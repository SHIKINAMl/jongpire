"""
1 対局の記録を組み立てる（データベースには触らない）

GameSession が対局の始まり・局の終わり・対局の終わりで呼び、対局の終わりに MatchRecord を作る。
記録はランキング（players / player_stats）と、学習・CPU 調整用のデータ（matches / match_players / match_rounds）に使う。

ランキングの値（プレイヤーごと）:
- max_damage   … 和了 1 回で相手に与えた損失（loser_loss。単騎の 2 倍・強襲の追加ダメージを含む）の最大
- match_points … 1 対局の累計獲得点数（cumulative_earned_points。30000 で勝ち）
- rounds       … 1 対局で最後まで打った局数
"""
import time
from dataclasses import dataclass, field
from typing import Any, Optional

POINT_VICTORY = 30000
# ランキングに入れる対戦の種類（CPU 戦は記録だけ残し、ランキングには入れない）
RANKED_MODES = ("public", "private")


@dataclass(frozen=True)
class PlayerIdentity:
    """接続しているプレイヤーの ID と表示名（register / identify で確かめたもの）。名前を登録する前は name が None。"""
    player_id: str
    name: Optional[str]
    # データベースで秘密の文字列を確かめられたか。確かめられなかった名乗り（データベースが使えないとき）は、
    # 名前の表示にだけ使い、ランキングには入れない
    verified: bool = True


@dataclass
class Seat:
    client_id: str
    identity: Optional[PlayerIdentity] = None  # 名乗っていない人間・CPU は None
    cpu_difficulty: Optional[str] = None  # CPU の席だけ


@dataclass
class PlayerResult:
    seat: int
    client_id: str
    player_id: Optional[str]
    name: Optional[str]
    cpu_difficulty: Optional[str]
    final_health: int
    match_points: int
    max_damage: int
    won: bool

    @property
    def is_cpu(self) -> bool:
        return self.cpu_difficulty is not None


@dataclass
class MatchRecord:
    mode: str  # "public" / "private" / "cpu"
    started_at: float
    ended_at: float
    rounds: int  # 最後まで打った局数
    end_reason: str  # 決着の理由（engine.game_end_reason）、途中で終わったときは "cancelled:理由"
    finished: bool  # 決着まで打ったか（切断・放置で終わったら False）
    players: list[PlayerResult]
    round_logs: list[dict] = field(default_factory=list)  # 局ごとの記録（学習・CPU 調整用）

    @property
    def ranked(self) -> bool:
        """ランキングに入れるか。対人戦を決着まで打ったときだけ。"""
        return self.finished and self.mode in RANKED_MODES


def _tiles(wall: list[int], indexes) -> list[int]:
    return [wall[i] for i in sorted(indexes) if 0 <= i < len(wall)]


class MatchRecorder:
    """1 対局分の記録係。"""

    def __init__(self, mode: str, seats: list[Seat]):
        self.mode = mode
        self.seats = seats
        self.started_at = time.time()
        self.round_logs: list[dict] = []
        self._max_damage = {seat.client_id: 0 for seat in seats}
        self.finished = False

    def record_round(self, engine: Any, is_draw: bool) -> None:
        """
        局の終わりに呼ぶ（次の局の準備で状態が消える前に、同期的に）。
        各プレイヤーの山・手牌・待ち・捨て牌・掛け金と、精算結果を残す。
        """
        liquidation = None if is_draw else engine.get_last_liquidation_result()
        if liquidation:
            winner = liquidation.get("winner_id")
            if winner in self._max_damage:
                self._max_damage[winner] = max(self._max_damage[winner], int(liquidation.get("loser_loss") or 0))
        players = []
        for p in engine.state.players:
            players.append({
                "client_id": p.player_id,
                "wall": list(p.wall),
                "hand": _tiles(p.wall, p.hand),
                "waits": list(p.waits),
                "discards": list(p.discards),
                "bet": p.bet,
                "health": p.health,
                "cumulative_earned_points": p.cumulative_earned_points,
            })
        self.round_logs.append({
            "round_number": engine.state.round_state.round_number,
            "dora_id": engine.state.round_state.dora_id,
            "is_draw": bool(is_draw),
            "liquidation": dict(liquidation) if liquidation else None,
            "players": players,
        })

    @staticmethod
    def _winner(engine: Any) -> Optional[str]:
        """勝った方の client_id（training/skill_game.game_winner と同じ決め方＋掛け金を払えなかった方の負け）。"""
        players = list(engine.state.players)
        if engine.game_end_reason == "cumulative_earned_points":
            reached = [p.player_id for p in players if p.cumulative_earned_points >= POINT_VICTORY]
            if len(reached) == 1:
                return reached[0]
        unpayable = [p.player_id for p in players if p.player_id in getattr(engine, "bet_unpayable_players", ())]
        if len(unpayable) == 1:
            return next(p.player_id for p in players if p.player_id != unpayable[0])
        if len(players) != 2 or players[0].health == players[1].health:
            return None
        return max(players, key=lambda p: p.health).player_id

    def build(self, engine: Optional[Any], finished: bool, reason: Optional[str] = None) -> MatchRecord:
        """
        対局の記録を作る。finished=False（切断・放置）なら、ランキングには入れない記録になる。
        engine がない（開始前に破棄された）ときは、HP などは 0 で埋める。
        """
        self.finished = True
        winner = self._winner(engine) if (engine is not None and finished) else None
        by_id = {p.player_id: p for p in engine.state.players} if engine is not None else {}
        results = []
        for index, seat in enumerate(self.seats):
            state = by_id.get(seat.client_id)
            results.append(PlayerResult(
                seat=index,
                client_id=seat.client_id,
                # 名前を登録したプレイヤーだけを結びつける（名前の前はデータベースにプレイヤーがいない）
                player_id=(seat.identity.player_id
                           if seat.identity and seat.identity.verified and seat.identity.name else None),
                name=seat.identity.name if seat.identity else None,
                cpu_difficulty=seat.cpu_difficulty,
                final_health=state.health if state is not None else 0,
                match_points=state.cumulative_earned_points if state is not None else 0,
                max_damage=self._max_damage.get(seat.client_id, 0),
                won=winner == seat.client_id,
            ))
        end_reason = (engine.game_end_reason if engine is not None else None) or "unknown"
        if not finished:
            end_reason = f"cancelled:{reason or 'unknown'}"
        return MatchRecord(
            mode=self.mode,
            started_at=self.started_at,
            ended_at=time.time(),
            rounds=len(self.round_logs),
            end_reason=end_reason,
            finished=finished,
            players=results,
            round_logs=self.round_logs,
        )
