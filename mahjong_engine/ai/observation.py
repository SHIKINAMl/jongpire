"""
CPU の視界（CpuObservation）

CPU が判断に使ってよい情報だけを詰める。エンジンの状態を CPU に直接読ませないための壁。

見てよい情報（人間のクライアントが画面に出すものと同じ）:
- 自分の山・聴牌例・手牌・待ち、ドラ
- 相手の捨て牌（順番付き）、透視で公開された相手の牌
- 相手の透視で公開された自分の牌（透視の結果は両者に通知される。相手に読まれている牌）
- 対局を通して覚えた相手のくせ（和了った牌・捨てた順番。tactics.OpponentProfile）
- 両者の HP、掛け金、公開された役強化、スキル使用の通知
- 局数、流局の持ち越し回数

見てはいけない情報: 相手の山の未公開部分、相手の手牌、相手の待ち
（サーバーは dealing_completed などで相手の山もクライアントへ送っているが、画面には出していない）
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..engine.game_engine import PERSPECTIVE_REVEAL_COUNT, GameEngine
from ..engine.game_state import SkillType
from .tactics import OpponentProfile

# CPU が使うスキル（特殊勝利は使わない）
CPU_SKILLS = (SkillType.MULLIGAN, SkillType.PERSPECTIVE, SkillType.BOOST_HAND, SkillType.ASSAULT)


@dataclass
class PublicInfo:
    """WebSocket の通知から CPU 自身が覚えておく公開情報。"""

    opponent_opening_boosts: dict[str, int] = field(default_factory=dict)
    opponent_boost_use_count: int = 0
    opponent_skills_this_round: list[str] = field(default_factory=list)
    my_skills_this_round: dict[str, int] = field(default_factory=dict)
    # 自分の透視で、相手の山のその色の牌を全部見た色（公開された枚数が規定より少なかった＝もう残っていない）
    complete_suits: set[str] = field(default_factory=set)
    # マリガンで見て牌山に戻った牌（牌ID）。牌山にあるので、相手の山にはない
    known_reserved: list[int] = field(default_factory=list)
    # 対局を通して覚える相手のくせ（局が変わっても消さない）
    opponent_profile: OpponentProfile = field(default_factory=OpponentProfile)

    def reset_round(self) -> None:
        self.opponent_skills_this_round = []
        self.my_skills_this_round = {}
        self.complete_suits = set()
        self.known_reserved = []

    def record_mulligan(self, pending: dict, changes: list[dict]) -> None:
        """マリガンの結果を覚える。見た牌のうち手に入れなかった牌は、牌山に戻っている。"""
        returned = Counter(pending["options"])
        returned.subtract(c["new_tile"] for c in changes)
        self.known_reserved.extend(returned.elements())

    def record_round_end(self, my_id: str, my_discards, opponent_discards, liquidation: Optional[dict]) -> None:
        """局の終わりに相手のくせを覚える。相手の和了り牌は、放銃した自分の最後の捨て牌。"""
        my_discards = list(my_discards)
        opponent_won = bool(liquidation) and liquidation.get("winner_id") not in (None, my_id)
        self.opponent_profile.record_round(
            opponent_discards,
            opponent_won,
            my_discards[-1] if opponent_won and my_discards else None,
            bool(liquidation and liquidation.get("is_tanki_wait")),
        )

    def record_my_perspective(self, suit: Optional[str], newly_exposed_count: int) -> None:
        """自分の透視の結果を覚える。規定の枚数より少なければ、その色はもう全部見えている。"""
        if suit is not None and newly_exposed_count < PERSPECTIVE_REVEAL_COUNT:
            self.complete_suits.add(suit)


@dataclass(frozen=True)
class MyView:
    client_id: str
    wall: tuple[int, ...]
    hand: tuple[int, ...]  # 手牌確定前はサーバーの聴牌例（wall index）
    waits: tuple[int, ...]  # 牌種（下位5ビット）
    discarded_wall_indexes: frozenset[int]
    health: int
    cumulative_earned_points: int  # 和了で得た累計点数（精算結果で両者に公開。30000 で勝ち）
    bet: int
    boost_hand_bonus: dict[str, int]
    special_victory_count: int
    assault_used_this_round: bool
    skill_costs: dict[str, int]
    bet_max: int
    bet_unit: int
    boost_hand_targets: int
    can_mulligan: bool  # 交換用の予備牌が残っているか
    known_reserved: tuple[int, ...]  # マリガンで見て牌山に戻った牌（相手の山にはない）
    exposed_to_opponent: dict[int, int]  # 相手の透視で公開された自分の wall index → 牌ID（相手に読まれている牌）
    skill_health_floor: int  # スキルを使った後に残さなければならない HP（流局の続きの局の積み上がった掛け金）
    # ボルテージ（河から両者に分かる）: 点・切れているか・切れた後に再開する点
    voltage_points: int
    voltage_broken: bool
    voltage_resume_points: int


@dataclass(frozen=True)
class OpponentView:
    client_id: str
    wall_size: int
    exposed: dict[int, int]  # 透視で公開された wall index → 牌ID
    discards: tuple[int, ...]  # 捨て牌（牌ID、捨てた順）
    health: int
    cumulative_earned_points: int  # 和了で得た累計点数（精算結果で両者に公開。30000 で勝ち）
    bet: int
    opening_boosts: dict[str, int]
    boost_use_count: int
    special_victory_count: int
    skills_this_round: tuple[str, ...]
    complete_suits: frozenset[str]  # 相手の山のその色の牌を全部見た色


@dataclass(frozen=True)
class CpuObservation:
    phase: Optional[str]
    round_number: int
    carry_over_draw_count: int
    dora_id: Optional[int]
    me: MyView
    opponent: OpponentView
    i_am_first_player: bool  # 打牌フェーズの先手か（discard_phase_started で公開される）
    my_discard_count: int
    my_skills_this_round: dict[str, int]
    opponent_profile: Optional[OpponentProfile] = None  # 対局を通して覚えた相手のくせ


def build_observation(engine: GameEngine, client_id: str, public: PublicInfo) -> CpuObservation:
    """エンジンの状態から、CPU が見てよい情報だけを取り出す。"""
    me = engine.get_player_by_id(client_id)
    opponent = next(p for p in engine.state.players if p.player_id != client_id)
    round_state = engine.state.round_state
    status = round_state.status

    max_bet, bet_unit = engine.get_bet_rule(me)

    my_view = MyView(
        client_id=client_id,
        wall=tuple(me.wall),
        hand=tuple(me.hand),
        waits=tuple(w & 0b11111 for w in me.waits),
        discarded_wall_indexes=frozenset(me.discarded_wall_indexes),
        health=me.health,
        cumulative_earned_points=me.cumulative_earned_points,
        bet=me.bet,
        boost_hand_bonus=dict(engine._effective_boost_bonus_map(me)),
        special_victory_count=me.special_victory_count,
        assault_used_this_round=me.assault_used_this_round,
        skill_costs={skill.value: engine.get_skill_cost(me, skill) for skill in CPU_SKILLS},
        bet_max=max_bet,
        bet_unit=bet_unit,
        boost_hand_targets=me.boost_hand_targets,
        can_mulligan=bool(round_state.reserved_tiles),
        known_reserved=tuple(public.known_reserved),
        exposed_to_opponent={idx: me.wall[idx] for idx in sorted(me.exposed_hand_indexes) if idx < len(me.wall)},
        skill_health_floor=engine.skill_health_floor(me),
        voltage_points=me.voltage_points,
        voltage_broken=me.voltage_broken,
        voltage_resume_points=me.voltage_resume_points,
    )

    opponent_view = OpponentView(
        client_id=opponent.player_id,
        wall_size=len(opponent.wall),
        exposed={idx: opponent.wall[idx] for idx in sorted(opponent.exposed_hand_indexes) if idx < len(opponent.wall)},
        discards=tuple(opponent.discards),
        health=opponent.health,
        cumulative_earned_points=opponent.cumulative_earned_points,
        bet=opponent.bet,
        opening_boosts=dict(public.opponent_opening_boosts),
        boost_use_count=public.opponent_boost_use_count,
        special_victory_count=opponent.special_victory_count,
        skills_this_round=tuple(public.opponent_skills_this_round),
        complete_suits=frozenset(public.complete_suits),
    )

    return CpuObservation(
        phase=status.value if status else None,
        round_number=round_state.round_number,
        carry_over_draw_count=engine._carry_over_draw_count,
        dora_id=round_state.dora_id,
        me=my_view,
        opponent=opponent_view,
        i_am_first_player=engine.state.players[round_state.first_player_index].player_id == client_id,
        my_discard_count=len(me.discards),
        my_skills_this_round=dict(public.my_skills_this_round),
        opponent_profile=public.opponent_profile,
    )
