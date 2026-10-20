"""
麻雀ゲームエンジン
"""
import logging
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional
import random

from .game_state import GameState, RoundStatus, SkillType, PlayerState, get_skill_cost, get_bet_rule, apply_skill_cost_rate
from .tile_wall import RoundDeal, generate_round_deal
from .hand_analyzer import HandAnalyzer
from .yaku import Yaku

logger = logging.getLogger(__name__)

# 1 局で各プレイヤーが捨てる枚数。両者がこの枚数を捨て終えたら流局
DISCARDS_PER_PLAYER = 17

# 透視で指定できる色と、その色に入る牌種（下位5ビット）
PERSPECTIVE_SUITS: dict[str, range] = {
    "manzu": range(0, 9),    # 萬子
    "pinzu": range(9, 18),   # 筒子
    "souzu": range(18, 27),  # 索子
    "jihai": range(27, 29),  # 字牌（東・西）
}
# 透視 1 回で公開する枚数
PERSPECTIVE_REVEAL_COUNT = 3
# 透視を 1 局に使える回数（1 人あたり）
PERSPECTIVE_MAX_USES_PER_ROUND = 5
# マリガンで交換に出す枚数・牌山から出す候補の枚数。
# 自分の山から 3 枚を出し、出した 3 枚 + 候補 3 枚の 6 枚から 3 枚を選ぶ
MULLIGAN_CANDIDATE_COUNT = 3

# 精算倍率の段階（満貫・跳満・倍満・三倍満・役満・ダブル役満）
MULTIPLIER_TIERS = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
MULTIPLIER_TIER_LABELS = ("満貫", "跳満", "倍満", "三倍満", "役満", "ダブル役満")
# 段階に届く最低の翻数
_TIER_MIN_HAN = (4, 6, 8, 11, 13, 26)


# ボルテージの段階（点 → 和了ったときの獲得額の倍率）と、連続が切れた後に再開する点
_VOLTAGE_TIERS = ((14, 2.0), (9, 1.6), (5, 1.3), (2, 1.1), (0, 1.0))
_VOLTAGE_RESUME = ((14, 9), (9, 5), (5, 2), (0, 0))
VOLTAGE_MAX_POINTS = 14


def voltage_multiplier(points: int) -> float:
    """ボルテージの点から、勝者の獲得額の倍率を返す。"""
    return next(rate for floor, rate in _VOLTAGE_TIERS if points >= floor)


def voltage_resume_points(points: int) -> int:
    """連続が切れたとき、次に初めての種類を捨てたら再開する点（の 1 つ手前）。"""
    return next(resume for floor, resume in _VOLTAGE_RESUME if points >= floor)


def next_voltage(points: int, broken: bool, resume: int, is_new_kind: bool) -> tuple[int, bool, int]:
    """
    1 枚捨てた後のボルテージ（点, 切れているか, 再開する点）。GameEngine._apply_voltage_discard と同じ規則。
    どちらの河にもまだ出ていない種類を捨てると 1 点上がり（切れていたら再開する点から）、出た種類を捨てると切れる。
    """
    if is_new_kind:
        start = resume if broken else points
        return min(VOLTAGE_MAX_POINTS, start + 1), False, resume
    return 0, True, voltage_resume_points(points)


def han_to_tier(han: int) -> Optional[int]:
    """翻数から精算倍率の段階（MULTIPLIER_TIERS の番号）を返す。満貫未満は None（和了れない）。"""
    tier = None
    for i, min_han in enumerate(_TIER_MIN_HAN):
        if han >= min_han:
            tier = i
    return tier


def boost_tier_cap(yaku_name: str) -> int:
    """
    役強化でその役が昇格させられる上限の段階。和了りやすい（翻数の低い）役ほど上限が低い。
    1・2 翻 → 跳満、3・4 翻 → 倍満、清一色（6 翻）→ 役満、役満 → ダブル役満。
    """
    han = Yaku.get_han_by_name(yaku_name)
    if han <= 2:
        return 1
    if han <= 5:
        return 2
    if han < 13:
        return 4
    return 5


def promoted_tier(han: int, yaku: list[str], boost_map: dict[str, int]) -> Optional[int]:
    """
    役強化を含めた精算倍率の段階。満貫未満は None（役強化があっても和了れない）。

    手に含まれる強化済みの役 1 つにつき 1 段昇格する。上限は、含まれる強化済みの役の上限のうち高い方。
    元の段階が上限以上なら昇格しない（下がることはない）。
    """
    base = han_to_tier(han)
    if base is None:
        return None
    boosted = {name for name in yaku if boost_map.get(name, 0) > 0}
    if not boosted:
        return base
    cap = max(boost_tier_cap(name) for name in boosted)
    return max(base, min(base + len(boosted), cap))


def promoted_multiplier(han: int, yaku: list[str], boost_map: dict[str, int]) -> float:
    """役強化を含めた精算倍率。満貫未満は 0。"""
    tier = promoted_tier(han, yaku, boost_map)
    return 0.0 if tier is None else MULTIPLIER_TIERS[tier]

# 配牌生成（満貫聴牌形の探索）を行うスレッド。全マッチで共有する。
_DEAL_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="deal")


def tiles_to_wall_indexes(wall: list[int], tiles: list[int]) -> Optional[list[int]]:
    """
    牌IDの並びを、山（wall）の index の並びへ変換する。変換できなければ None。

    ドラ/赤ドラを取り違えると翻数が変わり満貫を割るため、同一の牌を優先し、なければ同じ牌種を使う。
    """
    used = [False] * len(wall)
    indexes: list[int] = []
    for tile in tiles:
        found = next((i for i, t in enumerate(wall) if not used[i] and t == tile), None)
        if found is None:
            kind = tile & 0b11111
            found = next((i for i, t in enumerate(wall) if not used[i] and (t & 0b11111) == kind), None)
        if found is None:
            return None
        used[found] = True
        indexes.append(found)
    return indexes


class GameEngine:
    """麻雀ゲームエンジン"""

    def __init__(
        self,
        max_rounds: int = 25,
        deal_factory: Optional[Callable[[int], RoundDeal]] = None,
    ):
        """
        ゲームエンジンを初期化

        Args:
            max_rounds: 最大ラウンド数（デフォルト25）
            deal_factory: 配牌を作る関数（引数は人数）。None なら本番の generate_round_deal を
                別スレッドで動かす。学習用のシミュレーターは、事前に作った配牌を返す関数を渡す
        """
        self.state = GameState()
        self.num_players = 2
        self.max_rounds = max_rounds
        self._carry_over_bets = False
        self._carry_over_draw_count = 0
        self._last_liquidation_result: Optional[dict] = None
        self._next_round_ready_players: set[str] = set()
        self._pending_agari: Optional[dict] = None
        self._voltage_seen_base_ids: set[int] = set() #　両者の河に出た牌を保存する集合
        # 次局の配牌（バックグラウンドで生成中 or 生成済み）
        self._next_deal: Optional[Future] = None
        self._deal_factory = deal_factory
        # 掛け金を払えずに決着したときの、払えなかったプレイヤー（game_end で知らせる。HP の多い方が払えないこともある）
        self.bet_unpayable_players: list[str] = []
        # この局で起きたこと（スキル・マリガンの選択・打牌）を起きた順に残す。記録（mahjong_engine/records）が
        # 局の終わりに読む。配牌のたびに空にする
        self.round_events: list[dict] = []
        # この局の配牌時に特典で公開した牌（initial_perspective_count を持つプレイヤーの分）
        self._initial_perspective_reveals: list[dict] = []
        # 対局が終わった理由（game_end の victory_method になる）
        self.game_end_reason: Optional[str] = None

        # 各種コールバック
        # 準備フェーズ
        self.on_dealt: Optional[Callable[[], None]] = None
        self.on_selected: Optional[Callable[[], None]] = None
        self.on_bet: Optional[Callable[[], None]] = None

        # 打牌フェーズ
        self.on_discard_started: Optional[Callable[[], None]] = None
        self.on_discarded: Optional[Callable[[str, int], None]] = None
        self.on_agari_pending: Optional[Callable[[str, str, int], None]] = None
        self.on_skill_casted: Optional[Callable[[str, SkillType, int, dict], None]] = None
        self.on_special_victory_won: Optional[Callable[[str], None]] = None  # player_id

        self.on_round_start: Optional[Callable[[], None]] = None
        self.on_round_end: Optional[Callable[[bool], None]] = None  # is_draw を受け取る
        self.on_game_end: Optional[Callable[[], None]] = None
        self.on_phase_change: Optional[Callable[[RoundStatus], None]] = None

    def initialize_players(self, player_ids: list[str]):
        """プレイヤーを初期化"""
        self.state.players = [
            PlayerState(
                player_id=player_id,
                health=20000,  # 初期体力
            )
            for player_id in player_ids
        ]

    def start_game(self, max_rounds: int = 25):
        """ゲームを開始"""
        self.max_rounds = max_rounds
        logger.info("ゲーム開始: max_rounds=%d", max_rounds)
        self._start_round()

    def prepare_next_deal(self) -> Future:
        """
        次局の配牌をバックグラウンドで生成し始める。既に生成中・生成済みならそれを返す。
        返り値の Future が完了していれば、_deal_tiles は待たずに配牌できる。
        """
        if self._deal_factory is not None:
            # 差し込まれた配牌は配るときに作る（先読みしない）。待つものはないので完了済みの Future を返す
            ready: Future = Future()
            ready.set_result(None)
            return ready
        if self._next_deal is None:
            self._next_deal = _DEAL_EXECUTOR.submit(generate_round_deal, self.num_players)
        return self._next_deal

    def _take_next_deal(self) -> RoundDeal:
        """用意しておいた配牌を取り出す。未用意・生成失敗時はその場で生成する。"""
        if self._deal_factory is not None:
            return self._deal_factory(self.num_players)
        future, self._next_deal = self._next_deal, None
        if future is not None:
            try:
                return future.result()
            except Exception:
                logger.exception("バックグラウンドの配牌生成に失敗したため、その場で生成します")
        return generate_round_deal(self.num_players)

    def dispose(self) -> None:
        """マッチ破棄時に、未着手の配牌生成を取り消す。"""
        if self._next_deal is not None:
            self._next_deal.cancel()
            self._next_deal = None

    def _start_round(self):
        """局を開始"""
        logger.info("局開始: round=%d", self.state.round_state.round_number)
        # 1局目や先読みが間に合わなかった場合に備え、配牌の生成を先に始めておく
        self.prepare_next_deal()
        self._invoke_callback(self.on_round_start)

        # 配牌を実行
        self._set_phase(RoundStatus.DEALING)
        self._deal_tiles()

    def _deal_tiles(self):
        """各プレイヤーに牌を配る"""
        # 局ごとに新しい牌山から配る（通常は局中に先読みした配牌を使う）
        deal = self._take_next_deal()
        hands = deal.hands

        # 配牌後に牌山へ残った実牌を MULLIGAN の交換候補にする。
        self.state.round_state.reserved_tiles = list(deal.reserved_tiles)

        def _to_wall_indexes(wall_tiles: list[int], hand_tiles: list[int]) -> list[int]:
            indexes = tiles_to_wall_indexes(wall_tiles, hand_tiles)
            if indexes is None:
                raise ValueError("deal が返した聴牌例を wall index へ変換できませんでした")
            return indexes

        for player, (wall, hand_tiles) in zip(self.state.players, hands):
            # 配られた牌。マリガンで書き換えるので、配牌そのもの（deal）とは別のリストにする
            player.wall = list(wall)
            player.hand = _to_wall_indexes(wall, hand_tiles)  # 手牌例を wall index で保持

        self.state.round_state.dora_id = deal.dora_id
        self._pending_agari = None
        self.round_events = []
        self._apply_initial_perspectives()

        # 局中に次局の配牌を用意しておき、局開始時にすぐ配れるようにする
        self.prepare_next_deal()

        self._invoke_callback(self.on_dealt)

        self._set_phase(RoundStatus.HAND_SELECTION)

    def _apply_initial_perspectives(self) -> None:
        """配牌時に、特典を持つプレイヤーへ相手の牌を無料で公開する。"""
        self._initial_perspective_reveals = []
        for player in self.state.players:
            if player.initial_perspective_count <= 0:
                continue
            target = next((p for p in self.state.players if p.player_id != player.player_id), None)
            if target is None:
                continue
            exposed = self._expose_random_tiles(target, player.initial_perspective_count)
            self._initial_perspective_reveals.append({
                "player_id": player.player_id,
                "target_id": target.player_id,
                "exposed_indexes": exposed,
            })
            self.round_events.append({
                "type": "skill", "player": player.player_id, "skill": SkillType.PERSPECTIVE.value,
                "source": "cpu_perk", "cost": 0, "health_after": player.health,
                "perspective": {"suit": None, "exposed_indexes": sorted(exposed),
                                "exposed_tiles": [target.wall[i] for i in sorted(exposed)]},
            })

    def get_initial_perspective_reveals(self) -> list[dict]:
        """この局の配牌時に特典で公開した牌を返す。"""
        return [dict(reveal) for reveal in self._initial_perspective_reveals]

    def selected_hand(self) -> None:
        """手牌の選択フェーズを完了"""
        self._invoke_callback(self.on_selected)

        if self._carry_over_bets:
            # 流局後は同じ掛け金を維持し、ベットフェーズをスキップ
            unpayable = self._players_unable_to_cover_carry_over_bet()
            if unpayable:
                logger.warning("持ち越し掛け金を支払えないプレイヤーがいるためゲーム終了: %s", unpayable)
                self.bet_unpayable_players = unpayable
                self._on_game_end("bet_unpayable")
                return
            self.bet()
            return

        # 通常局は最低掛け金を支払えるかチェック
        unpayable = self._players_unable_to_cover_min_bet()
        if unpayable:
            logger.warning("最低掛け金を支払えないプレイヤーがいるためゲーム終了: %s", unpayable)
            self.bet_unpayable_players = unpayable
            self._on_game_end("bet_unpayable")
            return

        self._set_phase(RoundStatus.BETTING)

    def bet(self) -> None:
        """掛け金設定フェーズを完了し、打牌フェーズを開始"""
        dead_players = [player.player_id for player in self.state.players if player.health < 0]
        if dead_players:
            logger.info("ベット確定時のHPマイナスによるゲーム終了: players=%s", dead_players)
            self._on_game_end("hp_zero")
            return

        self._invoke_callback(self.on_bet)
        self._set_phase(RoundStatus.DISCARD)
        self.state.round_state.current_player_index = random.randrange(0, self.num_players)
        self.state.round_state.first_player_index = self.state.round_state.current_player_index
        self._invoke_callback(self.on_discard_started)

    def get_bet_rule(self, player: PlayerState) -> tuple[int, int]:
        """プレイヤーの現在状態に対する掛け金ルール(上限, 単位)を返す。"""
        return get_bet_rule(player.special_victory_count)

    def skill_health_floor(self, player: PlayerState) -> int:
        """
        スキルを使った後に残さなければならない HP。流局の続きの局では、積み上がった掛け金
        （掛け金 × (流局の回数 + 1)）。それ以外は 0（掛け金はスキルの後に決めるので、払える額を賭ければよい）。
        """
        if not self._carry_over_bets:
            return 0
        return self.carry_over_stake(player, self._carry_over_draw_count)

    @staticmethod
    def carry_over_stake(player: PlayerState, draw_count: int) -> int:
        """流局が draw_count 回続いた後の局で精算される掛け金（掛け金 × (流局の回数 + 1)）。"""
        return player.bet * (draw_count + 1)

    def get_minimum_bet(self, player: PlayerState) -> int:
        """プレイヤーの現在状態に対する最低掛け金を返す。"""
        _, unit = self.get_bet_rule(player)
        return unit

    def place_bet(self, player: PlayerState, bet_amount: int) -> bool:
        """
        掛け金を検証して設定する。

        掛け金は先払いしない（HP はここでは減らない）。和了の精算でだけ「掛け金 × 倍率」が動く。
        賭けられるのは HP 以下の額だけ。
        """
        if not isinstance(bet_amount, int):
            return False

        max_bet, unit = self.get_bet_rule(player)

        if bet_amount < unit:
            return False
        if bet_amount > max_bet:
            return False
        if bet_amount % unit != 0:
            return False
        if player.health < bet_amount:
            return False

        player.bet = bet_amount
        player.base_bet = bet_amount
        return True

    def _players_unable_to_cover_min_bet(self) -> list[str]:
        """最低掛け金を HP で払えないプレイヤー。"""
        return [p.player_id for p in self.state.players if p.health < self.get_minimum_bet(p)]

    def _players_unable_to_cover_carry_over_bet(self) -> list[str]:
        """流局持ち越し時、積み上がった掛け金（掛け金 × (流局の回数 + 1)）を HP で払えないプレイヤー。"""
        return [
            p.player_id for p in self.state.players
            if p.health < self.carry_over_stake(p, self._carry_over_draw_count)
        ]

    def select_hand(self, hand_indexes: list[int], player: PlayerState, force: bool = False) -> bool:
        """
        プレイヤーが手牌を選択したときの処理

        Args:
            hand_indexes: 選択された手牌の山牌内 index リスト
            player: プレイヤーの状態
            force: True の場合、非聴牌形でも確定を許可する

        Returns:
            手牌が有効であれば True、そうでなければ False
        """
        # index から牌 ID に変換してテンパイ判定
        hand_tiles = [player.wall[idx] for idx in hand_indexes]
        is_tenpai = HandAnalyzer.is_tenpai(hand_tiles, player.wall)

        if not force and not is_tenpai:
            return False

        player.hand = hand_indexes  # 手牌を index で保持
        player.waits = HandAnalyzer.get_tenpai_waiting_tiles(hand_tiles, player.wall) if is_tenpai else []

        return True

    # ========== スキル処理 ==========

    @staticmethod
    def normalize_boost_target_yaku_name(yaku_name: str, allow_blocked: bool = False) -> Optional[str]:
        """BOOST_HAND の入力役名を正規化する。`役名` / `役名+N` / `(役名)+N` を許可。"""
        if not isinstance(yaku_name, str):
            return None

        raw = yaku_name.strip()
        if not raw:
            return None

        # 表示形式 `役名+N` または `(役名)+N` が渡ってきた場合、基底の役名へ戻す。
        m = re.fullmatch(r"(.+?)\+(\d+)", raw)
        base_name = m.group(1).strip() if m else raw

        if base_name.startswith("(") and base_name.endswith(")") and len(base_name) > 2:
            base_name = base_name[1:-1].strip()

        # 立直・ドラ・赤ドラは BOOST_HAND の強化対象外
        blocked_yaku_names = {
            Yaku.RICHI.japanese_name,
            Yaku.DORA.japanese_name,
            Yaku.AKA_DORA.japanese_name,
        }
        if not allow_blocked and base_name in blocked_yaku_names:
            return None

        if Yaku.get_han_by_name(base_name) == -1:
            return None

        return base_name

    def _normalized_boost_bonus_map(self, player: PlayerState) -> dict[str, int]:
        """プレイヤーの BOOST_HAND ボーナス辞書を正規化して返す。"""
        normalized: dict[str, int] = {}
        for raw_name, count in player.boost_hand_bonus.items():
            if not isinstance(count, int) or count <= 0:
                continue
            base_name = self.normalize_boost_target_yaku_name(str(raw_name))
            if base_name is None:
                continue
            normalized[base_name] = normalized.get(base_name, 0) + count
        return normalized

    def _effective_boost_bonus_map(self, player: PlayerState) -> dict[str, int]:
        """役強化ボーナス辞書を返す。"""
        return self._normalized_boost_bonus_map(player)

    @staticmethod
    def get_opening_boost_candidates() -> list[str]:
        """開始時恒常強化の候補役（1翻/2翻）を返す。"""
        blocked_yaku_names = {
            Yaku.RICHI.japanese_name,
            Yaku.DORA.japanese_name,
            Yaku.AKA_DORA.japanese_name,
        }
        return [
            yaku.japanese_name
            for yaku in Yaku
            if yaku.han in (1, 2) and yaku.japanese_name not in blocked_yaku_names
        ]

    def assign_opening_boost(self, player: PlayerState, yaku_name: str, bonus_han: int = 1) -> bool:
        """プレイヤーに開始時恒常強化を付与する。"""
        if not isinstance(bonus_han, int) or bonus_han <= 0:
            return False

        normalized_name = self.normalize_boost_target_yaku_name(yaku_name)
        if normalized_name is None or normalized_name in self._effective_boost_bonus_map(player):
            return False

        # 強化は役ごとに 1 段の昇格（bonus_han は互換のため受け取るが、段数は増えない）
        player.boost_hand_bonus[normalized_name] = 1
        return True

    def assign_random_opening_boosts(self, rng: random.Random | None = None) -> list[dict]:
        """
        開始時の恒常強化を、各プレイヤーに opening_boost_count 個ずつ付与する（役は重複なし。スキルと同じ 1 段の昇格）。

        Returns:
            付与した強化（{"client_id", "yaku_name", "bonus_han"} のリスト。opening_boost_assigned で送る形）
        """
        rng = rng or random
        candidates = self.get_opening_boost_candidates()
        assigned: list[dict] = []
        for player in self.state.players:
            for yaku_name in rng.sample(candidates, min(player.opening_boost_count, len(candidates))):
                if self.assign_opening_boost(player, yaku_name, bonus_han=1):
                    assigned.append({"client_id": player.player_id, "yaku_name": yaku_name, "bonus_han": 1})
        return assigned

    def _build_display_yaku_list(self, yaku_list: list[str], bonus_map: dict[str, int]) -> list[str]:
        """表示用役名を `役名+回数` 形式で組み立てる。"""
        display: list[str] = []
        for name in yaku_list:
            bonus_count = bonus_map.get(name, 0)
            display.append(f"{name}+{bonus_count}" if bonus_count > 0 else name)
        return display

    def cast_skill(self, user: PlayerState, skill_type: SkillType, target: PlayerState | None = None, **options) -> Optional[int]:
        """
        スキルを発動する（統一インターフェース）

        Args:
            user: スキル使用者
            skill_type: スキル種別
            target: 相手を対象とするスキルの場合は相手プレイヤー
            **options: スキル固有のオプション
                - yaku_name: BOOST_HAND の対象役名
                - target_hand_indexes: MULLIGAN で交換に出す wall 内インデックス 3 つ

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        # HP コスト計算
        try:
            cost = self.get_skill_cost(user, skill_type)
        except KeyError:
            logger.error("スキルコスト取得失敗: skill_type=%s", skill_type)
            return None

        # HP 不足チェック
        if user.health < cost:
            logger.debug("スキル HP 不足: player=%s  skill=%s  cost=%d  health=%d",
                         user.player_id, skill_type.value, cost, user.health)
            return None

        # 流局の続きの局では、積み上がった掛け金を払えなくなるほど HP を使えない（自滅を防ぐ）
        if user.health - cost < self.skill_health_floor(user):
            logger.debug("スキル 持ち越し掛け金を下回る: player=%s  skill=%s  cost=%d  health=%d  floor=%d",
                         user.player_id, skill_type.value, cost, user.health, self.skill_health_floor(user))
            return None

        # スキル固有の前提条件チェック
        validation_error = self._validate_skill_cast(user, skill_type, target, options)
        if validation_error:
            logger.debug("スキル前提条件エラー: player=%s  skill=%s", user.player_id, skill_type.value)
            return None

        # HP 消費
        user.health -= cost
        logger.info("スキル発動: player=%s  skill=%s  cost=%d  health_after=%d",
                    user.player_id, skill_type.value, cost, user.health)

        # スキル効果を適用
        exposed_tiles = self._apply_skill_effect(user, skill_type, target, options)
        self._record_skill_event(user, skill_type, cost, target, options, exposed_tiles)

        # コールバック呼び出し
        self._invoke_callback(self.on_skill_casted, user.player_id, skill_type, cost, exposed_tiles)

        return cost

    def _record_skill_event(
        self, user: PlayerState, skill_type: SkillType, cost: int, target: Optional[PlayerState],
        options: dict, exposed_tiles: dict,
    ) -> None:
        """使ったスキルを局の出来事に残す（種類・コストと、透視で見えた牌・マリガンの候補・強化した役）。"""
        event: dict = {
            "type": "skill", "player": user.player_id, "skill": skill_type.value, "source": "skill",
            "cost": cost, "health_after": user.health,
        }
        if skill_type == SkillType.PERSPECTIVE:
            perspective = exposed_tiles.get("__perspective__") or {}
            indexes = list(perspective.get("new_indexes", []))
            event["perspective"] = {
                "suit": perspective.get("suit"),
                "exposed_indexes": indexes,
                "exposed_tiles": [target.wall[i] for i in indexes] if target is not None else [],
            }
        elif skill_type == SkillType.MULLIGAN:
            pending = user.pending_mulligan or {}
            # 選んだ結果は、この後の mulligan_select の出来事に残る
            event["mulligan"] = {"wall_indexes": list(pending.get("wall_indexes", [])),
                                 "options": list(pending.get("options", []))}
        elif skill_type == SkillType.BOOST_HAND:
            event["boost_hand"] = {"yaku_names": list(options.get("yaku_names") or [])}
        self.round_events.append(event)

    def get_skill_cost(self, player: PlayerState, skill_type: SkillType) -> int:
        """プレイヤーの特典（スキルコストの倍率）を反映したスキルコストを返す。"""
        base_cost = get_skill_cost(skill_type, player.special_victory_count)
        return apply_skill_cost_rate(base_cost, player.skill_cost_rate)

    def _validate_skill_cast(self, user: PlayerState, skill_type: SkillType, target: PlayerState | None, options: dict) -> bool:
        """
        スキル発動の前提条件をチェック

        Returns:
            エラーがあれば True、正常ならば False
        """
        if skill_type == SkillType.BOOST_HAND:
            # 1回で強化できる役の数は boost_hand_targets まで。
            # 強化は役ごとに 1 回だけ（同じ役を重ねて指定する・強化済みの役をもう一度強化することはできない）
            yaku_names = options.get("yaku_names")
            if yaku_names is None:
                yaku_names = [options.get("yaku_name")]
            if not isinstance(yaku_names, list) or not 1 <= len(yaku_names) <= user.boost_hand_targets:
                return True
            normalized_names = [self.normalize_boost_target_yaku_name(name) for name in yaku_names]
            if any(name is None for name in normalized_names):
                return True
            if len(set(normalized_names)) != len(normalized_names):
                return True
            boosted = self._effective_boost_bonus_map(user)
            if any(name in boosted for name in normalized_names):
                return True
            options["yaku_names"] = normalized_names

        elif skill_type == SkillType.MULLIGAN:
            def valid_index(idx) -> bool:
                return (
                    isinstance(idx, int) and not isinstance(idx, bool)
                    and 0 <= idx < len(user.wall) and idx not in user.discarded_wall_indexes
                )

            indexes = options.get("target_hand_indexes")
            if (
                not isinstance(indexes, list) or len(indexes) != MULLIGAN_CANDIDATE_COUNT
                or len(set(indexes)) != len(indexes) or not all(valid_index(i) for i in indexes)
            ):
                return True
            # 予備牌チェック
            if not self.state.round_state.reserved_tiles:
                return True
            # 前のマリガンの牌を選び終えるまでは、次を使えない
            if user.pending_mulligan is not None:
                return True

        elif skill_type == SkillType.PERSPECTIVE:
            if target is None:
                return True
            if user.perspective_uses_this_round >= PERSPECTIVE_MAX_USES_PER_ROUND:
                return True
            suit = options.get("suit")
            if suit is not None:
                # 色を指定したときは、その色が相手の山に残っていなくても使える（全額払い、0 枚公開）
                if suit not in PERSPECTIVE_SUITS:
                    return True
            else:
                unrevealed_indexes = [idx for idx in range(len(target.wall)) if idx not in target.exposed_hand_indexes]
                if not unrevealed_indexes:
                    return True

        elif skill_type == SkillType.ASSAULT:
            if user.assault_used_this_round:
                return True

        return False

    def _apply_skill_effect(self, user: PlayerState, skill_type: SkillType, target: PlayerState | None, options: dict) -> dict:
        """
        スキル効果を適用し、公開牌セットを返す

        Returns:
            公開された牌インデックス（PERSPECTIVE の場合は player_id ごとの辞書）
        """
        exposed_tiles: dict[str, set[int]] = {}

        if skill_type == SkillType.SPECIAL_VICTORY:
            user.special_victory_count += 1
            if user.special_victory_count >= 3:
                self._invoke_callback(self.on_special_victory_won, user.player_id)

        elif skill_type == SkillType.BOOST_HAND:
            for yaku_name in options["yaku_names"]:
                user.boost_hand_bonus[yaku_name] = 1

        elif skill_type == SkillType.PERSPECTIVE:
            # 使用者の牌は公開しない。相手（target）の山のうち、指定した色の未公開の牌から公開する
            # （色の指定がなければ山全体から。色を送らない古いクライアントのため）
            user.perspective_uses_this_round += 1
            if target is not None and target.player_id != user.player_id and target.wall:
                suit = options.get("suit")
                before = set(target.exposed_hand_indexes)
                kinds = PERSPECTIVE_SUITS[suit] if suit is not None else None
                exposed_tiles[target.player_id] = set(self._expose_random_tiles(target, PERSPECTIVE_REVEAL_COUNT, kinds))
                exposed_tiles["__perspective__"] = {
                    "suit": suit,
                    "new_indexes": sorted(target.exposed_hand_indexes - before),
                }

        elif skill_type == SkillType.MULLIGAN:
            # 牌山から候補を出し、選ばせる（選ぶのは resolve_mulligan）。
            # 候補は選び終えるまで牌山から抜いておき、手に入れなかった牌は牌山へ戻す
            reserved = self.state.round_state.reserved_tiles
            drawn = random.sample(reserved, min(MULLIGAN_CANDIDATE_COUNT, len(reserved)))
            for tile in drawn:
                reserved.remove(tile)
            indexes = list(options["target_hand_indexes"])
            user.pending_mulligan = {"wall_indexes": indexes, "options": [user.wall[i] for i in indexes] + drawn}
            exposed_tiles["__mulligan__"] = dict(user.pending_mulligan)

        elif skill_type == SkillType.ASSAULT:
            user.assault_used_this_round = True
            user.assault_active_this_round = True

        return exposed_tiles

    def _expose_random_tiles(self, target: PlayerState, count: int, kinds: Optional[range] = None) -> list[int]:
        """
        相手の山から未公開の牌を count 枚ランダムに公開する。足りなければある分だけ公開する。

        Args:
            kinds: 指定したときは、この牌種（下位5ビット）の牌からだけ選ぶ

        Returns:
            公開済みを含む、相手の全公開 index（昇順）
        """
        unrevealed_indexes = [
            idx for idx in range(len(target.wall))
            if idx not in target.exposed_hand_indexes
            and (kinds is None or (target.wall[idx] & 0b11111) in kinds)
        ]
        if unrevealed_indexes:
            exposed = random.sample(unrevealed_indexes, min(count, len(unrevealed_indexes)))
            target.exposed_hand_indexes.update(exposed)
        return sorted(target.exposed_hand_indexes)

    def use_skill(
        self,
        player: PlayerState,
        skill_type: SkillType,
        yaku_name: str | None = None,
        yaku_names: list[str] | None = None,
    ) -> Optional[int]:
        """
        プレイヤーがスキルを使用する（自分を対象とするスキル）

        Args:
            player: 使用プレイヤー
            skill_type: スキル種別（BOOST_HAND, SPECIAL_VICTORY など）
            yaku_name: BOOST_HAND 使用時に、翻を上げる対象役名
            yaku_names: BOOST_HAND で複数の役を同時に上げるときの対象役名（boost_hand_targets 件まで）

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        return self.cast_skill(player, skill_type, yaku_name=yaku_name, yaku_names=yaku_names)

    def use_skill_on_opponent(self, user: PlayerState, opponent: PlayerState, skill_type: SkillType) -> Optional[int]:
        """
        プレイヤーがスキルを使用する（相手を対象とするスキル）

        Args:
            user: 使用プレイヤー
            opponent: 相手プレイヤー
            skill_type: スキル種別（PERSPECTIVE など）

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        return self.cast_skill(user, skill_type, target=opponent)

    def execute_skill(self, user: PlayerState, skill_type: SkillType, data: dict) -> Optional[int]:
        """
        クライアントの skill action の data を受け取ってスキルを使う。

        Returns:
            実際に支払ったHP コスト。入力が不正・前提条件を満たさないときは None
        """
        if skill_type == SkillType.BOOST_HAND:
            # 複数の役を同時に強化する（件数と役名の検証は _validate_skill_cast で行う）
            yaku_names = data.get("yaku_names")
            if yaku_names is not None:
                if not isinstance(yaku_names, list):
                    return None
                return self.use_skill(user, skill_type, yaku_names=yaku_names)
            yaku_name = self.normalize_boost_target_yaku_name(data.get("yaku_name"))
            if yaku_name is None:
                return None
            return self.use_skill(user, skill_type, yaku_name)

        if skill_type == SkillType.PERSPECTIVE:
            target = next((p for p in self.state.players if p.player_id != user.player_id), None)
            if target is None:
                return None
            suit = data.get("suit")
            if suit is not None and suit not in PERSPECTIVE_SUITS:
                return None
            return self.cast_skill(user, skill_type, target=target, suit=suit)

        if skill_type == SkillType.MULLIGAN:
            return self.cast_skill(user, skill_type, target_hand_indexes=data.get("target_hand_indexes"))

        if skill_type in (SkillType.SPECIAL_VICTORY, SkillType.ASSAULT):
            return self.use_skill(user, skill_type)
        return None

    def resolve_mulligan(self, player: PlayerState, choice) -> Optional[dict]:
        """
        マリガンで手に入れる牌を選ぶ。choice は options（出した 3 枚 + 候補 3 枚）の番号 3 つ。
        選んだ順に、出した位置へ入る（[0, 1, 2] なら出した牌をそのまま残す）。手に入れなかった牌は牌山へ戻す。

        Returns:
            {"changes": [{"wall_index", "old_tile", "new_tile"}, ...]}。選択待ちがない・choice が不正なら None
        """
        pending = player.pending_mulligan
        if pending is None:
            return None
        options, indexes = pending["options"], pending["wall_indexes"]
        if not isinstance(choice, list) or len(choice) != len(indexes) or len(set(choice)) != len(choice):
            return None
        if not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(options) for c in choice):
            return None

        changes = [
            {"wall_index": wall_idx, "old_tile": player.wall[wall_idx], "new_tile": options[option]}
            for wall_idx, option in zip(indexes, choice)
        ]
        for change in changes:
            player.wall[change["wall_index"]] = change["new_tile"]
        self.state.round_state.reserved_tiles.extend(t for i, t in enumerate(options) if i not in choice)
        player.pending_mulligan = None
        self.round_events.append({
            "type": "mulligan_select", "player": player.player_id, "choices": list(choice),
            "changes": [dict(c) for c in changes],
        })
        return {"changes": changes}

    def _apply_voltage_discard(self, player: PlayerState, tile_id: int) -> None:
        """打牌に応じて、そのプレイヤーのボルテージを更新する。"""
        base_id = tile_id & 0b11111
        is_new = base_id not in self._voltage_seen_base_ids
        self._voltage_seen_base_ids.add(base_id)
        player.voltage_points, player.voltage_broken, player.voltage_resume_points = next_voltage(
            player.voltage_points, player.voltage_broken, player.voltage_resume_points, is_new,
        )

    def discard(self, player_id: str, wall_index: int) -> bool:
        """
        プレイヤーが牌を捨てたときの処理

        Args:
            player_id: 捨てたプレイヤーの ID
            wall_index: 捨てる牌の wall 内インデックス

        Returns:
            捨てた牌で上がりが成立した場合は True、それ以外は False
        """
        if self._pending_agari is not None:
            logger.warning("和了入力待ち中のため打牌不可: pending=%s", self._pending_agari)
            return False

        # 打牌開始時に直近の精算結果をクリア
        self._last_liquidation_result = None

        discarding_player = self.get_player_by_id(player_id)

        if discarding_player is None:
            return False

        if wall_index < 0 or wall_index >= len(discarding_player.wall):
            return False

        if wall_index in discarding_player.discarded_wall_indexes:
            logger.warning("既に打牌済みの wall_index が指定された: player=%s wall_index=%d", player_id, wall_index)
            return False

        winning_player = next(
            (player for player in self.state.players if player.player_id != player_id),
            None,
        )

        discarded_tile = discarding_player.wall[wall_index]
        discarding_player.discards.append(discarded_tile)
        discarding_player.discarded_wall_indexes.add(wall_index)

        self._apply_voltage_discard(discarding_player, discarded_tile) # ボルテージを適用する
        self.round_events.append({
            "type": "discard", "player": player_id, "wall_index": wall_index, "tile": discarded_tile,
            "voltage_points": discarding_player.voltage_points, "voltage_broken": discarding_player.voltage_broken,
        })

        self._invoke_callback(self.on_discarded, player_id, discarded_tile)

        # 打牌後、相手の待ち牌なら和了入力待ちへ移行（フリテン時は不可）
        discarded_tile_base = discarded_tile & 0b11111
        winning_waits = []
        winning_discard_bases = set()
        if winning_player is not None:
            winning_waits = [tile & 0b11111 for tile in winning_player.waits]
            winning_discard_bases = {tile & 0b11111 for tile in winning_player.discards}

        if (
            winning_player is not None
            and discarded_tile_base in winning_waits
            and discarded_tile_base not in winning_discard_bases
            and self.can_agari(winning_player, discarding_player, discarded_tile)
        ):
            logger.info(
                "和了入力待ち: discarder=%s winner=%s tile=%d tile_base=%d",
                player_id,
                winning_player.player_id,
                discarded_tile,
                discarded_tile_base,
            )
            self._pending_agari = {
                "winner_id": winning_player.player_id,
                "loser_id": player_id,
                "winning_tile": discarded_tile,
            }
            self._invoke_callback(
                self.on_agari_pending,
                winning_player.player_id,
                player_id,
                discarded_tile,
            )
            return False

        if winning_player is not None and discarded_tile_base in winning_waits and discarded_tile_base in winning_discard_bases:
            logger.info(
                "フリテンのため和了入力待ちを作成しない: discarder=%s winner=%s tile=%d tile_base=%d",
                player_id,
                winning_player.player_id,
                discarded_tile,
                discarded_tile_base,
            )

        if all(len(player.discards) >= DISCARDS_PER_PLAYER for player in self.state.players):
            logger.info(
                "流局: all players reached %d discards (%s)",
                DISCARDS_PER_PLAYER,
                {player.player_id: len(player.discards) for player in self.state.players},
            )
            self.end_round(is_draw=True)
            return False

        self._advance_player()
        return False

    def get_pending_agari(self) -> Optional[dict]:
        """和了入力待ち情報を取得する。"""
        return dict(self._pending_agari) if self._pending_agari is not None else None

    def resolve_pending_agari(self, player_id: str, accept: bool) -> Optional[bool]:
        """和了入力待ちを解決する。accept=True で精算、False で見送り。"""
        pending = self._pending_agari
        if pending is None:
            return None

        if pending.get("winner_id") != player_id:
            return None

        self._pending_agari = None
        winning_tile = pending.get("winning_tile")

        if accept:
            winner = self.get_player_by_id(player_id)
            if winner is None:
                return False

            if winning_tile is not None:
                winning_tile_base = winning_tile & 0b11111
                winner_discard_bases = {tile & 0b11111 for tile in winner.discards}
                if winning_tile_base in winner_discard_bases:
                    logger.info("フリテンのため和了不可: winner=%s tile_base=%d", winner.player_id, winning_tile_base)
                    if all(len(player.discards) >= DISCARDS_PER_PLAYER for player in self.state.players):
                        logger.info(
                            "和了拒否（フリテン）後に流局: all players reached %d discards (%s)",
                            DISCARDS_PER_PLAYER,
                            {player.player_id: len(player.discards) for player in self.state.players},
                        )
                        self.end_round(is_draw=True)
                        return False

                    self._advance_player()
                    return False

            try:
                winning_hand_tiles = [winner.wall[idx] for idx in winner.hand]
            except (IndexError, TypeError) as exc:
                logger.error(
                    "和了解決時の手牌 index 変換失敗: winner=%s hand=%s wall_len=%d error=%s",
                    winner.player_id,
                    winner.hand,
                    len(winner.wall),
                    exc,
                )
                return False

            return self.liquidation(player_id, winning_hand_tiles + [winning_tile], winning_tile=winning_tile)

        if all(len(player.discards) >= DISCARDS_PER_PLAYER for player in self.state.players):
            logger.info(
                "和了見送り後に流局: all players reached %d discards (%s)",
                DISCARDS_PER_PLAYER,
                {player.player_id: len(player.discards) for player in self.state.players},
            )
            self.end_round(is_draw=True)
            return False

        self._advance_player()
        return False

    @staticmethod
    def _get_multiplier_label(tier: Optional[int]) -> str:
        """精算倍率の段階から呼称を返す。"""
        return "満貫未満" if tier is None else MULTIPLIER_TIER_LABELS[tier]

    def _get_voltage_multiplier(self, points: int) -> float:
        """ボルテージポイントから勝者獲得額の倍率を返す。"""
        return voltage_multiplier(points)

    def _get_voltage_level(self, points: int) -> int:
        """ボルテージポイントから段階番号を返す。0〜4。"""
        if points >= 14:
            return 4
        if points >= 9:
            return 3
        if points >= 5:
            return 2
        if points >= 2:
            return 1
        return 0

    def _get_voltage_resume_points(self, points: int) -> int:
        """ブレイク後、次の初出牌で再開するポイントを返す。"""
        return voltage_resume_points(points)

    @staticmethod
    def _is_tanki_wait_agari(hand: list[int], winning_tile: int, winner_waits: list[int]) -> bool:
        """
        単騎待ちボーナス（放銃した側の損失が 2 倍）の対象の和了か。
        待ちが 1 種類だけで、その牌で和了ったとき。雀頭の待ちかどうかは問わない（嵌張・辺張の 1 種類待ちも含む）。
        hand は判定に使わない（呼び出し側の形をそろえるために受け取る）。
        """
        if winning_tile is None or not winner_waits:
            return False
        wait_bases = {wait & 0b11111 for wait in winner_waits}
        return wait_bases == {winning_tile & 0b11111}

    def _is_ippatsu_agari(self, winner: PlayerState, loser: PlayerState, base_yaku_list: list[str]) -> bool:
        """一発成立かどうかを返す。両者の一打目までを一発圏とする。"""
        if Yaku.RICHI.japanese_name not in base_yaku_list:
            return False

        return len(winner.discards) <= 1 and len(loser.discards) == 1

    def _is_houtei_raoyui(self, loser: PlayerState, winning_tile: Optional[int]) -> bool:
        """河底撈魚成立かどうかを返す。後手側の最後の打牌（DISCARDS_PER_PLAYER 打目）を最後の捨て牌とみなす。"""
        if winning_tile is None or len(loser.discards) != DISCARDS_PER_PLAYER:
            return False

        first_player = self.state.players[self.state.round_state.first_player_index]
        return loser.player_id != first_player.player_id

    def _get_win_context_yaku(
        self,
        winner: PlayerState,
        loser: PlayerState,
        winning_tile: Optional[int],
        base_yaku_list: list[str],
    ) -> list[str]:
        """和了時の状況役を返す。初期の手牌評価には使わない。"""
        context_yaku: list[str] = []

        if winning_tile is None:
            return context_yaku

        if self._is_ippatsu_agari(winner, loser, base_yaku_list):
            context_yaku.append(Yaku.IPPATSU.japanese_name)

        if self._is_houtei_raoyui(loser, winning_tile):
            context_yaku.append(Yaku.KAWA_ZO.japanese_name)

        return context_yaku

    def _calc_agari_han(
        self,
        winner: PlayerState,
        loser: PlayerState,
        hand: list[int],
        winning_tile: Optional[int],
    ) -> tuple[list[str], int, Optional[int], dict[str, int]]:
        """
        和了したときの役・翻数・精算倍率の段階を数える（状況役を含む）。

        Returns:
            (役のリスト, 翻数, 役強化を含めた段階（満貫未満は None）, 役強化の辞書)
        """
        base_yaku_list = HandAnalyzer.enum_yaku(hand, winning_tile=winning_tile)
        base_yaku_list += self._get_win_context_yaku(winner, loser, winning_tile, base_yaku_list)

        # 役強化は翻数ではなく段階を上げるので、満貫に届くかどうかは変わらない
        boost_bonus_map = self._effective_boost_bonus_map(winner)
        han = sum(Yaku.get_han_by_name(name) for name in base_yaku_list)
        return base_yaku_list, han, promoted_tier(han, base_yaku_list, boost_bonus_map), boost_bonus_map

    def can_agari(self, winner: PlayerState, loser: PlayerState, winning_tile: int) -> bool:
        """
        その牌でロンしたら満貫以上で精算されるか。

        満貫未満の手は和了れない。打牌の時点でこれを確かめ、和了れない牌ではロン確認を出さない。
        数え方は liquidation と同じ（状況役の一発・河底撈魚を含む。役強化では満貫に届かない）。
        """
        try:
            hand = [winner.wall[idx] for idx in winner.hand] + [winning_tile]
        except (IndexError, TypeError):
            return False
        if not HandAnalyzer.is_win(hand):
            return False
        _, _, tier, _ = self._calc_agari_han(winner, loser, hand, winning_tile)
        return tier is not None

    def liquidation(self, player_id: str, hand: list[int], winning_tile: Optional[int] = None) -> bool:
        """
        清算処理

        Args:
            player_id: 清算対象のプレイヤーのID
            hand: 清算対象の手牌
        Returns:
            上がりが成立していれば True、そうでなければ False
        """
        winner = self.get_player_by_id(player_id)
        if winner is None:
            return False

        loser = next((player for player in self.state.players if player.player_id != player_id), None)
        if loser is None:
            return False

        is_win = HandAnalyzer.is_win(hand)
        base_yaku_list, han, tier, boost_bonus_map = self._calc_agari_han(winner, loser, hand, winning_tile)
        is_mangan = tier is not None
        if not is_win or not is_mangan:
            logger.debug("上がり条件不成立: player=%s  is_win=%s  check_mangan=%s", player_id, is_win, is_mangan)
            return False

        is_tanki_wait = self._is_tanki_wait_agari(hand, winning_tile, winner.waits)

        # 役倍率（跳満 1.5倍 / 倍満 2倍 / 三倍満 3倍 / 役満 4倍 / ダブル役満 8倍。役強化で昇格）
        display_yaku_list = self._build_display_yaku_list(base_yaku_list, boost_bonus_map)
        base_tier = han_to_tier(han)
        multiplier = MULTIPLIER_TIERS[tier]
        logger.info("精算: winner=%s  yaku=%s  han=%d  tier=%s→%s  multiplier=%.1f",
                    winner.player_id, display_yaku_list, han, MULTIPLIER_TIER_LABELS[base_tier],
                    MULTIPLIER_TIER_LABELS[tier], multiplier)

        # ボルテージの倍率
        voltage_multiplier = self._get_voltage_multiplier(
            winner.voltage_points
        )

        # 流局持ち越しは「支払い設定額を固定」し、清算時のみ累積回数を掛ける。
        carry_rounds = self._carry_over_draw_count + 1
        winner_effective_bet = winner.bet * carry_rounds
        loser_effective_bet = loser.bet * carry_rounds

        # 勝者: 累積済み掛け金 × 役倍率分 x ボルテージ倍率を獲得
        winner_original_gain = int(winner_effective_bet * multiplier * voltage_multiplier)
        assault_applied = winner.assault_active_this_round
        assault_bonus_damage = winner_original_gain if assault_applied else 0
        winner_gain = 0 if assault_applied else winner_original_gain
        winner.health += winner_gain
        winner.cumulative_earned_points += winner_gain

        # 敗者: 単騎待ち和了時のみ支払いを倍化（勝者獲得量は据え置き）
        loser_loss_multiplier = 2 if is_tanki_wait else 1
        loser_loss = int(loser_effective_bet * multiplier * loser_loss_multiplier) + assault_bonus_damage
        # 残りの HP を超える損失は、マイナスのまま記録する（どれだけ超えたかを見せるため）。
        # 0 以下になれば end_round でその場でゲーム終了になる
        loser.health = loser.health - loser_loss

        self._last_liquidation_result = {
            "winner_id": winner.player_id,
            "loser_id": loser.player_id,
            "yaku": display_yaku_list,
            "base_yaku": base_yaku_list,
            "display_yaku": display_yaku_list,
            "base_han": han,
            "bonus_han": 0,  # 役強化は翻を上げなくなった（互換のため残す）
            "han": han,
            "multiplier": multiplier,
            "multiplier_label": MULTIPLIER_TIER_LABELS[tier],
            "base_multiplier": MULTIPLIER_TIERS[base_tier],
            "promoted_tiers": tier - base_tier,
            "voltage_multiplier": voltage_multiplier,
            "winner_voltage_points": winner.voltage_points,
            "winner_voltage_level": self._get_voltage_level(winner.voltage_points),
            "winner_bet": winner_effective_bet,
            "loser_bet": loser_effective_bet,
            "winner_original_gain": winner_original_gain,
            "winner_gain": winner_gain,
            "loser_loss": loser_loss,
            "loser_loss_multiplier": loser_loss_multiplier,
            "assault_applied": assault_applied,
            "assault_bonus_damage": assault_bonus_damage,
            "is_tanki_wait": is_tanki_wait,
            "carry_over_draw_count": self._carry_over_draw_count,
            "winner_health": winner.health,
            "loser_health": loser.health,
        }

        self.end_round()
        return True

    def _prepare_next_round(self) -> None:
        """次局開始に必要な状態を準備する。"""
        self.state.round_state.round_number += 1
        self._pending_agari = None
        self._voltage_seen_base_ids.clear()
        self.state.players = [
            PlayerState(
                player_id=p.player_id,
                health=p.health,
                hand=[],
                wall=[],
                waits=[],
                discards=[],
                discarded_wall_indexes=set(),
                bet=p.bet if self._carry_over_bets else 0,
                base_bet=p.base_bet if self._carry_over_bets else 0,
                cumulative_earned_points=p.cumulative_earned_points,
                special_victory_count=p.special_victory_count,
                boost_hand_bonus=p.boost_hand_bonus.copy(),
                exposed_hand_indexes=set(),  # 局が変わるたびにリセット（手牌が変わるため古い indexes は無効）
                skill_cost_rate=p.skill_cost_rate,
                boost_hand_targets=p.boost_hand_targets,
                opening_boost_count=p.opening_boost_count,
                initial_perspective_count=p.initial_perspective_count,
            )
            for p in self.state.players
        ]

    def get_next_round_ready_players(self) -> list[str]:
        """次局進行承認済みプレイヤー一覧を返す。"""
        return sorted(self._next_round_ready_players)

    def confirm_next_round(self, player_id: str) -> bool:
        """プレイヤーが次局進行を承認する。全員承認で次局開始。"""
        if self.state.round_state.status != RoundStatus.ROUND_END_WAITING:
            return False

        player = self.get_player_by_id(player_id)
        if player is None:
            return False

        self._next_round_ready_players.add(player_id)
        logger.info(
            "次局承認: player=%s ready=%s/%s",
            player_id,
            len(self._next_round_ready_players),
            self.num_players,
        )

        if len(self._next_round_ready_players) < self.num_players:
            return True

        self._next_round_ready_players.clear()
        self._prepare_next_round()
        self._start_round()
        return True

    def end_round(self, is_draw: bool = False) -> None:
        """局を終了"""
        logger.info("局終了: is_draw=%s  round=%d", is_draw, self.state.round_state.round_number)
        if self.state.round_state.round_number >= self.max_rounds:
            logger.info("最大ラウンド到達によるゲーム終了: round=%d", self.state.round_state.round_number)
            self._invoke_callback(self.on_round_end, is_draw)
            self._on_game_end("max_rounds")
            return

        # 精算後に HP が 0 以下のプレイヤーがいれば、その場でゲーム終了。
        # 流局では HP は減らない（全額を賭けて 0 になった場合は、下の「次局の掛け金を払えない」で終わる）
        if not is_draw and any(p.health <= 0 for p in self.state.players):
            dead = [p.player_id for p in self.state.players if p.health <= 0]
            logger.info("HP 0 によるゲーム終了: players=%s", dead)
            self._invoke_callback(self.on_round_end, is_draw)
            self._on_game_end("hp_zero")
            return

        # 勝利時の獲得累計点数が 30000 に到達したらゲーム終了
        point_winner = next((p for p in self.state.players if p.cumulative_earned_points >= 30000), None)
        if point_winner is not None:
            logger.info(
                "累計獲得点数到達によるゲーム終了: player=%s cumulative_earned_points=%d",
                point_winner.player_id,
                point_winner.cumulative_earned_points,
            )
            self._invoke_callback(self.on_round_end, is_draw)
            self._on_game_end("cumulative_earned_points")
            return

        # 次の局の掛け金を払えなければ、その場でゲーム終了。次の局を始めてから手牌選択の後に気づくのでは遅い。
        # 流局後は、同じ掛け金が積み上がって「掛け金 × (流局の回数 + 1)」で精算される。掛け金は先払いしないので
        # 流局しても HP は減らないが、積み上がった額を HP で払えなければ負け（先払いのころ、もう一度同じ額を
        # 払えなければ負けだったのと同じ）。流局以外は最低掛け金
        unpayable = [
            p.player_id for p in self.state.players
            if p.health < (self.carry_over_stake(p, self._carry_over_draw_count + 1) if is_draw else self.get_minimum_bet(p))
        ]
        if unpayable:
            logger.info("次局の掛け金を払えないためゲーム終了: players=%s is_draw=%s", unpayable, is_draw)
            self.bet_unpayable_players = unpayable
            self._invoke_callback(self.on_round_end, is_draw)
            self._on_game_end("bet_unpayable")
            return

        if is_draw:
            self._carry_over_draw_count += 1
        else:
            self._carry_over_draw_count = 0

        self._carry_over_bets = is_draw
        self._next_round_ready_players.clear()
        self._set_phase(RoundStatus.ROUND_END_WAITING)
        self._invoke_callback(self.on_round_end, is_draw)

    def _on_game_end(self, reason: str) -> None:
        """
        ゲーム終了時の処理

        Args:
            reason: 終わった理由（"hp_zero" / "cumulative_earned_points" / "bet_unpayable" / "max_rounds"）
        """
        if self.game_end_reason is None:
            self.game_end_reason = reason
        self._invoke_callback(self.on_game_end)

    def _set_phase(self, new_status: RoundStatus) -> None:
        """ラウンドフェーズを変更"""

        if self.state.round_state.status != new_status:
            self.state.round_state.status = new_status

            self._invoke_callback(self.on_phase_change, new_status)

    def _invoke_callback(self, callback: Optional[Callable], *args) -> None:
        if callback:
            try:
                callback(*args)
            except Exception:
                logger.exception("コールバック %s で例外発生", getattr(callback, '__name__', repr(callback)))

    def get_last_liquidation_result(self) -> Optional[dict]:
        """直近の精算結果を取得する。"""
        return self._last_liquidation_result

    def get_current_player(self) -> PlayerState:
        """現在のプレイヤーを取得"""
        return self.state.players[self.state.round_state.current_player_index]

    def get_player_by_id(self, player_id: str) -> PlayerState:
        """プレイヤーIDからプレイヤー状態を取得"""
        return next((p for p in self.state.players if p.player_id == player_id), None)

    def update_player_state(self, player_id: str, **kwargs) -> None:
        """プレイヤーの状態を更新"""
        player = self.get_player_by_id(player_id)
        if not player:
            return

        for key, value in kwargs.items():
            if key in player.__dataclass_fields__:
                setattr(player, key, value)

    def get_waits(self, hand_indexes: list[int], player: PlayerState) -> list[tuple[int, bool, list[str], list[str], int, float, str]]:
        """
        手牌から待ち牌を取得

        Args:
            hand_indexes: 手牌の山内 index リスト
            player: プレイヤー

        Returns:
            待ち牌のリスト
                - 待ち牌のID
                - 満貫以上の待ちかどうか
                - 表示用役のリスト（役名+強化回数）
                - 生の役のリスト
                - 翻数（役強化は翻を上げない）
                - 点数倍率（役強化の昇格込み）
                - 点数倍率の呼称
        """
        # index から牌 ID に変換
        hand_tiles = [player.wall[idx] for idx in hand_indexes]
        boost_bonus_map = self._effective_boost_bonus_map(player)
        waits = self.evaluate_waits(hand_tiles, player.wall, self.state.round_state.dora_id, boost_bonus_map)
        if waits is None:
            return None

        result = []
        for w, base_yaku_list, han, multiplier in waits:
            display_yaku_list = self._build_display_yaku_list(base_yaku_list, boost_bonus_map)
            multiplier_label = self._get_multiplier_label(promoted_tier(han, base_yaku_list, boost_bonus_map))
            result.append((w, multiplier > 0, display_yaku_list, base_yaku_list, han, multiplier, multiplier_label))
        return result

    @staticmethod
    def evaluate_waits(
        hand_tiles: list[int],
        wall: list[int],
        dora_id: Optional[int],
        boost_bonus_map: dict[str, int],
    ) -> Optional[list[tuple[int, list[str], int, float]]]:
        """
        聴牌なら、待ちごとに (和了牌の牌ID, 役のリスト, 翻数, 精算倍率（役強化の昇格込み。満貫未満は 0）) を返す。
        聴牌でなければ None。
        ドラの待ちはドラの牌IDで役を数える。
        """
        waits = HandAnalyzer.get_tenpai_waiting_tiles(hand_tiles, wall)
        if not waits:
            return None
        result = []
        for kind in waits:
            tile = kind + 32 if kind == dora_id else kind
            yaku = HandAnalyzer.enum_yaku(hand_tiles + [tile], winning_tile=tile)
            han = sum(Yaku.get_han_by_name(y) for y in yaku)
            result.append((tile, yaku, han, promoted_multiplier(han, yaku, boost_bonus_map)))
        return result


    def _advance_player(self):
        """プレイヤーを次に進める"""
        self.state.round_state.current_player_index = (self.state.round_state.current_player_index + 1) % self.num_players

    def get_game_state(self) -> dict:
        """ゲーム状態を辞書で取得（API用）"""
        return {
            "status": self.state.round_state.status.value if self.state.round_state.status else None,
            "round": self.state.round_state.round_number,
            "dora_id": self.state.round_state.dora_id,
            "current_player": self.get_current_player().player_id,
            "players": [
                {
                    "id": p.player_id,
                    "health": p.health,
                    "cumulative_earned_points": p.cumulative_earned_points,
                    "hand": p.hand,
                    "wall": p.wall,
                    "waits": p.waits,
                    "discards": p.discards
                }
                for p in self.state.players
            ]
        }