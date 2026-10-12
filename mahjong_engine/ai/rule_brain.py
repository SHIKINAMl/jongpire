"""
よわい／ふつう の考え方

規則ごとに点数を付け、点数の高い行動を選ぶ。重み・しきい値はすべて cpu_config.json から受け取る。
入力は CpuObservation（視界）だけで、エンジンの状態は読まない。
出力は人間のクライアントと同じ action の data。
"""
import itertools
import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ..engine.game_engine import (
    MULLIGAN_CANDIDATE_COUNT,
    PERSPECTIVE_SUITS,
    GameEngine,
    next_voltage,
    promoted_multiplier,
    tiles_to_wall_indexes,
    voltage_multiplier,
)
from ..engine.hand_analyzer import HandAnalyzer
from ..engine.game_state import SkillType
from .observation import CpuObservation

logger = logging.getLogger(__name__)

HONOR_START = 27  # 牌種 27 以降は字牌（東・西）
POINT_VICTORY = 30000  # 和了で得た累計点数がこれに届くと勝ち（GameEngine.end_round）
HAND_SIZE = 13
OPPONENT_HAND_SIZE = 13
# 相手が手牌に持つ牌と、待ちの牌種との近さ（同じ牌種 / 1 つ隣 / 2 つ隣）
CLOSENESS_BY_DISTANCE = (1.0, 0.6, 0.3)


def tile_kind(tile: int) -> int:
    """牌ID から牌種（下位5ビット）を取り出す。"""
    return tile & 0b11111


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class WaitInfo:
    kind: int
    han: int
    multiplier: float  # 満貫未満なら 0（和了れない）
    yaku: tuple[str, ...]
    available: float  # 和了れそうな枚数の見積もり
    tanki: bool = False  # 単騎待ちボーナスの対象（待ちが 1 種類）。和了ると相手の損失が 2 倍


@dataclass(frozen=True)
class HandPlan:
    hand_indexes: tuple[int, ...]
    waits: tuple[WaitInfo, ...]
    score: float
    strength: float  # 0〜1。掛け金やスキルの判断に使う
    # この案を選んだときに比べた候補。山が変わらない限り、選び直しは探し直さずに採点し直すだけで済む
    candidates: tuple[tuple[int, ...], ...] = field(default=(), compare=False)

    @property
    def best_han(self) -> int:
        return max((w.han for w in self.waits), default=0)

    @property
    def wait_kinds(self) -> set[int]:
        return {w.kind for w in self.waits}

    @property
    def expected_multiplier(self) -> float:
        """和了れそうな枚数で重み付けした精算倍率。"""
        winning = [w for w in self.waits if w.multiplier > 0]
        total = sum(w.available for w in winning)
        if total <= 0:
            return max((w.multiplier for w in winning), default=0.0)
        return sum(w.multiplier * w.available for w in winning) / total


def possibly_held_exposed_kinds(obs: CpuObservation) -> Counter:
    """
    透視で公開された相手の牌のうち、まだ捨てられていない可能性がある牌（牌種ごとの枚数）。

    どの位置の牌が捨てられたかは通知されないため、同じ牌ID の捨て牌を1枚ずつ差し引いて見積もる。
    """
    remaining = Counter(obs.opponent.exposed.values())
    remaining.subtract(Counter(obs.opponent.discards))
    held: Counter = Counter()
    for tile, count in remaining.items():
        if count > 0:
            held[tile_kind(tile)] += count
    return held


def rest_indexes(obs: CpuObservation, hand_indexes: tuple[int, ...]) -> list[int]:
    """手牌以外で、まだ打っていない wall index。"""
    hand = set(hand_indexes)
    return [
        i for i in range(len(obs.me.wall))
        if i not in hand and i not in obs.me.discarded_wall_indexes
    ]


def boostable_yaku(plan: "HandPlan") -> list[str]:
    """役強化で強化できる、この手牌で和了れる待ちの役。"""
    return sorted({
        name
        for wait in plan.waits if wait.multiplier > 0
        for name in wait.yaku
        if GameEngine.normalize_boost_target_yaku_name(name) is not None
    })


def replan_needs_search(skill: str) -> bool:
    """スキルの後、手牌の候補を探し直す必要があるか（山が変わるのはマリガンだけ）。"""
    return skill == SkillType.MULLIGAN.value


# ========== 安全な手（考え方が失敗したときに必ず通る行動） ==========

def fallback_hand_indexes(obs: CpuObservation) -> list[int]:
    if len(obs.me.hand) == HAND_SIZE:
        return list(obs.me.hand)
    return list(range(min(HAND_SIZE, len(obs.me.wall))))


def fallback_bet(obs: CpuObservation) -> int:
    return obs.me.bet_unit


def fallback_discard(obs: CpuObservation) -> int:
    candidates = rest_indexes(obs, obs.me.hand)
    if not candidates:
        candidates = [i for i in range(len(obs.me.wall)) if i not in obs.me.discarded_wall_indexes]
    return candidates[0]


@dataclass(frozen=True)
class _HandContext:
    """手牌の候補によらない、視界から決まる値（候補ごとに数え直さないため）。"""
    wall: list[int]
    own_counts: Counter
    opponent_discards: Counter
    held_exposed: Counter
    known_reserved: Counter  # 牌山にあると分かっている牌（相手からは出てこない）

# マリガンで手に入れる牌を選ぶときの、1 回の手牌探索の制限時間（秒）。組合せごとに探すので短くする
MULLIGAN_SEARCH_TIME_LIMIT = 0.3
# マリガンで手に入れる牌を選ぶときの、1 回の手牌探索の候補の数。20 通りを探すので、手牌選択（candidate_limit）より少なくする
MULLIGAN_CANDIDATE_LIMIT = 40


class RuleBrain:
    """よわい（level="easy"）／ふつう（level="normal"）の考え方。"""

    def __init__(self, level: str, config: dict, rng: Optional[random.Random] = None):
        if level not in ("easy", "normal"):
            raise ValueError(f"Unsupported level: {level}")
        self.level = level
        self.config = config
        self.rng = rng or random.Random()

    # ========== 手牌選択 ==========

    @staticmethod
    def _context(obs: CpuObservation) -> _HandContext:
        wall = list(obs.me.wall)
        return _HandContext(
            wall=wall,
            own_counts=Counter(tile_kind(t) for t in wall),
            opponent_discards=Counter(tile_kind(t) for t in obs.opponent.discards),
            held_exposed=possibly_held_exposed_kinds(obs),
            known_reserved=Counter(tile_kind(t) for t in obs.me.known_reserved),
        )

    def plan_hand(
        self, obs: CpuObservation, time_limit: Optional[float] = None, candidate_limit: Optional[int] = None,
    ) -> Optional[HandPlan]:
        """
        聴牌の候補を挙げて点数を付け、一番高いものを返す。満貫の候補がなければ None。
        聴牌形の探索は重いので、イベントループの外（別スレッド）から呼ぶ。

        Args:
            time_limit: 探索の制限時間（秒）。省略すると設定の search_time_limit
            candidate_limit: 探す候補の数。省略すると設定の candidate_limit
        """
        cfg = self.config["hand_selection"]
        wall = list(obs.me.wall)

        candidates: list[tuple[int, ...]] = []
        if len(obs.me.hand) == HAND_SIZE:
            candidates.append(tuple(sorted(obs.me.hand)))
        # 候補が少ない山では全列挙になって終わらないことがあるので、制限時間で打ち切る
        deadline = time.monotonic() + (cfg["search_time_limit"] if time_limit is None else time_limit)
        found = HandAnalyzer.search_tenpai(
            wall, wall, obs.dora_id, limit=candidate_limit or cfg["candidate_limit"], rng=self.rng, deadline=deadline,
        )
        for tiles in found:
            indexes = tiles_to_wall_indexes(wall, tiles)
            if indexes is not None and tuple(sorted(indexes)) not in candidates:
                candidates.append(tuple(sorted(indexes)))
        return self._best_of(obs, tuple(candidates))

    def replan(self, obs: CpuObservation, plan: Optional[HandPlan], skill: str) -> Optional[HandPlan]:
        """
        スキルを使った後に手牌を選び直す。

        山が変わるマリガンの後だけ探し直す。透視（見える牌）・役強化（翻数）の後は、
        前の候補を採点し直すだけで足りる。強襲は手牌の評価に関係しない。
        """
        if plan is None or replan_needs_search(skill):
            return self.plan_hand(obs)
        if skill == SkillType.ASSAULT.value:
            return plan
        return self._best_of(obs, plan.candidates)

    def _best_of(self, obs: CpuObservation, candidates: tuple[tuple[int, ...], ...]) -> Optional[HandPlan]:
        ctx = self._context(obs)
        best: Optional[HandPlan] = None
        for indexes in candidates:
            plan = self.evaluate_hand(obs, indexes, ctx=ctx)
            if plan is not None and (best is None or plan.score > best.score):
                best = plan
        if best is None:
            return None
        return HandPlan(best.hand_indexes, best.waits, best.score, best.strength, candidates)

    def evaluate_hand(
        self,
        obs: CpuObservation,
        hand_indexes: tuple[int, ...],
        ctx: Optional[_HandContext] = None,
    ) -> Optional[HandPlan]:
        """手牌に点数を付ける。聴牌でない・満貫の待ちがない場合は None。"""
        cfg = self.config["hand_selection"]
        ctx = ctx or self._context(obs)
        hand_tiles = [ctx.wall[i] for i in hand_indexes]
        scored = GameEngine.evaluate_waits(hand_tiles, ctx.wall, obs.dora_id, obs.me.boost_hand_bonus)
        if scored is None:
            return None

        rest_counts = Counter(tile_kind(ctx.wall[i]) for i in rest_indexes(obs, hand_indexes))
        waits: list[WaitInfo] = []
        for winning_tile, yaku, han, multiplier in scored:
            kind = tile_kind(winning_tile)
            live = max(0, 4 - ctx.own_counts[kind] - ctx.opponent_discards[kind] - ctx.known_reserved[kind])
            available = max(
                0.0,
                live
                + cfg["bonus_wait_seen_in_opponent_wall"] * ctx.held_exposed[kind]
                - cfg["penalty_wait_in_own_wall"] * rest_counts[kind],
            )
            # 単騎待ちボーナスは、待ちが 1 種類なら形を問わず付く（GameEngine._is_tanki_wait_agari）
            tanki = len(scored) == 1
            waits.append(WaitInfo(kind, han, multiplier, tuple(yaku), available, tanki))

        score = self._score_waits(waits)
        if score is None:
            return None
        strength = _clamp(score / cfg["strong_hand_score"], 0.0, 1.0)
        return HandPlan(tuple(sorted(hand_indexes)), tuple(waits), score, strength)

    def _score_waits(self, waits) -> Optional[float]:
        """待ちの点数。和了れる（満貫以上の）待ちがなければ None。"""
        cfg = self.config["hand_selection"]
        winning = [w for w in waits if w.multiplier > 0]
        if not winning:
            return None
        # 単騎は和了ると相手の損失が 2 倍になる分、tanki_bonus だけ高く見る
        return sum(
            (w.multiplier ** cfg["weight_han_multiplier"]) * (w.available ** cfg["weight_wait_count"])
            * (1 + cfg["tanki_bonus"] if w.tanki else 1)
            for w in winning
        )

    def best_boost_combo(
        self, obs: CpuObservation, plan: HandPlan
    ) -> tuple[Optional[tuple[str, ...]], float]:
        """
        役強化で強化する役の組合せのうち、点数がいちばん上がるもの（と、そのときの点数）。
        上がる組合せがなければ (None, plan.score)。手牌の評価はやり直さず、待ちの倍率を昇格し直して採点する。
        強化済みの役はもう強化できない。1 回で boost_hand_targets 個まで（同じ役は重ねられない）。
        """
        best_combo: Optional[tuple[str, ...]] = None
        best_score = plan.score
        current = obs.me.boost_hand_bonus
        candidates = [name for name in boostable_yaku(plan) if name not in current]
        for size in range(1, min(obs.me.boost_hand_targets, len(candidates)) + 1):
            for combo in itertools.combinations(candidates, size):
                boost_map = {**current, **{name: 1 for name in combo}}
                boosted = [
                    WaitInfo(w.kind, w.han, promoted_multiplier(w.han, list(w.yaku), boost_map), w.yaku, w.available, w.tanki)
                    for w in plan.waits
                ]
                score = self._score_waits(boosted) or 0.0
                if score > best_score:
                    best_combo, best_score = combo, score
        return best_combo, best_score

    def final_hand(self, obs: CpuObservation, plan: Optional[HandPlan]) -> list[int]:
        """スキルを使い終えた後に確定する手牌（wall index）。規則では、点数がいちばん高い候補。"""
        return list(plan.hand_indexes) if plan is not None else fallback_hand_indexes(obs)

    # ========== スキル ==========

    def skill_features(self, obs: CpuObservation, plan: HandPlan) -> dict[str, float]:
        """
        スキルを使うかどうかの判断材料（視界から分かることだけ）。学習用の測定（training/skill_strategy.py）と
        同じものを使い、cpu_config.json の skill.value_model の重みを掛けてスキルの効果を見積もる。

        - 和了りの見込み: 手の点数・期待倍率・最大倍率・和了れる待ちの数・残っていそうな待ち牌の数
        - 自分の牌: 手牌の外にある自分の待ち牌（山から出ない死に牌）・山で重なっている牌（マリガンで手放せる牌）・
          役強化で点数がどれだけ上がるか・強化済みの役の数
        - 相手の牌の予想: 相手の山に残っていそうな枚数がいちばん多い色の枚数（透視で見える量）・公開済みの割合・
          相手の役強化の数（相手の手の高さ）
        """
        winning = [w for w in plan.waits if w.multiplier > 0]
        wall_counts = Counter(tile_kind(t) for t in obs.me.wall)
        rest = rest_indexes(obs, plan.hand_indexes)
        rest_kinds = [tile_kind(obs.me.wall[i]) for i in rest]
        _combo, boosted_score = self.best_boost_combo(obs, plan)

        seen = Counter(tile_kind(t) for t in obs.opponent.exposed.values())
        seen.update(tile_kind(t) for t in obs.opponent.discards)
        unseen_by_suit = [
            sum(max(0, 4 - wall_counts[k] - seen[k]) for k in kinds) for kinds in PERSPECTIVE_SUITS.values()
        ]
        return {
            "score": plan.score,
            "expected_multiplier": plan.expected_multiplier,
            "max_multiplier": max((w.multiplier for w in winning), default=0.0),
            "winning_waits": float(len(winning)),
            "available": sum(w.available for w in winning),
            "dead_waits": float(sum(1 for k in rest_kinds if k in plan.wait_kinds)),
            "duplicated_rest": float(sum(1 for k in rest_kinds if wall_counts[k] >= 3)),
            "boost_gain_ratio": (boosted_score - plan.score) / max(plan.score, 1e-9),
            "my_boosts": float(len(obs.me.boost_hand_bonus)),
            "opp_unseen_best_suit": float(max(unseen_by_suit)),
            "opp_exposed_ratio": len(obs.opponent.exposed) / max(1, obs.opponent.wall_size),
            "opp_boosts": float(len(obs.opponent.opening_boosts) + obs.opponent.boost_use_count),
        }

    def bet_features(self, obs: CpuObservation, plan: HandPlan) -> dict[str, float]:
        """
        掛け金の判断材料（視界から分かることだけ）。skill_features に、相手の様子と対局の状況を足したもの。
        学習用の記録（training/bet_data.py）と同じものを使い、cpu_config.json の bet.model で
        和了る確率・放銃する確率・倍率を見積もる。

        - 相手がこの局で使ったスキル（マリガン・透視は手や守りが良くなる、役強化は倍率が上がる）
        - 両者の HP・累計獲得点数、流局の持ち越しの回数
        """
        opp_skills = Counter(obs.opponent.skills_this_round)
        return {
            **self.skill_features(obs, plan),
            "opp_used_mulligan": float(opp_skills[SkillType.MULLIGAN.value]),
            "opp_used_perspective": float(opp_skills[SkillType.PERSPECTIVE.value]),
            "opp_used_boost_hand": float(opp_skills[SkillType.BOOST_HAND.value]),
            "opp_used_assault": float(opp_skills[SkillType.ASSAULT.value]),
            "my_health": obs.me.health / 20000,
            "opp_health": obs.opponent.health / 20000,
            "my_points": obs.me.cumulative_earned_points / 30000,
            "opp_points": obs.opponent.cumulative_earned_points / 30000,
            "carry_over": float(obs.carry_over_draw_count),
        }

    def next_skill(self, obs: CpuObservation, plan: Optional[HandPlan]) -> Optional[dict[str, Any]]:
        """次に使うスキルの action の data を返す。使わないなら None。手牌選択フェーズで繰り返し呼ぶ。"""
        if plan is None:
            return None
        for decide in (self._decide_perspective, self._decide_mulligan, self._decide_boost_hand, self._decide_assault):
            data = decide(obs, plan)
            if data is not None:
                return data
        return None

    def _can_afford(self, obs: CpuObservation, skill: str) -> bool:
        # 流局の続きの局では、積み上がった掛け金を払えなくなるほど使わない（サーバーも弾く）
        reserve = max(self.config["skill"]["min_health_after_use"], obs.me.skill_health_floor)
        return obs.me.health - obs.me.skill_costs[skill] >= reserve

    def _uses(self, obs: CpuObservation, skill: str) -> int:
        return obs.my_skills_this_round.get(skill, 0)

    def _value_model(self, skill: str) -> Optional[dict]:
        """スキルの効果の見積もり式（ふつう以上）。よわいは持たず、決まった条件で使う。"""
        return self.config["skill"].get("value_model", {}).get(skill)

    def skill_stake(self, obs: CpuObservation, plan: HandPlan) -> float:
        """
        効果の見積もりに使う、この局の掛け金（流局の持ち越しを含む）。
        効果は掛け金にほぼ比例する。弱い手は掛け金が低いが、スキルで手が良くなれば上げるので、下限を設ける。
        """
        cfg = self.config["skill"]
        carry = obs.carry_over_draw_count
        bet = obs.me.bet if carry > 0 and obs.me.bet > 0 else self.rule_bet(obs, plan)
        return max(bet, cfg["stake_floor"]) * (carry + 1)

    def estimate_skill_value(self, obs: CpuObservation, plan: HandPlan, skill: str) -> Optional[float]:
        """
        スキルを（もう 1 回）使ったときの、この局の HP の差の見込み（コストは含めない）。見積もり式がなければ None。

        掛け金 1000 あたりの効果 = 切片 + Σ 重み × 判断材料（skill_features）。同じ局の 2 回目以降は
        repeat_factor を掛ける。それに、この局の掛け金 / 1000 を掛ける。
        """
        per_1000 = self.skill_value_per_1000(obs, plan, skill)
        if per_1000 is None:
            return None
        return per_1000 * self.skill_stake(obs, plan) / 1000

    def skill_value_per_1000(
        self, obs: CpuObservation, plan: HandPlan, skill: str, features: Optional[dict[str, float]] = None,
    ) -> Optional[float]:
        """掛け金 1000 あたりの効果の見積もり（2 回目以降は repeat_factor 込み）。見積もり式がなければ None。"""
        model = self._value_model(skill)
        if model is None:
            return None
        features = features if features is not None else self.skill_features(obs, plan)
        per_1000 = model["intercept"] + sum(w * features.get(name, 0.0) for name, w in model["weights"].items())
        return per_1000 * model["repeat_factor"] ** self._uses(obs, skill)

    def effective_cost(self, obs: CpuObservation, skill: str) -> float:
        """
        判断に使うコスト。使った後の HP が少ないほど重く見る（HP を減らすほど、放銃で倒されやすくなる）。
        コスト × value_margin × (1 + hp_risk.weight × 足りない割合)。足りない割合は safe_health に対する不足分。
        """
        cfg = self.config["skill"]
        cost = obs.me.skill_costs[skill]
        risk = cfg["hp_risk"]
        shortfall = max(0.0, risk["safe_health"] - (obs.me.health - cost)) / max(1, risk["safe_health"])
        # スキルごとの value_margin があればそちらを使う（役強化は高いので、よほど得なときだけにする）
        margin = cfg.get(skill, {}).get("value_margin", cfg["value_margin"])
        return cost * margin * (1 + risk["weight"] * shortfall)

    def _worth_using(self, obs: CpuObservation, plan: HandPlan, skill: str, extra_value: float = 0.0) -> bool:
        """見積もった効果（＋後の局まで続く分）が、判断に使うコスト以上か。"""
        value = self.estimate_skill_value(obs, plan, skill)
        return value is not None and value + extra_value >= self.effective_cost(obs, skill)

    def _decide_perspective(self, obs: CpuObservation, plan: HandPlan) -> Optional[dict[str, Any]]:
        cfg = self.config["skill"]["perspective"]
        skill = "perspective"
        if not cfg["enabled"] or self._uses(obs, skill) >= cfg["max_uses_per_round"]:
            return None
        if not self._can_afford(obs, skill) or obs.me.health < cfg["min_health"]:
            return None
        unrevealed = obs.opponent.wall_size - len(obs.opponent.exposed)
        if unrevealed <= 0:
            return None
        if self._value_model(skill) is not None:
            if not self._worth_using(obs, plan, skill):
                return None
        elif self.level == "normal":
            # 見積もり式がないとき: 公開済みが多いほど、追加で見える価値は下がる
            value = cfg["value"] * unrevealed / max(1, obs.opponent.wall_size)
            if value < obs.me.skill_costs[skill]:
                return None
        return {"skill_type": skill, "suit": self.choose_perspective_suit(obs, plan)}

    def choose_perspective_suit(self, obs: CpuObservation, plan: HandPlan, prefer_waits: bool = False) -> str:
        """
        透視で公開させる色を選ぶ。

        相手の山に残っていそうな枚数（4 枚 − 自分の山 − 公開済み − 相手の捨て牌）を色ごとに数え、
        いちばん多い色を選ぶ。相手の山の様子がいちばん分かり、放銃を避けるのに効く
        （自分の待ち牌の色を選ぶより効果が大きかった。training/skill_value.py で測定）。
        ふつうは、同じ枚数なら自分の待ち牌が多い色を選ぶ。
        prefer_waits=True なら、自分の待ち牌が残っていそうな色を先に選ぶ（比較の実験用）。
        """
        own = Counter(tile_kind(t) for t in obs.me.wall)
        seen = Counter(tile_kind(t) for t in obs.opponent.exposed.values())
        seen.update(tile_kind(t) for t in obs.opponent.discards)
        unseen = {kind: max(0, 4 - own[kind] - seen[kind]) for kinds in PERSPECTIVE_SUITS.values() for kind in kinds}

        def score(suit: str) -> tuple:
            kinds = PERSPECTIVE_SUITS[suit]
            total = sum(unseen[k] for k in kinds)
            if self.level == "easy":
                return (total,)
            waits = sum(unseen[k] for k in kinds if k in plan.wait_kinds)
            return (waits, total) if prefer_waits else (total, waits)

        return max(PERSPECTIVE_SUITS, key=score)

    def _decide_mulligan(self, obs: CpuObservation, plan: HandPlan) -> Optional[dict[str, Any]]:
        cfg = self.config["skill"]["mulligan"]
        skill = "mulligan"
        if not cfg["enabled"] or self._uses(obs, skill) >= cfg["max_uses_per_round"]:
            return None
        if not obs.me.can_mulligan or not self._can_afford(obs, skill):
            return None
        if self._value_model(skill) is not None:
            # 強い手に使うと損になる（測定）ので、手の様子から効果を見積もって決める
            if not self._worth_using(obs, plan, skill):
                return None
        elif plan.strength >= cfg["max_strength"]:
            return None
        return self.mulligan_skill_data(obs, plan)

    def mulligan_skill_data(self, obs: CpuObservation, plan: HandPlan) -> Optional[dict[str, Any]]:
        """マリガンの action の data（役に立っていない 3 枚を出す）。出せる牌が足りなければ None。"""
        offer = self._least_useful_rest(obs, plan)[:MULLIGAN_CANDIDATE_COUNT]
        if len(offer) < MULLIGAN_CANDIDATE_COUNT:
            return None
        return {"skill_type": "mulligan", "target_hand_indexes": offer}

    def _least_useful_rest(self, obs: CpuObservation, plan: HandPlan) -> list[int]:
        """
        手牌以外の牌を、役に立っていない順に並べる（マリガンで手放す候補）。
        自分の待ち牌（山から出ず、捨てればフリテン）→ 自分の山に何枚も重なっている牌 → それ以外。
        """
        counts = Counter(tile_kind(t) for t in obs.me.wall)
        rest = rest_indexes(obs, plan.hand_indexes)
        return sorted(rest, key=lambda i: (
            tile_kind(obs.me.wall[i]) not in plan.wait_kinds,
            -counts[tile_kind(obs.me.wall[i])],
            self.rng.random(),
        ))

    def _score_wall(self, obs: CpuObservation, wall: list[int]) -> float:
        """山がこうなったときの一番良い手牌の点数（満貫の手がなければ -inf）。"""
        plan = self.plan_hand(replace(obs, me=replace(obs.me, wall=tuple(wall))), time_limit=MULLIGAN_SEARCH_TIME_LIMIT,
                              candidate_limit=MULLIGAN_CANDIDATE_LIMIT)
        return plan.score if plan is not None else float("-inf")

    def choose_mulligan(self, obs: CpuObservation, pending: dict) -> list[int]:
        """
        マリガンで残す 3 枚（engine.resolve_mulligan に渡す options の番号）。
        6 枚から 3 枚の組合せ（20 通り）をすべて試し、一番良い手牌が作れるものを選ぶ。同点なら出した牌を残す。
        """
        indexes, options = pending["wall_indexes"], pending["options"]
        best, best_score = list(range(len(indexes))), float("-inf")
        for combo in itertools.combinations(range(len(options)), len(indexes)):
            wall = list(obs.me.wall)
            for wall_idx, option in zip(indexes, combo):
                wall[wall_idx] = options[option]
            score = self._score_wall(obs, wall)
            if score > best_score:
                best, best_score = list(combo), score
        return best

    def _decide_boost_hand(self, obs: CpuObservation, plan: HandPlan) -> Optional[dict[str, Any]]:
        cfg = self.config["skill"]["boost_hand"]
        skill = "boost_hand"
        if not cfg["enabled"] or self._uses(obs, skill) >= 1:
            return None
        if not self._can_afford(obs, skill):
            return None

        combo, score = self.best_boost_combo(obs, plan)
        if combo is None:
            return None
        if self._value_model(skill) is not None:
            # この局の効果に、後の局まで続く分（1 局あたりの効果 × 見込みの局数）を足して決める。
            # 強化済みの役が多いほど、新しく強化した役が和了りに絡む局は減るので、1 つごとに future_decay を掛ける
            future = cfg["future_value_per_1000"] * cfg["future_rounds"] * cfg["future_stake"] / 1000
            future *= cfg["future_decay"] ** len(obs.me.boost_hand_bonus)
            if not self._worth_using(obs, plan, skill, extra_value=future):
                return None
            return {"skill_type": skill, "yaku_names": list(combo)}
        if obs.me.health < cfg["min_health"]:
            return None
        gain_ratio = (score - plan.score) / max(plan.score, 1e-9)
        if self.level == "normal" and gain_ratio < cfg["min_gain_ratio"]:
            return None
        return {"skill_type": skill, "yaku_names": list(combo)}

    def _decide_assault(self, obs: CpuObservation, plan: HandPlan) -> Optional[dict[str, Any]]:
        cfg = self.config["skill"]["assault"]
        skill = "assault"
        if not cfg["enabled"] or obs.me.assault_used_this_round or self._uses(obs, skill) >= 1:
            return None
        if not self._can_afford(obs, skill):
            return None

        use = plan.best_han >= cfg["min_han"]
        if not use and self.level == "normal" and cfg["kill_ratio"] > 0:
            # 強襲は自分の獲得を相手への追加ダメージに変える。和了っても自分の累計獲得点数は増えないので、
            # 点数で勝つ道を捨てることになる（対局の多くは累計獲得点数で決まる。training/cpu_match.py で、
            # 倒しきれるか分からないときにも使うと勝率が下がった）。そこで、和了れば確実に倒しきれるときだけ使う:
            # 相手の損失は「相手の掛け金 × 倍率 + 強襲の追加ダメージ（自分の獲得）」。相手の掛け金はまだ分からない
            # ので最低額と見積もる。普通に和了れば累計獲得点数で勝てるなら使わない
            multiplier = plan.expected_multiplier
            gain = self.rule_bet(obs, plan) * multiplier
            opponent_loss_without = obs.me.bet_unit * multiplier
            if obs.me.cumulative_earned_points + gain >= POINT_VICTORY:
                return None
            use = gain > 0 and opponent_loss_without < obs.opponent.health <= opponent_loss_without + gain * cfg["kill_ratio"]
        if not use:
            return None
        return {"skill_type": skill}

    # ========== 掛け金 ==========

    def choose_bet(self, obs: CpuObservation, plan: Optional[HandPlan]) -> int:
        """この局の掛け金。"""
        return self.rule_bet(obs, plan)

    def rule_bet(self, obs: CpuObservation, plan: Optional[HandPlan]) -> int:
        """
        規則で決める掛け金。スキルの効果の見積もりなど、考え方の中で「賭ける予定の額」として使うのもこれ
        （学習したモデルは choose_bet だけを置き換える）。
        """
        cfg = self.config["bet"]
        unit = obs.me.bet_unit
        max_bet = min(obs.me.bet_max, obs.me.health // unit * unit)
        if max_bet <= unit:
            return unit

        # 和了ると「掛け金 × 倍率」を得て、放銃すると「掛け金 × 倍率」を失う（先払いはない）。
        if plan is None:
            return unit
        if "model" in cfg:
            return self._choose_bet_by_model(obs, plan, unit, max_bet)

        # 見積もり式がないとき: 和了れる見込みの倍率が設定値に届くときだけ、手の強さに応じて上げる
        if plan.expected_multiplier < cfg["min_multiplier_to_raise"]:
            return unit

        ratio = 0.0
        for step in sorted(cfg["steps"], key=lambda s: s["min_strength"]):
            if plan.strength >= step["min_strength"]:
                ratio = step["bet_ratio"]

        lead = _clamp((obs.me.health - obs.opponent.health) / 20000, -1.0, 1.0)
        ratio = _clamp(ratio + cfg["health_lead_factor"] * lead, 0.0, 1.0)
        return unit + int((max_bet - unit) * ratio) // unit * unit

    def bet_outlook(self, obs: CpuObservation, plan: HandPlan) -> dict[str, float]:
        """
        この局の見込み（掛け金 1 あたり）。bet.model の式で、和了る確率・放銃する確率・得る額・失う額を見積もる。
        edge = 和了る確率 × 得る額 − 放銃する確率 × 失う額（掛け金 1 あたりの、この局の HP の増減の見込み）
        """
        model = self.config["bet"]["model"]
        features = self.bet_features(obs, plan)

        def linear(key: str) -> float:
            entry = model[key]
            return entry["intercept"] + sum(w * features.get(name, 0.0) for name, w in entry["weights"].items())

        p_win = 1 / (1 + math.exp(-_clamp(linear("win"), -30, 30)))
        p_lose = 1 / (1 + math.exp(-_clamp(linear("lose"), -30, 30)))
        if p_win + p_lose > 1:  # 2 つの式は別々に作ったので、足して 1 を超えたら割り戻す
            p_win, p_lose = p_win / (p_win + p_lose), p_lose / (p_win + p_lose)
        gain, loss = max(0.0, linear("gain")), max(0.0, linear("loss"))
        return {"p_win": p_win, "p_lose": p_lose, "gain": gain, "loss": loss,
                "edge": p_win * gain - p_lose * loss}

    def _choose_bet_by_model(self, obs: CpuObservation, plan: HandPlan, unit: int, max_bet: int) -> int:
        """
        見積もり式で掛け金を決める。
        1. 見込み（edge）が 0 以下なら最低額。edge_for_max_bet に届けば上限。その間は比例
        2. HP の差で少し上げ下げする（リードしていれば攻める。health_lead_factor）
        3. 和了れば累計獲得点数 30000 に届くなら上限（points_push）
        4. 放銃で失う額の悪い場合を払っても ruin_reserve 以上の HP が残る額まで（倒されないため）
        """
        cfg = self.config["bet"]
        outlook = self.bet_outlook(obs, plan)
        if outlook["edge"] <= 0:
            return unit
        ratio = _clamp(outlook["edge"] / cfg["edge_for_max_bet"], 0.0, 1.0)
        lead = _clamp((obs.me.health - obs.opponent.health) / 20000, -1.0, 1.0)
        ratio = _clamp(ratio + cfg["health_lead_factor"] * lead, 0.0, 1.0)
        bet = unit + (max_bet - unit) * ratio
        if cfg["points_push"] and obs.me.cumulative_earned_points + max_bet * outlook["gain"] >= POINT_VICTORY:
            bet = max_bet
        safe = (obs.me.health - cfg["ruin_reserve"]) / cfg["worst_loss_multiplier"]
        bet = min(bet, safe)
        return max(unit, int(bet) // unit * unit)

    # ========== 打牌 ==========

    def choose_discard(self, obs: CpuObservation) -> int:
        cfg = self.config["discard"]
        candidates = rest_indexes(obs, obs.me.hand)
        if not candidates:
            return fallback_discard(obs)

        opponent_discard_kinds = {tile_kind(t) for t in obs.opponent.discards}
        # どちらかの河に出た種類（ボルテージ: 出ていない種類を捨てると上がり、出た種類を捨てると切れる）
        seen_kinds = opponent_discard_kinds | {tile_kind(obs.me.wall[i]) for i in obs.me.discarded_wall_indexes}
        my_waits = set(obs.me.waits)
        held_exposed = possibly_held_exposed_kinds(obs)
        # 公開牌が相手の手牌に入っている見込み。相手が捨て進めるほど、残った公開牌は手牌である可能性が高い
        undiscarded = max(OPPONENT_HAND_SIZE, obs.opponent.wall_size - len(obs.opponent.discards))
        in_hand_probability = OPPONENT_HAND_SIZE / undiscarded

        # 点数は牌種で決まるので、牌種ごとに 1 回だけ数える（同点の選び方だけは牌ごとに散らす）
        kind_scores: dict[int, float] = {}
        for kind in {tile_kind(obs.me.wall[i]) for i in candidates}:
            score = 0.0
            if kind in opponent_discard_kinds:
                score += cfg["weight_genbutsu"]  # 相手が捨てた種類はフリテンで必ず安全
            if kind in my_waits:
                score += cfg["weight_own_wait"]  # 自分の待ちを捨てると自分がフリテンになる
            score += cfg["weight_danger"] * self._danger(
                kind, opponent_discard_kinds, held_exposed, in_hand_probability, obs.opponent.complete_suits,
            )
            score += cfg["weight_voltage"] * (self._voltage_potential(obs, kind not in seen_kinds) - 1.0)
            kind_scores[kind] = score
        return max(candidates, key=lambda i: kind_scores[tile_kind(obs.me.wall[i])] + self.rng.random() * 1e-3)

    @staticmethod
    def _voltage_potential(obs: CpuObservation, is_new_kind: bool) -> float:
        """
        その種類を捨てた後に見込めるボルテージの倍率。続いていればいまの倍率、切れていれば
        次に初めての種類を捨てて再開したときの倍率（ボルテージは和了ったときの獲得額に掛かる）。
        """
        points, broken, resume = next_voltage(
            obs.me.voltage_points, obs.me.voltage_broken, obs.me.voltage_resume_points, is_new_kind,
        )
        return voltage_multiplier(resume + 1 if broken else points)

    def _danger(
        self,
        kind: int,
        opponent_discard_kinds: set[int],
        held_exposed: Counter,
        in_hand_probability: float,
        complete_suits: frozenset[str] = frozenset(),
    ) -> float:
        """その牌種が相手の待ちである見込み（0 なら安全）。"""
        cfg = self.config["discard"]
        if kind in opponent_discard_kinds:
            return 0.0
        if self._impossible_wait(kind, held_exposed, complete_suits):
            return 0.0

        if kind >= HONOR_START:
            danger = cfg["danger_honor"]
        else:
            number = kind % 9
            if number in (0, 8):
                danger = cfg["danger_terminal"]
            elif number in (1, 7):
                danger = cfg["danger_2_8"]
            else:
                danger = cfg["danger_middle"]

            # 筋: 相手が3つ離れた同じ色の牌を捨てていれば、両面待ちの可能性が下がる
            suit_start = kind - number
            suji = [suit_start + n for n in (number - 3, number + 3) if 0 <= n <= 8]
            if any(s in opponent_discard_kinds for s in suji):
                danger *= cfg["suji_factor"]

        if cfg["weight_exposed_neighbor"] > 0 and held_exposed:
            nearby = sum(count * self._closeness(kind, exposed) for exposed, count in held_exposed.items())
            danger *= 1 + cfg["weight_exposed_neighbor"] * in_hand_probability * nearby
        return danger

    @staticmethod
    def _impossible_wait(kind: int, held_exposed: Counter, complete_suits: frozenset[str]) -> bool:
        """
        相手の山のその色の牌を全部見ていて、kind が相手の待ちになりえないか。

        相手の手牌は相手の山の一部なので、その色の手牌は「見えた牌のうち、まだ捨てられていないもの」に限られる。
        待ちの形（両面・嵌張・辺張・単騎・双碰）は、どれも待ち牌の 2 つ隣までの牌を手牌に必要とするので、
        その範囲にまだ捨てられていない牌がなければ、kind で和了られることはない。
        """
        for suit, kinds in PERSPECTIVE_SUITS.items():
            if kind in kinds:
                break
        else:
            return False
        if suit not in complete_suits:
            return False
        if kind >= HONOR_START:
            return held_exposed[kind] == 0  # 字牌は順子にならないので、同じ牌がなければ待てない
        return not any(held_exposed[k] for k in kinds if abs(k - kind) <= 2)

    @staticmethod
    def _closeness(kind: int, other: int) -> float:
        """other を手牌に持つ相手が kind を待っている見込みの近さ。"""
        if kind == other:
            return CLOSENESS_BY_DISTANCE[0]
        if kind >= HONOR_START or other >= HONOR_START or kind // 9 != other // 9:
            return 0.0
        distance = abs(kind - other)
        return CLOSENESS_BY_DISTANCE[distance] if distance < len(CLOSENESS_BY_DISTANCE) else 0.0


def create_brain(config: dict, rng: Optional[random.Random] = None) -> RuleBrain:
    """
    設定の brain から考え方を作る。つよい（model）は、学習済みモデル（model_path）があればそれで打つ。
    モデルがない・読めないときは、規則で打つ（設定はつよいのもの）。
    """
    brain = config["brain"]
    if brain == "model":
        # numpy がない・モデルが読めないときも、マッチを止めずに規則で打つ
        try:
            from .model_brain import ModelBrain, model_available, resolve_model_path

            path = resolve_model_path(config.get("model_path", ""))
            if model_available(path):
                return ModelBrain({**config, "model_path": str(path)}, rng, mode="greedy")
            logger.warning("つよいのモデルがないので、規則で打ちます: %s", path)
        except Exception:
            logger.exception("つよいのモデルを使えないので、規則で打ちます")
    level = "easy" if brain == "easy" else "normal"
    return RuleBrain(level, config, rng)
