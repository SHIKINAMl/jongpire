"""
つよい（学習したモデル）の考え方

RuleBrain（規則で打つつよい）をそのまま土台にし、勝ち負けに効く 3 つの判断だけを学習したネットで決める:
- 手牌: 探した聴牌の候補から 1 つ選ぶ（final_hand）
- スキル: 次に使うスキル（使わない・透視・マリガン・役強化・強襲）（next_skill）
- 掛け金: 上限までの何割を賭けるか（BET_LEVELS）（choose_bet）
打牌・マリガンで残す牌・ロン・透視の色・役強化の役は RuleBrain のまま
（打牌は残り 21 枚から 17 枚を捨てるだけで、勝ち負けへの効きが小さかった。training/tune_cpu.py）。

ネットは小さな全結合（tanh）。重みは training/rl_train.py が .npz に書き出し、ここでは numpy で計算する。
mode:
- "greedy" … いちばん確率の高い行動（本番）。ただし variation.model_temperature が 0 より大きければ、
  ネットの確率を温度で鋭くしてから確率どおりに選ぶ（毎回同じ行動にならず、人間に読まれにくい）
- "sample" … 確率どおりに選ぶ（学習中のセルフプレイ）
- "teacher" … 規則どおりに打ち、その行動を手本として記録する（模倣学習のデータ集め）
recorder（リスト）を渡すと、判断ごとに入力・行動・確率を記録する（学習用）。
"""
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .cpu_config import BET_FEATURES
from .observation import CpuObservation
from .rule_brain import HandPlan, RuleBrain, fallback_hand_indexes

SKILL_ACTIONS = ("none", "perspective", "mulligan", "boost_hand", "assault")
COST_SKILLS = ("perspective", "mulligan", "boost_hand", "assault")
VALUE_SKILLS = ("perspective", "mulligan", "boost_hand")
BET_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
HEADS = ("hand", "skill", "bet")

# 判断材料の目安の大きさ（だいたい 0〜1 に収める）
_FEATURE_SCALE = {
    "score": 10.0, "expected_multiplier": 2.0, "max_multiplier": 2.0, "winning_waits": 4.0, "available": 10.0,
    "dead_waits": 4.0, "duplicated_rest": 8.0, "boost_gain_ratio": 1.0, "my_boosts": 4.0,
    "opp_unseen_best_suit": 30.0, "opp_boosts": 4.0, "carry_over": 2.0,
}
CANDIDATE_FEATURES = (
    "score", "expected_multiplier", "max_multiplier", "winning_waits", "available", "dead_waits",
    "duplicated_rest", "boost_gain_ratio",
)
CANDIDATE_SIZE = len(CANDIDATE_FEATURES) + 3  # + 一番良い候補との点数の比・規則が選ぶ候補か・1 種類待ち（単騎待ちボーナス）
STATE_SIZE = len(BET_FEATURES) + 5 + len(VALUE_SKILLS) + 2 * len(COST_SKILLS) + 4


def _scaled(features: dict[str, float], names) -> list[float]:
    return [features.get(name, 0.0) / _FEATURE_SCALE.get(name, 1.0) for name in names]


class Mlp:
    """全結合（隠れ層は tanh、出力は線形）。重みは {名前}.W0, {名前}.b0, ... の順。"""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers

    @classmethod
    def from_npz(cls, data, name: str) -> "Mlp":
        layers = []
        i = 0
        while f"{name}.W{i}" in data:
            layers.append((np.asarray(data[f"{name}.W{i}"], dtype=np.float64), np.asarray(data[f"{name}.b{i}"], dtype=np.float64)))
            i += 1
        if not layers:
            raise ValueError(f"モデルに {name} の重みがありません")
        return cls(layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x


def load_model(path: str) -> dict[str, Mlp]:
    with np.load(path) as data:
        return {head: Mlp.from_npz(data, head) for head in HEADS}


def _softmax_choice(logits: np.ndarray, mask: Optional[np.ndarray], rng: random.Random, sample: bool) -> tuple[int, float]:
    """(選んだ番号, その確率の対数)。mask が False の行動は選ばない。"""
    logits = np.asarray(logits, dtype=np.float64).copy()
    if mask is not None:
        logits[~mask] = -1e9
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if sample:
        r, acc, index = rng.random(), 0.0, int(np.argmax(probs))
        for i, p in enumerate(probs):
            acc += p
            if r < acc and (mask is None or mask[i]):
                index = i
                break
    else:
        index = int(np.argmax(probs))
    return index, math.log(max(probs[index], 1e-12))


class ModelBrain(RuleBrain):
    """手牌・スキル・掛け金を学習したネットで決める考え方（それ以外は RuleBrain）。"""

    def __init__(
        self,
        config: dict,
        rng: Optional[random.Random] = None,
        nets: Optional[dict[str, Mlp]] = None,
        mode: str = "greedy",
        recorder: Optional[list] = None,
    ):
        super().__init__("normal", config, rng)
        if mode not in ("greedy", "sample", "teacher"):
            raise ValueError(f"Unsupported mode: {mode}")
        if nets is None and mode != "teacher":
            nets = load_model(config["model_path"])
        self.nets = nets
        self.mode = mode
        self.recorder = recorder

    # ========== 入力 ==========

    def state_vector(self, obs: CpuObservation, plan: HandPlan) -> np.ndarray:
        """スキル・掛け金の判断の入力（STATE_SIZE 次元）。"""
        features = self.bet_features(obs, plan)
        vector = _scaled(features, BET_FEATURES)
        if "model" in self.config["bet"]:
            outlook = self.bet_outlook(obs, plan)
            vector += [outlook["p_win"], outlook["p_lose"], outlook["gain"] / 4, outlook["loss"] / 4, outlook["edge"]]
        else:
            vector += [0.0] * 5
        for skill in VALUE_SKILLS:
            value = self.skill_value_per_1000(obs, plan, skill, features)
            vector.append((value or 0.0) / 1000)
        vector += [obs.me.skill_costs[s] / 5000 for s in COST_SKILLS]
        vector += [self._uses(obs, s) for s in COST_SKILLS]
        vector += [
            obs.round_number / 30,
            obs.me.bet_max / 5000,
            float(obs.me.assault_used_this_round),
            float(obs.me.can_mulligan),
        ]
        return np.asarray(vector, dtype=np.float64)

    def candidate_matrix(self, obs: CpuObservation, plan: HandPlan) -> tuple[np.ndarray, list[tuple[int, ...]], int]:
        """手牌の候補ごとの入力（候補数 × CANDIDATE_SIZE）、候補の手牌、規則が選ぶ候補の番号。"""
        ctx = self._context(obs)
        rows, hands, plans = [], [], []
        for indexes in plan.candidates or (plan.hand_indexes,):
            candidate = self.evaluate_hand(obs, indexes, ctx=ctx)
            if candidate is not None:
                plans.append(candidate)
                hands.append(candidate.hand_indexes)
        if not plans:
            return np.zeros((0, CANDIDATE_SIZE)), [], 0
        best = max(range(len(plans)), key=lambda i: plans[i].score)
        best_score = max(plans[best].score, 1e-9)
        for i, candidate in enumerate(plans):
            features = self.skill_features(obs, candidate)
            single_wait = float(any(w.tanki for w in candidate.waits))
            rows.append(_scaled(features, CANDIDATE_FEATURES) + [candidate.score / best_score, float(i == best), single_wait])
        return np.asarray(rows, dtype=np.float64), hands, best

    def _record(self, head: str, x: np.ndarray, action: int, logp: float, mask: Optional[np.ndarray], state: np.ndarray) -> None:
        if self.recorder is not None:
            self.recorder.append({
                "head": head, "x": x, "action": action, "logp": logp,
                "mask": mask, "state": state,
            })

    def _choose(self, head: str, x: np.ndarray, mask: Optional[np.ndarray], teacher_action: int) -> tuple[int, float]:
        if self.mode == "teacher":
            return teacher_action, 0.0
        logits = self.nets[head](x)
        if head == "hand":
            logits = logits.reshape(-1)
        if self.mode == "greedy":
            temperature = self.config["variation"]["model_temperature"]
            if temperature > 0:
                index, _ = _softmax_choice(np.asarray(logits) / temperature, mask, self.rng, sample=True)
                return index, 0.0
        return _softmax_choice(logits, mask, self.rng, sample=self.mode == "sample")

    # ========== 手牌 ==========

    def final_hand(self, obs: CpuObservation, plan: Optional[HandPlan]) -> list[int]:
        if plan is None:
            return fallback_hand_indexes(obs)
        x, hands, best = self.candidate_matrix(obs, plan)
        if len(hands) <= 1:
            return list(plan.hand_indexes)
        index, logp = self._choose("hand", x, None, best)
        self._record("hand", x, index, logp, None, self.state_vector(obs, plan))
        return list(hands[index])

    # ========== スキル ==========

    def skill_options(self, obs: CpuObservation, plan: HandPlan) -> list[Optional[dict[str, Any]]]:
        """
        SKILL_ACTIONS の順に、使える行動の data（使えないものは None。「使わない」は {}）。
        乱数は使わない（マリガンで出す牌の選び方の乱数を戻す）。手本の記録で、規則と同じ手順を保つため。
        """
        state = self.rng.getstate()
        try:
            return self._skill_options(obs, plan)
        finally:
            self.rng.setstate(state)

    def _skill_options(self, obs: CpuObservation, plan: HandPlan) -> list[Optional[dict[str, Any]]]:
        cfg = self.config["skill"]
        options: list[Optional[dict[str, Any]]] = [{}]
        unrevealed = obs.opponent.wall_size - len(obs.opponent.exposed)
        ok = cfg["perspective"]["enabled"] and self._uses(obs, "perspective") < cfg["perspective"]["max_uses_per_round"]
        options.append({"skill_type": "perspective", "suit": self.choose_perspective_suit(obs, plan)}
                       if ok and unrevealed > 0 and self._can_afford(obs, "perspective") else None)
        ok = cfg["mulligan"]["enabled"] and self._uses(obs, "mulligan") < cfg["mulligan"]["max_uses_per_round"]
        options.append(self.mulligan_skill_data(obs, plan)
                       if ok and obs.me.can_mulligan and self._can_afford(obs, "mulligan") else None)
        combo = None
        if cfg["boost_hand"]["enabled"] and self._uses(obs, "boost_hand") < 1 and self._can_afford(obs, "boost_hand"):
            combo, _score = self.best_boost_combo(obs, plan)
        options.append({"skill_type": "boost_hand", "yaku_names": list(combo)} if combo else None)
        ok = (cfg["assault"]["enabled"] and not obs.me.assault_used_this_round
              and self._uses(obs, "assault") < 1 and self._can_afford(obs, "assault"))
        options.append({"skill_type": "assault"} if ok else None)
        return options

    def next_skill(self, obs: CpuObservation, plan: Optional[HandPlan]) -> Optional[dict[str, Any]]:
        if plan is None:
            return None
        options = self.skill_options(obs, plan)
        mask = np.array([option is not None for option in options])
        if mask.sum() <= 1:
            return None
        teacher = 0
        if self.mode == "teacher":
            data = super().next_skill(obs, plan)
            if data is not None:
                teacher = SKILL_ACTIONS.index(data["skill_type"])
                if not mask[teacher]:
                    return data  # 規則が別の条件で使う（記録しない）
                options[teacher] = data
        state = self.state_vector(obs, plan)
        index, logp = self._choose("skill", state, mask, teacher)
        self._record("skill", state, index, logp, mask, state)
        return options[index] or None

    # ========== 掛け金 ==========

    def choose_bet(self, obs: CpuObservation, plan: Optional[HandPlan]) -> int:
        unit = obs.me.bet_unit
        max_bet = min(obs.me.bet_max, obs.me.health // unit * unit)
        if plan is None or max_bet <= unit:
            return unit
        teacher = 0
        if self.mode == "teacher":
            bet = super().choose_bet(obs, plan)
            ratio = (bet - unit) / (max_bet - unit)
            teacher = min(range(len(BET_LEVELS)), key=lambda i: abs(BET_LEVELS[i] - ratio))
        state = self.state_vector(obs, plan)
        index, logp = self._choose("bet", state, None, teacher)
        self._record("bet", state, index, logp, None, state)
        if self.mode == "teacher":
            return bet
        bet = unit + (max_bet - unit) * BET_LEVELS[index]
        bet = max(unit, int(bet) // unit * unit)
        # 本番だけ、ブラフ・強い手で小さく賭けるのを混ぜる（選び方自体が揺れるので、細かい揺らぎは足さない）
        return self.vary_bet(obs, plan, bet, noise=False) if self.mode == "greedy" else bet


def resolve_model_path(path: str) -> Optional[Path]:
    """model_path を絶対パスにする。相対パスは mahjong_engine/ai/ からの位置（サーバーの作業フォルダによらない）。"""
    if not path:
        return None
    resolved = Path(path)
    return resolved if resolved.is_absolute() else Path(__file__).resolve().parent / resolved


def model_available(path: Optional[Path]) -> bool:
    return path is not None and path.is_file()
