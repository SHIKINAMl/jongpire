"""
相手のくせの記憶と、特定の戦術への対策・採用

人間は CPU の決まった動きを読んで、それを逆手に取る（例: CPU が一九字牌を早く捨てるのを読んで、
一九字牌を待ちにする）。そこで CPU も、対局を通して相手のくせを公開情報から覚え、
- 対策: 相手がよく和了る種類の牌は、捨てるときに危険と見る（danger_multiplier）
- 採用: 相手が早く捨てる種類の牌は、自分の待ちにすると和了りやすいと見る（wait_value_multiplier）
をする。戦術ごとの重みは cpu_config.json の tactics に置く。

戦術を足すときは、TACTICS に名前を足し、分類（kind_category）と 2 つの倍率に効かせ方を書く。
いまある戦術:
- yaochu … 一九字牌待ち（2026-10 の人間とのテストプレイで使われた）
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

HONOR_START = 27  # 牌種 27 以降は字牌（東・西）
TILE_KIND_COUNT = 29

# 牌の分類。相手のくせは分類ごとに数える
CATEGORIES = ("honor", "terminal", "simple")
# 待ちの牌種が分類ごとに何種類あるか（相手にくせがないときの、和了り牌の分類の割合の目安）
_KINDS_PER_CATEGORY = {"honor": 2, "terminal": 6, "simple": 21}
BASE_SHARE = {c: n / TILE_KIND_COUNT for c, n in _KINDS_PER_CATEGORY.items()}

# 戦術の名前と、効かせる分類
TACTICS: dict[str, tuple[str, ...]] = {
    "yaochu": ("honor", "terminal"),
}


def kind_category(kind: int) -> str:
    if kind >= HONOR_START:
        return "honor"
    return "terminal" if kind % 9 in (0, 8) else "simple"


@dataclass
class OpponentProfile:
    """
    対局を通して覚える相手のくせ（公開情報だけ。局が変わっても消さない）。

    - 相手が和了った牌の分類（和了り牌は放銃した自分の最後の捨て牌なので、誰でも分かる）
    - 相手が捨てた牌の分類と、捨てた順番（0 = 1 打目、1 = 最後）
    """

    rounds: int = 0
    wins: int = 0
    tanki_wins: int = 0
    win_categories: Counter = field(default_factory=Counter)
    discard_counts: Counter = field(default_factory=Counter)
    discard_turn_sums: Counter = field(default_factory=Counter)

    def record_round(
        self,
        opponent_discards: Iterable[int],
        opponent_won: bool,
        winning_tile: Optional[int],
        is_tanki: bool = False,
    ) -> None:
        """局の終わりに 1 局分を覚える。"""
        discards = list(opponent_discards)
        self.rounds += 1
        last = max(1, len(discards) - 1)
        for turn, tile in enumerate(discards):
            category = kind_category(tile & 0b11111)
            self.discard_counts[category] += 1
            self.discard_turn_sums[category] += turn / last
        if opponent_won and winning_tile is not None:
            self.wins += 1
            self.win_categories[kind_category(winning_tile & 0b11111)] += 1
            if is_tanki:
                self.tanki_wins += 1

    def win_ratio(self, category: str, prior_strength: float) -> float:
        """
        相手の和了り牌のうち、その分類の割合が目安（BASE_SHARE）の何倍か。
        prior_strength 局分の「くせのない相手」を混ぜてならす（和了りが少ないうちは 1 に近い）。
        """
        base = BASE_SHARE[category]
        share = (self.win_categories[category] + prior_strength * base) / (self.wins + prior_strength)
        return share / base

    def early_discard(self, category: str, min_samples: int) -> Optional[float]:
        """
        その分類の牌を相手がどれだけ早く捨てるか（0.5 = ふつう、1 = 最初に、0 = 最後に）。
        捨てたのを見た枚数が min_samples に届かなければ None（まだ分からない）。
        """
        count = self.discard_counts[category]
        if count < max(1, min_samples):
            return None
        return 1.0 - self.discard_turn_sums[category] / count


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def danger_multiplier(kind: int, profile: Optional[OpponentProfile], tactics_cfg: dict) -> float:
    """
    対策: 捨てる牌の危険度に掛ける倍率。相手がこの分類でよく和了っていれば上げる。
    prior は「人間はこの戦術を使ってくる」と最初から見込む倍率（和了りを見る前から効く）。
    """
    category = kind_category(kind)
    multiplier = 1.0
    for name, categories in TACTICS.items():
        cfg = tactics_cfg.get(name)
        if not cfg or category not in categories:
            continue
        multiplier *= cfg["counter_prior"]
        if profile is not None and cfg["counter_weight"] > 0:
            ratio = profile.win_ratio(category, cfg["prior_strength"])
            multiplier *= _clamp(1.0 + cfg["counter_weight"] * (ratio - 1.0), 0.5, cfg["max_multiplier"])
    return multiplier


def wait_value_multiplier(kind: int, profile: Optional[OpponentProfile], tactics_cfg: dict) -> float:
    """
    採用: 自分の待ちの価値に掛ける倍率。相手がこの分類の牌を早く捨てる（くせがある）なら、
    その分類を待ちにすると早く和了れるので上げる。遅く捨てるなら下げる。
    """
    if profile is None:
        return 1.0
    category = kind_category(kind)
    multiplier = 1.0
    for name, categories in TACTICS.items():
        cfg = tactics_cfg.get(name)
        if not cfg or category not in categories or cfg["adopt_weight"] <= 0:
            continue
        early = profile.early_discard(category, cfg["min_samples"])
        if early is None:
            continue
        multiplier *= _clamp(1.0 + cfg["adopt_weight"] * (early - 0.5) * 2.0, 0.5, cfg["max_multiplier"])
    return multiplier


def softmax_pick(scores: list[float], temperature: float, rng) -> int:
    """
    点数に応じた確率で番号を選ぶ（点数が高いほど選ばれやすい）。temperature が 0 以下なら最高点。
    同じ局面でも毎回同じ選択にならないようにして、人間に読まれにくくする。
    """
    if not scores:
        raise ValueError("scores is empty")
    best = max(range(len(scores)), key=lambda i: scores[i])
    if temperature <= 0 or len(scores) == 1:
        return best
    top = scores[best]
    weights = [math.exp(max(-60.0, (s - top) / temperature)) for s in scores]
    r = rng.random() * sum(weights)
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i
    return best
