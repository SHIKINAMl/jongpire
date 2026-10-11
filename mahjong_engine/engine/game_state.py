"""
麻雀ゲームの状態管理
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class RoundStatus(Enum):
    """ゲーム局のステータス"""
    DEALING = "dealing"  # 配牌フェーズ
    HAND_SELECTION = "hand_selection"  # 手牌選択フェーズ
    BETTING = "betting"  # 掛け金設定フェーズ
    DISCARD = "discard"  # 打牌フェーズ
    LIQUIDATION = "liquidation"  # 清算フェーズ
    ROUND_END_WAITING = "round_end_waiting"  # 次局進行待ち


class SkillType(Enum):
    """プレイヤーが使用できるスキルの種別"""
    MULLIGAN = "mulligan"  # 手牌交換スキル
    BOOST_HAND = "boost_hand"  # 指定役を含む和了の精算倍率を 1 段昇格（役ごとに 1 回）
    PERSPECTIVE = "perspective"  # 相手の手牌 3 枚を公開
    ASSAULT = "assault"  # 勝利時獲得点を0にし、同値を追加ダメージ化
    SPECIAL_VICTORY = "special_victory"  # 3 回目使用で勝利


# HP コスト・掛け金ルール統合テーブル
# インデックスは special_victory_count に対応（0回 / 1回 / 2回以上）
# コストはすべて 100 の倍数にする。
# 0 回の行のマリガン・透視・役強化は暫定値（効果の測定から、掛け金 2500 前後で元が取れる額。
# training/skill_value.py）。役強化は対局全体の測定（training/skill_game.py）で、3000 では勝率が +7% と
# 安すぎたため上げた。その後、切りのいい数にそろえた（マリガン・透視 500、役強化 5000）。
# 1 回・2 回以上の行はまだ見直していない
HP_COST_TABLE: list[dict] = [
    {
        "skill_costs": {
            SkillType.MULLIGAN: 500,
            SkillType.PERSPECTIVE: 500,
            SkillType.BOOST_HAND: 5000,
            SkillType.ASSAULT: 3000,
            SkillType.SPECIAL_VICTORY: 30000,
        },
        "bet_max": 5000,
        "bet_unit": 200,
    },
    {
        "skill_costs": {
            SkillType.MULLIGAN: 1000,
            SkillType.PERSPECTIVE: 3000,
            SkillType.BOOST_HAND: 9000,
            SkillType.ASSAULT: 3000,
            SkillType.SPECIAL_VICTORY: 30000,
        },
        "bet_max": 10000,
        "bet_unit": 1000,
    },
    {
        "skill_costs": {
            SkillType.MULLIGAN: 800,
            SkillType.PERSPECTIVE: 4500,
            SkillType.BOOST_HAND: 8000,
            SkillType.ASSAULT: 3000,
            SkillType.SPECIAL_VICTORY: 45000,
        },
        "bet_max": 50000,
        "bet_unit": 3000,
    },
]


def _cost_index(special_victory_count: int) -> int:
    """HP_COST_TABLE の参照インデックスを返す。"""
    return min(special_victory_count, 2)


def get_skill_cost(skill_type: SkillType, special_victory_count: int) -> int:
    """
    スキルの HP コストを取得

    Args:
        skill_type: スキル種別
        special_victory_count: SPECIAL_VICTORY の累計使用回数

    Returns:
        スキルの HP コスト
    """
    index = _cost_index(special_victory_count)
    return HP_COST_TABLE[index]["skill_costs"][skill_type]


# スキルコストの刻み（コストは常にこの倍数）
COST_STEP = 100


def apply_skill_cost_rate(cost: int, rate: float) -> int:
    """
    スキルコストに倍率を掛け、100 の倍数に丸める（例: 400 × 0.6 = 240 → 200、300 × 0.6 = 180 → 200）。
    コストのあるスキルは 0 にならない（最低 100）。

    Args:
        cost: 倍率を掛ける前のコスト
        rate: 倍率（1.0 なら変えない）

    Returns:
        倍率を掛けたあとのコスト
    """
    if cost <= 0:
        return 0
    return max(COST_STEP, int(cost * rate / COST_STEP + 0.5) * COST_STEP)


def get_bet_rule(special_victory_count: int) -> tuple[int, int]:
    """
    掛け金ルール（上限, 単位）を取得

    Args:
        special_victory_count: SPECIAL_VICTORY の累計使用回数

    Returns:
        (掛け金上限, 掛け金単位)
    """
    index = _cost_index(special_victory_count)
    row = HP_COST_TABLE[index]
    return row["bet_max"], row["bet_unit"]

@dataclass
class PlayerState:
    """プレイヤーの状態を管理"""
    player_id: str  # プレイヤー ID
    hand: list[int] = field(default_factory=list)  # 現在の手牌（インデックスまたはタイル ID）
    wall: list[int] = field(default_factory=list)  # この局で配られた牌山（全 34 枚）
    waits: list[int] = field(default_factory=list)  # テンパイ時の待ち牌リスト
    discards: list[int] = field(default_factory=list)  # この局で捨てた牌のリスト
    discarded_wall_indexes: set[int] = field(default_factory=set)  # 打牌済みの wall index
    health: int = 20000  # 現在の HP
    bet: int = 0  # この局の掛け金
    base_bet: int = 0  # 流局持ち越し時に加算する基準掛け金（最初に設定した額）
    cumulative_earned_points: int = 0  # 勝利時の獲得量だけを足す累計点数
    special_victory_count: int = 0  # SPECIAL_VICTORY 累計使用回数（対局を通じて持続）
    boost_hand_bonus: dict = field(default_factory=dict)  # 役強化ボーナス {役名: 1}。その役を含む和了の倍率を 1 段昇格（スキル/開始時付与を含み対局を通じて持続）
    exposed_hand_indexes: set = field(default_factory=set)  # PERSPECTIVE で公開された手牌インデックス（局ごとにリセット）
    assault_active_this_round: bool = False  # ASSAULT 発動済みなら、この局の和了時効果を有効化
    assault_used_this_round: bool = False  # ASSAULT の局内使用回数制限（1局1回）
    # MULLIGAN の選択待ち（{"wall_indexes": 交換に出した位置 3 つ, "options": [出した 3 枚, 候補 3 枚]}）。局ごとにリセット
    pending_mulligan: Optional[dict] = None
    voltage_points: int = 0 # 現在のボルテージ値
    voltage_broken: bool = False # ボルテージが継続中か
    voltage_resume_points: int = 0 # ボルテージを再開するとき、どこから再開するのか
    # CPU（とくしゅ）の特典。人間は既定値のまま（対局を通じて持続）
    skill_cost_rate: float = 1.0  # スキルコストの倍率
    boost_hand_targets: int = 1  # BOOST_HAND 1回で強化できる役の数
    opening_boost_count: int = 1  # 開始時恒常強化の数
    initial_perspective_count: int = 0  # 配牌時に無料で公開する相手の牌の枚数


@dataclass
class RoundState:
    """1 局の状態を管理"""
    round_number: int  # 現在の局番号
    current_player_index: int  # 現在のプレイヤーインデックス（0 または 1）
    first_player_index: int = 0  # 打牌フェーズの先手プレイヤーインデックス
    status: Optional[RoundStatus] = None  # 局内のステータス
    dora_id: Optional[int] = None  # ドラのタイル ID
    reserved_tiles: list = field(default_factory=list)  # 手牌交換用の予備牌（全 47 枚）


@dataclass
class GameState:
    """ゲーム全体の状態"""
    players: list[PlayerState] = field(default_factory=list)
    round_state: RoundState = field(default_factory=lambda: RoundState(
        round_number=1,
        current_player_index=0
    ))