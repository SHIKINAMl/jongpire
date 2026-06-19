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
    BOOST_HAND = "boost_hand"  # 指定役の翻数 +1
    PERSPECTIVE = "perspective"  # 相手の手牌 3 枚を公開
    SPECIAL_VICTORY = "special_victory"  # 3 回目使用で勝利


# HP コスト・掛け金ルール統合テーブル
# インデックスは special_victory_count に対応（0回 / 1回 / 2回以上）
HP_COST_TABLE: list[dict] = [
    {
        "skill_costs": {
            SkillType.MULLIGAN: 1200,
            SkillType.PERSPECTIVE: 1500,
            SkillType.BOOST_HAND: 10000,
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
    special_victory_count: int = 0  # SPECIAL_VICTORY 累計使用回数（対局を通じて持続）
    boost_hand_bonus: dict = field(default_factory=dict)  # 役強化ボーナス {役名: 追加翻数}（スキル/開始時付与を含み対局を通じて持続）
    exposed_hand_indexes: set = field(default_factory=set)  # PERSPECTIVE で公開された手牌インデックス（局ごとにリセット）


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