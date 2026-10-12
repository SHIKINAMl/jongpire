"""
CPU 対戦の設定値

調整できる値はすべて cpu_config.json に置く。ファイルは CPU 戦のマッチを作るたびに読み直すので、
書き換えればサーバーを止めずに次のマッチから反映される（進行中のマッチは変わらない）。

- 環境変数 CPU_CONFIG_PATH で別のファイルを指定できる（調整用・テスト用）
- 難易度ごとのブロックは "extends" で別の難易度を土台にし、違う値だけを書ける
- 読み込みに失敗したら直前に正しく読めた設定を使い続ける。マッチは止めない
- 不正な項目（型違い・範囲外）は既定値に戻してログに出す
"""
import copy
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DIFFICULTIES = ("easy", "normal", "hard", "special")
BRAINS = ("easy", "normal", "model")
CONFIG_PATH_ENV = "CPU_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("cpu_config.json")

# スキルの効果の見積もりに使う判断材料（RuleBrain.skill_features と同じ名前）
SKILL_FEATURES = (
    "score", "expected_multiplier", "max_multiplier", "winning_waits", "available", "dead_waits",
    "duplicated_rest", "boost_gain_ratio", "my_boosts", "opp_unseen_best_suit", "opp_exposed_ratio", "opp_boosts",
)


# 掛け金の判断材料（RuleBrain.bet_features と同じ名前）
BET_FEATURES = SKILL_FEATURES + (
    "opp_used_mulligan", "opp_used_perspective", "opp_used_boost_hand", "opp_used_assault",
    "my_health", "opp_health", "my_points", "opp_points", "carry_over",
)


def _linear(intercept: float, **weights: float) -> dict:
    """「切片 + Σ 重み × 判断材料」の既定値（掛け金の見積もり式）。書いていない判断材料の重みは 0。"""
    return {"intercept": float(intercept), "weights": {name: float(weights.get(name, 0.0)) for name in BET_FEATURES}}


def _value_model(intercept: float, repeat_factor: float, **weights: float) -> dict:
    """「掛け金 1000 あたりの効果 ≒ 切片 + Σ 重み × 判断材料」の既定値。書いていない判断材料の重みは 0。"""
    return {
        "intercept": float(intercept),
        "weights": {name: float(weights.get(name, 0.0)) for name in SKILL_FEATURES},
        "repeat_factor": float(repeat_factor),
    }


# 人間と同じ（＝特典なし）ときの値。これと違う値を持つ CPU だけが game_started で perks を公開する
HUMAN_PERKS = {
    "initial_perspective_count": 0,
    "opening_boost_count": 1,
    "boost_hand_targets": 1,
    "skill_cost_rate": 1.0,
}

_THINK_TIME_DEFAULT = {
    "hand_selection": [1.5, 3.0],
    "skill": [0.6, 1.2],
    "bet": [0.8, 1.5],
    "discard": [0.6, 1.5],
    "agari": [0.3, 0.6],
    "next_round": [0.5, 1.0],
}

# cpu_config.json が最初から読めなかったときの値。項目の型・既定値の基準にもなる。
# 中身は同梱の cpu_config.json と揃えておく（調整は json 側で行う）
_BUILTIN_CONFIG: dict[str, Any] = {
    "easy": {
        "brain": "easy",
        "think_time": {
            "hand_selection": [2.0, 4.0],
            "skill": [0.8, 1.5],
            "bet": [1.0, 2.0],
            "discard": [0.8, 2.0],
            "agari": [0.3, 0.6],
            "next_round": [0.5, 1.0],
        },
        "perks": dict(HUMAN_PERKS),
        "hand_selection": {
            "candidate_limit": 30,
            "search_time_limit": 1.0,
            "weight_han_multiplier": 1.0,
            "weight_wait_count": 1.0,
            "penalty_wait_in_own_wall": 0.5,
            "bonus_wait_seen_in_opponent_wall": 0.0,
            "strong_hand_score": 8.0,
            # 単騎待ち（和了ると相手の損失が 2 倍）の待ちの点数を、1 + tanki_bonus 倍に見る
            "tanki_bonus": 0.0,
        },
        "skill": {
            "min_health_after_use": 8000,
            "mulligan": {"enabled": True, "max_strength": 0.4, "max_uses_per_round": 1},
            "perspective": {"enabled": True, "min_health": 20000, "max_uses_per_round": 1, "value": 0},
            "assault": {"enabled": True, "min_han": 8, "kill_ratio": 0.0},
            "boost_hand": {"enabled": True, "min_health": 25000, "min_gain_ratio": 0.0},
        },
        "bet": {
            "min_multiplier_to_raise": 1.0,
            "steps": [
                {"min_strength": 0.0, "bet_ratio": 0.0},
                {"min_strength": 0.6, "bet_ratio": 0.2},
                {"min_strength": 0.85, "bet_ratio": 0.4},
            ],
            "health_lead_factor": 0.0,
        },
        "discard": {
            "weight_genbutsu": 100.0,
            "weight_own_wait": -50.0,
            "weight_danger": -10.0,
            "danger_honor": 0.3,
            "danger_terminal": 0.6,
            "danger_2_8": 0.8,
            "danger_middle": 1.0,
            "suji_factor": 0.5,
            "weight_exposed_neighbor": 0.0,
            "weight_voltage": 0.0,
        },
    },
    # ふつうは、つよいの設定を土台に、弱めた値だけを書く（人とのテストプレイで強すぎたときに下方修正する場所）。
    # つよいの設定は、強化学習のモデルができるまで規則で打つときの設定であり、学習の相手・手本にもする
    "normal": {
        "extends": "hard",
        "brain": "normal",
        "think_time": copy.deepcopy(_THINK_TIME_DEFAULT),
    },
    "hard": {
        "brain": "model",
        # 学習したモデル（mahjong_engine/ai/ からの位置）。ファイルがなければ規則で打つ
        "model_path": "models/hard.npz",
        "think_time": {
            "hand_selection": [1.0, 2.0],
            "skill": [0.4, 0.8],
            "bet": [0.5, 1.0],
            "discard": [0.5, 1.0],
            "agari": [0.3, 0.6],
            "next_round": [0.5, 1.0],
        },
        "perks": dict(HUMAN_PERKS),
        "hand_selection": {
            # 探す聴牌の候補の数。いちばん効く設定値で、20 → 勝率 39%、80 → 60%、160 → 80 に 58%、
            # 320 → 80 に 62%（training/tune_cpu.py・cpu_match.py）。探索は search_time_limit で打ち切るので、
            # 遅いサーバーでは見つかる数が減る。サーバーの余裕を見て決める
            "candidate_limit": 160,
            "search_time_limit": 1.0,
            "weight_han_multiplier": 1.0,
            "weight_wait_count": 1.0,
            "penalty_wait_in_own_wall": 0.5,
            "bonus_wait_seen_in_opponent_wall": 0.3,
            "strong_hand_score": 8.0,
            # 単騎待ち（和了ると相手の損失が 2 倍）の待ちの点数を、1 + tanki_bonus 倍に見る
            "tanki_bonus": 0.0,
        },
        "skill": {
            "min_health_after_use": 5000,
            # 見積もった効果が「コスト × value_margin」以上なら使う（1 未満なら少し損でも使う）
            "value_margin": 0.8,
            # 効果の見積もりに使う掛け金の下限。0 なら実際に賭ける予定の額で見積もる（大きく賭ける局でだけ使う）。
            # 1000 にすると最低額しか賭けない局でも使い、スキルなしの CPU に負けた（training/cpu_match.py）
            "stake_floor": 0,
            # 使った後の HP が safe_health を下回るほど、コストを重く見る（不足の割合 × weight だけ増やす）
            "hp_risk": {"safe_health": 15000, "weight": 1.0},
            "mulligan": {"enabled": True, "max_strength": 0.6, "max_uses_per_round": 2},
            "perspective": {"enabled": True, "min_health": 0, "max_uses_per_round": 2, "value": 1000},
            "assault": {"enabled": True, "min_han": 99, "kill_ratio": 1.0},
            "boost_hand": {
                "enabled": True, "min_health": 15000, "min_gain_ratio": 0.3,
                # 役強化は対局の最後まで続く。後の局の 1 局あたりの効果（掛け金 1000 あたり）× 見込みの局数を足す。
                # 対局全体の測定（training/skill_game.py）では、1 局目の役強化で勝率 +7%（HP 6000〜7000 相当）。
                # 累計獲得点数の決着が早まる分が大きく、1 局ごとの HP の差の測定より大きい
                # future_stake は後の局で賭ける見込みの額
                "future_value_per_1000": 600, "future_rounds": 10, "future_decay": 0.5, "future_stake": 1000,
                # 役強化だけの value_margin（skill.value_margin より優先）。コスト 6000 では使うほど勝率が下がった
                # （training/cpu_match.py。0.8 で 1 対局 1.2 回・勝率 51%、使わないと 55%）。2.5 で 0.2 回・54%
                "value_margin": 2.5,
            },
            # 掛け金 1000 あたりの効果の見積もり式（training/fit_skill_model.py で測定から作る）。
            # 既定値は 8000 局の測定（training/data/skill_raw.jsonl）から前進選択で作ったもの:
            #   透視   … 相手の山に残っていそうな枚数がいちばん多い色の枚数が多いほど効く
            #   マリガン … 最大倍率が低く、残っていそうな待ち牌が少ない（弱い）手ほど効く。強い手では損
            #   役強化 … 期待倍率が高く、強化で点数が大きく上がる手ほど効く
            "value_model": {
                "perspective": _value_model(-1520.72, 0.651, opp_unseen_best_suit=60.284),
                "mulligan": _value_model(1365.34, 0.341, max_multiplier=-533.005, available=-69.862),
                "boost_hand": _value_model(-1179.97, 0.0, expected_multiplier=828.505, boost_gain_ratio=1499.732),
            },
        },
        "bet": {
            "min_multiplier_to_raise": 1.0,
            "steps": [
                {"min_strength": 0.0, "bet_ratio": 0.0},
                {"min_strength": 0.5, "bet_ratio": 0.15},
                {"min_strength": 0.8, "bet_ratio": 0.4},
                {"min_strength": 0.95, "bet_ratio": 0.7},
            ],
            "health_lead_factor": 0.1,
            # ここから下は、見積もり式（model）があるときの決め方（RuleBrain._choose_bet_by_model）。
            # 1 あたりの見込み = 和了る確率 × 得る額 − 放銃する確率 × 失う額。
            # 見込みが edge_for_max_bet に届けば上限まで賭け、0 以下なら最低額にする
            "edge_for_max_bet": 0.5,
            # 放銃で失う額の悪い場合（掛け金 × worst_loss_multiplier）を払っても、ruin_reserve 以上の HP が残る額まで
            "worst_loss_multiplier": 3.0,
            "ruin_reserve": 2000,
            # 和了れば累計獲得点数 30000 に届くなら、見込みがプラスである限り上限まで賭ける
            "points_push": True,
            # 和了る確率（win）・放銃する確率（lose）はロジスティック、得る額（gain）・失う額（loss）は掛け金 1 あたり。
            # training/fit_bet_model.py で記録から作る。既定値は判断材料を使わない平均的な値
            "model": {
                "win": _linear(-0.4),
                "lose": _linear(-0.4),
                "gain": _linear(1.2),
                "loss": _linear(1.3),
            },
        },
        "discard": {
            "weight_genbutsu": 100.0,
            "weight_own_wait": -50.0,
            "weight_danger": -10.0,
            "danger_honor": 0.3,
            "danger_terminal": 0.6,
            "danger_2_8": 0.8,
            "danger_middle": 1.0,
            "suji_factor": 0.5,
            "weight_exposed_neighbor": 1.5,
            # ボルテージ: 捨てた後に見込める倍率 − 1 にこの重みを掛けて足す（出ていない種類は上げ、出た種類は切る）
            "weight_voltage": 0.0,
        },
    },
    "special": {
        "extends": "normal",
        "perks": {
            "initial_perspective_count": 7,
            "opening_boost_count": 3,
            "boost_hand_targets": 2,
            "skill_cost_rate": 0.6,
        },
    },
}


def _non_negative(value: Any) -> bool:
    return value >= 0


def _unit_interval(value: Any) -> bool:
    return 0 <= value <= 1


# 型だけでは足りない項目の範囲。キーは「難易度を除いたパス」
_RANGE_RULES: dict[str, Callable[[Any], bool]] = {
    "brain": lambda v: v in BRAINS,
    "perks.initial_perspective_count": _non_negative,
    "perks.opening_boost_count": _non_negative,
    "perks.boost_hand_targets": lambda v: v >= 1,
    "perks.skill_cost_rate": lambda v: v > 0,
    "hand_selection.candidate_limit": lambda v: v >= 1,
    "hand_selection.search_time_limit": lambda v: v > 0,
    "hand_selection.strong_hand_score": lambda v: v > 0,
    "skill.mulligan.max_uses_per_round": _non_negative,
    "skill.value_margin": lambda v: v > 0,
    "skill.boost_hand.value_margin": lambda v: v > 0,
    "skill.stake_floor": _non_negative,
    "skill.boost_hand.future_rounds": _non_negative,
    "skill.boost_hand.future_decay": _unit_interval,
    "skill.boost_hand.future_stake": _non_negative,
    "skill.hp_risk.safe_health": _non_negative,
    "skill.hp_risk.weight": _non_negative,
    "skill.perspective.max_uses_per_round": _non_negative,
    "bet.min_multiplier_to_raise": _non_negative,
    "bet.edge_for_max_bet": lambda v: v > 0,
    "bet.worst_loss_multiplier": lambda v: v > 0,
    "bet.ruin_reserve": _non_negative,
    "bet.steps.min_strength": _unit_interval,
    "bet.steps.bet_ratio": _unit_interval,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """base に override を重ねた新しい辞書を返す（辞書は再帰的に、それ以外は置き換え）。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(value: Any, default: Any, path: str, rule_path: str, problems: list[str]) -> Any:
    """default と同じ形かを確かめ、不正な項目は既定値に戻した値を返す。"""
    if isinstance(default, dict):
        if not isinstance(value, dict):
            problems.append(f"{path}: 辞書が必要です")
            return copy.deepcopy(default)
        cleaned = {}
        for key, default_child in default.items():
            child_path = f"{path}.{key}"
            child_rule = f"{rule_path}.{key}" if rule_path else key
            if key not in value:
                cleaned[key] = copy.deepcopy(default_child)
                continue
            cleaned[key] = _validate(value[key], default_child, child_path, child_rule, problems)
        for key in value:
            if key not in default:
                problems.append(f"{path}.{key}: 知らない項目なので無視します")
        return cleaned

    if isinstance(default, list):
        if not isinstance(value, list) or not value:
            problems.append(f"{path}: 空でない配列が必要です")
            return copy.deepcopy(default)
        if default and isinstance(default[0], dict):
            return [_validate(item, default[0], f"{path}[{i}]", rule_path, problems) for i, item in enumerate(value)]
        # 数値の組（think_time の [最短, 最長] など）
        if len(value) != len(default) or not all(_is_number(v) and v >= 0 for v in value):
            problems.append(f"{path}: 0 以上の数値 {len(default)} 個が必要です")
            return copy.deepcopy(default)
        if len(value) == 2 and value[0] > value[1]:
            problems.append(f"{path}: [最小, 最大] の順で指定してください")
            return copy.deepcopy(default)
        return [float(v) for v in value]

    if isinstance(default, bool):
        valid = isinstance(value, bool)
    elif isinstance(default, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        valid = _is_number(value)
        if valid:
            value = float(value)
    elif isinstance(default, str):
        valid = isinstance(value, str)
    else:
        valid = True

    if valid:
        rule = _RANGE_RULES.get(rule_path)
        if rule is not None and not rule(value):
            problems.append(f"{path}: 範囲外の値です ({value!r})")
            return copy.deepcopy(default)
        return value

    problems.append(f"{path}: {type(default).__name__} が必要です ({value!r})")
    return copy.deepcopy(default)


def _resolve(raw: dict, schema: Optional[dict[str, dict]]) -> dict[str, dict]:
    """
    難易度ごとのブロックを extends をたどって組み立て、schema で検証する。

    Args:
        raw: 設定ファイルの中身
        schema: 項目の型・既定値の基準（組み立て済み）。None なら検証しない（同梱の既定値を組み立てるとき）
    """
    resolved: dict[str, dict] = {}
    problems: list[str] = []

    def resolve_block(name: str, stack: tuple[str, ...]) -> dict:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f"extends が循環しています: {' -> '.join(stack + (name,))}")

        block = raw.get(name)
        if block is None:
            if schema is None:
                raise ValueError(f"{name} がありません")
            resolved[name] = copy.deepcopy(schema[name])
            return resolved[name]
        if not isinstance(block, dict):
            raise ValueError(f"{name}: 辞書が必要です")

        parent = block.get("extends")
        if parent is not None:
            if parent not in DIFFICULTIES:
                raise ValueError(f"{name}.extends: 知らない難易度です ({parent!r})")
            base = resolve_block(parent, stack + (name,))
        else:
            base = schema[name] if schema is not None else {}

        merged = _deep_merge(base, {k: v for k, v in block.items() if k != "extends"})
        if schema is not None:
            merged = _validate(merged, schema[name], name, "", problems)
        resolved[name] = merged
        return merged

    for difficulty in DIFFICULTIES:
        resolve_block(difficulty, ())

    for problem in problems:
        logger.warning("CPU 設定: %s", problem)
    return resolved


def _with_model_path(config: dict[str, dict]) -> dict[str, dict]:
    """model_path はつよいだけが持つが、どの難易度でも参照できるように空文字で埋める。"""
    for block in config.values():
        block.setdefault("model_path", "")
    return config


_BUILTIN_RESOLVED = _with_model_path(_resolve(_BUILTIN_CONFIG, None))
_last_good: Optional[dict[str, dict]] = None
_load_lock = threading.Lock()


def config_path() -> Path:
    """いま読む設定ファイルの場所。"""
    override = os.getenv(CONFIG_PATH_ENV)
    return Path(override) if override else DEFAULT_CONFIG_PATH


def _load_resolved() -> dict[str, dict]:
    """
    設定ファイルを読み直して、難易度ごとの設定を返す（呼び出し側で書き換えないこと）。

    読み込みに失敗したときは直前に正しく読めた設定を返す（まだ一度も読めていなければ同梱の既定値）。
    """
    global _last_good
    path = config_path()
    with _load_lock:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("最上位は辞書が必要です")
            _last_good = _with_model_path(_resolve(raw, _BUILTIN_RESOLVED))
        except (OSError, ValueError) as exc:
            # json.JSONDecodeError は ValueError の一種
            logger.warning("CPU 設定を読めなかったため、直前の設定を使います: path=%s error=%s", path, exc)
        return _last_good if _last_good is not None else _BUILTIN_RESOLVED


def load_cpu_config() -> dict[str, dict]:
    """設定ファイルを読み直して、難易度ごとの設定を返す（書き換えてよいコピー）。"""
    return copy.deepcopy(_load_resolved())


@dataclass(frozen=True)
class CpuProfile:
    """1つの CPU 席の設定。マッチ作成時に確定し、対局中は変わらない。"""

    difficulty: str
    config: dict

    @property
    def perks(self) -> dict:
        return dict(self.config["perks"])

    def to_public_dict(self) -> dict:
        """game_started で公開する内容。特典が人間と同じなら perks は付けない。"""
        public: dict[str, Any] = {"difficulty": self.difficulty}
        if self.perks != HUMAN_PERKS:
            public["perks"] = self.perks
        return public


def create_cpu_profile(difficulty: str) -> CpuProfile:
    """難易度から CPU 席の設定を作る。設定ファイルはここで読み直す。"""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return CpuProfile(difficulty=difficulty, config=copy.deepcopy(_load_resolved()[difficulty]))
