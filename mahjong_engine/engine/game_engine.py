"""
麻雀ゲームエンジン
"""
import logging
import re
from collections import Counter
from typing import Callable, Optional
import random

from .game_state import GameState, RoundStatus, SkillType, PlayerState, get_skill_cost, get_bet_rule
from .tile_wall import TileWall
from .hand_analyzer import HandAnalyzer
from .yaku import Yaku

logger = logging.getLogger(__name__)


class GameEngine:
    """麻雀ゲームエンジン"""

    def __init__(self, max_rounds: int = 25):
        """
        ゲームエンジンを初期化

        Args:
            max_rounds: 最大ラウンド数（デフォルト25）
        """
        self.state = GameState()
        self.tile_wall = TileWall()
        self.num_players = 2
        self.max_rounds = max_rounds
        self._carry_over_bets = False
        self._last_liquidation_result: Optional[dict] = None
        self._next_round_ready_players: set[str] = set()
        self._pending_agari: Optional[dict] = None

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

    def _start_round(self):
        """局を開始"""
        logger.info("局開始: round=%d", self.state.round_state.round_number)
        self._invoke_callback(self.on_round_start)

        # 配牌を実行
        self._set_phase(RoundStatus.DEALING)
        self._deal_tiles()

    def _deal_tiles(self):
        """各プレイヤーに牌を配る"""
        # 局ごとに牌山を再生成しないと、2局目以降で牌不足になる。
        self.tile_wall.reset()
        self.tile_wall.shuffle()

        hands = [self.tile_wall.deal() for _ in range(self.num_players)]

        # MULLIGAN 交換用の reserved_tiles を計算
        # 全 116 牌 - (配られた 68 牌 + ドラの 1 牌) = 47 牌
        dealt_tiles = [tile for wall, _ in hands for tile in wall] + [self.tile_wall.dora_id]
        used_set = set(dealt_tiles)
        reserved = [t for t in range(116) if t not in used_set]
        self.state.round_state.reserved_tiles = reserved

        def _to_wall_indexes(wall_tiles: list[int], hand_tiles: list[int]) -> list[int]:
            used = [False] * len(wall_tiles)
            indexes: list[int] = []
            for tile in hand_tiles:
                tile_base = tile & 0b11111
                found_idx = None
                for idx, wall_tile in enumerate(wall_tiles):
                    if used[idx]:
                        continue
                    if (wall_tile & 0b11111) == tile_base:
                        found_idx = idx
                        break
                if found_idx is None:
                    raise ValueError("deal が返した聴牌例を wall index へ変換できませんでした")
                used[found_idx] = True
                indexes.append(found_idx)
            return indexes

        for player, (wall, hand_tiles) in zip(self.state.players, hands):
            player.wall = wall  # 配られた牌
            player.hand = _to_wall_indexes(wall, hand_tiles)  # 手牌例を wall index で保持

        self.state.round_state.dora_id = self.tile_wall.dora_id
        self._pending_agari = None

        self._invoke_callback(self.on_dealt)

        self._set_phase(RoundStatus.HAND_SELECTION)

    def selected_hand(self) -> None:
        """手牌の選択フェーズを完了"""
        self._invoke_callback(self.on_selected)

        if self._carry_over_bets:
            # 流局後は同じ掛け金を維持し、ベットフェーズをスキップ
            if not self._can_all_players_cover_carry_over_bet():
                logger.warning("持ち越し掛け金を支払えないプレイヤーがいるためゲーム終了")
                self._on_game_end()
                return
            self.bet()
            return

        # 通常局は最低掛け金を支払えるかチェック
        if not self._can_all_players_cover_min_bet():
            logger.warning("最低掛け金を支払えないプレイヤーがいるためゲーム終了")
            self._on_game_end()
            return

        self._set_phase(RoundStatus.BETTING)

    def bet(self) -> None:
        """掛け金設定フェーズを完了し、打牌フェーズを開始"""
        self._invoke_callback(self.on_bet)
        self._set_phase(RoundStatus.DISCARD)
        self.state.round_state.current_player_index = random.randrange(0, self.num_players)
        self.state.round_state.first_player_index = self.state.round_state.current_player_index
        self._invoke_callback(self.on_discard_started)

    def get_bet_rule(self, player: PlayerState) -> tuple[int, int]:
        """プレイヤーの現在状態に対する掛け金ルール(上限, 単位)を返す。"""
        return get_bet_rule(player.special_victory_count)

    def get_minimum_bet(self, player: PlayerState) -> int:
        """プレイヤーの現在状態に対する最低掛け金を返す。"""
        _, unit = self.get_bet_rule(player)
        return unit

    def place_bet(self, player: PlayerState, bet_amount: int) -> bool:
        """掛け金を検証して設定する。"""
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
        return True

    def _can_all_players_cover_min_bet(self) -> bool:
        """全プレイヤーが最低掛け金以上のHPを保持しているか。"""
        for player in self.state.players:
            required = self.get_minimum_bet(player)
            if player.health < required:
                player.health = 0
                return False
        return True

    def _can_all_players_cover_carry_over_bet(self) -> bool:
        """流局持ち越し時、全プレイヤーが前局掛け金以上のHPを保持しているか。"""
        for player in self.state.players:
            if player.health < player.bet:
                player.health = 0
                return False
        return True

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
        if normalized_name is None:
            return False

        player.boost_hand_bonus[normalized_name] = player.boost_hand_bonus.get(normalized_name, 0) + bonus_han
        return True

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
                - target_hand_index: MULLIGAN の交換対象インデックス

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        # HP コスト計算
        try:
            cost = get_skill_cost(skill_type, user.special_victory_count)
        except KeyError:
            logger.error("スキルコスト取得失敗: skill_type=%s", skill_type)
            return None

        # HP 不足チェック
        if user.health < cost:
            logger.debug("スキル HP 不足: player=%s  skill=%s  cost=%d  health=%d",
                         user.player_id, skill_type.value, cost, user.health)
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

        # コールバック呼び出し
        self._invoke_callback(self.on_skill_casted, user.player_id, skill_type, cost, exposed_tiles)

        return cost

    def _validate_skill_cast(self, user: PlayerState, skill_type: SkillType, target: PlayerState | None, options: dict) -> bool:
        """
        スキル発動の前提条件をチェック

        Returns:
            エラーがあれば True、正常ならば False
        """
        if skill_type == SkillType.BOOST_HAND:
            if "yaku_name" not in options or not options["yaku_name"]:
                return True
            normalized_yaku_name = self.normalize_boost_target_yaku_name(options["yaku_name"])
            if normalized_yaku_name is None:
                return True
            options["yaku_name"] = normalized_yaku_name

        elif skill_type == SkillType.MULLIGAN:
            target_index = options.get("target_hand_index")
            if not isinstance(target_index, int) or target_index < 0 or target_index >= len(user.wall):
                return True
            if target_index in user.discarded_wall_indexes:
                return True
            # 予備牌チェック
            if not self.state.round_state.reserved_tiles:
                return True

        elif skill_type == SkillType.PERSPECTIVE:
            if target is None:
                return True
            unrevealed_indexes = [idx for idx in target.hand if idx not in target.exposed_hand_indexes]
            if not unrevealed_indexes:
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
            yaku_name = options["yaku_name"]
            user.boost_hand_bonus[yaku_name] = user.boost_hand_bonus.get(yaku_name, 0) + 1

        elif skill_type == SkillType.PERSPECTIVE:
            participants = [user]
            if target is not None and target.player_id != user.player_id:
                participants.append(target)

            for participant in participants:
                if not participant.hand:
                    continue

                unrevealed_indexes = [
                    idx for idx in participant.hand
                    if idx not in participant.exposed_hand_indexes
                ]
                if not unrevealed_indexes:
                    exposed_tiles[participant.player_id] = set(participant.exposed_hand_indexes)
                    continue

                exposed = random.sample(unrevealed_indexes, min(3, len(unrevealed_indexes)))
                participant.exposed_hand_indexes.update(exposed)
                exposed_tiles[participant.player_id] = set(participant.exposed_hand_indexes)

        elif skill_type == SkillType.MULLIGAN:
            target_index = options["target_hand_index"]
            # target_hand_index は wall 内インデックス(0-33)として扱う
            wall_idx = target_index
            old_tile_id = user.wall[wall_idx]
            new_tile_id = random.choice(self.state.round_state.reserved_tiles)
            self.state.round_state.reserved_tiles.remove(new_tile_id)
            self.state.round_state.reserved_tiles.append(old_tile_id)
            # wall 内の指定位置の牌を差し替える
            user.wall[wall_idx] = new_tile_id

        return exposed_tiles

    def use_skill(self, player: PlayerState, skill_type: SkillType, yaku_name: str | None = None) -> Optional[int]:
        """
        プレイヤーがスキルを使用する（自分を対象とするスキル）

        Args:
            player: 使用プレイヤー
            skill_type: スキル種別（BOOST_HAND, SPECIAL_VICTORY など）
            yaku_name: BOOST_HAND 使用時に必須。翻を上げる対象役名

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        return self.cast_skill(player, skill_type, yaku_name=yaku_name)

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

    def use_mulligan(self, player: PlayerState, target_hand_index: int) -> Optional[int]:
        """
        プレイヤーがMULLIGAN スキルを使用する（手牌を交換）

        Args:
            player: プレイヤー
            target_hand_index: 交換対象の wall 内インデックス（0-33）

        Returns:
            実際に支払ったHP コスト。失敗時は None
        """
        return self.cast_skill(player, SkillType.MULLIGAN, target_hand_index=target_hand_index)

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

        self._invoke_callback(self.on_discarded, player_id, discarded_tile)

        # 打牌後、相手の待ち牌なら和了入力待ちへ移行
        discarded_tile_base = discarded_tile & 0b11111
        winning_waits = []
        if winning_player is not None:
            winning_waits = [tile & 0b11111 for tile in winning_player.waits]

        if winning_player is not None and discarded_tile_base in winning_waits:
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

        if all(len(player.discards) >= 16 for player in self.state.players):
            logger.info(
                "流局: all players reached 16 discards (%s)",
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

        if all(len(player.discards) >= 16 for player in self.state.players):
            logger.info(
                "和了見送り後に流局: all players reached 16 discards (%s)",
                {player.player_id: len(player.discards) for player in self.state.players},
            )
            self.end_round(is_draw=True)
            return False

        self._advance_player()
        return False

    def _get_liquidation_multiplier(self, han: int) -> float:
        """翻数から精算倍率を返す。"""
        if han >= 26:
            return 8.0
        if han >= 13:
            return 4.0
        if han >= 11:
            return 3.0
        if han >= 8:
            return 2.0
        if han >= 6:
            return 1.5
        return 1.0

    def _is_tanki_wait_agari(self, hand: list[int], winning_tile: int, winner_waits: list[int]) -> bool:
        """単騎待ちでの和了かどうかを判定する。"""
        if winning_tile is None or not winner_waits:
            return False

        winning_base = winning_tile & 0b11111
        wait_bases = [wait & 0b11111 for wait in winner_waits]

        # 単騎待ちは待ち牌が1種類で、その牌で和了している必要がある。
        if len(set(wait_bases)) != 1 or wait_bases[0] != winning_base:
            return False

        counter = Counter(tile & 0b11111 for tile in hand)
        if counter[winning_base] < 2:
            return False

        temp_counter = counter.copy()
        temp_counter[winning_base] -= 2
        return any(len(melds) == 4 for melds in HandAnalyzer._generate_melds(temp_counter))

    def _is_ippatsu_agari(self, winner: PlayerState, loser: PlayerState, base_yaku_list: list[str]) -> bool:
        """一発成立かどうかを返す。両者の一打目までを一発圏とする。"""
        if Yaku.RICHI.japanese_name not in base_yaku_list:
            return False

        return len(winner.discards) <= 1 and len(loser.discards) == 1

    def _is_houtei_raoyui(self, loser: PlayerState, winning_tile: Optional[int]) -> bool:
        """河底撈魚成立かどうかを返す。後手側の16打目を最後の捨て牌とみなす。"""
        if winning_tile is None or len(loser.discards) != 16:
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
        base_yaku_list = HandAnalyzer.enum_yaku(hand, winning_tile=winning_tile)
        base_yaku_list += self._get_win_context_yaku(winner, loser, winning_tile, base_yaku_list)

        # 満貫判定にはスキルによる翻上昇分も含める
        boost_bonus_map = self._effective_boost_bonus_map(winner)
        base_han = sum(Yaku.get_han_by_name(name) for name in base_yaku_list)
        bonus_han = sum(boost_bonus_map.get(name, 0) for name in base_yaku_list)
        han = base_han + bonus_han
        is_mangan = han >= 4
        if not is_win or not is_mangan:
            logger.debug("上がり条件不成立: player=%s  is_win=%s  check_mangan=%s", player_id, is_win, is_mangan)
            return False

        is_tanki_wait = self._is_tanki_wait_agari(hand, winning_tile, winner.waits)

        # 役倍率（跳満 1.5倍 / 倍満 2倍 / 三倍満 3倍 / 役満 4倍）
        display_yaku_list = self._build_display_yaku_list(base_yaku_list, boost_bonus_map)
        multiplier = self._get_liquidation_multiplier(han)
        logger.info("精算: winner=%s  yaku=%s  base_han=%d  bonus_han=%d  multiplier=%.1f",
                    winner.player_id, display_yaku_list, base_han, bonus_han, multiplier)

        # 勝者: 自身の賭け金 × 自身の役倍率分を獲得
        winner_gain = int(winner.bet * multiplier)
        winner.health += winner_gain

        # 敗者: 単騎待ち和了時のみ支払いを倍化（勝者獲得量は据え置き）
        loser_loss_multiplier = 2 if is_tanki_wait else 1
        loser_loss = int(loser.bet * multiplier * loser_loss_multiplier)
        loser.health = max(0, loser.health - loser_loss)

        self._last_liquidation_result = {
            "winner_id": winner.player_id,
            "loser_id": loser.player_id,
            "yaku": display_yaku_list,
            "base_yaku": base_yaku_list,
            "display_yaku": display_yaku_list,
            "base_han": base_han,
            "bonus_han": bonus_han,
            "han": han,
            "multiplier": multiplier,
            "winner_bet": winner.bet,
            "loser_bet": loser.bet,
            "winner_gain": winner_gain,
            "loser_loss": loser_loss,
            "loser_loss_multiplier": loser_loss_multiplier,
            "is_tanki_wait": is_tanki_wait,
            "winner_health": winner.health,
            "loser_health": loser.health,
        }

        self.end_round()
        return True

    def _prepare_next_round(self) -> None:
        """次局開始に必要な状態を準備する。"""
        self.state.round_state.round_number += 1
        self._pending_agari = None
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
                special_victory_count=p.special_victory_count,
                boost_hand_bonus=p.boost_hand_bonus.copy(),
                exposed_hand_indexes=p.exposed_hand_indexes.copy(),
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
            self._on_game_end()
            return

        # 精算後に HP が 0 以下のプレイヤーがいればゲーム終了
        if any(p.health <= 0 for p in self.state.players):
            dead = [p.player_id for p in self.state.players if p.health <= 0]
            logger.info("HPゼロによるゲーム終了: players=%s", dead)
            self._invoke_callback(self.on_round_end, is_draw)
            self._on_game_end()
            return

        self._carry_over_bets = is_draw
        self._next_round_ready_players.clear()
        self._set_phase(RoundStatus.ROUND_END_WAITING)
        self._invoke_callback(self.on_round_end, is_draw)

    def _on_game_end(self) -> None:
        """ゲーム終了時の処理"""
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

    def get_waits(self, hand_indexes: list[int], player: PlayerState) -> list[tuple[int, bool, list[str], list[str]]]:
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
        """
        # index から牌 ID に変換
        hand_tiles = [player.wall[idx] for idx in hand_indexes]
        
        if not HandAnalyzer.is_tenpai(hand_tiles, player.wall):
            return None

        waits = HandAnalyzer.get_tenpai_waiting_tiles(hand_tiles, player.wall)
        waits = [w+32 if w == self.state.round_state.dora_id else w for w in waits]

        result = []
        boost_bonus_map = self._effective_boost_bonus_map(player)
        for w in waits:
            base_yaku_list = HandAnalyzer.enum_yaku(hand_tiles + [w], winning_tile=w)
            base_han = sum(Yaku.get_han_by_name(y) for y in base_yaku_list)
            bonus = sum(boost_bonus_map.get(y, 0) for y in base_yaku_list)
            display_yaku_list = self._build_display_yaku_list(base_yaku_list, boost_bonus_map)
            result.append((w, base_han + bonus >= 4, display_yaku_list, base_yaku_list))
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
                    "hand": p.hand,
                    "wall": p.wall,
                    "waits": p.waits,
                    "discards": p.discards
                }
                for p in self.state.players
            ]
        }