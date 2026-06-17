import asyncio
import heapq
import logging
import random
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from ..engine.game_engine import GameEngine
from ..engine.game_state import RoundStatus, SkillType, PlayerState
from ..engine.hand_analyzer import HandAnalyzer

logger = logging.getLogger(__name__)


class GameSession:
	"""ゲーム進行に関する処理を担当"""

	def __init__(
		self,
		lock: asyncio.Lock,
		matches: Dict[str, Any],
		active_match_by_client: Dict[str, str],
		game_engines: Dict[str, GameEngine],
		available_match_numbers: List[int],
		send_to_client: Callable[[str, Dict[str, Any]], Awaitable[None]],
		broadcast_match_members: Callable[[str, Dict[str, Any]], Awaitable[None]],
	):
		self._lock = lock
		self._matches = matches
		self._active_match_by_client = active_match_by_client
		self._game_engines = game_engines
		self._available_match_numbers = available_match_numbers
		self._send_to_client = send_to_client
		self._broadcast_match_members = broadcast_match_members
		self._pending_low_hand_confirmations: Dict[str, List[int]] = {}
		self._confirmed_hand_players_by_match: Dict[str, Set[str]] = {}

	async def _send_error(self, client_id: str, message: str) -> None:
		"""クライアントへのエラー通知"""
		try:
			await self._send_to_client(client_id, {"type": "error", "message": message})
		except Exception:
			pass

	async def _respond_to_client(self, client_id: str, payload: Dict[str, Any]) -> bool:
		"""クライアント応答。送信失敗時はエラー通知を試行する。"""
		try:
			await self._send_to_client(client_id, payload)
			return True
		except Exception:
			await self._send_error(client_id, "Failed to deliver response")
			return False

	async def _ensure_phase(
		self,
		engine: GameEngine,
		client_id: str,
		required_status: RoundStatus,
		action_name: str,
	) -> bool:
		"""現在フェーズが required_status かを検証し、違えば統一エラーを返す。"""
		if engine.state.round_state.status == required_status:
			return True

		cur = engine.state.round_state.status.value if engine.state.round_state.status else None
		await self._send_error(
			client_id,
			f"{action_name} は {required_status.value.upper()} フェーズでのみ実行可能 (current={cur})",
		)
		return False

	def _validate_hand_indexes(self, wall: List[int], wall_indexes: Any) -> Optional[str]:
		"""hand_indexes の形式・範囲を検証し、エラー文言を返す。正常時は None。"""
		if not isinstance(wall_indexes, list) or not wall_indexes:
			return "hand_indexes は空でない整数配列が必要です"

		if not all(isinstance(idx, int) for idx in wall_indexes):
			return "hand_indexes の各要素は整数でなければなりません"

		if len(set(wall_indexes)) != len(wall_indexes):
			return "hand_indexes に重複値が含まれています"

		for idx in wall_indexes:
			if idx < 0 or idx >= len(wall):
				return f"hand_indexes[値={idx}] が wall 範囲外です (wall長={len(wall)})"

		return None

	def _convert_tiles_to_wall_indexes(self, wall: List[int], tiles: List[int]) -> Optional[List[int]]:
		"""牌IDの並びを、対応する wall index の並びへ変換する（重複牌対応）。"""
		if not isinstance(tiles, list):
			return None

		used = [False] * len(wall)
		result: List[int] = []

		for tile in tiles:
			found_index = None
			tile_base = tile & 0b11111
			for idx, wall_tile in enumerate(wall):
				if used[idx]:
					continue
				# dora/aka など上位ビット差で不一致にならないよう牌種で比較
				if (wall_tile & 0b11111) == tile_base:
					found_index = idx
					break

			if found_index is None:
				return None

			used[found_index] = True
			result.append(found_index)

		return result

	def _find_any_tenpai_example_indexes(self, wall: List[int], dora_id: int) -> Optional[List[int]]:
		"""wall から満貫以上の聴牌形を再探索して、最初に index 変換できる例を返す。"""
		try:
			candidates = HandAnalyzer.search_tenpai(wall, wall, dora_id)
		except Exception:
			return None

		for hand_tiles in candidates:
			indexes = self._convert_tiles_to_wall_indexes(wall, hand_tiles)
			if indexes is not None:
				return indexes

		return None

	def _create_task_callback(
		self,
		handler: Callable[..., Awaitable[None]],
		*fixed_args,
	) -> Callable[..., None]:
		"""コールバックから非同期ハンドラを fire-and-forget で起動する。"""
		def _callback(*args) -> None:
			asyncio.create_task(handler(*fixed_args, *args))

		return _callback

	def _clear_pending_confirmations_for_engine(self, engine: GameEngine) -> None:
		for p in engine.state.players:
			self._pending_low_hand_confirmations.pop(p.player_id, None)

	def _mark_hand_selection_confirmed(self, match_id: str, client_id: str) -> None:
		self._confirmed_hand_players_by_match.setdefault(match_id, set()).add(client_id)

	def _unmark_hand_selection_confirmed(self, match_id: str, client_id: str) -> None:
		confirmed = self._confirmed_hand_players_by_match.get(match_id)
		if confirmed is None:
			return
		confirmed.discard(client_id)
		if not confirmed:
			self._confirmed_hand_players_by_match.pop(match_id, None)

	def _clear_hand_selection_confirmations(self, match_id: str) -> None:
		self._confirmed_hand_players_by_match.pop(match_id, None)

	def _are_all_hand_selections_confirmed(self, match_id: str, engine: GameEngine) -> bool:
		confirmed = self._confirmed_hand_players_by_match.get(match_id, set())
		return len(confirmed) >= engine.num_players

	def _apply_opening_boosts(self, engine: GameEngine) -> List[Dict[str, Any]]:
		"""ゲーム開始時に各プレイヤーへ恒常強化を1つずつ付与する。"""
		candidates = GameEngine.get_opening_boost_candidates()
		if not candidates:
			return []

		assigned: List[Dict[str, Any]] = []
		for player in engine.state.players:
			yaku_name = random.choice(candidates)
			if not engine.assign_opening_boost(player, yaku_name, bonus_han=1):
				continue
			assigned.append(
				{
					"client_id": player.player_id,
					"yaku_name": yaku_name,
					"bonus_han": 1,
				}
			)

		return assigned

	async def start_match(self, match: Any) -> None:
		try:
			await self._start_match_inner(match)
		except Exception:
			logger.exception("マッチ開始に失敗: match_id=%s", getattr(match, 'match_id', '?'))
			raise

	async def _start_match_inner(self, match: Any) -> None:
		engine = GameEngine(max_rounds=4)
		engine.initialize_players(match.players)
		original_deal_tiles = engine._deal_tiles
		round_start_done = asyncio.Event()
		dealing_phase_done = asyncio.Event()

		engine.on_dealt = self._create_task_callback(self.on_dealt, match.match_id)
		engine.on_selected = self._create_task_callback(self.on_selected, match.match_id)
		engine.on_bet = self._create_task_callback(self.on_bet, match.match_id)

		engine.on_discard_started = self._create_task_callback(self.on_discard_started, match.match_id)
		engine.on_discarded = self._create_task_callback(self.on_discarded, match.match_id)
		engine.on_agari_pending = self._create_task_callback(self.on_agari_pending, match.match_id)
		engine.on_skill_casted = self._create_task_callback(self.on_skill_casted, match.match_id)

		def _on_round_start() -> None:
			round_start_done.clear()
			dealing_phase_done.clear()

			async def _notify_round_start() -> None:
				try:
					await self.on_round_start(match.match_id)
				finally:
					round_start_done.set()

			async def _wait_and_start_deal() -> None:
				await round_start_done.wait()
				await dealing_phase_done.wait()
				original_deal_tiles()

			asyncio.create_task(_notify_round_start())
			asyncio.create_task(_wait_and_start_deal())

		engine.on_round_start = _on_round_start
		engine.on_round_end = self._create_task_callback(self.on_round_end, match.match_id)
		engine.on_game_end = self._create_task_callback(self.on_game_end, match.match_id)
		engine.on_special_victory_won = self._create_task_callback(self.on_special_victory_won, match.match_id)

		def _on_phase_change(new_status: RoundStatus) -> None:
			async def _notify_phase_change() -> None:
				try:
					await self.on_phase_change(match.match_id, new_status)
				finally:
					if new_status == RoundStatus.DEALING:
						dealing_phase_done.set()

			asyncio.create_task(_notify_phase_change())

		engine.on_phase_change = _on_phase_change
		engine._deal_tiles = lambda: None

		payload = {
			"type": "game_started",
			"data": {
				"match_id": match.match_id,
				"players": [{"client_id": cid} for cid in match.players],
			},
		}

		await asyncio.gather(
			*(self._respond_to_client(cid, payload) for cid in match.players),
			return_exceptions=True,
		)

		self._game_engines[match.match_id] = engine
		logger.info("マッチ開始: match_id=%s  players=%s", match.match_id, match.players)

		opening_boosts = self._apply_opening_boosts(engine)
		if opening_boosts:
			await self._broadcast_match_members(
				match.match_id,
				{
					"type": "opening_boost_assigned",
					"data": {
						"boosts": opening_boosts,
					},
				},
			)

		engine.start_game(1000)

	async def handle_game_action(self, client_id: str, data: Dict[str, Any]) -> None:
		"""
		クライアントからのゲームアクションを処理
		Args:
			client_id: アクションを送信したクライアントのID
			data: アクションのデータ
		"""
		match_id = self._active_match_by_client.get(client_id)
		if not match_id or match_id not in self._game_engines:
			await self._send_error(client_id, "Not in game")
			return

		engine = self._game_engines[match_id]
		action_type = data.get("action")
		action_data = data.get("data")

		if not action_type:
			await self._send_error(client_id, "No action type specified")
			return
		if action_data is None:
			action_data = {}
		if not isinstance(action_data, dict):
			await self._send_error(client_id, "Invalid action data")
			return

		action_handlers = {
			"status": self._status,
			"is_tenpai": self._is_tenpai,
			"skill": self._skill,
			"select": self._select,
			"select_confirm": self._select_confirm,
			"bet": self._bet,
			"discard": self._discard,
			"agari": self._agari,
			"next_round": self._next_round,
		}
		handler = action_handlers.get(action_type)
		if handler is None:
			await self._send_error(client_id, "Unsupported action type")
			return

		try:
			await handler(engine, client_id, action_data)
		except Exception:
			tb = traceback.format_exc()
			logger.error("アクション処理中に例外: client=%s  action=%s\n%s",
				client_id, action_type, tb)
			await self._send_error(client_id, f"Internal error while processing '{action_type}'")

	def _serialize_round_state(self, engine: GameEngine) -> Dict[str, Any]:
		round_state = engine.state.round_state
		return {
			"round_number": round_state.round_number,
			"current_player_index": round_state.current_player_index,
			"first_player_index": round_state.first_player_index,
			"status": round_state.status.value if round_state.status else None,
			"dora_id": round_state.dora_id,
			"reserved_tiles": list(round_state.reserved_tiles),
		}

	def _serialize_player_state(self, player: PlayerState) -> Dict[str, Any]:
		return {
			"player_id": player.player_id,
			"hand": list(player.hand),
			"wall": list(player.wall),
			"waits": list(player.waits),
			"discards": list(player.discards),
			"discarded_wall_indexes": sorted(player.discarded_wall_indexes),
			"health": player.health,
			"bet": player.bet,
			"special_victory_count": player.special_victory_count,
			"boost_hand_bonus": dict(player.boost_hand_bonus),
			"exposed_hand_indexes": sorted(player.exposed_hand_indexes),
		}

	async def _status(self, engine: GameEngine, client_id: str, _action_data: Dict[str, Any]) -> None:
		"""状態取得アクションの処理"""
		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		opponent = next((p for p in engine.state.players if p.player_id != client_id), None)
		all_player_states = [self._serialize_player_state(p) for p in engine.state.players]

		await self._respond_to_client(
			client_id,
			{
				"type": "status",
				"data": {
					"game_state": engine.get_game_state(),
					"round_state": self._serialize_round_state(engine),
					"player_state": self._serialize_player_state(player),
					"opponent_player_state": self._serialize_player_state(opponent) if opponent else None,
					"player_states": all_player_states,
				},
			},
		)

	async def _is_tenpai(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""手牌の聴牌判定の処理"""

		if not await self._ensure_phase(engine, client_id, RoundStatus.HAND_SELECTION, "is_tenpai"):
			return

		wall_indexes = action_data.get("wall_indexes")

		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		validation_error = self._validate_hand_indexes(player.wall, wall_indexes)
		if validation_error is not None:
			await self._send_error(client_id, validation_error.replace("hand_indexes", "wall_indexes"))
			return

		waits = engine.get_waits(wall_indexes, player)
		if waits is not None:
			await self._respond_to_client(client_id, {
				"type": "is_tenpai",
				"data": {
					"waits": [
						{
							"tile" : w[0],
							"mangan_or_more" : w[1],
							"yaku" : w[2],
							"base_yaku": w[3],
						} for w in waits
					]
				},
			})

		else:
			await self._respond_to_client(client_id, {
				"type": "not_tenpai",
				"message": "Hand is not in tenpai",
			})

	async def _skill(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""スキルアクションの処理"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.HAND_SELECTION, "skill"):
			return

		# スキル種別の検証
		raw_skill_type = action_data.get("skill_type")
		if not isinstance(raw_skill_type, str):
			await self._send_error(client_id, "Invalid skill_type")
			return

		try:
			skill_type = SkillType(raw_skill_type)
		except ValueError:
			await self._send_error(client_id, f"Unsupported skill_type: {raw_skill_type}")
			return

		# プレイヤーの取得
		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		cost = await self._execute_skill(engine, player, skill_type, action_data)

		if cost is None:
			try:
				from ..engine.game_state import get_skill_cost
				skill_cost = get_skill_cost(skill_type, player.special_victory_count)
				detail = f" (cost={skill_cost}, health={player.health})"
			except Exception:
				detail = ""
			await self._send_error(client_id, f"スキル '{skill_type.value}' の発動に失敗しました{detail}")
			return

		# レスポンス（Python の snake_case を JSON のキャメルケースに変換）
		await self._respond_to_client(client_id, {
			"type": "skill_accepted",
			"data": {
				"skillType": skill_type.value,
				"cost": cost,
				"currentHealth": player.health,
			},
		})

	async def _execute_skill(self, engine: GameEngine, user: PlayerState, skill_type: SkillType, action_data: Dict[str, Any]) -> Optional[int]:
		"""
		スキルを実行（種別ごとの入力検証 + engine 呼び出し）

		Returns:
			HP コスト。失敗時は None
		"""
		if skill_type == SkillType.BOOST_HAND:
			# 対象役名を検証
			yaku_name = action_data.get("yaku_name")
			normalized_yaku_name = GameEngine.normalize_boost_target_yaku_name(yaku_name)
			if normalized_yaku_name is None:
				return None
			return engine.use_skill(user, skill_type, normalized_yaku_name)

		elif skill_type == SkillType.SPECIAL_VICTORY:
			return engine.use_skill(user, skill_type)

		elif skill_type == SkillType.PERSPECTIVE:
			# 相手を取得
			target = next((p for p in engine.state.players if p.player_id != user.player_id), None)
			if target is None:
				return None
			return engine.use_skill_on_opponent(user, target, skill_type)

		elif skill_type == SkillType.MULLIGAN:
			# 交換対象インデックスを検証
			target_index = action_data.get("target_hand_index")
			if not isinstance(target_index, int) or target_index < 0 or target_index >= len(user.wall):
				return None
			return engine.use_mulligan(user, target_index)

		return None

	async def _select(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""手牌選択アクションの処理"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.HAND_SELECTION, "select"):
			return

		match_id = self._active_match_by_client.get(client_id)
		if not match_id:
			await self._send_error(client_id, "Not in game")
			return

		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		wall_indexes = action_data.get("hand_indexes", action_data.get("hand"))

		validation_error = self._validate_hand_indexes(player.wall, wall_indexes)
		if validation_error is not None:
			await self._send_error(client_id, validation_error)
			return

		waits = engine.get_waits(wall_indexes, player)
		is_tenpai = waits is not None
		is_mangan_or_more = bool(waits and any(w[1] for w in waits))

		if not is_mangan_or_more:
			reason = "not_tenpai" if not is_tenpai else "below_mangan"
			self._pending_low_hand_confirmations[client_id] = wall_indexes.copy()
			self._unmark_hand_selection_confirmed(match_id, client_id)
			await self._respond_to_client(client_id, {
				"type": "hand_selection_confirmation_required",
				"data": {
					"reason": reason,
					"message": "満貫以上ではありません。確定する場合は select_confirm を送信してください。",
					"hand_indexes": wall_indexes,
				},
			})
			return

		if engine.select_hand(wall_indexes, player):
			self._pending_low_hand_confirmations.pop(client_id, None)
			self._mark_hand_selection_confirmed(match_id, client_id)
			# player.hand は wall 内 index を保持するため、クライアントへは牌 ID に変換して送信する
			hand_tiles = [player.wall[idx] for idx in player.hand]

			await self._respond_to_client(client_id, {
				"type": "hand_selection_accepted",
				"data": {
					"hand": hand_tiles,  # 牌 ID
					"waits": player.waits,
					"wall": player.wall,
				},
			})

			if self._are_all_hand_selections_confirmed(match_id, engine):
				self._clear_hand_selection_confirmations(match_id)
				engine.selected_hand()

		else:
			await self._send_error(client_id, "手牌が聴牌であるか、満貫以上の役の可能性が必要です")

	async def _select_confirm(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""二段階目の手牌確定アクション（満貫以下・非聴牌でも確定可能）"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.HAND_SELECTION, "select_confirm"):
			return

		match_id = self._active_match_by_client.get(client_id)
		if not match_id:
			await self._send_error(client_id, "Not in game")
			return

		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		wall_indexes = action_data.get("hand_indexes", action_data.get("hand"))
		validation_error = self._validate_hand_indexes(player.wall, wall_indexes)
		if validation_error is not None:
			await self._send_error(client_id, validation_error)
			return

		pending = self._pending_low_hand_confirmations.get(client_id)
		if pending is None:
			await self._send_error(client_id, "select を先に実行してください")
			return

		if pending != wall_indexes:
			await self._send_error(client_id, "select_confirm の hand_indexes が直前の select と一致しません")
			return

		if not engine.select_hand(wall_indexes, player, force=True):
			await self._send_error(client_id, "手牌の確定に失敗しました")
			return

		self._pending_low_hand_confirmations.pop(client_id, None)
		self._mark_hand_selection_confirmed(match_id, client_id)
		hand_tiles = [player.wall[idx] for idx in player.hand]

		await self._respond_to_client(client_id, {
			"type": "hand_selection_accepted",
			"data": {
				"hand": hand_tiles,
				"waits": player.waits,
				"wall": player.wall,
				"forced": True,
			},
		})

		if self._are_all_hand_selections_confirmed(match_id, engine):
			self._clear_hand_selection_confirmations(match_id)
			engine.selected_hand()

	async def _bet(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""掛け金設定アクションの処理"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.BETTING, "bet"):
			return

		player = engine.get_player_by_id(client_id)
		if player is None:
			await self._send_error(client_id, "Player not found")
			return

		bet_amount = action_data.get("bet_amount", action_data.get("bet"))
		if not isinstance(bet_amount, int):
			await self._send_error(client_id, f"bet_amount は整数で必要です (got={type(bet_amount).__name__})")
			return

		max_bet, bet_unit = engine.get_bet_rule(player)
		if not engine.place_bet(player, bet_amount):
			await self._send_error(
				client_id,
				f"Invalid bet_amount (max={max_bet}, unit={bet_unit}, health={player.health})",
			)
			return

		await self._respond_to_client(client_id, {
			"type": "bet_accepted",
			"data": {
				"bet_amount": bet_amount,
				"max_bet": max_bet,
				"bet_unit": bet_unit,
			},
		})

		if all(p.bet > 0 for p in engine.state.players):
			engine.bet()

	async def _discard(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""打牌アクションの処理"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.DISCARD, "discard"):
			return

		if engine.get_pending_agari() is not None:
			await self._send_error(client_id, "和了入力待ち中のため打牌できません")
			return

		player = engine.get_current_player()
		if player.player_id != client_id:
			await self._send_error(client_id, f"現在の手番ではありません (current_player={player.player_id})")
			return

		if "wall_index" not in action_data:
			await self._send_error(client_id, "No wall_index specified")
			return

		wall_index = action_data["wall_index"]
		if not isinstance(wall_index, int):
			await self._send_error(client_id, f"wall_index は整数で必要です (got={type(wall_index).__name__})")
			return

		if wall_index < 0 or wall_index >= len(player.wall):
			await self._send_error(client_id, f"wall_index が範囲外です (index={wall_index}, wall長={len(player.wall)})")
			return

		if wall_index in player.discarded_wall_indexes:
			await self._send_error(client_id, f"wall_index={wall_index} は既に打牌済みです")
			return

		tile = player.wall[wall_index]
		is_win = engine.discard(client_id, wall_index)
		liquidation_result = engine.get_last_liquidation_result() if is_win else None

		await self._respond_to_client(client_id, {
			"type": "discard_accepted",
			"data": {
				"wall_index": wall_index,
				"tile": tile,
				"is_win": is_win,
				"liquidation": liquidation_result,
			},
		})

	async def _agari(self, engine: GameEngine, client_id: str, action_data: Dict[str, Any]) -> None:
		"""和了入力アクションの処理"""
		if not await self._ensure_phase(engine, client_id, RoundStatus.DISCARD, "agari"):
			return

		if engine.get_player_by_id(client_id) is None:
			await self._send_error(client_id, "Player not found")
			return

		accept = action_data.get("accept", True)
		if not isinstance(accept, bool):
			await self._send_error(client_id, f"accept は bool で必要です (got={type(accept).__name__})")
			return

		result = engine.resolve_pending_agari(client_id, accept)
		if result is None:
			await self._send_error(client_id, "和了入力待ちが存在しないか、操作権限がありません")
			return

		await self._respond_to_client(
			client_id,
			{
				"type": "agari_accepted",
				"data": {
					"accepted": accept,
					"is_win": bool(result),
					"liquidation": engine.get_last_liquidation_result() if result else None,
				},
			},
		)

	async def _next_round(self, engine: GameEngine, client_id: str, _action_data: Dict[str, Any]) -> None:
		"""次局進行承認アクションの処理"""
		match_id = self._active_match_by_client.get(client_id)
		if not match_id:
			await self._send_error(client_id, "Not in game")
			return

		if not await self._ensure_phase(engine, client_id, RoundStatus.ROUND_END_WAITING, "next_round"):
			return

		if engine.get_player_by_id(client_id) is None:
			await self._send_error(client_id, "Player not found")
			return

		already_ready = client_id in engine.get_next_round_ready_players()
		if not engine.confirm_next_round(client_id):
			await self._send_error(client_id, "Failed to accept next round")
			return

		ready_players = engine.get_next_round_ready_players()
		is_still_waiting = engine.state.round_state.status == RoundStatus.ROUND_END_WAITING
		ready_count = len(ready_players) if is_still_waiting else engine.num_players
		await self._respond_to_client(client_id, {
			"type": "next_round_accepted",
			"data": {
				"already_ready": already_ready,
				"ready_players": ready_players,
				"ready_count": ready_count,
				"required_count": engine.num_players,
				"started": not is_still_waiting,
			},
		})

		if is_still_waiting:
			await self._broadcast_match_members(
				match_id,
				{
					"type": "next_round_waiting",
					"data": {
						"ready_players": ready_players,
						"ready_count": len(ready_players),
						"required_count": engine.num_players,
					},
				},
			)

	async def on_dealt(self, match_id: str) -> None:
		"""配牌完了時の処理"""
		engine = self._game_engines.get(match_id)
		self._clear_hand_selection_confirmations(match_id)
		if engine is not None:
			self._clear_pending_confirmations_for_engine(engine)

		wall = [(p.wall, p.hand) for p in self._game_engines[match_id].state.players]
		dora = self._game_engines[match_id].state.round_state.dora_id

		hand_payload = []
		for i, h in enumerate(wall):
			wall_tiles = h[0]
			tenpai_example = h[1]

			# 現在は GameEngine 側で wall index を保持する。
			if (
				isinstance(tenpai_example, list)
				and all(isinstance(idx, int) for idx in tenpai_example)
				and all(0 <= idx < len(wall_tiles) for idx in tenpai_example)
			):
				tenpai_example_indexes = tenpai_example
			else:
				tenpai_example_indexes = self._convert_tiles_to_wall_indexes(
					wall_tiles,
					tenpai_example,
				)

			if tenpai_example_indexes is None:
				# 稀に牌IDの差異で変換できないケースがあるため、wallから再探索して補完
				tenpai_example_indexes = self._find_any_tenpai_example_indexes(wall_tiles, dora)
				if tenpai_example_indexes is None:
					logger.warning("tenpai_example の index 変換に失敗: match_id=%s player_index=%s", match_id, i)
					tenpai_example_indexes = []

			hand_payload.append(
				{
					"client_id": await self.resolve_client_id(match_id, i),
					"wall": wall_tiles,
					"tenpai_examples": tenpai_example_indexes,
				}
			)

		await self._broadcast_match_members(
			match_id,
			{
				"type": "dealing_completed",
				"dora_id": dora,
				"hands": hand_payload,
			},
		)

	async def on_selected(self, match_id: str) -> None:
		"""手牌選択完了の処理"""
		payload = [
			{
				"client_id": await self.resolve_client_id(match_id, i),
				"hand": [self._game_engines[match_id].state.players[i].wall[idx] for idx in player.hand],  # indexから牌 ID に変換
				"waits": player.waits,
				"wall": player.wall,
			}
			for i, player in enumerate(self._game_engines[match_id].state.players)
		]

		await self._broadcast_match_members(
			match_id,
			{
				"type": "hand_selection_completed",
				"data": {
					"hands": payload,
				},
			},
		)

	async def on_bet(self, match_id: str) -> None:
		"""掛け金設定完了時の処理"""
		bet = [(p.bet) for p in self._game_engines[match_id].state.players]
		bet_payload = [
			{
				"client_id": await self.resolve_client_id(match_id, i),
				"bet": b,
			}
			for i, b in enumerate(bet)
		]

		await self._broadcast_match_members(
			match_id,
			{
				"type": "bet_completed",
				"data": {
					"bets": bet_payload,
				},
			},
		)

	async def on_discard_started(self, match_id: str) -> None:
		"""打牌フェーズ開始時の処理"""
		engine = self._game_engines[match_id]

		await self._broadcast_match_members(
			match_id,
			{
				"type": "discard_phase_started",
				"data": {
					"first_player":  engine.get_current_player().player_id,
				},
			},
		)

	async def on_discarded(self, match_id: str, player_id: str, tile_id: int) -> None:
		"""打牌実行時の処理"""
		await self._broadcast_match_members(
			match_id,
			{
				"type": "discard_completed",
				"data": {
					"player_id": player_id,
					"tile": tile_id,
				},
			},
		)

	async def on_agari_pending(self, match_id: str, winner_id: str, loser_id: str, tile_id: int) -> None:
		"""和了入力待ち発生時の処理"""
		await self._broadcast_match_members(
			match_id,
			{
				"type": "agari_pending",
				"data": {
					"winner_id": winner_id,
					"loser_id": loser_id,
					"tile": tile_id,
				},
			},
		)

	async def on_skill_casted(self, match_id: str, player_id: str, skill_type: SkillType, cost: int, exposed_indexes: Any) -> None:
		"""スキル使用時の処理"""
		player = self._game_engines[match_id].get_player_by_id(player_id)

		exposed_by_player: Dict[str, List[int]] = {}
		exposed_flat: List[int] = []
		if isinstance(exposed_indexes, dict):
			for cid, indexes in exposed_indexes.items():
				try:
					normalized = sorted(int(i) for i in indexes)
				except Exception:
					normalized = []
				exposed_by_player[str(cid)] = normalized
				exposed_flat.extend(normalized)
			exposed_flat = sorted(set(exposed_flat))
		elif exposed_indexes:
			try:
				exposed_flat = sorted(int(i) for i in exposed_indexes)
			except Exception:
				exposed_flat = []

		await self._broadcast_match_members(
			match_id,
			{
				"type": "skill_casted",
				"data": {
					"player_id": player_id,
					"skillType": skill_type.value,
					"cost": cost,
					"health": player.health if player else None,
					"exposedHandIndexes": exposed_flat,
					"exposedHandIndexesByPlayer": exposed_by_player,
				},
			},
		)

	async def on_round_start(self, match_id: str) -> None:
		"""ラウンド開始時の処理"""
		await self._broadcast_match_members(
			match_id,
			{
				"type": "round_start",
				"round": self._game_engines[match_id].state.round_state.round_number,
			},
		)

	async def on_round_end(self, match_id: str, is_draw: bool = False) -> None:
		"""ラウンド終了時の処理"""
		engine = self._game_engines.get(match_id)
		self._clear_hand_selection_confirmations(match_id)
		liquidation = engine.get_last_liquidation_result() if engine and not is_draw else None

		await self._broadcast_match_members(
			match_id,
			{
				"type": "round_end",
				"data": {
					"is_draw": is_draw,
					"liquidation": liquidation,
				},
			},
		)

		if engine is not None and engine.state.round_state.status == RoundStatus.ROUND_END_WAITING:
			await self._broadcast_match_members(
				match_id,
				{
					"type": "next_round_waiting",
					"data": {
						"ready_players": engine.get_next_round_ready_players(),
						"ready_count": 0,
						"required_count": engine.num_players,
					},
				},
			)

	async def on_special_victory_won(self, match_id: str, player_id: str) -> None:
		"""特殊勝利（3回目使用）による勝利時の処理"""
		engine = self._game_engines.get(match_id)
		if not engine:
			return

		# 勝者情報をブロードキャスト
		await self._broadcast_match_members(
			match_id,
			{
				"type": "special_victory_won",
				"data": {
					"player_id": player_id,
				},
			},
		)

		# ゲーム終了処理を直接呼ぶ（on_game_end が呼ばれる）
		await self.on_game_end(match_id)

	async def on_game_end(self, match_id: str) -> None:
		"""ゲーム終了時の処理"""
		logger.info("ゲーム終了: match_id=%s", match_id)
		engine = self._game_engines.get(match_id)
		self._clear_hand_selection_confirmations(match_id)
		if engine:
			self._clear_pending_confirmations_for_engine(engine)

			final_scores = {}
			for p in engine.state.players:
				resolved_client_id = await self.resolve_client_id(match_id, p.player_id)
				score_key = resolved_client_id if resolved_client_id is not None else str(p.player_id)
				final_scores[score_key] = p.health

			await self._broadcast_match_members(
				match_id,
				{
					"type": "game_end",
					"final_scores": final_scores,
				},
			)

			async with self._lock:
				self._game_engines.pop(match_id, None)
				match = self._matches.pop(match_id, None)
				if match_id.startswith("M") and match_id[1:].isdigit():
					heapq.heappush(self._available_match_numbers, int(match_id[1:]))
				for player_id in (match.players if match else []):
					self._active_match_by_client.pop(player_id, None)

	async def on_phase_change(self, match_id: str, new_status: RoundStatus) -> None:
		"""ラウンドステータス変更時の処理"""
		await self._broadcast_match_members(
			match_id,
			{
				"type": "phase_change",
				"new_status": new_status.value,
			},
		)

	async def resolve_client_id(self, match_id: str, engine_player_id: Any) -> Optional[str]:
		try:
			index = int(engine_player_id)
		except (TypeError, ValueError):
			return None

		async with self._lock:
			match = self._matches.get(match_id)
			if match is None:
				return None
			if index < 0 or index >= len(match.players):
				return None
			return match.players[index]