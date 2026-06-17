"""
WebSocket API サーバー

ゲームの流れ:
1. 接続したプレイヤーを待機キューへ追加
2. 2人揃ったらマッチ成立
3. その2人に game_started を通知
"""

import asyncio
import heapq
import websockets
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from .game_session import GameSession

@dataclass
class MatchSession:
	"""マッチ情報"""

	match_id: str
	players: List[str] = field(default_factory=list)
	status: str = "in_game"
	created_at: float = field(default_factory=time.time)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"match_id": self.match_id,
			"players": list(self.players),
			"status": self.status,
			"created_at": self.created_at,
		}


class WebSocketGameServer:
	"""WebSocket ゲームサーバー（2人マッチング）"""

	def __init__(self, host: str = "127.0.0.1", port: int = 8765, max_players: int = 2):
		self.host = host
		self.port = port
		self.max_players = max_players

		self._server = None
		self._websockets = None
		self._lock = asyncio.Lock()

		self._connections: Set[Any] = set()
		self._client_seq = 0
		self._available_client_numbers: List[int] = []
		self._match_seq = 0
		self._available_match_numbers: List[int] = []

		self._client_id_by_socket: Dict[Any, str] = {}
		self._socket_by_client_id: Dict[str, Any] = {}
		self._client_name: Dict[str, str] = {}

		# マッチング待機キュー（先着順）
		self._waiting_queue: List[str] = []

		# マッチ情報
		self._matches: Dict[str, MatchSession] = {}
		self._active_match_by_client: Dict[str, str] = {}

		self.game_engines: Dict[str, Any] = {}
		self._game_session = GameSession(
			lock=self._lock,
			matches=self._matches,
			active_match_by_client=self._active_match_by_client,
			game_engines=self.game_engines,
			available_match_numbers=self._available_match_numbers,
			send_to_client=self._send_to_client,
			broadcast_match_members=self._broadcast_match_members,
		)

	async def start(self) -> None:
		"""サーバーを起動"""
		if self._server is not None:
			return

		self._websockets = websockets
		self._server = await websockets.serve(self._on_connect, self.host, self.port)
		print(f"[WebSocketGameServer] started ws://{self.host}:{self.port}")

	async def stop(self) -> None:
		"""サーバーを停止"""
		if self._server is None:
			return

		self._server.close()
		await self._server.wait_closed()
		self._server = None

		for ws in list(self._connections):
			await self._safe_close(ws)

		self._connections.clear()
		self._client_id_by_socket.clear()
		self._socket_by_client_id.clear()
		self._client_name.clear()
		self._waiting_queue.clear()
		self._matches.clear()
		self._active_match_by_client.clear()
		self.game_engines.clear()

		print("[WebSocketGameServer] stopped")

	async def wait_closed(self) -> None:
		"""サーバー終了待ち"""
		if self._server is not None:
			await self._server.wait_closed()

	async def _on_connect(self, websocket) -> None:
		"""クライアント接続時の処理"""
		client_id = await self._register_client(websocket)

		try:
			await self._send_json(
				websocket,
				{
					"type": "connected",
					"data": {
						"client_id": client_id,
					},
				}
			)

			print(f"[WebSocketGameServer] client connected: {client_id}")

			async for raw_message in websocket:
				await self._handle_message(websocket, raw_message)
		except Exception as exc:
			await self._send_json(websocket, {"type": "error", "message": str(exc)})
		finally:
			await self._unregister_client(websocket)

	async def _register_client(self, websocket) -> str:
		"""クライアントを登録して一意の client_id を割り当てる"""
		async with self._lock:
			if self._available_client_numbers:
				client_number = heapq.heappop(self._available_client_numbers)
			else:
				self._client_seq += 1
				client_number = self._client_seq

			client_id = f"C{client_number:04d}"
			self._connections.add(websocket)
			self._client_id_by_socket[websocket] = client_id
			self._socket_by_client_id[client_id] = websocket
			self._client_name[client_id] = client_id
			return client_id

	async def _unregister_client(self, websocket) -> None:
		"""クライアントの登録を解除し、関連するリソースをクリーンアップする"""
		notify_sockets: List[Any] = []
		cancel_payload: Optional[Dict[str, Any]] = None
		requeue_targets: List[str] = []

		print(f"[WebSocketGameServer] client disconnected: {self._client_id_by_socket.get(websocket)}")

		async with self._lock:
			client_id = self._client_id_by_socket.pop(websocket, None)
			if client_id is None:
				return

			if client_id.startswith("C") and client_id[1:].isdigit():
				heapq.heappush(self._available_client_numbers, int(client_id[1:]))

			self._connections.discard(websocket)
			self._socket_by_client_id.pop(client_id, None)
			self._client_name.pop(client_id, None)

			# 待機キューから除外
			if client_id in self._waiting_queue:
				self._waiting_queue.remove(client_id)

			# 対局中マッチから除外
			match_id = self._active_match_by_client.pop(client_id, None)
			if match_id and match_id in self._matches:
				match = self._matches.pop(match_id)
				if match_id.startswith("M") and match_id[1:].isdigit():
					heapq.heappush(self._available_match_numbers, int(match_id[1:]))
				# ゲームエンジンもクリーンアップ
				self.game_engines.pop(match_id, None)
				other_players = [cid for cid in match.players if cid != client_id]

				for other_id in other_players:
					self._active_match_by_client.pop(other_id, None)
					if other_id in self._socket_by_client_id and other_id not in self._waiting_queue:
						requeue_targets.append(other_id)

				cancel_payload = {
					"type": "match_cancelled",
					"data": {
						"match_id": match_id,
						"reason": "player_disconnected",
					},
				}
				notify_sockets = [self._socket_by_client_id.get(cid) for cid in other_players]

			# 残ったプレイヤーは再度マッチング待機へ戻す
			for target_id in requeue_targets:
				self._waiting_queue.append(target_id)

		if cancel_payload:
			await asyncio.gather(
				*(self._send_json(ws, cancel_payload) for ws in notify_sockets if ws is not None),
				return_exceptions=True,
			)

		#await self._broadcast_matchmaking_state()
		await self._try_make_match()

	async def _handle_message(self, websocket, raw_message: str) -> None:
		"""クライアントからのメッセージを処理する"""
		try:
			data = json.loads(raw_message)
			print(f"[WebSocketGameServer] received message: {data} from {self._client_id_by_socket.get(websocket)}")
		except json.JSONDecodeError:
			await self._send_json(websocket, {"type": "error", "message": "Invalid JSON"})
			return

		msg_type = data.get("type")
		if not msg_type:
			await self._send_json(websocket, {"type": "error", "message": "type is required"})
			return

		if msg_type == "ping":
			await self._send_json(websocket, {
				"type": "ping",
				"data": {
					"ts": time.time()
				},
			})
			return

#		if msg_type == "status":
#			client_id = self._client_id_by_socket.get(websocket)
#			if client_id:
#				await self._send_matchmaking_status(websocket, client_id)
#			return

		if msg_type == "join":
			client_id = self._client_id_by_socket.get(websocket)
			if client_id:
				await self._enqueue_for_matchmaking(client_id)
			return

		if msg_type == "action":
			client_id = self._client_id_by_socket.get(websocket)
			if client_id:
				await self._game_session.handle_game_action(client_id, data.get("data", {}))
			return

		await self._send_json(websocket, {"type": "error", "message": f"Unknown type: {msg_type}"})

	async def _enqueue_for_matchmaking(self, client_id: str) -> None:
		"""プレイヤーを待機キューへ追加し、揃えば自動でマッチさせる"""
		async with self._lock:
			if client_id in self._active_match_by_client:
				await self._send_to_client(client_id, {"type": "error", "message": "Already in a match"})
				return
			if client_id in self._waiting_queue:
				await self._send_to_client(client_id, {"type": "error", "message": "Already in matchmaking queue"})
				return
			if client_id not in self._socket_by_client_id:
				await self._send_to_client(client_id, {"type": "error", "message": "Client not connected"})
				return

			self._waiting_queue.append(client_id)

		await self._send_waiting_message(client_id)
		#await self._broadcast_matchmaking_state()
		await self._try_make_match()

	async def _try_make_match(self) -> None:
		"""待機キュー先頭から2人を取り出してマッチ開始"""
		match: Optional[MatchSession] = None
		player_ids: List[str] = []

		async with self._lock:
			if len(self._waiting_queue) < self.max_players:
				return

			player_ids = self._waiting_queue[: self.max_players]
			del self._waiting_queue[: self.max_players]

			if self._available_match_numbers:
				match_number = heapq.heappop(self._available_match_numbers)
			else:
				self._match_seq += 1
				match_number = self._match_seq

			match_id = f"M{match_number:04d}"
			match = MatchSession(match_id=match_id, players=list(player_ids), status="in_game")
			self._matches[match_id] = match

			for cid in player_ids:
				self._active_match_by_client[cid] = match_id

		if match is None:
			return

		await self._game_session.start_match(match)
		#await self._broadcast_matchmaking_state()

	async def _send_waiting_message(self, client_id: str) -> None:
		"""マッチング待機中のクライアントに現在の待機状況を送信"""
		queue_pos = 0
		queue_size = 0
		async with self._lock:
			if client_id in self._waiting_queue:
				queue_pos = self._waiting_queue.index(client_id) + 1
			queue_size = len(self._waiting_queue)

		await self._send_to_client(
			client_id,
			{
				"type": "matching_waiting",
				#"data": {
				#	"queue_position": queue_pos,
				#	"queue_size": queue_size,
				#	"need_players": max(self.max_players - queue_size, 0),
				#},
			},
		)

	async def _send_matchmaking_status(self, websocket, client_id: str) -> None:
		"""クライアントにマッチメイキングの状態を送信"""
		queue_pos = None
		queue_size = 0
		match_payload = None

		async with self._lock:
			queue_size = len(self._waiting_queue)
			if client_id in self._waiting_queue:
				queue_pos = self._waiting_queue.index(client_id) + 1

			match_id = self._active_match_by_client.get(client_id)
			if match_id and match_id in self._matches:
				match_payload = self._matches[match_id].to_dict()

		await self._send_json(
			websocket,
			{
				"type": "matchmaking_status",
				"data": {
					"in_queue": queue_pos is not None,
					"queue_position": queue_pos,
					"queue_size": queue_size,
					"in_game": match_payload is not None,
					"match": match_payload,
				},
			},
		)

	async def _broadcast_matchmaking_state(self) -> None:
		"""マッチメイキングの状態を全クライアントに送信"""
		async with self._lock:
			payload = {
				"type": "matchmaking_state",
				"data": {
					"queue_size": len(self._waiting_queue),
					"active_matches": len(self._matches),
				},
			}
		await self._broadcast(payload)

	async def _send_to_client(self, client_id: str, payload: Dict[str, Any]) -> None:
		websocket = self._socket_by_client_id.get(client_id)
		await self._send_json(websocket, payload)

	async def _broadcast_match_members(self, match_id: str, payload: Dict[str, Any]) -> None:
		"""マッチのメンバーにペイロードを送信"""
		async with self._lock:
			match = self._matches.get(match_id)
			if match is None:
				return
			player_ids = list(match.players)

		await asyncio.gather(
			*(self._send_to_client(cid, payload) for cid in player_ids),
			return_exceptions=True,
		)

	async def _broadcast(self, payload: Dict[str, Any]) -> None:
		await asyncio.gather(
			*(self._send_json(ws, payload) for ws in list(self._connections)),
			return_exceptions=True,
		)

	async def _send_json(self, websocket, payload: Dict[str, Any]) -> None:
		if websocket is None:
			return
		try:
			await websocket.send(json.dumps(payload, ensure_ascii=False))
		except Exception:
			pass

	async def _safe_close(self, websocket) -> None:
		try:
			await websocket.close()
		except Exception:
			pass


if __name__ == "__main__":
	async def main():
		server = WebSocketGameServer()
		await server.start()
		await server.wait_closed()

	asyncio.run(main())