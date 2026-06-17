"""
Flask + Gunicorn 向け WebSocket サーバー。

既存の非同期ゲームロジック（WebSocketGameServer / GameSession）を
バックグラウンドの asyncio ループで動かし、Flask-Sock の同期 WebSocket と橋渡しする。
"""

import atexit
import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from typing import Any, Dict

from flask import Flask, jsonify
from flask_sock import Sock

from .websocket_server import WebSocketGameServer


logger = logging.getLogger(__name__)


class AsyncLoopThread:
    """バックグラウンドの asyncio ループを管理する。"""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro: Any) -> concurrent.futures.Future:
        if self._loop is None:
            raise RuntimeError("asyncio loop is not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


class FlaskWebSocketBridge:
    """Flask-Sock と既存ゲームサーバーを接続するブリッジ。"""

    def __init__(self, max_players: int = 2) -> None:
        self._loop_thread = AsyncLoopThread()
        port = int(os.getenv("PORT", "8000"))
        self._server = WebSocketGameServer(host="0.0.0.0", port=port, max_players=max_players)
        self._send_locks: Dict[int, threading.Lock] = {}
        self._send_locks_guard = threading.Lock()

    def _get_send_lock(self, websocket: Any) -> threading.Lock:
        key = id(websocket)
        with self._send_locks_guard:
            lock = self._send_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._send_locks[key] = lock
            return lock

    def _drop_send_lock(self, websocket: Any) -> None:
        key = id(websocket)
        with self._send_locks_guard:
            self._send_locks.pop(key, None)

    def _send_sync(self, websocket: Any, text: str) -> None:
        lock = self._get_send_lock(websocket)
        with lock:
            websocket.send(text)

    async def _send_json(self, websocket: Any, payload: Dict[str, Any]) -> None:
        if websocket is None:
            return

        text = json.dumps(payload, ensure_ascii=False)
        try:
            await asyncio.to_thread(self._send_sync, websocket, text)
        except Exception:
            pass

    async def _safe_close(self, websocket: Any) -> None:
        try:
            await asyncio.to_thread(websocket.close)
        except Exception:
            pass
        finally:
            self._drop_send_lock(websocket)

    def _submit(self, coro: Any) -> concurrent.futures.Future:
        return self._loop_thread.submit(coro)

    def handle_connection(self, websocket: Any) -> None:
        # 既存ロジックの送信・クローズ処理を Flask-Sock 用に差し替える。
        self._server._send_json = self._send_json
        self._server._safe_close = self._safe_close

        registered = False
        try:
            client_id = self._submit(self._server._register_client(websocket)).result()
            registered = True
            logger.info("websocket connected: %s", client_id)

            self._submit(
                self._server._send_json(
                    websocket,
                    {"type": "connected", "data": {"client_id": client_id}},
                )
            ).result()

            while True:
                raw_message = websocket.receive()
                if raw_message is None:
                    break
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8", errors="ignore")
                self._submit(self._server._handle_message(websocket, raw_message)).result()

        except Exception as exc:
            logger.exception("websocket handling failed")
            if registered:
                self._submit(self._server._send_json(websocket, {"type": "error", "message": str(exc)})).result()
        finally:
            if registered:
                logger.info("websocket disconnected: %s", self._server._client_id_by_socket.get(websocket))
                self._submit(self._server._unregister_client(websocket)).result()
            self._drop_send_lock(websocket)

    def shutdown(self) -> None:
        self._loop_thread.stop()


# Gunicorn から import されるプロセスごとに 1 つ作る。
_bridge = FlaskWebSocketBridge(max_players=2)


def create_app() -> Flask:
    app = Flask(__name__)
    sock = Sock(app)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok"})

    @sock.route("/ws")
    def websocket_endpoint(ws: Any) -> None:
        _bridge.handle_connection(ws)

    @sock.route("/")
    def websocket_endpoint_root(ws: Any) -> None:
        # クライアントがパス未指定で接続してきた場合の後方互換。
        _bridge.handle_connection(ws)

    return app


app = create_app()
atexit.register(_bridge.shutdown)
