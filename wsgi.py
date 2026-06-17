"""
Render 向けエントリーポイント。

Flask/Gunicorn なし。websockets ライブラリで asyncio サーバーを直接起動し、
/ と /healthz で HTTP ヘルスチェックにも応答する。
"""
import asyncio
import http
import json
import os
import signal

import websockets

from mahjong_engine.communication.websocket_server import WebSocketGameServer

PORT = int(os.getenv("PORT", "8000"))
_game_server = WebSocketGameServer(host="0.0.0.0", port=PORT)


async def _process_request(path, request_headers):
    """/ と /healthz への HTTP リクエストに 200 を返す。それ以外は WS アップグレードへ。"""
    if path in ("/", "/healthz"):
        body = json.dumps({"status": "ok"}).encode("utf-8")
        return http.HTTPStatus.OK, {}, body
    return None  # WebSocket アップグレードを続行


async def main() -> None:
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set_result, None)
        except (NotImplementedError, RuntimeError):
            pass

    async with websockets.serve(
        _game_server._on_connect,
        "0.0.0.0",
        PORT,
        process_request=_process_request,
    ) as server:
        print(f"[server] listening ws://0.0.0.0:{PORT}/ws")
        await stop

    print("[server] stopped")


if __name__ == "__main__":
    asyncio.run(main())
