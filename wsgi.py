"""
Render 向けエントリーポイント。

starlette + uvicorn で起動する。
HTTP GET/HEAD (ヘルスチェック) と WebSocket (/ws) を同じポートで扱う。
"""
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from mahjong_engine.communication.websocket_server import WebSocketGameServer

_game_server = WebSocketGameServer(host="0.0.0.0", port=0)


class _WSAdapter:
    """
    starlette WebSocket を既存の WebSocketGameServer が期待する
    websockets ライブラリのインターフェースに変換する。
    """

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def send(self, text: str) -> None:
        await self._ws.send_text(text)

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            data = await self._ws.receive_text()
            return data
        except (WebSocketDisconnect, Exception):
            raise StopAsyncIteration

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


async def health(request):
    return JSONResponse({"status": "ok"})


async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    adapter = _WSAdapter(websocket)
    await _game_server._on_connect(adapter)


app = Starlette(
    routes=[
        Route("/", health),
        Route("/healthz", health),
        WebSocketRoute("/ws", ws_endpoint),
    ]
)
