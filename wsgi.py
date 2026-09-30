"""
Render 向けエントリーポイント。

starlette + uvicorn で起動する。
HTTP GET/HEAD (ヘルスチェック) と WebSocket (/ws) を同じポートで扱う。
"""
import os
from contextlib import asynccontextmanager

from mahjong_engine.utils.memory import configure_malloc

# 長時間稼働でメモリが OS に返らず膨らむのを抑える（アリーナ生成前に設定する）
configure_malloc()

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from mahjong_engine.communication.websocket_server import WebSocketGameServer

_game_server = WebSocketGameServer(host="0.0.0.0", port=0)
_required_token = os.getenv("TOKEN", "")


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


def _is_authorized(conn) -> bool:
    """TOKEN 未設定なら常に許可。設定時は Bearer / X-Token / ?token= のいずれかで照合する。"""
    if not _required_token:
        return True

    auth_header = conn.headers.get("authorization", "")
    token_header = conn.headers.get("x-token", "")
    query_token = conn.query_params.get("token", "")

    supplied_token = ""
    if auth_header.lower().startswith("bearer "):
        supplied_token = auth_header[7:].strip()
    elif token_header:
        supplied_token = token_header.strip()
    elif query_token:
        supplied_token = query_token.strip()

    return supplied_token == _required_token


async def health(_request):
    return JSONResponse({"status": "ok"})


async def stats(request):
    """メモリリーク調査用に、サーバー内部の保持件数と RSS を返す。"""
    if not _is_authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse(_game_server.get_stats())


async def ws_endpoint(websocket: WebSocket):
    if not _is_authorized(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    adapter = _WSAdapter(websocket)
    await _game_server._on_connect(adapter)


@asynccontextmanager
async def lifespan(_app):
    _game_server.start_housekeeping()
    try:
        yield
    finally:
        await _game_server.stop_housekeeping()


app = Starlette(
    routes=[
        Route("/", health),
        Route("/healthz", health),
        Route("/stats", stats),
        WebSocketRoute("/ws", ws_endpoint),
    ],
    lifespan=lifespan,
)
