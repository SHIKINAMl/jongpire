"""
Mahjong Engine - 17歩ゲーム用エンジン

モジュール構成：
- engine: ゲームエンジンとゲーム状態管理
- communication: WebSocket API とメッセージハンドリング
- ai: AI プレイヤーと学習
- utils: ユーティリティ（牌変換など）
- output: ログ出力とゲーム結果の管理
- examples: 使用例とデモ
"""

from .engine import GameEngine, GameState
from .communication import WebSocketGameServer
from .utils import TileConverter

__all__ = [
    'GameEngine',
    'GameState',
    'WebSocketGameServer',
    'TileConverter',
]