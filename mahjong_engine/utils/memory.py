"""
プロセスのメモリ管理ユーティリティ

長時間稼働するサーバー向けに、glibc malloc の設定・返却と RSS の取得を行う。
Linux (glibc) 以外の環境では何もしない。
"""
import ctypes
import ctypes.util
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# glibc の mallopt パラメータ番号
_M_ARENA_MAX = -8

_libc = None
if sys.platform.startswith("linux"):
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        # musl など malloc_trim を持たない libc では使わない
        _libc.malloc_trim
        _libc.mallopt
    except (OSError, AttributeError):
        _libc = None


def configure_malloc(arena_max: int = 2) -> None:
    """
    malloc のアリーナ数を制限する（環境変数 MALLOC_ARENA_MAX と同等）。
    アリーナが増えるほど解放済みメモリが OS に返りにくくなるため、起動直後に呼ぶ。
    """
    if _libc is None:
        return
    if _libc.mallopt(_M_ARENA_MAX, arena_max) != 1:
        logger.warning("mallopt(M_ARENA_MAX, %s) failed", arena_max)


def trim_malloc() -> None:
    """解放済みで未使用のヒープを OS に返す。"""
    if _libc is None:
        return
    _libc.malloc_trim(0)


def current_rss_bytes() -> Optional[int]:
    """現在の RSS (常駐メモリ量) をバイトで返す。取得できない環境では None。"""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None
