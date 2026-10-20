"""
記録の窓口（GameSession / WebSocketGameServer から使う）

- register … 新しいプレイヤーの ID と秘密の文字列を発行する（WebGL 版は起動するたびに。メモリだけ）
- identify … 発行済みの ID と秘密の文字列で名乗る（起動中に接続が切れて、つなぎ直したとき）
- set_name … 名乗ったプレイヤーの名前を登録・変更する（初めての登録で、プレイヤーをデータベースに書く）
- submit   … 対局の記録を書き込み待ちの列に入れる。書き込みは別のタスクで行い、ゲームの進行を止めない。
             失敗したら間をあけてやり直す（Neon が止まっていて起きるまでの数秒など）
- ranking  … ランキングの上位と、自分の順位を返す（上位は少しの間ためておき、データベースへの問い合わせを減らす）

データベースにつながらないときも、ゲームは止めない（記録が残らないだけ）。
"""
import asyncio
import hashlib
import logging
import os
import secrets
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Optional, Union

from .match_recorder import MatchRecord, PlayerIdentity
from .store import RANKING_COLUMNS, VERIFY_MISMATCH, VERIFY_UNKNOWN, MemoryStore, PostgresStore

logger = logging.getLogger(__name__)

# クライアントの PlayerProfile.MaxNameLength（契約書の下線に収まる長さ）と同じにする
NAME_MAX_LENGTH = 10
SECRET_MIN_LENGTH = 16
SECRET_MAX_LENGTH = 128
# 発行する秘密の文字列の乱数のバイト数（token_urlsafe で 43 文字）
SECRET_BYTES = 32
# 発行しただけで名前のない ID を覚えておく時間と数（起動中の再接続のため。名前を登録したらデータベースに移る）
ISSUED_TTL_SEC = 24 * 60 * 60
ISSUED_MAX = 20000
RANKING_MAX_LIMIT = 50
RANKING_CACHE_SEC = 30.0
# 書き込み待ちの列の上限。あふれたら古いものから捨てる（データベースが長く止まっていても、メモリを使い切らない）
QUEUE_MAX = 500
# 書き込みのやり直し: 待ち時間（秒）を倍にしていき、この回数で諦める（合計でおよそ 10 分）
RETRY_DELAYS = (2, 4, 8, 15, 30, 60, 60, 60, 60, 60, 60, 60)
CONNECT_RETRY_SEC = 30.0
IDENTIFY_TIMEOUT_SEC = 15.0


@dataclass(frozen=True)
class IdentifyResult:
    identity: Optional[PlayerIdentity]
    # invalid_player_id / invalid_secret / unknown_player / secret_mismatch / invalid_name / unavailable
    error: Optional[str] = None
    persistent: bool = False  # データベースで確かめられたか（False なら、この接続の間だけの名乗り）


@dataclass(frozen=True)
class RegisterResult:
    identity: PlayerIdentity
    secret: str  # クライアントに渡して端末に保存してもらう（サーバーにはハッシュだけを残す）
    persistent: bool = False


def normalize_name(name: object) -> Optional[str]:
    """表示名を整える（前後の空白を除き、制御文字を消す）。空・長すぎるときは None。"""
    if not isinstance(name, str):
        return None
    cleaned = "".join(ch for ch in unicodedata.normalize("NFC", name) if unicodedata.category(ch)[0] != "C").strip()
    if not cleaned or len(cleaned) > NAME_MAX_LENGTH:
        return None
    return cleaned


def hash_secret(secret: str) -> str:
    # 秘密の文字列はサーバーが発行する十分に長い乱数なので、ソルトなしの SHA-256 で足りる
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class RecordService:
    def __init__(self, store: Union[MemoryStore, PostgresStore]):
        self.store = store
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._connected = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._ranking_cache: dict[tuple[str, int], tuple[float, list]] = {}
        # 発行しただけで、まだ名前のない ID（player_id → (秘密の文字列のハッシュ, 発行した時刻)）
        self._issued: dict[str, tuple[str, float]] = {}

    @property
    def persistent(self) -> bool:
        return self.store.persistent and self._connected.is_set()

    # ========== 起動・停止 ==========

    def start(self) -> None:
        """データベースへの接続と書き込みのタスクを始める（つながるまで待たない）。イベントループの中で呼ぶ。"""
        if self._tasks:
            return
        self._tasks = [asyncio.create_task(self._connect_loop()), asyncio.create_task(self._writer_loop())]

    async def stop(self, flush_timeout: float = 10.0) -> None:
        """書き込み待ちをできるだけ書いてから止める（Render の再起動・デプロイのとき）。"""
        if self._connected.is_set() and not self._queue.empty():
            try:
                await asyncio.wait_for(self._queue.join(), flush_timeout)
            except asyncio.TimeoutError:
                logger.warning("書き込めなかった対局の記録があります: %d 件", self._queue.qsize())
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        await self.store.close()

    async def _connect_loop(self) -> None:
        while True:
            try:
                await self.store.connect()
                self._connected.set()
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("記録用のデータベースにつながりません。%.0f 秒後にやり直します", CONNECT_RETRY_SEC)
                await asyncio.sleep(CONNECT_RETRY_SEC)

    async def _writer_loop(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self._connected.wait()
                await self._save_with_retry(record)
            finally:
                self._queue.task_done()

    async def _save_with_retry(self, record: MatchRecord) -> None:
        for attempt, delay in enumerate((0,) + RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self.store.save_match(record)
                if record.ranked:
                    self._ranking_cache.clear()
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("対局の記録を書けませんでした（%d 回目）", attempt + 1)
        logger.error("対局の記録を諦めました: mode=%s rounds=%d", record.mode, record.rounds)

    # ========== ID の発行・名乗り・名前 ==========

    async def _db_call(self, coro):
        """データベースの操作（つながるまで待つ）。メモリの保存先ならそのまま。"""
        if not self.store.persistent:
            return await coro
        await asyncio.wait_for(self._connected.wait(), IDENTIFY_TIMEOUT_SEC)
        return await asyncio.wait_for(coro, IDENTIFY_TIMEOUT_SEC)

    def _prune_issued(self) -> None:
        now = time.monotonic()
        for player_id, (_hash, issued_at) in list(self._issued.items()):
            if now - issued_at > ISSUED_TTL_SEC:
                del self._issued[player_id]
        while len(self._issued) >= ISSUED_MAX:
            self._issued.pop(next(iter(self._issued)))  # いちばん古いもの

    async def register(self) -> RegisterResult:
        """
        新しいプレイヤーの ID と秘密の文字列を発行する（名前はまだない）。

        WebGL 版のクライアントは ID を端末に残さず、起動するたびに発行してもらう。開いただけ・名前を付けずに
        遊んだだけのプレイヤーをデータベースに溜めないよう、発行はメモリだけで行い、名前を登録したとき
        （set_name）に初めてデータベースに書く。発行した ID は、起動中の再接続（identify）のために
        ISSUED_TTL_SEC の間だけ覚えておく。
        """
        self._prune_issued()
        player_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(SECRET_BYTES)
        self._issued[player_id] = (hash_secret(secret), time.monotonic())
        return RegisterResult(PlayerIdentity(player_id, None), secret, persistent=self.store.persistent)

    async def identify(self, player_id: object, secret: object, name: object = None) -> IdentifyResult:
        """
        発行済みの ID と秘密の文字列で名乗る（起動中に接続が切れて、つなぎ直したとき）。名前は登録済みのものを返す。
        name を付けると、名乗りに続けて名前も登録・変更する（set_name と同じ）。
        """
        try:
            canonical_id = str(uuid.UUID(str(player_id)))
        except (ValueError, TypeError, AttributeError):
            return IdentifyResult(None, "invalid_player_id")
        if not isinstance(secret, str) or not SECRET_MIN_LENGTH <= len(secret) <= SECRET_MAX_LENGTH:
            return IdentifyResult(None, "invalid_secret")
        secret_hash = hash_secret(secret)

        issued = self._issued.get(canonical_id)
        if issued is not None:
            # 発行しただけで、まだ名前のない（データベースにいない）プレイヤー
            if issued[0] != secret_hash:
                return IdentifyResult(None, "secret_mismatch")
            identity = PlayerIdentity(canonical_id, None)
        else:
            try:
                status, stored_name = await self._db_call(self.store.verify_player(canonical_id, secret_hash))
            except asyncio.CancelledError:
                raise
            except Exception:
                # データベースが使えなくても遊べるようにする（この接続の対局はランキングに入らない）
                logger.exception("名乗りを確かめられませんでした: player_id=%s", canonical_id)
                return IdentifyResult(PlayerIdentity(canonical_id, normalize_name(name), verified=False),
                                      persistent=False)
            if status == VERIFY_UNKNOWN:
                return IdentifyResult(None, "unknown_player")
            if status == VERIFY_MISMATCH:
                return IdentifyResult(None, "secret_mismatch")
            identity = PlayerIdentity(canonical_id, stored_name)
        if name is not None:
            identity, error = await self.set_name(identity, name)
            if error is not None:
                return IdentifyResult(None, error)
        return IdentifyResult(identity, persistent=self.store.persistent)

    async def set_name(self, identity: PlayerIdentity, name: object) -> tuple[Optional[PlayerIdentity], Optional[str]]:
        """
        名前を登録・変更する。(新しい名乗り, 失敗の理由)。理由は invalid_name / unavailable。
        発行しただけのプレイヤーは、ここで初めてデータベースに書く。
        """
        clean_name = normalize_name(name)
        if clean_name is None:
            return None, "invalid_name"
        if not identity.verified:
            # データベースで確かめられていない名乗り: 表示にだけ使う（保存しない）
            return PlayerIdentity(identity.player_id, clean_name, verified=False), None
        issued = self._issued.get(identity.player_id)
        try:
            if issued is not None:
                await self._db_call(self.store.create_player(identity.player_id, issued[0], clean_name))
                ok = True
            else:
                ok = await self._db_call(self.store.set_name(identity.player_id, clean_name))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("名前を保存できませんでした: player_id=%s", identity.player_id)
            return None, "unavailable"
        if not ok:
            return None, "unavailable"
        if issued is not None:
            self._issued.pop(identity.player_id, None)  # データベースに移ったので、もう覚えておかない
        self._ranking_cache.clear()  # ランキングの名前も変わる
        return PlayerIdentity(identity.player_id, clean_name), None

    # ========== 対局の記録 ==========

    def submit(self, record: MatchRecord) -> None:
        """書き込み待ちの列に入れる（すぐ戻る）。"""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                logger.warning("書き込み待ちがあふれたため、いちばん古い対局の記録を捨てました")
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(record)

    # ========== ランキング ==========

    async def ranking(self, kind: object, limit: object = 10, player_id: Optional[str] = None) -> dict:
        if kind not in RANKING_COLUMNS:
            raise ValueError(f"Unknown ranking kind: {kind!r}")
        limit = max(1, min(RANKING_MAX_LIMIT, int(limit) if isinstance(limit, int) and not isinstance(limit, bool) else 10))
        if self.store.persistent:
            await asyncio.wait_for(self._connected.wait(), IDENTIFY_TIMEOUT_SEC)

        key = (kind, limit)
        cached = self._ranking_cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < RANKING_CACHE_SEC:
            top = cached[1]
        else:
            top = await self.store.top(kind, limit)
            self._ranking_cache[key] = (time.monotonic(), top)

        entries, previous, rank = [], None, 0
        for index, (name, value) in enumerate(top, start=1):
            if value != previous:
                rank, previous = index, value  # 同じ値は同じ順位
            entries.append({"rank": rank, "name": name, "value": value})
        me = None
        if player_id is not None:
            mine = await self.store.rank_of(kind, player_id)
            if mine is not None:
                me = {"rank": mine[0], "value": mine[1]}
        return {"kind": kind, "entries": entries, "me": me}

    async def my_records(self, player_id: str) -> dict:
        """
        自分の記録（ランキングの 3 つの値と順位・対局数・勝利数）。まだ記録がなければ値は 0・順位は None。
        """
        if self.store.persistent:
            await asyncio.wait_for(self._connected.wait(), IDENTIFY_TIMEOUT_SEC)
        stats = await self.store.stats_of(player_id) or {}
        result = {"matches": stats.get("matches", 0), "wins": stats.get("wins", 0)}
        for kind in RANKING_COLUMNS:
            mine = await self.store.rank_of(kind, player_id) if stats else None
            result[kind] = {"value": mine[1] if mine else 0, "rank": mine[0] if mine else None}
        return result

    def queue_size(self) -> int:
        return self._queue.qsize()


def create_record_service() -> RecordService:
    """DATABASE_URL があれば Postgres、なければメモリに記録する。"""
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return RecordService(PostgresStore(dsn))
    logger.info("DATABASE_URL がないので、記録はメモリに置きます（サーバーを止めると消えます）")
    return RecordService(MemoryStore())
