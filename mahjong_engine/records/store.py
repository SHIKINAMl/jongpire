"""
記録の保存先

- PostgresStore … 本番（Neon の Postgres）。環境変数 DATABASE_URL に接続文字列を入れると使う
- MemoryStore   … DATABASE_URL がないとき（手元の開発・テスト）。サーバーを止めると消える

どちらも同じメソッドを持ち、RecordService から使う。
テーブルはサーバーの起動時に CREATE TABLE IF NOT EXISTS で作る（schema.sql と同じ内容）。
"""
import json
import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .match_recorder import MatchRecord

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# ランキングの種類 → player_stats の列
RANKING_COLUMNS = {
    "max_damage": "max_damage",
    "match_points": "max_match_points",
    "rounds": "max_rounds",
}


# 名乗りの確かめの結果
VERIFY_OK = "ok"
VERIFY_UNKNOWN = "unknown"  # その ID は登録されていない
VERIFY_MISMATCH = "mismatch"  # 秘密の文字列が違う


@dataclass
class _MemoryPlayer:
    name: Optional[str]
    secret_hash: str
    stats: dict = field(default_factory=lambda: {
        "max_damage": 0, "max_match_points": 0, "max_rounds": 0, "matches": 0, "wins": 0,
    })


class MemoryStore:
    """メモリに置く保存先（DATABASE_URL がないとき）。"""

    persistent = False

    def __init__(self) -> None:
        self.players: dict[str, _MemoryPlayer] = {}
        self.matches: list[MatchRecord] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def create_player(self, player_id: str, secret_hash: str, name: Optional[str] = None) -> None:
        self.players[player_id] = _MemoryPlayer(name, secret_hash)

    async def verify_player(self, player_id: str, secret_hash: str) -> tuple[str, Optional[str]]:
        """(結果, 名前)。結果は VERIFY_OK / VERIFY_UNKNOWN / VERIFY_MISMATCH。"""
        player = self.players.get(player_id)
        if player is None:
            return VERIFY_UNKNOWN, None
        if player.secret_hash != secret_hash:
            return VERIFY_MISMATCH, None
        return VERIFY_OK, player.name

    async def set_name(self, player_id: str, name: str) -> bool:
        player = self.players.get(player_id)
        if player is None:
            return False
        player.name = name
        return True

    async def save_match(self, record: MatchRecord) -> None:
        self.matches.append(record)
        if not record.ranked:
            return
        for result in record.players:
            player = self.players.get(result.player_id) if result.player_id else None
            if player is None:
                continue
            stats = player.stats
            stats["max_damage"] = max(stats["max_damage"], result.max_damage)
            stats["max_match_points"] = max(stats["max_match_points"], result.match_points)
            stats["max_rounds"] = max(stats["max_rounds"], record.rounds)
            stats["matches"] += 1
            stats["wins"] += int(result.won)

    async def stats_of(self, player_id: str) -> Optional[dict]:
        player = self.players.get(player_id)
        if player is None or player.stats["matches"] == 0:
            return None
        return dict(player.stats)

    def _sorted(self, kind: str) -> list[tuple[str, str, int]]:
        column = RANKING_COLUMNS[kind]
        # 名前を登録していないプレイヤーはランキングに出さない
        rows = [(pid, p.name, p.stats[column]) for pid, p in self.players.items()
                if p.stats["matches"] > 0 and p.name is not None]
        return sorted(rows, key=lambda r: (-r[2], r[0]))

    async def top(self, kind: str, limit: int) -> list[tuple[str, int]]:
        return [(name, value) for _pid, name, value in self._sorted(kind)[:limit]]

    async def rank_of(self, kind: str, player_id: str) -> Optional[tuple[int, int]]:
        rows = self._sorted(kind)
        mine = next((r for r in rows if r[0] == player_id), None)
        if mine is None:
            return None
        return 1 + sum(1 for r in rows if r[2] > mine[2]), mine[2]


# 接続文字列の問い合わせのうち、asyncpg が知らないもの。残すと Postgres の設定として送られ、接続が失敗する
# （Neon がコピーさせる接続文字列には channel_binding=require が付いている）
_UNSUPPORTED_DSN_PARAMS = ("channel_binding",)


def clean_dsn(dsn: str) -> str:
    """asyncpg で使えるように接続文字列を整える（知らない問い合わせを除く。sslmode などは残す）。"""
    parts = urlsplit(dsn.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in _UNSUPPORTED_DSN_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query)))


class PostgresStore:
    """Neon などの Postgres に置く保存先（asyncpg）。"""

    persistent = True

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        import asyncpg  # DATABASE_URL を使うときだけ要る

        # Neon は使わない間は止まり、最初の接続で起きる（数秒かかる）ので、待ち時間を長めにとる
        # statement_cache_size=0: Neon の接続プール（ホスト名に -pooler が付く接続文字列）でも使えるように、
        # 準備済みの文を使い回さない
        self._pool = await asyncpg.create_pool(
            clean_dsn(self._dsn), min_size=0, max_size=4, timeout=30, command_timeout=30, statement_cache_size=0,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        logger.info("記録用のデータベースに接続しました")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_player(self, player_id: str, secret_hash: str, name: Optional[str] = None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO players (player_id, name, secret_hash) VALUES ($1::uuid, $2, $3)",
                player_id, name, secret_hash,
            )

    async def verify_player(self, player_id: str, secret_hash: str) -> tuple[str, Optional[str]]:
        """(結果, 名前)。確かめられたら last_seen_at を更新する。"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT name, secret_hash FROM players WHERE player_id = $1::uuid", player_id)
            if row is None:
                return VERIFY_UNKNOWN, None
            if row["secret_hash"] != secret_hash:
                return VERIFY_MISMATCH, None
            await conn.execute("UPDATE players SET last_seen_at = now() WHERE player_id = $1::uuid", player_id)
        return VERIFY_OK, row["name"]

    async def set_name(self, player_id: str, name: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE players SET name = $2, last_seen_at = now() WHERE player_id = $1::uuid RETURNING player_id",
                player_id, name,
            )
        return row is not None

    async def save_match(self, record: MatchRecord) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                match_id = await conn.fetchval(
                    """
                    INSERT INTO matches (mode, started_at, ended_at, rounds, end_reason, finished, ranked)
                    VALUES ($1, to_timestamp($2), to_timestamp($3), $4, $5, $6, $7) RETURNING id
                    """,
                    record.mode, record.started_at, record.ended_at, record.rounds, record.end_reason,
                    record.finished, record.ranked,
                )
                await conn.executemany(
                    """
                    INSERT INTO match_players
                        (match_id, seat, client_id, player_id, name, cpu_difficulty, final_health, match_points,
                         max_damage, won)
                    VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8, $9, $10)
                    """,
                    [(match_id, r.seat, r.client_id, r.player_id, r.name, r.cpu_difficulty, r.final_health,
                      r.match_points, r.max_damage, r.won) for r in record.players],
                )
                if record.round_logs:
                    await conn.executemany(
                        "INSERT INTO match_rounds (match_id, round_index, data) VALUES ($1, $2, $3::jsonb)",
                        [(match_id, i, json.dumps(log, ensure_ascii=False)) for i, log in enumerate(record.round_logs)],
                    )
                if not record.ranked:
                    return
                for r in record.players:
                    if r.player_id is None:
                        continue
                    # 大きい方を残す（最大打点・1 対局の累計獲得点数・局数）。対局数・勝利数は足す
                    await conn.execute(
                        """
                        INSERT INTO player_stats (player_id, max_damage, max_match_points, max_rounds, matches, wins)
                        VALUES ($1::uuid, $2, $3, $4, 1, $5)
                        ON CONFLICT (player_id) DO UPDATE SET
                            max_damage = GREATEST(player_stats.max_damage, EXCLUDED.max_damage),
                            max_match_points = GREATEST(player_stats.max_match_points, EXCLUDED.max_match_points),
                            max_rounds = GREATEST(player_stats.max_rounds, EXCLUDED.max_rounds),
                            matches = player_stats.matches + 1,
                            wins = player_stats.wins + EXCLUDED.wins,
                            updated_at = now()
                        """,
                        r.player_id, r.max_damage, r.match_points, record.rounds, int(r.won),
                    )

    async def stats_of(self, player_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT max_damage, max_match_points, max_rounds, matches, wins
                FROM player_stats WHERE player_id = $1::uuid
                """,
                player_id,
            )
        return dict(row) if row else None

    async def top(self, kind: str, limit: int) -> list[tuple[str, int]]:
        column = RANKING_COLUMNS[kind]  # 列名は決まった値だけ（利用者の入力は入らない）
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT p.name, s.{column} AS value FROM player_stats s JOIN players p USING (player_id)
                WHERE p.name IS NOT NULL
                ORDER BY s.{column} DESC, s.player_id LIMIT $1
                """,
                limit,
            )
        return [(row["name"], int(row["value"])) for row in rows]

    async def rank_of(self, kind: str, player_id: str) -> Optional[tuple[int, int]]:
        column = RANKING_COLUMNS[kind]
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT s.{column} AS value,
                       (SELECT count(*) FROM player_stats o JOIN players op USING (player_id)
                        WHERE op.name IS NOT NULL AND o.{column} > s.{column}) + 1 AS rank
                FROM player_stats s JOIN players p USING (player_id)
                WHERE s.player_id = $1::uuid AND p.name IS NOT NULL
                """,
                player_id,
            )
        return (int(row["rank"]), int(row["value"])) if row else None
