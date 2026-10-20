-- 記録用のテーブル（Postgres）。サーバーの起動時に毎回流す（あれば何もしない）。
-- 列を足すときは ALTER TABLE ... ADD COLUMN IF NOT EXISTS を下に書き足す（既存のデータを消さない）。

-- プレイヤー。player_id と秘密の文字列はサーバーが発行する（register。WebGL 版は起動するたびに新しく発行し、端末に残さない）。
-- 発行しただけではここに書かず、名前を登録したとき（set_name）に書く（開いただけの人を溜めない）
CREATE TABLE IF NOT EXISTS players (
    player_id    uuid PRIMARY KEY,
    name         text,
    secret_hash  text NOT NULL,  -- 秘密の文字列の SHA-256（他人が同じ ID をかたって記録を書き換えないため）
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now()
);

-- ランキング用の値（対人戦を決着まで打った対局だけで数える）
CREATE TABLE IF NOT EXISTS player_stats (
    player_id        uuid PRIMARY KEY REFERENCES players (player_id),
    max_damage       bigint NOT NULL DEFAULT 0,  -- 最大打点: 和了 1 回で相手に与えた損失の最大
    max_match_points bigint NOT NULL DEFAULT 0,  -- 1 対局の累計獲得点数の最大
    max_rounds       integer NOT NULL DEFAULT 0, -- 1 対局で打った局数の最大
    matches          integer NOT NULL DEFAULT 0,
    wins             integer NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS player_stats_max_damage ON player_stats (max_damage DESC);
CREATE INDEX IF NOT EXISTS player_stats_max_match_points ON player_stats (max_match_points DESC);
CREATE INDEX IF NOT EXISTS player_stats_max_rounds ON player_stats (max_rounds DESC);

-- 対局（CPU 戦・途中で終わった対局も残す。学習・CPU 調整用）
CREATE TABLE IF NOT EXISTS matches (
    id         bigserial PRIMARY KEY,
    mode       text NOT NULL,          -- public / private / cpu
    started_at timestamptz NOT NULL,
    ended_at   timestamptz NOT NULL,
    rounds     integer NOT NULL,       -- 最後まで打った局数
    end_reason text NOT NULL,          -- 決着の理由。途中で終わったら cancelled:理由
    finished   boolean NOT NULL,
    ranked     boolean NOT NULL        -- ランキングに入れたか
);
CREATE INDEX IF NOT EXISTS matches_ended_at ON matches (ended_at);

-- 対局の席ごとの結果
CREATE TABLE IF NOT EXISTS match_players (
    match_id       bigint NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    seat           integer NOT NULL,
    client_id      text NOT NULL,      -- 対局中の ID（match_rounds.data の players[].client_id と同じ）
    player_id      uuid,               -- 名乗っていない人間・CPU は NULL
    name           text,
    cpu_difficulty text,               -- CPU の席だけ
    final_health   integer NOT NULL,
    match_points   integer NOT NULL,
    max_damage     integer NOT NULL,
    won            boolean NOT NULL,
    PRIMARY KEY (match_id, seat)
);
CREATE INDEX IF NOT EXISTS match_players_player_id ON match_players (player_id);

-- 局ごとの記録（山・手牌・待ち・捨て牌・掛け金・精算結果。JSON）
-- 2026-10-20: 名前の登録を ID の発行と分けたので、名前のないプレイヤーを許す（前の版で作ったテーブル向け）
ALTER TABLE players ALTER COLUMN name DROP NOT NULL;

CREATE TABLE IF NOT EXISTS match_rounds (
    match_id    bigint NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    round_index integer NOT NULL,
    data        jsonb NOT NULL,
    PRIMARY KEY (match_id, round_index)
);
