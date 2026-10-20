# 対局の記録とランキング（クライアント向けの手引き）

プレイヤー名・対局の記録・ランキングを、サーバーが Postgres（Neon）に残す。
**クライアントはデータベースに直接つながない。** 名前の登録もランキングの取得も、いつもの WebSocket のメッセージで行う
（データベースの接続文字列をクライアントやリポジトリに入れないこと。入れると誰でも記録を書き換えられる）。

サーバーのコード: このフォルダ（`mahjong_engine/records/`）。テーブルは `schema.sql`。

## 目次

1. [クライアントでやること（まとめ）](#1-クライアントでやることまとめ)
2. [起動時に ID を発行してもらう（register）](#2-起動時に-id-を発行してもらうregister)
3. [接続が切れたときに名乗り直す（identify）](#3-接続が切れたときに名乗り直すidentify)
4. [名前を登録・変更する（set_name）](#4-名前を登録変更するset_name)
5. [対局中の名前（game_started）](#5-対局中の名前game_started)
6. [ランキングを出す（ranking）](#6-ランキングを出すranking)
7. [自分の記録を出す（my_records）](#7-自分の記録を出すmy_records)
8. [ランキングの値の決まり](#8-ランキングの値の決まり)
9. [データベースから記録を取り出す（開発者向け）](#9-データベースから記録を取り出す開発者向け)
10. [サーバーの準備（Neon と Render）](#10-サーバーの準備neon-と-render)

## 1. クライアントでやること（まとめ）

ログインはない。**WebGL 版は端末に ID を残さず、起動するたび（開き直し・再読み込みのたび）にサーバーに ID を発行してもらう。**
ID と秘密の文字列は、起動している間だけメモリに持つ（接続が切れたときに同じプレイヤーへ戻るため）。

| いつ | 送る | 受け取る |
|---|---|---|
| 起動して最初の `connected` を受け取ったとき | `register` | `registered`（ID・秘密の文字列）→ メモリに持つ |
| 起動中に接続が切れて、つなぎ直したとき | `identify`（メモリの ID・秘密の文字列） | `identified` / `identify_failed` |
| ゲーム内の名前登録のとき（契約書で名前を書いた後） | `set_name`（名前） | `name_set` / `set_name_failed` |
| 名前を変えたとき | `set_name`（新しい名前） | `name_set` |
| ランキング画面を開いたとき | `ranking`（種類・件数） | `ranking` |
| 自分の記録を見るとき | `my_records` | `my_records` |
| 対局が始まったとき | — | `game_started` の `players[].name` に両者の名前 |

```
起動 ─ connected ─ register ─ registered（ID・秘密の文字列をメモリに持つ）
名前登録のフェイズ ・ 名前の変更 ─ set_name ─ name_set
（起動中に切断）─ connected ─ identify ─┬─ identified（同じプレイヤー・同じ名前に戻る）
                                        └─ identify_failed → register からやり直し、名前を set_name で送り直す
```

- `register` / `identify` は `join` より前に送る。名乗らなくても遊べる（記録はランキングに入らない）
- サーバーのデータベースが使えないときも遊べる（`persistent` が `false`）
- **名前を登録した後の対局だけがランキングに入る**（名前を付ける前の対局は、プレイヤーに結びつけない）

起動するたびに別のプレイヤーになるので、次のことに気をつける:

- ランキングに、同じ人が同じ名前で何度も並ぶことがある（起動ごとに別のプレイヤーとして記録される）
- `my_records` と `ranking` の `me` は、いま起動している間のプレイヤーの分だけ（前回の起動の記録は自分のものとして出ない）
- サーバーは、名前を登録したときに初めてプレイヤーをデータベースに書く（開いただけの人はデータベースに残らない）

## 2. 起動時に ID を発行してもらう（register）

起動して最初の `connected` を受け取ったら送る。1 つの接続で 1 回だけ発行する（2 回目は `register_failed {reason: "already_registered"}`）。

```json
{"type": "register"}
```

返事（`registered`）:

```json
{"type": "registered", "data": {"player_id": "UUID", "secret": "43 文字の秘密の文字列", "persistent": true}}
```

- `player_id` と `secret` は、起動している間だけメモリ（static 変数など）に持つ。端末には保存しない
- **`secret` は画面やログに出さない**（他人が同じ ID をかたって名前を書き換えないため）。サーバーもハッシュだけを持ち、ログでは伏せる
- この接続は、そのまま名乗った扱いになる（続けて `set_name` を送れる）
- `persistent` が `false` なら、データベースが使えない（遊ぶことはできるが、記録は残らない）

**cookie は使わない:** ゲームのページ（itch.io など）とサーバー（Render）は別のサイトなので、
ブラウザがサーバーの cookie を保存しない・送らないことが多い。

```csharp
namespace KillingMahjong.Core
{
    /// <summary>サーバーが発行した ID と秘密の文字列。起動している間だけ持つ（端末には保存しない）。</summary>
    public static class PlayerIdentity
    {
        public static string Id { get; private set; } = "";
        public static string Secret { get; private set; } = "";
        public static bool HasIdentity => !string.IsNullOrEmpty(Id) && !string.IsNullOrEmpty(Secret);

        /// <summary>registered を受け取ったら呼ぶ。</summary>
        public static void Set(string playerId, string secret) { Id = playerId; Secret = secret; }

        /// <summary>identify_failed を受け取ったら呼んで、register からやり直す。</summary>
        public static void Clear() { Id = ""; Secret = ""; }
    }
}
```

接続したときの分かれ道:

```csharp
// connected を受け取ったとき
if (PlayerIdentity.HasIdentity) SendIdentify();   // 起動中の再接続
else client.SendAsync("{\"type\": \"register\"}");  // 起動して最初の接続
```

## 3. 接続が切れたときに名乗り直す（identify）

起動中に接続が切れて、つなぎ直したときに送る（`join` より前）。同じプレイヤー・同じ名前に戻る。
起動して最初の接続では送らない（`register` を送る）。

```json
{"type": "identify", "data": {"player_id": "UUID", "secret": "メモリに持っている秘密の文字列"}}
```

成功（`identified`）:

```json
{"type": "identified", "data": {"player_id": "UUID", "name": "登録済みの名前（未登録なら null）", "persistent": true}}
```

- `data` に `"name"` を付けて送ると、名乗りに続けて名前も登録・変更する（`set_name` を別に送るのと同じ）

失敗（`identify_failed`）。どれも `PlayerIdentity.Clear()` して `register` からやり直し、名前があれば `set_name` で送り直す:

| reason | 意味 |
|---|---|
| `invalid_player_id` / `invalid_secret` | 持っている値の形が壊れている |
| `unknown_player` | その ID をサーバーが覚えていない（名前の前の ID を 24 時間たって忘れた・サーバーが再起動した など） |
| `secret_mismatch` | 秘密の文字列が違う |

`JsonUtility` で送る（文字列の連結で JSON を作らない）:

```csharp
[Serializable] public class IdentifyData { public string player_id; public string secret; }
[Serializable] public class IdentifyMessage { public string type = "identify"; public IdentifyData data; }

string json = JsonUtility.ToJson(new IdentifyMessage
{
    data = new IdentifyData { player_id = PlayerIdentity.Id, secret = PlayerIdentity.Secret },
});
```

受け取りは、ほかのメッセージと同じく `IServerMessageHandler` を足して `registered` / `register_failed` /
`identified` / `identify_failed` / `name_set` / `set_name_failed` を扱う。

## 4. 名前を登録・変更する（set_name）

`register` か `identify` で名乗った接続から送る。ゲーム内の名前登録のフェイズ（チュートリアルの契約書）と、名前の変更で使う。

```json
{"type": "set_name", "data": {"name": "表示名"}}
```

名前の決まり（サーバーでも同じように整える）:

- 1〜10 文字（`PlayerProfile.MaxNameLength` と同じ）
- 前後の空白（全角の空白も）は除く。制御文字（改行など）は消す
- **名前に `"` や `\` が入ることがあるので、JSON は `JsonUtility.ToJson` で作る**

成功（`name_set`）:

```json
{"type": "name_set", "data": {"player_id": "UUID", "name": "整えた後の名前"}}
```

- 表示はこの `name` を使う。ランキングの名前も新しい名前になる（記録は引き継ぐ）
- 対局中に送ってもよいが、相手の画面の名前が変わるのは次の対局から

失敗（`set_name_failed`）:

| reason | 意味 | クライアントでやること |
|---|---|---|
| `not_identified` | まだ `register` / `identify` をしていない | 先に名乗る |
| `invalid_name` | 名前が空・11 文字以上 | 名前を入れ直してもらう |
| `unavailable` | データベースに書けなかった | 少し待って送り直す（遊ぶことはできる） |

## 5. 対局中の名前（game_started）

`game_started` の `players` に `name` が入る。名乗っていない人・名前を登録していない人・CPU は `null`。

```json
{"type": "game_started", "data": {"match_id": "M0001", "mode": "public", "players": [
  {"client_id": "C0001", "health": 20000, "is_cpu": false, "name": "あかね"},
  {"client_id": "C0002", "health": 20000, "is_cpu": false, "name": null}
]}}
```

`JsonUtility` は `null` の文字列を `null`（または空）にする。空なら「名無し」や「CPU（つよい）」などを表示する。

## 6. ランキングを出す（ranking）

```json
{"type": "ranking", "data": {"kind": "max_damage", "limit": 10}}
```

| kind | 表示名の例 | 値 |
|---|---|---|
| `max_damage` | 最大打点 | 和了 1 回で相手に与えた損失の最大 |
| `match_points` | 累計獲得点数 | 1 対局の累計獲得点数の最大 |
| `rounds` | 最大局数 | 1 対局で打った局数の最大 |

- `limit` は 1〜50（省略すると 10）
- 種類ごとに 1 回ずつ送る（3 つのタブなら 3 回）

返事:

```json
{"type": "ranking", "data": {
  "kind": "max_damage",
  "entries": [
    {"rank": 1, "name": "あかね", "value": 12000},
    {"rank": 1, "name": "そら", "value": 12000},
    {"rank": 3, "name": "ひかり", "value": 9000}
  ],
  "me": {"rank": 15, "value": 3000}
}}
```

- 同じ値は同じ順位（上の例は 1 位が 2 人で、次は 3 位）
- `me` は自分の順位。名乗っていない・まだランキングに入る対局がないときは `null`。
  **`JsonUtility` は `null` のオブジェクトを `rank = 0` のオブジェクトにするので、`rank == 0` を「順位なし」として扱う**
- 上位の一覧はサーバーで 30 秒ためる。対局の直後は少し遅れて反映されることがある（ランキングに入る対局が終わると、ためた一覧は捨てる）
- 種類が違うと `{"type": "error", "message": "Unknown ranking kind: ..."}`

```csharp
[Serializable] public class RankingEntry { public int rank; public string name; public long value; }
[Serializable] public class RankingMe { public int rank; public long value; }
[Serializable] public class RankingData { public string kind; public RankingEntry[] entries; public RankingMe me; }
[Serializable] public class RankingMessage { public string type; public RankingData data; }

RankingMessage msg = JsonUtility.FromJson<RankingMessage>(jsonString);
bool ranked = msg.data.me != null && msg.data.me.rank > 0;
```

## 7. 自分の記録を出す（my_records）

`register` か `identify` で名乗った後に送る（名乗っていないと `{"type": "error", "message": "Identify first"}`）。

```json
{"type": "my_records"}
```

```json
{"type": "my_records", "data": {
  "matches": 12, "wins": 7,
  "max_damage": {"value": 12000, "rank": 1},
  "match_points": {"value": 31000, "rank": 4},
  "rounds": {"value": 18, "rank": null}
}}
```

- まだランキングに入る対局がなければ、値は 0・`rank` は `null`（`JsonUtility` では 0）
- `matches` / `wins` は、ランキングに入れた対局（対人戦を決着まで打ったもの）だけを数える。
  端末の `PlayerStatsManager`（CPU 戦も数える通算戦績）とは数が違う

## 8. ランキングの値の決まり

- **ランキングに入るのは、対人戦（公開・非公開）を決着まで打った対局だけ。**
  CPU 戦・途中で切断した対局・放置で破棄された対局は、記録（学習・CPU 調整用）には残すがランキングには入れない
- 名乗っていない人・名前を登録する前の対局はランキングに入らない（名前を登録した後の対局から入る）
- `max_damage`（最大打点）は**相手に与えた損失**（単騎の 2 倍・強襲の追加ダメージを含む）。
  端末の `PlayerStatsManager.BestScore`（自己ベスト打点）は**自分が得た点**（`winner_gain`）なので、同じ和了でも値が違うことがある。
  画面で並べるときは、どちらの値かが分かるように書く（または、どちらかに揃える。揃える場合はサーバーとクライアントのどちらを直すか相談）

## 9. データベースから記録を取り出す（開発者向け）

クライアントからは取り出さない。開発者が集計・学習に使うとき:

### Neon の SQL Editor で見る

Neon のダッシュボードの「SQL Editor」で、そのまま SQL を書ける。

```sql
-- 最大打点のランキング（上位 20）
SELECT p.name, s.max_damage, s.matches, s.wins
FROM player_stats s JOIN players p USING (player_id)
ORDER BY s.max_damage DESC LIMIT 20;

-- 最近の対局（CPU 戦も含む）
SELECT m.id, m.mode, m.ended_at, m.rounds, m.end_reason, m.ranked,
       string_agg(coalesce(mp.name, mp.cpu_difficulty, '名無し') || ':' || mp.final_health, ' / ' ORDER BY mp.seat) AS players
FROM matches m JOIN match_players mp ON mp.match_id = m.id
GROUP BY m.id ORDER BY m.ended_at DESC LIMIT 50;

-- CPU 戦の難易度ごとの勝率（人間側から見て）
SELECT cpu.cpu_difficulty, count(*) AS games, avg((human.won)::int) AS human_win_rate
FROM matches m
JOIN match_players cpu ON cpu.match_id = m.id AND cpu.cpu_difficulty IS NOT NULL
JOIN match_players human ON human.match_id = m.id AND human.cpu_difficulty IS NULL
WHERE m.finished GROUP BY cpu.cpu_difficulty;

-- 局ごとの記録（山・手牌・待ち・捨て牌・掛け金・精算結果）
SELECT match_id, round_index, data FROM match_rounds WHERE match_id = 123 ORDER BY round_index;
```

### 手元に書き出す（学習用）

接続文字列は Neon のダッシュボードの「Connection string」。**自分の端末の環境変数に入れ、ファイルやチャットに貼らない。**

```bash
psql "$DATABASE_URL" -c "\copy (SELECT match_id, round_index, data FROM match_rounds) TO 'match_rounds.csv' CSV HEADER"
```

Python なら `asyncpg` / `psycopg` で `SELECT ... FROM match_rounds` を読み、`data`（JSON）を学習の入力に変える。

### 局ごとの記録（match_rounds.data）の中身

```json
{
  "round_number": 3, "dora_id": 12, "is_draw": false,
  "liquidation": {"winner_id": "C0001", "loser_id": "C0002", "han": 7, "multiplier": 1.5, "loser_loss": 4500, "...": "..."},
  "players": [
    {"client_id": "C0001", "wall": [34 枚の牌ID], "hand": [13 枚], "waits": [待ち], "discards": [捨てた順],
     "bet": 1000, "health": 21500, "cumulative_earned_points": 4500}
  ]
}
```

`players[].client_id` は `match_players.client_id` と同じなので、`match_id` と `client_id` で結べば、その席の `player_id`・名前・CPU の難易度が分かる。

## 10. サーバーの準備（Neon と Render）

1. [Neon](https://neon.tech) でプロジェクトを作る（無料プラン）。リージョンは Render のサービスに近いところ
2. Neon の「Connection string」をコピーする（`postgresql://...?sslmode=require` の形）
3. Render のサービスの Environment に `DATABASE_URL` として貼る（コードやリポジトリには書かない）
4. デプロイすると、起動時にテーブルを作る（`mahjong_engine/records/schema.sql`。あれば何もしない）

確かめ方: `/stats?token=...` の `records_persistent` が `true`。ログに「記録用のデータベースに接続しました」が出る。

Neon は使わない間は止まり、最初の接続で起きる（数秒かかる）。その間の名乗り・書き込みは、起きるまで待ってから行う。
書き込みに失敗したら間をあけてやり直す（約 10 分まで）。それでも書けなければ、その対局の記録は諦める（ゲームは止めない）。
