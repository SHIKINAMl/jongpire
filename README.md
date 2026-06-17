# killing-mahjong-battle backend (Flask + Gunicorn)

本番向けの WebSocket サーバーは Flask + Gunicorn で起動できます。

## 1. 依存インストール

```bash
pip install -r requirements.txt
```

## 2. 起動

```bash
python wsgi.py
```

- WebSocket endpoint (Render): `wss://jongpire.onrender.com/ws`
- Health check: `https://jongpire.onrender.com/healthz`

## 3. 運用メモ

- このゲームは 2 人マッチングのため、プロセス間で状態共有しない構成では `-w 1` を推奨します。
- 複数ワーカーにしたい場合は、接続のスティッキー化か外部状態ストアによる共有が必要です。
