# killing-mahjong-battle backend (Flask + Gunicorn)

本番向けの WebSocket サーバーは Flask + Gunicorn で起動できます。

## 1. 依存インストール

```bash
pip install -r requirements.txt
```

## 2. 起動

```bash
gunicorn -w 1 -k gthread --threads 8 -b 0.0.0.0:8000 wsgi:app
```

- WebSocket endpoint: `ws://<host>:8000/ws`
- Health check: `http://<host>:8000/healthz`

## 3. 運用メモ

- このゲームは 2 人マッチングのため、プロセス間で状態共有しない構成では `-w 1` を推奨します。
- 複数ワーカーにしたい場合は、接続のスティッキー化か外部状態ストアによる共有が必要です。
