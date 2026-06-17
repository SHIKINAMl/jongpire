# killing-mahjong-battle backend

本番向けの WebSocket サーバー

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