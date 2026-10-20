"""
対局の記録とランキング（データベース）

- match_recorder … 1 対局の記録を組み立てる
- store          … 保存先（Postgres / メモリ）
- service        … 名乗り・書き込み・ランキングの窓口
"""
from .match_recorder import MatchRecord, MatchRecorder, PlayerIdentity, Seat
from .service import IdentifyResult, RecordService, RegisterResult, create_record_service

__all__ = [
    "IdentifyResult", "MatchRecord", "MatchRecorder", "PlayerIdentity", "RecordService", "RegisterResult", "Seat",
    "create_record_service",
]
