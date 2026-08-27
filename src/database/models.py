# ==========================================
# 1. Database Model (SQLite Table)
# ==========================================


from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AuditLogEntry(Base):
    __tablename__ = "audit_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(50), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    caller_id = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)  # Stored as a serialized JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AuditEntry(call_id='{self.call_id}', action='{self.action}')>"


class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(100), nullable=False, unique=True, index=True)
    status = Column(String(50), nullable=False)
    audio_filename = Column(String(255), nullable=True)
    transcript_text = Column(Text, nullable=True)
    summary_json = Column(JSON, nullable=True)
    qa_scores_json = Column(JSON, nullable=True)
    report_json = Column(JSON, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    trace_id = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<CallRecord(call_id='{self.call_id}', status='{self.status}')>"


class TranscriptionCache(Base):
    __tablename__ = "transcription_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audio_hash = Column(String(128), nullable=False, unique=True, index=True)
    transcription = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TranscriptionCache(audio_hash='{self.audio_hash}')>"

