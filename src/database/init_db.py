import os
import sqlite3
from src.utils.config import get_logger

logger = get_logger("init_db")

# Database path
DB_PATH = os.getenv("TRANSCRIPTION_CACHE_DB", "data/agent.db")


def init_transcription_cache_table():
    """Create transcription_cache table if it doesn't exist."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcription_cache (
                audio_hash TEXT PRIMARY KEY,
                caller_id TEXT,
                call_id TEXT,
                filename TEXT,
                transcription TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_caller_id
            ON transcription_cache(caller_id)
        """)

        conn.commit()
        conn.close()
        logger.info("transcription_cache table initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize transcription_cache table: {e}")
        raise


def init_audit_entries_table():
    """Create audit_entries table if it doesn't exist."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id VARCHAR(50) NOT NULL,
                action VARCHAR(100) NOT NULL,
                caller_id VARCHAR(100) NOT NULL,
                details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ix_audit_entries_call_id
            ON audit_entries (call_id)
        """)

        conn.commit()
        conn.close()
        logger.info("audit_entries table initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audit_entries table: {e}")
        raise


def init_db():
    """Initialize all database tables."""
    logger.info("Starting database initialization...")
    init_transcription_cache_table()
    logger.info("transcription_cache table initialized successfully")
    init_audit_entries_table()
    logger.info("audit_entries table initialized successfully")
    logger.info("Database initialization complete")


if __name__ == "__main__":
    init_db()
