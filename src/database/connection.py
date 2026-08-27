import os
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

_session_factories = {}


def get_engine(db_url: str = None) -> Engine:
    """Create SQLite engine with optional encryption support.

    Args:
        db_url: Database URL. Defaults to sqlite:///call_center.db

    Returns:
        SQLAlchemy Engine instance
    """
    if db_url is None:
        db_url = f"sqlite:///{os.getenv("TRANSCRIPTION_CACHE_DB", "data/agent.db")}"

    engine = create_engine(db_url, echo=False)

    encryption_key = os.getenv("DB_ENCRYPTION_KEY", "Encryption_Key")
    if encryption_key:
        @event.listens_for(engine, "connect")
        def set_encryption_key(dbapi_conn, _connection_record):
            dbapi_conn.execute(f"PRAGMA key='{encryption_key}'")

    return engine


def get_session(engine: Engine = None) -> Session:
    """Get or create a session for the given engine.

    Sessions are cached by engine id to avoid creating multiple factories
    for the same engine.

    Args:
        engine: SQLAlchemy Engine instance. If None, creates a default engine.

    Returns:
        SQLAlchemy Session instance
    """
    if engine is None:
        engine = get_engine()

    engine_id = id(engine)

    if engine_id not in _session_factories:
        _session_factories[engine_id] = sessionmaker(bind=engine)

    return _session_factories[engine_id]()


@contextmanager
def session_scope(engine: Engine = None):
    """Context manager for database sessions with automatic commit/rollback.

    Usage:
        with session_scope() as session:
            # Use session here
            session.add(obj)
        # Auto-commits on success, rolls back on exception

    Args:
        engine: SQLAlchemy Engine instance. If None, creates a default engine.

    Yields:
        SQLAlchemy Session instance
    """
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
