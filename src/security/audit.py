import json
import logging
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import AuditLogEntry
from src.database.models import Base
from tests import data
import os

# Setup basic internal logging to see what is happening under the hood
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_logger")

_CACHE_DB_PATH = os.getenv("TRANSCRIPTION_CACHE_DB", "data/agent.db")

# ==========================================
# 2. AuditLogger Implementation
# ==========================================
class AuditLogger:
    """Manages the lifecycle of database sessions specifically for creating audit entries."""
    
    def __init__(self, session_factory):
        """Initializes with a pre-configured sessionmaker factory."""
        self.session_factory = session_factory
        self._session = None

    def __enter__(self) -> "AuditLogger":
        """Opens a database session and explicitly begins a transaction."""
        self._session = self.session_factory()
        self._session.begin()
        return self

    def log(self, call_id: str, action: str, caller_id: str, details: dict | list | str = None):
        """Creates an AuditEntry and adds it to the active session workspace."""
        if not self._session or not self._session.is_active:
            raise RuntimeError("Database write failed: No active session scope exists. Use a 'with' block.")

        # Safely convert dict/list data into a JSON string for SQLite Text storage
        serialized_data = json.dumps(details) if isinstance(details, (dict, list)) else str(details) if details else None

        # Instantiate the database entity
        entry = AuditLogEntry(
            call_id=call_id,
            action=action,
            caller_id=caller_id,
            details=serialized_data
        )

        # Track the entity in the current session
        self._session.add(entry)
        logger.info(f"Queued log entry in session workspace: call_id={call_id}, action={action}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handles the final commit or rollback, and safely closes the database connection."""
        try:
            if exc_type is not None:
                # An error occurred inside the 'with' block; discard changes safely
                logger.error(f"Transaction failed. Rolling back changes. Error: {exc_val}")
                self._session.rollback()
            else:
                # Success; permanently write to SQLite disk
                self._session.commit()
                logger.info("Audit transaction committed to SQLite successfully.")
        finally:
            # Crucial for SQLite to prevent 'Database is locked' errors
            self._session.close()
            
        return False  # Let any underlying non-db exceptions bubble up naturally



# ==========================================
# 3. Setup and Usage Example
# ==========================================
if __name__ == "__main__":
    # Create a SQLite database file at _CACHE_DB_PATH
    engine = create_engine(f"sqlite:///{_CACHE_DB_PATH}", echo=False)

    # conn =sqlite3.connect(_CACHE_DB_PATH)
    
    # Create the 'audit_entries' table in the database
    Base.metadata.create_all(engine)
    
    # Create the global session factory
    SessionFactory = sessionmaker(bind=engine)

    # --- Scenario 1: Successful Writes ---
    print("\n--- Running Scenario 1 (Success) ---")
    
    # Pass the session factory into your logger
    with AuditLogger(SessionFactory) as audit:
        # Log a plain string action
        audit.log(
            call_id="call_9921a", 
            action="USER_LOGIN", 
            caller_id="caller_502",
            details="User ID 502 authenticated via web."
        )
        
        # Log complex structured data (auto-serializes to JSON string)
        payload = {"browser": "Chrome", "ip_address": "192.168.1.1", "status": "success"}
        audit.log(
            call_id="call_9921a", 
            action="API_REQUEST_METADATA", 
            caller_id="caller_502",
            details=payload
        )
    
    # --- Scenario 2: Error Handling and Safety Rollback ---
    print("\n--- Running Scenario 2 (Automatic Rollback on Error) ---")
    try:
        with AuditLogger(SessionFactory) as audit:
            audit.log(
                call_id="call_8832b", 
                action="SENSITIVE_PROCESS_START", 
                caller_id="caller_8832b",
                details={"step": 1}
            )
            
            # Simulate a severe runtime failure midway through your process
            print("Executing application logic... something goes horribly wrong!")
            raise ValueError("Payment gateway connection timed out.")
            
            # This line will never execute, and the 'SENSITIVE_PROCESS_START' log will be wiped
            audit.log(call_id="call_8832b", action="SENSITIVE_PROCESS_END")
            
    except ValueError as e:
        print(f"Caught expected application exception: {e}")

    # --- Verify Database State ---
    print("\n--- Verifying Final Database Entries ---")
    db_session = SessionFactory()
    all_entries = db_session.query(AuditLogEntry).all()
    print(f"Total rows written to SQLite: {len(all_entries)}")
    for item in all_entries:
        print(f" - Found Row: ID={item.id}, CallID={item.call_id}, Action={item.action}, Details={item.details}")
    db_session.close()
