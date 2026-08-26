import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, CallRecord, AuditLogEntry


@pytest.fixture
def temp_db():
    """Create a temporary in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal, engine


@pytest.fixture
def session(temp_db):
    """Provide a database session for tests."""
    SessionLocal, engine = temp_db
    session = SessionLocal()
    yield session
    session.close()


class TestCallRecordPersistence:
    """Test CallRecord insert-and-retrieve operations."""

    def test_insert_call_record_with_minimal_data(self, session):
        """Test inserting a CallRecord with minimal required fields."""
        call_record = CallRecord(
            call_id="CALL-001",
            status="completed",
        )
        session.add(call_record)
        session.commit()

        # Retrieve and verify
        retrieved = session.query(CallRecord).filter_by(call_id="CALL-001").first()
        assert retrieved is not None
        assert retrieved.call_id == "CALL-001"
        assert retrieved.status == "completed"
        assert retrieved.audio_filename is None
        assert retrieved.transcript_text is None

    def test_insert_call_record_with_full_data(self, session):
        """Test inserting a CallRecord with all fields populated."""
        report_data = {
            "call_id": "CALL-002",
            "status": "completed",
            "summary": "Customer inquiry resolved",
        }
        qa_scores_data = {
            "professionalism": 4,
            "empathy": 5,
            "problem_resolution": 4,
            "compliance": 5,
            "communication_clarity": 4,
            "overall_score": 4.4,
            "justification": "Agent handled call professionally",
        }

        call_record = CallRecord(
            call_id="CALL-002",
            status="completed",
            audio_filename="call_recording.mp3",
            transcript_text="Customer: Hello. Agent: Hi, how can I help?",
            summary_json=json.dumps({"summary": "Issue resolved"}),
            qa_scores_json=json.dumps(qa_scores_data),
            report_json=json.dumps(report_data),
            processed_at=datetime.now(timezone.utc),
            trace_id="trace-12345",
        )
        session.add(call_record)
        session.commit()

        # Retrieve and verify all fields
        retrieved = session.query(CallRecord).filter_by(call_id="CALL-002").first()
        assert retrieved is not None
        assert retrieved.call_id == "CALL-002"
        assert retrieved.status == "completed"
        assert retrieved.audio_filename == "call_recording.mp3"
        assert "Customer: Hello" in retrieved.transcript_text
        assert retrieved.summary_json is not None
        assert retrieved.qa_scores_json is not None
        assert retrieved.report_json is not None
        assert retrieved.trace_id == "trace-12345"

        # Verify JSON data can be deserialized
        summary_data = json.loads(retrieved.summary_json)
        assert summary_data["summary"] == "Issue resolved"

        qa_data = json.loads(retrieved.qa_scores_json)
        assert qa_data["professionalism"] == 4
        assert qa_data["overall_score"] == 4.4

    def test_retrieve_call_record_by_call_id(self, session):
        """Test retrieving CallRecord by call_id."""
        call_ids = ["CALL-A", "CALL-B", "CALL-C"]
        for call_id in call_ids:
            record = CallRecord(call_id=call_id, status="pending")
            session.add(record)
        session.commit()

        # Retrieve specific record
        retrieved = session.query(CallRecord).filter_by(call_id="CALL-B").first()
        assert retrieved is not None
        assert retrieved.call_id == "CALL-B"

        # Verify others exist
        all_records = session.query(CallRecord).all()
        assert len(all_records) == 3

    def test_call_record_unique_constraint_on_call_id(self, session):
        """Test that call_id is unique in CallRecord."""
        record1 = CallRecord(call_id="UNIQUE-001", status="completed")
        session.add(record1)
        session.commit()

        # Attempt to insert duplicate call_id
        record2 = CallRecord(call_id="UNIQUE-001", status="failed")
        session.add(record2)

        with pytest.raises(Exception):  # SQLAlchemy raises IntegrityError
            session.commit()

    def test_call_record_status_field_values(self, session):
        """Test CallRecord with various status values."""
        statuses = ["pending", "processing", "completed", "failed", "error"]
        for idx, status in enumerate(statuses):
            record = CallRecord(call_id=f"STATUS-{idx}", status=status)
            session.add(record)
        session.commit()

        # Verify each status
        for idx, status in enumerate(statuses):
            retrieved = session.query(CallRecord).filter_by(call_id=f"STATUS-{idx}").first()
            assert retrieved.status == status

    def test_call_record_with_null_optional_fields(self, session):
        """Test that optional fields can be null."""
        record = CallRecord(
            call_id="CALL-MINIMAL",
            status="processing",
            audio_filename=None,
            transcript_text=None,
            summary_json=None,
            qa_scores_json=None,
            report_json=None,
            processed_at=None,
            trace_id=None,
        )
        session.add(record)
        session.commit()

        retrieved = session.query(CallRecord).filter_by(call_id="CALL-MINIMAL").first()
        assert retrieved is not None
        assert retrieved.audio_filename is None
        assert retrieved.transcript_text is None
        assert retrieved.summary_json is None
        assert retrieved.trace_id is None

    def test_call_record_timestamps(self, session):
        """Test that processed_at timestamp is stored correctly."""
        now = datetime.utcnow()  # Use naive datetime to match model default
        record = CallRecord(
            call_id="CALL-TIMESTAMP",
            status="completed",
            processed_at=now,
        )
        session.add(record)
        session.commit()

        retrieved = session.query(CallRecord).filter_by(call_id="CALL-TIMESTAMP").first()
        assert retrieved.processed_at is not None
        # Allow small time diff due to DB rounding
        time_diff = abs((retrieved.processed_at - now).total_seconds())
        assert time_diff < 1


class TestAuditLogEntryPersistence:
    """Test AuditLogEntry insert-and-retrieve operations."""

    def test_insert_audit_log_entry_with_minimal_data(self, session):
        """Test inserting an AuditLogEntry with minimal required fields."""
        audit_entry = AuditLogEntry(
            call_id="CALL-AUDIT-001",
            action="call_started",
            caller_id="user-123",
        )
        session.add(audit_entry)
        session.commit()

        # Retrieve and verify
        retrieved = (
            session.query(AuditLogEntry)
            .filter_by(call_id="CALL-AUDIT-001")
            .first()
        )
        assert retrieved is not None
        assert retrieved.call_id == "CALL-AUDIT-001"
        assert retrieved.action == "call_started"
        assert retrieved.caller_id == "user-123"
        assert retrieved.details is None

    def test_insert_audit_log_entry_with_details(self, session):
        """Test inserting an AuditLogEntry with JSON details."""
        details_data = {
            "department": "billing",
            "call_duration": 300,
            "queue_wait_time": 45,
        }
        audit_entry = AuditLogEntry(
            call_id="CALL-AUDIT-002",
            action="call_completed",
            caller_id="user-456",
            details=json.dumps(details_data),
        )
        session.add(audit_entry)
        session.commit()

        # Retrieve and verify
        retrieved = (
            session.query(AuditLogEntry)
            .filter_by(call_id="CALL-AUDIT-002")
            .first()
        )
        assert retrieved is not None
        assert retrieved.details is not None

        # Verify JSON can be deserialized
        details = json.loads(retrieved.details)
        assert details["department"] == "billing"
        assert details["call_duration"] == 300

    def test_audit_log_entry_indexing_by_call_id(self, session):
        """Test efficient retrieval using indexed call_id."""
        call_ids = ["CALL-IDX-A", "CALL-IDX-B", "CALL-IDX-A"]
        for i, call_id in enumerate(call_ids):
            audit_entry = AuditLogEntry(
                call_id=call_id,
                action=f"action_{i}",
                caller_id=f"user-{i}",
            )
            session.add(audit_entry)
        session.commit()

        # Query by indexed call_id
        entries = session.query(AuditLogEntry).filter_by(call_id="CALL-IDX-A").all()
        assert len(entries) == 2

        entries = session.query(AuditLogEntry).filter_by(call_id="CALL-IDX-B").all()
        assert len(entries) == 1

    def test_audit_log_entry_actions(self, session):
        """Test various action types in AuditLogEntry."""
        actions = [
            "call_started",
            "transcription_complete",
            "qa_scored",
            "report_generated",
            "escalated",
            "call_ended",
        ]
        for idx, action in enumerate(actions):
            audit_entry = AuditLogEntry(
                call_id=f"CALL-ACTION-{idx}",
                action=action,
                caller_id=f"agent-{idx}",
            )
            session.add(audit_entry)
        session.commit()

        # Verify all actions are stored
        for idx, action in enumerate(actions):
            retrieved = (
                session.query(AuditLogEntry)
                .filter_by(call_id=f"CALL-ACTION-{idx}")
                .first()
            )
            assert retrieved.action == action

    def test_audit_log_entry_created_at_timestamp(self, session):
        """Test that created_at timestamp is automatically set."""
        audit_entry = AuditLogEntry(
            call_id="CALL-AUDIT-TIMESTAMP",
            action="test_action",
            caller_id="user-timestamp",
        )
        session.add(audit_entry)
        session.commit()

        retrieved = (
            session.query(AuditLogEntry)
            .filter_by(call_id="CALL-AUDIT-TIMESTAMP")
            .first()
        )
        assert retrieved.created_at is not None
        assert isinstance(retrieved.created_at, datetime)

    def test_audit_log_entry_multiple_entries_per_call(self, session):
        """Test multiple audit entries for the same call tracking full lifecycle."""
        call_id = "CALL-LIFECYCLE-001"
        actions_sequence = [
            ("call_started", "agent-alice"),
            ("transcription_started", "system"),
            ("transcription_complete", "system"),
            ("qa_evaluation_started", "system"),
            ("qa_evaluation_complete", "system"),
            ("report_generated", "system"),
            ("call_ended", "agent-alice"),
        ]

        for action, caller_id in actions_sequence:
            audit_entry = AuditLogEntry(
                call_id=call_id,
                action=action,
                caller_id=caller_id,
                details=json.dumps({"step": action}),
            )
            session.add(audit_entry)
        session.commit()

        # Retrieve all entries for this call
        entries = session.query(AuditLogEntry).filter_by(call_id=call_id).all()
        assert len(entries) == len(actions_sequence)

        # Verify actions are in correct order
        retrieved_actions = [e.action for e in entries]
        expected_actions = [a[0] for a in actions_sequence]
        assert retrieved_actions == expected_actions

    def test_audit_log_entry_with_special_characters_in_details(self, session):
        """Test that special characters in JSON details are preserved."""
        special_details = {
            "message": "Error: 'Invalid syntax' & \"bad code\"",
            "emoji": "🔴 Alert",
            "unicode": "こんにちは",
        }
        audit_entry = AuditLogEntry(
            call_id="CALL-SPECIAL-CHARS",
            action="error_logged",
            caller_id="system",
            details=json.dumps(special_details, ensure_ascii=False),
        )
        session.add(audit_entry)
        session.commit()

        retrieved = (
            session.query(AuditLogEntry)
            .filter_by(call_id="CALL-SPECIAL-CHARS")
            .first()
        )
        assert retrieved is not None

        deserialized = json.loads(retrieved.details)
        assert "Invalid syntax" in deserialized["message"]
        assert deserialized["emoji"] == "🔴 Alert"

    def test_audit_log_entry_large_details_payload(self, session):
        """Test that large JSON details can be stored."""
        large_details = {
            "transcript": "This is a very long transcript " * 100,
            "metadata": {f"field_{i}": f"value_{i}" for i in range(100)},
        }
        audit_entry = AuditLogEntry(
            call_id="CALL-LARGE-PAYLOAD",
            action="transcript_stored",
            caller_id="system",
            details=json.dumps(large_details),
        )
        session.add(audit_entry)
        session.commit()

        retrieved = (
            session.query(AuditLogEntry)
            .filter_by(call_id="CALL-LARGE-PAYLOAD")
            .first()
        )
        assert retrieved is not None
        assert retrieved.details is not None

        deserialized = json.loads(retrieved.details)
        assert len(deserialized["metadata"]) == 100


class TestCrossTableIntegration:
    """Test interactions between CallRecord and AuditLogEntry."""

    def test_create_call_record_with_matching_audit_entries(self, session):
        """Test creating a call record with associated audit log entries."""
        call_id = "CROSS-TABLE-001"

        # Create call record
        call_record = CallRecord(
            call_id=call_id,
            status="completed",
            audio_filename="recording.mp3",
            processed_at=datetime.now(timezone.utc),
        )
        session.add(call_record)

        # Create audit entries for same call
        for i in range(3):
            audit_entry = AuditLogEntry(
                call_id=call_id,
                action=f"step_{i}",
                caller_id=f"user-{i}",
            )
            session.add(audit_entry)

        session.commit()

        # Verify both are retrievable
        retrieved_record = (
            session.query(CallRecord).filter_by(call_id=call_id).first()
        )
        assert retrieved_record is not None

        audit_entries = (
            session.query(AuditLogEntry).filter_by(call_id=call_id).all()
        )
        assert len(audit_entries) == 3

    def test_audit_trail_for_call_status_transitions(self, session):
        """Test tracking call status changes through audit log."""
        call_id = "STATUS-CHANGE-001"

        # Create initial call record
        call_record = CallRecord(call_id=call_id, status="pending")
        session.add(call_record)
        session.commit()

        # Log status transition
        audit_entry = AuditLogEntry(
            call_id=call_id,
            action="status_changed",
            caller_id="system",
            details=json.dumps({"from": "pending", "to": "processing"}),
        )
        session.add(audit_entry)
        session.commit()

        # Update call record
        call_record.status = "processing"
        session.commit()

        # Verify audit trail
        audit_entries = (
            session.query(AuditLogEntry).filter_by(call_id=call_id).all()
        )
        assert len(audit_entries) == 1
        assert "status_changed" in audit_entries[0].action

        # Verify updated record
        updated_record = (
            session.query(CallRecord).filter_by(call_id=call_id).first()
        )
        assert updated_record.status == "processing"
