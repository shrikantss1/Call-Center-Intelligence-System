

import os
import re
import tempfile
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.graph.state import PIIScanResult, PipeLineState
from src.security.audit import AuditLogger
from src.utils.audio import (
    extract_audio_properties,
    validate_audio,
    validate_audio_duration,
)
from src.utils.config import get_logger

logger = get_logger("intake")

# Setup SQLAlchemy session factory for audit logging
_CACHE_DB_PATH = os.getenv("TRANSCRIPTION_CACHE_DB", "data/agent.db")
_engine = create_engine(f"sqlite:///{_CACHE_DB_PATH}", echo=False)
Base.metadata.create_all(_engine)
_SessionFactory = sessionmaker(bind=_engine)

# Define PII regex patterns for metadata scanning
# SSN: e.g. 123-45-6789
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # credit card: match 16 digits optionally separated by spaces or dashes in groups of 4
    "credit_card": re.compile(r"\b(?:\d[ -]*?){16}\b"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    # phone: simple patterns for US-style numbers (e.g. (123) 456-7890, 123-456-7890, 1234567890)
    "phone": re.compile(r"\b(?:\+?1[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b"),
}


def scan_metadata_for_pii(metadata: dict[str, str]) -> PIIScanResult:
    """Scan metadata fields `caller_id` and `department` for PII.

    Returns PIIScanResult with pii_detected and list of affected field names.
    """
    affected: list[str] = []
    if not metadata:
        return PIIScanResult(pii_detected=False, affected_fields=affected)

    for field in ("caller_id", "department"):
        val = metadata.get(field)
        if not val:
            continue
        for _name, pattern in PII_PATTERNS.items():
            if pattern.search(val):
                affected.append(field)
                break

    return PIIScanResult(pii_detected=bool(affected), affected_fields=affected)


# Constants and helpers for failed intake results
_EMPTY_AUDIO_PROPS = None

def _make_failed_result(reason: str):
    return {
        "is_valid": False,
        "reason": reason,
        "properties": _EMPTY_AUDIO_PROPS,
    }


def run_intake(state: PipeLineState) -> PipeLineState:
    """
    Run the intake process on the given state.

    Args:
        state (PipeLineState): The current state of the pipeline.

    Returns:
        PipeLineState: The updated state after running the intake process.
    """
    try:
        # generate uuid for the audio input
        if "audio_input" in state and state["audio_input"] is not None:
            state["audio_input"].call_id = str(uuid.uuid4())
            logger.info(f"Generated intake call_id={state['audio_input'].call_id} for file={state['audio_input'].filename}")
        else:
            logger.warning("Intake called without audio_input in state.")
            return state

        audio_input = state["audio_input"]
        call_id = audio_input.call_id
        caller_id = audio_input.caller_id or "unknown"
        filename = audio_input.filename
        if not filename:
            logger.info("Audio filename missing; creating temporary file for intake validation.")
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(audio_input.audio_bytes)
                        filename = temp_file.name

        logger.info(f"Starting intake validation for call_id={call_id}, filename={filename}, caller_id={caller_id}")

        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="INTAKE_STARTED",
                caller_id=caller_id,
                details={"filename": filename}
            )

        validation_result = validate_audio(audio_input.audio_bytes, filename)
        if not validation_result.is_valid:
            logger.error(f"Audio validation failed for call_id={call_id}: {validation_result.error}")
            with AuditLogger(_SessionFactory) as audit:
                audit.log(
                    call_id=call_id,
                    action="AUDIO_VALIDATION_FAILED",
                    caller_id=caller_id,
                    details={"filename": filename, "error": validation_result.error}
                )
            state["intake_result"] = _make_failed_result(validation_result.error)
            state["state"] = "intake_failed"
            state["error"] = validation_result.error
            return state

        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="AUDIO_VALIDATION_PASSED",
                caller_id=caller_id,
                details={"filename": filename}
            )

        # Extract audio properties
        logger.info(f"Extracting audio properties for call_id={call_id}")
        properties = extract_audio_properties(audio_input.audio_bytes, filename)

        validation_result = validate_audio_duration(properties.duration_seconds)
        if not validation_result.is_valid:
            logger.error(f"Duration validation failed for call_id={call_id}: {validation_result.error}")
            with AuditLogger(_SessionFactory) as audit:
                audit.log(
                    call_id=call_id,
                    action="DURATION_VALIDATION_FAILED",
                    caller_id=caller_id,
                    details={"filename": filename, "duration_seconds": properties.duration_seconds, "error": validation_result.error}
                )
            state["intake_result"] = _make_failed_result(validation_result.error)
            state["state"] = "intake_failed"
            state["error"] = validation_result.error
            return state

        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="DURATION_VALIDATION_PASSED",
                caller_id=caller_id,
                details={"filename": filename, "duration_seconds": properties.duration_seconds}
            )

        # scan caller_id and department metadata fields for PII using regex patterns.
        metadata = {
            "caller_id": audio_input.caller_id,
            "department": audio_input.department
        }

        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="PII_SCAN_STARTED",
                caller_id=caller_id,
                details={"filename": filename}
            )

        pii_result = scan_metadata_for_pii(metadata)
        if pii_result.pii_detected:
            logger.warning(f"PII detected in metadata fields for call_id={call_id}: {pii_result.affected_fields}")
            with AuditLogger(_SessionFactory) as audit:
                audit.log(
                    call_id=call_id,
                    action="PII_DETECTED",
                    caller_id=caller_id,
                    details={"filename": filename, "affected_fields": pii_result.affected_fields}
                )
            state["pii_scan"] = pii_result
            state["intake_result"] = _make_failed_result(f"PII detected in metadata fields: {pii_result.affected_fields}")
            state["state"] = "intake_failed"
            state["error"] = f"PII detected in metadata fields: {pii_result.affected_fields}"
            return state
        else:
            logger.info(f"No PII detected in metadata fields for call_id={call_id}")
            with AuditLogger(_SessionFactory) as audit:
                audit.log(
                    call_id=call_id,
                    action="PII_SCAN_PASSED",
                    caller_id=caller_id,
                    details={"filename": filename}
                )
            state["pii_scan"] = pii_result



        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="INTAKE_COMPLETED",
                caller_id=caller_id,
                details={"filename": filename, "duration_seconds": properties.duration_seconds}
            )

        state["intake_result"] = {
                "is_valid": True,
                "reason": None,
                "properties": properties
            }
        state["state"] = "intake_complete"
        logger.info(f"Intake completed successfully for call_id={call_id}, duration_seconds={properties.duration_seconds}")
        return state

    except Exception as e:
        call_id = state["audio_input"].call_id if state.get("audio_input") else "unknown"
        caller_id = state["audio_input"].caller_id if state.get("audio_input") else "unknown"
        filename = state["audio_input"].filename if state.get("audio_input") else "unknown"
        logger.exception(f"Intake failed for call_id={call_id}, filename={filename}: {e!s}")

        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=call_id,
                action="INTAKE_FAILED",
                caller_id=caller_id,
                details={"filename": filename, "error": str(e)}
            )
        state["state"] = "intake_failed"
        state["error"] = str(e)
        raise
