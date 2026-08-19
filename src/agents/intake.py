

import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List

from graph.state import PIIScanResult, PipeLineState
from utils.audio import validate_audio
import uuid
from utils.audio import extract_audio_properties, validate_audio_duration



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


def scan_metadata_for_pii(metadata: Dict[str, str]) -> PIIScanResult:
    """Scan metadata fields `caller_id` and `department` for PII.

    Returns PIIScanResult with pii_detected and list of affected field names.
    """
    affected: List[str] = []
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
_EMPTY_PII = PIIScanResult(pii_detected=False, affected_fields=[])


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
    # generate uuid for the audio input
    if "audio_input" in state and state["audio_input"] is None:
        state["audio_input"].caller_id = str(uuid.uuid4())

    validation_result = validate_audio(state["audio_input"].audio_bytes, state["audio_input"].filename)
    if not validation_result.is_valid:
        state["intake_result"] = _make_failed_result(validation_result.error)
        state["pii_scan"] = _EMPTY_PII
    else:
        # Assuming extract_audio_properties is a function that extracts properties from the audio bytes
        properties = extract_audio_properties(state["audio_input"].audio_bytes, state["audio_input"].filename)

        validation_result = validate_audio_duration(properties.duration_seconds)
        if not validation_result.is_valid:
            state["intake_result"] = _make_failed_result(validation_result.error)
            state["pii_scan"] = _EMPTY_PII
            return state
        
        # scan caller_id and department metadata fields for PII using regex patterns.
        metadata = {
            "caller_id": state["audio_input"].caller_id,
            "department": state["audio_input"].department
        }
        pii_result = scan_metadata_for_pii(metadata)
        if pii_result.pii_detected:
            print(f"PII detected in metadata fields: {pii_result.affected_fields}")
            state["pii_scan"] = pii_result
            state["intake_result"] = _make_failed_result(f"PII detected in metadata fields: {pii_result.affected_fields}")
            return state
        else:
            print("No PII detected in metadata fields.")
            state["pii_scan"] = pii_result


        with tempfile.NamedTemporaryFile(suffix=state["audio_input"].filename.split('.')[-1], delete=False) as temp_file:
            temp_file.write(state["audio_input"].audio_bytes)
            state["intake_result"] = {
                "is_valid": True,
                "reason": None,
                "properties": properties
            }

    return state
