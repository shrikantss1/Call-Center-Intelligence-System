


from src.utils.config import get_logger
from time import time

from src.utils.llm_factory import llm
from src.graph.state import SummaryResult
from src.graph.state import PipeLineState
from src.security.audit import AuditLogger
from src.database.connection import get_engine
from sqlalchemy.orm import sessionmaker

logger = get_logger("summarization")

_engine = get_engine()
_SessionFactory = sessionmaker(bind=_engine)

def run_summarization(state: PipeLineState) -> PipeLineState:
    """
    Run the summarization process on the given state.

    Args:
        state (PipeLineState): The current state of the pipeline.

    Returns:
        PipeLineState: The updated state after running the summarization process.
    """
    # Check if transcription is available in the state
    if "transcription" not in state or state["transcription"] is None:
        error_msg = "Transcription not available for summarization."
        with AuditLogger(_SessionFactory) as audit:
            audit.log(
                call_id=state.get("transcription").call_id if state.get("transcription") else "unknown",
                action="SUMMARIZATION_FAILED",
                caller_id="unknown",
                details={"error": error_msg, "state": "summarization_failed"}
            )
        logger.error(error_msg)
        state["summary"] = {
            "is_valid": False,
            "reason": error_msg,
            "summary": None,
        }
        state["state"] = "summarization_failed"
        state["error"] = error_msg
        return state

    transcription_segments = state["transcription"].segments
    call_id = state["transcription"].call_id
    with AuditLogger(_SessionFactory) as audit:
        audit.log(
            call_id=call_id or "unknown",
            action="SUMMARIZATION_STARTED",
            caller_id="unknown",
            details={"segment_count": len(transcription_segments)}
        )
    # Format the segments into a single string for summarization
    formatted_transcription = "\n".join(
        [f"{segment.start}-{segment.end}: {segment.speaker}: {segment.text}" for segment in transcription_segments]
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            summary = llm.with_structured_output(SummaryResult).invoke(formatted_transcription)
            state["summary"] = {
                "is_valid": True,
                "reason": None,
                "summary": summary.summary,
                "call_id": state["transcription"].call_id,
            }
            state["state"] = "summarization_complete"
            break  # Exit the retry loop if successful
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = min(2 ** attempt, 10)
                logger.error(f"Summarization attempt {attempt + 1} failed: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                error_msg = f"Summarization failed after {max_retries} attempts: {e}"
                with AuditLogger(_SessionFactory) as audit:
                    audit.log(
                        call_id=call_id or "unknown",
                        action="SUMMARIZATION_FAILED",
                        caller_id="unknown",
                        details={"error": error_msg, "state": "summarization_failed"}
                    )
                logger.error(error_msg)
                state["summary"] = {
                            "is_valid": False,
                            "reason": error_msg,
                            "summary": None,
                        }
                state["state"] = "summarization_failed"
                state["error"] = error_msg
                return state

    return state