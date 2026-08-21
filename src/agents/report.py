import json
from datetime import datetime, timezone

from src.graph.state import PipeLineState, CallReport, QAScoringResult
from src.database.models import CallRecord
from src.database.connection import get_session
from src.utils.config import get_logger

logger = get_logger("report")


def compile_report(state: PipeLineState) -> PipeLineState:
    """Assemble a report from all upstream pipeline results and update state.

    Args:
        state: The PipeLineState containing all processed results

    Returns:
        The updated PipeLineState with call_report field populated
    """

    logger.info(f"Compiling report for call_id: {state.get('audio_input').call_id if state.get('audio_input') else 'unknown'}")
    audio_input = state.get("audio_input")
    call_id = audio_input.call_id if audio_input else "unknown"

    transcript_text = None
    transcription = None
    if state.get("transcription"):
        trans = state["transcription"]
        transcription = trans
        transcript_text = " ".join(seg.text for seg in trans.segments)

    summary = state.get("summary")
    if summary is not None:
        if hasattr(summary, "summary"):
            summary = summary.summary
        elif isinstance(summary, dict):
            summary = summary.get("summary")

    logger.info(f"Compiling report for call_id: {call_id}, transcript length: {len(transcript_text) if transcript_text else 0}, summary length: {len(summary) if summary else 0}")

    qa_scores = state.get("qa_score")
    if qa_scores is not None and isinstance(qa_scores, dict):
        nested = qa_scores.get("qa_score") if isinstance(qa_scores.get("qa_score"), dict) else qa_scores
        try:
            qa_scores = QAScoringResult.model_validate(nested)
        except Exception:
            qa_scores = None

    pii_scan = None
    if state.get("pii_scan"):
        pii_scan = state["pii_scan"]

    status = state.get("state", "completed")
    logger.info(f"Compiling report for call_id: {call_id}, status: {status}")
    error = state.get("error")
    if error:
        status = "failed"

    try:
        report = CallReport(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            audio_filename="abc", #audio_input.filename if audio_input else None,
            transcription=transcription,
            transcript_text=transcript_text,
            summary=summary,
            qa_scores=qa_scores,
            pii_scan=pii_scan,
            status=status,
            error=error,
        )
        logger.info(f"status: {report.status}, audio_filename: {report.audio_filename}")
    except Exception as e:
        logger.error(f"Failed to create CallReport for call_id: {call_id}, error: {e}", exc_info=True)
        raise

    state = {
        "call_report": report
    }
    logger.info(f"status: {state['call_report'].status}, audio_filename: {state['call_report'].audio_filename}")
    return state


def persist_report(state: PipeLineState) -> PipeLineState:
    """Write a CallRecord row to the database using compiled report data.

    Args:
        state: The PipeLineState containing all processed results

    Returns:
        The updated state
    """

    logger.info(f"Persisting report for call_id: {state.get('call_report').call_id if state.get('call_report') else 'unknown'}")
    try:
        state = compile_report(state)
        report = state["call_report"]
        logger.info(f"Persisting report for call_id: {report.call_id}, status: {report.status}, audio_filename: {report.audio_filename}")
        session = get_session()
        try:
            call_record = CallRecord(
                call_id=report.call_id,
                status=report.status,
                audio_filename=report.audio_filename,
                transcript_text=report.transcript_text,
                summary_json=json.dumps({"summary": report.summary}) if report.summary else None,
                qa_scores_json=report.qa_scores.model_dump_json() if report.qa_scores else None,
                report_json=report.model_dump_json(),
                processed_at=datetime.now(timezone.utc),
            )
            session.add(call_record)
            session.commit()
            state["state"] = "persisted"
        finally:
            session.close()

        return state

    except Exception as e:
        state["state"] = "persistence_failed"
        state["error"] = str(e)
        return state
    