from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.graph.edges import (
    route_after_injection_check,
    route_after_intake,
    route_after_qa,
    route_after_transcription,
)
from src.graph.state import PipeLineState, TranscriptionResult, TranscriptionSegment
from src.security.pii_redactor import redact_pii


@traceable
def intake_step(state: PipeLineState) -> PipeLineState:
    """Run the intake validation and preserve the result in state."""
    from src.agents.intake import run_intake
    from src.utils.config import get_logger

    logger = get_logger("workflow")
    logger.info("Intake step started")

    state["state"] = "intake_started"
    state = run_intake(state)

    intake_state = state.get("state", "unknown")
    logger.info(f"Intake step completed with state: {intake_state}")
    if state.get("error"):
        logger.error(f"Intake error: {state.get('error')}")

    return state


@traceable
def transcription_node(state: PipeLineState) -> PipeLineState:
    """Transcribe the call audio and store the result on state."""
    from src.agents.transcription import transcribe_audio
    from src.utils.config import get_logger

    logger = get_logger("workflow")
    logger.info("Transcription node started")

    state["state"] = "transcribing"
    state = transcribe_audio(state)

    transcription = state.get("transcription")
    if transcription:
        segment_count = len(getattr(transcription, "segments", []))
        logger.info(f"Transcription node completed with {segment_count} segments")
    else:
        logger.warning("Transcription node completed but transcription is None")

    return state


@traceable
def injection_check_node(state: PipeLineState) -> PipeLineState:
    """Stop the pipeline if the transcription contains injection content."""
    from src.utils.config import get_logger

    logger = get_logger("workflow")
    transcription = state.get("transcription")
    if transcription is None:
        logger.error("No transcription available for injection check")
        state["state"] = "error"
        state["error"] = "No transcription available for injection check"
        return state

    injection_detected = getattr(transcription, "injection_detected", False)
    injection_reason = getattr(transcription, "injection_reason", None)

    if injection_detected:
        logger.warning(f"Injection detected: {injection_reason}")
        state["state"] = "flagged_for_review"
        # state["error"] = injection_reason or "prompt injection detected"
        return state

    state["state"] = "ready_for_redaction"
    return state


@traceable
def pii_redaction_node(state: PipeLineState) -> PipeLineState:
    """Redact PII from transcript segments before summarization."""
    transcription = state.get("transcription")
    if transcription is None:
        state["state"] = "error"
        state["error"] = "No transcription available for PII redaction"
        return state

    if not hasattr(transcription, "segments"):
        state["state"] = "error"
        state["error"] = "Transcription payload is not in the expected format"
        return state

    redacted_segments = []
    for segment in transcription.segments:
        redacted_text = redact_pii(segment.text)
        redacted_segments.append(
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=redacted_text,
                speaker=segment.speaker,
                confidence=segment.confidence,
            )
        )

    state["transcription"] = TranscriptionResult(
        segments=redacted_segments,
        injection_detected=transcription.injection_detected,
        injection_reason=transcription.injection_reason,
        call_id=transcription.call_id,
    )
    state["state"] = "pii_redacted"
    return state


@traceable
def summarize_and_qa_node(state: PipeLineState) -> PipeLineState:
    """Summarize the redacted transcript and run QA scoring."""
    from src.agents.qa_scoring import run_qa_scoring
    from src.agents.summarization import run_summarization

    state = run_summarization(state)
    state = run_qa_scoring(state)
    state["state"] = "summary_and_qa_complete"
    return state


@traceable
def report_node(state: PipeLineState) -> PipeLineState:
    """Compile and persist the final call report."""
    from src.agents.report import persist_report

    state = persist_report(state)
    state["state"] = state.get("state", "completed")
    return state


@traceable
def error_node(state: PipeLineState) -> PipeLineState:
    """Persist the call record even on pipeline error without changing status."""
    from src.agents.report import persist_report

    state = persist_report(state)

    return state


@traceable
def supervisor_review_node(state: PipeLineState) -> PipeLineState:
    """Route a flagged call to human review."""
    state["state"] = "supervisor_review"
    state["error"] = state.get("error") or "Call requires supervisor review"
    return state


def build_workflow() -> StateGraph:
    """Build a simple LangGraph workflow for the call processing pipeline."""
    workflow = StateGraph(PipeLineState)
    workflow.add_node("intake_step", intake_step)
    workflow.add_node("transcribe_step", transcription_node)
    workflow.add_node("injection_check_step", injection_check_node)
    workflow.add_node("pii_redact_step", pii_redaction_node)
    workflow.add_node("summarize_and_qa_step", summarize_and_qa_node)
    workflow.add_node("report_step", report_node)
    workflow.add_node("error_step", error_node)
    workflow.add_node("supervisor_step", supervisor_review_node)

    workflow.set_entry_point("intake_step")

    # Conditional edges with routing logic
    workflow.add_conditional_edges("intake_step", route_after_intake)
    workflow.add_conditional_edges("transcribe_step", route_after_transcription)
    workflow.add_conditional_edges("injection_check_step", route_after_injection_check)
    workflow.add_conditional_edges("summarize_and_qa_step", route_after_qa)

    # Regular edges for non-conditional paths
    workflow.add_edge("pii_redact_step", "summarize_and_qa_step")

    # Terminal edges to END
    workflow.add_edge("report_step", END)
    workflow.add_edge("error_step", END)
    workflow.add_edge("supervisor_step", END)

    return workflow


__all__ = [
    "build_workflow",
    "error_node",
    "injection_check_node",
    "intake_step",
    "pii_redaction_node",
    "report_node",
    "summarize_and_qa_node",
    "supervisor_review_node",
    "traceable",
    "transcription_node",
]
