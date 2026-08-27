from src.utils.config import get_logger
from time import sleep
from src.utils.llm_factory import llm
from src.graph.state import QAScoringResult, PipeLineState
from src.security.audit import AuditLogger
from src.database.connection import get_engine
from sqlalchemy.orm import sessionmaker

logger = get_logger("qa_scoring")

_engine = get_engine()
_SessionFactory = sessionmaker(bind=_engine)

DIMENSION_WEIGHTS = {
    "professionalism": 0.15,
    "empathy": 0.20,
    "problem_resolution": 0.30,
    "compliance": 0.20,
    "communication_clarity": 0.15,
}

def get_qa_scoring_prompt() -> str:
    """Return the system prompt for QA scoring with philosophy, rubrics, and guidelines."""
    return """You are an expert quality assurance coach for call center interactions. Your role is to evaluate calls fairly and constructively, using clear standards that distinguish truly competent performance from exceptional work.

## SCORING PHILOSOPHY

The scale 1-5 is calibrated as follows. **A score of 3 represents baseline competent performance** — the agent handled the call correctly, professionally, and met all core requirements. This is the expected standard. Do not inflate scores.

- **1-2**: Below baseline. Agent struggled with basics: rudeness, misunderstanding, incomplete resolution, or compliance breaches.
- **3**: Baseline competent. Agent performed the role correctly. No problems, but nothing exceptional. This is the target standard.
- **4**: Above baseline. Agent exceeded expectations in meaningful ways: extra empathy, proactive solutions, excellent clarity.
- **5**: Exceptional. Could serve as a training exemplar. Masterful handling across multiple dimensions.

## DIMENSION RUBRICS

### Professionalism (Scale 1-5)
- **5**: Impeccable tone, courteous throughout, manages emotions perfectly. Sets an exemplary standard.
- **4**: Professional and respectful, handles difficult moments well. Minor tone inconsistencies that don't undermine the interaction.
- **3**: Baseline professional. Courteous, appropriate tone, no rude or dismissive moments. Meets expectations.
- **2**: Generally professional but moments of impatience, slight condescension, or defensiveness.
- **1**: Rude, dismissive, or repeatedly unprofessional behavior that damages the interaction.

### Empathy (Scale 1-5)
- **5**: Exceptional empathy. Validates concerns authentically, acknowledges frustration, makes the caller feel heard. Proactively reassuring.
- **4**: Genuine empathy. Acknowledges caller's situation, shows understanding. Mostly warm and supportive.
- **3**: Baseline empathy. Acknowledges the issue, responds appropriately. Neither cold nor notably warm. Meets expectations.
- **2**: Limited empathy. Focuses on process over the caller's feelings. May seem dismissive of concerns.
- **1**: No empathy. Dismissive of caller emotions or indifferent to their situation.

### Problem Resolution (Scale 1-5)
- **5**: Exceptional resolution. Solved the problem completely, went above and beyond, offered preventive guidance or alternatives.
- **4**: Strong resolution. Solved the core issue efficiently. Caller likely satisfied.
- **3**: Baseline resolution. Problem was resolved adequately. Caller should be able to move forward. Meets expectations.
- **2**: Partial resolution. Problem partially addressed or solution unclear. Caller may need follow-up.
- **1**: Failed to resolve. Problem unaddressed or made worse.

### Compliance (Scale 1-5)
- **5**: Perfect compliance. All required procedures followed, all regulatory/policy requirements met. Exemplary adherence.
- **4**: Strong compliance. All major requirements met. Possible minor procedural gaps that don't undermine compliance.
- **3**: Baseline compliance. Followed all core procedural and regulatory requirements. Meets expectations.
- **2**: Compliance issues. Skipped steps or ignored minor policies, but no major violations.
- **1**: Serious compliance breach. Violated significant policy or regulatory requirements.

**Note on compliance**: Only flag genuine procedural or regulatory violations. Do NOT penalize for communication style, call length, or efficiency. Short calls are not deficient—they can show competence.

### Communication Clarity (Scale 1-5)
- **5**: Exceptional clarity. Articulate, well-organized, explains concepts simply, avoids jargon. Caller never confused.
- **4**: Clear communication. Easy to follow, good use of summaries or confirmations. Minor clarity gaps.
- **3**: Baseline clarity. Agent explains clearly enough for caller to understand the information and next steps. Meets expectations.
- **2**: Some confusion. Agent's explanations are somewhat unclear or disorganized. Caller may ask for clarification.
- **1**: Confusing or unclear. Poor articulation, contradictions, or jargon without explanation.

## JUSTIFICATION GUIDELINES

- **Cite specific moments** in the call using MM:SS timestamp format (e.g., "At 02:15, the agent...").
- **Write like a coach, not a critic**. Frame feedback constructively: "At 01:20, the agent could have acknowledged the caller's frustration more explicitly" rather than "The agent was uncaring."
- **Be concrete**. Don't just say "good empathy"—cite the moment that demonstrated it.
- **Connect to the rubric**. Explain why a score aligns with the 1-5 rubric you're using.
- **Segment the justification by dimension**. Produce a separate justification for each score in the output JSON, using distinct keys such as "Professionalism": "<justification>", "Empathy": "<justification>", "Problem_Resolution": "<justification>", "Compliance": "<justification>", and "Communication_Clarity": "<justification>". Add a new line after every justification for readability.

## OUTPUT INSTRUCTIONS

Provide scores for all five dimensions (professionalism, empathy, problem_resolution, compliance, communication_clarity), each 1-5. The system will compute the overall_score using weighted averaging, so do not worry about calculating it—just focus on scoring each dimension fairly and writing a clear, coaching-style justification that cites timestamps.

If you detect a genuine compliance violation (not a style preference), flag it in compliance_flag and explain the violation in compliance_details."""


def run_qa_scoring(state: PipeLineState) -> PipeLineState:
    """
    Run QA scoring on the given state.

    Evaluates the call across five dimensions using an LLM, then computes the
    overall_score using fixed dimension weights (overriding the LLM's value).

    Args:
        state (PipeLineState): The current state of the pipeline.

    Returns:
        PipeLineState: The updated state after QA scoring.
    """
    if "transcription" not in state or state["transcription"] is None:
        error_msg = "Transcription not available for QA scoring."
        logger.error("QA scoring skipped: %s", error_msg)
        state["qa_score"] = {
            "is_valid": False,
            "reason": error_msg,
            "qa_score": None,
        }
        state["state"] = "qa_scoring_failed"
        state["error"] = error_msg
        return state

    if "summary" not in state or state["summary"] is None:
        error_msg = "Summary not available for QA scoring."
        logger.error("QA scoring skipped: %s", error_msg)
        state["qa_score"] = {
            "is_valid": False,
            "reason": error_msg,
            "qa_score": None,
        }
        state["state"] = "qa_scoring_failed"
        state["error"] = error_msg
        return state

    transcription = state["transcription"]
    summary = state["summary"]
    call_id = transcription.call_id

    # Format transcript with timestamps
    formatted_transcript = "\n".join(
        [f"[{_format_timestamp(segment.start)}-{_format_timestamp(segment.end)}] {segment.speaker}: {segment.text}"
         for segment in transcription.segments]
    )

    # Build user message with transcript and summary
    user_message = f"""Please evaluate this call interaction:

## CALL TRANSCRIPT
{formatted_transcript}

## CALL SUMMARY
{summary.get('summary', summary) if isinstance(summary, dict) else summary.summary}

Provide fair, baseline-anchored scores across all five dimensions. Be specific with your feedback and cite timestamps."""

    logger.info("Starting QA scoring for call_id=%s with %s transcript segments and summary available=%s", call_id or "unknown", len(transcription.segments), summary is not None)
    with AuditLogger(_SessionFactory) as audit:
        audit.log(
            call_id=call_id or "unknown",
            action="QA_SCORING_STARTED",
            caller_id="unknown",
            details={"transcript_segments": len(transcription.segments), "summary_available": summary is not None}
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info("QA scoring attempt %s/%s for call_id=%s", attempt + 1, max_retries, call_id or "unknown")
            scoring_result = llm.with_structured_output(QAScoringResult).invoke(
                [
                    {"role": "system", "content": get_qa_scoring_prompt()},
                    {"role": "user", "content": user_message},
                ]
            )

            computed_overall = (
                scoring_result.professionalism * DIMENSION_WEIGHTS["professionalism"]
                + scoring_result.empathy * DIMENSION_WEIGHTS["empathy"]
                + scoring_result.problem_resolution * DIMENSION_WEIGHTS["problem_resolution"]
                + scoring_result.compliance * DIMENSION_WEIGHTS["compliance"]
                + scoring_result.communication_clarity * DIMENSION_WEIGHTS["communication_clarity"]
            )

            scoring_result.overall_score = round(computed_overall, 2)
            scoring_result.call_id = call_id

            state["qa_score"] = scoring_result
            state["state"] = "qa_scoring_complete"
            logger.info("QA scoring completed for call_id=%s with overall_score=%s", call_id or "unknown", scoring_result.overall_score)
            break

        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = min(2 ** attempt, 10)
                logger.warning(
                    "QA scoring attempt %s/%s failed for call_id=%s: %s. Retrying in %s seconds...",
                    attempt + 1,
                    max_retries,
                    call_id or "unknown",
                    e,
                    sleep_time,
                )
                sleep(sleep_time)
            else:
                error_msg = f"QA scoring failed after {max_retries} attempts: {e}"
                with AuditLogger(_SessionFactory) as audit:
                    audit.log(
                        call_id=call_id or "unknown",
                        action="QA_SCORING_FAILED",
                        caller_id="unknown",
                        details={"error": error_msg, "state": "qa_scoring_failed"}
                    )
                state["qa_score"] = {
                    "is_valid": False,
                    "reason": error_msg,
                    "qa_score": None,
                }
                logger.error("QA scoring failed for call_id=%s: %s", call_id or "unknown", e)
                state["state"] = "qa_scoring_failed"
                state["error"] = error_msg
                return state

    return state


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
