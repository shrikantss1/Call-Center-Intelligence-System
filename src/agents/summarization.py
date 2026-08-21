


from src.utils.config import get_logger
from time import time

from src.utils.llm_factory import llm
from src.graph.state import SummaryResult
from src.graph.state import PipeLineState

logger = get_logger("summarization")

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
        # If transcription is not available, we cannot proceed with summarization
        state["summary"] = {
            "is_valid": False,
            "reason": "Transcription not available for summarization.",
            "summary": None,
        }
        return state

    transcription_segments = state["transcription"].segments
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
            break  # Exit the retry loop if successful
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = min(2 ** attempt, 10)
                logger.error(f"Summarization attempt {attempt + 1} failed: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                state["summary"] = {
                            "is_valid": False,
                            "reason": f"Summarization failed after {max_retries} attempts: {e}",
                            "summary": None,
                        }
                return state

    return state