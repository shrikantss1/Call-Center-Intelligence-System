"""Processing utilities for call analysis."""

import json
import io
import wave
import numpy as np
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from src.graph.state import PipeLineState, AudioInput
from src.graph.workflow import build_workflow
from src.agents.report import generate_report_pdf, generate_report_json
from src.utils.config import get_logger

logger = get_logger("process")

def process_call(
    audio_data: Tuple[int, np.ndarray],
    caller_id: Optional[str] = None,
    department: Optional[str] = None,
) -> Tuple[str, str, str, Optional[str], Optional[str], str]:
    """
    Process a call recording through the analysis pipeline.

    Args:
        audio_data: Tuple of (sample_rate, audio_array) from gr.Audio
        caller_id: Optional caller ID
        department: Optional department

    Returns:
        Tuple of (transcript, summary, qa_scores_md, pdf_path, json_path, error_message)
    """
    if audio_data is None:
        return "", "", "", None, None, ""

    sample_rate, audio_array = audio_data

    # Convert numpy array to bytes
    # (audio_array * 32767).astype(np.int16).tobytes()
    audio_bytes = convert_to_safe_bytes((sample_rate, audio_array))

    # Create initial state with audio input
    initial_state: PipeLineState = {
        "audio_input": AudioInput(
            audio_bytes=audio_bytes,
            call_id=caller_id or None,
            department=department or None,
        ),
    }

    # Build and run workflow
    workflow = build_workflow()
    agent = workflow.compile()
    final_state = agent.invoke(initial_state)
    logger.info(f"Final state after processing: {final_state.get('state', 'unknown')}")

    # Check for errors from state
    error_message = final_state.get("error", "")
    if error_message:
        return "", "", "", None, None, f"❌ Error: {error_message}"

    # Extract results from state
    transcription = final_state.get("transcription")
    if transcription and hasattr(transcription, "segments"):
        transcript = "\n".join([f"[{seg.speaker}] - {seg.text}" for seg in transcription.segments])
    else:
        transcript = ""

    summary_data = final_state.get("summary", {})
    if hasattr(summary_data, "dict"):
        summary_data = summary_data.dict()

    qa_data = final_state.get("qa_score", {})
    if hasattr(qa_data, "dict"):
        qa_data = qa_data.dict()
    call_report = final_state.get("call_report")

    # Format summary markdown
    if isinstance(summary_data, str):
        summary_data = json.loads(summary_data)
    summary_text = summary_data.get("summary", "No summary available") if isinstance(summary_data, dict) else "No summary available"
    summary_md = f"### Summary\n\n{summary_text}"

    # Format QA scores markdown
    if isinstance(qa_data, str):
        qa_data = json.loads(qa_data)
    qa_md = "### QA Scores\n\n"
    if qa_data and isinstance(qa_data, dict):
        qa_md += f"- **Professionalism**: {qa_data.get('professionalism', 'N/A')}/5\n"
        qa_md += f"- **Empathy**: {qa_data.get('empathy', 'N/A')}/5\n"
        qa_md += f"- **Problem Resolution**: {qa_data.get('problem_resolution', 'N/A')}/5\n"
        qa_md += f"- **Compliance**: {qa_data.get('compliance', 'N/A')}/5\n"
        qa_md += f"- **Communication Clarity**: {qa_data.get('communication_clarity', 'N/A')}/5\n"
        qa_md += f"- **Overall Score**: {qa_data.get('overall_score', 'N/A')}/5\n"
        if qa_data.get("justification"):
            qa_md += f"- **Justification**: \n {qa_data.get('justification')}\n"
    else:
        qa_md += "No QA scores available"

    # Generate PDF and JSON reports
    pdf_path = None
    json_path = None

    if call_report:
        try:
            pdf_bytes = generate_report_pdf({"call_report": call_report})
            pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            pdf_file.write(pdf_bytes)
            pdf_file.close()
            pdf_path = pdf_file.name
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")

        try:
            json_content = generate_report_json({"call_report": call_report})
            json_file = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
            json_file.write(json_content)
            json_file.close()
            json_path = json_file.name
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")

    return transcript, summary_md, qa_md, pdf_path, json_path, ""

def convert_to_safe_bytes(audio_data):
    sampling_rate, audio_array = audio_data

    # Check if the array is floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        # Clip to guarantee values don't exceed boundaries, then scale
        audio_array = np.clip(audio_array, -1.0, 1.0)
        int16_array = (audio_array * 32767).astype(np.int16)
    else:
        # It's already integers (like int16), just ensure the type matches
        int16_array = audio_array.astype(np.int16)

    # Handle mono vs stereo
    if len(int16_array.shape) == 1:
        n_channels = 1
    else:
        n_channels = int16_array.shape[1]

    # Wrap raw PCM in WAV format with proper header
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)  # int16 = 2 bytes
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(int16_array.tobytes())

    return wav_buffer.getvalue()
