"""Pipeline service for call analysis and reporting."""

import json
import wave
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

from src.graph.state import PipeLineState, AudioInput, QAScoringResult
from src.app_globals import get_agent
from src.agents.report import generate_report_pdf, generate_report_json
from src.utils.config import get_logger

logger = get_logger("pipeline")

# Module-level temp file tracking for rolling cleanup
_temp_files = []
_MAX_TEMP_FILES = 50


@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    transcript: str
    summary: str
    qa_scores: str
    pdf_path: Optional[str]
    json_path: Optional[str]
    error: str


def _cleanup_old_temp_files():
    """Remove temp files beyond the 50-file cap, keeping most recent."""
    global _temp_files
    if len(_temp_files) > _MAX_TEMP_FILES:
        files_to_remove = _temp_files[:-_MAX_TEMP_FILES]
        for filepath in files_to_remove:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temp file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {filepath}: {e}")
        _temp_files = _temp_files[-_MAX_TEMP_FILES:]


def _add_temp_file(filepath: str):
    """Track a temp file for cleanup."""
    global _temp_files
    _temp_files.append(filepath)
    _cleanup_old_temp_files()


def _load_audio_from_file(filepath: str) -> Tuple[int, np.ndarray]:
    """Load audio from a file and convert to numpy array.

    Args:
        filepath: Path to the audio file

    Returns:
        Tuple of (sample_rate, audio_array)

    Raises:
        ValueError: If the file cannot be decoded or is not a valid audio file
    """
    if not os.path.exists(filepath):
        raise ValueError("Audio file not found")

    # Try to load with pydub for format flexibility
    if AudioSegment is not None:
        try:
            audio = AudioSegment.from_file(filepath)
            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples())

            # Handle stereo/mono
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Convert to float in range [-1, 1]
            samples = samples.astype(np.float32) / 32768.0

            return sample_rate, samples
        except Exception as e:
            error_msg = str(e)
            if "ffmpeg" in error_msg.lower() or "decode" in error_msg.lower():
                raise ValueError("Invalid or corrupted audio file. Please ensure the file is a valid audio format (MP3, WAV, etc.)")
            raise ValueError(f"Failed to load audio file: {error_msg}")

    # Fallback to wave module for WAV files only
    try:
        with wave.open(filepath, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)
            samples = np.frombuffer(audio_data, dtype=np.int16)

            if n_channels == 2:
                samples = samples.reshape((-1, 2))

            samples = samples.astype(np.float32) / 32768.0
            return sample_rate, samples
    except Exception as e:
        raise ValueError("Invalid or corrupted audio file. Please ensure the file is a valid audio format (MP3, WAV, etc.)")


def _write_audio_to_wav(audio_data: Union[Tuple[int, np.ndarray], str]) -> Tuple[str, bytes]:
    """Convert audio to WAV format. Handles both numpy arrays and file paths.

    Args:
        audio_data: Either a tuple of (sample_rate, audio_array) or a filepath string

    Returns:
        Tuple of (path to the temporary WAV file, WAV file bytes)

    Raises:
        ValueError: If audio cannot be read or converted
    """
    # If it's a filepath, load the audio first
    if isinstance(audio_data, str):
        sample_rate, audio_array = _load_audio_from_file(audio_data)
    else:
        sample_rate, audio_array = audio_data

    # Ensure audio is in int16 format
    if np.issubdtype(audio_array.dtype, np.floating):
        audio_array = np.clip(audio_array, -1.0, 1.0)
        int16_array = (audio_array * 32767).astype(np.int16)
    else:
        int16_array = audio_array.astype(np.int16)

    # Determine channels
    n_channels = 1 if len(int16_array.shape) == 1 else int16_array.shape[1]

    # Create WAV data in memory first
    wav_buffer = tempfile.SpooledTemporaryFile()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_array.tobytes())

    # Get bytes from buffer
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()
    wav_buffer.close()

    # Write to temp file for storage
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = temp_file.name
    temp_file.write(wav_bytes)
    temp_file.close()

    _add_temp_file(wav_path)
    return wav_path, wav_bytes


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to [MM:SS] format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def _format_transcript(final_state: dict) -> str:
    """Format transcript with [MM:SS] Speaker: text tags and [LOW CONF] markers.

    Args:
        final_state: The final pipeline state

    Returns:
        Formatted transcript string
    """
    transcription = final_state.get("transcription")
    if not transcription:
        logger.warning("No transcription object found in final state")
        return ""

    if not hasattr(transcription, "segments"):
        logger.warning(f"Transcription object missing 'segments' attribute. Type: {type(transcription)}")
        return ""

    logger.debug(f"Formatting {len(transcription.segments)} transcription segments")
    lines = []
    for seg in transcription.segments:
        timestamp = _format_timestamp(seg.start)
        speaker = seg.speaker or "Unknown"
        text = seg.text
        confidence_marker = ""

        # Add [LOW CONF] marker if confidence is below threshold
        if seg.confidence < 0.5:
            confidence_marker = " [LOW CONF]"

        line = f"{timestamp} {speaker}: {text}{confidence_marker}"
        lines.append(line)

    result = "\n".join(lines)
    logger.debug(f"Formatted transcript: {len(result)} characters, {len(lines)} lines")
    return result


def format_summary(summary_data: dict) -> str:
    """Format summary data as markdown.

    Args:
        summary_data: Summary data from pipeline state

    Returns:
        Formatted summary markdown
    """
    if isinstance(summary_data, str):
        summary_data = json.loads(summary_data) if summary_data else {}

    if not isinstance(summary_data, dict):
        return "### Summary\n\nNo summary available"

    summary_text = summary_data.get("summary", "No summary available")
    return f"### Summary\n\n{summary_text}"


def format_qa(qa_data: dict) -> str:
    """Format QA scores as markdown.

    Args:
        qa_data: QA scoring data from pipeline state

    Returns:
        Formatted QA markdown
    """
    qa_md = "### QA Scores\n\n"

    if isinstance(qa_data, str):
        try:
            qa_data = json.loads(qa_data)
        except json.JSONDecodeError:
            qa_data = {}

    if not isinstance(qa_data, dict) or not qa_data:
        qa_md += "No QA scores available"
        return qa_md

    # Handle Pydantic model case
    if hasattr(qa_data, 'model_dump'):
        qa_data = qa_data.model_dump()

    qa_md += f"- **Professionalism**: {qa_data.get('professionalism', 'N/A')}/5\n"
    qa_md += f"- **Empathy**: {qa_data.get('empathy', 'N/A')}/5\n"
    qa_md += f"- **Problem Resolution**: {qa_data.get('problem_resolution', 'N/A')}/5\n"
    qa_md += f"- **Compliance**: {qa_data.get('compliance', 'N/A')}/5\n"
    qa_md += f"- **Communication Clarity**: {qa_data.get('communication_clarity', 'N/A')}/5\n"
    qa_md += f"- **Overall Score**: {qa_data.get('overall_score', 'N/A')}/5\n"

    if qa_data.get("justification"):
        qa_md += f"- **Justification**: \n {qa_data.get('justification')}\n"

    return qa_md


def process_call(
    audio_data: Union[str, Tuple[int, np.ndarray]],
    caller_id: Optional[str] = None,
    department: Optional[str] = None,
) -> PipelineResult:
    """Process a call recording through the analysis pipeline.

    Args:
        audio_data: Either a filepath to an audio file or tuple of (sample_rate, audio_array)
        caller_id: Optional caller ID
        department: Optional department

    Returns:
        PipelineResult with transcript, summary, qa_scores, and report paths
    """
    if audio_data is None:
        return PipelineResult(
            transcript="",
            summary="",
            qa_scores="",
            pdf_path=None,
            json_path=None,
            error="No audio data provided",
        )

    try:
        # Write audio to temp WAV file and get bytes (handles both filepath and tuple formats)
        wav_path, wav_bytes = _write_audio_to_wav(audio_data)

        # Create initial state with audio input
        initial_state: PipeLineState = {
            "audio_input": AudioInput(
                audio_bytes=wav_bytes,
                filename=wav_path,
                caller_id=caller_id or None,
                department=department or None,
            ),
        }

        # Get compiled agent from app globals
        agent = get_agent()
        final_state = agent.invoke(initial_state)
        logger.info(f"Final state after processing: {final_state.get('state', 'unknown')}")
        logger.debug(f"Final state keys: {list(final_state.keys())}")
        logger.debug(f"Transcription in state: {'transcription' in final_state}, Type: {type(final_state.get('transcription'))}")

        # Check for errors from state
        error_message = final_state.get("error", "")
        if error_message:
            logger.error(f"Pipeline encountered error: {error_message}")
            return PipelineResult(
                transcript="",
                summary="",
                qa_scores="",
                pdf_path=None,
                json_path=None,
                error=f"❌ Error: {error_message}",
            )

        # Format transcript with timestamps and confidence markers
        transcript = _format_transcript(final_state)
        logger.info(f"Transcript formatted: {len(transcript) if transcript else 0} characters")

        # Format summary using helper function
        summary_data = final_state.get("summary", {})
        if hasattr(summary_data, "dict"):
            summary_data = summary_data.dict()
        summary_md = format_summary(summary_data)

        # Format QA scores using helper function
        qa_data = final_state.get("qa_score", {})
        if hasattr(qa_data, "dict"):
            qa_data = qa_data.dict()
        qa_md = format_qa(qa_data)

        # Get call report for PDF/JSON generation
        call_report = final_state.get("call_report")

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
                _add_temp_file(pdf_path)
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")

            try:
                json_content = generate_report_json({"call_report": call_report})
                json_file = tempfile.NamedTemporaryFile(
                    suffix=".json", mode="w", delete=False
                )
                json_file.write(json_content)
                json_file.close()
                json_path = json_file.name
                _add_temp_file(json_path)
            except Exception as e:
                logger.error(f"Error generating JSON: {e}")

        return PipelineResult(
            transcript=transcript,
            summary=summary_md,
            qa_scores=qa_md,
            pdf_path=pdf_path,
            json_path=json_path,
            error="",
        )

    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}", exc_info=True)
        error_msg = str(e).strip()
        # Truncate very long error messages (e.g., ffmpeg output)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return PipelineResult(
            transcript="",
            summary="",
            qa_scores="",
            pdf_path=None,
            json_path=None,
            error=f"❌ {error_msg}",
        )
