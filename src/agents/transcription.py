import os
import sys
import re
import hashlib
import json
import sqlite3
from pathlib import Path
from io import BytesIO

from faster_whisper import WhisperModel
import torch
from src.graph.state import PipeLineState, TranscriptionResult, TranscriptionSegment
from src.utils.config import get_logger

logger = get_logger("transcription")

# define module level global _model and _model_size for singleton
_model = None
_model_size = None

# Database cache path
_CACHE_DB_PATH = os.getenv("TRANSCRIPTION_CACHE_DB", "data/agent.db")


def _get_whisper_model(model_size: str = "small") -> WhisperModel:
    global _model, _model_size
    if _model is None or _model_size != model_size:
        logger.info(f"Loading Whisper model of size '{model_size}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_size = model_size
        logger.info(f"Whisper model of size '{model_size}' loaded.")
    return _model

model = _get_whisper_model(os.getenv("WHISPER_MODEL_SIZE", "small"))


def _compute_audio_hash(audio_bytes: bytes) -> str:
    """Compute SHA-256 hash of audio bytes, reading in 8 KB chunks."""
    sha256_hash = hashlib.sha256()
    for chunk in _iter_chunks(audio_bytes, chunk_size=8192):
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _iter_chunks(data: bytes, chunk_size: int):
    """Yield successive chunks from bytes."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def _check_cache(audio_hash: str) -> TranscriptionResult | None:
    """Query transcription_cache table by hash. Returns TranscriptionResult if found."""
    try:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transcription FROM transcription_cache WHERE audio_hash = ?",
            (audio_hash,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            transcription_json = json.loads(row[0])
            segments = [
                TranscriptionSegment(**seg) for seg in transcription_json["segments"]
            ]
            logger.info(f"Cache hit for audio hash: {audio_hash}")
            return TranscriptionResult(segments=segments)
        return None
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def _save_cache(audio_hash: str, caller_id: str | None, filename: str, transcription: TranscriptionResult) -> None:
    """Insert transcription result into cache."""
    try:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        cursor = conn.cursor()
        transcription_json = json.dumps(transcription.model_dump())
        cursor.execute(
            """INSERT INTO transcription_cache (audio_hash, caller_id, filename, transcription)
               VALUES (?, ?, ?, ?)""",
            (audio_hash, caller_id, filename, transcription_json)
        )
        conn.commit()
        conn.close()
        logger.info(f"Cached transcription for audio hash: {audio_hash}")
    except sqlite3.IntegrityError:
        logger.debug(f"Audio hash already in cache: {audio_hash}")
    except Exception as e:
        logger.warning(f"Failed to cache transcription: {e}")


def _clean_transcript_text(text: str) -> str:
    """
    Clean transcript text by removing artifacts and normalizing patterns.

    Removes:
    - [BLANK_AUDIO] tags
    - Non-speech labels ([music], [applause], etc.)
    - YouTube-style footers ("thanks for watching", etc.)
    - Four or more repeated dots
    - Collapsed repeated phrases/words
    """
    if not text:
        return text

    # Remove [BLANK_AUDIO] tags
    text = re.sub(r'\[BLANK_AUDIO\]', '', text, flags=re.IGNORECASE)

    # Remove non-speech labels in brackets (music, applause, silence, etc.)
    text = re.sub(
        r'\[(?:music|applause|laughter|silence|background noise|noise|static|sound effect|crosstalk|pause|breathing|laugh)\]',
        '', text, flags=re.IGNORECASE
    )

    # Remove YouTube-style footers
    footer_patterns = [
        r'thanks for watching',
        r'don\'t forget to subscribe',
        r'like and subscribe',
        r'please subscribe',
        r'hit the subscribe button',
    ]
    for pattern in footer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Replace four or more dots with a single dot
    text = re.sub(r'\.{4,}', '.', text)

    # Collapse repeated word sequences
    words = text.split()
    i = 0
    cleaned = []

    while i < len(words):
        found_repeat = False
        # Try different sequence lengths (up to 4 words, and only if at least 2 repeats exist)
        for seq_len in range(min(4, (len(words) - i) // 2), 0, -1):
            if i + seq_len * 2 <= len(words):
                seq = [w.lower() for w in words[i:i + seq_len]]
                next_seq = [w.lower() for w in words[i + seq_len:i + seq_len * 2]]

                if seq == next_seq:
                    # Found repeating sequence, add once and skip repeats
                    cleaned.extend(words[i:i + seq_len])
                    i += seq_len
                    # Skip additional repeats of the same sequence
                    while i + seq_len <= len(words) and [w.lower() for w in words[i:i + seq_len]] == seq:
                        i += seq_len
                    found_repeat = True
                    break

        if not found_repeat:
            cleaned.append(words[i])
            i += 1

    text = ' '.join(cleaned)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Patterns that strongly indicate agent speech (first speaker)
_AGENT_PATTERNS = re.compile(
    r"(?i)(thank you for calling|how (?:can|may) I (?:help|assist)|"
    r"my name is|this call (?:may|will) be|for quality (?:and|assurance)|"
    r"is there anything else|have a (?:great|good|wonderful) day)"
)
# Patterns that strongly indicate customer speech
_CUSTOMER_PATTERNS = re.compile(
    r"(?i)(I(?:'m| am) calling (?:about|because|to)|I (?:need|want|have a)|"
    r"my account|my order|my bill|can you help|I was charged)"
)

class SpeakerDiarizer:
    """Heuristic speaker detection using gaps, questions, and content patterns."""

    def assign_speakers(self, segments: list[dict] | object) -> list[str]:
        segments = list(segments)
        labels = ["Agent", "Customer"]
        assignments: list[str] = []
        current = 0  # 0 = Agent, 1 = Customer

        for i, seg in enumerate(segments):
            text = seg.text.strip() if seg.text else ""

            if i == 0:
                # First segment: check if it sounds like an agent greeting
                if _AGENT_PATTERNS.search(text):
                    current = 0
                elif _CUSTOMER_PATTERNS.search(text):
                    current = 1
                # else default to Agent (call center convention)
            else:
                gap = seg.start - segments[i - 1].end
                prev_text = segments[i - 1].text.strip() if segments[i - 1].text else ""

                # Content-based: strong signal overrides gap heuristic
                if _AGENT_PATTERNS.search(text) and current != 0:
                    current = 0
                elif _CUSTOMER_PATTERNS.search(text) and current != 1:
                    current = 1
                # Gap-based: speaker likely changed
                elif gap > 1.2:
                    current = 1 - current
                # Question followed by answer = speaker change
                elif prev_text.endswith("?"):
                    current = 1 - current
                # Short affirmation after long segment = different speaker
                elif len(text.split()) <= 3 and len(prev_text.split()) > 10:
                    current = 1 - current

            assignments.append(labels[current])
        return assignments


_diarizer = SpeakerDiarizer()

def _get_diarizer() -> SpeakerDiarizer:
    return _diarizer


def transcribe_audio(state: PipeLineState) -> PipeLineState:
    """
    Transcribe the audio input in the given state using the Whisper model.
    Uses cache to avoid redundant transcriptions for identical audio.

    Args:
        state (PipeLineState): The current state of the pipeline.
    """
    if "audio_input" not in state or state["audio_input"] is None:
        logger.error("No audio input found in the state.")
        return state

    audio_input = state["audio_input"]
    audio_bytes = audio_input.audio_bytes
    filename = audio_input.filename
    caller_id = audio_input.caller_id

    # Compute hash and check cache
    audio_hash = _compute_audio_hash(audio_bytes)
    cached_result = _check_cache(audio_hash)
    if cached_result:
        state["transcription"] = cached_result
        return state

    logger.info(f"Transcribing audio file: {filename}")
    audio_stream = BytesIO(audio_bytes)
    segments, info = model.transcribe(
        audio_stream,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=False)

    diarizer = _get_diarizer()
    segments = list(segments)
    speakers = diarizer.assign_speakers(segments)

    transcription_segments = []
    for i, seg in enumerate(segments):
        logprob_conf = max(0, min(1, 1 + seg.avg_logprob))
        speech_conf = 1 - seg.no_speech_prob
        transcription_segments.append(TranscriptionSegment(
            start=seg.start,
            end=seg.end,
            speaker=(speakers[i] if i < len(speakers) else "Unknown"),
            text=_clean_transcript_text(seg.text),
            confidence=round(logprob_conf * 0.7 + speech_conf * 0.3, 4)
        ))

    transcription_result = TranscriptionResult(segments=transcription_segments)
    _save_cache(audio_hash, caller_id, filename, transcription_result)
    state["transcription"] = transcription_result
    return state
