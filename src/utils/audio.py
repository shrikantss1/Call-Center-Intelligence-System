import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from src.utils.config import get_logger
try:
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.mp4 import MP4
except ImportError:
    MP3 = FLAC = MP4 = None

# Audio file magic bytes signatures
AUDIO_SIGNATURES = {
    'mp3': [b'ID3', b'\xff\xfb', b'\xff\xfa', b'\xff\xe3', b'\xff\xe2'],
    'wav': [b'RIFF'],
    'flac': [b'fLaC'],
    'ogg': [b'OggS'],
    'm4a': [b'ftypisom', b'ftypmp42'],
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes
MAX_DURATION_SECONDS = 3600  # 1 hour in seconds
SUPPORTED_FORMATS = {'mp3', 'wav', 'flac', 'ogg', 'm4a'}

logger = get_logger("audio_utils")

class AudioValidationError(Exception):
    pass

@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


@dataclass
class AudioProperties:
    frame_count: int
    sample_rate: int
    channel_count: int
    duration_seconds: float


def validate_audio(file_contents: bytes, file_path: str) -> ValidationResult:
    """
    Validates if an audio file is supported, not empty, under 50 MB, and within max duration.
    Takes file contents and file path as input.
    Returns a ValidationResult with is_valid and error fields.
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.is_file():
        return ValidationResult(is_valid=False, error="File does not exist.")

    file_size = len(file_contents)
    if file_size == 0:
        return ValidationResult(is_valid=False, error="File is empty.")

    if file_size > MAX_FILE_SIZE:
        return ValidationResult(
            is_valid=False,
            error=f"File is too large ({file_size / (1024*1024):.2f} MB). Maximum allowed size is 50 MB."
        )

    detected_format = detect_audio_format(file_contents)

    if detected_format is None:
        return ValidationResult(
            is_valid=False,
            error="Unsupported or unrecognized audio format."
        )

    try:
        audio_properties = extract_audio_properties(file_contents, file_path)
        duration_validation = validate_audio_duration(audio_properties.duration_seconds)
        if not duration_validation.is_valid:
            return duration_validation
    except AudioValidationError as e:
        return ValidationResult(is_valid=False, error=str(e))

    return ValidationResult(is_valid=True)


def validate_audio_duration(duration_seconds: float) -> ValidationResult:
    """
    Validates if audio duration is within the maximum allowed duration.
    Args:
        duration_seconds: Duration of the audio file in seconds.
    Returns:
        ValidationResult with is_valid and error fields.
    """
    if duration_seconds < 0:
        return ValidationResult(is_valid=False, error="Audio duration cannot be negative.")

    if duration_seconds == 0:
        return ValidationResult(is_valid=False, error="Audio file has zero duration.")

    if duration_seconds > MAX_DURATION_SECONDS:
        minutes = int(MAX_DURATION_SECONDS / 60)
        return ValidationResult(
            is_valid=False,
            error=f"Audio file is too long ({duration_seconds:.2f} seconds / {duration_seconds/60:.2f} minutes). "
                  f"Maximum allowed duration is {MAX_DURATION_SECONDS} seconds ({minutes} minutes)."
        )

    return ValidationResult(is_valid=True)


def detect_audio_format(file_contents: bytes) -> Optional[str]:
    """Detect audio format from file contents' magic bytes (first 12 bytes)."""
    if not file_contents or len(file_contents) == 0:
        return None

    header = file_contents[:12]
    logger.info(f"Detecting audio format from header bytes: {header.hex()}")
    for format_name, signatures in AUDIO_SIGNATURES.items():
        if any(header.startswith(sig) for sig in signatures):
            return format_name
    return None


def extract_audio_properties(file_contents: bytes, filepath: str) -> AudioProperties:
    """
    Extracts audio properties (frame count, sample rate, channel count) from audio file contents.
    Uses wave module for WAV files and mutagen for other formats.
    Detects format from file's magic bytes, not file extension.
    Raises AudioValidationError on corrupt or unreadable files.
    """
    file_path = Path(filepath)

    # Detect format from magic bytes instead of file extension
    detected_format = detect_audio_format(file_contents)

    try:
        if detected_format == 'wav':
            return _extract_wav_properties(file_path)
        elif detected_format == 'mp3':
            return _extract_mp3_properties(file_path)
        elif detected_format == 'flac':
            return _extract_flac_properties(file_path)
        elif detected_format == 'm4a':
            return _extract_mp4_properties(file_path)
        else:
            raise AudioValidationError(f"Unsupported or unrecognized audio format")
    except Exception as e:
        raise AudioValidationError(f"Failed to extract audio properties: {str(e)}")


def _extract_wav_properties(file_path: Path) -> AudioProperties:
    """Extract properties from WAV file using wave module."""
    try:
        with wave.open(str(file_path), 'rb') as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            duration_seconds = frame_count / sample_rate if sample_rate > 0 else 0

            return AudioProperties(
                frame_count=frame_count,
                sample_rate=sample_rate,
                channel_count=channel_count,
                duration_seconds=duration_seconds
            )
    except wave.Error as e:
        raise AudioValidationError(f"Corrupt or unreadable WAV file: {str(e)}")


def _extract_mutagen_properties(audio_obj, get_frame_count) -> AudioProperties:
    """Extract properties from mutagen audio object."""
    sample_rate = audio_obj.info.sample_rate
    channel_count = audio_obj.info.channels
    duration_seconds = audio_obj.info.length
    frame_count = get_frame_count(audio_obj, sample_rate, duration_seconds)

    return AudioProperties(
        frame_count=frame_count,
        sample_rate=sample_rate,
        channel_count=channel_count,
        duration_seconds=duration_seconds
    )


def _extract_mp3_properties(file_path: Path) -> AudioProperties:
    """Extract properties from MP3 file using mutagen."""
    if MP3 is None:
        raise AudioValidationError("mutagen library not installed. Install with: pip install mutagen")

    try:
        audio = MP3(str(file_path))
        return _extract_mutagen_properties(
            audio,
            lambda _a, sr, d: int(d * sr) if sr > 0 else 0
        )
    except Exception as e:
        raise AudioValidationError(f"Corrupt or unreadable MP3 file: {str(e)}")


def _extract_flac_properties(file_path: Path) -> AudioProperties:
    """Extract properties from FLAC file using mutagen."""
    if FLAC is None:
        raise AudioValidationError("mutagen library not installed. Install with: pip install mutagen")

    try:
        audio = FLAC(str(file_path))
        return _extract_mutagen_properties(
            audio,
            lambda a, _sr, _d: a.info.total_samples
        )
    except Exception as e:
        raise AudioValidationError(f"Corrupt or unreadable FLAC file: {str(e)}")


def _extract_mp4_properties(file_path: Path) -> AudioProperties:
    """Extract properties from MP4/M4A file using mutagen."""
    if MP4 is None:
        raise AudioValidationError("mutagen library not installed. Install with: pip install mutagen")

    try:
        audio = MP4(str(file_path))
        return _extract_mutagen_properties(
            audio,
            lambda _a, sr, d: int(d * sr) if sr > 0 else 0
        )
    except Exception as e:
        raise AudioValidationError(f"Corrupt or unreadable MP4/M4A file: {str(e)}")
