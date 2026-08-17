import pytest
import wave
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.utils.audio import (
    validate_audio,
    validate_audio_duration,
    detect_audio_format,
    extract_audio_properties,
    _extract_wav_properties,
    _extract_mp3_properties,
    _extract_flac_properties,
    _extract_mp4_properties,
    _extract_mutagen_properties,
    AudioValidationError,
    ValidationResult,
    AudioProperties,
    MAX_FILE_SIZE,
    MAX_DURATION_SECONDS,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_valid(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error is None

    def test_validation_result_invalid_with_error(self):
        error_msg = "Test error"
        result = ValidationResult(is_valid=False, error=error_msg)
        assert result.is_valid is False
        assert result.error == error_msg


class TestAudioProperties:
    """Test AudioProperties dataclass."""

    def test_audio_properties_creation(self):
        props = AudioProperties(
            frame_count=44100,
            sample_rate=44100,
            channel_count=2,
            duration_seconds=1.0,
        )
        assert props.frame_count == 44100
        assert props.sample_rate == 44100
        assert props.channel_count == 2
        assert props.duration_seconds == 1.0

    def test_audio_properties_mono(self):
        props = AudioProperties(
            frame_count=48000,
            sample_rate=48000,
            channel_count=1,
            duration_seconds=1.0,
        )
        assert props.channel_count == 1


class TestValidateAudioDuration:
    """Test audio duration validation."""

    def test_duration_within_limit(self):
        duration = 1800.0  # 30 minutes
        result = validate_audio_duration(duration)
        assert result.is_valid is True
        assert result.error is None

    def test_duration_exactly_at_limit(self):
        duration = MAX_DURATION_SECONDS  # Exactly 3600 seconds
        result = validate_audio_duration(duration)
        assert result.is_valid is True
        assert result.error is None

    def test_duration_exceeds_limit(self):
        duration = MAX_DURATION_SECONDS + 1.0  # 3601 seconds
        result = validate_audio_duration(duration)
        assert result.is_valid is False
        assert "too long" in result.error.lower()
        assert "3600" in result.error

    def test_duration_significantly_exceeds_limit(self):
        duration = 7200.0  # 2 hours
        result = validate_audio_duration(duration)
        assert result.is_valid is False
        assert "too long" in result.error.lower()

    def test_duration_zero(self):
        duration = 0.0
        result = validate_audio_duration(duration)
        assert result.is_valid is False
        assert "zero duration" in result.error.lower()

    def test_duration_negative(self):
        duration = -100.0
        result = validate_audio_duration(duration)
        assert result.is_valid is False
        assert "cannot be negative" in result.error.lower()

    def test_duration_very_short(self):
        duration = 0.5  # 0.5 seconds
        result = validate_audio_duration(duration)
        assert result.is_valid is True
        assert result.error is None

    def test_duration_one_second(self):
        duration = 1.0
        result = validate_audio_duration(duration)
        assert result.is_valid is True
        assert result.error is None

    def test_duration_near_limit(self):
        duration = 3599.9  # Just under 1 hour
        result = validate_audio_duration(duration)
        assert result.is_valid is True
        assert result.error is None

    def test_duration_error_message_format(self):
        duration = 7200.0  # 2 hours
        result = validate_audio_duration(duration)
        assert result.is_valid is False
        # Verify error message contains useful info
        assert "seconds" in result.error.lower() or "minutes" in result.error.lower()


class TestDetectAudioFormat:
    """Test audio format detection from magic bytes."""

    def test_detect_mp3_format(self):
        file_contents = b"ID3" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "mp3"

    def test_detect_mp3_format_ffb(self):
        file_contents = b"\xff\xfb" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "mp3"

    def test_detect_wav_format(self):
        file_contents = b"RIFF" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "wav"

    def test_detect_flac_format(self):
        file_contents = b"fLaC" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "flac"

    def test_detect_ogg_format(self):
        file_contents = b"OggS" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "ogg"

    def test_detect_m4a_format_isom(self):
        file_contents = b"ftypisom" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "m4a"

    def test_detect_m4a_format_mp42(self):
        file_contents = b"ftypmp42" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected == "m4a"

    def test_detect_unknown_format(self):
        file_contents = b"\x00\x01\x02\x03" + b"\x00" * 100
        detected = detect_audio_format(file_contents)
        assert detected is None

    def test_detect_format_empty_contents(self):
        detected = detect_audio_format(b"")
        assert detected is None

    def test_detect_format_none_contents(self):
        detected = detect_audio_format(None)
        assert detected is None


class TestValidateAudio:
    """Test audio file validation."""

    def test_validate_nonexistent_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            file_path = f.name
        Path(file_path).unlink()

        file_contents = b"ID3" + b"\x00" * 100
        result = validate_audio(file_contents, file_path)
        assert result.is_valid is False
        assert "does not exist" in result.error.lower()

    def test_validate_empty_contents(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            result = validate_audio(b"", file_path)
            assert result.is_valid is False
            assert "empty" in result.error.lower()
        finally:
            Path(file_path).unlink()

    def test_validate_file_too_large(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            large_contents = b"ID3" + b"\x00" * (MAX_FILE_SIZE + 100)
            result = validate_audio(large_contents, file_path)
            assert result.is_valid is False
            assert "too large" in result.error.lower()
        finally:
            Path(file_path).unlink()

    def test_validate_unsupported_format(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"\x00\x01\x02\x03")
            f.flush()
            file_path = f.name

        try:
            file_contents = b"\x00\x01\x02\x03"
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is False
            assert "unsupported" in result.error.lower()
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_valid_mp3(self, mock_extract):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            mock_extract.return_value = AudioProperties(
                frame_count=44100,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=1.0
            )
            file_contents = b"ID3" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_valid_wav(self, mock_extract):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"RIFF" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            mock_extract.return_value = AudioProperties(
                frame_count=44100,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=1.0
            )
            file_contents = b"RIFF" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_valid_flac(self, mock_extract):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            f.write(b"fLaC" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            mock_extract.return_value = AudioProperties(
                frame_count=44100,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=1.0
            )
            file_contents = b"fLaC" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_valid_m4a(self, mock_extract):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            f.write(b"ftypisom" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            mock_extract.return_value = AudioProperties(
                frame_count=44100,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=1.0
            )
            file_contents = b"ftypisom" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_audio_duration_exceeds_limit(self, mock_extract):
        """Test validation fails when audio duration exceeds MAX_DURATION_SECONDS."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            # Mock audio properties with duration exceeding limit
            mock_extract.return_value = AudioProperties(
                frame_count=158760000,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=3601.0  # Exceeds 3600 second limit
            )

            file_contents = b"ID3" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is False
            assert "too long" in result.error.lower()
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_audio_duration_at_limit(self, mock_extract):
        """Test validation passes when audio duration is exactly at MAX_DURATION_SECONDS."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            # Mock audio properties with duration at limit
            mock_extract.return_value = AudioProperties(
                frame_count=158760000,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=3600.0  # Exactly at limit
            )

            file_contents = b"ID3" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_audio_duration_under_limit(self, mock_extract):
        """Test validation passes when audio duration is under MAX_DURATION_SECONDS."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            # Mock audio properties with duration under limit
            mock_extract.return_value = AudioProperties(
                frame_count=44100000,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=1000.0  # Under limit
            )

            file_contents = b"ID3" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is True
            assert result.error is None
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_audio_zero_duration(self, mock_extract):
        """Test validation fails when audio duration is zero."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"RIFF" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            # Mock audio properties with zero duration
            mock_extract.return_value = AudioProperties(
                frame_count=0,
                sample_rate=44100,
                channel_count=2,
                duration_seconds=0.0  # Zero duration
            )

            file_contents = b"RIFF" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is False
            assert "zero duration" in result.error.lower()
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.extract_audio_properties")
    def test_validate_audio_extraction_error(self, mock_extract):
        """Test validation fails gracefully when audio properties extraction fails."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            f.write(b"fLaC" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            # Mock extraction to raise an error
            mock_extract.side_effect = AudioValidationError("Failed to extract properties")

            file_contents = b"fLaC" + b"\x00" * 100
            result = validate_audio(file_contents, file_path)
            assert result.is_valid is False
            assert "failed to extract" in result.error.lower()
        finally:
            Path(file_path).unlink()


class TestExtractWavProperties:
    """Test WAV properties extraction."""

    def test_extract_wav_properties_mono(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            # Create a simple WAV file
            with wave.open(f.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b"\x00\x00" * 44100)
            file_path = Path(f.name)

        try:
            props = _extract_wav_properties(file_path)
            assert props.channel_count == 1
            assert props.sample_rate == 44100
            assert props.frame_count == 44100
            assert props.duration_seconds == 1.0
        finally:
            file_path.unlink()

    def test_extract_wav_properties_stereo(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            with wave.open(f.name, "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(b"\x00\x00" * 48000 * 2)
            file_path = Path(f.name)

        try:
            props = _extract_wav_properties(file_path)
            assert props.channel_count == 2
            assert props.sample_rate == 48000
            assert props.duration_seconds == 1.0
        finally:
            file_path.unlink()

    def test_extract_wav_properties_zero_sample_rate(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            file_path = Path(f.name)

        try:
            with patch("wave.open") as mock_wave:
                mock_wav = MagicMock()
                mock_wav.getnchannels.return_value = 2
                mock_wav.getsampwidth.return_value = 2
                mock_wav.getframerate.return_value = 0
                mock_wav.getnframes.return_value = 100
                mock_wave.return_value.__enter__.return_value = mock_wav

                props = _extract_wav_properties(file_path)
                assert props.duration_seconds == 0.0
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_extract_wav_corrupt_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"not a valid wav file")
            f.flush()
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_wav_properties(file_path)
            assert "corrupt" in str(exc_info.value).lower()
        finally:
            file_path.unlink()


class TestExtractMutagenProperties:
    """Test mutagen properties extraction helper."""

    def test_extract_mutagen_properties(self):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.length = 120.5

        get_frame_count = lambda _a, sr, d: int(d * sr)

        props = _extract_mutagen_properties(mock_audio, get_frame_count)

        assert props.sample_rate == 44100
        assert props.channel_count == 2
        assert props.duration_seconds == 120.5
        assert props.frame_count == int(120.5 * 44100)

    def test_extract_mutagen_properties_with_total_samples(self):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 48000
        mock_audio.info.channels = 1
        mock_audio.info.length = 60.0
        mock_audio.info.total_samples = 2880000

        get_frame_count = lambda a, _sr, _d: a.info.total_samples

        props = _extract_mutagen_properties(mock_audio, get_frame_count)

        assert props.sample_rate == 48000
        assert props.channel_count == 1
        assert props.frame_count == 2880000
        assert props.duration_seconds == 60.0


class TestExtractMP3Properties:
    """Test MP3 properties extraction."""

    @patch("src.utils.audio.MP3")
    def test_extract_mp3_properties(self, mock_mp3_class):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.length = 180.0
        mock_mp3_class.return_value = mock_audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            file_path = Path(f.name)

        try:
            props = _extract_mp3_properties(file_path)
            assert props.sample_rate == 44100
            assert props.channel_count == 2
            assert props.duration_seconds == 180.0
            assert props.frame_count == int(180.0 * 44100)
        finally:
            file_path.unlink()

    @patch("src.utils.audio.MP3", None)
    def test_extract_mp3_mutagen_not_installed(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_mp3_properties(file_path)
            assert "mutagen" in str(exc_info.value).lower()
        finally:
            file_path.unlink()

    @patch("src.utils.audio.MP3")
    def test_extract_mp3_corrupt_file(self, mock_mp3_class):
        mock_mp3_class.side_effect = Exception("Corrupt MP3")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_mp3_properties(file_path)
            assert "corrupt" in str(exc_info.value).lower()
        finally:
            file_path.unlink()


class TestExtractFLACProperties:
    """Test FLAC properties extraction."""

    @patch("src.utils.audio.FLAC")
    def test_extract_flac_properties(self, mock_flac_class):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.length = 240.0
        mock_audio.info.total_samples = 10584000
        mock_flac_class.return_value = mock_audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            file_path = Path(f.name)

        try:
            props = _extract_flac_properties(file_path)
            assert props.sample_rate == 44100
            assert props.channel_count == 2
            assert props.duration_seconds == 240.0
            assert props.frame_count == 10584000
        finally:
            file_path.unlink()

    @patch("src.utils.audio.FLAC", None)
    def test_extract_flac_mutagen_not_installed(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_flac_properties(file_path)
            assert "mutagen" in str(exc_info.value).lower()
        finally:
            file_path.unlink()

    @patch("src.utils.audio.FLAC")
    def test_extract_flac_corrupt_file(self, mock_flac_class):
        mock_flac_class.side_effect = Exception("Corrupt FLAC")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_flac_properties(file_path)
            assert "corrupt" in str(exc_info.value).lower()
        finally:
            file_path.unlink()


class TestExtractMP4Properties:
    """Test MP4/M4A properties extraction."""

    @patch("src.utils.audio.MP4")
    def test_extract_mp4_properties(self, mock_mp4_class):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 48000
        mock_audio.info.channels = 2
        mock_audio.info.length = 300.0
        mock_mp4_class.return_value = mock_audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            file_path = Path(f.name)

        try:
            props = _extract_mp4_properties(file_path)
            assert props.sample_rate == 48000
            assert props.channel_count == 2
            assert props.duration_seconds == 300.0
            assert props.frame_count == int(300.0 * 48000)
        finally:
            file_path.unlink()

    @patch("src.utils.audio.MP4", None)
    def test_extract_mp4_mutagen_not_installed(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_mp4_properties(file_path)
            assert "mutagen" in str(exc_info.value).lower()
        finally:
            file_path.unlink()

    @patch("src.utils.audio.MP4")
    def test_extract_mp4_corrupt_file(self, mock_mp4_class):
        mock_mp4_class.side_effect = Exception("Corrupt MP4")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as f:
            file_path = Path(f.name)

        try:
            with pytest.raises(AudioValidationError) as exc_info:
                _extract_mp4_properties(file_path)
            assert "corrupt" in str(exc_info.value).lower()
        finally:
            file_path.unlink()


class TestRealAudioFiles:
    """Test with real audio files from tests/data."""

    def test_validate_real_mp3_file(self):
        """Test validation of real MP3 file: 1735404531.458927.mp3"""
        test_file_path = Path(__file__).parent / "data" / "1735404531.458927.mp3"

        assert test_file_path.exists(), f"Test file not found: {test_file_path}"

        with open(test_file_path, 'rb') as f:
            file_contents = f.read()

        result = validate_audio(file_contents, str(test_file_path))
        assert result.is_valid is True
        assert result.error is None

    @patch("src.utils.audio.MP3")
    def test_extract_properties_real_mp3_file(self, mock_mp3_class):
        """Test properties extraction from real MP3 file: 1735404531.458927.mp3"""
        test_file_path = Path(__file__).parent / "data" / "1735404531.458927.mp3"

        assert test_file_path.exists(), f"Test file not found: {test_file_path}"

        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.length = 120.0
        mock_mp3_class.return_value = mock_audio

        with open(test_file_path, 'rb') as f:
            file_contents = f.read()

        props = extract_audio_properties(file_contents, str(test_file_path))
        assert isinstance(props, AudioProperties)
        assert props.sample_rate == 44100
        assert props.channel_count == 2
        assert props.duration_seconds == 120.0


class TestExtractAudioProperties:
    """Test main audio properties extraction."""

    def test_extract_properties_wav_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            with wave.open(f.name, "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b"\x00\x00" * 44100 * 2)
            file_path = f.name

        try:
            with open(file_path, 'rb') as f:
                file_contents = f.read()
            props = extract_audio_properties(file_contents, file_path)
            assert isinstance(props, AudioProperties)
            assert props.sample_rate == 44100
            assert props.channel_count == 2
            assert props.duration_seconds == 1.0
        finally:
            Path(file_path).unlink()

    @patch("src.utils.audio.MP3")
    def test_extract_properties_mp3_file(self, mock_mp3_class):
        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.length = 60.0
        mock_mp3_class.return_value = mock_audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(b"ID3" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            file_contents = b"ID3" + b"\x00" * 100
            props = extract_audio_properties(file_contents, file_path)
            assert props.sample_rate == 44100
            assert props.channel_count == 2
        finally:
            Path(file_path).unlink()

    def test_extract_properties_unsupported_format(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".unknown") as f:
            f.write(b"\x00\x01\x02\x03" + b"\x00" * 100)
            f.flush()
            file_path = f.name

        try:
            file_contents = b"\x00\x01\x02\x03" + b"\x00" * 100
            with pytest.raises(AudioValidationError) as exc_info:
                extract_audio_properties(file_contents, file_path)
            assert "unsupported" in str(exc_info.value).lower()
        finally:
            Path(file_path).unlink()

    def test_extract_properties_corrupt_wav(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"RIFF" + b"\x00" * 10)
            f.flush()
            file_path = f.name

        try:
            file_contents = b"RIFF" + b"\x00" * 10
            with pytest.raises(AudioValidationError) as exc_info:
                extract_audio_properties(file_contents, file_path)
            assert "failed to extract" in str(exc_info.value).lower()
        finally:
            Path(file_path).unlink()
