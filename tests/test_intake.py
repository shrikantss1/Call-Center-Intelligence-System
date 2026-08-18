import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.agents.intake import run_intake, scan_metadata_for_pii
from src.graph.state import AudioInput, AudioProperties, PipeLineState, PIIScanResult
from src.utils.audio import ValidationResult


class TestScanMetadataForPII:
    """Test PII scanning in metadata fields."""

    def test_no_pii_detected_with_empty_metadata(self):
        """Test that empty metadata returns no PII detected."""
        result = scan_metadata_for_pii({})
        assert result.pii_detected is False
        assert result.affected_fields == []

    def test_no_pii_detected_with_none_metadata(self):
        """Test that None metadata returns no PII detected."""
        result = scan_metadata_for_pii(None)
        assert result.pii_detected is False
        assert result.affected_fields == []

    def test_ssn_detected_in_caller_id(self):
        """Test that SSN pattern in caller_id is detected."""
        metadata = {
            "caller_id": "John Doe 123-45-6789",
            "department": "Sales"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "caller_id" in result.affected_fields

    def test_ssn_detected_in_department(self):
        """Test that SSN pattern in department is detected."""
        metadata = {
            "caller_id": "John Doe",
            "department": "Sales 123-45-6789"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "department" in result.affected_fields

    def test_credit_card_detected_in_caller_id(self):
        """Test that credit card pattern in caller_id is detected."""
        metadata = {
            "caller_id": "1234 5678 9012 3456",
            "department": "Support"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "caller_id" in result.affected_fields

    def test_email_detected_in_caller_id(self):
        """Test that email pattern in caller_id is detected."""
        metadata = {
            "caller_id": "contact: john.doe@example.com",
            "department": "Support"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "caller_id" in result.affected_fields

    def test_phone_number_detected_in_caller_id(self):
        """Test that phone number pattern in caller_id is detected."""
        metadata = {
            "caller_id": "Call me at 123-456-7890",
            "department": "Support"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "caller_id" in result.affected_fields

    def test_phone_number_without_formatting_detected(self):
        """Test that unformatted phone number is detected."""
        metadata = {
            "caller_id": "Contact 1234567890",
            "department": "Support"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert "caller_id" in result.affected_fields

    def test_no_pii_detected_with_valid_metadata(self):
        """Test that valid metadata without PII returns no detection."""
        metadata = {
            "caller_id": "John Smith",
            "department": "Sales"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is False
        assert result.affected_fields == []

    def test_pii_detected_in_both_fields(self):
        """Test that PII detected in both caller_id and department."""
        metadata = {
            "caller_id": "123-45-6789",
            "department": "dept@example.com"
        }
        result = scan_metadata_for_pii(metadata)
        assert result.pii_detected is True
        assert len(result.affected_fields) == 2
        assert "caller_id" in result.affected_fields
        assert "department" in result.affected_fields


class TestRunIntakePositiveCase:
    """Test positive case for intake processing."""

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_valid_audio_intake_success(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that valid audio passes all validation and returns valid=True."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 120.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)
        mock_pii_scan.return_value = PIIScanResult(pii_detected=False, affected_fields=[])

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="test.mp3",
            caller_id="John Smith",
            department="Sales"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is True
        assert result_state["intake_result"]["reason"] is None
        assert result_state["intake_result"]["properties"] == mock_audio_props
        assert result_state["pii_scan"].pii_detected is False

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_valid_audio_intake_with_uuid_generation(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that caller_id is generated when not provided."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 60.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)
        mock_pii_scan.return_value = PIIScanResult(pii_detected=False, affected_fields=[])

        # Create input state without caller_id
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="test.wav",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is True
        assert result_state["pii_scan"].pii_detected is False


class TestRunIntakeEmptyAudio:
    """Test intake with empty audio bytes."""

    @patch('src.agents.intake.validate_audio')
    def test_empty_audio_bytes_validation_fails(self, mock_validate_audio):
        """Test that empty audio bytes are rejected."""
        # Setup mock to return validation failure
        mock_validate_audio.return_value = ValidationResult(
            is_valid=False,
            error="File is empty."
        )

        # Create input state with empty bytes
        audio_input = AudioInput(
            audio_bytes=b"",
            filename="empty.mp3",
            caller_id="Test Caller",
            department="Sales"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert result_state["intake_result"]["reason"] == "File is empty."
        assert result_state["intake_result"]["properties"] is None
        assert result_state["pii_scan"].pii_detected is False


class TestRunIntakeUnsupportedFormat:
    """Test intake with unsupported audio format."""

    @patch('src.agents.intake.validate_audio')
    def test_unsupported_audio_format_with_invalid_bytes(self, mock_validate_audio):
        """Test that audio with unsupported format bytes is rejected."""
        # Setup mock to return unsupported format error
        mock_validate_audio.return_value = ValidationResult(
            is_valid=False,
            error="Unsupported or unrecognized audio format."
        )

        # Create input state with invalid audio bytes (first 12 bytes don't match any supported format)
        invalid_audio_bytes = b"INVALID_DATA_NOT_SUPPORTED_FORMAT_BYTES"
        audio_input = AudioInput(
            audio_bytes=invalid_audio_bytes,
            filename="invalid.xyz",
            caller_id="Test Caller",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert result_state["intake_result"]["reason"] == "Unsupported or unrecognized audio format."
        assert result_state["intake_result"]["properties"] is None
        assert result_state["pii_scan"].pii_detected is False

    @patch('src.agents.intake.validate_audio')
    def test_multiple_unsupported_formats(self, mock_validate_audio):
        """Test various unsupported format signatures."""
        unsupported_signatures = [
            b"RANDOM_BYTES_12345678901234567",  # Random bytes
            b"MIDI_DATA_HERE",  # MIDI signature
            b"GIF89a_INVALID_AUDIO",  # GIF format (image, not audio)
        ]

        for invalid_bytes in unsupported_signatures:
            mock_validate_audio.return_value = ValidationResult(
                is_valid=False,
                error="Unsupported or unrecognized audio format."
            )

            audio_input = AudioInput(
                audio_bytes=invalid_bytes,
                filename=f"invalid_{unsupported_signatures.index(invalid_bytes)}.bin",
                caller_id="Test",
                department="Support"
            )
            state: PipeLineState = {
                "audio_input": audio_input,
            }

            result_state = run_intake(state)

            assert result_state["intake_result"]["is_valid"] is False
            assert "Unsupported" in result_state["intake_result"]["reason"]


class TestRunIntakePIIValidation:
    """Test PII detection during intake processing."""

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_pii_detected_in_caller_id(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that PII in caller_id prevents valid intake."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 30.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)

        # Mock PII scan to detect PII
        mock_pii_scan.return_value = PIIScanResult(
            pii_detected=True,
            affected_fields=["caller_id"]
        )

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="test.mp3",
            caller_id="SSN: 123-45-6789",
            department="Sales"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
            "intake_result": {"is_valid": True, "reason": None, "properties": None}
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert "PII detected" in result_state["intake_result"]["reason"]
        assert "caller_id" in result_state["intake_result"]["reason"]
        assert result_state["pii_scan"].pii_detected is True
        assert "caller_id" in result_state["pii_scan"].affected_fields

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_pii_detected_in_department(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that PII in department prevents valid intake."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 45.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)

        # Mock PII scan to detect PII in department
        mock_pii_scan.return_value = PIIScanResult(
            pii_detected=True,
            affected_fields=["department"]
        )

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="test.wav",
            caller_id="John Doe",
            department="Support 555-123-4567"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
            "intake_result": {"is_valid": True, "reason": None, "properties": None}
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert "PII detected" in result_state["intake_result"]["reason"]
        assert "department" in result_state["intake_result"]["reason"]
        assert result_state["pii_scan"].pii_detected is True

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_pii_detected_in_multiple_fields(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that PII in multiple fields is properly detected."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 60.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)

        # Mock PII scan to detect PII in both fields
        mock_pii_scan.return_value = PIIScanResult(
            pii_detected=True,
            affected_fields=["caller_id", "department"]
        )

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="test.flac",
            caller_id="john@example.com",
            department="dept@company.com"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
            "intake_result": {"is_valid": True, "reason": None, "properties": None}
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert result_state["pii_scan"].pii_detected is True
        assert len(result_state["pii_scan"].affected_fields) == 2


class TestRunIntakeDurationValidation:
    """Test audio duration validation during intake."""

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    def test_audio_duration_exceeds_maximum(self, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that audio exceeding maximum duration is rejected."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 4000.0  # Exceeds 3600 second max
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(
            is_valid=False,
            error="Audio file is too long (4000.00 seconds / 66.67 minutes). Maximum allowed duration is 3600 seconds (60 minutes)."
        )

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"valid_audio_data",
            filename="too_long.mp3",
            caller_id="Test",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert "too long" in result_state["intake_result"]["reason"]
        assert result_state["pii_scan"].pii_detected is False

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    def test_audio_duration_zero(self, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test that audio with zero duration is rejected."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 0.0
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(
            is_valid=False,
            error="Audio file has zero duration."
        )

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"",
            filename="zero_duration.wav",
            caller_id="Test",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is False
        assert "zero duration" in result_state["intake_result"]["reason"]


class TestRunIntakeEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_minimum_valid_audio_duration(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test audio with minimum valid duration (> 0 seconds)."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 0.1  # Very short but valid
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)
        mock_pii_scan.return_value = PIIScanResult(pii_detected=False, affected_fields=[])

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"short_audio",
            filename="short.mp3",
            caller_id="Test",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is True

    @patch('src.agents.intake.validate_audio')
    @patch('src.agents.intake.extract_audio_properties')
    @patch('src.agents.intake.validate_audio_duration')
    @patch('src.agents.intake.scan_metadata_for_pii')
    def test_maximum_valid_audio_duration(self, mock_pii_scan, mock_duration_validate, mock_extract_props, mock_validate_audio):
        """Test audio at maximum valid duration (3600 seconds / 1 hour)."""
        # Setup mocks
        mock_validate_audio.return_value = ValidationResult(is_valid=True)

        mock_audio_props = MagicMock()
        mock_audio_props.duration_seconds = 3600.0  # Exactly 1 hour
        mock_extract_props.return_value = mock_audio_props

        mock_duration_validate.return_value = ValidationResult(is_valid=True)
        mock_pii_scan.return_value = PIIScanResult(pii_detected=False, affected_fields=[])

        # Create input state
        audio_input = AudioInput(
            audio_bytes=b"long_audio_data" * 1000,
            filename="max_duration.mp3",
            caller_id="Test",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        # Run intake
        result_state = run_intake(state)

        # Assertions
        assert result_state["intake_result"]["is_valid"] is True

    @patch('src.agents.intake.validate_audio')
    def test_file_size_exceeds_maximum(self, mock_validate_audio):
        """Test audio file that exceeds maximum file size."""
        mock_validate_audio.return_value = ValidationResult(
            is_valid=False,
            error="File is too large (51.00 MB). Maximum allowed size is 50 MB."
        )

        # Create large audio bytes (simulate 51 MB)
        large_audio = b"x" * (51 * 1024 * 1024)

        audio_input = AudioInput(
            audio_bytes=large_audio,
            filename="large_file.mp3",
            caller_id="Test",
            department="Support"
        )
        state: PipeLineState = {
            "audio_input": audio_input,
        }

        result_state = run_intake(state)

        assert result_state["intake_result"]["is_valid"] is False
        assert "too large" in result_state["intake_result"]["reason"]
