import pytest
from pydantic import ValidationError

from src.graph.state import TranscriptionSegment, TranscriptionResult


def test_transcription_segment_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        TranscriptionSegment(start=0.0, end=1.0, text="hello", confidence=-0.1)

    with pytest.raises(ValidationError):
        TranscriptionSegment(start=0.0, end=1.0, text="hello", confidence=1.5)

    segment = TranscriptionSegment(start=0.0, end=1.0, text="hello", confidence=0.82)
    assert segment.confidence == 0.82

