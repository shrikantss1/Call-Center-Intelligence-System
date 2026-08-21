
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    audio_bytes: bytes
    filename: str
    caller_id: Optional[str] = None
    call_id: Optional[str] = None
    department: Optional[str] = None
    timestamp: Optional[datetime] = None


class AudioProperties(BaseModel):
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]


class IntakeResult(BaseModel):
    is_valid: bool
    reason: Optional[str]
    properties: Optional[AudioProperties]


@dataclass
class PIIScanResult:
    pii_detected: bool
    affected_fields: List[str]

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class TranscriptionResult(BaseModel):
    segments: List[TranscriptionSegment]
    injection_detected: bool = False
    injection_reason: Optional[str] = None
    call_id: Optional[str] = None

class SummaryResult(BaseModel):
    summary: str = Field(..., description="The summarized text of the transcription.")
    is_valid: bool = True
    reason: Optional[str] = None
    call_id: Optional[str] = None

class PipeLineState(TypedDict, total=False):
    audio_input: AudioInput
    intake_result: IntakeResult
    transcription: Optional[TranscriptionResult] = None
    summary: Optional[SummaryResult] = None
    qa_score: Optional[float] = None
    pii_scan: PIIScanResult = None
    error: Optional[str] = None
    state: Optional[str] = None

    

