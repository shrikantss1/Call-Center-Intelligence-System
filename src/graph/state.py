
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

class QAScoringResult(BaseModel):
    professionalism: int = Field(..., ge=1, le=5, description="Agent professionalism score (1-5)")
    empathy: int = Field(..., ge=1, le=5, description="Agent empathy score (1-5)")
    problem_resolution: int = Field(..., ge=1, le=5, description="Problem resolution effectiveness (1-5)")
    compliance: int = Field(..., ge=1, le=5, description="Compliance adherence (1-5)")
    communication_clarity: int = Field(..., ge=1, le=5, description="Communication clarity (1-5)")
    overall_score: float = Field(..., ge=1.0, le=5.0, description="Computed overall score (overridden after LLM response)")
    justification: str = Field(..., description="Coaching-style justification with timestamp citations (MM:SS format)")
    compliance_flag: bool = Field(default=False, description="True if genuine procedural violations detected")
    compliance_details: Optional[str] = Field(default=None, description="Details on compliance issues if flagged")
    is_valid: bool = True
    reason: Optional[str] = None
    call_id: Optional[str] = None

class CallReport(BaseModel):
    call_id: str
    timestamp: str
    audio_filename: Optional[str] = None
    transcription: Optional[TranscriptionResult] = None
    transcript_text: Optional[str] = None
    summary: Optional[str] = None
    qa_scores: Optional[QAScoringResult] = None
    pii_scan: Optional[PIIScanResult] = None
    status: str
    error: Optional[str] = None

class PipeLineState(TypedDict, total=False):
    audio_input: AudioInput
    intake_result: IntakeResult
    transcription: Optional[TranscriptionResult] = None
    summary: Optional[SummaryResult] = None
    qa_score: Optional[QAScoringResult] = None
    pii_scan: Optional[PIIScanResult] = None
    call_report: Optional[CallReport] = None
    error: Optional[str] = None
    state: Optional[str] = None

    

