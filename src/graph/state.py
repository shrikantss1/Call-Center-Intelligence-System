
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    audio_bytes: bytes
    filename: str | None = None
    caller_id: str | None = None
    call_id: str | None = None
    department: str | None = None
    timestamp: datetime | None = None


class AudioProperties(BaseModel):
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int | None


class IntakeResult(BaseModel):
    is_valid: bool
    reason: str | None
    properties: AudioProperties | None


@dataclass
class PIIScanResult:
    pii_detected: bool
    affected_fields: list[str]

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class TranscriptionResult(BaseModel):
    segments: list[TranscriptionSegment]
    injection_detected: bool = False
    injection_reason: str | None = None
    call_id: str | None = None

class SummaryResult(BaseModel):
    summary: str = Field(..., description="The summarized text of the transcription.")
    is_valid: bool = True
    reason: str | None = None
    call_id: str | None = None

class QAScoringResult(BaseModel):
    professionalism: int = Field(..., ge=1, le=5, description="Agent professionalism score (1-5)")
    empathy: int = Field(..., ge=1, le=5, description="Agent empathy score (1-5)")
    problem_resolution: int = Field(..., ge=1, le=5, description="Problem resolution effectiveness (1-5)")
    compliance: int = Field(..., ge=1, le=5, description="Compliance adherence (1-5)")
    communication_clarity: int = Field(..., ge=1, le=5, description="Communication clarity (1-5)")
    overall_score: float = Field(..., ge=1.0, le=5.0, description="Computed overall score (overridden after LLM response)")
    justification: str = Field(..., description="Coaching-style justification with timestamp citations (MM:SS format)")
    compliance_flag: bool = Field(default=False, description="True if genuine procedural violations detected")
    compliance_details: str | None = Field(default=None, description="Details on compliance issues if flagged")
    is_valid: bool = True
    reason: str | None = None
    call_id: str | None = None

class CallReport(BaseModel):
    call_id: str
    timestamp: str
    audio_filename: str | None = None
    transcription: TranscriptionResult | None = None
    transcript_text: str | None = None
    summary: str | None = None
    qa_scores: QAScoringResult | None = None
    pii_scan: PIIScanResult | None = None
    status: str
    error: str | None = None

class PipeLineState(TypedDict, total=False):
    audio_input: AudioInput
    intake_result: IntakeResult
    transcription: TranscriptionResult | None = None
    summary: SummaryResult | None = None
    qa_score: QAScoringResult | None = None
    pii_scan: PIIScanResult | None = None
    call_report: CallReport | None = None
    error: str | None = None
    state: str | None = None



