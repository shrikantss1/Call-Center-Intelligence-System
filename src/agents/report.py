import io
import json
from dataclasses import asdict
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.database.connection import get_session
from src.database.models import CallRecord
from src.graph.state import CallReport, PipeLineState, QAScoringResult
from src.utils.config import get_logger

logger = get_logger("report")


def compile_report(state: PipeLineState) -> PipeLineState:
    """Assemble a report from all upstream pipeline results and update state.

    Args:
        state: The PipeLineState containing all processed results

    Returns:
        The updated PipeLineState with call_report field populated
    """

    audio_input = state.get("audio_input")
    call_id = audio_input.call_id if audio_input else "unknown"
    logger.info("Compiling report for call_id=%s", call_id)

    transcript_text = None
    transcription = None
    if state.get("transcription"):
        trans = state["transcription"]
        transcription = trans
        transcript_text = " ".join(seg.text for seg in trans.segments)

    summary = state.get("summary")
    if summary is not None:
        if hasattr(summary, "summary"):
            summary = summary.summary
        elif isinstance(summary, dict):
            summary = summary.get("summary")

    logger.info(
        "Report assembly inputs: call_id=%s, transcript_length=%s, summary_length=%s",
        call_id,
        len(transcript_text) if transcript_text else 0,
        len(summary) if summary else 0,
    )

    qa_scores = state.get("qa_score")
    if qa_scores is not None and isinstance(qa_scores, dict):
        nested = qa_scores.get("qa_score") if isinstance(qa_scores.get("qa_score"), dict) else qa_scores
        try:
            qa_scores = QAScoringResult.model_validate(nested)
        except Exception:
            qa_scores = None

    pii_scan = None
    if state.get("pii_scan"):
        pii_scan = state["pii_scan"]

    status = state.get("state", "completed")
    error = state.get("error")
    logger.info("Compiling report for call_id=%s with pipeline_status=%s", call_id, status)
    # if error:
    #     status = "failed"

    try:
        report = CallReport(
            call_id=call_id,
            timestamp=datetime.now(UTC).isoformat(),
            audio_filename="abc", #audio_input.filename if audio_input else None,
            transcription=transcription,
            transcript_text=transcript_text,
            summary=summary,
            qa_scores=qa_scores,
            pii_scan=pii_scan,
            status=status,
            error=error,
        )
        logger.info("Call report built successfully for call_id=%s with status=%s", call_id, report.status)
    except Exception:
        logger.exception("Failed to create CallReport for call_id=%s", call_id)
        raise

    state = {
        "call_report": report
    }
    logger.info("Prepared final report state for call_id=%s, status=%s", call_id, state["call_report"].status)
    return state


def persist_report(state: PipeLineState) -> PipeLineState:
    """Write a CallRecord row to the database using compiled report data.

    Args:
        state: The PipeLineState containing all processed results

    Returns:
        The updated state
    """

    logger.info("Persisting report for call_id=%s", state.get("call_report").call_id if state.get("call_report") else "unknown")
    try:
        original_state = state.get("state")
        state = compile_report(state)
        report = state["call_report"]
        logger.info("Writing report to database for call_id=%s with status=%s", report.call_id, report.status)
        session = get_session()
        try:
            call_record = CallRecord(
                call_id=report.call_id,
                status=report.status,
                audio_filename=report.audio_filename,
                transcript_text=report.transcript_text,
                summary_json=json.dumps({"summary": report.summary}) if report.summary else None,
                qa_scores_json=report.qa_scores.model_dump_json() if report.qa_scores else None,
                report_json=report.model_dump_json(),
                processed_at=datetime.now(UTC),
            )
            session.add(call_record)
            session.commit()
            if original_state in ("intake_failed", "error", "flagged_for_review", "supervisor_review"):
                state["state"] = original_state
            else:
                state["state"] = "persisted"
            logger.info("Report persisted successfully for call_id=%s", report.call_id)
        finally:
            session.close()

        return state

    except Exception as e:
        state["state"] = "persistence_failed"
        state["error"] = str(e)
        logger.exception("Report persistence failed for call_id=%s", state.get("call_report").call_id if state.get("call_report") else "unknown")
        return state


def generate_report_pdf(state: PipeLineState) -> bytes:
    """Generate a PDF report from the call report state.

    Args:
        state: The PipeLineState containing the compiled call report

    Returns:
        PDF document as bytes
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
        spaceBefore=8
    )

    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
    )
    call_report = state.get("call_report")
    if not call_report:
        story.append(Paragraph("Error: No call report found", styles['Normal']))
        doc.build(story)
        return buffer.getvalue()

    # Title
    story.append(Paragraph(f"Call Report: {call_report.call_id}", title_style))
    story.append(Spacer(1, 12))

    # Summary Section
    story.append(Paragraph("Summary", section_style))
    summary_data = [
        ["Call ID", Paragraph(str(call_report.call_id or "N/A"), cell_style)],
        ["Timestamp", Paragraph(str(call_report.timestamp or "N/A"), cell_style)],
        ["Status", Paragraph((call_report.status or "unknown").upper(), cell_style)],
    ]

    if call_report.summary:
        summary = call_report.summary
        summary_data.append(["Summary", Paragraph(call_report.summary, cell_style)])

    summary_table = Table(summary_data, colWidths=[1.5*inch, 4.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EBF4FF")),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # QA Scores Section
    if call_report.qa_scores:
        story.append(Paragraph("QA Scores by Dimension", section_style))

        qa_data = [["Dimension", "Score", "Status"]]
        qa_dict = call_report.qa_scores.model_dump() if hasattr(call_report.qa_scores, 'model_dump') else call_report.qa_scores

        for key, value in qa_dict.items():
            if key in ['professionalism', 'empathy', 'problem_resolution', 'compliance', 'communication_clarity', 'overall_score'] and value is not None:
                try:
                    score_val = float(value) if isinstance(value, (int, float)) else 0
                    status = "PASS" if score_val >= 3 else "FAIL"
                    qa_data.append([
                        key.replace('_', ' ').title(),
                        f"{score_val:.1f}",
                        status
                    ])
                except (ValueError, TypeError):
                    continue

        if len(qa_data) > 1:
            qa_table = Table(qa_data, colWidths=[2*inch, 1.5*inch, 1*inch])
            qa_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(qa_table)
            story.append(Spacer(1, 12))

        if qa_dict.get("justification"):
            story.append(Paragraph("Justification", section_style))
            story.append(Paragraph(qa_dict["justification"], styles['Normal']))
            story.append(Spacer(1, 12))
    # Compliance & Security Section
    if call_report.pii_scan:
        story.append(Paragraph("Compliance & Security Flags", section_style))

        pii_dict = asdict(call_report.pii_scan) if hasattr(call_report.pii_scan, '__dataclass_fields__') else (call_report.pii_scan.model_dump() if hasattr(call_report.pii_scan, 'model_dump') else {})

        compliance_data = [["Check", "Status"]]
        for key, value in pii_dict.items():
            if isinstance(value, bool):
                status = "🚩 FLAGGED" if value else "✓ CLEAR"
                compliance_data.append([key.replace('_', ' ').title(), status])
            elif isinstance(value, list) and value:
                compliance_data.append([key.replace('_', ' ').title(), ", ".join(map(str, value))])

        if len(compliance_data) > 1:
            compliance_table = Table(compliance_data, colWidths=[3*inch, 2*inch])
            compliance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#742A2A")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#FEF5F5")]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(compliance_table)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_report_json(state: PipeLineState) -> str:
    """Generate a JSON representation of the call report.

    Args:
        state: The PipeLineState containing the compiled call report

    Returns:
        JSON string with the report data, formatted with 2-space indentation
    """
    call_report = state.get("call_report")
    if not call_report:
        logger.warning("No call report found in state for JSON export")
        return "{}"

    return call_report.model_dump_json(indent=2)
