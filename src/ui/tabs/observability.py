"""Observability tab for the Call Center Intelligence System UI."""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
from sqlalchemy import desc, func

from src.database.connection import get_engine, session_scope
from src.database.models import CallRecord, AuditLogEntry


def _get_pipeline_metrics() -> Dict:
    """Query database for pipeline metrics."""
    with session_scope(get_engine()) as session:
        total_calls = session.query(func.count(CallRecord.id)).scalar() or 0

        completed = (
            session.query(func.count(CallRecord.id))
            .filter(CallRecord.status.in_(["summary_and_qa_complete", "report_generated"]))
            .scalar()
            or 0
        )

        failed = (
            session.query(func.count(CallRecord.id))
            .filter(CallRecord.status.in_(["INTAKE_FAILED", "TRANSCRIPTION_FAILED", "SUMMARIZATION_FAILED", "QA_SCORING_FAILED"]))
            .scalar()
            or 0
        )

        flagged = (
            session.query(func.count(CallRecord.id))
            .filter(CallRecord.status == "FLAGGED_FOR_REVIEW")
            .scalar()
            or 0
        )

        success_rate = (completed / total_calls * 100) if total_calls > 0 else 0

        qa_records = session.query(CallRecord.qa_scores_json).filter(CallRecord.qa_scores_json.isnot(None)).all()
        avg_qa_score = 0.0
        compliance_flag_count = 0

        if qa_records:
            scores = []
            for row in qa_records:
                try:
                    qa_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(qa_data, dict):
                        if "overall_score" in qa_data:
                            scores.append(float(qa_data["overall_score"]))
                        if qa_data.get("compliance_flag", False):
                            compliance_flag_count += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            avg_qa_score = sum(scores) / len(scores) if scores else 0.0

        total_audit_events = session.query(func.count(AuditLogEntry.id)).scalar() or 0

    return {
        "total_calls": total_calls,
        "completed": completed,
        "failed": failed,
        "flagged": flagged,
        "success_rate": success_rate,
        "avg_qa_score": avg_qa_score,
        "compliance_flags": compliance_flag_count,
        "total_audit_events": total_audit_events,
    }


def _get_langsmith_status() -> str:
    """Check LangSmith integration status."""
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    trace_id = os.getenv("LANGSMITH_TRACE_ID", "").strip()
    langsmith_url = os.getenv("LANGSMITH_URL", "").strip()

    if langsmith_api_key and trace_id:
        status = "✅ **LangSmith Integration: Active**\n\n- API Key: Configured\n- Trace ID: Configured"
        if langsmith_url:
            status += f"\n- [Open LangSmith Dashboard]({langsmith_url})"
        return status
    elif langsmith_api_key:
        status = "⚠️ **LangSmith Integration: Partial**\n\n- API Key: Configured\n- Trace ID: Missing"
        if langsmith_url:
            status += f"\n- [Open LangSmith Dashboard]({langsmith_url})"
        return status
    else:
        return "❌ **LangSmith Integration: Inactive**\n\n- API Key: Not configured\n- Trace ID: Not configured"


def _get_audit_events() -> pd.DataFrame:
    """Query database for the 20 most recent audit events."""
    with session_scope(get_engine()) as session:
        audit_entries = (
            session.query(AuditLogEntry)
            .order_by(desc(AuditLogEntry.created_at))
            .limit(20)
            .all()
        )

        data = []
        for entry in audit_entries:
            timestamp = entry.created_at.isoformat() if entry.created_at else ""
            data.append({
                "Timestamp": timestamp,
                "Call ID": entry.call_id,
                "Action": entry.action,
                "Details": entry.details or "",
            })

        df = pd.DataFrame(data)
        return df if not df.empty else pd.DataFrame(columns=["Timestamp", "Call ID", "Action", "Details"])


def _refresh_metrics() -> Tuple[str, pd.DataFrame]:
    """Refresh both metrics and audit events."""
    metrics = _get_pipeline_metrics()
    metrics_md = f"""
### 📊 Pipeline Metrics

| Metric | Value |
|--------|-------|
| **Total Calls** | {metrics['total_calls']} |
| **Completed** | {metrics['completed']} |
| **Failed** | {metrics['failed']} |
| **Flagged for Review** | {metrics['flagged']} |
| **Success Rate** | {metrics['success_rate']:.1f}% |
| **Average QA Score** | {metrics['avg_qa_score']:.2f}/5.0 |
| **Compliance Flags** | {metrics['compliance_flags']} |
| **Total Audit Events** | {metrics['total_audit_events']} |
"""
    audit_df = _get_audit_events()
    return metrics_md, audit_df


def create_observability_tab() -> Dict:
    """
    Create the Observability tab with metrics, LangSmith status, and audit events.

    Must be called within a gr.Blocks() context.

    Returns:
        dict: Dictionary containing component references under "components" key,
              and a "refresh_handler" function to wire tab select events
    """
    gr.Markdown("## 🔍 Observability & Monitoring")

    metrics_md, initial_audit_df = _refresh_metrics()

    metrics_display = gr.Markdown(value=metrics_md)

    gr.Markdown("### 🔗 Integration Status")
    langsmith_status = gr.Markdown(value=_get_langsmith_status())

    gr.Markdown("### 📋 Recent Audit Events (Last 20)")
    audit_dataframe = gr.Dataframe(
        value=initial_audit_df,
        headers=["Timestamp", "Call ID", "Action", "Details"],
        type="pandas",
        interactive=False,
    )

    refresh_button = gr.Button("🔄 Refresh Now", variant="secondary")

    def _on_refresh():
        return _refresh_metrics()

    refresh_button.click(
        fn=_on_refresh,
        inputs=None,
        outputs=[metrics_display, audit_dataframe],
    )

    def wire_tab_select_refresh(tab_component):
        """
        Wire the tab select event to auto-refresh metrics and audit events.

        Usage in main app:
            tab_result = create_observability_tab()
            with gr.Tab("Observability") as obs_tab:
                # tab creation already done above
                pass
            tab_result["refresh_handler"](obs_tab)
        """
        tab_component.select(
            fn=_on_refresh,
            inputs=None,
            outputs=[metrics_display, audit_dataframe],
        )

    return {
        "components": {
            "metrics_display": metrics_display,
            "langsmith_status": langsmith_status,
            "audit_dataframe": audit_dataframe,
            "refresh_button": refresh_button,
        },
        "refresh_handler": wire_tab_select_refresh,
    }


if __name__ == "__main__":
    with gr.Blocks(title="Observability") as demo:
        gr.Markdown("# Observability")
        result = create_observability_tab()
    demo.launch()
