"""Analyze Call tab for the Call Center Intelligence System UI."""

import gradio as gr

from src.services.pipeline import process_call


def _show_status() -> gr.update:
    """Show the processing status message."""
    return gr.update(
        value=get_processing_status_message(estimated_duration=120.0),
        visible=True,
    )


def _hide_status() -> gr.update:
    """Hide the processing status message."""
    return gr.update(visible=False)


def _process_call_with_error_display(audio_data, caller_id, department):
    """Wrapper to process call and format error display."""
    result = process_call(audio_data, caller_id, department)

    # Update error display
    error_display = gr.update(value=result.error, visible=bool(result.error))

    # Update file downloads with visibility based on availability
    pdf_update = gr.update(
        value=result.pdf_path,
        visible=bool(result.pdf_path)
    )
    json_update = gr.update(
        value=result.json_path,
        visible=bool(result.json_path)
    )

    return (
        result.transcript,
        result.summary,
        result.qa_scores,
        pdf_update,
        json_update,
        error_display
    )


def create_analyze_tab() -> dict:
    """
    Create the Analyze Call tab with audio input, metadata, and result display.

    Must be called within a gr.Blocks() context.

    Returns:
        dict: Dictionary containing component references under "components" key
    """
    # Audio Input Section
    gr.Markdown("### Audio Input")
    audio_input = gr.Audio(
        type="filepath",
        sources=["upload", "microphone"],
        label="Upload Audio or Record",
    )

    # Metadata Section
    gr.Markdown("### Call Metadata (Optional)")
    with gr.Row():
        caller_id = gr.Textbox(
            label="Caller ID",
            placeholder="Enter caller ID",
            scale=1,
        )
        department = gr.Textbox(
            label="Department",
            placeholder="Enter department",
            scale=1,
        )

    # Analyze Button
    analyze_button = gr.Button(
        "Analyze Call",
        variant="primary",
    )

    # Status Indicator (Initially Hidden)
    status_markdown = gr.Markdown(
        visible=False,
        elem_id="status-indicator",
    )

    # Error Display Section
    error_display = gr.Textbox(
        label="Error",
        lines=4,
        max_lines=10,
        interactive=False,
        visible=False,
        elem_id="error-display",
    )

    # Transcription Section
    gr.Markdown("### Transcription")
    transcript_output = gr.Textbox(
        label="Transcript",
        lines=15,
        max_lines=30,
        interactive=False,
    )

    # Results Section
    gr.Markdown("### Analysis Results")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Summary")
            summary_markdown = gr.Markdown(
                value="",
            )
        with gr.Column(scale=1):
            gr.Markdown("#### QA Scoring")
            qa_markdown = gr.Markdown(
                value="",
            )

    # Download Section
    gr.Markdown("### Download Results")
    with gr.Row():
        pdf_download = gr.File(
            label="📄 Download PDF Report",
            interactive=False,
            visible=False,
            type="filepath",
        )
        json_download = gr.File(
            label="📋 Download JSON Report",
            interactive=False,
            visible=False,
            type="filepath",
        )

    # Wire the Analyze button with .click().then().then()
    analyze_button.click(
        fn=_show_status,
        inputs=None,
        outputs=status_markdown,
    ).then(
        fn=_process_call_with_error_display,
        inputs=[audio_input, caller_id, department],
        outputs=[transcript_output, summary_markdown, qa_markdown, pdf_download, json_download, error_display],
    ).then(
        fn=_hide_status,
        inputs=None,
        outputs=status_markdown,
    )

    return {
        "components": {
            "audio_input": audio_input,
            "caller_id": caller_id,
            "department": department,
            "analyze_button": analyze_button,
            "status_markdown": status_markdown,
            "error_display": error_display,
            "transcript_output": transcript_output,
            "summary_markdown": summary_markdown,
            "qa_markdown": qa_markdown,
            "pdf_download": pdf_download,
            "json_download": json_download,
        },
    }


def get_processing_status_message(estimated_duration: float = 60.0) -> str:
    """
    Generate a status message to display during processing.

    Args:
        estimated_duration: Estimated processing duration in seconds

    Returns:
        Markdown formatted status message
    """
    minutes = int(estimated_duration // 60)
    seconds = int(estimated_duration % 60)

    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    return f"""
🔄 **Processing Your Call...**

**Estimated Duration:** {time_str}

⚠️ **Please do not refresh the page** while the analysis is in progress.

This may result in:
- Lost analysis progress
- Incomplete data
- Session interruption

Your analysis will be completed shortly. Thank you for your patience!
    """

if __name__ == "__main__":
    with gr.Blocks(title="Analyze Call") as demo:
        gr.Markdown("# Analyze Call")
        result = create_analyze_tab()
    demo.launch()
