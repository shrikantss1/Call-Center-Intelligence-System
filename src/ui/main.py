"""Main Gradio app builder for Call Center Intelligence System."""

import gradio as gr

from src.ui.tabs.analyze import create_analyze_tab
from src.ui.tabs.observability import create_observability_tab


def build_app() -> gr.Blocks:
	"""Build the main Gradio app with Analyze and Observability tabs.

	Returns:
		Compiled Gradio Blocks instance
	"""
	with gr.Blocks(title="Call Center Intelligence System") as demo:
		gr.Markdown("# 📞 Call Center Intelligence System")
		gr.Markdown("Analyze call recordings with AI-powered insights, transcription, and QA scoring.")

		with gr.Tabs():
			with gr.TabItem("Analyze Call", id="analyze"):
				create_analyze_tab()

			with gr.TabItem("Observability", id="observability"):
				create_observability_tab()

	return demo
