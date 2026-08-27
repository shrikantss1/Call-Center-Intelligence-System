"""Thin entrypoint for Call Center Intelligence System."""

import os

import gradio as gr
from sqlalchemy.orm import sessionmaker

from src.agents.transcription import _get_whisper_model
from src.app_globals import set_workflow_and_agent
from src.database.connection import get_engine
from src.database.init_db import init_db
from src.graph.workflow import build_workflow
from src.security.audit import AuditLogger
from src.ui.main import build_app
from src.utils.config import get_logger, load_config

logger = get_logger("app")

# ============ INITIALIZATION ============
config = load_config()
logger.info("✓ Configuration loaded")

engine = get_engine()
logger.info("✓ Database engine created")

init_db()
logger.info("✓ Database initialized")

whisper_size = config.whisper_model_size or "small"
logger.info(f"Loading Whisper model (size: {whisper_size})...")
_get_whisper_model(whisper_size)
logger.info("✓ Whisper model loaded")

workflow_graph = build_workflow()
agent = workflow_graph.compile()
set_workflow_and_agent(workflow_graph, agent)
logger.info("✓ Workflow compiled and registered")

session_factory = sessionmaker(bind=engine)
audit_logger = AuditLogger(session_factory)
logger.info("✓ AuditLogger initialized")

# ============ BUILD & LAUNCH ============
demo = build_app()
logger.info("✓ Gradio app built")

# Determine server_name: HuggingFace Spaces vs local
server_name = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
logger.info(f"Launching on {server_name}:7860")

if __name__ == "__main__":
    demo.launch(server_name=server_name, share=False, show_error=True, theme=gr.themes.Soft())
