"""Global app instance holders for workflow and agent.

These are populated during app initialization and used by pipeline.py.
"""

from typing import Optional
from langgraph.graph import StateGraph

# Module-level holders for workflow and agent
_workflow: Optional[StateGraph] = None
_agent = None


def set_workflow_and_agent(workflow: StateGraph, agent):
    """Set the compiled workflow and agent from app.py."""
    global _workflow, _agent
    _workflow = workflow
    _agent = agent


def get_workflow():
    """Get the compiled workflow."""
    if _workflow is None:
        raise RuntimeError("Workflow not initialized. Call set_workflow_and_agent() first.")
    return _workflow


def get_agent():
    """Get the compiled agent."""
    if _agent is None:
        raise RuntimeError("Agent not initialized. Call set_workflow_and_agent() first.")
    return _agent
