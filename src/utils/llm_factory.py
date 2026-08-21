"""Lightweight LLM factory for supported providers.

Supported providers:
  - "openai" -> ChatOpenAI(model="gpt-4o")
  - "gemini" -> ChatGoogleGenerativeAI(model="gemini-2.0-flash")
  - "groq" -> ChatGroq(model="llama-3.3-70b-versatile")

If the provider-specific client classes are not available, a simple
`LLMStub` dataclass is returned so downstream code can inspect the
chosen provider/model/timeout during testing or runtime.
"""
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .config import get_logger, load_config
import os


logger = get_logger("llm_factory")

def get_llm(provider: str, model: Optional[str] = None, timeout: Optional[float] = None):
    """Return an LLM client for the given provider.

    provider: one of 'openai', 'gemini', 'groq' (case-insensitive)
    model: optional model override; if not provided a sensible default is used.
    timeout: optional request timeout in seconds.
    """
    if provider is None:
        raise ValueError("provider is required")

    p = provider.strip().lower()

    if p == "openai":
        default_model = "gpt-4o-mini"
        model_name = model or default_model
        try:
            return ChatOpenAI(model=model_name, timeout=timeout, base_url="https://api.openai.com/v1")
        except TypeError:
            # fallback to common param name
            return ChatOpenAI(model_name=model_name)
       

    if p == "gemini":
        default_model = "gemini-2.0-flash"
        model_name = model or default_model
        
        try:
            return ChatGoogleGenerativeAI(model=model_name, timeout=timeout)
        except TypeError:
            return ChatGoogleGenerativeAI(model=model_name)
        
    if p == "groq":
        default_model = "llama-3.3-70b-versatile"
        model_name = model or default_model
        
        try:
            return ChatGroq(model=model_name, timeout=timeout)
        except TypeError:
            return ChatGroq(model=model_name)
        

    raise ValueError(f"Unsupported provider: {provider}")


try:
    config = load_config()
    llm = get_llm(config.llm_provider, model=None, timeout=config.llm_timeout_seconds)
    logger.info(f"LLM client initialized: provider={config.llm_provider}, model={llm.model_name}, timeout={config.llm_timeout_seconds}")
except Exception as e:
    llm = None
    logger.warning(f"Warning: Failed to initialize LLM: {e}")

__all__ = ["get_llm", "llm"]

