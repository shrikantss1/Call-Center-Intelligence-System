"""Configuration loader using python-dotenv.

Loads environment variables and returns a frozen `Config` dataclass.
Supported env vars:
  LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY,
  WHISPER_MODEL_SIZE, CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_HALT_RATIO,
  DB_PATH, DB_ENCRYPTION_KEY, MAX_RETRIES_PER_NODE, LLM_TIMEOUT_SECONDS
"""
from dataclasses import dataclass
import os
from typing import Optional
import logging
import sys

from dotenv import load_dotenv


# Load environment from .env (if present) into os.environ
load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    return val


def _get_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got: {val}")


def _get_float(name: str, default: Optional[float] = None) -> Optional[float]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a float, got: {val}")


@dataclass(frozen=True)
class Config:
    llm_provider: str
    openai_api_key: Optional[str]
    google_api_key: Optional[str]
    groq_api_key: Optional[str]
    whisper_model_size: Optional[str]
    confidence_threshold: Optional[float]
    low_confidence_halt_ratio: Optional[float]
    db_path: Optional[str]
    db_encryption_key: Optional[str]
    max_retries_per_node: Optional[int]
    llm_timeout_seconds: Optional[float]


def load_config() -> Config:
    """Read environment variables and return a frozen `Config`.

    Raises:
        RuntimeError: if a required variable is missing.
        ValueError: if an environment variable cannot be parsed to the expected type.
    """
    # Required variable: LLM_PROVIDER
    llm_provider = _get_env("LLM_PROVIDER")
    if not llm_provider:
        raise RuntimeError("Environment variable LLM_PROVIDER is required")

    cfg = Config(
        llm_provider=llm_provider,
        openai_api_key=_get_env("OPENAI_API_KEY"),
        google_api_key=_get_env("GOOGLE_API_KEY"),
        groq_api_key=_get_env("GROQ_API_KEY"),
        whisper_model_size=_get_env("WHISPER_MODEL_SIZE"),
        confidence_threshold=_get_float("CONFIDENCE_THRESHOLD"),
        low_confidence_halt_ratio=_get_float("LOW_CONFIDENCE_HALT_RATIO"),
        db_path=_get_env("DB_PATH"),
        db_encryption_key=_get_env("DB_ENCRYPTION_KEY"),
        max_retries_per_node=_get_int("MAX_RETRIES_PER_NODE"),
        llm_timeout_seconds=_get_float("LLM_TIMEOUT_SECONDS"),
    )

    return cfg


__all__ = ["Config", "load_config"]
