"""Configuration: environment variables, .env, and Streamlit secrets."""
import os
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PROJECT_DIR = ROOT_DIR / "sample_project"
DEFAULT_MODEL = "openai/gpt-oss-20b"

load_dotenv(ROOT_DIR / ".env")

_SECRET_PATTERNS = [
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


def redact(text: str) -> str:
    """Remove anything that looks like an API key from text shown to users."""
    text = str(text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    key = os.getenv("GROQ_API_KEY")
    if key and len(key) > 8:
        text = text.replace(key, "[REDACTED]")
    return text


def _from_streamlit_secrets(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)  # raises if no secrets file exists
        return str(value) if value else None
    except Exception:
        return None


def get_setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then Streamlit secrets, then default."""
    return os.getenv(name) or _from_streamlit_secrets(name) or default


def get_api_key() -> str | None:
    dotenv_key = dotenv_values(ROOT_DIR / ".env").get("GROQ_API_KEY")
    if dotenv_key and dotenv_key.strip().lower() != "your_groq_api_key_here":
        return dotenv_key.strip()

    key = get_setting("GROQ_API_KEY")
    if key and key.strip().lower() == "your_groq_api_key_here":
        return None
    return key.strip() if key and key.strip() else None


def get_model_name() -> str:
    return get_setting("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
