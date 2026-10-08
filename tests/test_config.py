import pytest

from app import config
from app.llm import GroqClient, MissingAPIKeyError


def test_api_key_from_project_dotenv_takes_priority_over_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GROQ_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "stale-environment-key")

    assert config.get_api_key() == "dotenv-key"


def test_placeholder_in_dotenv_falls_back_to_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=your_groq_api_key_here\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "environment-key")

    assert config.get_api_key() == "environment-key"


def test_blank_dotenv_falls_back_to_streamlit_secrets(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GROQ_API_KEY=   \n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        config, "_from_streamlit_secrets", lambda _: "streamlit-key"
    )

    assert config.get_api_key() == "streamlit-key"


def test_explicit_empty_client_key_does_not_fall_back_to_config(monkeypatch):
    monkeypatch.setattr(
        "app.llm.get_api_key",
        lambda: pytest.fail("unexpected config fallback"),
    )

    with pytest.raises(MissingAPIKeyError):
        GroqClient(api_key="")
