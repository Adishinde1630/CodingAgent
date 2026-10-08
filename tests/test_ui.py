"""Smoke tests that drive the real Streamlit script with streamlit.testing (no network, scripted LLM)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.llm import RateLimitError
from test_agent_tools import FakeLLM, SAMPLE, TASK

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("app.config.get_api_key", lambda: None)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_loads_with_header_and_tabs():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "AI Coding Agent"
    assert not at.text_input
    assert "app.py" in at.session_state["fm"].list_files()
    assert at.multiselect[0].value == ["app.py", "validators.py"]
    assert [t.label for t in at.tabs] == ["Task", "Project Files", "Agent Plan", "Changes", "Diff", "Validation", "Final Summary"]


def test_upload_mode_requires_zip_instead_of_using_sample_project():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.radio[0].set_value("Upload ZIP").run()

    assert not at.exception
    assert "Upload a valid project ZIP" in " ".join(e.value for e in at.warning)
    assert "Upload a ZIP file to view its project files." in " ".join(i.value for i in at.info)


def test_missing_api_key_shows_clear_error():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.text_area[0].set_value(TASK)
    _button(at, "Run Coding Agent").click().run()
    assert not at.exception
    assert any("GROQ_API_KEY is not set" in e.value for e in at.error)


def test_partial_selection_update_does_not_crash_ui(monkeypatch):
    class FailingSelectionLLM:
        def generate_json(self, stage, system, prompt, **kwargs):
            if stage == "understand":
                return {"summary": "Inspect the project", "keywords": []}
            if stage == "select_files":
                raise RuntimeError("selection unavailable")
            raise AssertionError(f"Unexpected LLM stage: {stage}")

    monkeypatch.setattr("app.config.get_api_key", lambda: "backend-test-key")
    monkeypatch.setattr("app.config.get_model_name", lambda: "backend-test-model")
    monkeypatch.setattr("app.llm.GroqClient", lambda **kwargs: FailingSelectionLLM())
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.text_area[0].set_value("Inspect the project").run()
    _button(at, "Run Coding Agent").click().run()

    assert not at.exception
    assert "selection unavailable" in at.session_state["result"]["fatal_error"]
    assert at.session_state["result"]["relevant_files"] == []


def test_empty_task_shows_error():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    _button(at, "Run Coding Agent").click().run()
    assert any("enter a coding task" in e.value for e in at.error)


def test_full_run_with_scripted_llm(monkeypatch):
    client_config = {}

    def fake_groq_client(api_key=None, model=None):
        client_config.update(api_key=api_key, model=model)
        return FakeLLM(SAMPLE)

    monkeypatch.setattr("app.config.get_api_key", lambda: "backend-test-key")
    monkeypatch.setattr("app.config.get_model_name", lambda: "backend-test-model")
    monkeypatch.setattr("app.llm.GroqClient", fake_groq_client)
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    _button(at, "Use demo task").click().run()
    assert at.text_area[0].value == TASK
    _button(at, "Run Coding Agent").click().run()
    assert not at.exception, at.exception
    assert client_config == {"api_key": "backend-test-key", "model": "backend-test-model"}
    result = at.session_state["result"]
    assert result["validation_passed"] and len(result["modified_files"]) == 4
    at.text_area[0].set_value(TASK + " ").run()
    assert at.session_state["result"]["validation_passed"]
    all_md = " ".join(m.value for m in at.markdown)
    assert "Tests executed" in all_md and "Tests passed" in all_md and "Modified" not in at.error
    diff_blocks = [c.value for c in at.code if c.language == "diff"]
    assert any("+def is_valid_email" in d for d in diff_blocks)
    # the revert button really restores the sample project
    _button(at, "Revert all changes").click().run()
    assert "is_valid_email" not in at.session_state["fm"].read_file("validators.py")


def test_plan_rate_limit_is_shown_without_crashing(monkeypatch):
    class PlanRateLimited(FakeLLM):
        def generate_json(self, stage, system, prompt, **kwargs):
            if stage == "plan":
                self.calls.append(stage)
                raise RateLimitError("Groq rate limit reached: daily quota exceeded. Please wait and try again.")
            return super().generate_json(stage, system, prompt, **kwargs)

    monkeypatch.setattr("app.config.get_api_key", lambda: "backend-test-key")
    monkeypatch.setattr("app.config.get_model_name", lambda: "backend-test-model")
    monkeypatch.setattr("app.llm.GroqClient", lambda **kwargs: PlanRateLimited(SAMPLE))
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.text_area[0].set_value("Add a ping endpoint.")
    _button(at, "Run Coding Agent").click().run()

    assert not at.exception
    assert at.session_state["result"]["rate_limited"]
    assert "daily quota exceeded" in at.session_state["result"]["fatal_error"]
    assert any("Please wait for the Groq rate limit to reset" in warning.value for warning in at.warning)
    assert not any("Agent crashed" in error.value for error in at.error)
