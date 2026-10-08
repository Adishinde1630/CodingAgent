"""Tests for the agent's tools, safety checks and the full LangGraph workflow (with a scripted fake LLM)."""
import json
import shutil
from pathlib import Path

import pytest

from app import code_analyzer, diff_manager, validator
from app.agent import CodingAgent, run_agent, run_agent_to_end
from app.file_manager import FileAccessError, FileManager, MAX_FILE_BYTES
from app.llm import MissingAPIKeyError, GroqClient, LLMError, RateLimitError, parse_json
from app.tools import AgentTools

SAMPLE = Path(__file__).resolve().parents[1] / "sample_project"
TASK = "Add input validation to the user registration API and write tests for invalid email addresses."


@pytest.fixture()
def workspace(tmp_path):
    dest = tmp_path / "ws"
    shutil.copytree(SAMPLE, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dest


@pytest.fixture()
def tools(workspace):
    return AgentTools(FileManager(workspace), test_timeout=60)


# ----------------------------------------------------------------- file safety
@pytest.mark.parametrize("bad", ["../secret.py", "../../etc/passwd", "a/../../x.py", "/etc/passwd",
                                 "C:/Windows/x.py", "~/x.py", ".env", "sub/.env.local", ".git/config",
                                 "__pycache__/x.py", "key.pem", "image.png", "", "a\x00b.py"])
def test_blocked_paths(workspace, bad):
    fm = FileManager(workspace)
    with pytest.raises(FileAccessError):
        fm.resolve(bad)


def test_write_outside_workspace_rejected(tools, workspace):
    res = tools.write_file("../evil.py", "x = 1\n")
    assert not res["ok"] and "traversal" in res["error"]
    assert not (workspace.parent / "evil.py").exists()


def test_symlink_escape_rejected(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    fm = FileManager(workspace)
    with pytest.raises(FileAccessError):
        fm.write_file("link/pwn.py", "x = 1\n")
    assert not (outside / "pwn.py").exists()


def test_env_and_secret_files_not_listed_or_readable(workspace):
    (workspace / ".env").write_text("GROQ_API_KEY=gsk_" + "a" * 40)
    (workspace / "config.py").write_text('KEY = "gsk_' + "b" * 40 + '"\n')
    fm = FileManager(workspace)
    assert ".env" not in fm.list_files()
    with pytest.raises(FileAccessError):
        fm.read_file(".env")
    with pytest.raises(FileAccessError, match="secret"):
        fm.read_file("config.py")


def test_file_not_found_and_size_limit(workspace):
    fm = FileManager(workspace)
    with pytest.raises(FileAccessError, match="not found"):
        fm.read_file("nope.py")
    (workspace / "big.py").write_text("x = 1\n" * (MAX_FILE_BYTES // 5))
    with pytest.raises(FileAccessError, match="too large"):
        fm.read_file("big.py")
    assert "big.py" not in fm.list_files()


def test_write_tracks_original_and_rollback(tools, workspace):
    original = (workspace / "validators.py").read_text()
    assert tools.write_file("validators.py", original + "\nX = 1\n")["status"] == "modified"
    assert tools.write_file("tests/test_new.py", "def test_x():\n    assert True\n")["status"] == "created"
    assert {c["path"] for c in tools.fm.changed_files()} == {"validators.py", "tests/test_new.py"}
    tools.fm.rollback()
    assert (workspace / "validators.py").read_text() == original
    assert not (workspace / "tests/test_new.py").exists()


def test_unchanged_write_not_reported_as_modified(tools, workspace):
    same = (workspace / "validators.py").read_text()
    assert tools.write_file("validators.py", same)["status"] == "unchanged"
    assert tools.fm.changed_files() == []


# ------------------------------------------------------------------- tools
def test_list_and_read_multiple_files(tools):
    files = tools.list_files()["files"]
    assert {"app.py", "validators.py", "models.py", "tests/test_app.py"} <= set(files)
    res = tools.read_files(["app.py", "validators.py", "missing.py", "../x.py"])
    assert set(res["files"]) == {"app.py", "validators.py"}
    assert set(res["errors"]) == {"missing.py", "../x.py"}


def test_search_finds_registration_files(tools):
    paths = [m["path"] for m in tools.search_code(TASK)["matches"]]
    assert paths[0] == "app.py" and "validators.py" in paths


def test_search_code_uses_metadata_without_reading_files(tools, monkeypatch):
    def fail_read(*args, **kwargs):
        raise AssertionError("candidate search must not read file contents")

    monkeypatch.setattr(tools, "read_files", fail_read)
    assert tools.search_code(TASK)["matches"]


def test_prompt_file_context_is_bounded_and_secret_safe():
    from app import prompts

    contents = {
        f"module_{index}.py": "value = 1\n" * 2_000
        for index in range(10)
    }
    contents[".env"] = "UPLOADED_SECRET_VALUE"
    rendered = prompts.changes("Make a small change", ["Update one module"], contents)

    assert "UPLOADED_SECRET_VALUE" not in rendered
    assert rendered.count("<file path=") <= 6
    assert len(rendered) < 17_000
    assert "truncated for model context" in rendered


def test_truncated_source_is_not_overwritten(tools):
    original = "x = 1\n...[truncated for model context]"
    (tools.fm.root / "large.py").write_text("x = 1\n" * 2_000, encoding="utf-8")
    state = {
        "file_contents": {"large.py": original},
        "proposed_changes": [{"path": "large.py", "content": "x = 2\n"}],
    }

    result = CodingAgent(llm=None, tools=tools).apply_changes(state)

    assert not result["apply_results"][0]["ok"]
    assert "truncated" in result["apply_results"][0]["error"]
    assert (tools.fm.root / "large.py").read_text(encoding="utf-8") == "x = 1\n" * 2_000


def test_code_analyzer_helpers(tools):
    tree = code_analyzer.build_tree(tools.list_files()["files"])
    assert "tests/" in tree and "app.py" in tree
    outline = code_analyzer.outline_file("app.py", (tools.fm.root / "app.py").read_text())
    assert "def register_user(" in outline
    assert code_analyzer.related_tests("app.py", ["tests/test_app.py"]) == ["tests/test_app.py"]


def test_diff_is_real(tools):
    tools.write_file("validators.py", (tools.fm.root / "validators.py").read_text() + "\nX = 1\n")
    d = tools.get_diff()
    assert "diff --git a/validators.py b/validators.py" in d["diff"] and "+X = 1" in d["diff"]
    assert d["files"][0]["added"] == 2
    assert diff_manager.make_diff({"a.py": "x\n"}, {"a.py": "x\n"})[0] == ""


def test_pytest_really_runs(tools):
    res = tools.run_tests()
    assert res["executed"] and res["passed"] and res["exit_code"] == 0
    assert "8 passed" in res["stdout"]


def test_pytest_failure_reported(tools):
    tools.write_file("tests/test_fail.py", "def test_bad():\n    assert 1 == 2\n")
    res = tools.run_tests()
    assert res["executed"] and not res["passed"] and res["exit_code"] == 1
    assert "assert 1 == 2" in res["stdout"]


def test_pytest_timeout(tools):
    tools.write_file("tests/test_slow.py", "import time\n\ndef test_slow():\n    time.sleep(30)\n")
    tools.test_timeout = 2
    res = tools.run_tests()
    assert res["timed_out"] and not res["passed"]


def test_validate_project_detects_syntax_error(tools, workspace):
    (workspace / "broken.py").write_text("def f(:\n")
    res = tools.validate_project()
    assert not res["passed"] and res["syntax_errors"][0]["path"] == "broken.py"


def test_test_subprocess_does_not_receive_api_key(tools, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "z" * 40)
    tools.write_file("tests/test_env.py", "import os\n\ndef test_no_key():\n    assert 'GROQ_API_KEY' not in os.environ\n")
    assert tools.run_tests()["passed"]


# --------------------------------------------------------------------- LLM
def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("app.llm.get_api_key", lambda: None)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        GroqClient()


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} done') == {"a": 2}
    with pytest.raises(Exception):
        parse_json("not json")


def test_error_mapping():
    client = GroqClient(api_key="dummy-key-for-test", model="m")

    class Err(Exception):
        def __init__(self, status_code, message):
            self.status_code, self.message = status_code, message

    name = lambda e: type(client._map_error(e)).__name__
    assert name(Err(429, "Rate limit reached for model")) == "RateLimitError"
    assert name(Err(401, "Invalid API Key")) == "InvalidAPIKeyError"
    assert name(Err(404, "The model `x` does not exist")) == "ModelNotFoundError"
    assert name(Err(400, "The model `x` has been decommissioned")) == "ModelNotFoundError"
    assert name(Err(400, "json_validate_failed: Failed to generate JSON")) == "_BadJSON"
    assert "too large" in str(client._map_error(Err(413, "Request too large for model")))
    assert "gsk_" not in str(client._map_error(Err(500, "boom gsk_" + "q" * 40)))
    assert "[REDACTED]" in str(client._map_error(Err(400, "bad gsk_" + "q" * 40)))


def test_rate_limit_is_not_retried_and_includes_retry_after():
    from types import SimpleNamespace as NS

    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=2)
    calls = []

    class Err(Exception):
        status_code = 429
        message = "Daily quota exceeded for this model"
        response = NS(headers={"retry-after": "45"})

    def create(**kwargs):
        calls.append(kwargs)
        raise Err()

    client._client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(RateLimitError) as err:
        client.generate_json("plan", "system", "prompt")
    assert len(calls) == 1
    assert "Daily quota exceeded" in str(err.value)
    assert "Retry after 45 seconds" in str(err.value)


def test_generate_json_with_mocked_groq_client(monkeypatch, caplog):
    """Drive GroqClient.generate_json through the real code path with a fake SDK client."""
    from types import SimpleNamespace as NS
    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=0)
    replies = iter(["not json at all", '{"ok": true}'])
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return NS(choices=[NS(message=NS(content=next(replies)), finish_reason="stop")])

    client._client = NS(chat=NS(completions=NS(create=create)))
    assert client.generate_json("t", "sys", "prompt") == {"ok": True}      # retried once after bad JSON
    assert seen["response_format"] == {"type": "json_object"} and seen["model"] == "m"
    assert seen["messages"][0]["role"] == "system"
    assert "tools" not in seen and "tool_choice" not in seen
    assert "Tools enabled: no" in caplog.text
    assert "Tool choice: none (not sent; no tools)" in caplog.text


def test_groq_allows_and_executes_provided_tools(caplog):
    from types import SimpleNamespace as NS

    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=0)
    requests = []
    responses = iter([
        NS(choices=[NS(
            message=NS(
                content=None,
                tool_calls=[NS(
                    id="call-tree",
                    function=NS(
                        name="repo_browser.print_tree",
                        arguments='{"path": "", "depth": 3}',
                    ),
                )],
            ),
            finish_reason="tool_calls",
        )]),
        NS(choices=[NS(
            message=NS(content='{"relevant_files": []}', tool_calls=None),
            finish_reason="stop",
        )]),
    ])

    def create(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client._client = NS(chat=NS(completions=NS(create=create)))
    output = client.generate_json(
        "select_files",
        "system",
        "Find the relevant files.",
        tools=[{
            "type": "function",
            "function": {
                "name": "repo_browser_print_tree",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        tool_handlers={
            "repo_browser_print_tree": lambda path="", depth=3: {
                "tree": "app.py\n", "file_count": 1, "path": path, "depth": depth,
            }
        },
    )

    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["tools"][0]["function"]["name"] == "repo_browser_print_tree"
    assert "response_format" not in requests[0]
    assert all(request["tool_choice"] == "auto" for request in requests)
    assert all(len(request["tools"]) == 1 for request in requests)
    assert caplog.text.count("Tools enabled: yes") == 2
    assert caplog.text.count("Tool choice: auto") == 2
    assert requests[1]["messages"][-2]["role"] == "assistant"
    assert requests[1]["messages"][-2]["tool_calls"][0]["id"] == "call-tree"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-tree",
        "content": '{"tree": "app.py\\n", "file_count": 1, "path": "", "depth": 3}',
    }
    assert output["relevant_files"] == []
    assert output["_tool_results"] == [{
        "name": "repo_browser.print_tree",
        "result": {"tree": "app.py\n", "file_count": 1, "path": "", "depth": 3},
    }]


def test_duplicate_groq_tool_calls_reuse_cached_result(caplog):
    from types import SimpleNamespace as NS

    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=0)
    requests = []
    executions = []

    def tool_response(call_id, path=""):
        return NS(choices=[NS(
            message=NS(
                content=None,
                tool_calls=[NS(
                    id=call_id,
                    function=NS(
                        name="repo_browser.print_tree",
                        arguments=json.dumps({"path": path, "depth": 3}),
                    ),
                )],
            ),
            finish_reason="tool_calls",
        )])

    responses = iter(
        [tool_response(f"call-{i}") for i in range(4)]
        + [NS(choices=[NS(
            message=NS(content='{"relevant_files": []}', tool_calls=None),
            finish_reason="stop",
        )])]
    )

    def create(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client._client = NS(chat=NS(completions=NS(create=create)))
    result = client.generate_json(
        "select_files",
        "system",
        "Select relevant files.",
        tools=[{
            "type": "function",
            "function": {
                "name": "repo_browser_print_tree",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        tool_handlers={
            "repo_browser_print_tree": lambda **kwargs: (
                executions.append(kwargs) or {"tree": "app.py"}
            )
        },
    )

    assert result["relevant_files"] == []
    assert result["_tool_call_count"] == 1
    assert not result["_tool_limit_reached"]
    assert len(executions) == 1
    assert len(requests) == 5
    assert caplog.text.count("Duplicate tool call detected") == 3


def test_groq_enforces_maximum_actual_tool_calls():
    from types import SimpleNamespace as NS
    from app.llm import MAX_TOOL_CALLS

    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=0)
    requests = []
    executions = []

    def tool_response(call_id, path):
        return NS(choices=[NS(
            message=NS(
                content=None,
                tool_calls=[NS(
                    id=call_id,
                    function=NS(
                        name="repo_browser.print_tree",
                        arguments=json.dumps({"path": path, "depth": 3}),
                    ),
                )],
            ),
            finish_reason="tool_calls",
        )])

    responses = iter(
        [tool_response(f"call-{i}", f"dir-{i}") for i in range(MAX_TOOL_CALLS)]
        + [tool_response("call-over-limit", "dir-over-limit")]
        + [NS(choices=[NS(
            message=NS(content='{"relevant_files": []}', tool_calls=None),
            finish_reason="stop",
        )])]
    )

    def create(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client._client = NS(chat=NS(completions=NS(create=create)))
    result = client.generate_json(
        "select_files",
        "system",
        "Select relevant files.",
        tools=[{
            "type": "function",
            "function": {
                "name": "repo_browser_print_tree",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        tool_handlers={
            "repo_browser_print_tree": lambda **kwargs: (
                executions.append(kwargs) or {"tree": kwargs["path"]}
            )
        },
    )

    assert len(executions) == MAX_TOOL_CALLS
    assert result["_tool_call_count"] == MAX_TOOL_CALLS
    assert result["_tool_limit_reached"]
    assert len(requests) == MAX_TOOL_CALLS + 2
    assert all(request.get("tools") for request in requests[:MAX_TOOL_CALLS])
    assert all("tools" not in request for request in requests[MAX_TOOL_CALLS:])


def test_repeated_tool_call_loop_stops_at_iteration_limit():
    from types import SimpleNamespace as NS
    from app.llm import MAX_TOOL_ITERATIONS

    client = GroqClient(api_key="dummy-key-for-test", model="m", retries=0)
    requests = []
    executions = []

    def create(**kwargs):
        requests.append(kwargs)
        return NS(choices=[NS(message=NS(
            content=None,
            tool_calls=[NS(
                id=f"call-{len(requests)}",
                function=NS(
                    name="repo_browser.print_tree",
                    arguments='{"path": "", "depth": 3}',
                ),
            )],
        ), finish_reason="tool_calls")])

    client._client = NS(chat=NS(completions=NS(create=create)))
    result = client.generate_json(
        "select_files",
        "system",
        "Select relevant files.",
        tools=[{
            "type": "function",
            "function": {
                "name": "repo_browser_print_tree",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        tool_handlers={
            "repo_browser_print_tree": lambda **kwargs: (
                executions.append(kwargs) or {"tree": "app.py"}
            )
        },
    )

    assert len(requests) == MAX_TOOL_ITERATIONS + 1
    assert len(executions) == 1
    assert result["_tool_call_count"] == 1
    assert result["_tool_limit_reached"]


def test_invalid_key_from_real_sdk_is_friendly():
    """Real Groq SDK object, rejected key / no network -> a friendly LLMError (never a traceback)."""
    client = GroqClient(api_key="gsk_" + "x" * 40, model="llama-3.3-70b-versatile", retries=0)
    with pytest.raises(LLMError) as err:
        client.generate_json("t", "sys", "give me JSON")
    assert "gsk_" not in str(err.value)


# ---------------------------------------------------------- full agent flow
class FakeLLM:
    """Scripted stand-in for the Groq model. Produces a plausible solution to the demo task."""

    def __init__(self, workspace: Path, break_first: bool = False, empty_task_summary_fails: bool = False):
        self.ws, self.break_first, self.calls = workspace, break_first, []
        self.summary_fails = empty_task_summary_fails

    def generate_json(self, stage, system, prompt, **_):
        self.calls.append(stage)
        if stage == "understand":
            return {"summary": "Add email validation", "task_type": "feature",
                    "keywords": ["register_user", "email", "validators"], "expected_changes": "validators + app + tests"}
        if stage == "select_files":
            return {"relevant_files": [{"path": p, "reason": "relevant"} for p in
                    ["app.py", "validators.py", "tests/test_app.py", "tests/test_validators.py", "../../etc/passwd", "ghost.py"]]}
        if stage == "plan":
            return {"plan": ["1. Inspect register_user", "Add is_valid_email to validators.py", "Wire it into app.py",
                             "Add tests", "Run pytest", "Review diff"]}
        if stage == "changes":
            if self.break_first and self.calls.count("changes") == 1:
                return {"changes": [{"path": "validators.py", "content": "def broken(:\n", "explanation": "oops"},
                                    {"path": "../escape.py", "content": "x = 1\n", "explanation": "bad path"},
                                    {"path": "app.py", "content": (SAMPLE / "app.py").read_text().replace(
                                        "from validators import is_strong_password, is_valid_username",
                                        "from validators import is_strong_password, is_valid_username, is_valid_email")
                                        .replace('    if not is_valid_username(username):',
                                                 '    if not is_valid_email(email):\n        return 400, {"error": "Invalid email address"}\n\n    if not is_valid_username(username):'),
                                     "explanation": "uses a validator that does not exist yet"}]}
            return self._good_changes()
        if stage == "summary":
            if self.summary_fails:
                raise LLMError("rate limited")
            return {"summary": "Added email validation.", "why": "Registration accepted any email.",
                    "limitations": ["Regex-based validation"]}
        raise AssertionError(stage)

    def _good_changes(self):
        read = lambda p: (SAMPLE / p).read_text()  # always start from the pristine sample
        validators = read("validators.py") + '''

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\\-]+@[A-Za-z0-9\\-]+(\\.[A-Za-z0-9\\-]+)*\\.[A-Za-z]{2,}$")


def is_valid_email(email: str) -> bool:
    """Basic email format check (single @, a domain with a dot, no spaces)."""
    return isinstance(email, str) and len(email) <= 254 and bool(EMAIL_PATTERN.match(email))
'''
        app = (read("app.py")
               .replace("from validators import is_strong_password, is_valid_username",
                        "from validators import is_strong_password, is_valid_email, is_valid_username")
               .replace("    if not is_valid_username(username):",
                        '    if not is_valid_email(email):\n        return 400, {"error": "Invalid email address"}\n\n    if not is_valid_username(username):'))
        test_app = read("tests/test_app.py") + '''

import pytest


@pytest.mark.parametrize("bad_email", ["plainaddress", "no@tld", "@missing.com", "spaces in@x.com", "a@b@c.com"])
def test_register_invalid_email(bad_email):
    store = UserStore()
    status, body = register_user(make_payload(email=bad_email), store)
    assert status == 400
    assert body["error"] == "Invalid email address"
    assert store.count() == 0
'''
        test_val = read("tests/test_validators.py").replace(
            "from validators import is_strong_password, is_valid_username",
            "from validators import is_strong_password, is_valid_email, is_valid_username") + '''

def test_email_validation():
    assert is_valid_email("alice@example.com")
    assert not is_valid_email("alice@")
    assert not is_valid_email(None)
'''
        return {"changes": [{"path": "validators.py", "content": validators, "explanation": "add is_valid_email"},
                            {"path": "app.py", "content": app, "explanation": "reject invalid emails"},
                            {"path": "tests/test_app.py", "content": test_app, "explanation": "API tests"},
                            {"path": "tests/test_validators.py", "content": test_val, "explanation": "unit tests"},
                            ]}


def test_full_workflow_demo_task(workspace):
    fm = FileManager(workspace)
    llm = FakeLLM(workspace)
    state = run_agent_to_end(CodingAgent(llm, AgentTools(fm)), TASK, max_attempts=2)

    assert not state.get("fatal_error"), state.get("fatal_error")
    assert llm.calls == ["understand", "select_files", "plan", "changes", "summary"]
    # multiple files really read; invalid / hallucinated paths were dropped
    assert set(state["file_contents"]) == {"app.py", "validators.py", "tests/test_app.py", "tests/test_validators.py"}
    assert state["plan"][0] == "Inspect register_user"          # numbering stripped
    assert state["changes"] == state["proposed_changes"]
    # real modifications, real diff, real pytest run
    assert {m["path"] for m in state["modified_files"]} == {"validators.py", "app.py", "tests/test_app.py", "tests/test_validators.py"}
    assert "+def is_valid_email" in state["diff"] and "diff --git a/app.py b/app.py" in state["diff"]
    assert state["validation_passed"] and state["test_results"]["exit_code"] == 0
    assert "passed" in state["test_results"]["summary"]
    assert "is_valid_email" in (workspace / "validators.py").read_text()
    assert state["final_summary"]["summary"].startswith("Added email validation")


def test_repair_loop_recovers_from_bad_code(workspace):
    fm = FileManager(workspace)
    llm = FakeLLM(workspace, break_first=True)
    state = run_agent_to_end(CodingAgent(llm, AgentTools(fm)), TASK, max_attempts=2)
    assert llm.calls.count("changes") == 2                       # one repair attempt happened
    assert state["attempt_history"][0]["passed"] is False         # ImportError -> pytest failed
    assert state["attempt_history"][1]["passed"] is True
    assert any("syntax error" in e for e in state["errors"])      # broken code was never written
    assert any("traversal" in e for e in state["errors"])
    assert not (workspace.parent / "escape.py").exists()
    assert state["validation_passed"]


def test_failure_is_reported_and_auto_rollback(workspace):
    original = (workspace / "app.py").read_text()
    fm = FileManager(workspace)
    llm = FakeLLM(workspace, break_first=True)
    state = run_agent_to_end(CodingAgent(llm, AgentTools(fm)), TASK, max_attempts=1, auto_rollback=True)
    assert not state["validation_passed"] and state["test_results"]["exit_code"] != 0
    assert state["rolled_back"] and (workspace / "app.py").read_text() == original
    assert state["diff"]                                           # diff captured before rollback


def test_summary_falls_back_when_llm_fails(workspace):
    llm = FakeLLM(workspace, empty_task_summary_fails=True)
    state = run_agent_to_end(CodingAgent(llm, AgentTools(FileManager(workspace))), TASK)
    assert "Modified" in state["final_summary"]["summary"] and state["validation_passed"]


def test_empty_task_is_rejected(workspace):
    state = run_agent_to_end(CodingAgent(FakeLLM(workspace), AgentTools(FileManager(workspace))), "   ")
    assert "enter a coding task" in state["fatal_error"]
    assert not state.get("modified_files")


def test_llm_failure_stops_gracefully(workspace):
    class Boom:
        def generate_json(self, *a, **k):
            raise LLMError("Groq rejected the API key")
    state = run_agent_to_end(CodingAgent(Boom(), AgentTools(FileManager(workspace))), TASK)
    assert state["fatal_error"] == "Groq rejected the API key"


def test_plan_rate_limit_stops_before_code_generation(workspace):
    class PlanRateLimited(FakeLLM):
        def generate_json(self, stage, system, prompt, **kwargs):
            if stage == "plan":
                self.calls.append(stage)
                raise RateLimitError("Groq rate limit reached: daily quota exceeded. Please wait and try again.")
            return super().generate_json(stage, system, prompt, **kwargs)

    llm = PlanRateLimited(workspace)
    state = run_agent_to_end(CodingAgent(llm, AgentTools(FileManager(workspace))), "Add a ping endpoint.")

    assert llm.calls == ["understand", "select_files", "plan"]
    assert state["rate_limited"]
    assert "daily quota exceeded" in state["fatal_error"]
    assert state["errors"] == [state["fatal_error"]]
    assert state["plan"] == ""
    assert state["changes"] == []
    assert state["relevant_files"]
    assert state["modified_files"] == []
    assert state["diff"] == ""
    assert state["test_results"] == ""
    assert state["final_summary"] == ""


def test_relevant_files_are_initialized_and_empty_selection_ends_gracefully(workspace):
    class EmptySelection:
        def generate_json(self, stage, system, prompt, **kwargs):
            if stage == "understand":
                return {"summary": "Find a missing feature", "keywords": []}
            if stage == "select_files":
                return {"relevant_files": []}
            raise AssertionError(f"Unexpected LLM stage: {stage}")

    agent = CodingAgent(EmptySelection(), AgentTools(FileManager(workspace)))
    agent.tools.search_code = lambda *args, **kwargs: {"matches": []}
    first_event = next(run_agent(agent, "Implement an unrelated feature"))
    assert "relevant_files" in first_event[2]
    assert first_event[2]["relevant_files"] == []

    state = run_agent_to_end(agent, "Implement an unrelated feature")
    assert not state.get("fatal_error")
    assert state["relevant_files"] == []
    assert state["no_relevant_files"]
    assert "No relevant files were identified" in state["final_summary"]["summary"]


# ------------------------------------------- Groq 400 "tool_use_failed" (model hallucinates a tool call)
TOOL_LEAK_MESSAGE = ("Error code: 400 - {'error': {'message': 'Tool choice is none, but model called a tool', "
                     "'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': "
                     "'{\"name\": \"repo_browser.print_tree\", \"arguments\": {\"path\": \"\", \"depth\": 3}\\n}'}}")


class _Groq400(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code, self.message = 400, message


def _client_with(replies, seen):
    from types import SimpleNamespace as NS
    client = GroqClient(api_key="dummy-key-for-test", model="openai/gpt-oss-20b", retries=0)
    replies = iter(replies)

    def create(**kwargs):
        seen.append(kwargs)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return NS(choices=[NS(message=NS(content=reply, tool_calls=None), finish_reason="stop")])

    client._client = NS(chat=NS(completions=NS(create=create)))
    return client


def test_tool_use_failed_is_retryable_not_fatal():
    client = GroqClient(api_key="dummy-key-for-test", model="m")
    assert type(client._map_error(_Groq400(TOOL_LEAK_MESSAGE))).__name__ == "_ToolCallLeak"


def test_model_tool_call_leak_recovers_on_retry():
    seen = []
    client = _client_with([_Groq400(TOOL_LEAK_MESSAGE), '{"ok": true}'], seen)
    assert client.generate_json("select_files", "sys", "prompt") == {"ok": True}
    assert len(seen) == 2
    assert "NO tools" in seen[0]["messages"][0]["content"]                 # prevented up front
    assert "Do NOT call any tool" in seen[1]["messages"][1]["content"]     # stronger instruction on retry
    assert "tools" not in seen[1] and seen[1]["response_format"] == {"type": "json_object"}


def test_persistent_tool_call_leak_gives_actionable_error():
    seen = []
    client = _client_with([_Groq400(TOOL_LEAK_MESSAGE)] * 3, seen)
    with pytest.raises(LLMError) as err:
        client.generate_json("select_files", "sys", "prompt")
    assert "kept trying to call tools" in str(err.value) and "GROQ_MODEL" in str(err.value)
    assert "tool_use_failed" not in str(err.value) and len(seen) == 3
