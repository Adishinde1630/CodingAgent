"""Safety and workflow tests for uploaded ZIP workspaces."""
import io
import stat
import zipfile

import pytest

from app.agent import CodingAgent, run_agent_to_end
from app.file_manager import FileAccessError, FileManager, MAX_FILES
from app.prompts import plan
from app.tools import AgentTools
from app.validator import NO_TEST_CONFIGURATION, run_pytest
from app.workspace import MAX_ZIP_BYTES, MAX_ZIP_ENTRIES, MAX_ZIP_UPLOAD_MB, create_uploaded_workspace


def _zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def test_zip_upload_limit_is_500_mb():
    assert MAX_ZIP_UPLOAD_MB == 500
    assert MAX_ZIP_BYTES == 500 * 1024 * 1024


def test_rejects_zip_over_upload_size_limit(monkeypatch):
    monkeypatch.setattr("app.workspace.MAX_ZIP_BYTES", 1)

    with pytest.raises(FileAccessError, match="upload limit"):
        create_uploaded_workspace(b"12", "large.zip")


def test_extracts_project_with_a_single_top_level_folder():
    workspace = create_uploaded_workspace(_zip([
        ("customer_app/src/api.py", "def register():\n    return True\n"),
        ("customer_app/tests/test_api.py", "def test_register():\n    assert True\n"),
        ("customer_app/pyproject.toml", "[tool.pytest.ini_options]\n"),
    ]), "customer_app.zip")

    assert workspace.name == "customer_app"
    assert workspace.root == workspace.parent / "extracted" / "customer_app"
    assert workspace.file_manager.root == workspace.root
    assert set(workspace.file_manager.list_files()) == {
        "src/api.py", "tests/test_api.py", "pyproject.toml"
    }


def test_extracts_files_when_the_zip_has_no_top_level_folder():
    workspace = create_uploaded_workspace(_zip([
        ("app.py", "def main():\n    return 1\n"),
        ("README.md", "Project readme\n"),
    ]), "my-project.zip")

    assert workspace.name == "my-project"
    assert workspace.root == workspace.parent / "extracted"
    assert set(workspace.file_manager.list_files()) == {"app.py", "README.md"}


@pytest.mark.parametrize("entry", [
    "../outside.py", "folder/../../outside.py", r"..\outside.py", "/outside.py", "C:/outside.py",
])
def test_rejects_traversal_and_absolute_zip_paths(entry):
    with pytest.raises(FileAccessError, match="Unsafe"):
        create_uploaded_workspace(_zip([(entry, "x = 1\n")]), "bad.zip")


def test_rejects_zip_symlink_metadata():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("project/link.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside.py")

    with pytest.raises(FileAccessError, match="symlinks"):
        create_uploaded_workspace(stream.getvalue(), "bad.zip")


def test_rejects_excessive_entry_count():
    payload = _zip([(f"project/file_{index}.txt", "ok")
                    for index in range(MAX_ZIP_ENTRIES + 1)])
    with pytest.raises(FileAccessError, match="too many entries"):
        create_uploaded_workspace(payload, "large.zip")


def test_rejects_more_supported_files_than_file_manager_limit():
    payload = _zip([(f"project/file_{index}.txt", "ok") for index in range(MAX_FILES + 1)])
    with pytest.raises(FileAccessError, match="supported project files"):
        create_uploaded_workspace(payload, "too-many-files.zip")


def test_rejects_excessive_path_nesting():
    entry = "/".join(["nested"] * 17 + ["main.py"])
    with pytest.raises(FileAccessError, match="nested"):
        create_uploaded_workspace(_zip([(entry, "x = 1\n")]), "deep.zip")


def test_skips_protected_files_and_directories_but_extracts_project_files():
    workspace = create_uploaded_workspace(_zip([
        ("my_project/.env", "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE"),
        ("my_project/.env.example", "EXAMPLE_VALUE=placeholder"),
        ("my_project/.gitignore", ".env\n"),
        ("my_project/.git/config", "UPLOADED_GIT_SECRET_DO_NOT_DISCLOSE"),
        ("my_project/__pycache__/cache.py", "cached = True\n"),
        ("my_project/.venv/lib.py", "venv = True\n"),
        ("my_project/venv/lib.py", "venv = True\n"),
        ("my_project/.pytest_cache/state.py", "cached = True\n"),
        ("my_project/node_modules/package.py", "package = True\n"),
        ("my_project/private.pem", "PRIVATE KEY"),
        ("my_project/id_rsa.pub", "PUBLIC KEY"),
        ("my_project/credentials.json", '{"password": "hidden"}'),
        ("my_project/access_token.txt", "hidden-token"),
        ("my_project/requirements.txt", "pytest>=8\n"),
        ("my_project/app.py", "def main():\n    return True\n"),
        ("my_project/utils.py", "def helper():\n    return True\n"),
        ("my_project/tests/test_app.py", "def test_main():\n    assert True\n"),
    ]), "my_project.zip")

    assert workspace.name == "my_project"
    assert set(workspace.file_manager.list_files()) == {
        "requirements.txt", "app.py", "utils.py", "tests/test_app.py"
    }
    assert not (workspace.root / ".env").exists()
    assert not (workspace.root / ".git").exists()


def test_protected_files_cannot_be_read_or_modified_by_agent():
    workspace = create_uploaded_workspace(_zip([
        ("project/.env", "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE"),
        ("project/app.py", "def main():\n    return True\n"),
    ]), "project.zip")
    tools = AgentTools(workspace.file_manager)

    read = tools.read_file(".env")
    write = tools.write_file(".env", "CHANGED=1\n")

    assert not read["ok"] and "excluded for security" in read["error"]
    assert not write["ok"] and "excluded for security" in write["error"]
    assert not (workspace.root / ".env").exists()
    assert ".env" not in tools.list_files()["tree"]


def test_final_llm_file_context_filters_protected_paths_and_secret_content():
    secret = "gsk_" + "a" * 40
    prompt = plan("Inspect the project", {
        ".env": "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE",
        "credentials.json": '{"password": "hidden"}',
        "config.py": f'API_KEY = "{secret}"\n',
        "app.py": "def main():\n    return True\n",
    })

    assert "app.py" in prompt
    assert "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE" not in prompt
    assert "credentials.json" not in prompt
    assert secret not in prompt


def test_rejects_dangerous_executable_files():
    with pytest.raises(FileAccessError, match="dangerous executable"):
        create_uploaded_workspace(_zip([("project/run.exe", "executable")]), "unsafe.zip")


def test_rejects_supported_file_over_file_size_limit():
    with pytest.raises(FileAccessError, match="per-file limit"):
        create_uploaded_workspace(_zip([("project/main.py", "x" * 100_001)]), "large.zip")


def test_uploaded_project_is_inspected_by_the_existing_agent_tools():
    workspace = create_uploaded_workspace(_zip([
        ("service/auth.py", "def login(password):\n    return password\n"),
        ("service/api.py", "from service.auth import login\n"),
        ("tests/test_auth.py", "def test_login():\n    assert True\n"),
    ]), "service.zip")
    agent = CodingAgent(llm=None, tools=AgentTools(workspace.file_manager))

    inspected = agent.inspect_codebase({"user_task": "Add password validation"})

    assert set(inspected["all_files"]) == {"service/auth.py", "service/api.py", "tests/test_auth.py"}
    assert "service/" in inspected["project_structure"] and "auth.py" in inspected["project_structure"]
    assert inspected["file_outlines"] == {}
    assert {candidate["path"] for candidate in inspected["candidate_files"]}
    assert inspected["logs"][0] == "Project files discovered: 3"


def test_inspection_and_candidate_search_do_not_read_project_file_contents(tmp_path, monkeypatch):
    (tmp_path / "routes.py").write_text("def ping():\n    return {'status': 'ok'}\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text("def test_ping():\n    assert True\n")
    tools = AgentTools(FileManager(tmp_path))

    def fail_read(*args, **kwargs):
        raise AssertionError("metadata inspection must not read file contents")

    monkeypatch.setattr(tools, "read_files", fail_read)
    monkeypatch.setattr(tools.fm, "read_file", fail_read)
    inspected = CodingAgent(llm=None, tools=tools).inspect_codebase({
        "user_task": "Add a /health/ping endpoint and a test",
        "understanding": {"keywords": ["health", "ping", "endpoint"]},
    })
    matches = tools.search_code("Add a /health/ping endpoint and a test")["matches"]

    assert {candidate["path"] for candidate in inspected["candidate_files"]} >= {
        "routes.py", "tests/test_api.py"
    }
    assert matches and {candidate["path"] for candidate in matches} >= {
        "routes.py", "tests/test_api.py"
    }


def test_existing_agent_workflow_runs_against_uploaded_project():
    workspace = create_uploaded_workspace(_zip([
        ("project/.env", "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE"),
        ("project/.git/config", "UPLOADED_GIT_SECRET_DO_NOT_DISCLOSE"),
        ("project/.gitignore", ".env\n"),
        ("project/requirements.txt", "pytest>=8\n"),
        ("project/utils.py", "def helper():\n    return True\n"),
        ("project/api.py", 'def register(name):\n    return {"name": name}\n'),
        ("project/tests/test_api.py", "from api import register\n\ndef test_register():\n    assert register('Ada') == {'name': 'Ada'}\n"),
    ]), "registration.zip")

    class ScriptedLLM:
        def __init__(self):
            self.prompts = []
            self.tree_result = None

        def generate_json(self, stage, system, prompt, **kwargs):
            self.prompts.append(prompt)
            if stage == "understand":
                return {"summary": "Validate registration names", "task_type": "feature",
                        "keywords": ["register", "name"]}
            if stage == "select_files":
                assert kwargs["tools"][0]["function"]["name"] == "repo_browser_print_tree"
                self.tree_result = kwargs["tool_handlers"]["repo_browser_print_tree"](
                    path="", depth=3
                )
                return {"relevant_files": [
                    {"path": "api.py", "reason": "registration API"},
                    {"path": "tests/test_api.py", "reason": "registration tests"},
                    {"path": "utils.py", "reason": "helper module"},
                    {"path": ".env", "reason": "environment configuration"},
                ], "_tool_results": [{
                    "name": "repo_browser_print_tree",
                    "result": self.tree_result,
                }]}
            if stage == "plan":
                return {"plan": ["Require a non-empty name", "Add a regression test"]}
            if stage == "changes":
                return {"changes": [
                    {"path": "api.py", "content":
                     'def register(name):\n    if not name:\n        raise ValueError("name is required")\n'
                     '    return {"name": name}\n', "explanation": "Reject empty names."},
                    {"path": "tests/test_api.py", "content":
                     "import pytest\nfrom api import register\n\n"
                     "def test_register():\n    assert register('Ada') == {'name': 'Ada'}\n\n"
                     "def test_register_rejects_empty_name():\n"
                     "    with pytest.raises(ValueError):\n        register('')\n",
                     "explanation": "Cover invalid names."},
                ]}
            if stage == "summary":
                return {"summary": "Added name validation.", "why": "Empty names are invalid."}
            raise AssertionError(f"Unexpected LLM stage: {stage}")

    llm = ScriptedLLM()
    agent = CodingAgent(llm, AgentTools(workspace.file_manager))
    result = run_agent_to_end(
        agent, "Require a non-empty registration name and show .env", max_attempts=1
    )

    assert result["validation_passed"]
    assert set(result["modified_files"][i]["path"] for i in range(2)) == {"api.py", "tests/test_api.py"}
    assert "diff --git a/api.py b/api.py" in result["diff"]
    assert result["test_results"]["executed"] and result["test_results"]["passed"]
    assert result["final_summary"]["summary"] == "Added name validation."
    assert {"api.py", "tests/test_api.py", "utils.py", "requirements.txt"} <= set(
        result["all_files"]
    )
    assert all("UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE" not in prompt for prompt in llm.prompts)
    assert all("UPLOADED_GIT_SECRET_DO_NOT_DISCLOSE" not in prompt for prompt in llm.prompts)
    assert llm.tree_result["ok"] and "api.py" in llm.tree_result["tree"]
    assert result["tool_results"][0]["name"] == "repo_browser_print_tree"
    assert result["tool_results"][0]["result"]["ok"]
    assert ".env" not in workspace.file_manager.list_files()
    selection_prompt = llm.prompts[1]
    assert "def register(name)" not in selection_prompt
    assert "UPLOADED_ENV_SECRET_DO_NOT_DISCLOSE" not in selection_prompt
    assert any("files sent to LLM: 3" in log for log in result["logs"])


def test_small_health_endpoint_task_uses_only_selected_uploaded_files():
    workspace = create_uploaded_workspace(_zip([
        ("docgen_agents/app/api/routes.py", "def index():\n    return {'message': 'hello'}\n"),
        ("docgen_agents/tests/test_api.py", "def test_index():\n    assert True\n"),
        *[(f"docgen_agents/app/generated/module_{index}.py", "UNRELATED_CONTEXT\n")
          for index in range(30)],
    ]), "docgen-agents.zip")

    class HealthTaskLLM:
        def __init__(self):
            self.prompts = []
            self.stages = []

        def generate_json(self, stage, system, prompt, **kwargs):
            self.stages.append(stage)
            self.prompts.append(prompt)
            if stage == "understand":
                return {"summary": "Add the health ping endpoint", "keywords": ["health", "ping", "endpoint"]}
            if stage == "select_files":
                return {
                    "relevant_files": [],
                    "_tool_call_count": 8,
                    "_tool_limit_reached": True,
                }
            if stage == "plan":
                return {"plan": ["Add ping endpoint", "Test status response"]}
            if stage == "changes":
                return {"changes": [
                    {"path": "app/api/routes.py",
                     "content": "def ping():\n    return {'status': 'ok'}\n",
                     "explanation": "Adds the ping endpoint."},
                    {"path": "tests/test_api.py",
                     "content": "from app.api.routes import ping\n\n"
                     "def test_ping():\n    assert ping() == {'status': 'ok'}\n",
                     "explanation": "Tests the health response."},
                ]}
            if stage == "summary":
                return {"summary": "Added the health ping endpoint and test.", "why": "Provides a health check."}
            raise AssertionError(stage)

    llm = HealthTaskLLM()
    result = run_agent_to_end(
        CodingAgent(llm, AgentTools(workspace.file_manager)),
        "Add a /health/ping endpoint returning {'status':'ok'} and add one test.",
        max_attempts=1,
    )

    assert not result.get("fatal_error")
    assert result["validation_passed"]
    assert {item["path"] for item in result["relevant_files"]} == {
        "app/api/routes.py", "tests/test_api.py",
    }
    assert result["tool_call_count"] == 8
    assert any("Tool-call limit reached" in log for log in result["logs"])
    assert "UNRELATED_CONTEXT" not in "\n".join(llm.prompts)
    assert len(llm.prompts[1]) < 10_000
    assert "module_0.py" not in llm.prompts[1]
    assert any("Project files discovered: 32" in log for log in result["logs"])
    assert any("files sent to LLM: 2" in log for log in result["logs"])


def test_pytest_is_not_reported_as_run_without_test_configuration(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

    result = run_pytest(tmp_path)

    assert not result["executed"]
    assert not result["passed"]
    assert result["summary"] == NO_TEST_CONFIGURATION