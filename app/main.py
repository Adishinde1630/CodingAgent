"""Streamlit UI for the AI Coding Agent.  Run with:  streamlit run app/main.py"""
import shutil
import hashlib
import sys
import tempfile
from pathlib import Path

# `streamlit run app/main.py` puts app/ (not the repo root) on sys.path - add the root for `import app.*`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.agent import NODE_LABELS, CodingAgent, run_agent  # noqa: E402
from app.code_analyzer import build_tree  # noqa: E402
from app.file_manager import FileAccessError, FileManager  # noqa: E402
from app.llm import GroqClient, LLMError, MissingAPIKeyError  # noqa: E402
from app.tools import AgentTools  # noqa: E402
from app.workspace import MAX_ZIP_UPLOAD_MB, ProjectWorkspace, create_uploaded_workspace  # noqa: E402

SAMPLE_PROJECT_NAME = "sample_project"
SAMPLE_SOURCE = "Sample Project"
UPLOAD_SOURCE = "Upload ZIP"
DEMO_TASK = "Add input validation to the user registration API and write tests for invalid email addresses."
LANGS = {".py": "python", ".md": "markdown", ".json": "json", ".toml": "toml", ".yaml": "yaml",
         ".yml": "yaml", ".html": "html", ".css": "css", ".js": "javascript", ".sql": "sql", ".ini": "ini"}

st.set_page_config(page_title="AI Coding Agent", page_icon="🤖", layout="wide")


# ----------------------------------------------------------------- workspace
def _save_active_workspace() -> None:
    active_key = st.session_state.get("active_workspace_key")
    active = st.session_state.get("workspaces", {}).get(active_key)
    if active:
        active["result"] = st.session_state.get("result")
        active["file_sel"] = st.session_state.get("file_sel")


def _activate_workspace(key: str) -> None:
    active_key = st.session_state.get("active_workspace_key")
    if active_key is not None:
        _save_active_workspace()
    workspace = st.session_state.workspaces[key]
    st.session_state.active_workspace_key = key
    st.session_state.fm = workspace["file_manager"]
    st.session_state.workspace_parent = str(workspace["parent"])
    st.session_state.project_root = str(workspace["root"])
    st.session_state.project_name = workspace["name"]
    st.session_state.result = workspace.get("result")
    if workspace.get("file_sel") is None:
        st.session_state.pop("file_sel", None)
    else:
        st.session_state["file_sel"] = workspace["file_sel"]


def _store_workspace(key: str, workspace: ProjectWorkspace) -> None:
    old = st.session_state.workspaces.get(key)
    if old:
        shutil.rmtree(old["parent"], ignore_errors=True)
    st.session_state.workspaces[key] = {
        "parent": workspace.parent,
        "root": workspace.root,
        "name": workspace.name,
        "file_manager": workspace.file_manager,
        "result": None,
        "file_sel": None,
    }
    if st.session_state.get("active_workspace_key") == key:
        st.session_state.active_workspace_key = None
    _activate_workspace(key)


def new_workspace(source: str = SAMPLE_SOURCE) -> None:
    """Create a fresh temporary copy for the selected project source."""
    if source == SAMPLE_SOURCE:
        parent = Path(tempfile.mkdtemp(prefix="coding_agent_ws_"))
        target = parent / SAMPLE_PROJECT_NAME
        try:
            shutil.copytree(config.SAMPLE_PROJECT_DIR, target,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            workspace = ProjectWorkspace(parent, target, SAMPLE_PROJECT_NAME, FileManager(target))
        except Exception:
            shutil.rmtree(parent, ignore_errors=True)
            raise
        _store_workspace("sample", workspace)
        return

    zip_bytes = st.session_state.get("uploaded_zip_bytes")
    zip_name = st.session_state.get("uploaded_zip_name")
    if not zip_bytes or not zip_name:
        raise FileAccessError("Upload a project ZIP before creating an uploaded workspace.")
    workspace = create_uploaded_workspace(zip_bytes, zip_name)
    _store_workspace("upload", workspace)


def _deactivate_workspace() -> None:
    _save_active_workspace()
    st.session_state.active_workspace_key = None
    st.session_state.result = None
    st.session_state["file_sel"] = []


def set_demo_task() -> None:
    st.session_state.task_text = DEMO_TASK


def revert_changes() -> None:
    restored = st.session_state.fm.rollback()
    st.session_state.result = {**st.session_state.result, "reverted": restored}


st.session_state.setdefault("project_source", SAMPLE_SOURCE)
st.session_state.setdefault("workspaces", {})
if "sample" not in st.session_state.workspaces:
    new_workspace(SAMPLE_SOURCE)


def lang_for(path: str) -> str | None:
    return LANGS.get(Path(path).suffix.lower())


def safe(text) -> str:
    return config.redact(text or "")


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    project_source = st.radio("Project Source", [SAMPLE_SOURCE, UPLOAD_SOURCE], key="project_source")
    uploaded_file = None
    if project_source == UPLOAD_SOURCE:
        uploaded_file = st.file_uploader("Upload your project ZIP", type=["zip"],
                                         max_upload_size=MAX_ZIP_UPLOAD_MB, key="project_zip")
        if uploaded_file is not None:
            zip_bytes = uploaded_file.getvalue()
            upload_digest = hashlib.sha256(zip_bytes).hexdigest()
            if upload_digest != st.session_state.get("uploaded_zip_digest"):
                try:
                    workspace = create_uploaded_workspace(zip_bytes, uploaded_file.name)
                    _store_workspace("upload", workspace)
                    st.session_state.uploaded_zip_bytes = zip_bytes
                    st.session_state.uploaded_zip_name = uploaded_file.name
                    st.session_state.uploaded_zip_digest = upload_digest
                    st.session_state.upload_error = None
                    st.session_state.upload_error_digest = None
                except Exception as exc:
                    st.session_state.upload_error = safe(exc)
                    st.session_state.upload_error_digest = upload_digest
            if st.session_state.get("upload_error_digest") == upload_digest:
                st.error(st.session_state.upload_error)

    upload_failed = bool(uploaded_file is not None and
                         st.session_state.get("upload_error_digest") == hashlib.sha256(
                             uploaded_file.getvalue()).hexdigest())
    project_ready = False
    if project_source == SAMPLE_SOURCE:
        _activate_workspace("sample")
        project_ready = True
    elif not upload_failed and "upload" in st.session_state.workspaces:
        _activate_workspace("upload")
        project_ready = True
    else:
        _deactivate_workspace()

    repairs = st.slider("Automatic repair attempts", 0, 3, 1,
                        help="If validation fails, feed the real error output back to the model and retry.")
    auto_rollback = st.checkbox("Auto-rollback if validation fails", value=False)
    fresh_start = st.checkbox("Start from a fresh project each run", value=True)
    if st.button("Reset workspace"):
        try:
            new_workspace(project_source)
            st.rerun()
        except FileAccessError as exc:
            st.error(safe(exc))

fm: FileManager | None = st.session_state.get("fm") if project_ready else None
project_name = st.session_state.get("project_name") if project_ready else None
project_label = (SAMPLE_SOURCE if project_source == SAMPLE_SOURCE
                 else f"Uploaded: {project_name}" if project_name else UPLOAD_SOURCE)

# -------------------------------------------------------------------- header
st.title("AI Coding Agent")
st.caption("An AI agent that understands a codebase, plans coding changes, modifies files, and validates the result.")

tab_task, tab_files, tab_plan, tab_changes, tab_diff, tab_validation, tab_summary = st.tabs(
    ["Task", "Project Files", "Agent Plan", "Changes", "Diff", "Validation", "Final Summary"])

# ---------------------------------------------------------------------- task
with tab_task:
    st.subheader("Coding Task")
    st.write(f"Working on project: **{project_label}**")
    if not project_ready:
        st.warning("Upload a valid project ZIP to begin working on an uploaded project.")
    task_text = st.text_area("Describe the coding task...", key="task_text", height=140,
                             placeholder=(f"Example: {DEMO_TASK}" if project_source == SAMPLE_SOURCE
                                          else "Describe a task for the uploaded project..."))
    c1, c2, _ = st.columns([1, 1, 4])
    run_clicked = c1.button("Run Coding Agent", type="primary")
    if project_source == SAMPLE_SOURCE:
        c2.button("Use demo task", on_click=set_demo_task)
    progress = st.container()

    if run_clicked:
        task = task_text.strip()
        llm = None
        if not task:
            st.error("Please enter a coding task.")
        elif not project_ready:
            st.error("Upload a valid project ZIP before running the agent.")
        else:
            try:
                llm = GroqClient(api_key=config.get_api_key() or "", model=config.get_model_name())
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:  # e.g. SDK problem
                st.error(f"Could not initialise the Groq client: {safe(exc)}")
        if llm:
            if fresh_start:
                try:
                    new_workspace(project_source)
                except FileAccessError as exc:
                    st.error(safe(exc))
                    llm = None
            else:
                st.session_state.fm.reset_baseline()
            if llm:
                fm = st.session_state.fm
                agent = CodingAgent(llm, AgentTools(fm))
                final: dict = {}
                with progress:
                    with st.status("Running coding agent...", expanded=True) as status:
                        try:
                            for node, update, state in run_agent(agent, task, max_attempts=1 + repairs,
                                                                 auto_rollback=auto_rollback):
                                label = NODE_LABELS.get(node, node)
                                if node == "generate_changes" and state.get("attempt", 1) > 1:
                                    label += f" (repair attempt {state.get('attempt', 1)})"
                                if update.get("fatal_error"):
                                    st.write(f"❌ {label}: {safe(update['fatal_error'])}")
                                else:
                                    st.write(f"✅ {label}")
                                if node == "select_relevant_files":
                                    relevant = state.get("relevant_files", [])
                                    if relevant:
                                        st.markdown("**Relevant files:** " + ", ".join(
                                            f"`{r['path']}`" for r in relevant
                                        ))
                                    elif state.get("no_relevant_files"):
                                        st.info(state.get("final_summary", {}).get(
                                            "summary", "No relevant files were identified."
                                        ))
                                if node == "create_plan":
                                    plan = update.get("plan", [])
                                    if plan:
                                        st.markdown("### Agent Plan\n" + "\n".join(
                                            f"{i}. {s}" for i, s in enumerate(plan, 1)
                                        ))
                                if node == "run_validation":
                                    validation_passed = update.get("validation_passed", False)
                                    test_results = update.get("test_results", {})
                                    st.write(("✓ Tests passed" if validation_passed else "✗ Validation failed")
                                             + f" — {test_results.get('summary', '')}")
                                final = state
                        except Exception as exc:  # last-resort safety net
                            final = {**final, "fatal_error": f"Agent crashed: {safe(exc)}"}
                        if final.get("no_relevant_files"):
                            status.update(label="No relevant files identified", state="complete")
                        elif final.get("rate_limited"):
                            status.update(label="Agent stopped because the LLM rate limit was reached.",
                                          state="error")
                        elif final.get("fatal_error"):
                            status.update(label="Agent stopped with an error", state="error")
                        elif final.get("validation_passed"):
                            status.update(label="Done - validation passed", state="complete")
                        else:
                            status.update(label="Finished - validation did not pass", state="error")
                st.session_state.result = dict(final)

    result = st.session_state.get("result")
    if result and result.get("rate_limited"):
        st.warning("Agent stopped because the LLM rate limit was reached. "
                   "Please wait for the Groq rate limit to reset and try again.")
    if result and result.get("fatal_error"):
        st.error(safe(result["fatal_error"]))

result = st.session_state.get("result")
EMPTY = "Run the coding agent from the **Task** tab to see results here."

# ------------------------------------------------------------- project files
with tab_files:
    st.subheader(f"Codebase: {project_label}")
    if not project_ready or fm is None:
        st.info("Upload a ZIP file to view its project files.")
    else:
        files = fm.list_files()
        modified = {m["path"]: m["status"] for m in fm.changed_files()}
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Project tree**")
            st.code(build_tree(files), language=None)
            if result and result.get("relevant_files"):
                st.markdown("**Files the agent selected**")
                for r in result.get("relevant_files", []):
                    st.markdown(f"- `{r['path']}` — {r['reason']}")
        with right:
            st.markdown("**File contents** (select one or more files)")
            if project_source == SAMPLE_SOURCE:
                defaults = [f for f in ("app.py", "validators.py") if f in files]
            else:
                defaults = [f for f in files if f.endswith(".py") and not Path(f).name.startswith("test_")][:2]
            if not defaults:
                defaults = files[:1]
            selected_before = st.session_state.get("file_sel", defaults)
            st.session_state["file_sel"] = [f for f in selected_before if f in files]
            selected = st.multiselect("Files", files, key="file_sel", label_visibility="collapsed")
            for path in selected:
                tag = f" ({modified[path]} by agent)" if path in modified else ""
                with st.expander(f"{path}{tag}", expanded=True):
                    try:
                        st.code(fm.read_file(path), language=lang_for(path))
                    except FileAccessError as exc:
                        st.error(safe(exc))

# ---------------------------------------------------------------------- plan
with tab_plan:
    st.subheader("Agent Plan")
    if not result or not result.get("plan"):
        st.info(EMPTY)
    else:
        u = result.get("understanding", {})
        st.markdown(f"**Task understanding:** {u.get('summary', '')}  \n*Type:* `{u.get('task_type', '')}`")
        st.markdown("**Relevant files**")
        for r in result.get("relevant_files", []):
            st.markdown(f"- `{r['path']}` — {r['reason']}")
        st.markdown("**Plan**\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(result["plan"], 1)))
        with st.expander("Agent log"):
            st.code("\n".join(safe(l) for l in result.get("logs", [])), language=None)

# ------------------------------------------------------------------- changes
with tab_changes:
    st.subheader("Modified Files")
    if not result:
        st.info(EMPTY)
    else:
        mods = result.get("modified_files", [])
        if not mods:
            st.warning("The agent did not modify any files.")
        explanations = {c["path"]: c["explanation"] for c in result.get("proposed_changes", [])}
        for m in mods:
            with st.expander(f"{m['path']} — {m['status']}", expanded=False):
                if explanations.get(m["path"]):
                    st.markdown(f"**What and why:** {explanations[m['path']]}")
                try:
                    st.code(fm.read_file(m["path"]) if not result.get("reverted") and not result.get("rolled_back")
                            else "(changes were reverted)", language=lang_for(m["path"]))
                except FileAccessError as exc:
                    st.error(safe(exc))
        problems = [r for r in result.get("apply_results", []) if not r.get("ok")]
        if problems:
            st.markdown("**Changes the safety checks refused**")
            for r in problems:
                st.error(safe(r.get("error", "")))

# ---------------------------------------------------------------------- diff
with tab_diff:
    st.subheader("Diff")
    if not result:
        st.info(EMPTY)
    elif not result.get("diff"):
        st.info("No differences - nothing was modified.")
    else:
        if result.get("rolled_back") or result.get("reverted"):
            st.info("These changes have been rolled back; the diff shows what the agent had changed.")
        st.code(safe(result["diff"]), language="diff")
        st.dataframe([{"file": s["path"], "status": s["status"], "+": s["added"], "-": s["removed"]}
                      for s in result.get("diff_stats", [])], hide_index=True)
        st.download_button("Download patch", safe(result["diff"]), file_name="coding_agent.patch")

# ---------------------------------------------------------------- validation
with tab_validation:
    st.subheader("Validation Result")
    if not result or not result.get("test_results"):
        st.info(EMPTY)
    else:
        t, syn = result["test_results"], result.get("syntax_results", {})
        st.markdown(("✓ Tests executed" if t["executed"] else "✗ Tests did not run (see stderr)"))
        if not t["executed"] and t.get("summary", "").startswith("No supported test configuration"):
            st.warning(t["summary"])
        if t["timed_out"]:
            st.markdown("✗ Tests timed out")
        elif t["executed"]:
            st.markdown("✓ Tests passed" if t["passed"] else "✗ Tests failed")
        st.markdown("✓ Syntax and import checks passed" if syn.get("passed") else "✗ Syntax/import checks failed")
        st.caption(f"{t.get('summary', '')} · exit code {t.get('exit_code')} · {t.get('duration')}s · `{t.get('command')}`")
        for e in syn.get("syntax_errors", []) + syn.get("import_errors", []):
            st.error(f"{e['path']}: {safe(e['error'])}")
        if len(result.get("attempt_history", [])) > 1 or any(not h["passed"] for h in result.get("attempt_history", [])):
            st.markdown("**Attempts**")
            st.dataframe([{"attempt": h["attempt"], "passed": h["passed"], "pytest": h["summary"]}
                          for h in result["attempt_history"]], hide_index=True)
        with st.expander("pytest stdout", expanded=not t["passed"]):
            st.code(safe(t["stdout"]) or "(empty)", language=None)
        if t["stderr"]:
            with st.expander("pytest stderr"):
                st.code(safe(t["stderr"]), language=None)

# ------------------------------------------------------------------- summary
with tab_summary:
    st.subheader("Summary")
    if not result or not result.get("final_summary"):
        st.info(EMPTY)
    else:
        fs = result["final_summary"]
        st.write(safe(fs["summary"]))
        st.subheader("Files Modified")
        mods = result.get("modified_files", [])
        st.markdown("\n".join(f"- `{m['path']}` ({m['status']})" for m in mods) or "_No files were modified._")
        st.subheader("Why")
        st.write(safe(fs["why"]))
        st.subheader("Validation")
        t = result.get("test_results", {})
        if result.get("validation_passed"):
            st.success(f"✓ Tests executed and passed — {t.get('summary', '')}")
        else:
            st.error(f"✗ Validation did not pass — {t.get('summary', '')}")
        with st.expander("pytest output"):
            st.code(safe(t.get("stdout", "")), language=None)
        st.subheader("Diff")
        st.code(safe(result.get("diff", "")) or "(no changes)", language="diff")
        st.subheader("Limitations")
        st.markdown("\n".join(f"- {safe(x)}" for x in fs["limitations"]))
        if result.get("rolled_back"):
            st.info("Validation failed, so the changes were rolled back automatically.")
        elif result.get("reverted") is not None:
            st.info("All changes were reverted.")
        elif mods:
            st.button("Revert all changes", on_click=revert_changes)
