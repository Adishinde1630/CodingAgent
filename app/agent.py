"""LangGraph workflow for the coding agent.

understand_task -> inspect_codebase -> select_relevant_files -> create_plan -> generate_changes
-> apply_changes -> generate_diff -> run_validation -> (repair loop) -> summarize_result

The LLM only makes decisions (understanding, file choice, plan, code, wording). Everything that
touches the project - listing, reading, writing, diffing, testing - is done by deterministic tools,
and the final report is built from those tool results.
"""
from __future__ import annotations

import ast
import re
from typing import Callable, Iterator

from langgraph.graph import END, START, StateGraph

from app import code_analyzer, prompts
from app.config import redact
from app.llm import LLMError, RateLimitError, parse_json
from app.state import ACCUMULATING_KEYS, AgentState
from app.tools import AgentTools

MAX_RELEVANT_FILES = 6
MAX_FILE_CONTEXT_CHARS = 6_000
MAX_CONTEXT_CHARS = 16_000
MAX_CHANGED_FILES = 8
REPO_BROWSER_PRINT_TREE = {
    "type": "function",
    "function": {
        "name": "repo_browser_print_tree",
        "description": "Print the current project file tree (repo_browser.print_tree).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional project-relative directory path."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "additionalProperties": False,
        },
    },
}

NODE_LABELS = {
    "understand_task": "Understanding the task",
    "inspect_codebase": "Inspecting project structure",
    "select_relevant_files": "Selecting and reading relevant files",
    "create_plan": "Creating implementation plan",
    "generate_changes": "Generating code changes",
    "apply_changes": "Applying changes safely",
    "generate_diff": "Generating diff",
    "run_validation": "Running validation and tests",
    "summarize_result": "Summarizing result",
}


def _strip_fences(code: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\n(.*?)\n```\s*$", code, flags=re.S)
    return m.group(1) if m else code


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


class CodingAgent:
    def __init__(self, llm, tools: AgentTools):
        self.llm = llm
        self.tools = tools
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        nodes: list[tuple[str, Callable]] = [
            ("understand_task", self.understand_task),
            ("inspect_codebase", self.inspect_codebase),
            ("select_relevant_files", self.select_relevant_files),
            ("create_plan", self.create_plan),
            ("generate_changes", self.generate_changes),
            ("apply_changes", self.apply_changes),
            ("generate_diff", self.generate_diff),
            ("run_validation", self.run_validation),
            ("summarize_result", self.summarize_result),
        ]
        g = StateGraph(AgentState)
        for name, fn in nodes:
            g.add_node(name, self._guard(fn))
        g.add_edge(START, "understand_task")
        names = [n for n, _ in nodes]
        for src, nxt in zip(names, names[1:]):
            if src == "run_validation":
                continue
            g.add_conditional_edges(src, self._route(nxt), {nxt: nxt, END: END})
        g.add_conditional_edges("run_validation", self._after_validation,
                                {"generate_changes": "generate_changes",
                                 "summarize_result": "summarize_result", END: END})
        g.add_edge("summarize_result", END)
        return g.compile()

    @staticmethod
    def _route(nxt: str):
        def router(state: AgentState) -> str:
            return END if state.get("fatal_error") or state.get("no_relevant_files") else nxt
        return router

    @staticmethod
    def _after_validation(state: AgentState) -> str:
        if state.get("fatal_error"):
            return END
        if not state.get("validation_passed") and state.get("attempt", 0) < state.get("max_attempts", 1):
            return "generate_changes"          # automatic repair loop
        return "summarize_result"

    @staticmethod
    def _guard(fn: Callable) -> Callable:
        """Turn exceptions inside a node into a fatal_error instead of crashing the graph."""
        def node(state: AgentState) -> dict:
            try:
                return fn(state)
            except RateLimitError as exc:
                msg = str(exc)
                return {"fatal_error": msg, "rate_limited": True, "errors": [msg],
                        "logs": [f"{fn.__name__}: failed - {msg}"]}
            except LLMError as exc:
                msg = str(exc)
            except Exception as exc:  # unexpected tool/logic failure
                msg = f"Unexpected error in {fn.__name__}: {redact(exc)}"
            return {"fatal_error": msg, "errors": [msg], "logs": [f"{fn.__name__}: failed - {msg}"]}
        return node

    # ------------------------------------------------------------------ nodes
    def understand_task(self, state: AgentState) -> dict:
        task = (state.get("user_task") or "").strip()
        if not task:
            raise LLMError("Please enter a coding task.")
        prompt = prompts.understand(task)
        data = self.llm.generate_json(
            "understand", prompts.SYSTEM_PROMPT, prompt, max_output_tokens=384
        )
        keywords = [str(k) for k in data.get("keywords", []) if isinstance(k, (str, int))][:10]
        understanding = {
            "summary": str(data.get("summary") or task),
            "task_type": str(data.get("task_type") or "other"),
            "keywords": keywords,
            "expected_changes": str(data.get("expected_changes") or ""),
        }
        return {"understanding": understanding, "logs": [f"Understood task: {understanding['summary']}"]}

    def inspect_codebase(self, state: AgentState) -> dict:
        listing = self.tools.list_files()
        if not listing["ok"] or not listing["files"]:
            raise RuntimeError(listing.get("error") or "No readable project files found.")
        context_files = [
            path for path in listing["files"] if not code_analyzer.is_generated_path(path)
        ]
        query = state.get("user_task", "")
        understanding = state.get("understanding", {})
        if isinstance(understanding, dict):
            query += " " + " ".join(
                str(keyword) for keyword in understanding.get("keywords", [])
            )
        candidates = self.tools.search_code(query, max_results=12).get("matches", [])
        logs = [
            f"Project files discovered: {len(listing['files'])}",
            f"Candidate files: {len(candidates)}",
        ]
        return {"all_files": listing["files"],
                "project_structure": code_analyzer.build_tree(context_files),
                "candidate_files": candidates, "file_outlines": {}, "logs": logs}

    def select_relevant_files(self, state: AgentState) -> dict:
        task = state.get("user_task", "")
        u = state.get("understanding", {})
        if not isinstance(u, dict):
            u = {}
        query = f"{task} {' '.join(u.get('keywords', []))}"
        candidates = state.get("candidate_files", [])
        if not candidates:
            search = self.tools.search_code(query, max_results=12)
            candidates = search.get("matches", []) if isinstance(search, dict) else []
        all_files = state.get("all_files", [])
        project_structure = state.get("project_structure", "")
        file_outlines = state.get("file_outlines", {})

        selection_prompt = prompts.select_files(
            task, u, project_structure, file_outlines, candidates
        )
        data = self.llm.generate_json(
            "select_files",
            prompts.SYSTEM_PROMPT,
            selection_prompt,
            max_output_tokens=512,
            tools=[REPO_BROWSER_PRINT_TREE],
            tool_handlers={"repo_browser_print_tree": self.tools.print_tree},
        )
        tool_results = data.get("_tool_results", []) if isinstance(data, dict) else []
        if not isinstance(tool_results, list):
            tool_results = []
        tool_call_count = data.get("_tool_call_count", 0) if isinstance(data, dict) else 0
        if not isinstance(tool_call_count, int):
            tool_call_count = 0
        tool_limit_reached = bool(data.get("_tool_limit_reached")) if isinstance(data, dict) else False
        if tool_limit_reached:
            logs = [
                "Tool-call limit reached. Continuing with the information already collected."
            ]
        else:
            logs = []
        if isinstance(data, str):
            try:
                data = parse_json(data)
            except (TypeError, ValueError):
                data = [
                    path for path in all_files
                    if re.search(rf"(?<![\w./-]){re.escape(path)}(?![\w./-])", data)
                ]
        if isinstance(data, dict):
            requested = data.get("relevant_files", [])
        elif isinstance(data, list):
            requested = data
        else:
            requested = []
        if not isinstance(requested, list):
            requested = []

        chosen: list[dict] = []
        for item in requested:
            path = item.get("path") if isinstance(item, dict) else item
            if (isinstance(path, str) and path in all_files
                    and path not in [c["path"] for c in chosen]):
                try:
                    clean, absolute = self.tools.fm.resolve(path)
                except Exception:
                    continue
                if clean != path or not absolute.is_file():
                    continue
                reason = str(item.get("reason", "")) if isinstance(item, dict) else ""
                chosen.append({"path": path, "reason": reason or "selected by the model"})
        if not chosen:  # fallback to keyword ranking if the model returned nothing usable
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                path = candidate.get("path")
                if not isinstance(path, str) or path not in all_files:
                    continue
                try:
                    clean, absolute = self.tools.fm.resolve(path)
                except Exception:
                    continue
                if clean != path or not absolute.is_file():
                    continue
                chosen.append({
                    "path": path,
                    "reason": f"keyword match (score {candidate.get('score', 0)})",
                })
                for t in code_analyzer.related_tests(path, all_files):
                    if t not in [x["path"] for x in chosen]:
                        chosen.append({"path": t, "reason": "test file for a relevant module"})
                if len(chosen) >= MAX_RELEVANT_FILES:
                    break
            logs.append("Model returned no valid files - used keyword ranking fallback")
        chosen = chosen[:MAX_RELEVANT_FILES]
        if not chosen:
            message = ("No relevant files were identified for this task. "
                       "Please provide a more specific coding request.")
            return {
                "relevant_files": [],
                "file_contents": {},
                "tool_results": tool_results,
                "tool_call_count": tool_call_count,
                "no_relevant_files": True,
                "final_summary": {
                    "summary": message,
                    "why": "No existing project files matched the task.",
                    "limitations": [],
                },
                "logs": logs + [message],
            }

        chosen = chosen[:MAX_RELEVANT_FILES]
        read = self.tools.read_files([c["path"] for c in chosen])
        read_files = read.get("files", {}) if isinstance(read, dict) else {}
        contents, total = {}, 0
        for path, text in read_files.items():
            remaining = MAX_CONTEXT_CHARS - total
            if remaining <= 0:
                logs.append(f"Skipped {path}: context size limit reached")
                continue
            marker = "\n...[truncated for model context]"
            limit = min(MAX_FILE_CONTEXT_CHARS, remaining)
            if len(text) > limit:
                text = text[:max(0, limit - len(marker))] + marker
                logs.append(f"Truncated {path} to fit the model context limit")
            contents[path] = text
            total += len(text)
        if not contents:
            errors = read.get("errors", {}) if isinstance(read, dict) else {}
            message = ("No relevant files were identified for this task. "
                       "Please provide a more specific coding request.")
            return {
                "relevant_files": [],
                "file_contents": {},
                "tool_results": tool_results,
                "tool_call_count": tool_call_count,
                "no_relevant_files": True,
                "final_summary": {
                    "summary": message,
                    "why": "; ".join(str(error) for error in errors.values())
                    or "No selected project files could be read.",
                    "limitations": [],
                },
                "logs": logs + [message],
            }
        chosen = [c for c in chosen if c["path"] in contents]
        context_prompt = prompts.plan(task, contents)
        logs.insert(0, f"Relevant files: {len(chosen)}; files sent to LLM: {len(contents)}; "
                        f"selection context: ~{prompts.context_size(selection_prompt):,} characters; "
                        f"file context: ~{prompts.context_size(context_prompt):,} characters")
        logs.insert(1, f"Read {len(contents)} files: {', '.join(contents)}")
        return {"relevant_files": chosen, "file_contents": contents, "logs": logs,
                "tool_results": tool_results,
                "tool_call_count": tool_call_count,
                "errors": [f"{p}: {e}" for p, e in read.get("errors", {}).items()]}

    def create_plan(self, state: AgentState) -> dict:
        prompt = prompts.plan(state.get("user_task", ""), state.get("file_contents", {}))
        data = self.llm.generate_json(
            "plan", prompts.SYSTEM_PROMPT, prompt, max_output_tokens=768
        )
        steps = [re.sub(r"^\s*\d+[.)]\s*", "", str(s)).strip() for s in data.get("plan", []) if str(s).strip()]
        if not steps:
            raise LLMError("The model did not produce an implementation plan.")
        return {
            "plan": steps[:12],
            "logs": [
                f"Plan created with {len(steps[:12])} steps; context: "
                f"~{prompts.context_size(prompt):,} characters"
            ],
        }

    def generate_changes(self, state: AgentState) -> dict:
        attempt = state.get("attempt", 0) + 1
        # Always show the model the CURRENT on-disk state of relevant + already modified files.
        paths = [r["path"] for r in state.get("relevant_files", [])]
        paths += [m["path"] for m in state.get("modified_files", []) if m["path"] not in paths]
        current_read = self.tools.read_files(paths)
        current = current_read.get("files", {}) or state.get("file_contents", {})
        current = dict(list(current.items())[:MAX_RELEVANT_FILES])
        prompt = prompts.changes(
            state.get("user_task", ""), state.get("plan", []), current,
            state.get("feedback", ""),
        )
        data = self.llm.generate_json(
            "changes", prompts.SYSTEM_PROMPT, prompt, max_output_tokens=3072
        )
        proposed = []
        for item in data.get("changes", []):
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("content"), str):
                proposed.append({"path": item["path"].strip(), "content": _strip_fences(item["content"]),
                                 "explanation": str(item.get("explanation", ""))})
        proposed = proposed[:MAX_CHANGED_FILES]
        logs = [
            f"Attempt {attempt}: model proposed changes to {len(proposed)} file(s); "
            f"files sent to LLM: {len(current)}; context: "
            f"~{prompts.context_size(prompt):,} characters"
        ]
        errors = [] if proposed else [f"Attempt {attempt}: the model proposed no file changes"]
        return {"changes": proposed, "proposed_changes": proposed, "attempt": attempt,
                "logs": logs, "errors": errors}

    def apply_changes(self, state: AgentState) -> dict:
        results, errors = [], []
        for ch in state.get("proposed_changes", []):
            path, content = ch["path"], ch["content"]
            if "[truncated for model context]" in state.get("file_contents", {}).get(path, ""):
                msg = f"{path}: rejected - file was truncated in the model context"
                results.append({"path": path, "ok": False, "status": "rejected", "error": msg})
                errors.append(msg)
                continue
            if not content.endswith("\n"):
                content += "\n"
            if path.endswith(".py"):               # never write code that does not even parse
                try:
                    ast.parse(content)
                except SyntaxError as exc:
                    msg = f"{path}: rejected - syntax error line {exc.lineno}: {exc.msg}"
                    results.append({"path": path, "ok": False, "status": "rejected", "error": msg})
                    errors.append(msg)
                    continue
            res = self.tools.write_file(path, content)   # path validation + backup happen inside
            results.append(res)
            if not res["ok"]:
                errors.append(f"{path}: {res['error']}")
        modified = self.tools.fm.changed_files()          # derived from the real filesystem
        logs = [f"{r['path']}: {r.get('status', 'failed')}" for r in results]
        return {"apply_results": results, "modified_files": modified, "logs": logs, "errors": errors}

    def generate_diff(self, state: AgentState) -> dict:
        res = self.tools.get_diff()
        if not res["ok"]:
            raise RuntimeError(res["error"])
        return {"diff": res["diff"], "diff_stats": res["files"],
                "logs": [f"Diff generated for {len(res['files'])} file(s)"]}

    def run_validation(self, state: AgentState) -> dict:
        syntax = self.tools.validate_project()
        tests = self.tools.run_tests()
        modified = state.get("modified_files", [])
        passed = bool(modified) and syntax["passed"] and tests["executed"] and tests["passed"]

        feedback = ""
        if not passed:
            parts = []
            if not modified:
                rejected = [e for e in state.get("errors", []) if "rejected" in e or "blocked" in e]
                parts.append("No files were modified. " + " ".join(rejected[-3:]))
            for e in syntax["syntax_errors"] + syntax["import_errors"]:
                parts.append(f"{e['path']}: {e['error']}")
            if not tests["passed"]:
                if not tests["executed"] and tests.get("summary", "").startswith("No supported test configuration"):
                    parts.append(tests["summary"])
                else:
                    parts.append("pytest output:\n" + _truncate((tests["stdout"] + "\n" + tests["stderr"]).strip()[-6000:], 6000))
            feedback = "\n".join(parts)
        attempt = state.get("attempt", 1)
        history = {"attempt": attempt, "passed": passed, "summary": tests.get("summary", "")}
        return {"syntax_results": syntax, "test_results": tests, "validation_passed": passed,
                "feedback": feedback, "attempt_history": [history],
                "logs": [f"Validation {'passed' if passed else 'FAILED'}: {tests.get('summary') or 'no test summary'}"]}

    def summarize_result(self, state: AgentState) -> dict:
        modified = state.get("modified_files", [])
        passed = state.get("validation_passed", False)
        logs, rolled_back = [], False
        if state.get("auto_rollback") and not passed and modified:
            rolled_back = bool(self.tools.rollback()["ok"])
            logs.append("Validation failed - changes were rolled back automatically")

        explanations = {c["path"]: c["explanation"] for c in state.get("proposed_changes", [])}
        tests = state.get("test_results", {})
        facts = {
            "files_actually_modified": [{**m, "explanation": explanations.get(m["path"], "")} for m in modified],
            "plan": state.get("plan", []),
            "validation_passed": passed,
            "attempts_used": state.get("attempt", 0),
            "pytest_summary": tests.get("summary"), "pytest_exit_code": tests.get("exit_code"),
            "pytest_timed_out": tests.get("timed_out"),
            "pytest_output_tail": (tests.get("stdout", "") or "")[-1000:],
            "syntax_errors": state.get("syntax_results", {}).get("syntax_errors", []),
            "diff_excerpt": _truncate(state.get("diff", ""), 2000),
            "changes_rolled_back": rolled_back,
            "errors": state.get("errors", [])[-6:],
        }
        summary = None
        try:
            prompt = prompts.summary(state.get("user_task", ""), facts)
            data = self.llm.generate_json(
                "summary", prompts.SYSTEM_PROMPT, prompt, max_output_tokens=384
            )
            summary = {"summary": str(data.get("summary", "")), "why": str(data.get("why", "")),
                       "limitations": [str(x) for x in data.get("limitations", [])][:5]}
            logs.append(f"Summary context: ~{prompts.context_size(prompt):,} characters")
        except RateLimitError:
            raise
        except LLMError as exc:
            logs.append(f"Summary model call failed ({exc}); using the tool-based summary")
        if not summary or not summary["summary"]:
            names = ", ".join(m["path"] for m in modified) or "no files"
            summary = {"summary": f"Modified {names}. Validation {'passed' if passed else 'failed'}.",
                       "why": "See the plan and diff for the reasoning behind each change.", "limitations": []}
        summary["limitations"] += [
            "Passing tests only shows the checks that exist; review the diff before using the change.",
            "The model rewrites whole files and may misunderstand ambiguous tasks.",
        ]
        return {"final_summary": summary, "rolled_back": rolled_back, "logs": logs + ["Final summary created"]}


# ------------------------------------------------------------------ public API
def _merge(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key in ACCUMULATING_KEYS:
            state[key] = list(state.get(key, [])) + list(value)
        else:
            state[key] = value


def run_agent(agent: CodingAgent, task: str, max_attempts: int = 2, auto_rollback: bool = False) -> Iterator[tuple[str, dict, dict]]:
    """Run the graph, yielding (node_name, update, accumulated_state) after every node."""
    state: dict = {
        "user_task": (task or "").strip(),
        "max_attempts": max(1, max_attempts),
        "auto_rollback": auto_rollback,
        "understanding": {},
        "project_structure": "",
        "all_files": [],
        "candidate_files": [],
        "tool_results": [],
        "tool_call_count": 0,
        "file_outlines": {},
        "relevant_files": [],
        "no_relevant_files": False,
        "file_contents": {},
        "plan": "",
        "changes": [],
        "proposed_changes": [],
        "apply_results": [],
        "modified_files": [],
        "diff": "",
        "diff_stats": [],
        "syntax_results": {},
        "test_results": "",
        "validation_passed": False,
        "feedback": "",
        "attempt": 0,
        "rolled_back": False,
        "final_summary": "",
        "fatal_error": "",
        "rate_limited": False,
        "errors": [],
        "logs": [],
        "attempt_history": [],
    }
    for chunk in agent.graph.stream(dict(state), stream_mode="updates"):
        for node, update in chunk.items():
            if update:
                _merge(state, update)
                yield node, update, state


def run_agent_to_end(agent: CodingAgent, task: str, **kwargs) -> dict:
    final: dict = {}
    for _, _, state in run_agent(agent, task, **kwargs):
        final = state
    return final
