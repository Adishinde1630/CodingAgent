"""LangGraph state shared by all agent nodes."""
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # inputs / settings
    user_task: str
    max_attempts: int          # total code-generation attempts (1 = no automatic repair)
    auto_rollback: bool
    # understanding & inspection
    understanding: dict
    project_structure: str
    all_files: list
    candidate_files: list
    tool_results: list
    tool_call_count: int
    file_outlines: dict
    relevant_files: list       # [{"path", "reason"}], initialized per run
    no_relevant_files: bool
    file_contents: dict        # path -> content actually read by tools
    # planning & changes
    plan: str | list
    changes: list
    proposed_changes: list     # [{"path", "content", "explanation"}]
    apply_results: list        # what the write tool really did per file
    modified_files: list       # [{"path", "status"}] derived from the real filesystem
    diff: str
    diff_stats: list
    # validation
    syntax_results: dict
    test_results: str | dict
    validation_passed: bool
    feedback: str              # failure info fed back into the next attempt
    attempt: int
    rolled_back: bool
    # output
    final_summary: str | dict
    fatal_error: str
    rate_limited: bool
    # accumulating fields (lists are appended, not replaced)
    errors: Annotated[list, operator.add]
    logs: Annotated[list, operator.add]
    attempt_history: Annotated[list, operator.add]


ACCUMULATING_KEYS = ("errors", "logs", "attempt_history")
