"""Shared protected-path checks for project ingestion and LLM context."""
from __future__ import annotations

import re

from app.file_manager import BLOCKED_DIRS, BLOCKED_NAME_PATTERNS

_BLOCKED_DIRS = {name.casefold() for name in BLOCKED_DIRS}
_SECRET_FILE_NAMES = re.compile(
    r"(?:^|[._-])(?:credentials?|tokens?|secrets?)(?:[._-]|$)", re.I
)


def is_protected_path(path: str) -> bool:
    """Return whether a path belongs to a protected file or directory."""
    if not isinstance(path, str):
        return False
    parts = tuple(part for part in path.replace("\\", "/").split("/") if part)
    if any(part.casefold() in _BLOCKED_DIRS for part in parts):
        return True
    if not parts:
        return False
    name = parts[-1]
    return (
        name.casefold() == ".gitignore"
        or _SECRET_FILE_NAMES.search(name) is not None
        or any(pattern.match(name) for pattern in BLOCKED_NAME_PATTERNS)
    )
