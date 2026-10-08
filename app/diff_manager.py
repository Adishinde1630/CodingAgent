"""Git-style unified diffs computed from saved originals vs. current content."""
from __future__ import annotations

import difflib


def make_diff(originals: dict[str, str | None], currents: dict[str, str | None]) -> tuple[str, list[dict]]:
    """Return (diff_text, per_file_stats). Unchanged files are omitted."""
    chunks: list[str] = []
    stats: list[dict] = []
    for path in sorted(originals):
        old, new = originals[path], currents.get(path)
        if old == new or new is None:
            continue
        old_lines = (old or "").splitlines()
        new_lines = new.splitlines()
        header = [f"diff --git a/{path} b/{path}"]
        if old is None:
            header += ["new file mode 100644", "--- /dev/null", f"+++ b/{path}"]
        else:
            header += [f"--- a/{path}", f"+++ b/{path}"]
        body = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=3))[2:]  # drop ---/+++ from difflib
        added = sum(1 for l in body if l.startswith("+"))
        removed = sum(1 for l in body if l.startswith("-"))
        chunks.append("\n".join(header + body))
        stats.append({"path": path, "status": "added" if old is None else "modified",
                      "added": added, "removed": removed})
    return ("\n".join(chunks) + "\n" if chunks else ""), stats
