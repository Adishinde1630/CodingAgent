"""Lightweight code analysis: project tree, file outlines and keyword search/ranking."""
from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "add", "make", "write", "when", "then",
    "should", "would", "could", "please", "need", "want", "all", "any", "also", "are", "was", "has",
    "have", "not", "but", "use", "using", "new", "existing", "code", "file", "files", "function", "tests",
    "test", "api", "update", "create", "implement", "ensure", "check", "input",
}
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "ed", "es", "s")


def build_tree(paths: list[str]) -> str:
    """Render paths as an ASCII tree."""
    tree: dict = {}
    for p in paths:
        node = tree
        for part in PurePosixPath(p).parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0]))  # dirs first
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if child else ''}")
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(lines)


def outline_file(path: str, content: str) -> str:
    """Compact description of a file (signatures for Python, first lines otherwise)."""
    if path.endswith(".py"):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return "(could not parse - syntax error)"
        out: list[str] = []
        doc = ast.get_docstring(tree)
        if doc:
            out.append(f'"""{doc.splitlines()[0]}"""')
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"def {node.name}({ast.unparse(node.args)})" + _doc(node))
            elif isinstance(node, ast.ClassDef):
                out.append(f"class {node.name}" + _doc(node))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"    def {sub.name}({ast.unparse(sub.args)})")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(ast.unparse(node))
        return "\n".join(out) or "(empty)"
    lines = [l.strip() for l in content.splitlines() if l.strip()][:5]
    return "\n".join(lines) or "(empty)"


def _doc(node) -> str:
    doc = ast.get_docstring(node)
    return f"  # {doc.splitlines()[0]}" if doc else ""


def stem(word: str) -> str:
    """Crude stemmer: strip a common suffix, then keep at most 6 letters so that
    'registration' and 'register' (or 'validation' and 'validator') share a stem."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:6]


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    """Split free text / identifiers into unique lowercase keyword stems."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.replace("_", " "))
    seen: list[str] = []
    for w in words:
        w = w.lower()
        if len(w) < 3 or w in STOPWORDS:
            continue
        s = stem(w)
        if s not in seen:
            seen.append(s)
    return seen[:limit]


def search_code(files: dict[str, str], query: str, max_results: int = 10) -> list[dict]:
    """Rank files by keyword relevance to ``query`` (filename, definitions, content hits)."""
    keywords = extract_keywords(query)
    results = []
    for path, content in files.items():
        lowered_path = path.lower()
        lines = content.splitlines()
        score = 0
        matched: list[tuple[int, str]] = []
        for kw in keywords:
            if kw in lowered_path:
                score += 5
            hits, def_bonus = 0, 0
            for n, line in enumerate(lines, 1):
                low = line.lower()
                if kw in low:
                    hits += 1
                    if re.match(r"\s*(def|class)\s", low):
                        def_bonus = 3            # at most once per keyword
                    if len(matched) < 5 and (n, line.strip()) not in matched:
                        matched.append((n, line.strip()[:120]))
            score += min(hits, 5) + def_bonus
        if score and (path.startswith("tests/") or PurePosixPath(path).name.startswith("test_")):
            score = max(1, score - 3)        # prefer source files; tests are secondary context
        if score:
            results.append({"path": path, "score": score, "matches": matched[:5]})
    results.sort(key=lambda r: (-r["score"], r["path"]))
    return results[:max_results]


def search_paths(paths: list[str], query: str, max_results: int = 10) -> list[dict]:
    """Rank candidates using task words and path metadata without reading file contents."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", query))
    keywords = list(dict.fromkeys(
        stem(word.lower()) for word in words
        if len(word) >= 3 and word.lower() not in STOPWORDS
    ))
    low_query = query.lower()
    results = []
    for path in paths:
        normalized = path.lower()
        if is_generated_path(path):
            continue
        name = PurePosixPath(path).name.lower()
        path_words = set(re.findall(r"[a-z0-9]+", normalized.replace("_", " ")))
        score = sum(5 for keyword in keywords if keyword in normalized)
        score += sum(2 for keyword in keywords if keyword in path_words)

        if any(term in low_query for term in ("endpoint", "route", "health", "api")):
            if name in {"route.py", "routes.py", "api.py", "views.py"} or "/api/" in f"/{normalized}/":
                score += 8
            if name in {"app.py", "main.py", "server.py"}:
                score += 5
        if any(term in low_query for term in ("test", "tests", "pytest")):
            if name.startswith("test_") or "/tests/" in f"/{normalized}/":
                score += 6
        if name in {"main.py", "app.py", "server.py"}:
            score += 1
            if any(term in low_query for term in ("register", "registration", "endpoint", "api")):
                score += 10
        if path.lower().endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java")):
            score += 1
        if name.startswith("test_") or "/tests/" in f"/{normalized}/":
            score = max(1, score - 2)
        if score:
            results.append({"path": path, "score": score, "matches": []})
    results.sort(key=lambda result: (-result["score"], result["path"]))
    return results[:max_results]


def is_generated_path(path: str) -> bool:
    """Identify common generated/build artifacts that should not enter model context."""
    normalized = path.lower()
    parts = set(PurePosixPath(normalized).parts)
    name = PurePosixPath(path).name.lower()
    return bool(
        parts.intersection({"build", "dist", "coverage", "generated", ".next", "vendor"})
        or name.endswith((".min.js", ".min.css"))
        or ".generated." in name
    )


def related_tests(path: str, all_files: list[str]) -> list[str]:
    """Test files that look like they cover ``path`` (tests/test_<stem>.py)."""
    stem_name = PurePosixPath(path).stem
    return [f for f in all_files if PurePosixPath(f).name == f"test_{stem_name}.py"]
