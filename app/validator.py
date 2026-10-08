"""Validation: syntax checks, import checks and a sandboxed pytest run."""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 60
MAX_OUTPUT_CHARS = 20_000
NO_TEST_CONFIGURATION = (
    "No supported test configuration detected. Code changes were generated, but automated validation "
    "could not be executed."
)


def _safe_env() -> dict[str, str]:
    """Minimal environment for subprocesses - API keys are deliberately NOT passed on."""
    keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
            "LANG", "LC_ALL", "VIRTUAL_ENV")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ""})
    return env


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) > MAX_OUTPUT_CHARS:
        value = "...[truncated]...\n" + value[-MAX_OUTPUT_CHARS:]
    return value


def check_syntax(root: Path, files: list[str]) -> list[dict]:
    """Parse every .py file; return a list of syntax errors."""
    errors = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            source = (root / rel).read_text(encoding="utf-8")
            ast.parse(source, filename=rel)
        except SyntaxError as exc:
            errors.append({"path": rel, "line": exc.lineno, "error": f"SyntaxError: {exc.msg}"})
        except (OSError, UnicodeDecodeError) as exc:
            errors.append({"path": rel, "line": None, "error": f"Could not read file: {exc}"})
    return errors


def check_imports(root: Path, files: list[str], timeout: int = 30) -> list[dict]:
    """Import each top-level, non-test module in a subprocess to catch import-time errors."""
    modules = [
        f[:-3] for f in files
        if f.endswith(".py") and "/" not in f and not f.startswith("test_") and f != "conftest.py"
    ]
    if not modules:
        return []
    code = (
        "import sys, importlib, traceback\n"
        "sys.path.insert(0, '.')\n"
        "bad = 0\n"
        "for m in sys.argv[1:]:\n"
        "    try:\n        importlib.import_module(m)\n"
        "    except BaseException as e:\n"
        "        bad += 1\n        print(f'{m}: {type(e).__name__}: {e}')\n"
        "sys.exit(1 if bad else 0)\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code, *modules], cwd=root, capture_output=True,
                              text=True, timeout=timeout, env=_safe_env())
    except subprocess.TimeoutExpired:
        return [{"path": "(imports)", "line": None, "error": f"Import check timed out after {timeout}s"}]
    if proc.returncode == 0:
        return []
    return [{"path": "(imports)", "line": None, "error": line}
            for line in _text(proc.stdout + proc.stderr).strip().splitlines() if line.strip()]


def run_pytest(root: Path, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run pytest inside the workspace with a timeout and capture the real output."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header", "-p", "no:cacheprovider"]
    start = time.monotonic()
    result = {"command": "pytest -q --tb=short", "executed": False, "passed": False, "timed_out": False,
              "exit_code": None, "stdout": "", "stderr": "", "summary": "", "duration": 0.0}
    root = Path(root)
    if not ((root / "pytest.ini").is_file() or (root / "pyproject.toml").is_file()
            or (root / "tests").is_dir()):
        result["summary"] = NO_TEST_CONFIGURATION
        result["duration"] = round(time.monotonic() - start, 2)
        return result
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout,
                              env=_safe_env())
    except subprocess.TimeoutExpired as exc:
        result.update(timed_out=True, stdout=_text(exc.stdout), stderr=_text(exc.stderr),
                      summary=f"Test run timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        result.update(stderr=f"Could not start pytest: {exc}", summary="Could not start pytest")
    else:
        stdout, stderr = _text(proc.stdout), _text(proc.stderr)
        lines = [l for l in stdout.strip().splitlines() if l.strip()]
        summary = lines[-1] if lines else ""
        if proc.returncode == 5:
            summary = "No tests were collected"
        result.update(executed=True, exit_code=proc.returncode, stdout=stdout, stderr=stderr,
                      passed=proc.returncode == 0, summary=re.sub(r"^=+\s*|\s*=+$", "", summary))
    result["duration"] = round(time.monotonic() - start, 2)
    return result
