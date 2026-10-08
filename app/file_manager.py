"""Sandboxed file access for the agent.

Every path the agent (or the LLM) supplies goes through ``FileManager.resolve``
which blocks traversal, secrets, VCS/venv folders and unsupported file types.
Originals are recorded on first write so changes can be diffed and rolled back.
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

from app.config import contains_secret

BLOCKED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".idea", ".vscode", ".streamlit", "site-packages",
}
BLOCKED_NAME_PATTERNS = [
    re.compile(r"^\.env(\..*)?$", re.I),
    re.compile(r"^secrets?\.(toml|json|ya?ml)$", re.I),
    re.compile(r".*\.(pem|key|p12|pfx|crt)$", re.I),
    re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$", re.I),
    re.compile(r"^\.netrc$", re.I),
]
ALLOWED_EXTENSIONS = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
    ".html", ".css", ".js", ".sql",
}
MAX_FILE_BYTES = 100_000
MAX_FILES = 500


class FileAccessError(Exception):
    """Raised for any rejected or failed file operation (safe to show to users)."""


class FileManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileAccessError(f"Workspace does not exist: {self.root.name}")
        # path -> original content (None when the file did not exist before)
        self._originals: dict[str, str | None] = {}

    # ------------------------------------------------------------------ paths
    def resolve(self, rel_path: str) -> tuple[str, Path]:
        """Validate a user/LLM supplied path; return (clean_posix_path, absolute_path)."""
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise FileAccessError("Invalid path: path must be a non-empty string.")
        raw = rel_path.strip().replace("\\", "/")
        if "\x00" in raw:
            raise FileAccessError("Invalid path: null byte.")
        if raw.startswith("/") or raw.startswith("~") or re.match(r"^[A-Za-z]:", raw):
            raise FileAccessError(f"Invalid path '{rel_path}': absolute paths are not allowed.")
        pure = PurePosixPath(raw)
        parts = [p for p in pure.parts if p not in (".", "")]
        if not parts:
            raise FileAccessError("Invalid path: empty path.")
        if ".." in parts:
            raise FileAccessError(f"Invalid path '{rel_path}': path traversal is not allowed.")
        for part in parts:
            if part in BLOCKED_DIRS or part == ".git":
                raise FileAccessError(f"Access to '{rel_path}' is blocked (protected folder '{part}').")
        name = parts[-1]
        if any(p.match(name) for p in BLOCKED_NAME_PATTERNS):
            raise FileAccessError(f"Access to '{rel_path}' is blocked (sensitive file).")
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise FileAccessError(
                f"Unsupported file type for '{rel_path}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        clean = "/".join(parts)
        absolute = (self.root / clean).resolve()
        if not absolute.is_relative_to(self.root):
            raise FileAccessError(f"Invalid path '{rel_path}': resolves outside the workspace.")
        return clean, absolute

    # ---------------------------------------------------------------- reading
    def list_files(self) -> list[str]:
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in BLOCKED_DIRS)
            for fname in sorted(filenames):
                full = Path(dirpath) / fname
                rel = full.relative_to(self.root).as_posix()
                try:
                    self.resolve(rel)
                    if full.is_symlink() or full.stat().st_size > MAX_FILE_BYTES:
                        continue
                except (FileAccessError, OSError):
                    continue
                found.append(rel)
                if len(found) >= MAX_FILES:
                    return sorted(found)
        return sorted(found)

    def read_file(self, rel_path: str) -> str:
        clean, absolute = self.resolve(rel_path)
        if not absolute.exists() or not absolute.is_file():
            raise FileAccessError(f"File not found: {clean}")
        size = absolute.stat().st_size
        if size > MAX_FILE_BYTES:
            raise FileAccessError(f"File too large ({size} bytes, limit {MAX_FILE_BYTES}): {clean}")
        try:
            data = absolute.read_bytes()
            if b"\x00" in data:
                raise FileAccessError(f"Binary file not supported: {clean}")
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise FileAccessError(f"File is not valid UTF-8 text: {clean}")
        except OSError as exc:
            raise FileAccessError(f"Could not read {clean}: {exc.strerror or exc}")
        if contains_secret(text):
            raise FileAccessError(f"Access to '{clean}' is blocked: it appears to contain a secret.")
        return text

    def exists(self, rel_path: str) -> bool:
        _, absolute = self.resolve(rel_path)
        return absolute.is_file()

    # ---------------------------------------------------------------- writing
    def write_file(self, rel_path: str, content: str) -> str:
        """Write a file inside the workspace. Returns 'created', 'modified' or 'unchanged'."""
        clean, absolute = self.resolve(rel_path)
        if not isinstance(content, str):
            raise FileAccessError(f"Content for '{clean}' must be text.")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise FileAccessError(f"Refusing to write '{clean}': content exceeds {MAX_FILE_BYTES} bytes.")
        if contains_secret(content):
            raise FileAccessError(f"Refusing to write '{clean}': content appears to contain a secret.")
        if absolute.exists() and not absolute.is_file():
            raise FileAccessError(f"'{clean}' is not a regular file.")
        existed = absolute.is_file()
        old = absolute.read_text(encoding="utf-8") if existed else None
        if clean not in self._originals:           # save original content once
            self._originals[clean] = old
        if existed and old == content:
            return "unchanged"
        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
            if not absolute.parent.resolve().is_relative_to(self.root):
                raise FileAccessError(f"Invalid path '{rel_path}': resolves outside the workspace.")
            absolute.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise FileAccessError(f"Could not write {clean}: {exc.strerror or exc}")
        return "modified" if existed else "created"

    # ------------------------------------------------------ change tracking
    @property
    def originals(self) -> dict[str, str | None]:
        return dict(self._originals)

    def current_content(self, clean_path: str) -> str | None:
        _, absolute = self.resolve(clean_path)
        return absolute.read_text(encoding="utf-8") if absolute.is_file() else None

    def changed_files(self) -> list[dict]:
        """Files whose content really differs from the saved original."""
        changes = []
        for path, original in sorted(self._originals.items()):
            current = self.current_content(path)
            if current == original:
                continue
            changes.append({"path": path, "status": "added" if original is None else "modified"})
        return changes

    def rollback(self) -> list[str]:
        """Restore every touched file to its original content (delete files the agent created)."""
        restored = []
        for path, original in self._originals.items():
            _, absolute = self.resolve(path)
            if original is None:
                if absolute.is_file():
                    absolute.unlink()
                    restored.append(path)
            elif not absolute.is_file() or absolute.read_text(encoding="utf-8") != original:
                absolute.write_text(original, encoding="utf-8", newline="\n")
                restored.append(path)
        self._originals.clear()
        return sorted(restored)

    def reset_baseline(self) -> None:
        """Treat the current workspace state as the new 'original' for future diffs."""
        self._originals.clear()
