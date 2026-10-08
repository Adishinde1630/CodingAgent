"""The agent's tools. Each returns a plain dict and never raises - failures become {'ok': False}."""
from __future__ import annotations

from app import code_analyzer, diff_manager, validator
from app.config import redact
from app.file_manager import FileAccessError, FileManager
from app.protected_files import is_protected_path


class AgentTools:
    def __init__(self, fm: FileManager, test_timeout: int = validator.DEFAULT_TIMEOUT):
        self.fm = fm
        self.test_timeout = test_timeout

    def list_files(self) -> dict:
        try:
            files = self.fm.list_files()
            return {"ok": True, "files": files, "tree": code_analyzer.build_tree(files)}
        except Exception as exc:
            return {"ok": False, "error": redact(exc), "files": [], "tree": ""}

    def print_tree(self, path: str = "", depth: int = 3) -> dict:
        """Return workspace tree metadata for the optional Groq repo-browser tool."""
        if not isinstance(path, str) or not isinstance(depth, int) or depth < 1:
            return {"ok": False, "error": "Invalid tree path or depth.", "tree": "", "file_count": 0}
        listing = self.list_files()
        prefix = path.strip().replace("\\", "/").strip("/")
        files = listing["files"]
        if prefix:
            files = [name for name in files if name == prefix or name.startswith(prefix + "/")]
        if depth < 8:
            files = [
                name for name in files
                if len(name.split("/")) - (len(prefix.split("/")) if prefix else 0) <= depth
            ]
        return {
            "tree": code_analyzer.build_tree(files),
            "file_count": len(files),
            "ok": listing["ok"],
        }

    def read_file(self, path: str) -> dict:
        if is_protected_path(path):
            return {"ok": False, "path": path,
                    "error": "Protected files are excluded for security reasons."}
        try:
            clean, _ = self.fm.resolve(path)
            return {"ok": True, "path": clean, "content": self.fm.read_file(path)}
        except FileAccessError as exc:
            return {"ok": False, "path": path, "error": str(exc)}

    def read_files(self, paths: list[str]) -> dict:
        files, errors = {}, {}
        for p in paths:
            res = self.read_file(p)
            if res["ok"]:
                files[res["path"]] = res["content"]
            else:
                errors[str(p)] = res["error"]
        return {"ok": bool(files), "files": files, "errors": errors}

    def search_code(self, query: str, max_results: int = 10) -> dict:
        try:
            return {
                "ok": True,
                "matches": code_analyzer.search_paths(self.fm.list_files(), query, max_results),
            }
        except Exception as exc:
            return {"ok": False, "error": redact(exc), "matches": []}

    def write_file(self, path: str, content: str) -> dict:
        if is_protected_path(path):
            return {"ok": False, "path": path,
                    "error": "Protected files are excluded for security reasons."}
        try:
            clean, _ = self.fm.resolve(path)
            return {"ok": True, "path": clean, "status": self.fm.write_file(path, content)}
        except FileAccessError as exc:
            return {"ok": False, "path": path, "error": str(exc)}

    def get_diff(self) -> dict:
        try:
            originals = self.fm.originals
            currents = {p: self.fm.current_content(p) for p in originals}
            diff, stats = diff_manager.make_diff(originals, currents)
            return {"ok": True, "diff": diff, "files": stats}
        except Exception as exc:
            return {"ok": False, "error": redact(exc), "diff": "", "files": []}

    def run_tests(self) -> dict:
        return validator.run_pytest(self.fm.root, timeout=self.test_timeout)

    def validate_project(self) -> dict:
        files = self.fm.list_files()
        syntax = validator.check_syntax(self.fm.root, files)
        imports = [] if syntax else validator.check_imports(self.fm.root, files)
        return {"passed": not syntax and not imports, "checked_files": [f for f in files if f.endswith(".py")],
                "syntax_errors": syntax, "import_errors": imports}

    def rollback(self) -> dict:
        try:
            return {"ok": True, "restored": self.fm.rollback()}
        except Exception as exc:
            return {"ok": False, "error": redact(exc), "restored": []}
