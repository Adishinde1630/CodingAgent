"""Temporary project workspace creation, including safe ZIP extraction."""
from __future__ import annotations

import io
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from app.file_manager import FileAccessError, FileManager, MAX_FILES, MAX_FILE_BYTES
from app.protected_files import is_protected_path

MAX_ZIP_UPLOAD_MB = 500
MAX_ZIP_BYTES = MAX_ZIP_UPLOAD_MB * 1024 * 1024
MAX_ZIP_ENTRIES = 1000
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 50 * 1024 * 1024
MAX_PATH_DEPTH = 16
_DANGEROUS_EXTENSIONS = {
    ".app", ".bat", ".bin", ".com", ".dll", ".dylib", ".exe", ".jar", ".msi",
    ".ps1", ".scr", ".sh", ".so",
}
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.I)


@dataclass
class ProjectWorkspace:
    parent: Path
    root: Path
    name: str
    file_manager: FileManager


def _safe_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    raw = info.filename
    windows_path = PureWindowsPath(raw)
    normalized = raw.replace("\\", "/")
    if (not raw or "\x00" in raw or normalized.startswith("/") or windows_path.is_absolute()
            or windows_path.drive or re.match(r"^[A-Za-z]:", normalized)):
        raise FileAccessError(f"Unsafe ZIP path: {raw!r}.")
    path = normalized[:-1] if info.is_dir() and normalized.endswith("/") else normalized
    parts = tuple(path.split("/"))
    if (not parts or any(part in ("", ".", "..") for part in parts)
            or len(parts) > MAX_PATH_DEPTH):
        raise FileAccessError(f"Unsafe or excessively nested ZIP path: {raw!r}.")
    if any(part.endswith((".", " ")) or _WINDOWS_RESERVED.match(part) for part in parts):
        raise FileAccessError(f"Unsafe ZIP path: {raw!r}.")
    return parts


def create_uploaded_workspace(zip_bytes: bytes, upload_name: str) -> ProjectWorkspace:
    """Extract supported text files to a private workspace without using extractall."""
    if not zip_bytes:
        raise FileAccessError("The uploaded ZIP file is empty.")
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise FileAccessError(f"ZIP file exceeds the {MAX_ZIP_BYTES // (1024 * 1024)} MB upload limit.")

    parent = Path(tempfile.mkdtemp(prefix="coding_agent_upload_"))
    extraction_root = parent / "extracted"
    extraction_root.mkdir()
    extracted_paths: list[tuple[str, ...]] = []
    seen_paths: set[str] = set()
    total_archive_size = 0
    total_extracted_size = 0

    try:
        try:
            archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except (OSError, zipfile.BadZipFile) as exc:
            raise FileAccessError("The uploaded file is not a valid ZIP archive.") from exc

        with archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_ENTRIES:
                raise FileAccessError(f"ZIP contains too many entries (limit {MAX_ZIP_ENTRIES}).")
            manager = FileManager(extraction_root)

            for info in entries:
                parts = _safe_parts(info)
                mode = (info.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if kind == stat.S_IFLNK:
                    raise FileAccessError(f"ZIP symlinks are not allowed: {info.filename!r}.")
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise FileAccessError(f"Special ZIP entries are not allowed: {info.filename!r}.")
                if info.is_dir():
                    continue
                if kind == stat.S_IFDIR:
                    raise FileAccessError(f"Invalid ZIP file entry: {info.filename!r}.")
                if info.file_size < 0 or info.file_size > MAX_ARCHIVE_BYTES:
                    raise FileAccessError("ZIP contains an entry exceeding the extraction size limit.")
                total_archive_size += info.file_size
                if total_archive_size > MAX_ARCHIVE_BYTES:
                    raise FileAccessError("ZIP expands beyond the 100 MB archive limit.")

                rel_path = "/".join(parts)
                if is_protected_path(rel_path):
                    continue
                name = parts[-1]
                if Path(name).suffix.lower() in _DANGEROUS_EXTENSIONS:
                    raise FileAccessError(f"ZIP contains a dangerous executable file: {info.filename!r}.")

                try:
                    clean, destination = manager.resolve(rel_path)
                except FileAccessError as exc:
                    if "Unsupported file type" in str(exc):
                        continue
                    raise
                if clean.casefold() in seen_paths:
                    raise FileAccessError(f"ZIP contains duplicate paths: {info.filename!r}.")
                seen_paths.add(clean.casefold())
                if info.file_size > MAX_FILE_BYTES:
                    raise FileAccessError(
                        f"ZIP file '{clean}' exceeds the {MAX_FILE_BYTES}-byte per-file limit."
                    )
                total_extracted_size += info.file_size
                if total_extracted_size > MAX_EXTRACTED_BYTES:
                    raise FileAccessError("ZIP contains more than 50 MB of supported project files.")

                try:
                    with archive.open(info) as source:
                        content = source.read(MAX_FILE_BYTES + 1)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise FileAccessError(f"Could not safely read ZIP entry '{clean}'.") from exc
                if len(content) != info.file_size:
                    raise FileAccessError(f"ZIP entry size did not match its metadata: '{clean}'.")
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if b"\x00" in content:
                    continue
                if len(text.encode("utf-8")) > MAX_FILE_BYTES:
                    raise FileAccessError(f"ZIP file '{clean}' exceeds the per-file limit.")

                if len(extracted_paths) >= MAX_FILES:
                    raise FileAccessError(f"ZIP contains more than {MAX_FILES} supported project files.")
                if not destination.resolve().is_relative_to(extraction_root.resolve()):
                    raise FileAccessError(f"ZIP path escapes the temporary workspace: '{clean}'.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                for parent_path in destination.parents:
                    if parent_path == extraction_root:
                        break
                    if parent_path.is_symlink():
                        raise FileAccessError(f"ZIP path contains a symlink: '{clean}'.")
                try:
                    with destination.open("xb") as target:
                        target.write(content)
                except FileExistsError as exc:
                    raise FileAccessError(f"ZIP contains conflicting paths: '{clean}'.") from exc
                extracted_paths.append(parts)

        if not extracted_paths:
            raise FileAccessError("The ZIP contains no supported UTF-8 text project files.")

        top_level = {parts[0] for parts in extracted_paths}
        one_project_folder = len(top_level) == 1 and all(len(parts) > 1 for parts in extracted_paths)
        if one_project_folder:
            project_name = next(iter(top_level))
            project_root = extraction_root / project_name
        else:
            project_name = PureWindowsPath(upload_name).name
            project_name = Path(project_name).stem or "uploaded-project"
            project_root = extraction_root
        if not project_root.is_dir():
            raise FileAccessError("Could not determine the uploaded project root.")
        return ProjectWorkspace(parent, project_root, project_name, FileManager(project_root))
    except Exception:
        shutil.rmtree(parent, ignore_errors=True)
        raise