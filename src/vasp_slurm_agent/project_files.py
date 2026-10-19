"""Project file tools. Edits affect drafts, never prepared calculation inputs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


MAX_PREVIEW_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_PRIVATE_NAMES = {"potcar", "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json",
                  "auth.json", ".env", "vasp_license"}
_SECRET = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|"
    r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b|"
    r"(?im:(?:^|[,{])\s*[\"']?(?:api[_-]?key|password|private[_-]?key|otp|access[_-]?token)"
    r"[\"']?\s*[:=]\s*[\"']?\S+)"
)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _checked(workspace, project_id, relative, *, draft=False, must_exist=False):
    value = os.fspath(relative)
    path = PurePosixPath(value)
    allowed = {"files"} if draft else {"files", "runs", "artifacts"}
    if path.is_absolute() or not path.parts or path.parts[0] not in allowed:
        raise ValueError("Choose a file inside this project's files, runs or artifacts folder.")
    if any(part.startswith(".") or part.lower() in _PRIVATE_NAMES
           or part.lower().startswith("potcar.") for part in path.parts):
        raise ValueError("Use the secure setup tools for credentials and private potential libraries.")
    return workspace.resolve_path(project_id, value, must_exist=must_exist)


def _safe_content(data):
    # Binary scientific inputs are retained privately and never sent by this tool.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    if _SECRET.search(text):
        raise ValueError("This file appears to contain a credential. Use the secure connection settings.")


@contextmanager
def _lock(root):
    from .workflow import _acquire, _release

    path = root / ".files.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        _acquire(stream)
        try:
            yield
        finally:
            _release(stream)


def _atomic(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".draft-", delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_file(workspace, project_id, relative, data, *, expected_sha256=None):
    """Create a draft, or replace exactly the revision the user reviewed."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, bytes) or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a file smaller than 512 MiB.")
    _safe_content(data)
    root = workspace.project_paths(project_id)["root"]
    with _lock(root):
        previous = None
        written = False
        try:
            # Archive and delete acquire this same database write transaction.
            with workspace.edit_project(project_id) as record_event:
                destination = _checked(workspace, project_id, relative, draft=True)
                if destination.exists():
                    if not destination.is_file():
                        raise ValueError("Choose a regular file.")
                    previous = destination.read_bytes()
                    if expected_sha256 is None or _hash(previous) != expected_sha256:
                        raise ValueError("The file changed. Open its current revision before saving.")
                elif expected_sha256 is not None:
                    raise ValueError("The original file is missing. Save a new file explicitly.")
                digest = _hash(data)
                if previous is not None and previous != data:
                    # Revision files stay private and out of the file browser.
                    history = root / "files" / ".history"
                    if history.is_symlink():
                        raise ValueError("Invalid file history folder.")
                    snapshot = history / (_hash(previous) + ".bin")
                    if snapshot.is_symlink():
                        raise ValueError("Invalid file revision.")
                    if not snapshot.exists():
                        _atomic(snapshot, previous)
                _atomic(destination, data)
                written = True
                record_event("file.saved", {
                    "path": PurePosixPath(relative).as_posix(), "sha256": digest,
                    "previous_sha256": _hash(previous) if previous is not None else None,
                    "size": len(data),
                })
        except BaseException:
            if written:
                if previous is None:
                    destination.unlink(missing_ok=True)
                else:
                    _atomic(destination, previous)
            raise
        return {"path": PurePosixPath(relative).as_posix(), "sha256": digest, "size": len(data)}


def import_file(workspace, project_id, source, *, name=None):
    """Copy one explicitly selected local file into project drafts."""
    source = Path(source).expanduser()
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a regular local file smaller than 512 MiB.")
    name = name or source.name
    if Path(name).name != name or name in {"", ".", ".."} or "\\" in name:
        raise ValueError("Enter a file name, without folders.")
    return write_file(workspace, project_id, (PurePosixPath("files") / name).as_posix(), source.read_bytes())


def read_file(workspace, project_id, relative):
    """Return a bounded text preview and the revision required to save an edit."""
    path = _checked(workspace, project_id, relative, must_exist=True)
    if not path.is_file() or path.stat().st_size > MAX_PREVIEW_BYTES:
        raise ValueError("Text preview supports regular files up to 2 MiB. Download larger files.")
    data = path.read_bytes()
    _safe_content(data)
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("This is a binary file. Download it to view locally.") from None
    return {"path": PurePosixPath(relative).as_posix(), "content": content, "sha256": _hash(data),
            "size": len(data), "editable": PurePosixPath(relative).parts[0] == "files"}


def list_files(workspace, project_id, *, limit=500):
    """List visible regular files without reading their contents."""
    if type(limit) is not int or not 1 <= limit <= 5000:
        raise ValueError("Choose a file limit between 1 and 5000.")
    roots = workspace.project_paths(project_id)
    result = []
    for name in ("files", "runs", "artifacts"):
        for directory, folders, files in os.walk(roots[name], followlinks=False):
            folders[:] = sorted(folder for folder in folders
                                if not folder.startswith(".") and not (Path(directory) / folder).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                relative = path.relative_to(roots["root"])
                try:
                    checked = _checked(workspace, project_id, relative.as_posix(), must_exist=True)
                    if not checked.is_file():
                        continue
                    stat = checked.stat()
                except (ValueError, OSError):
                    continue
                result.append({"path": relative.as_posix(), "size": stat.st_size,
                               "modified_ns": stat.st_mtime_ns,
                               "editable": relative.parts[0] == "files"})
                if len(result) >= limit:
                    return result
    return result
