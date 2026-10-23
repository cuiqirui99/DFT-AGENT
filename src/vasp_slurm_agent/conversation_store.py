"""Private, persistent task conversations and their original structures."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _folder(folder):
    path = Path(folder).expanduser().absolute()
    if path.is_symlink() or path.parent.is_symlink() or path.parent.name != ".conversations":
        raise ValueError("Invalid conversation folder.")
    if not path.is_dir():
        raise ValueError("The conversation folder is unavailable.")
    return path.resolve()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Invalid conversation file path.")
    parts = Path(relative).parts
    if any(part in {"..", "."} for part in parts):
        raise ValueError("Invalid conversation file path.")
    candidate = root.joinpath(*parts)
    if any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise ValueError("Conversation files cannot be symbolic links.")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Invalid conversation file path.")
    return candidate


def _run_path(root, value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Invalid conversation run path.")
    path = Path(value)
    runs_root = root.parent.parent
    try:
        relative = path.relative_to(runs_root)
    except ValueError:
        raise ValueError("A conversation run must stay in its run folder.") from None
    if not relative.parts or relative.parts[0] == ".conversations":
        raise ValueError("Invalid conversation run path.")
    checked = _inside(runs_root, str(relative))
    if checked.exists() and not checked.is_dir():
        raise ValueError("Invalid conversation run folder.")
    return checked


def _validate(root, record):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Invalid conversation record.")
    try:
        identifier = str(uuid.UUID(record["id"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ValueError("Invalid conversation identity.") from None
    if identifier != root.name or record["id"] != identifier or record.get("engine") not in {"vasp", "qe", "abinit", "cp2k", "elk"}:
        raise ValueError("Invalid conversation identity or engine.")
    if type(record.get("revision")) is not int or record["revision"] < 0:
        raise ValueError("Invalid conversation revision.")
    if not isinstance(record.get("goal"), str):
        raise ValueError("Invalid conversation goal.")
    if not isinstance(record.get("messages"), list) or any(
        not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
        or not isinstance(message.get("content"), str) for message in record["messages"]
    ):
        raise ValueError("Invalid conversation messages.")
    if record.get("pending") is not None and not isinstance(record["pending"], dict):
        raise ValueError("Invalid pending conversation action.")
    if (not isinstance(record.get("runs"), list) or any(not isinstance(run, str) for run in record["runs"])
            or len(set(record["runs"])) != len(record["runs"])):
        raise ValueError("Invalid conversation runs.")
    for run in record["runs"]:
        _run_path(root, run)
    if record.get("active_run") is not None and record["active_run"] not in record["runs"]:
        raise ValueError("The active run is not part of this conversation.")
    if record.get("source") not in {"source.cif", "POSCAR"}:
        raise ValueError("Invalid conversation structure path.")
    files = [{"path": record["source"], "sha256": record.get("source_sha256")}]
    images = record.get("images")
    if not isinstance(images, list):
        raise ValueError("Invalid conversation images.")
    seen = set()
    for image in images:
        if not isinstance(image, dict) or set(image) != {"path", "sha256"}:
            raise ValueError("Invalid conversation image.")
        path = image["path"]
        if not isinstance(path, str) or not path.startswith("images/") or path in seen:
            raise ValueError("Invalid conversation image path.")
        seen.add(path)
        files.append(image)
    for item in files:
        path = _inside(root, item["path"])
        if not path.is_file() or _digest(path) != item["sha256"]:
            raise ValueError("A saved conversation structure has changed or is missing.")
    for field in ("created_at", "updated_at"):
        if not isinstance(record.get(field), str):
            raise ValueError("Invalid conversation timestamp.")
        try:
            datetime.fromisoformat(record[field])
        except ValueError:
            raise ValueError("Invalid conversation timestamp.") from None
    return record


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field in conversation record.")
        result[key] = value
    return result


def load(folder):
    """Read a conversation and verify that its copied structures are unchanged."""
    root = _folder(folder)
    path = _inside(root, "conversation.json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON number.")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read the saved conversation.") from exc
    return _validate(root, record)


def _atomic_write(path, record):
    encoded = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".conversation-", delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lock(root):
    path = _inside(root, ".conversation.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Invalid conversation lock.")
        if os.name == "nt":
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def create(runs_root, engine, source_path, *, images=None):
    """Copy only the selected structure inputs into a new private conversation."""
    if engine not in {"vasp", "qe", "abinit", "cp2k", "elk"}:
        raise ValueError("Choose VASP, Quantum ESPRESSO, ABINIT, CP2K or Elk.")
    parent = Path(runs_root).expanduser().resolve() / ".conversations"
    if parent.is_symlink():
        raise ValueError("The conversation folder cannot be a symbolic link.")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent.chmod(0o700)
    root = parent / str(uuid.uuid4())
    root.mkdir(mode=0o700)
    try:
        def copy(source, destination):
            source = Path(source).expanduser()
            if source.is_symlink() or not source.is_file():
                raise ValueError("Choose a regular structure file.")
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o600)
            return _digest(destination)

        name = "source.cif" if Path(source_path).suffix.lower() == ".cif" else "POSCAR"
        source_hash = copy(source_path, root / name)
        copied_images = []
        for index, image in enumerate(images or [], 1):
            (root / "images").mkdir(mode=0o700, exist_ok=True)
            suffix = ".cif" if Path(image).suffix.lower() == ".cif" else ".vasp"
            relative = f"images/{index:03d}{suffix}"
            copied_images.append({"path": relative, "sha256": copy(image, root / relative)})
        now = _now()
        record = {"schema_version": 1, "id": root.name, "engine": engine, "goal": "",
                  "messages": [], "pending": None, "active_run": None, "runs": [], "revision": 0,
                  "source": name, "source_sha256": source_hash, "images": copied_images,
                  "created_at": now, "updated_at": now}
        _validate(root, record)
        _atomic_write(root / "conversation.json", record)
    except Exception:
        shutil.rmtree(root)
        raise
    return root


def save(folder, record, *, expected_revision=None):
    """Atomically save one turn; reject edits based on an older saved revision."""
    root = _folder(folder)
    updated = deepcopy(record)
    _validate(root, updated)
    expected = updated["revision"] if expected_revision is None else expected_revision
    with _lock(root):
        current = load(root)
        if type(expected) is not int or expected != current["revision"]:
            raise ValueError("This conversation changed. Reload it before sending another message.")
        for field in ("id", "engine", "source", "source_sha256", "images", "created_at"):
            if updated[field] != current[field]:
                raise ValueError("The original conversation inputs cannot be changed.")
        updated["revision"] = current["revision"] + 1
        updated["updated_at"] = _now()
        _atomic_write(_inside(root, "conversation.json"), updated)
    return updated


def list_conversations(runs_root, engine=None):
    """List readable conversations, newest first, without opening other engines."""
    if engine is not None and engine not in {"vasp", "qe", "abinit", "cp2k", "elk"}:
        raise ValueError("Choose VASP, Quantum ESPRESSO, ABINIT, CP2K or Elk.")
    parent = Path(runs_root).expanduser().resolve() / ".conversations"
    if parent.is_symlink() or not parent.is_dir():
        return []
    records = []
    for folder in parent.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            record = load(folder)
        except (OSError, ValueError, TypeError):
            continue
        if engine is not None and record["engine"] != engine:
            continue
        title = next((message["content"].strip() for message in record["messages"]
                      if message["role"] == "user" and message["content"].strip()), "New conversation")
        records.append({**record, "folder": str(folder), "title": title.splitlines()[0][:100]})
    return sorted(records, key=lambda record: record["updated_at"], reverse=True)


def attach_run(folder, run_dir):
    """Save a portable transcript snapshot and a reference to its live conversation."""
    root = _folder(folder)
    record = load(root)
    run = _run_path(root, str(Path(run_dir).expanduser().absolute()))
    if not run.is_dir():
        raise ValueError("Prepare the run before attaching its conversation.")
    snapshot = {**record, "conversation_id": record["id"], "conversation_path": str(root)}
    _atomic_write(_inside(run, "conversation.json"), snapshot)
    return snapshot
