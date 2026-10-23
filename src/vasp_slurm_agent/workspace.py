"""Private project storage. Solver files remain the execution authority."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import uuid

from .paths import app_home
from .secret_guard import reject_ordinary_text, redact


_MIGRATIONS = ((1, "001_workspace.sql"), (2, "002_connections.sql"), (3, "003_credential_slots.sql"),
               (4, "004_step_attempts.sql"), (5, "005_open_source_engines.sql"))
_FOLDERS = ("files", "runs", "memory", "config", "artifacts")
_ENGINES = {"vasp", "qe", "abinit", "cp2k", "elk", "wannier90", "python"}
_SECRET_KEYS = {"password", "passphrase", "apikey", "privatekey", "otp", "totp",
                "credential", "credentials", "secret", "secretkey", "authorization",
                "accesstoken", "refreshtoken", "bearertoken", "devicecode"}
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
             "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id(value):
    try:
        normalized = str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid workspace identity.") from None
    if normalized != value:
        raise ValueError("Invalid workspace identity.")
    return normalized


def _text(value, name, maximum=200, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise ValueError(f"Invalid {name}.")
    if not empty and (not value.strip() or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Enter a valid {name}.")
    reject_ordinary_text(value)
    return value if empty else value.strip()


def _json(value):
    """Reject credential fields, including nested environment-variable names."""
    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("Metadata keys must be strings.")
                normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                if normalized in _SECRET_KEYS or any(normalized.endswith(suffix) for suffix in
                        ("apikey", "password", "privatekey", "accesstoken", "refreshtoken")):
                    raise ValueError("Credentials belong in secure settings, not project metadata.")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, str):
            reject_ordinary_text(item)
    check(value)
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Project metadata must be finite JSON data.") from error


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("Project metadata must be an object.")
    return _json(value)


def _migration_sql(name):
    return resources.files("vasp_slurm_agent").joinpath("migrations", name).read_text(encoding="utf-8")


def _statements(script):
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending
            pending = ""
    if pending.strip():
        raise ValueError("Incomplete workspace migration.")


class Workspace:
    """Project ownership, chat history and a read-through index of solver runs.

    Public IDs are UUID strings except workflow step IDs, which may be short
    names. No method submits, cancels, modifies or certifies a calculation.
    """

    def __init__(self, home=None):
        location = Path(home if home is not None else app_home()).expanduser().absolute()
        if location.is_symlink():
            raise ValueError("The workspace cannot be a symbolic link.")
        location.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.home = location.resolve()
        self.home.chmod(0o700)
        self.projects_dir = self.home / "projects"
        if self.projects_dir.is_symlink():
            raise ValueError("The projects folder cannot be a symbolic link.")
        self.projects_dir.mkdir(mode=0o700, exist_ok=True)
        self.projects_dir.chmod(0o700)
        self.db_path = self.home / "workspace.sqlite3"
        if self.db_path.is_symlink():
            raise ValueError("The workspace database cannot be a symbolic link.")
        descriptor = os.open(self.db_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Invalid workspace database.")
        finally:
            os.close(descriptor)
        self.db_path.chmod(0o600)
        self._migrate()

    @contextmanager
    def _connection(self):
        if self.db_path.is_symlink() or not self.db_path.is_file():
            raise ValueError("The workspace database is unavailable.")
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=10000")
            yield db
        finally:
            db.close()
            for suffix in ("", "-wal", "-shm"):
                path = Path(str(self.db_path) + suffix)
                try:
                    if path.is_file() and not path.is_symlink():
                        path.chmod(0o600)
                except FileNotFoundError:
                    # Another process closed the last connection and SQLite removed the file.
                    pass

    @contextmanager
    def _transaction(self, *, write=True):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _migrate(self):
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            # Table rebuilds follow SQLite's documented procedure: references
            # are checked once, after every migration, before the commit.
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("BEGIN IMMEDIATE")
            try:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version > _MIGRATIONS[-1][0]:
                    raise ValueError("This workspace requires a newer DFT Agent version.")
                for number, name in _MIGRATIONS:
                    if number <= version:
                        continue
                    if number != version + 1:
                        raise ValueError("Workspace migrations must be consecutive.")
                    for statement in _statements(_migration_sql(name)):
                        db.execute(statement)
                    db.execute("INSERT INTO schema_migrations VALUES (?, ?, ?)", (number, name, _now()))
                    db.execute(f"PRAGMA user_version={number}")
                    version = number
                if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("A workspace migration left inconsistent references.")
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _project_row(self, db, project_id, *, writable=False):
        row = db.execute("SELECT * FROM projects WHERE id=?", (_id(project_id),)).fetchone()
        if row is None:
            raise ValueError("Project not found.")
        if writable and (row["archived_at"] or row["deleted_at"]):
            raise ValueError("Restore the project before changing its contents.")
        return row

    def _project(self, row):
        return {**dict(row), "archived": bool(row["archived_at"]), "deleted": bool(row["deleted_at"]),
                "root": str(self.projects_dir / row["id"])}

    def _event(self, db, project_id, kind, details):
        kind = _text(kind, "event kind", 100)
        cursor = db.execute("INSERT INTO events(project_id,kind,details,created_at) VALUES (?,?,?,?)",
                            (project_id, kind, _object(details), _now()))
        return cursor.lastrowid

    def create_project(self, name):
        name, identifier, now = _text(name, "project name"), str(uuid.uuid4()), _now()
        root = self.projects_dir / identifier
        if self.projects_dir.is_symlink():
            raise ValueError("Invalid projects folder.")
        root.mkdir(mode=0o700)
        try:
            for folder in _FOLDERS:
                (root / folder).mkdir(mode=0o700)
            with self._transaction() as db:
                db.execute("INSERT INTO projects(id,name,created_at,updated_at) VALUES (?,?,?,?)",
                           (identifier, name, now, now))
                self._event(db, identifier, "project.created", {"name": name})
                return self._project(self._project_row(db, identifier))
        except BaseException:
            shutil.rmtree(root)
            raise

    def list_projects(self, include_archived=False, include_deleted=False):
        with self._transaction(write=False) as db:
            rows = db.execute("SELECT * FROM projects WHERE (? OR archived_at IS NULL) "
                              "AND (? OR deleted_at IS NULL) ORDER BY COALESCE(opened_at,updated_at) DESC,id",
                              (bool(include_archived), bool(include_deleted))).fetchall()
            return [self._project(row) for row in rows]

    def get_project(self, project_id):
        with self._transaction(write=False) as db:
            return self._project(self._project_row(db, project_id))

    def open_project(self, project_id):
        self.project_paths(project_id)
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            db.execute("UPDATE projects SET opened_at=?,revision=revision+1 WHERE id=?", (_now(), project_id))
            self._event(db, project_id, "project.opened", {})
            return self._project(self._project_row(db, project_id))

    def rename_project(self, project_id, name):
        name = _text(name, "project name")
        with self._transaction() as db:
            before = self._project_row(db, project_id)
            if before["deleted_at"]:
                raise ValueError("Restore the deleted project before renaming it.")
            db.execute("UPDATE projects SET name=?,updated_at=?,revision=revision+1 WHERE id=?", (name, _now(), project_id))
            self._event(db, project_id, "project.renamed", {"before": before["name"], "name": name})
            return self._project(self._project_row(db, project_id))

    def archive_project(self, project_id, archived=True):
        if type(archived) is not bool:
            raise ValueError("archived must be boolean.")
        with self._transaction() as db:
            row = self._project_row(db, project_id)
            if row["deleted_at"]:
                raise ValueError("Restore the deleted project first.")
            now = _now()
            db.execute("UPDATE projects SET archived_at=?,updated_at=?,revision=revision+1 WHERE id=?",
                       (now if archived else None, now, project_id))
            self._event(db, project_id, "project.archived" if archived else "project.unarchived", {})
            return self._project(self._project_row(db, project_id))

    def delete_project(self, project_id, confirmation=None):
        """Hide a project reversibly; never remove its files or remote jobs."""
        with ExitStack() as locks:
            with self._transaction() as db:
                row = self._project_row(db, project_id)
                if confirmation != row["name"]:
                    raise ValueError("Enter the project name to confirm deletion.")
                if row["deleted_at"]:
                    return self._project(row)
                self._check_inactive(db, project_id, locks)
                now = _now()
                db.execute("UPDATE projects SET deleted_at=?,updated_at=?,revision=revision+1 WHERE id=?", (now, now, project_id))
                self._event(db, project_id, "project.deleted", {"files_retained": True})
                return self._project(self._project_row(db, project_id))

    def restore_project(self, project_id):
        with self._transaction() as db:
            self._project_row(db, project_id)
            now = _now()
            db.execute("UPDATE projects SET deleted_at=NULL,archived_at=NULL,updated_at=?,revision=revision+1 WHERE id=?", (now, project_id))
            self._event(db, project_id, "project.restored", {})
            return self._project(self._project_row(db, project_id))

    def _resolve(self, project_id, relative, *, must_exist=False):
        _id(project_id)
        value = os.fspath(relative)
        if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
            raise ValueError("Invalid project path.")
        parts = PurePosixPath(value)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Paths must stay inside their project.")
        root = self.projects_dir / project_id
        if self.projects_dir.is_symlink() or root.is_symlink() or not root.is_dir():
            raise ValueError("The project folder is unavailable.")
        candidate = root
        for part in parts.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("Project paths cannot use symbolic links.")
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Paths must stay inside their project.")
        if must_exist and not candidate.exists():
            raise ValueError("The project file or folder is unavailable.")
        return candidate

    def resolve_path(self, project_id, relative, must_exist=False):
        """Resolve a portable relative path in one project, without following links."""
        self.get_project(project_id)
        return self._resolve(project_id, relative, must_exist=must_exist)

    def project_paths(self, project_id):
        self.get_project(project_id)
        paths = {"root": self._resolve(project_id, ".", must_exist=True)}
        paths.update({name: self._resolve(project_id, name, must_exist=True) for name in _FOLDERS})
        if any(not path.is_dir() for path in paths.values()):
            raise ValueError("A project directory is unavailable.")
        return paths

    def _relative(self, project_id, path):
        value = Path(path).expanduser()
        if value.is_absolute():
            try:
                value = value.relative_to(self.projects_dir / project_id)
            except ValueError:
                raise ValueError("The path belongs to another project or lies outside the workspace.") from None
        checked = self._resolve(project_id, value.as_posix(), must_exist=True)
        return checked, checked.relative_to(self.projects_dir / project_id).as_posix()

    def record_event(self, project_id, kind, details):
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            identifier = self._event(db, project_id, kind, details)
            row = dict(db.execute("SELECT * FROM events WHERE id=?", (identifier,)).fetchone())
            row["details"] = json.loads(row["details"])
            return row

    @contextmanager
    def edit_project(self, project_id):
        """Hold the writable-project guard while a caller writes and audits files.

        The caller must restore changed files if this context raises. The yielded
        callback appends an event within this same database transaction.
        """
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            yield lambda kind, details: self._event(db, project_id, kind, details)

    def events(self, project_id, limit=100):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Choose an event limit from 1 to 10000.")
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            return [{**dict(row), "details": json.loads(row["details"])} for row in db.execute(
                "SELECT * FROM events WHERE project_id=? ORDER BY id DESC LIMIT ?", (project_id, limit))]

    def _session(self, db, project_id, session_id, *, messages=True):
        self._project_row(db, project_id)
        row = db.execute("SELECT * FROM sessions WHERE project_id=? AND id=?", (project_id, _id(session_id))).fetchone()
        if row is None:
            raise ValueError("Conversation not found in this project.")
        result = dict(row)
        if messages:
            result["messages"] = [{**dict(item), "metadata": json.loads(item["metadata"])} for item in db.execute(
                "SELECT id,role,content,metadata,created_at FROM messages WHERE project_id=? AND session_id=? ORDER BY id",
                (project_id, session_id))]
        return result

    def create_session(self, project_id, title="New conversation"):
        title, identifier, now = _text(title, "conversation title"), str(uuid.uuid4()), _now()
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            db.execute("INSERT INTO sessions(id,project_id,title,created_at,updated_at) VALUES (?,?,?,?,?)", (identifier, project_id, title, now, now))
            self._event(db, project_id, "session.created", {"session_id": identifier})
            return self._session(db, project_id, identifier)

    def get_session(self, project_id, session_id):
        with self._transaction(write=False) as db:
            return self._session(db, project_id, session_id)

    def list_sessions(self, project_id, query=""):
        query = _text(query, "conversation search", 1000, empty=True)
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            return [dict(row) for row in db.execute(
                "SELECT * FROM sessions WHERE project_id=? AND (instr(lower(title),lower(?))>0 OR EXISTS "
                "(SELECT 1 FROM messages WHERE messages.project_id=sessions.project_id AND messages.session_id=sessions.id "
                "AND instr(lower(content),lower(?))>0)) ORDER BY updated_at DESC,id", (project_id, query, query))]

    @staticmethod
    def _revision(current, expected):
        if expected is not None and (type(expected) is not int or expected != current):
            raise ValueError("The conversation changed. Reload it before saving.")

    def append_message(self, project_id, session_id, role, content, metadata=None, expected_revision=None):
        """Append one turn atomically. Credentials must never be entered as chat."""
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("Choose a user, assistant or tool message.")
        content = _text(redact(content) if role != "user" else content, "message", 1_000_000, empty=True)
        encoded = _object({} if metadata is None else metadata)
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            session = self._session(db, project_id, session_id, messages=False)
            self._revision(session["revision"], expected_revision)
            now = _now()
            message = db.execute("INSERT INTO messages(project_id,session_id,role,content,metadata,created_at) VALUES (?,?,?,?,?,?)",
                                 (project_id, session_id, role, content, encoded, now))
            db.execute("UPDATE sessions SET revision=revision+1,updated_at=? WHERE project_id=? AND id=?", (now, project_id, session_id))
            self._event(db, project_id, "message.appended", {"session_id": session_id, "message_id": message.lastrowid, "role": role})
            return self._session(db, project_id, session_id)

    def update_session_title(self, project_id, session_id, title, expected_revision=None):
        title = _text(title, "conversation title")
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            session = self._session(db, project_id, session_id, messages=False)
            self._revision(session["revision"], expected_revision)
            db.execute("UPDATE sessions SET title=?,revision=revision+1,updated_at=? WHERE project_id=? AND id=?", (title, _now(), project_id, session_id))
            self._event(db, project_id, "session.renamed", {"session_id": session_id, "title": title})
            return self._session(db, project_id, session_id)

    def save_memory(self, project_id, key, text):
        key = _text(key, "memory name")
        text = _text(text, "memory text", 1_000_000, empty=True)
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            now = _now()
            db.execute("INSERT INTO memory(project_id,key,text,created_at,updated_at) VALUES (?,?,?,?,?) "
                       "ON CONFLICT(project_id,key) DO UPDATE SET text=excluded.text,updated_at=excluded.updated_at,revision=memory.revision+1",
                       (project_id, key, text, now, now))
            self._event(db, project_id, "memory.saved", {"key": key})
            return dict(db.execute("SELECT * FROM memory WHERE project_id=? AND key=?", (project_id, key)).fetchone())

    def list_memory(self, project_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            return [dict(row) for row in db.execute("SELECT * FROM memory WHERE project_id=? ORDER BY updated_at DESC,key", (project_id,))]

    def create_workflow(self, project_id, session_id, title, goal, steps):
        """Record a reviewed dependency graph; this does not prepare or submit it."""
        title, goal = _text(title, "workflow title"), _text(goal, "workflow goal", 100000, empty=True)
        if not isinstance(steps, list) or not steps:
            raise ValueError("A workflow needs at least one step.")
        prepared = []
        for item in steps:
            if not isinstance(item, dict) or set(item) - {"id", "name", "engine", "depends_on", "spec"}:
                raise ValueError("Invalid workflow step.")
            identifier = item.get("id", str(uuid.uuid4()))
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
                raise ValueError("Use a short, unique workflow step ID.")
            if item.get("engine") not in _ENGINES:
                raise ValueError("Choose VASP, QE, Wannier90 or Python for the step.")
            dependencies = item.get("depends_on", [])
            if not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies) or len(dependencies) != len(set(dependencies)):
                raise ValueError("Invalid workflow dependencies.")
            prepared.append({"id": identifier, "name": _text(item.get("name"), "step name"), "engine": item["engine"],
                             "depends_on": dependencies, "spec": _object(item.get("spec", {}))})
        identifiers = {item["id"] for item in prepared}
        if len(identifiers) != len(prepared):
            raise ValueError("Workflow step IDs must be unique.")
        pending = {item["id"]: set(item["depends_on"]) for item in prepared}
        if any(not values <= identifiers or key in values for key, values in pending.items()):
            raise ValueError("Dependencies must name another step in this workflow.")
        while pending:
            ready = {key for key, values in pending.items() if not values}
            if not ready:
                raise ValueError("Workflow dependencies must not contain a cycle.")
            pending = {key: values - ready for key, values in pending.items() if key not in ready}
        identifier, now = str(uuid.uuid4()), _now()
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            self._session(db, project_id, session_id, messages=False)
            db.execute("INSERT INTO workflows(id,project_id,session_id,title,goal,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                       (identifier, project_id, session_id, title, goal, now, now))
            for position, item in enumerate(prepared):
                db.execute("INSERT INTO steps(project_id,workflow_id,id,position,name,engine,spec) VALUES (?,?,?,?,?,?,?)",
                           (project_id, identifier, item["id"], position, item["name"], item["engine"], item["spec"]))
            for item in prepared:
                db.executemany("INSERT INTO dependencies VALUES (?,?,?,?)", [(project_id, identifier, item["id"], dependency) for dependency in item["depends_on"]])
            self._event(db, project_id, "workflow.created", {"workflow_id": identifier})
            return self._workflow(db, project_id, identifier)

    def _workflow(self, db, project_id, workflow_id):
        self._project_row(db, project_id)
        row = db.execute("SELECT * FROM workflows WHERE project_id=? AND id=?", (project_id, _id(workflow_id))).fetchone()
        if row is None:
            raise ValueError("Workflow not found in this project.")
        items = []
        for step in db.execute("SELECT * FROM steps WHERE project_id=? AND workflow_id=? ORDER BY position", (project_id, workflow_id)):
            dependencies = [item[0] for item in db.execute("SELECT depends_on FROM dependencies WHERE project_id=? AND workflow_id=? AND step_id=? ORDER BY depends_on", (project_id, workflow_id, step["id"]))]
            items.append({**dict(step), "spec": json.loads(step["spec"]), "depends_on": dependencies})
        return {**dict(row), "steps": items}

    def get_workflow(self, project_id, workflow_id):
        with self._transaction(write=False) as db:
            return self._workflow(db, project_id, workflow_id)

    def list_workflows(self, project_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            ids = [row[0] for row in db.execute("SELECT id FROM workflows WHERE project_id=? ORDER BY updated_at DESC,id", (project_id,))]
            return [self._workflow(db, project_id, identifier) for identifier in ids]

    @staticmethod
    def _state(folder):
        paths = [folder / name for name in ("run.json", "batch.json") if (folder / name).exists() or (folder / name).is_symlink()]
        if len(paths) != 1 or paths[0].is_symlink() or not paths[0].is_file() or paths[0].stat().st_size > 32 * 1024 * 1024:
            raise ValueError("The run must have one regular run.json or batch.json record.")
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("Duplicate run record field.")
                value[key] = item
            return value
        try:
            value = json.loads(paths[0].read_text(encoding="utf-8"), object_pairs_hook=pairs,
                               parse_constant=lambda item: (_ for _ in ()).throw(ValueError("Invalid run number.")))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("The run record is unreadable.") from error
        kind = "batch" if paths[0].name == "batch.json" else "run"
        identifier = value.get("batch_id" if kind == "batch" else "run_id") if isinstance(value, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier) or not isinstance(value.get("status"), str):
            raise ValueError("The run identity or status is missing.")
        engine = value.get("engine", "vasp")
        if engine not in _ENGINES:
            raise ValueError("Unknown run engine.")
        _object(value)
        return kind, identifier, engine, value

    def register_run(self, project_id, run_dir, workflow_id=None, step_id=None):
        """Index existing owned solver files, without changing their state."""
        self.get_project(project_id)
        folder, relative = self._relative(project_id, run_dir)
        if not folder.is_dir() or not relative.startswith("runs/"):
            raise ValueError("Runs must live inside this project's runs folder.")
        kind, native_id, engine, state = self._state(folder)
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            if step_id is not None and workflow_id is None:
                raise ValueError("A step must belong to a workflow.")
            if workflow_id is not None:
                workflow = self._workflow(db, project_id, workflow_id)
                if step_id is not None:
                    step = next((item for item in workflow["steps"] if item["id"] == step_id), None)
                    if step is None or step["engine"] != engine:
                        raise ValueError("The run does not match this workflow step and engine.")
            existing = db.execute("SELECT * FROM runs WHERE project_id=? AND (relative_path=? OR run_id=?)", (project_id, relative, native_id)).fetchall()
            if existing and (len(existing) != 1 or existing[0]["relative_path"] != relative or existing[0]["run_id"] != native_id or existing[0]["engine"] != engine or existing[0]["kind"] != kind):
                raise ValueError("The run identity is already registered at another path or has changed.")
            now = _now()
            if existing:
                old = existing[0]
                if workflow_id is None:
                    workflow_id, step_id = old["workflow_id"], old["step_id"]
                if old["workflow_id"] is not None and (old["workflow_id"], old["step_id"]) != (workflow_id, step_id):
                    raise ValueError("The run already belongs to another workflow step.")
                identifier = old["id"]
                db.execute("UPDATE runs SET status=?,state=?,workflow_id=?,step_id=?,updated_at=? WHERE id=?", (state["status"], _object(state), workflow_id, step_id, now, identifier))
            else:
                identifier = str(uuid.uuid4())
                db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, project_id, native_id, engine, kind, relative, state["status"], _object(state), workflow_id, step_id, now, now))
            self._event(db, project_id, "run.registered", {"id": identifier, "run_id": native_id, "path": relative})
            return self._run_record(db.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone())

    def _run_record(self, row):
        record = dict(row)
        record["run_dir"] = str(self.projects_dir / row["project_id"] / row["relative_path"])
        try:
            folder = self._resolve(row["project_id"], row["relative_path"], must_exist=True)
            kind, identifier, engine, state = self._state(folder)
            if (kind, identifier, engine) != (row["kind"], row["run_id"], row["engine"]):
                raise ValueError("The registered run identity changed.")
            record.update(state=state, status=state["status"])
        except (ValueError, OSError):
            record.update(state=None, status="unavailable", error="The registered run record is missing or changed.")
        return record

    def get_run(self, project_id, run_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            row = db.execute("SELECT * FROM runs WHERE project_id=? AND id=?", (project_id, _id(run_id))).fetchone()
            if row is None:
                raise ValueError("Run not found in this project.")
            return self._run_record(row)

    def list_runs(self, project_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            return [self._run_record(row) for row in db.execute("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC,id", (project_id,))]

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def register_artifact(self, project_id, path, kind="file", run_id=None):
        self.get_project(project_id)
        file, relative = self._relative(project_id, path)
        kind = _text(kind, "artifact kind", 100)
        if not file.is_file():
            raise ValueError("An artifact must be a regular project file.")
        before = file.stat()
        digest = self._file_hash(file)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("The artifact changed while being registered.")
        with self._transaction() as db:
            self._project_row(db, project_id, writable=True)
            if run_id is not None:
                run = db.execute("SELECT relative_path FROM runs WHERE project_id=? AND id=?", (project_id, _id(run_id))).fetchone()
                if run is None or not file.is_relative_to(self.projects_dir / project_id / run["relative_path"]):
                    raise ValueError("The artifact does not belong to this project's run.")
            now = _now()
            old = db.execute("SELECT * FROM artifacts WHERE project_id=? AND relative_path=?", (project_id, relative)).fetchone()
            identifier = old["id"] if old else str(uuid.uuid4())
            if old and old["run_id"] is not None and run_id is None:
                run_id = old["run_id"]
            db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(project_id,relative_path) DO UPDATE SET "
                       "kind=excluded.kind,run_id=excluded.run_id,size=excluded.size,sha256=excluded.sha256,updated_at=excluded.updated_at",
                       (identifier, project_id, relative, kind, run_id, after.st_size, digest, now, now))
            self._event(db, project_id, "artifact.registered", {"id": identifier, "path": relative, "sha256": digest})
            return self._artifact_record(db.execute("SELECT * FROM artifacts WHERE id=?", (identifier,)).fetchone())

    def _artifact_record(self, row):
        record = dict(row)
        record["path"] = str(self.projects_dir / row["project_id"] / row["relative_path"])
        try:
            record["available"] = self._resolve(row["project_id"], row["relative_path"], must_exist=True).is_file()
        except (ValueError, OSError):
            record["available"] = False
        return record

    def get_artifact(self, project_id, artifact_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            row = db.execute("SELECT * FROM artifacts WHERE project_id=? AND id=?", (project_id, _id(artifact_id))).fetchone()
            if row is None:
                raise ValueError("Artifact not found in this project.")
            return self._artifact_record(row)

    def list_artifacts(self, project_id):
        with self._transaction(write=False) as db:
            self._project_row(db, project_id)
            return [self._artifact_record(row) for row in db.execute("SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at DESC,id", (project_id,))]

    @staticmethod
    @contextmanager
    def _idle_lock(folder, name):
        from .workflow import _acquire, _release

        path = folder / name
        if path.is_symlink():
            raise ValueError("Invalid run lock.")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "r+b") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Invalid run lock.")
            try:
                # Windows can lock a byte beyond EOF. Writing first would touch
                # another worker's locked byte before we know the run is idle.
                _acquire(stream, blocking=False)
            except BlockingIOError as error:
                raise ValueError("A run is being monitored or updated. Stop monitoring before deleting this project.") from error
            try:
                yield
            finally:
                _release(stream)

    def _check_inactive(self, db, project_id, locks):
        """Check live registered and unregistered runs before hiding a project."""
        if db.execute("SELECT 1 FROM step_attempts WHERE project_id=? AND state IN "
                      "('planning','preparing','starting','active','reconnecting','cancelling') LIMIT 1",
                      (project_id,)).fetchone():
            raise ValueError("A project operation is active. Reconcile it before deleting this project.")
        root = self._resolve(project_id, "runs", must_exist=True)
        registered = db.execute("SELECT * FROM runs WHERE project_id=?", (project_id,)).fetchall()
        for row in registered:
            if self._run_record(row)["state"] is None:
                raise ValueError("A run is unavailable. Reconcile it before deleting this project.")
        folders = set()
        for directory, dirs, files in os.walk(root, followlinks=False):
            current = Path(directory)
            if any((current / name).is_symlink() for name in [*dirs, *files]):
                raise ValueError("Reconcile linked run files before deleting this project.")
            if "run.json" in files or "batch.json" in files:
                folders.add(current)
        for name in (".worker.lock", ".state.lock"):
            for folder in sorted(folders):
                locks.enter_context(self._idle_lock(folder, name))
        for folder in folders:
            try:
                _, _, _, state = self._state(folder)
            except (ValueError, OSError) as error:
                raise ValueError("A run is unreadable. Reconcile it before deleting this project.") from error
            verified_process = False
            if (state.get("workflow") == "script" and state.get("engine") in {"python", "wannier90"}
                    and state["status"] in {"completed_unverified", "unconverged", "suspect"}):
                from .script_workflow import load_run_context

                try:
                    context = load_run_context(folder)
                except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
                    raise ValueError("Active or uncertain jobs must be reconciled before deleting this project.") from error
                verified_process = (context["status"] == state["status"]
                                    and context["facts"]["run.process_success"]["value"] is True)
            if (not verified_process and (state["status"] not in {"planned", "succeeded", "completed", "failed", "cancelled"}
                                          or state.get("last_error"))) or state.get("remote_failures"):
                raise ValueError("Active or uncertain jobs must be reconciled before deleting this project.")
            stages = state.get("stages", [])
            if not isinstance(stages, list):
                raise ValueError("Invalid run state. Reconcile it before deleting this project.")
            for stage in stages:
                if not isinstance(stage, dict) or ((stage.get("job_id") or stage.get("staged")) and stage.get("scheduler_state") not in _TERMINAL):
                    raise ValueError("Active or uncertain jobs must be reconciled before deleting this project.")
