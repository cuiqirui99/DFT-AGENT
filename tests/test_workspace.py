"""Project isolation and failure behavior; no solver or remote execution."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from vasp_slurm_agent import workspace as module
from vasp_slurm_agent.workspace import Workspace


@pytest.fixture
def store(tmp_path):
    return Workspace(tmp_path / "workspace")


def make_run(store, project_id, name="task", *, engine="vasp", status="planned", **extra):
    root = store.project_paths(project_id)["runs"] / name
    root.mkdir(parents=True)
    state = {"schema_version": 2, "run_id": name, "engine": engine, "status": status,
             "stages": [], **extra}
    (root / "run.json").write_text(json.dumps(state))
    return root


def test_workspace_defaults_migration_and_private_files(store, monkeypatch):
    monkeypatch.setenv("DFT_AGENT_HOME", str(store.home))
    assert Workspace().home == store.home
    with store._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == module._MIGRATIONS[-1][0]
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert len(db.execute("SELECT * FROM schema_migrations").fetchall()) == len(module._MIGRATIONS)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    if os.name != "nt":
        assert store.home.stat().st_mode & 0o777 == 0o700
        assert store.db_path.stat().st_mode & 0o777 == 0o600


def test_failed_migration_rolls_back_schema_and_version(tmp_path, monkeypatch):
    original = module._migration_sql
    monkeypatch.setattr(module, "_migration_sql", lambda name: original(name) + "INVALID SQL;\n")
    home = tmp_path / "failed"
    with pytest.raises(sqlite3.DatabaseError):
        Workspace(home)
    with sqlite3.connect(home / "workspace.sqlite3") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    monkeypatch.setattr(module, "_migration_sql", original)
    assert Workspace(home).create_project("Recovered")["name"] == "Recovered"


def test_newer_database_is_never_downgraded(store):
    with store._connection() as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="newer"):
        Workspace(store.home)
    with store._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 99


def test_failed_project_creation_removes_only_its_new_folder(store, monkeypatch):
    existing = store.create_project("Existing")
    original = store._event
    def fail(*args):
        original(*args)
        raise RuntimeError("disk failure")
    monkeypatch.setattr(store, "_event", fail)
    with pytest.raises(RuntimeError):
        store.create_project("Will roll back")
    assert [p["id"] for p in store.list_projects()] == [existing["id"]]
    assert [p.name for p in store.projects_dir.iterdir()] == [existing["id"]]


def test_project_lifecycle_keeps_identity_files_and_history(store):
    project = store.create_project("Silicon")
    identifier = project["id"]
    paths = store.project_paths(identifier)
    note = paths["files"] / "notes.txt"
    note.write_text("Unpublished research")
    assert set(paths) == {"root", "files", "runs", "memory", "config", "artifacts"}
    assert store.rename_project(identifier, "Silicon SOC")["id"] == identifier
    assert store.open_project(identifier)["opened_at"]
    store.archive_project(identifier)
    assert store.list_projects() == []
    assert store.list_projects(include_archived=True)[0]["id"] == identifier
    with pytest.raises(ValueError, match="Restore"):
        store.create_session(identifier)
    store.archive_project(identifier, False)
    with pytest.raises(ValueError, match="name"):
        store.delete_project(identifier, confirmation="Silicon")
    assert store.delete_project(identifier, confirmation="Silicon SOC")["deleted"]
    assert note.read_text() == "Unpublished research"
    assert store.list_projects() == []
    assert store.list_projects(include_deleted=True)[0]["id"] == identifier
    restored = Workspace(store.home).restore_project(identifier)
    assert not restored["deleted"] and not restored["archived"]
    assert restored["root"] == project["root"]
    assert len(store.events(identifier)) == 7


def test_projects_isolate_sessions_memory_and_same_engine_runs(store):
    a, b = [store.create_project(name)["id"] for name in ("A", "B")]
    sa, sb = [store.create_session(p) for p in (a, b)]
    store.append_message(a, sa["id"], "user", "Private A goal")
    store.append_message(b, sb["id"], "user", "Private B goal")
    store.save_memory(a, "method", "A method")
    store.save_memory(b, "method", "B method")
    ra = store.register_run(a, make_run(store, a))
    rb = store.register_run(b, make_run(store, b))
    assert ra["run_id"] == rb["run_id"] and ra["id"] != rb["id"]
    restarted = Workspace(store.home)
    assert restarted.get_session(a, sa["id"])["messages"][0]["content"] == "Private A goal"
    assert restarted.list_sessions(a, "Private B") == []
    assert restarted.list_memory(a)[0]["text"] == "A method"
    assert len(restarted.list_runs(a)) == len(restarted.list_runs(b)) == 1
    for call in (lambda: store.get_session(a, sb["id"]), lambda: store.append_message(a, sb["id"], "user", "leak"),
                 lambda: store.get_run(a, rb["id"]), lambda: store.register_run(a, rb["run_dir"])):
        with pytest.raises(ValueError):
            call()


def test_append_and_title_use_revision_checks_and_are_atomic(store):
    p = store.create_project("Revision")["id"]
    session = store.create_session(p)
    new = store.append_message(p, session["id"], "user", "First", expected_revision=0)
    assert new["revision"] == 1
    with pytest.raises(ValueError, match="changed"):
        store.append_message(p, session["id"], "user", "Stale", expected_revision=0)
    with pytest.raises(ValueError, match="changed"):
        store.update_session_title(p, session["id"], "Stale title", expected_revision=0)
    result = store.update_session_title(p, session["id"], "New title", expected_revision=1)
    assert result["revision"] == 2 and result["title"] == "New title"
    assert [item["content"] for item in result["messages"]] == ["First"]
    assert len(store.events(p)) == 4


def test_two_writers_cannot_both_commit_the_same_session_revision(store):
    p = store.create_project("Concurrent")["id"]
    session = store.create_session(p)
    def write(content):
        try:
            return store.append_message(p, session["id"], "user", content, expected_revision=0)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, ("One", "Two")))
    assert sum(item is not None for item in results) == 1
    assert len(store.get_session(p, session["id"])["messages"]) == 1


def test_append_failure_does_not_leave_message_or_revision(store, monkeypatch):
    p = store.create_project("Atomic")["id"]
    session = store.create_session(p)
    monkeypatch.setattr(store, "_event", lambda *args: (_ for _ in ()).throw(RuntimeError("event failed")))
    with pytest.raises(RuntimeError):
        store.append_message(p, session["id"], "user", "Not committed")
    assert store.get_session(p, session["id"])["messages"] == []
    assert store.get_session(p, session["id"])["revision"] == 0


@pytest.mark.parametrize("metadata", [
    {"api_key": "x"}, {"environment": {"ANTHROPIC_API_KEY": "x"}}, {"nested": [{"password": "x"}]},
    {"private_key": "x"}, {"otp": "123456"}, {"access_token": "x"}, {"n": float("nan")},
])
def test_credentials_and_nonfinite_metadata_never_enter_database(store, metadata):
    p = store.create_project("Private")["id"]
    session = store.create_session(p)
    with pytest.raises(ValueError):
        store.append_message(p, session["id"], "user", "message", metadata=metadata)
    with pytest.raises(ValueError):
        store.record_event(p, "test", metadata)
    with pytest.raises(ValueError):
        store.create_workflow(p, session["id"], "Unsafe", "goal", [{"name": "SCF", "engine": "qe", "spec": metadata}])
    assert store.get_session(p, session["id"])["messages"] == []
    assert store.list_workflows(p) == []


def test_provider_usage_and_credential_references_are_not_secrets(store):
    p = store.create_project("Usage")["id"]
    s = store.create_session(p)
    metadata = {"input_tokens": 12, "output_tokens": 5, "credential_ref": "account-1", "api_key_ref": "account-2"}
    assert store.append_message(p, s["id"], "assistant", "Done", metadata)["messages"][0]["metadata"] == metadata


def test_paths_reject_foreign_absolute_parent_and_symlink_paths(store):
    a, b = [store.create_project(name)["id"] for name in ("A", "B")]
    other = store.project_paths(b)["files"] / "private.txt"
    other.write_text("B data")
    own = store.project_paths(a)["files"]
    (own / "linked").symlink_to(other.parent, target_is_directory=True)
    for path in (str(other), f"../{b}/files/private.txt", "files/../../x", "files/linked/private.txt", "C:\\private", "files/\\private"):
        with pytest.raises(ValueError):
            store.resolve_path(a, path)
    assert store.resolve_path(a, "files/new.txt") == own / "new.txt"
    with pytest.raises(ValueError, match="unavailable"):
        store.resolve_path(a, "files/new.txt", must_exist=True)
    assert other.read_text() == "B data"


def test_database_and_project_directory_symlinks_rejected(store, tmp_path):
    project = store.create_project("Linked")
    root = Path(project["root"])
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        store.project_paths(project["id"])
    external = tmp_path / "database"
    store.db_path.rename(external)
    store.db_path.symlink_to(external)
    with pytest.raises(ValueError):
        Workspace(store.home)


def test_mixed_engine_workflow_dependencies_and_run_binding(store):
    p = store.create_project("Mixed")["id"]
    s = store.create_session(p)
    steps = [
        {"id": "relax", "name": "Relax", "engine": "vasp"},
        {"id": "qe", "name": "QE SCF", "engine": "qe", "depends_on": ["relax"], "spec": {"fresh_scf": True}},
        {"id": "wannier", "name": "Wannier", "engine": "wannier90", "depends_on": ["qe"]},
        {"id": "plot", "name": "Plot", "engine": "python", "depends_on": ["wannier"]},
    ]
    workflow = store.create_workflow(p, s["id"], "VASP to QE", "Compare bands", steps)
    assert [step["engine"] for step in workflow["steps"]] == ["vasp", "qe", "wannier90", "python"]
    assert workflow["steps"][1]["depends_on"] == ["relax"]
    assert all(step["status"] == "planned" for step in workflow["steps"])
    run = make_run(store, p, engine="qe")
    with pytest.raises(ValueError, match="engine"):
        store.register_run(p, run, workflow["id"], "relax")
    registered = store.register_run(p, run, workflow["id"], "qe")
    assert registered["step_id"] == "qe"
    assert store.get_workflow(p, workflow["id"])["status"] == "planned"
    assert Workspace(store.home).get_workflow(p, workflow["id"])["steps"] == workflow["steps"]


@pytest.mark.parametrize("steps", [
    [],
    [{"id": "x", "name": "X", "engine": "vasp", "depends_on": ["absent"]}],
    [{"id": "x", "name": "X", "engine": "vasp", "depends_on": ["x"]}],
    [{"id": "a", "name": "A", "engine": "vasp", "depends_on": ["b"]}, {"id": "b", "name": "B", "engine": "qe", "depends_on": ["a"]}],
    [{"id": "x", "name": "X", "engine": "vasp"}, {"id": "x", "name": "Y", "engine": "qe"}],
    [{"name": "X", "engine": "unknown"}],
])
def test_invalid_workflows_leave_no_partial_graph(store, steps):
    p = store.create_project("Graph")["id"]
    s = store.create_session(p)
    with pytest.raises(ValueError):
        store.create_workflow(p, s["id"], "Invalid", "goal", steps)
    assert store.list_workflows(p) == []


def test_composite_foreign_keys_reject_cross_project_joins(store):
    a, b = [store.create_project(name)["id"] for name in ("A", "B")]
    sa, sb = [store.create_session(p) for p in (a, b)]
    with pytest.raises(ValueError):
        store.create_workflow(a, sb["id"], "Wrong", "goal", [{"name": "SCF", "engine": "qe"}])
    workflow = store.create_workflow(b, sb["id"], "B", "goal", [{"id": "scf", "name": "SCF", "engine": "qe"}])
    with pytest.raises(ValueError):
        store.get_workflow(a, workflow["id"])
    with pytest.raises(ValueError):
        store.register_run(a, make_run(store, a, engine="qe"), workflow["id"], "scf")
    with store._connection() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO messages(project_id,session_id,role,content,metadata,created_at) VALUES (?,?,?,?,?,?)", (a, sb["id"], "user", "leak", "{}", "now"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE workflows SET session_id=? WHERE id=?", (sa["id"], workflow["id"]))
    assert store.list_runs(a) == []


def test_run_registry_reads_live_state_without_certifying_success(store):
    p = store.create_project("Runs")["id"]
    folder = make_run(store, p)
    record = store.register_run(p, folder)
    original = json.loads((folder / "run.json").read_text())
    updated = {**original, "status": "running"}
    (folder / "run.json").write_text(json.dumps(updated))
    fresh = store.list_runs(p)[0]
    assert fresh["state"] == updated and fresh["status"] == "running"
    assert "accepted" not in fresh and "validated" not in fresh
    assert store.register_run(p, folder)["id"] == record["id"]
    updated["run_id"] = "replacement"
    (folder / "run.json").write_text(json.dumps(updated))
    assert store.get_run(p, record["id"])["state"] is None
    assert store.list_runs(p)[0]["status"] == "unavailable"
    with pytest.raises(ValueError):
        store.register_run(p, folder)


def test_same_run_identity_cannot_be_copied_into_second_registered_folder(store):
    p = store.create_project("Duplicates")["id"]
    first = make_run(store, p, "first", run_id="same")
    second = make_run(store, p, "second", run_id="same")
    store.register_run(p, first)
    with pytest.raises(ValueError, match="another path"):
        store.register_run(p, second)
    assert len(store.list_runs(p)) == 1


def test_artifacts_keep_registration_hash_and_ownership(store):
    a, b = [store.create_project(name)["id"] for name in ("A", "B")]
    folder = make_run(store, a)
    run = store.register_run(a, folder)
    path = folder / "result.csv"
    path.write_text("energy\n-10.3\n")
    artifact = store.register_artifact(a, path, "table", run["id"])
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact["available"]
    with pytest.raises(ValueError):
        store.get_artifact(b, artifact["id"])
    with pytest.raises(ValueError):
        store.register_artifact(b, path, "table", run["id"])
    outside = store.project_paths(a)["files"] / "unrelated.txt"
    outside.write_text("not in the run")
    with pytest.raises(ValueError):
        store.register_artifact(a, outside, run_id=run["id"])
    path.write_text("changed")
    assert store.get_artifact(a, artifact["id"])["sha256"] == artifact["sha256"]
    new = store.register_artifact(a, path)
    assert new["id"] == artifact["id"] and new["sha256"] != artifact["sha256"]
    assert new["run_id"] == run["id"]
    path.unlink()
    assert not store.get_artifact(a, artifact["id"])["available"]


@pytest.mark.parametrize("state", [
    {"status": "running"}, {"status": "queued"}, {"status": "needs_attention"},
    {"status": "planned", "stages": [{"staged": True, "job_id": None}]},
    {"status": "failed", "stages": [{"job_id": "123", "scheduler_state": "RUNNING"}]},
    {"status": "planned", "last_error": "Lost submission acknowledgment"},
    {"status": "completed_unverified", "engine": "python", "workflow": "script"},
])
def test_delete_refuses_active_or_uncertain_actual_state(store, state):
    p = store.create_project("Active")
    folder = make_run(store, p["id"], engine=state.get("engine", "vasp"))
    store.register_run(p["id"], folder)
    current = json.loads((folder / "run.json").read_text())
    (folder / "run.json").write_text(json.dumps({**current, **state}))
    with pytest.raises(ValueError, match="reconciled"):
        store.delete_project(p["id"], confirmation=p["name"])
    assert not store.get_project(p["id"])["deleted"]
    assert folder.is_dir()


def test_delete_checks_unregistered_runs_and_missing_registered_state(store):
    p = store.create_project("Unregistered")
    folder = make_run(store, p["id"], status="running")
    with pytest.raises(ValueError):
        store.delete_project(p["id"], confirmation=p["name"])
    state = json.loads((folder / "run.json").read_text())
    (folder / "run.json").write_text(json.dumps({**state, "status": "succeeded"}))
    store.register_run(p["id"], folder)
    (folder / "run.json").unlink()
    with pytest.raises(ValueError, match="unavailable"):
        store.delete_project(p["id"], confirmation=p["name"])


@pytest.mark.parametrize("lockname", [".worker.lock", ".state.lock"])
def test_delete_refuses_busy_worker_even_before_first_state_update(store, lockname):
    from vasp_slurm_agent.workflow import _acquire, _release

    p = store.create_project("Monitoring")
    folder = make_run(store, p["id"])
    store.register_run(p["id"], folder)
    with (folder / lockname).open("a+") as stream:
        _acquire(stream, blocking=False)
        try:
            with pytest.raises(ValueError, match="monitored or updated"):
                store.delete_project(p["id"], confirmation=p["name"])
        finally:
            _release(stream)
    assert not store.get_project(p["id"])["deleted"]
    assert store.delete_project(p["id"], confirmation=p["name"])["deleted"]


@pytest.mark.parametrize("content", [b"", b"existing lock metadata"])
def test_idle_guard_does_not_write_windows_lock_bytes(tmp_path, monkeypatch, content):
    class WindowsOS:
        name = "nt"
        def __getattr__(self, name):
            return getattr(os, name)
    monkeypatch.setattr(module, "os", WindowsOS())
    path = tmp_path / ".worker.lock"
    path.write_bytes(content)
    with Workspace._idle_lock(tmp_path, path.name):
        assert path.stat().st_size == len(content)
    assert path.read_bytes() == content


def test_events_are_append_only_and_scoped(store):
    a, b = [store.create_project(name)["id"] for name in ("A", "B")]
    event = store.record_event(a, "file.saved", {"path": "files/notes.txt"})
    assert store.events(a, 1)[0] == event
    assert all(item["project_id"] == b for item in store.events(b))
    with store._connection() as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE events SET kind='changed' WHERE id=?", (event["id"],))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM events WHERE id=?", (event["id"],))


def test_failed_upgrade_keeps_existing_schema_and_projects(store, monkeypatch):
    project = store.create_project("Before upgrade")
    original = module._migration_sql
    version = module._MIGRATIONS[-1][0]
    monkeypatch.setattr(module, "_MIGRATIONS", (*module._MIGRATIONS, (version + 1, "test_upgrade.sql")))
    monkeypatch.setattr(module, "_migration_sql", lambda name: "CREATE TABLE staged_upgrade (id INTEGER);\nINVALID SQL;\n" if name == "test_upgrade.sql" else original(name))
    with pytest.raises(sqlite3.DatabaseError):
        Workspace(store.home)
    assert store.get_project(project["id"])["name"] == "Before upgrade"
    with store._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == version
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='staged_upgrade'").fetchall()


def test_edit_project_audit_rolls_back_and_rejects_archived_project(store):
    project = store.create_project("Edit")
    with pytest.raises(RuntimeError):
        with store.edit_project(project["id"]) as event:
            event("file.saved", {"path": "files/test.txt"})
            raise RuntimeError("Write failed")
    assert [item["kind"] for item in store.events(project["id"])] == ["project.created"]
    with store.edit_project(project["id"]) as event:
        event("file.saved", {"path": "files/test.txt"})
    assert store.events(project["id"], 1)[0]["kind"] == "file.saved"
    store.archive_project(project["id"])
    with pytest.raises(ValueError, match="Restore"):
        with store.edit_project(project["id"]):
            pytest.fail("Archived project became writable")


@pytest.mark.parametrize('content', ['API_KEY=not-a-real-secret', 'password: example-only', 'OTP 123456', 'sk-abcdefghijklmnopqrstuvwxyz'])
def test_secret_chat_memory_and_metadata_never_persist(store, content):
    project = store.create_project('Private project')['id']
    session = store.create_session(project)['id']
    with pytest.raises(ValueError, match='secure'):
        store.append_message(project, session, 'user', content)
    with pytest.raises(ValueError, match='secure'):
        store.save_memory(project, 'Credentials', content)
    with pytest.raises(ValueError, match='secure'):
        store.record_event(project, 'note', {'text': content})
    assert store.get_session(project, session)['messages'] == []
    assert store.list_memory(project) == []
    assert content.encode() not in store.db_path.read_bytes()


def test_unexpected_secret_in_assistant_message_is_redacted(store):
    project = store.create_project('Private project')['id']
    session = store.create_session(project)['id']
    result = store.append_message(project, session, 'assistant', 'Unexpected password: example-only')
    assert 'example-only' not in result['messages'][0]['content']
    assert '[secret omitted]' in result['messages'][0]['content']
