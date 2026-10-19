import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import PureWindowsPath
import threading

import pytest

from vasp_slurm_agent import project_files
from vasp_slurm_agent.workspace import Workspace


@pytest.fixture
def project(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    item = workspace.create_project("Crystal")
    return workspace, item["id"]


def test_edits_require_the_current_revision_and_keep_previous_bytes(project):
    workspace, project_id = project
    first = project_files.write_file(workspace, project_id, "files/INCAR", "ENCUT = 520\n")
    changed = project_files.write_file(workspace, project_id, "files/INCAR", "ENCUT = 600\n",
                                       expected_sha256=first["sha256"])
    with pytest.raises(ValueError, match="changed"):
        project_files.write_file(workspace, project_id, "files/INCAR", "ENCUT = 700\n",
                                 expected_sha256=first["sha256"])
    preview = project_files.read_file(workspace, project_id, "files/INCAR")
    assert preview["content"] == "ENCUT = 600\n"
    assert preview["sha256"] == changed["sha256"]
    saved = workspace.project_paths(project_id)["files"] / ".history" / (first["sha256"] + ".bin")
    assert saved.read_bytes() == b"ENCUT = 520\n"
    assert [item["path"] for item in project_files.list_files(workspace, project_id)] == ["files/INCAR"]


def test_prepared_files_are_viewable_but_not_editable(project):
    workspace, project_id = project
    path = workspace.project_paths(project_id)["runs"] / "task" / "inputs" / "pw.in"
    path.parent.mkdir(parents=True)
    path.write_text("&CONTROL\n calculation = 'scf'\n/\n")
    assert project_files.read_file(workspace, project_id, "runs/task/inputs/pw.in")["editable"] is False
    with pytest.raises(ValueError):
        project_files.write_file(workspace, project_id, "runs/task/inputs/pw.in", "changed")
    assert "calculation" in path.read_text()


def test_project_boundaries_and_symlinks(project):
    workspace, project_id = project
    other = workspace.create_project("Other")["id"]
    project_files.write_file(workspace, other, "files/private.txt", "Research B")
    other_path = workspace.project_paths(other)["files"] / "private.txt"
    own_files = workspace.project_paths(project_id)["files"]
    (own_files / "link.txt").symlink_to(other_path)
    for relative in (str(other_path), f"files/../../{other}/files/private.txt", "files/link.txt"):
        with pytest.raises(ValueError):
            project_files.read_file(workspace, project_id, relative)
        with pytest.raises(ValueError):
            project_files.write_file(workspace, project_id, relative, "changed")
    assert project_files.list_files(workspace, project_id) == []
    assert other_path.read_text() == "Research B"


@pytest.mark.parametrize("name,content", [
    ("POTCAR", "private potential data"),
    ("POTCAR.gz", b"compressed data"),
    (".env", "MODEL=foo"),
    ("config.json", '{"api_key": "secret"}'),
    ("notes.txt", "api_key = never-store-this"),
    ("key.pem", "-----BEGIN OPENSSH PRIVATE KEY-----\nabc"),
])
def test_secret_or_potential_upload_rejected(project, name, content):
    workspace, project_id = project
    with pytest.raises(ValueError):
        project_files.write_file(workspace, project_id, "files/" + name, content)
    assert project_files.list_files(workspace, project_id) == []


def test_selected_import_is_private_and_does_not_replace_existing_file(project, tmp_path):
    workspace, project_id = project
    source = tmp_path / "Si.cif"
    source.write_text("data_Si\n_cell_length_a 5.43\n")
    receipt = project_files.import_file(workspace, project_id, source)
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    stored = workspace.project_paths(project_id)["files"] / "Si.cif"
    assert stored.read_bytes() == source.read_bytes()
    if __import__("os").name != "nt":
        assert stored.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="changed"):
        project_files.import_file(workspace, project_id, source)


def test_portable_paths_survive_import_edit_read_list_and_audit(project, tmp_path):
    workspace, project_id = project
    source = tmp_path / "Si.cif"
    source.write_text("data_Si\n")
    imported = project_files.import_file(workspace, project_id, source)
    assert imported["path"] == "files/Si.cif"
    saved = project_files.write_file(workspace, project_id, "files/subfolder/note.txt", "Original")
    changed = project_files.write_file(workspace, project_id, saved["path"], "Updated", expected_sha256=saved["sha256"])
    assert changed["path"] == "files/subfolder/note.txt"
    assert project_files.read_file(workspace, project_id, changed["path"])["path"] == changed["path"]
    assert {item["path"] for item in project_files.list_files(workspace, project_id)} == {imported["path"], changed["path"]}
    assert all("\\" not in event["details"]["path"] for event in workspace.events(project_id) if event["kind"] == "file.saved")


def test_project_path_validation_does_not_roundtrip_through_native_windows_path(project, monkeypatch):
    workspace, project_id = project
    monkeypatch.setattr(project_files, "Path", PureWindowsPath)
    resolved = project_files._checked(workspace, project_id, "files/nested/Si.cif")
    assert resolved == workspace.project_paths(project_id)["files"] / "nested" / "Si.cif"


@pytest.mark.parametrize("relative", [r"files\Si.cif", r"files/nested\Si.cif", "files/C:Si.cif", "files/../runs/Si.cif", "C:/files/Si.cif"])
def test_portable_project_paths_still_reject_windows_escapes(project, relative):
    workspace, project_id = project
    with pytest.raises(ValueError):
        project_files.write_file(workspace, project_id, relative, "data_Si\n")
    assert project_files.list_files(workspace, project_id) == []


def test_archived_project_is_read_only(project):
    workspace, project_id = project
    project_files.write_file(workspace, project_id, "files/note.txt", "Draft")
    workspace.archive_project(project_id)
    with pytest.raises(ValueError, match="Restore|reopen"):
        project_files.write_file(workspace, project_id, "files/new.txt", "New")
    assert project_files.read_file(workspace, project_id, "files/note.txt")["content"] == "Draft"


def test_preview_has_explicit_size_and_binary_limits(project, monkeypatch):
    workspace, project_id = project
    project_files.write_file(workspace, project_id, "files/binary.dat", b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="binary"):
        project_files.read_file(workspace, project_id, "files/binary.dat")
    project_files.write_file(workspace, project_id, "files/table.csv", "x,y\n1,2\n")
    monkeypatch.setattr(project_files, "MAX_PREVIEW_BYTES", 4)
    with pytest.raises(ValueError, match="preview"):
        project_files.read_file(workspace, project_id, "files/table.csv")


def test_audit_failure_restores_original_file(project, monkeypatch):
    workspace, project_id = project
    first = project_files.write_file(workspace, project_id, "files/note.txt", "Original")
    original_event = workspace._event

    def fail(db, identifier, kind, details):
        if kind == "file.saved":
            raise OSError("Database write failed")
        return original_event(db, identifier, kind, details)

    monkeypatch.setattr(workspace, "_event", fail)
    with pytest.raises(OSError, match="Database"):
        project_files.write_file(workspace, project_id, "files/note.txt", "Replacement",
                                 expected_sha256=first["sha256"])
    assert project_files.read_file(workspace, project_id, "files/note.txt")["content"] == "Original"
    with pytest.raises(OSError):
        project_files.write_file(workspace, project_id, "files/new.txt", "New")
    assert not workspace.resolve_path(project_id, "files/new.txt").exists()
    assert len([item for item in workspace.events(project_id) if item["kind"] == "file.saved"]) == 1


def test_archive_waits_for_file_edit_and_its_audit(project, monkeypatch):
    workspace, project_id = project
    entered, proceed, archiving = threading.Event(), threading.Event(), threading.Event()
    atomic = project_files._atomic

    def paused(path, data):
        entered.set()
        assert proceed.wait(5)
        return atomic(path, data)

    def archive():
        archiving.set()
        return workspace.archive_project(project_id)

    monkeypatch.setattr(project_files, "_atomic", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        edit = pool.submit(project_files.write_file, workspace, project_id, "files/note.txt", "Saved")
        try:
            assert entered.wait(5)
            archived = pool.submit(archive)
            assert archiving.wait(5)
        finally:
            proceed.set()
        assert edit.result(timeout=5)["size"] == 5
        assert archived.result(timeout=5)["archived"]
    events = workspace.events(project_id)
    assert [item["kind"] for item in events[:2]] == ["project.archived", "file.saved"]
    assert project_files.read_file(workspace, project_id, "files/note.txt")["content"] == "Saved"
