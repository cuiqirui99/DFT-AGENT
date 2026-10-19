"""Project transfers use parser fixtures, not new cluster acceptance evidence."""

from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath
import shutil

import pytest

from vasp_slurm_agent import project_artifacts as artifacts
from vasp_slurm_agent.workspace import Workspace
from test_continuation import parent
from test_explanation import accepted_run
from test_qe_context import accepted_qe
from test_batch_context import accepted_batch
from test_batch import setup


@pytest.fixture
def project(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    return workspace, workspace.create_project("Geometry")["id"]


def register(project, original):
    workspace, project_id = project
    destination = workspace.project_paths(project_id)["runs"] / "source"
    shutil.copytree(original, destination)
    return workspace.register_run(project_id, destination), destination


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_geometry_transfer_preserves_native_bytes_and_rechecks_evidence(project, request, engine):
    source = request.getfixturevalue("parent")[0] if engine == "vasp" else request.getfixturevalue("accepted_qe")
    record, root = register(project, source)
    workspace, project_id = project
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    inspected = artifacts.inspect_run(workspace, project_id, record["id"])
    assert inspected["accepted"] and inspected["structure_available"]
    assert inspected["process_success"]
    receipt = artifacts.transfer_structure(workspace, project_id, record["id"], None,
                                            "qe" if engine == "vasp" else "vasp", "artifacts/next/POSCAR")
    expected = root / "01_scf/outputs" / ("final_structure.vasp" if engine == "vasp" else "CONTCAR")
    actual = workspace.resolve_path(project_id, receipt["path"])
    assert actual.read_bytes() == expected.read_bytes()
    assert receipt["source_engine"] == engine
    assert receipt["units"] == {"lattice": "angstrom", "positions": "fractional"}
    assert receipt["species"] == ["Si", "Si"]
    assert receipt["requires_fresh_scf"]
    assert not receipt["electronic_state_reused"]
    assert not receipt["magnetic_state_transferred"]
    assert not receipt["pseudopotentials_transferred"]
    assert str(root) not in json.dumps(receipt)
    assert artifacts.verify_transfer(workspace, project_id, receipt) == receipt
    assert artifacts.transfer_structure(workspace, project_id, record["id"], None,
                                         receipt["target_engine"], receipt["path"]) == receipt
    assert len(workspace.list_artifacts(project_id)) == 2
    assert before == {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("change", ["source", "target", "proof", "record"])
def test_changed_transfer_is_rejected(project, parent, change):
    workspace, project_id = project
    record, root = register(project, parent[0])
    receipt = artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", "files/POSCAR")
    if change == "source":
        (root / "01_scf/outputs/OUTCAR").write_text("Changed")
        assert not artifacts.inspect_run(workspace, project_id, record["id"])["accepted"]
    elif change == "target":
        workspace.resolve_path(project_id, receipt["path"]).write_text("Changed")
    elif change == "proof":
        workspace.resolve_path(project_id, receipt["path"] + ".provenance.json").write_text("{}")
    else:
        receipt["species"] = ["Fe"]
    with pytest.raises(ValueError):
        artifacts.verify_transfer(workspace, project_id, receipt)


def test_foreign_run_and_paths_rejected(project, parent):
    workspace, project_id = project
    record, root = register(project, parent[0])
    other = workspace.create_project("Other")["id"]
    with pytest.raises(ValueError, match="not found"):
        artifacts.transfer_structure(workspace, other, record["id"], 0, "qe", "files/POSCAR")
    for path in ("../POSCAR", "runs/POSCAR", "config/POSCAR", "files/structure.cif", str(root / "POSCAR"), r"files\POSCAR", r"files/nested\POSCAR"):
        with pytest.raises(ValueError):
            artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", path)
    (root / "redirect").symlink_to(workspace.project_paths(other)["files"], target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        artifacts.inspect_run(workspace, project_id, record["id"])


def test_structure_provenance_paths_are_portable_on_windows(project, parent, monkeypatch):
    workspace, project_id = project
    record, _ = register(project, parent[0])
    def local_or_windows_path(value):
        return PureWindowsPath(value) if isinstance(value, str) and value.startswith(("files/", "artifacts/")) else Path(value)
    monkeypatch.setattr(artifacts, "Path", local_or_windows_path)
    receipt = artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", "artifacts/next/POSCAR")
    assert receipt["path"] == "artifacts/next/POSCAR"
    proof = workspace.resolve_path(project_id, "artifacts/next/POSCAR.provenance.json", must_exist=True)
    assert json.loads(proof.read_text())["path"] == receipt["path"]
    assert artifacts.verify_transfer(workspace, project_id, receipt) == receipt


def test_source_change_during_copy_rolls_back(project, parent, monkeypatch):
    workspace, project_id = project
    record, root = register(project, parent[0])
    original = artifacts._atomic
    def changing(path, data):
        original(path, data)
        if path.name == "POSCAR":
            (root / "01_scf/outputs/OUTCAR").write_text("Changed while copying")
    monkeypatch.setattr(artifacts, "_atomic", changing)
    with pytest.raises(ValueError):
        artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", "artifacts/new/POSCAR")
    assert not workspace.resolve_path(project_id, "artifacts/new/POSCAR").exists()
    assert not workspace.resolve_path(project_id, "artifacts/new/POSCAR.provenance.json").exists()
    assert not workspace.list_artifacts(project_id)


def test_archive_and_conflicting_files_are_not_overwritten(project, parent):
    workspace, project_id = project
    record, _ = register(project, parent[0])
    path = workspace.resolve_path(project_id, "files/POSCAR")
    path.write_text("Existing input")
    with pytest.raises((ValueError, FileNotFoundError)):
        artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", "files/POSCAR")
    assert path.read_text() == "Existing input"
    workspace.archive_project(project_id)
    with pytest.raises(ValueError, match="Restore"):
        artifacts.transfer_structure(workspace, project_id, record["id"], 0, "qe", "files/new.vasp")
    assert not workspace.resolve_path(project_id, "files/new.vasp").exists()


def test_batch_is_freshly_inspected_but_requires_member_for_transfer(project, accepted_batch):
    workspace, project_id = project
    record, root = register(project, accepted_batch)
    assert artifacts.inspect_run(workspace, project_id, record["id"])["accepted"]
    with pytest.raises(ValueError, match="not a batch"):
        artifacts.transfer_structure(workspace, project_id, record["id"], None, "qe", "files/POSCAR")
    output = next(root.rglob("OUTCAR"))
    output.write_text("Changed")
    assert not artifacts.inspect_run(workspace, project_id, record["id"])["accepted"]


def test_none_selects_last_accepted_structure_not_final_unaccepted_stage(project, parent, monkeypatch):
    workspace, project_id = project
    record, root = register(project, parent[0])
    original = artifacts.explanation.load_run_context
    context = original(root)
    state = record["state"]
    state["stages"].append({"folder": "02_bands", "name": "bands", "status": "planned"})
    (root / "run.json").write_text(json.dumps(state))
    context["facts"]["stage_2.accepted"] = {"value": False}
    monkeypatch.setattr(artifacts.explanation, "load_run_context", lambda root: deepcopy(context))
    seen = []
    def inspected(record, root, stage):
        seen.append(stage)
        source = root / "01_scf/outputs/final_structure.vasp"
        return {"source_stage": stage, "context_sha256": context["context_sha256"]}, source, artifacts._hash(source.read_bytes())
    monkeypatch.setattr(artifacts, "_inspect", inspected)
    receipt = artifacts.transfer_structure(workspace, project_id, record["id"], None, "qe", "files/POSCAR")
    assert receipt["source_stage"] == 0
    assert seen and set(seen) == {0}
