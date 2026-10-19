"""Project commands use the workspace without changing path-based execution."""

from importlib.resources import files
import json
from pathlib import Path
import sys

import pytest

from vasp_slurm_agent import cli


@pytest.fixture
def invoke(monkeypatch, capsys):
    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["dft-agent", *map(str, arguments)])
        status = cli.main()
        output = capsys.readouterr()
        return status, json.loads(output.out) if output.out.strip() else None, output.err

    return run


def test_project_cli_lifecycle_preserves_files_and_identity(invoke):
    status, project, error = invoke("project", "create", "Silicon study")
    assert status == 0 and not error
    identifier = project["id"]
    root = Path(project["root"])
    status, shown, _ = invoke("project", "show", identifier)
    assert status == 0 and shown["project"]["id"] == identifier
    assert set(shown["paths"]) == {"root", "files", "runs", "memory", "config", "artifacts"}
    assert all(Path(path).is_dir() for path in shown["paths"].values())
    source = Path(shown["paths"]["files"]) / "notes.txt"
    source.write_text("Keep this project evidence.\n")

    status, renamed, _ = invoke("project", "rename", identifier, "Silicon bands")
    assert status == 0 and renamed["name"] == "Silicon bands"
    assert renamed["id"] == identifier and Path(renamed["root"]) == root
    assert invoke("project", "open", identifier)[1]["project"]["opened_at"]
    assert invoke("project", "archive", identifier)[1]["archived"]
    assert invoke("project", "list")[1] == []
    assert [item["id"] for item in invoke("project", "list", "--include-archived")[1]] == [identifier]
    assert not invoke("project", "archive", identifier, "--unarchive")[1]["archived"]

    status, _, error = invoke("project", "delete", identifier, "--confirm-name", "Silicon study")
    assert status == 1 and error.startswith("Error:")
    assert not invoke("project", "show", identifier)[1]["project"]["deleted"]
    assert invoke("project", "delete", identifier, "--confirm-name", "Silicon bands")[1]["deleted"]
    assert invoke("project", "list")[1] == []
    assert invoke("project", "list", "--include-deleted")[1][0]["id"] == identifier
    assert source.read_text() == "Keep this project evidence.\n"
    status, restored, _ = invoke("project", "restore", identifier)
    assert status == 0 and not restored["deleted"]
    assert restored["id"] == identifier and Path(restored["root"]) == root
    assert source.read_text() == "Keep this project evidence.\n"


def test_project_cli_requires_explicit_identifier_and_deletion_name(invoke, monkeypatch):
    project = invoke("project", "create", "Project A")[1]
    monkeypatch.setattr(sys, "argv", ["dft-agent", "project", "delete", project["id"]])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert not invoke("project", "show", project["id"])[1]["project"]["deleted"]
    status, result, error = invoke("project", "show", "missing-project")
    assert status == 1 and result is None and error.startswith("Error:")


def test_project_lists_are_scoped_and_read_only(invoke):
    from vasp_slurm_agent import workflow
    from vasp_slurm_agent.workspace import Workspace

    first = invoke("project", "create", "First")[1]
    second = invoke("project", "create", "Second")[1]
    workspace = Workspace()
    session = workspace.create_session(first["id"], "Silicon bands")
    workspace.append_message(first["id"], session["id"], "user", "Calculate silicon bands.")
    paths = workspace.project_paths(first["id"])
    source = paths["files"] / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    run = paths["runs"] / "silicon"
    workflow.prepare_run(source, run, None, "scf", {})
    registered = workspace.register_run(first["id"], run)
    before = {item["id"]: item for item in workspace.list_projects()}
    original_state = (run / "run.json").read_bytes()

    status, sessions, _ = invoke("project", "sessions", first["id"])
    assert status == 0 and [item["id"] for item in sessions] == [session["id"]]
    assert sessions[0]["project_id"] == first["id"] and "messages" not in sessions[0]
    status, runs, _ = invoke("project", "runs", first["id"])
    assert status == 0 and [item["id"] for item in runs] == [registered["id"]]
    assert runs[0]["project_id"] == first["id"] and runs[0]["state"]["status"] == "planned"
    assert invoke("project", "sessions", second["id"])[:2] == (0, [])
    assert invoke("project", "runs", second["id"])[:2] == (0, [])
    assert {item["id"]: item for item in workspace.list_projects()} == before
    assert (run / "run.json").read_bytes() == original_state


def test_open_project_does_not_redirect_legacy_prepare(tmp_path, invoke, monkeypatch):
    from vasp_slurm_agent import workflow
    from vasp_slurm_agent.paths import default_config_path, default_runs_root

    def forbidden(*args, **kwargs):
        pytest.fail("Project management and input preparation must not contact the cluster.")

    monkeypatch.setattr(workflow, "SSHTransport", forbidden)
    monkeypatch.setattr(workflow, "start_worker", forbidden)
    defaults = default_config_path(), default_runs_root()
    project = invoke("project", "create", "Unrelated project")[1]
    assert invoke("project", "open", project["id"])[0] == 0
    assert (default_config_path(), default_runs_root()) == defaults
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    destination = tmp_path / "explicit-legacy-run"
    status, state, _ = invoke("prepare", source, destination, "--task", "scf")
    assert status == 0 and state["status"] == "planned"
    assert (destination / "run.json").is_file()
    assert all(stage["job_id"] is None for stage in state["stages"])
    assert invoke("project", "runs", project["id"])[:2] == (0, [])
