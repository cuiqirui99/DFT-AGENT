"""A model can inspect one project's records without reading another project."""

import json
from pathlib import Path
import shutil

import pytest

from vasp_slurm_agent import agent, project_context, project_files, workflow
from vasp_slurm_agent.workspace import Workspace
from test_explanation import accepted_run
from test_qe_context import accepted_qe


@pytest.fixture
def projects(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Project context must not call a model or cluster")
    monkeypatch.setattr(agent, "_request_structured", blocked)
    monkeypatch.setattr(workflow, "SSHTransport", blocked)
    store = Workspace(tmp_path / "workspace")
    first = store.create_project("Crystal A")["id"]
    second = store.create_project("Crystal B")["id"]
    return store, first, second


def session(store, project, marker):
    item = store.create_session(project, marker)
    store.append_message(project, item["id"], "user", marker)
    return item["id"]


def registered_run(store, project, source):
    destination = store.project_paths(project)["runs"] / "accepted"
    shutil.copytree(source, destination)
    return store.register_run(project, destination), destination


def test_owned_history_files_memory_and_workflows_only(projects):
    store, first, second = projects
    own = session(store, first, "Own research question")
    other = session(store, second, "Foreign research question")
    store.save_memory(first, "choice", "Keep the cell fixed")
    store.save_memory(second, "choice", "Foreign confidential finding")
    project_files.write_file(store, first, "files/Si.cif", "own-file-content")
    project_files.write_file(store, second, "files/CrSBr.cif", "foreign-file-content")
    private = store.project_paths(first)["config"] / "cluster.json"
    private.write_text('{"host":"do-not-export-host", "api_key":"do-not-export-key"}')
    own_work = store.create_workflow(first, own, title="Relax then phonons", goal="Check the structure", steps=[
        {"id": "relax", "name": "Relax", "engine": "vasp", "spec": {"private_value": "do-not-export-spec"}},
        {"id": "phonons", "name": "Phonons", "engine": "qe", "depends_on": ["relax"]},
    ])
    store.create_workflow(second, other, title="Foreign workflow", goal="Foreign method",
                          steps=[{"id": "scf", "name": "SCF", "engine": "qe"}])
    context = project_context.build_project_context(store, first, session_id=own)
    payload = json.dumps(context)
    assert "Own research question" in payload and "Keep the cell fixed" in payload
    assert context["files"] == [{"path": "files/Si.cif", "size": 16}]
    assert context["memory"][0]["verified"] is False
    assert context["workflows"][0]["id"] == own_work["id"]
    assert context["workflows"][0]["steps"][1]["depends_on"] == ["relax"]
    for hidden in ("Foreign", "foreign-file-content", "own-file-content", "do-not-export", str(store.project_paths(first)["root"])):
        assert hidden not in payload


@pytest.mark.parametrize("selector", ["session", "workflow", "artifact", "run"])
def test_foreign_record_rejected_before_evidence_read(projects, accepted_run, monkeypatch, selector):
    store, first, second = projects
    other_session = session(store, second, "Foreign")
    if selector == "session":
        arguments = {"session_id": other_session}
    elif selector == "workflow":
        item = store.create_workflow(second, other_session, title="Foreign", goal="Foreign",
                                     steps=[{"id": "scf", "name": "SCF", "engine": "qe"}])
        arguments = {"workflow_ids": [item["id"]]}
    elif selector == "artifact":
        path = store.project_paths(second)["artifacts"] / "result.csv"
        path.write_text("gap_ev\n1.0\n")
        item = store.register_artifact(second, path, kind="table")
        arguments = {"artifact_ids": [item["id"]]}
    else:
        item, _ = registered_run(store, second, accepted_run[0])
        arguments = {"run_ids": [item["id"]]}
    monkeypatch.setattr(project_context, "load_run_context", lambda *args: pytest.fail("Foreign evidence was read"))
    with pytest.raises(ValueError):
        project_context.build_project_context(store, first, **arguments)


def test_result_facts_refresh_after_output_changes(projects, accepted_run):
    store, project, _ = projects
    item, copied = registered_run(store, project, accepted_run[0])
    selected = {"run_ids": [item["id"]]}
    first = project_context.build_project_context(store, project, **selected)
    result = first["runs"][0]
    assert result["status"] == "succeeded"
    assert result["facts"]["stage_1.energy_per_atom"]["value"] == -5.0
    assert "source" not in result["facts"]["stage_1.energy_per_atom"]
    payload = json.dumps(first)
    assert "private-host" not in payload and "private-user" not in payload and "/private/remote" not in payload
    output = copied / accepted_run[1].relative_to(accepted_run[0]) / "vasprun.xml"
    output.write_text("changed solver output")
    second = project_context.build_project_context(store, project, **selected)
    assert second["runs"][0]["status"] == "needs_attention"
    assert "stage_1.energy_per_atom" not in second["runs"][0]["facts"]
    assert first["runs"][0]["context_sha256"] != second["runs"][0]["context_sha256"]
    assert first["context_sha256"] != second["context_sha256"]
    assert (accepted_run[1] / "vasprun.xml").read_text() == "verified solver fixture"


def test_qe_context_uses_current_solver_evidence(projects, accepted_qe):
    store, project, _ = projects
    item, copied = registered_run(store, project, accepted_qe)
    first = project_context.build_project_context(store, project, run_ids=[item["id"]])
    result = first["runs"][0]
    assert result["engine"] == "qe" and result["status"] == "succeeded"
    assert result["facts"]["stage_1.energy"]["value"] == pytest.approx(-272.11386245988)
    assert result["facts"]["stage_1.band_gap"]["value"] > 0
    assert "pseudo_manifest" not in json.dumps(first)
    xml = copied / "01_scf/outputs/data-file-schema.xml"
    xml.write_bytes(xml.read_bytes() + b"\nchanged")
    second = project_context.build_project_context(store, project, run_ids=[item["id"]])
    assert second["runs"][0]["status"] == "needs_attention"
    assert "stage_1.energy" not in second["runs"][0]["facts"]
    assert result["context_sha256"] != second["runs"][0]["context_sha256"]


def test_registered_identity_change_and_symlink_evidence_rejected(projects, accepted_run, monkeypatch):
    store, project, second = projects
    item, copied = registered_run(store, project, accepted_run[0])
    foreign = store.project_paths(second)["files"] / "unpublished.txt"
    foreign.write_text("unpublished")
    redirected = copied / "extra"
    redirected.symlink_to(foreign)
    monkeypatch.setattr(project_context, "load_run_context", lambda *args: pytest.fail("Redirected evidence was read"))
    with pytest.raises(ValueError, match="symbolic"):
        project_context.build_project_context(store, project, run_ids=[item["id"]])
    redirected.unlink()
    path = copied / "run.json"
    state = json.loads(path.read_text())
    state["run_id"] = "another-run"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity"):
        project_context.build_project_context(store, project, run_ids=[item["id"]])


def test_secure_material_omitted_from_files_and_messages(projects):
    store, project, _ = projects
    selected = session(store, project, "Analyze Si")
    # Legacy imported history can predate the ingress guard. Context still
    # redacts it; new ordinary messages are rejected before persistence.
    store.append_message(project, selected, "user", "Legacy imported message")
    with store._transaction() as db:
        db.execute("UPDATE messages SET content=? WHERE project_id=? AND session_id=? AND role='user'",
                   ("OPENAI_API_KEY=made-up-test-secret; password=another-test-secret", project, selected))
    store.append_message(project, selected, "assistant", "Source /private/research/Si.cif; ENCUT = 520 eV.")
    for name in ("POTCAR", "Si.UPF", "vasp_license", "credentials.json", "key.pem"):
        path = store.project_paths(project)["files"] / name
        path.write_text("restricted-file-content")
        store.register_artifact(project, path, kind="input")
    context = project_context.build_project_context(store, project, session_id=selected)
    payload = json.dumps(context)
    for hidden in ("made-up-test-secret", "another-test-secret", "/private/research", "restricted-file-content", "POTCAR", "Si.UPF", "key.pem"):
        assert hidden not in payload
    assert "520 eV" in payload
    assert context["artifacts"] == []


def test_history_limits_are_explicit_without_changing_saved_history(projects):
    store, project, _ = projects
    selected = session(store, project, "First constraint")
    store.append_message(project, selected, "assistant", "First answer")
    store.append_message(project, selected, "user", "Latest question")
    context = project_context.build_project_context(store, project, session_id=selected, limits={"messages": 2})
    assert context["omitted"]["messages"] == 1
    assert len(context["session"]["history"]) == 2
    assert len(store.get_session(project, selected)["messages"]) == 3
    with pytest.raises(ValueError, match="too large"):
        project_context.build_project_context(store, project, session_id=selected, limits={"characters": 50})


def test_context_restores_from_reopened_workspace(projects, tmp_path):
    store, project, _ = projects
    selected = session(store, project, "Continue from the accepted relaxation")
    store.save_memory(project, "constraint", "Keep SOC enabled")
    before = project_context.build_project_context(store, project, session_id=selected)
    reopened = Workspace(tmp_path / "workspace")
    assert project_context.build_project_context(reopened, project, session_id=selected) == before


@pytest.mark.parametrize("operation", ["archive", "delete"])
def test_inactive_project_has_no_model_context(projects, operation):
    store, project, _ = projects
    if operation == "archive":
        store.archive_project(project)
    else:
        store.delete_project(project, confirmation=store.get_project(project)["name"])
    with pytest.raises(ValueError, match="active project"):
        project_context.build_project_context(store, project)
