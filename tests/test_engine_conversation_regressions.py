"""Follow-up chats use checked geometry and keep failed calculations out of scientific facts."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from pymatgen.core import Structure

from vasp_slurm_agent import conversation, engine_agent, engine_context, engine_workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.workflow import _check_config
from test_conversation import SETTINGS, provider, route
from test_engine_adapters import _plan
from test_engine_workflow import DATA, PARAMETERS, _drive, _setup, _sha


def load_config(path):
    return ClusterConfig.load(path)


@pytest.fixture
def completed(tmp_path, request):
    engine = getattr(request, "param", "cp2k")
    run, cluster, _, _ = _setup(tmp_path, engine)
    assert _drive(run, cluster)["status"] == "succeeded"
    return run, load_config(run / "config.json")


@pytest.mark.parametrize("completed", ["abinit", "cp2k", "elk"], indirect=True)
def test_continuation_uses_accepted_geometry_and_keeps_it_across_revisions(completed, tmp_path, monkeypatch):
    run, config = completed
    source = engine_context.inspect_continuation(run, 0)
    original = Structure.from_file(DATA / "Si-perturbed.cif")
    accepted = Structure.from_file(source["structure_path"])
    assert not np.allclose(original.frac_coords, accepted.frac_coords)
    folder = conversation.open_run(run, tmp_path)
    first = _plan(["bands"])
    first["operations"] = [{"type": "supercell", "matrix": [2, 1, 1]}]
    revised = _plan(["bands"], {"line_density": 8})
    revised["operations"] = [{"type": "supercell", "matrix": [3, 1, 1]}]
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route("continue", 0)), (config.engine + "_task", first),
        ("dft_conversation", route()), (config.engine + "_task", revised),
    ])
    result = conversation.send(folder, "Double the relaxed cell and calculate bands.", SETTINGS, config=config)
    assert result["pending"]["kind"] == "engine_continuation"
    result = conversation.send(folder, "Use 3x1x1 instead and 8 path points.", SETTINGS, config=config)
    pending = result["pending"]
    assert pending["status"] == "ready", pending
    assert not remaining
    assert pending["plan"]["source_sha256"] == source["structure_sha256"]
    assert pending["context_sha256"] == source["context_sha256"]
    assert seen[1][1]["inherited_parameters"] == PARAMETERS[config.engine]
    assert seen[0][1]["accepted_stages"] == [{"index": 0, "task": "relax"}, {"index": 1, "task": "scf"}]
    assert "2" in json.dumps(seen[3][1]["history"])
    child = tmp_path / "followup"
    conversation.prepare(folder, child, config)
    state = json.loads((child / "run.json").read_text())
    _check_config(child, state)
    assert all(stage["job_id"] is None for stage in state["stages"])
    expected = accepted.copy()
    expected.make_supercell([3, 1, 1])
    actual = Structure.from_file(child / "01_scf/inputs/structure.cif")
    assert len(actual) == len(expected) == 6
    assert np.allclose(actual.frac_coords, expected.frac_coords, atol=1e-7)
    parameters = state["stages"][0]["parameters"]
    assert parameters["line_density"] == 8
    assert parameters["mesh"] == PARAMETERS[config.engine]["mesh"]
    provenance = json.loads((child / "parent.json").read_text())
    assert provenance["source_structure_sha256"] == source["structure_sha256"]
    assert provenance["source_context_sha256"] == source["context_sha256"]
    assert provenance["electronic_state_reused"] is False


def test_changed_continuation_evidence_blocks_preparation(completed, tmp_path, monkeypatch):
    run, config = completed
    folder = conversation.open_run(run, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("continue", 0)), ("cp2k_task", _plan(["bands"]))])
    result = conversation.send(folder, "Calculate bands from the relaxation.", SETTINGS, config=config)
    assert result["pending"]["status"] == "ready"
    (run / "01_relax/outputs/CONTCAR").write_text("changed after review\n")
    with pytest.raises(ValueError, match="accepted|changed"):
        conversation.prepare(folder, tmp_path / "child", config)
    assert not (tmp_path / "child").exists()


def test_explicit_original_source_discards_continuation_base(completed, tmp_path, monkeypatch):
    run, config = completed
    folder = conversation.open_run(run, tmp_path)
    provider(monkeypatch, [
        ("dft_conversation", route("continue", 0)), ("cp2k_task", _plan(["bands"])),
        ("dft_conversation", route(original_source=True)), ("cp2k_task", _plan(["scf"])),
    ])
    conversation.send(folder, "Bands from the relaxed structure.", SETTINGS, config=config)
    result = conversation.send(folder, "Start over from the original upload, SCF only.", SETTINGS, config=config)
    assert result["pending"]["kind"] == "plan" and result["planning_base"] is None
    assert result["pending"]["plan"]["source_sha256"] == _sha(folder / result["source"])


def test_failed_verified_logs_are_redacted_and_never_become_accepted_numbers(tmp_path):
    run, cluster, state, _ = _setup(tmp_path, "abinit")
    cluster.env["REPLAY_CRASH_AT_EXIT"] = "relax"
    engine_workflow.advance(run, cluster)
    remote = Path(state["remote_root"]) / "01_relax"
    with (remote / "slurm.err").open("a") as stream:
        stream.write("\npassword=example-private-value /private/research/input researcher@host.invalid\n")
    state = _drive(run, cluster)
    assert state["status"] == "needs_attention"
    context = engine_context.load_run_context(run)
    facts = context["facts"]
    assert facts["stage_1.evidence_verified"]["value"] is True
    assert facts["stage_1.accepted"]["value"] is False
    assert facts["stage_1.exit_code"]["value"] == 134
    reason = facts["stage_1.failure_reason"]["value"]
    assert "double free or corruption" in reason and "134" in reason
    assert "example-private-value" not in reason and "/private/research" not in reason and "host.invalid" not in reason
    assert len(reason) <= 3500
    assert not any(key.startswith("stage_1.") and ("energy" in key or "band_gap" in key) for key in facts)
    (run / "01_relax/outputs/slurm.err").write_text("forged diagnosis")
    facts = engine_context.load_run_context(run)["facts"]
    assert facts["stage_1.evidence_verified"]["value"] is False
    assert "stage_1.failure_reason" not in facts


@pytest.fixture
def failed_scf(tmp_path):
    run, cluster, state, _ = _setup(tmp_path, "cp2k")
    engine_workflow.advance(run, cluster)
    engine_workflow.advance(run, cluster)
    engine_workflow.advance(run, cluster)
    output = Path(state["remote_root"]) / "02_scf/cp2k.out"
    assert "SCF run converged" in output.read_text()
    output.write_text(output.read_text().replace("SCF run converged", "SCF run NOT converged"))
    state = engine_workflow.advance(run, cluster)
    assert state["status"] == "needs_attention" and state["current_stage"] == 1
    return run, load_config(run / "config.json")


def repair_proposal(source):
    plan = _plan([recipe["task"] for recipe in source["recipes"]])
    for stage, recipe in zip(plan["stages"], source["recipes"]):
        stage["parameters_json"] = json.dumps(recipe["parameters"])
    first = json.loads(plan["stages"][0]["parameters_json"])
    first["max_scf_steps"] = 200
    plan["stages"][0]["parameters_json"] = json.dumps(first)
    return plan


def test_repair_preserves_accepted_prior_geometry_and_frozen_settings(failed_scf, tmp_path, monkeypatch):
    run, config = failed_scf
    source = engine_context.inspect_repair(run)
    assert source["allowed_changes"] == {"max_scf_steps": {"before": 100, "maximum": 200}}
    approved = Structure.from_file(source["structure_path"])
    original = Structure.from_file(DATA / "Si-perturbed.cif")
    assert not np.allclose(original.frac_coords, approved.frac_coords)
    folder = conversation.open_run(run, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("repair")), ("cp2k_task", repair_proposal(source))])
    result = conversation.send(folder, "Repair SCF convergence and keep my remaining tasks.", SETTINGS, config=config)
    pending = result["pending"]
    assert pending["status"] == "ready", pending
    assert pending["kind"] == "engine_repair"
    assert pending["plan"]["changes"] == [{"stage": 0, "key": "max_scf_steps", "before": 100, "after": 200}]
    child = tmp_path / "retry"
    conversation.prepare(folder, child, config)
    state = json.loads((child / "run.json").read_text())
    _check_config(child, state)
    assert state["repair"]["attempt"] == 1
    assert [stage["name"] for stage in state["stages"]] == ["scf", "bands", "dos"]
    assert state["stages"][0]["parameters"]["max_scf_steps"] == 200
    actual = Structure.from_file(child / "01_scf/inputs/structure.cif")
    assert np.allclose(actual.frac_coords, approved.frac_coords, atol=1e-7)
    assert json.loads((child / "proposal.json").read_text())["execution"]["plan_sha256"] == state["plan_sha256"]


@pytest.mark.parametrize("change", ["tolerance", "functional", "geometry", "drop_stage", "too_many_steps"])
def test_repair_rejects_unapproved_changes(failed_scf, change):
    source = engine_context.inspect_repair(failed_scf[0])
    plan = repair_proposal(source)
    parameters = json.loads(plan["stages"][0]["parameters_json"])
    if change == "tolerance":
        parameters["eps_scf"] = 0.1
    elif change == "functional":
        parameters["functional"] = "LDA"
    elif change == "geometry":
        plan["operations"] = [{"type": "supercell", "matrix": [2, 1, 1]}]
    elif change == "drop_stage":
        plan["stages"].pop()
    else:
        parameters["max_scf_steps"] = 201
    plan["stages"][0]["parameters_json"] = json.dumps(parameters)
    with pytest.raises(ValueError, match="repair|remaining tasks"):
        engine_context.check_repair_plan(plan, source)


def test_repair_attempt_limit_is_checked_against_frozen_provenance(failed_scf):
    run, _ = failed_scf
    state = json.loads((run / "run.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    state["repair"] = plan["repair"] = {"attempt": 2, "max_attempts": 2, "source_run_id": "earlier", "changes": []}
    (run / "plan.json").write_text(json.dumps(plan))
    state["plan_sha256"] = _sha(run / "plan.json")
    (run / "run.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Two repair attempts"):
        engine_context.inspect_repair(run)


def test_crash_has_diagnostics_but_no_blind_parameter_retry(tmp_path, monkeypatch):
    run, cluster, _, _ = _setup(tmp_path, "abinit")
    cluster.env["REPLAY_CRASH_AT_EXIT"] = "relax"
    _drive(run, cluster)
    source = engine_context.inspect_repair(run)
    assert source["allowed_changes"] == {}
    folder = conversation.open_run(run, tmp_path)
    seen, _ = provider(monkeypatch, [("dft_conversation", route("repair")), ("abinit_task", repair_proposal(source))])
    result = conversation.send(folder, "Repair this run.", SETTINGS, config=load_config(run / "config.json"))
    assert result["pending"]["status"] == "needs_input"
    assert "double free or corruption" in json.dumps(seen[-1][1]["history"])


def test_cp2k_prompt_distinguishes_residual_from_energy():
    assert "dimensionless SCF residual" in engine_agent._system("cp2k")
    assert "must not be converted to EPS_SCF" in engine_agent._system("cp2k")


def test_planner_receives_configured_resources_without_connection_details(tmp_path, monkeypatch):
    config = ClusterConfig(engine="cp2k", host="cluster.invalid", user="tester", partition="cpu", tasks=1,
        remote_root="/scratch/tester/runs", engine_command="srun cp2k.psmp", engine_library_root="/scratch/tester/library",
        engine_library={name: {"sha256": "0" * 64} for name in ("BASIS_MOLOPT", "GTH_POTENTIALS")})
    seen, _ = provider(monkeypatch, [("cp2k_task", _plan(["scf"]))])
    proposal = engine_agent.draft_task("Run SCF.", DATA / "Si-perturbed.cif", SETTINGS, engine="cp2k", config=config)
    assert proposal["status"] == "ready"
    payload = seen[0][1]
    assert payload["environment"]["resources"] == {"nodes": 1, "tasks": 1, "cpus_per_task": 1, "walltime": "00:30:00"}
    assert payload["environment"]["library"] == [
        {"file": "BASIS_MOLOPT", "checksum_recorded": True}, {"file": "GTH_POTENTIALS", "checksum_recorded": True}]
    text = json.dumps(payload)
    assert config.remote_root not in text and config.engine_library_root not in text
    assert "host" not in payload["environment"] and "user" not in payload["environment"]


def project_run(fixture, tmp_path):
    from vasp_slurm_agent.connections import ConnectionRegistry
    from vasp_slurm_agent.project_execution import ProjectExecution
    from vasp_slurm_agent.workspace import Workspace

    source, config = fixture
    workspace = Workspace(tmp_path / "project-workspace")
    registry = ConnectionRegistry(workspace)
    project = workspace.create_project("Silicon")["id"]
    session = workspace.create_session(project)["id"]
    model = registry.create("model", "Fixture model", {"provider": "codex"})["id"]
    registry.bind(project, model)
    connection = registry.create("hpc", "Fixture cluster", {"host": "test.invalid", "user": "tester",
        "scheduler": "slurm", "remote_base": str(tmp_path / "remote")})["id"]
    registry.bind(project, connection)
    environment = registry.create_environment(connection, "CP2K", "cp2k", {
        "partition": "cpu", "tasks": 1, "engine_command": config.engine_command,
        "engine_library_root": config.engine_library_root, "engine_library": config.engine_library})["id"]
    graph = workspace.create_workflow(project, session, "Silicon", "Relax and spectra", [{
        "id": "si", "name": "Silicon", "engine": "cp2k", "depends_on": [], "spec": {
            "goal": "Relax and calculate spectra", "environment_id": environment,
            "source": {"kind": "file", "path": "files/Si.cif"}}}])
    execution = ProjectExecution(workspace, registry)
    root = workspace.project_paths(project)["runs"] / "original"
    shutil.copytree(source, root)
    registered = workspace.register_run(project, root, graph["id"], "si")
    previous = execution._new(project, graph["id"], graph["steps"][0])
    previous = execution._save(previous, run_id=registered["id"], state=registered["state"]["status"])
    return workspace, execution, project, session, graph["id"], model, root, previous


@pytest.mark.parametrize("interface", ["guided", "mcp"])
def test_project_continuation_is_available_in_both_chat_interfaces(completed, tmp_path, monkeypatch, interface):
    from vasp_slurm_agent import project_agent
    from vasp_slurm_agent.mcp_server import ProjectTools
    from test_project_agent import reply

    workspace, execution, pid, sid, wid, model, original, old = project_run(completed, tmp_path)
    original_state = (original / "run.json").read_bytes()
    replies = [("dft_conversation", route("plan", 0)), ("cp2k_task", _plan(["bands"]))]
    if interface == "guided":
        replies.insert(0, ("dft_project_action", reply("continue_step")))
    seen, remaining = provider(monkeypatch, replies)
    if interface == "guided":
        answer = project_agent.send(workspace, pid, sid, "Use the accepted relaxation for new bands.", model,
                                    workflow_id=wid, step_id="si")
        assert answer["action"] == "continue_step", answer
    else:
        tools = ProjectTools(workspace, pid, sid, model)
        names = {item["name"] for item in tools.definitions()}
        assert {"continue_step", "repair_step"} <= names
        answer = tools.call("continue_step", {"workflow_id": wid, "step_id": "si",
            "instructions": "Use the accepted relaxation for new bands."})
        assert answer["state"] == "ready", answer
    assert not remaining
    assert next(payload for name, payload in seen if name == "dft_conversation")["requested_action"] == "continue"
    attempt = execution.latest(pid, wid, "si")
    assert attempt["id"] != old["id"] and attempt["source_context"]["kind"] == "continuation"
    assert attempt["proposal"]["kind"] == "engine_continuation"
    prepared = execution.prepare_step(pid, wid, "si")
    assert prepared["run_id"] != old["run_id"] and len(workspace.list_runs(pid)) == 2
    assert (original / "run.json").read_bytes() == original_state
    assert prepared["tasks"] == ["scf", "bands"]
    from vasp_slurm_agent.project_context import build_project_context
    step = build_project_context(workspace, pid, workflow_ids=[wid])["workflows"][0]["steps"][0]
    assert step["name"] == "Silicon"
    assert step["current_attempt"] == {"number": 2, "tasks": ["scf", "bands"],
                                        "status": "prepared", "run_id": prepared["run_id"]}
    assert execution.attempts(pid, wid, "si")[0]["tasks"] == ["relax", "scf", "bands", "dos"]
    child = Path(workspace.get_run(pid, prepared["run_id"])["run_dir"])
    parent = json.loads((child / "parent.json").read_text())
    assert parent["source_stage"] == 0 and parent["source_structure_sha256"] == _sha(original / "01_relax/outputs/CONTCAR")


def test_mcp_repair_cannot_fall_back_to_original_upload(failed_scf, tmp_path, monkeypatch):
    from vasp_slurm_agent.mcp_server import ProjectTools

    workspace, execution, pid, sid, wid, model, original, old = project_run(failed_scf, tmp_path)
    source = engine_context.inspect_repair(original)
    provider(monkeypatch, [("dft_conversation", route(original_source=True)), ("cp2k_task", repair_proposal(source))])
    tools = ProjectTools(workspace, pid, sid, model)
    answer = tools.call("repair_step", {"workflow_id": wid, "step_id": "si", "instructions": "Increase the SCF iteration limit."})
    assert answer["state"] == "ready", answer
    assert answer["plan"]["changes"] == [{"stage": 0, "key": "max_scf_steps", "before": 100, "after": 200}]
    attempt = execution.latest(pid, wid, "si")
    assert attempt["proposal"]["kind"] == "engine_repair"
    assert attempt["proposal"]["plan"]["source_sha256"] == source["structure_sha256"]
    assert len(workspace.list_runs(pid)) == 1 and attempt["id"] != old["id"]


@pytest.mark.parametrize("engine", ["abinit", "cp2k", "elk"])
def test_spectrum_controls_do_not_insert_duplicate_scf_but_mesh_changes_do(tmp_path, engine):
    base = PARAMETERS[engine]
    spectral = {"line_density": 20, "dos_mesh": [6, 6, 6], "kpoints": [[0, 0, 0], [0.5, 0, 0]]}
    if engine == "cp2k":
        spectral["dos_broadening_ev"] = 0.2
    original = engine_workflow.prepare_plan(DATA / "Si-perturbed.cif", tmp_path / "base", None,
                                            ["scf"], base, engine=engine)
    revised = engine_workflow.prepare_plan(DATA / "Si-perturbed.cif", tmp_path / "spectral", None,
                                           ["scf"], {**base, **spectral}, engine=engine)
    name = original["stages"][0]["metadata"]["input_file"]
    assert name == revised["stages"][0]["metadata"]["input_file"]
    assert (tmp_path / "base/01_scf/inputs" / name).read_bytes() == (tmp_path / "spectral/01_scf/inputs" / name).read_bytes()
    state = engine_workflow.prepare_plan(DATA / "Si-perturbed.cif", tmp_path / "chain", None,
        ["relax", "scf", "bands", "dos"], stage_parameters=[base, base, {**base, "line_density": 20},
                                                            {**base, "dos_mesh": [6, 6, 6]}], engine=engine)
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands", "dos"]
    assert state["stages"][2]["restart_from"] == state["stages"][3]["restart_from"] == 1
    sequence = engine_workflow._sequence(["scf", "bands"], {}, [base, {**base, "mesh": [4, 4, 4]}])
    assert [stage["name"] for stage in sequence] == ["scf", "scf", "bands"]
