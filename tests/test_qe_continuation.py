import json
from pathlib import Path

import pytest

from vasp_slurm_agent import qe_continuation as continuation, qe_workflow
from test_qe_context import accepted_qe
from test_qe_workflow import FakeTransport, complete
from test_qe import xml_output


def test_bands_continuation_reuses_verified_scf_without_resubmission(accepted_qe, tmp_path):
    before = {str(path): qe_workflow._digest(path) for path in accepted_qe.rglob("*") if path.is_file()}
    inspected = continuation.inspect_continuation(accepted_qe, 0)
    target = tmp_path / "bands"
    state = continuation.prepare_continuation(accepted_qe, 0, target, None, ["bands"], expected_context_sha256=inspected["context_sha256"])
    assert [stage["name"] for stage in state["stages"]] == ["bands"]
    assert state["status"] == "planned"
    assert state["stages"][0]["external_checkpoint"]
    qe_workflow._check_config(target, state)
    assert json.loads((target / "parent.json").read_text())["scf_reused"]
    assert before == {str(path): qe_workflow._digest(path) for path in accepted_qe.rglob("*") if path.is_file()}
    assert not any(stage["job_id"] for stage in state["stages"])


def test_changed_cutoff_recalculates_charge(accepted_qe, tmp_path):
    state = continuation.prepare_continuation(accepted_qe, 0, tmp_path / "changed", None, ["bands"], {"ecutwfc": 80})
    assert [stage["name"] for stage in state["stages"]] == ["scf", "bands"]
    assert not state["stages"][0].get("external_checkpoint")


def test_stricter_scf_keeps_new_convergence_check(accepted_qe, tmp_path):
    state = continuation.prepare_continuation(accepted_qe, 0, tmp_path / "stricter", None, ["bands"], {"conv_thr": 1e-12})
    assert [stage["name"] for stage in state["stages"]] == ["scf", "bands"]
    assert state["stages"][0]["external_checkpoint"]
    assert state["stages"][0]["parameters"]["namelists"]["electrons"]["startingpot"] == "file"


def test_continuation_rejects_stale_approval(accepted_qe, tmp_path):
    with pytest.raises(ValueError, match="changed after planning"):
        continuation.prepare_continuation(accepted_qe, 0, tmp_path / "new", None, ["scf"], expected_context_sha256="0" * 64)


def test_continuation_receipt_tampering_blocks_before_network(accepted_qe, tmp_path):
    target = tmp_path / "bands"
    continuation.prepare_continuation(accepted_qe, 0, target, None, ["bands"])
    (target / "source" / "checkpoint.json").write_text("{}")
    transport = FakeTransport()
    state = qe_workflow.advance(target, transport)
    assert state["status"] == "needs_attention"
    assert not transport.commands


def fail_scf(root):
    state = qe_workflow.read_state(root)
    state["status"] = state["stages"][0]["status"] = "queued"
    qe_workflow._write(root / "run.json", state)
    transport = FakeTransport()
    complete(root, transport)
    transport.files["pw.out"] = b"convergence NOT achieved after 100 iterations\nJOB DONE.\n"
    transport.files["data-file-schema.xml"] = (root / "01_scf/outputs/data-file-schema.xml").read_bytes()
    failed = qe_workflow.advance(root, transport)
    assert failed["status"] == "needs_attention"
    return failed


def test_bounded_repair_uses_actual_failure_and_separate_run(accepted_qe, tmp_path):
    fail_scf(accepted_qe)
    before = {str(path): qe_workflow._digest(path) for path in accepted_qe.rglob("*") if path.is_file()}
    options = continuation.repair_options(accepted_qe)
    assert {entry["action"] for entry in options["actions"]} == {"increase_electronic_steps", "reduce_mixing"}
    target = tmp_path / "repair"
    state = continuation.prepare_repair(accepted_qe, target, "increase_electronic_steps", options["snapshot_sha256"])
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    assert state["stages"][0]["parameters"]["namelists"]["electrons"]["electron_maxstep"] == 200
    assert json.loads((target / "repair.json").read_text())["attempt"] == 1
    assert before == {str(path): qe_workflow._digest(path) for path in accepted_qe.rglob("*") if path.is_file()}


def test_repair_budget_and_stale_evidence(accepted_qe, tmp_path):
    fail_scf(accepted_qe)
    options = continuation.repair_options(accepted_qe)
    with pytest.raises(ValueError, match="changed"):
        continuation.prepare_repair(accepted_qe, tmp_path / "stale", "reduce_mixing", "0" * 64)
    qe_workflow._write(accepted_qe / "repair.json", {"attempt": 2})
    options = continuation.repair_options(accepted_qe)
    assert options["attempt"] == 3 and not options["actions"]


def test_unconfirmed_failure_cannot_repair(accepted_qe):
    fail_scf(accepted_qe)
    state = qe_workflow.read_state(accepted_qe)
    state["last_error"] = "SSH lost"
    qe_workflow._write(accepted_qe / "run.json", state)
    with pytest.raises(ValueError, match="Reconnect"):
        continuation.repair_options(accepted_qe)


def test_timeout_repair_rebuilds_requested_phonon_chain(accepted_qe, tmp_path, monkeypatch):
    inspected = continuation.inspect_continuation(accepted_qe, 0)
    source = tmp_path / "phonon"
    original = qe_workflow.prepare_plan(inspected["structure_path"], source, inspected["config"], ["phonon_bands"],
        {"qmesh": [2, 2, 2], "qpoints": [[0, 0, 0], [0.5, 0, 0]]})
    original.update(status="needs_attention", current_stage=2)
    failed = original["stages"][2]
    failed.update(status="needs_attention", scheduler_state="TIMEOUT")
    qe_workflow._write(source / "run.json", original)
    before = {str(path): qe_workflow._digest(path) for path in source.rglob("*") if path.is_file()}
    # Exercise preparation after failure evidence has been independently checked.
    monkeypatch.setattr(continuation, "_failed_snapshot", lambda run: (source, original, failed, 2, "time limit", 1, "snapshot"))
    calls = []
    def continued(run, seed, target, config, tasks, parameters=None, stage_parameters=None):
        calls.append(tasks)
        return qe_workflow.prepare_plan(inspected["structure_path"], target, config, tasks, parameters, stage_parameters=stage_parameters)
    monkeypatch.setattr(continuation, "prepare_continuation", continued)
    repaired = continuation.prepare_repair(source, tmp_path / "retry", "increase_walltime", "snapshot")
    assert calls == [["phonon_bands"]]
    assert [stage["name"] for stage in repaired["stages"]] == ["scf", "ph", "q2r", "matdyn"]
    assert not any(stage["job_id"] for stage in repaired["stages"])
    assert before == {str(path): qe_workflow._digest(path) for path in source.rglob("*") if path.is_file()}


def test_eigenvalue_repair_keeps_thresholds_and_reuses_accepted_scf(accepted_qe, tmp_path):
    inspected = continuation.inspect_continuation(accepted_qe, 0)
    root = tmp_path / "bands-failure"
    qe_workflow.prepare_plan(inspected["structure_path"], root, inspected["config"], ["scf", "bands"],
                            {"mesh": [6, 6, 6], "ecutwfc": 70, "conv_thr": 1e-10,
                             "namelists": {"electrons": {"diago_thr_init": 1e-12}}})
    transport = FakeTransport()
    qe_workflow.advance(root, transport)
    complete(root, transport)
    fixture = tmp_path / "scf-fixture"
    fixture.mkdir()
    xml_output(fixture, root / "01_scf/inputs/structure.cif")
    for name in ("data-file-schema.xml", "pw.out"):
        transport.files[name] = (fixture / name).read_bytes()
    transport.files["checkpoint.json"] = json.dumps({
        "scratch/dft_agent.save/data-file-schema.xml": {"size": (fixture / "data-file-schema.xml").stat().st_size, "sha256": qe_workflow._digest(fixture / "data-file-schema.xml")},
        "scratch/dft_agent.save/charge-density.dat": {"size": 4, "sha256": "0" * 64},
        "scratch/dft_agent.save/wfc1.dat": {"size": 4, "sha256": "0" * 64},
    }).encode()
    assert qe_workflow.advance(root, transport)["stages"][0]["status"] == "succeeded"
    qe_workflow.advance(root, transport)
    complete(root, transport)
    xml_output(fixture, root / "02_bands/inputs/structure.cif", calculation="bands", scf=True)
    transport.files["data-file-schema.xml"] = (fixture / "data-file-schema.xml").read_bytes()
    transport.files["pw.out"] = ("c_bands: 1 eigenvalues not converged\n" + "0.0 " * 20000
                                  + "\nEnd of band structure calculation\nJOB DONE.\n").encode()
    assert qe_workflow.advance(root, transport)["status"] == "needs_attention"
    before = {str(path): qe_workflow._digest(path) for path in root.rglob("*") if path.is_file()}
    options = continuation.repair_options(root)
    assert [entry["action"] for entry in options["actions"]] == ["use_cg_diagonalization"]
    target = tmp_path / "bands-repair"
    state = continuation.prepare_repair(root, target, "use_cg_diagonalization", options["snapshot_sha256"])
    assert [stage["name"] for stage in state["stages"]] == ["bands"]
    assert state["stages"][0]["external_checkpoint"]
    assert not any(stage["job_id"] for stage in state["stages"])
    metadata = state["stages"][0]["metadata"]
    assert metadata["parameters"]["mesh"] == [6, 6, 6]
    assert metadata["namelists"]["system"]["ecutwfc"] == 70
    assert metadata["namelists"]["electrons"] == {
        "conv_thr": 1e-10, "electron_maxstep": 120, "diago_thr_init": 1e-12,
        "diagonalization": "cg", "diago_cg_maxiter": 200, "diago_full_acc": True, "startingpot": "file"}
    assert json.loads((target / "repair.json").read_text())["attempt"] == 1
    assert before == {str(path): qe_workflow._digest(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", ["scf", "bands"])
def test_singular_davidson_overlap_offers_cg(accepted_qe, monkeypatch, name):
    state = qe_workflow.read_state(accepted_qe)
    failed = {"name": name, "scheduler_state": "FAILED", "parameters": {}}
    log = "     Computing kpt #:     2  of   110\n     Error in routine cdiaghg (116):\n     S matrix not positive definite\n"
    monkeypatch.setattr(continuation, "_failed_snapshot", lambda run: (accepted_qe, state, failed, 0, log, 1, "snapshot"))
    actions = continuation.repair_options(accepted_qe)["actions"]
    assert [entry["action"] for entry in actions] == ["use_cg_diagonalization"]
    assert "singular overlap" in actions[0]["reason"]


def test_cg_repair_stops_at_iteration_cap(accepted_qe, monkeypatch):
    state = qe_workflow.read_state(accepted_qe)
    failed = {"name": "bands", "scheduler_state": "COMPLETED", "parameters": {
        "namelists": {"electrons": {"diagonalization": "cg", "diago_cg_maxiter": 1000, "diago_full_acc": True}}}}
    monkeypatch.setattr(continuation, "_failed_snapshot", lambda run: (
        accepted_qe, state, failed, 0, "c_bands: 1 eigenvalues not converged", 1, "snapshot"))
    assert not continuation.repair_options(accepted_qe)["actions"]
