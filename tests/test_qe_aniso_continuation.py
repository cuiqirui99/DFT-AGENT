"""Continuation contracts; synthetic native streams are not material validation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from pymatgen.core import Structure

from vasp_slurm_agent import qe_continuation as continuation, qe_agent, qe_context, qe_epw, qe_reasoning, qe_wannier, qe_workflow
from vasp_slurm_agent.config import ClusterConfig
from test_qe import xml_output
from test_qe_agent import response
from test_qe_epw import checkpoint_fixture
from test_qe_wannier import settings


def aniso(**changes):
    return dict(temperatures=[10, 20], mu_star=.1, frequency_cutoff_ev=.4, **changes)


@pytest.fixture
def accepted_interpolation(tmp_path, monkeypatch):
    # The accepted-stage gate is isolated here. Real parser tests live in
    # test_qe_wannier and test_qe_epw; preparation and checkpoint I/O stay real.
    crystal = tmp_path / "POSCAR"
    Structure([[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]],
              ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=crystal, fmt="poscar")
    config = ClusterConfig(engine="qe", host="test.invalid", user="test", partition="cpu",
        remote_root="/scratch/jobs", qe_pseudo_root="/pseudo", qe_pseudopotentials={
            "Si": {"filename": "Si.UPF", "pseudo_type": "NC", "sha256": "a" * 64,
                   "functional": "PBE", "fully_relativistic": False}})
    recipe = settings()
    recipe["projections"] = ["Si:s", "Si:p"]
    root = tmp_path / "source-run"
    state = qe_workflow.prepare_plan(crystal, root, config, ["wannier_electron_phonon"], recipe)
    stage = state["stages"][3]
    output = root / stage["folder"] / "outputs"
    shutil.copytree(root / stage["folder"] / "inputs", output)
    xml_output(output, root / stage["source_path"])
    checkpoint_fixture(output)
    for name in ("epw.out", "dft_agent.wout", "dft_agent.win", "dft_agent.a2f"):
        (output / name).write_text("synthetic continuation fixture\n")
    native = qe_epw.read_checkpoint(output)
    checkpoint = {name: {"sha256": digest, "size": (output / name).stat().st_size}
                  for name, digest in native["sha256"].items()}
    checkpoint["dft_agent.a2f"] = {"sha256": qe_workflow._digest(output / "dft_agent.a2f"),
                                   "size": (output / "dft_agent.a2f").stat().st_size}
    qe_workflow._write(output / "checkpoint.json", checkpoint)
    qe_workflow._write(output / "pseudo_manifest.json", config.qe_pseudopotentials)
    stage.update(status="succeeded", result={"success": True,
        "checkpoint_sha256": qe_workflow._digest(output / "checkpoint.json")})
    state.update(status="succeeded", current_stage=3)
    qe_workflow._save(root, state)
    records = {str(path.relative_to(output)): {"sha256": qe_workflow._digest(path), "size": path.stat().st_size}
               for path in output.rglob("*") if path.is_file()}
    qe_workflow._write(output / "artifact_manifest.json", records)

    def context(path):
        assert Path(path).resolve() == root.resolve()
        payload = json.dumps({str(p.relative_to(root)): qe_workflow._digest(p)
                              for p in root.rglob("*") if p.is_file()}, sort_keys=True)
        return {"context_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "facts": {"stage_4.accepted": {"value": True}}, "artifacts": []}

    monkeypatch.setattr(qe_context, "load_run_context", context)
    monkeypatch.setattr(qe_wannier, "analyze_outputs", lambda *a, **kw: {"success": True})
    return root, output, config


def test_continue_only_anisotropic_and_freeze_every_native_member(accepted_interpolation, tmp_path):
    source, output, config = accepted_interpolation
    before = {str(p): qe_workflow._digest(p) for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "continued"
    state = continuation.prepare_continuation(source, 3, target, config, ["epw_aniso", "epw_aniso"],
        stage_parameters=[aniso(), {**aniso(), "temperatures": [30]}])
    assert [s["name"] for s in state["stages"]] == ["epw_aniso", "epw_aniso"]
    assert [s["request_index"] for s in state["stages"]] == [0, 1]
    assert state["current_stage"] == 0 and state["status"] == "planned"
    plan = json.loads((target / "plan.json").read_text())
    native = qe_epw.read_checkpoint(output)["sha256"]
    for name in {"qe-input.json", "epw.in", "epw.out", "data-file-schema.xml", "structure.cif",
                 "dft_agent.wout", "dft_agent.win", "dft_agent.a2f", *native}:
        frozen = "source/wannier/" + name
        assert (target / frozen).read_bytes() == (output / name).read_bytes()
        assert plan["sources"][frozen] == qe_workflow._digest(output / name)
    for stage in state["stages"]:
        assert stage["restart_from"] is stage["structure_from"] is None
        assert "evidence_from" not in stage and "auxiliary_from" not in stage
        assert stage["source_evidence"] == "source/wannier"
        assert stage["materialized"] and not stage["metadata"].get("preview_only")
        assert stage["metadata"]["checkpoint_sha256"] == native
        assert stage["job_id"] is None and not stage["staged"]
        text = (target / stage["folder"] / "inputs/epw.in").read_text()
        assert "laniso = .true." in text and "elph = .false." in text
    proof = json.loads((target / "source/checkpoint.json").read_text())
    assert proof["remote_directory"].endswith("/04_epw_interpolate")
    assert proof["cluster"]["host"] == config.host
    assert not any(p.name.startswith(("01_", "02_", "03_", "04_")) for p in target.iterdir())
    assert before == {str(p): qe_workflow._digest(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [("host", "other.invalid"), ("qe_bin_dir", "/new/qe/bin")])
def test_continuation_keeps_original_cluster_and_binary(accepted_interpolation, tmp_path, field, value):
    source, _, config = accepted_interpolation
    with pytest.raises(ValueError, match="original cluster"):
        continuation.prepare_continuation(source, 3, tmp_path / "changed", replace(config, **{field: value}),
                                          ["epw_aniso"], aniso())


@pytest.mark.parametrize("task", ["epw_iso", "epw_tc", "scf"])
def test_wannier_source_cannot_be_used_as_scalar_spectrum(accepted_interpolation, tmp_path, task):
    source, _, config = accepted_interpolation
    with pytest.raises(ValueError, match="with epw_aniso"):
        continuation.prepare_continuation(source, 3, tmp_path / "wrong-route", config, [task], aniso())


def test_remote_checkpoint_must_match_local_native_stream(accepted_interpolation):
    source, output, _ = accepted_interpolation
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    checkpoint["scratch/dft_agent.ephmat/ephmat1"]["sha256"] = "f" * 64
    qe_workflow._write(output / "checkpoint.json", checkpoint)
    state = qe_workflow.read_state(source)
    state["stages"][3]["result"]["checkpoint_sha256"] = qe_workflow._digest(output / "checkpoint.json")
    qe_workflow._save(source, state)
    with pytest.raises(ValueError, match="remote Wannier checkpoint differs"):
        continuation.inspect_continuation(source, 3)


def test_frozen_native_evidence_must_match_collected_manifest(accepted_interpolation, tmp_path):
    source, output, config = accepted_interpolation
    (output / "epw.out").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        continuation.prepare_continuation(source, 3, tmp_path / "changed", config, ["epw_aniso"], aniso())
    assert not (tmp_path / "changed").exists()


def test_new_temperature_needs_checkpoint_and_explicit_science(accepted_interpolation, tmp_path):
    source, _, config = accepted_interpolation
    with pytest.raises(ValueError, match="requires.*checkpoint"):
        continuation.prepare_continuation(source, 3, tmp_path / "no-restart", config, ["epw_aniso"], aniso(), reuse_checkpoint=False)
    with pytest.raises(ValueError, match="mu_star"):
        continuation.prepare_continuation(source, 3, tmp_path / "no-mu", config, ["epw_aniso"],
                                          {"temperatures": [20], "frequency_cutoff_ev": .4})


def continuation_decision(task, parameters):
    plan = response((task,), parameters)
    return {"continuation": {"source_stage": 3, "intent": plan["intent"], "stages": plan["stages"]}}


def test_reasoning_uses_explicit_anisotropic_recipe_without_wannier_settings():
    inspected = {"source_task": "epw_interpolate", "parameters": settings()}
    plan, recipes = qe_reasoning._continuation_plan(continuation_decision("epw_aniso", aniso()), inspected)
    assert recipes[0]["parameters"] == aniso()
    assert "mesh" not in recipes[0]["parameters"]
    with pytest.raises(ValueError, match="epw_aniso"):
        qe_reasoning._continuation_plan(continuation_decision("epw_iso", aniso()), inspected)
    inspected["source_task"] = "eph_lambda"
    inspected["parameters"]["mu_star"] = .1
    with pytest.raises(ValueError, match="epw_iso or epw_tc"):
        qe_reasoning._continuation_plan(continuation_decision("epw_aniso", aniso()), inspected)


@pytest.mark.parametrize("source,task", [("wannier_electron_phonon", "epw_aniso"),
                                        ("electron_phonon", "epw_iso"), ("electron_phonon", "epw_tc")])
def test_epw_requirements_follow_linked_source_without_adding_solver_parameters(source, task):
    plan = response((source, task))
    plan["stages"][0]["parameters_json"] = json.dumps({"functional": "LDA"})
    plan["stages"][1]["parameters_json"] = json.dumps(aniso())
    plan["stages"][1]["requirements"]["functional"] = "LDA"
    recipes = qe_agent._recipes(plan)
    assert recipes[1]["parameters"] == aniso()
    assert "functional" not in recipes[1]["parameters"]
    plan["stages"][1]["requirements"]["functional"] = "PBE"
    with pytest.raises(ValueError, match="functional.*differs"):
        qe_agent._recipes(plan)


@pytest.mark.parametrize("source,task", [("epw_interpolate", "epw_aniso"), ("eph_lambda", "epw_iso")])
def test_continuation_requirements_follow_frozen_source_method(source, task):
    decision = continuation_decision(task, aniso())
    decision["continuation"]["stages"][0]["requirements"]["functional"] = "LDA"
    inspected = {"source_task": source, "parameters": {"functional": "LDA", "mu_star": .1}}
    _, recipes = qe_reasoning._continuation_plan(decision, inspected)
    assert "functional" not in recipes[0]["parameters"]
    decision["continuation"]["stages"][0]["requirements"]["functional"] = "PBE"
    with pytest.raises(ValueError, match="functional.*differs"):
        qe_reasoning._continuation_plan(decision, inspected)


def test_frozen_evidence_change_blocks_before_network(accepted_interpolation, tmp_path):
    source, _, config = accepted_interpolation
    target = tmp_path / "continued"
    continuation.prepare_continuation(source, 3, target, config, ["epw_aniso"], aniso())
    (target / "source/wannier/epw.out").write_text("changed")
    class NoNetwork:
        def run(self, *a, **kw):
            pytest.fail("Changed evidence must not reach SSH")
    state = qe_workflow.advance(target, NoNetwork())
    assert state["status"] == "needs_attention"
    assert "source" in state["last_error"] and state["stages"][0]["job_id"] is None


def test_cli_prepares_wannier_chain_and_continues_native_source(accepted_interpolation, tmp_path, monkeypatch):
    import sys
    from vasp_slurm_agent import cli
    source, _, _ = accepted_interpolation
    monkeypatch.setattr(cli, "migrate_legacy_config", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dft-agent", "continue", str(source), "04_epw_interpolate",
        str(tmp_path / "cli-next"), "--tasks", "epw_aniso", "--parameters", json.dumps(aniso())])
    assert cli.main() == 0
    state = qe_workflow.read_state(tmp_path / "cli-next")
    assert [s["name"] for s in state["stages"]] == ["epw_aniso"]
    recipe = qe_workflow.read_state(source)["stages"][0]["parameters"]
    monkeypatch.setattr(sys, "argv", ["dft-agent", "prepare", str(source / "source/structure.vasp"),
        str(tmp_path / "cli-fresh"), "--task", "wannier_electron_phonon", "--config", str(source / "config.json"),
        "--parameters", json.dumps(recipe)])
    assert cli.main() == 0
    assert [s["name"] for s in qe_workflow.read_state(tmp_path / "cli-fresh")["stages"]] == [
        "wannier_scf", "wannier_ph", "wannier_nscf", "epw_interpolate"]
