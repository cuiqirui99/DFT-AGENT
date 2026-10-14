"""Frozen-source repairs reuse evidence, not completed source calculations."""
import json
import shutil

import pytest

from vasp_slurm_agent import qe_continuation as continuation, qe_workflow, qe_context
from vasp_slurm_agent.config import ClusterConfig
from test_qe_aniso_continuation import accepted_interpolation, aniso
from test_qe_workflow import FakeTransport, complete


@pytest.fixture
def frozen_timeout(tmp_path, monkeypatch):
    real_context = qe_context.load_run_context
    source, _, config = accepted_interpolation.__wrapped__(tmp_path, monkeypatch)

    def prepare(task, tasks=None):
        target = tmp_path / task
        selected = tasks or [task]
        recipes = [{**aniso(), **({"broadening_index": 1} if name != "epw_aniso" else {})}
                   for name in selected]
        state = continuation.prepare_continuation(source, 3, target, config, selected,
                                                 stage_parameters=recipes)
        # The source's acceptance gate is synthetic in this fixture. Failed
        # collection, input receipts, repair validation and rebuilding are real.
        monkeypatch.setattr(qe_context, "load_run_context", real_context)
        stage = state["stages"][0]
        inputs = target / stage["folder"] / "inputs"
        qe_workflow._write(inputs / "runtime.json", {"input_sha256": stage["metadata"]["input_sha256"]})
        qe_workflow._write(inputs / "pseudo_manifest.json", {})
        stage.update(job_id="8001", status="queued", staged=True,
                     runtime_sha256=qe_workflow._digest(inputs / "runtime.json"),
                     pseudo_manifest_sha256=qe_workflow._digest(inputs / "pseudo_manifest.json"))
        state.update(status="queued")
        qe_workflow._write(target / "run.json", state)
        transport = FakeTransport()
        complete(target, transport)
        transport.accounting = "TIMEOUT"
        failed = qe_workflow.advance(target, transport)
        assert failed["status"] == "needs_attention" and failed["stages"][0]["result"]["input_identity_verified"]
        return source, target, config

    return prepare


@pytest.mark.parametrize("task", ["epw_iso", "epw_tc", "epw_aniso"])
def test_timeout_repair_preserves_exact_frozen_epw_inputs(frozen_timeout, tmp_path, task):
    source, failed, config = frozen_timeout(task)
    before = {str(path.relative_to(failed)): qe_workflow._digest(path)
              for path in failed.rglob("*") if path.is_file()}
    old = qe_workflow.read_state(failed)
    old_plan = json.loads((failed / "plan.json").read_text())
    shutil.rmtree(source)  # Only the frozen continuation must be available locally.
    options = continuation.repair_options(failed)
    assert [item["action"] for item in options["actions"]] == ["increase_walltime"]
    target = tmp_path / "retry"
    repaired = continuation.prepare_repair(failed, target, "increase_walltime", options["snapshot_sha256"])
    assert [stage["name"] for stage in repaired["stages"]] == [task]
    assert repaired["status"] == "planned" and repaired["current_stage"] == 0
    assert repaired["run_id"] != old["run_id"] and repaired["remote_root"] != old["remote_root"]
    stage, original = repaired["stages"][0], old["stages"][0]
    assert stage["parameters"] == original["parameters"]
    assert stage["metadata"]["input_sha256"] == original["metadata"]["input_sha256"]
    assert stage["metadata"]["source_provenance"] == original["metadata"]["source_provenance"]
    assert stage.get("external_checkpoint") == original.get("external_checkpoint")
    assert not stage["job_id"] and not stage["staged"] and stage["materialized"]
    for name, digest in old_plan["sources"].items():
        assert qe_workflow._digest(target / name) == digest
    new_config = ClusterConfig.load(target / "config.json").to_dict()
    original_config = config.to_dict()
    assert new_config.pop("walltime") == "01:00:00"
    original_config.pop("walltime")
    assert new_config == original_config
    assert before == {str(path.relative_to(failed)): qe_workflow._digest(path)
                      for path in failed.rglob("*") if path.is_file()}
    qe_workflow._check_config(target, repaired)


def test_timeout_repair_keeps_mixed_scalar_and_anisotropic_stages(frozen_timeout, tmp_path):
    _, failed, _ = frozen_timeout("epw_tc", ["epw_tc", "epw_aniso"])
    options = continuation.repair_options(failed)
    state = continuation.prepare_repair(failed, tmp_path / "retry", "increase_walltime", options["snapshot_sha256"])
    assert [stage["name"] for stage in state["stages"]] == ["epw_tc", "epw_aniso"]
    assert "external_checkpoint" not in state["stages"][0]
    assert state["stages"][1]["external_checkpoint"]
    assert all(stage["restart_from"] is None and stage["structure_from"] is None for stage in state["stages"])


@pytest.mark.parametrize("member", ["dft_agent.a2f", "scratch/dft_agent.ephmat/ephmat1", "checkpoint"])
def test_changed_frozen_source_or_checkpoint_rejects_repair(frozen_timeout, tmp_path, member):
    _, failed, _ = frozen_timeout("epw_aniso")
    options = continuation.repair_options(failed)
    path = failed / ("source/checkpoint.json" if member == "checkpoint" else "source/wannier/" + member)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen source"):
        continuation.prepare_repair(failed, tmp_path / "retry", "increase_walltime", options["snapshot_sha256"])
    assert not (tmp_path / "retry").exists()
