"""Collection retries must not accept leftovers from an earlier download."""
import json

import pytest

from vasp_slurm_agent import qe_context, qe_workflow
from vasp_slurm_agent.transport import TransportError
from test_qe_context import accepted_qe
from test_qe_workflow import FakeTransport


def recollect(root):
    state = qe_workflow.read_state(root)
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    transport = FakeTransport()
    transport.files = {name: (output / name).read_bytes() for name in manifest}
    transport.queue, transport.accounting = "", "COMPLETED"
    state.update(status="collecting")
    stage.update(status="collecting", result={})
    qe_workflow._write(root / "run.json", state)
    return output, transport, stage["job_id"]


def snapshot(folder):
    return {path.relative_to(folder).as_posix(): qe_workflow._digest(path)
            for path in folder.rglob("*") if path.is_file()}


def test_recollection_does_not_use_stale_native_xml(accepted_qe):
    output, transport, job = recollect(accepted_qe)
    del transport.files["data-file-schema.xml"]
    state = qe_workflow.advance(accepted_qe, transport)
    assert state["status"] == "needs_attention"
    assert state["current_stage"] == 0
    assert not state["stages"][0]["result"]["success"]
    assert "data-file-schema.xml is missing" in state["stages"][0]["result"]["reason"]
    assert not (output / "data-file-schema.xml").exists()
    assert (output / "pw.out").read_bytes() == transport.files["pw.out"]
    assert not qe_context.load_run_context(accepted_qe)["facts"]["stage_1.accepted"]["value"]
    assert state["stages"][0]["job_id"] == job and not transport.uploads


@pytest.mark.parametrize("runtime", [b'{"changed": true}', None])
def test_collection_verifies_actual_runtime_bytes(accepted_qe, runtime):
    output, transport, _ = recollect(accepted_qe)
    if runtime is None:
        del transport.files["runtime.json"]
    else:
        transport.files["runtime.json"] = runtime
    state = qe_workflow.advance(accepted_qe, transport)
    result = state["stages"][0]["result"]
    assert state["status"] == "needs_attention"
    assert not result["success"] and not result["execution_verified"]
    assert "receipt is missing or inconsistent" in result["reason"]
    assert (output / "pw.out").is_file()


@pytest.mark.parametrize("changed", ["size", "sha256", "missing", "bytes"])
def test_invalid_transfer_receipt_preserves_previous_outputs(accepted_qe, monkeypatch, changed):
    output, transport, _ = recollect(accepted_qe)
    before = snapshot(output)
    original = transport.download_many

    def download(*args, **kwargs):
        records = original(*args, **kwargs)
        if changed == "missing":
            del records["pw.out"]
        elif changed == "bytes":
            (args[1] / "pw.out").write_bytes(b"changed after transfer")
        else:
            records["pw.out"][changed] = 0 if changed == "size" else "0" * 64
        return records

    monkeypatch.setattr(transport, "download_many", download)
    state = qe_workflow.advance(accepted_qe, transport)
    assert state["status"] == "needs_attention"
    assert "collection receipt" in state["last_error"]
    assert snapshot(output) == before


def test_interrupted_transfer_retains_evidence_and_resumes_same_job(accepted_qe, monkeypatch):
    output, transport, job = recollect(accepted_qe)
    before = snapshot(output)
    original = transport.download_many

    def interrupted(remote, folder, names, **kwargs):
        (folder / "pw.out").write_bytes(b"partial download")
        raise TransportError("Connection lost during collection")

    monkeypatch.setattr(transport, "download_many", interrupted)
    for _ in range(qe_workflow.MAX_REMOTE_FAILURES):
        state = qe_workflow.advance(accepted_qe, transport)
        assert snapshot(output) == before
    assert state["status"] == "needs_attention"
    monkeypatch.setattr(transport, "download_many", original)
    qe_workflow.resume(accepted_qe, transport)
    state = qe_workflow.advance(accepted_qe, transport)
    assert state["status"] == "succeeded"
    assert qe_context.load_run_context(accepted_qe)["facts"]["stage_1.accepted"]["value"]
    assert state["stages"][0]["job_id"] == job
    assert not transport.uploads and not transport.jobs
    assert not list(output.parent.glob(".qe-collect-*"))


def test_native_rejection_retains_new_failure_evidence(accepted_qe):
    output, transport, _ = recollect(accepted_qe)
    transport.files["data-file-schema.xml"] = transport.files["data-file-schema.xml"].replace(
        b"<scf_conv><convergence_achieved>true", b"<scf_conv><convergence_achieved>false")
    state = qe_workflow.advance(accepted_qe, transport)
    assert state["status"] == "needs_attention"
    assert "Electronic convergence" in state["stages"][0]["result"]["reason"]
    assert (output / "data-file-schema.xml").read_bytes() == transport.files["data-file-schema.xml"]
    assert (output / "pw.out").read_bytes() == transport.files["pw.out"]
    assert (output / "artifact_manifest.json").is_file() and (output / "qe-result.json").is_file()
