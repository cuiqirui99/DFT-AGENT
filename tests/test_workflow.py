from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result
from vasp_slurm_agent import workflow


class FakeTransport:
    """Model scheduler and file behavior; never connects to a real server."""

    def __init__(self):
        self.commands = []
        self.uploads = []
        self.files = {}
        self.remote_jobs = {}
        self.job_id = "80721"
        self.queue = "PENDING"
        self.accounting = ""
        self.exit_code = "0:0"
        self.lose_submit_reply = False
        self.cancel_count = 0
        self.cancel_error = False
        self.unavailable = False

    def run(self, command, timeout=60):
        self.commands.append(command)
        if self.unavailable:
            return Result(255, "", "Connection unavailable")
        if any(script in command for script in ("dispatch.py", "reconcile.py")) and command.startswith("python3 "):
            name = shlex.split(command)[3]
            if "--scheduler pbs" in command:
                self.job_id = "80721.pbs.example"
            if "--reconcile-only" in command and name not in self.remote_jobs:
                return Result(1, "", "No recorded submission; reconciliation cannot submit")
            self.remote_jobs.setdefault(name, self.job_id)
            if self.lose_submit_reply:
                self.lose_submit_reply = False
                return Result(255, "", "Connection lost after remote submission")
            return Result(0, json.dumps({"job_id": self.remote_jobs[name], "job_name": name}))
        if command.startswith("qstat "):
            phase = "R" if self.queue == "RUNNING" else "Q" if self.queue else "F"
            data = {"job_state": phase, "Exit_status": int(self.exit_code.split(":")[0])}
            if self.accounting == "CANCELLED": data["substate"] = 91
            return Result(0, json.dumps({"Jobs": {self.job_id: data}}))
        if command.startswith("squeue "):
            return Result(0, self.queue + "\n" if self.queue else "")
        if command.startswith("sacct "):
            return Result(0, f"{self.job_id}|{self.accounting}|{self.exit_code}|\n" if self.accounting else "")
        if command.startswith(("scancel ", "qdel ")):
            self.cancel_count += 1
            if self.cancel_error:
                return Result(1, "", "Invalid job id specified")
            return Result(0)
        if command.startswith("python3 -c "):
            return Result(0, json.dumps(list(self.files)))
        if command.startswith("command -v ") or command.startswith("mkdir -p "):
            return Result(0)
        pytest.fail(f"Unexpected remote command: {command}")

    def upload(self, source, destination):
        self.uploads.append((Path(source).name, destination))
        data = Path(source).read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    def download(self, source, destination):
        data = self.files[source.rsplit("/", 1)[-1]]
        Path(destination).write_bytes(data)
        return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


@pytest.mark.parametrize("accounting,expected", [("COMPLETED", "COMPLETED"), ("", "UNKNOWN")])
def test_completed_job_removed_from_squeue_uses_accounting(accounting, expected):
    class ExpiredQueue(FakeTransport):
        def run(self, command, timeout=60):
            if command.startswith("squeue "):
                return Result(1, "", "slurm_load_jobs error: Invalid job id specified")
            return super().run(command, timeout)

    transport = ExpiredQueue()
    transport.accounting = accounting
    assert workflow._poll(transport, {"job_id": transport.job_id}) == expected


def test_queue_connection_error_is_not_a_completed_job():
    transport = FakeTransport()
    transport.unavailable = True
    with pytest.raises(workflow.TransportError, match="Connection unavailable"):
        workflow._poll(transport, {"job_id": transport.job_id})


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    config = ClusterConfig(
        host="cluster.example.invalid", user="researcher", remote_root="/scratch/test",
        vasp_command="mpiexec vasp_std", potcar_root="/licensed/pbe", partition="cpu", scheduler=getattr(request, "param", "slurm"),
    )
    structure = tmp_path / "source.cif"
    structure.write_text("test structure for control semantics")

    def prepare(source, output, task, parameters, potcar_symbols):
        output.mkdir(parents=True)
        for name in ("POSCAR", "INCAR", "KPOINTS"):
            (output / name).write_text(f"test input {task}: {name}\n")
        metadata = {"potcar_labels": ["Si"], "formula": "Si2", "input_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("POSCAR", "INCAR", "KPOINTS")
        }}
        (output / "metadata.json").write_text(json.dumps(metadata))
        return metadata

    def no_network(*args, **kwargs):
        pytest.fail("Unexpected construction of real SSH transport")

    monkeypatch.setattr(workflow, "prepare_inputs", prepare)
    monkeypatch.setattr(workflow, "SSHTransport", no_network)
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: {"success": True, "reason": "Fake parser accepted outputs"})
    root = tmp_path / "run"
    workflow.prepare_run(structure, root, config)
    return root


@pytest.mark.parametrize("status", ["planned", "submitting"])
def test_poll_only_cannot_launch_a_stage(prepared, status):
    state = workflow.read_state(prepared)
    state["status"] = state["stages"][0]["status"] = status
    workflow._write(prepared / "run.json", state)
    # The fixture fails if an SSH transport is constructed.
    assert workflow.advance(prepared, poll_only=True) == state


def completed_files(root, transport, bad_hash=False):
    state = workflow.read_state(root)
    inputs = root / state["stages"][state["current_stage"]]["folder"] / "inputs"
    transport.files = {name: (inputs / name).read_bytes() for name in ("INCAR", "KPOINTS", "POSCAR")}
    lines = []
    for name in ("INCAR", "KPOINTS", "POSCAR"):
        digest = hashlib.sha256(transport.files[name]).hexdigest()
        lines.append(f"{'0' * 64 if bad_hash and name == 'INCAR' else digest}  {name}\n")
    transport.files["input_hashes.sha256"] = "".join(lines).encode()
    transport.files["CONTCAR"] = b"test output structure"
    transport.queue = ""
    transport.accounting = "COMPLETED"


def test_prepare_is_offline_and_awaits_explicit_submission(prepared):
    state = workflow.read_state(prepared)
    assert state["status"] == "planned"
    assert state["stages"][0]["job_id"] is None
    assert not state["stages"][0]["staged"]


def test_changed_approved_input_refuses_before_remote_submission(prepared):
    (prepared / "01_relax" / "inputs" / "INCAR").write_text("ENCUT = 900\n")
    transport = FakeTransport()
    state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    assert "changed" in state["last_error"]
    assert not transport.commands


def test_changed_cluster_snapshot_never_targets_different_host(prepared):
    config = json.loads((prepared / "config.json").read_text())
    config["host"] = "another.example.invalid"
    (prepared / "config.json").write_text(json.dumps(config))
    transport = FakeTransport()
    state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    assert not transport.commands


def test_lost_submission_reply_recovers_same_remote_job(prepared):
    transport = FakeTransport()
    transport.lose_submit_reply = True
    first = workflow.advance(prepared, transport)
    assert first["status"] == "submitting"
    assert first["stages"][0]["staged"]
    assert first["stages"][0]["job_id"] is None
    assert len(transport.remote_jobs) == 1
    first_upload_count = len(transport.uploads)
    # Reading state again simulates a restarted client with no in-memory job ID.
    second = workflow.advance(prepared, transport)
    assert second["status"] == "queued"
    assert second["stages"][0]["job_id"] == transport.job_id
    assert len(transport.remote_jobs) == 1
    assert len(transport.uploads) == first_upload_count
    assert second["run_id"] == first["run_id"]


def test_accounting_lag_preserves_identity_and_last_known_state(prepared):
    transport = FakeTransport()
    before = workflow.advance(prepared, transport)
    transport.queue = ""
    transport.accounting = ""
    after = workflow.advance(prepared, transport)
    assert after["stages"][0]["job_id"] == before["stages"][0]["job_id"]
    assert after["status"] == before["status"]
    assert after["stages"][0]["status"] == before["stages"][0]["status"]
    assert "no resubmission" in after["last_error"]
    assert len(transport.remote_jobs) == 1


def test_connection_failure_keeps_job_and_can_resume(prepared):
    transport = FakeTransport()
    before = workflow.advance(prepared, transport)
    transport.unavailable = True
    failed_poll = workflow.advance(prepared, transport)
    assert failed_poll["status"] == "queued"
    assert failed_poll["stages"][0]["job_id"] == before["stages"][0]["job_id"]
    transport.unavailable = False
    transport.queue = "RUNNING"
    resumed = workflow.advance(prepared, transport)
    assert resumed["status"] == "running"
    assert resumed["last_error"] is None
    assert len(transport.remote_jobs) == 1


def test_repeated_connection_failures_pause_and_resume_original_job(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    transport.unavailable = True
    for _ in range(workflow.MAX_REMOTE_FAILURES):
        stopped = workflow.advance(prepared, transport)
    assert stopped["status"] == "needs_attention"
    transport.unavailable = False
    restored = workflow.resume(prepared, transport)
    assert restored["status"] == "queued"
    assert restored["stages"][0]["job_id"] == transport.job_id
    assert restored["remote_failures"] == 0
    assert len(transport.remote_jobs) == 1


def test_lost_acknowledgement_can_reconcile_then_cancel(prepared):
    transport = FakeTransport()
    transport.lose_submit_reply = True
    workflow.advance(prepared, transport)
    assert workflow.cancel(prepared, transport)["status"] == "needs_attention"
    restored = workflow.resume(prepared, transport)
    assert restored["stages"][0]["job_id"] == transport.job_id
    assert restored["cancel_requested"]
    workflow.advance(prepared, transport)
    assert transport.cancel_count == 1
    transport.queue = ""
    transport.accounting = "CANCELLED"
    assert workflow.advance(prepared, transport)["status"] == "cancelled"
    assert len(transport.remote_jobs) == 1
    assert any("--reconcile-only" in command for command in transport.commands)


def test_resume_staged_run_without_receipt_never_blindly_submits(prepared):
    state = workflow.read_state(prepared)
    state["status"] = "needs_attention"
    state["stages"][0].update(status="submitting", staged=True)
    workflow._save(prepared, state)
    transport = FakeTransport()
    after = workflow.resume(prepared, transport)
    assert after["status"] == "needs_attention"
    assert not transport.remote_jobs


def test_cancel_of_paused_known_job_is_not_silently_ignored(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    transport.unavailable = True
    for _ in range(workflow.MAX_REMOTE_FAILURES):
        state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    transport.unavailable = False
    cancelled = workflow.cancel(prepared, transport)
    assert cancelled["cancel_requested"]
    assert transport.cancel_count == 1
    assert cancelled["stages"][0]["job_id"] == transport.job_id


def test_repeated_cancellation_rejection_preserves_job_and_stops_retrying(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    transport.cancel_error = True
    workflow.cancel(prepared, transport)
    for _ in range(workflow.MAX_REMOTE_FAILURES - 1):
        state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    assert state["cancel_requested"]
    assert state["stages"][0]["job_id"] == transport.job_id
    before = transport.cancel_count
    workflow.advance(prepared, transport)
    assert transport.cancel_count == before


@pytest.mark.parametrize("scheduler,exit_code", [("FAILED", "1:0"), ("TIMEOUT", "0:15"), ("COMPLETED", "1:0")])
def test_slurm_failure_cannot_be_overridden_by_parser_success(prepared, scheduler, exit_code):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport)
    transport.accounting = scheduler
    transport.exit_code = exit_code
    state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    assert not state["stages"][0]["result"]["success"]
    assert "Scheduler ended" in state["stages"][0]["result"]["reason"]


def test_input_checksum_mismatch_cannot_be_success(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport, bad_hash=True)
    state = workflow.advance(prepared, transport)
    assert state["status"] == "needs_attention"
    assert not state["stages"][0]["result"]["success"]
    assert not state["stages"][0]["result"]["input_identity_verified"]


def test_complete_collection_records_hashes_and_result(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport)
    state = workflow.advance(prepared, transport)
    assert state["status"] == "succeeded"
    output = prepared / "01_relax" / "outputs"
    assert json.loads((output / "result.json").read_text())["input_identity_verified"]
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, data in transport.files.items():
        assert manifest[name] == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    assert (output / "metadata.json").exists()


def test_cancel_requires_scheduler_confirmation(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    transport.queue = "RUNNING"
    requested = workflow.cancel(prepared, transport)
    assert requested["cancel_requested"]
    assert requested["status"] != "cancelled"
    assert transport.cancel_count == 1
    transport.queue = ""
    transport.accounting = "CANCELLED"
    confirmed = workflow.advance(prepared, transport)
    assert confirmed["status"] == "cancelled"
    assert confirmed["stages"][0]["job_id"] == transport.job_id


def test_already_gone_job_can_confirm_cancellation_even_if_scancel_fails(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    transport.queue = "RUNNING"
    workflow.cancel(prepared, transport)
    transport.cancel_error = True
    transport.queue = ""
    transport.accounting = "CANCELLED"
    state = workflow.advance(prepared, transport)
    assert state["status"] == "cancelled"
    assert len(transport.remote_jobs) == 1


def test_cancelling_unsubmitted_run_never_constructs_transport(prepared):
    state = workflow.cancel(prepared)
    assert state["status"] == "cancelled"
    assert state["stages"][0]["job_id"] is None


def test_second_worker_exits_without_touching_remote(prepared):
    with (prepared / ".worker.lock").open("a") as lock:
        workflow._acquire(lock, blocking=False)
        state = workflow.watch(prepared, interval=0)
    assert state["status"] == "planned"
    assert not (prepared / "worker.json").exists()


def test_zhegv_diagnosis_keeps_parser_reason_and_never_resubmits(prepared, monkeypatch):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    before = workflow.read_state(prepared)
    upload_count = len(transport.uploads)
    completed_files(prepared, transport)
    transport.accounting = "FAILED"
    transport.exit_code = "1:0"
    transport.files["slurm.out"] = b"Error EDDDAV: Call to ZHEGV failed. Returncode = 1 2 128\n"
    parser_reason = "ParseError: no element found: line 392, column 0"
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: {"success": False, "reason": parser_reason})
    state = workflow.advance(prepared, transport)
    result = state["stages"][0]["result"]
    assert result["failure_code"] == "VASP_EDDDAV_ZHEGV"
    assert result["parser_reason"] == parser_reason
    assert parser_reason in result["reason"]
    assert "ZHEGV" in result["reason"] and "Scheduler ended in FAILED" in result["reason"]
    assert "fewer MPI tasks" in result["recovery_hint"]
    assert "does not establish the cause" in result["recovery_hint"]
    assert state["status"] == "needs_attention"
    assert state["parameters"] == before["parameters"]
    assert state["stages"][0]["metadata"] == before["stages"][0]["metadata"]
    assert len(transport.uploads) == upload_count
    assert len(transport.remote_jobs) == 1
    assert workflow.advance(prepared, transport)["status"] == "needs_attention"
    assert len(transport.remote_jobs) == 1


@pytest.mark.parametrize("scheduler,log,code", [
    ("OUT_OF_MEMORY", b"", "OUT_OF_MEMORY"),
    ("FAILED", b"slurmstepd: error: Detected 1 oom-kill event in StepId=123.batch cgroup", "OUT_OF_MEMORY"),
    ("TIMEOUT", b"", "TIME_LIMIT"),
    ("FAILED", b"JOB 123 CANCELLED DUE TO TIME LIMIT", "TIME_LIMIT"),
])
def test_resource_failure_diagnostics_preserve_parser_reason(prepared, monkeypatch, scheduler, log, code):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport)
    transport.accounting = scheduler
    transport.exit_code = "1:0"
    transport.files["slurm.err"] = log
    parser_reason = "ValueError: Complete vasprun.xml is missing; scheduler completion is insufficient."
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: {"success": False, "reason": parser_reason})
    state = workflow.advance(prepared, transport)
    result = state["stages"][0]["result"]
    assert result["failure_code"] == code
    assert result["parser_reason"] == parser_reason
    assert parser_reason in result["reason"]
    assert result["recovery_hint"]
    assert not result["success"]


def test_unrecognized_failed_job_does_not_erase_original_parser_error(prepared, monkeypatch):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport)
    transport.accounting = "FAILED"
    transport.exit_code = "1:0"
    parser_reason = "ValueError: Final structure atom count differs from the input."
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: {"success": False, "reason": parser_reason})
    result = workflow.advance(prepared, transport)["stages"][0]["result"]
    assert result["parser_reason"] == parser_reason
    assert parser_reason in result["reason"]
    assert "Scheduler ended in FAILED" in result["reason"]


@pytest.mark.parametrize("prepared", ["pbs"], indirect=True)
def test_pbs_lost_ack_reconnect_and_cancel(prepared):
    transport = FakeTransport()
    transport.lose_submit_reply = True
    assert workflow.advance(prepared, transport)["stages"][0]["job_id"] is None
    state = workflow.advance(prepared, transport)
    assert state["stages"][0]["job_id"] == "80721.pbs.example"
    assert len(transport.remote_jobs) == 1
    transport.queue = "RUNNING"
    assert workflow.cancel(prepared, transport)["status"] != "cancelled"
    assert transport.cancel_count == 1
    transport.queue, transport.accounting = "", "CANCELLED"
    assert workflow.advance(prepared, transport)["status"] == "cancelled"
    assert any(command.startswith("qdel 80721.pbs.example") for command in transport.commands)


@pytest.mark.parametrize("prepared", ["pbs"], indirect=True)
def test_pbs_mock_output_still_requires_collection_and_input_identity(prepared):
    transport = FakeTransport()
    workflow.advance(prepared, transport)
    completed_files(prepared, transport, bad_hash=True)
    result = workflow.advance(prepared, transport)
    assert result["status"] == "needs_attention"
    assert result["stages"][0]["result"]["input_identity_verified"] is False
