from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from vasp_slurm_agent import qe_pairing_workflow as workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Transport:
    """Execute the frozen Python payload locally; simulate only SSH and Slurm."""
    def __init__(self, remote):
        self.remote = remote
        remote.mkdir()
        self.commands, self.jobs = [], {}
        self.queue, self.accounting = "PENDING", ""
        self.lose_reply = False
        self.cancel_count = 0

    def run(self, command, timeout=60):
        self.commands.append(command)
        words = shlex.split(command)
        if command.startswith("command -v "):
            return Result(0)
        if "PAIRING_RUNTIME " in command:
            # The transport simulates a Linux cluster using this test's Python.
            result = subprocess.run([sys.executable, "-c", words[-1]], text=True, capture_output=True, timeout=timeout)
            return Result(result.returncode, result.stdout, result.stderr)
        if words[:2] == ["python3", "-c"]:
            return Result(0, json.dumps([p.name for p in self.remote.iterdir() if p.is_file()]))
        if command.startswith("python3 ") and "/dispatch.py" in command:
            name = words[-2] if "--reconcile-only" in words else words[-1]
            if "--reconcile-only" in words and name not in self.jobs:
                return Result(1, "", "No recorded submission")
            self.jobs.setdefault(name, "8001")
            receipt = {"job_id": self.jobs[name], "job_name": name}
            (self.remote/"job.json").write_text(json.dumps(receipt))
            if self.lose_reply:
                self.lose_reply = False
                return Result(255, "", "outcome uncertain")
            return Result(0, json.dumps(receipt))
        if command.startswith("squeue "):
            return Result(0, self.queue)
        if command.startswith("sacct "):
            return Result(0, f"8001|{self.accounting}|0:0|" if self.accounting else "")
        if command.startswith("scancel "):
            self.cancel_count += 1
            return Result(0)
        pytest.fail(command)

    def upload_many(self, sources, remote, *, permitted_names):
        for source in sources:
            assert source.name in permitted_names
            (self.remote/source.name).write_bytes(source.read_bytes())
        return {}

    def download_many(self, remote, output, names, *, permitted_names):
        manifest = {}
        for name in names:
            assert name in permitted_names
            (output/name).write_bytes((self.remote/name).read_bytes())
            manifest[name] = {"size": (output/name).stat().st_size, "sha256": digest(output/name)}
        return manifest

    def execute(self):
        name = next(iter(self.jobs))
        env = dict(os.environ, SLURM_JOB_ID="8001", SLURM_JOB_NAME=name,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "execute.py"], cwd=self.remote, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        self.queue, self.accounting = "", "COMPLETED"


@pytest.fixture
def prepared(tmp_path):
    hr = tmp_path/"model_hr.dat"
    hr.write_text("Example model\n1\n3\n1 1 1\n0 0 0 1 1 0 0\n1 0 0 1 1 -1 0\n-1 0 0 1 1 -1 0\n")
    parameters = {"lattice_angstrom": [[1,0,0],[0,1,0],[0,0,1]], "orbital": "s",
                  "spin_convention": "spin_degenerate_no_soc", "u_ev": 1., "mu_ev": 0.,
                  "temperature_k": 11604.518121550322, "kmesh": [8,1,1], "matsubara_points": 32,
                  "channels": ["singlet", "triplet"], "double_counting": "Model only"}
    config = ClusterConfig(engine="qe", host="cluster.invalid", user="user", remote_root="/scratch/test",
                           partition="cpu", tasks=1, qe_pseudo_root="/unused")
    root = tmp_path/"run"
    state = workflow.prepare_run(hr, root, config, parameters, python_executable="python3", origin="model")
    transport = Transport(tmp_path/"remote")
    return root, transport, state


def complete(prepared):
    root, transport, _ = prepared
    assert workflow.advance(root, transport)["status"] == "queued"
    transport.execute()
    state = workflow.advance(root, transport)
    assert state["status"] == "succeeded", state["last_error"]
    return root, transport, state


def test_payload_lifecycle_and_verified_context(prepared):
    root, transport, state = complete(prepared)
    assert len(transport.jobs) == 1
    assert not state["stages"][0]["result"]["material_prediction_accepted"]
    assert (root/"01_pairing/outputs/execution.json").is_file()
    context = workflow.load_run_context(root)
    assert context["facts"]["run.accepted_stages"]["value"] == 1
    assert "stage_1.singlet.eigenvalue" in context["facts"]
    assert "cluster.invalid" not in json.dumps(context)
    assert "double_counting" not in json.dumps(context)
    assert context["facts"]["stage_1.material_prediction_accepted"]["value"] is False
    assert all(item["status"] == "verified" for item in context["artifacts"])


def test_lost_submission_reconciles_without_resubmission(prepared):
    root, transport, _ = prepared
    transport.lose_reply = True
    state = workflow.advance(root, transport)
    assert state["status"] == "needs_attention" and not state["stages"][0]["job_id"]
    state = workflow.resume(root, transport)
    assert state["status"] == "queued" and state["stages"][0]["job_id"] == "8001"
    assert len(transport.jobs) == 1
    assert "--reconcile-only" in transport.commands[-1]


def test_cancel_uncertain_submission_retains_job_identity(prepared):
    root, transport, _ = prepared
    transport.lose_reply = True
    workflow.advance(root, transport)
    assert workflow.cancel(root, transport)["status"] == "needs_attention"
    workflow.resume(root, transport)
    workflow.advance(root, transport)
    assert transport.cancel_count == 1
    transport.queue, transport.accounting = "", "CANCELLED"
    assert workflow.advance(root, transport)["status"] == "cancelled"
    assert len(transport.jobs) == 1


def test_changed_frozen_source_blocks_submission(prepared):
    root, transport, _ = prepared
    (root/"01_pairing/inputs/qe_pairing.py").write_text("changed")
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert not transport.commands


def test_cached_success_does_not_hide_output_tampering(prepared):
    root, _, state = complete(prepared)
    output = root/"01_pairing/outputs/result.json"
    result = json.loads(output.read_text())
    result["channels"][0]["eigenvalue"] = 17
    output.write_text(json.dumps(result))
    context = workflow.load_run_context(root)
    assert context["status"] == "needs_attention"
    assert context["facts"]["run.accepted_stages"]["value"] == 0
    assert "stage_1.singlet.eigenvalue" not in context["facts"]


def test_native_gap_check_rejects_relabelled_eigenvalue(prepared):
    root, _, _ = complete(prepared)
    output = root/"01_pairing/outputs"
    result = json.loads((output/"result.json").read_text())
    result["channels"][0]["eigenvalue"] = 17
    (output/"result.json").write_text(json.dumps(result))
    receipt = json.loads((output/"execution.json").read_text())
    receipt["output_sha256"]["result.json"] = digest(output/"result.json")
    (output/"execution.json").write_text(json.dumps(receipt))
    manifest = json.loads((output/"artifact_manifest.json").read_text())
    for name in ("result.json", "execution.json"):
        manifest[name]["sha256"] = digest(output/name)
    (output/"artifact_manifest.json").write_text(json.dumps(manifest))
    context = workflow.load_run_context(root)
    assert context["facts"]["run.accepted_stages"]["value"] == 0
    assert "stage_1.singlet.eigenvalue" not in context["facts"]


def test_scheduler_success_is_required(prepared):
    root, transport, _ = prepared
    workflow.advance(root, transport)
    transport.execute()
    transport.accounting = "FAILED"
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert workflow.load_run_context(root)["facts"]["run.accepted_stages"]["value"] == 0
