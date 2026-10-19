"""Real local Python payloads; SSH/scheduler and Wannier output are test doubles."""

from dataclasses import replace
import builtins
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from vasp_slurm_agent import script_workflow as script, explanation, workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.scheduler import get_scheduler
from vasp_slurm_agent.transport import Result
from vasp_slurm_agent.workspace import Workspace


class Transport:
    def __init__(self, remote, scheduler="slurm"):
        self.remote, self.scheduler = remote, scheduler
        self.remote.mkdir()
        self.commands, self.jobs = [], {}
        self.queue, self.accounting, self.exit_code = "PENDING", "", 0
        self.lose_reply, self.cancel_count = False, 0

    def run(self, command, timeout=60):
        self.commands.append(command)
        words = shlex.split(command)
        if command.startswith("set -e\n"):
            return Result(0)
        if words[:2] == ["python3", "-c"]:
            return Result(0, json.dumps([p.name for p in self.remote.iterdir() if p.is_file()]))
        if command.startswith("python3 ") and "/dispatch.py" in command:
            name = words[3]
            if "--reconcile-only" in words and name not in self.jobs:
                return Result(1, "", "No recorded submission")
            self.jobs.setdefault(name, "8101.pbs.example" if self.scheduler == "pbs" else "8101")
            receipt = {"job_id": self.jobs[name], "job_name": name}
            if self.scheduler != "slurm":
                receipt["scheduler"] = self.scheduler
            (self.remote / "job.json").write_text(json.dumps(receipt))
            if self.lose_reply:
                self.lose_reply = False
                return Result(255, "", "Submission outcome uncertain")
            return Result(0, json.dumps(receipt))
        if command.startswith("qstat "):
            record = {"job_state": "Q" if self.queue else "F", "Exit_status": self.exit_code}
            if self.accounting == "CANCELLED":
                record["substate"] = 91
            return Result(0, json.dumps({"Jobs": {"8101.pbs.example": record}}))
        if command.startswith("squeue "):
            return Result(0, self.queue)
        if command.startswith("sacct "):
            return Result(0, f"8101|{self.accounting}|{self.exit_code}:0|" if self.accounting else "")
        if command.startswith(("scancel ", "qdel ")):
            self.cancel_count += 1
            return Result(0)
        pytest.fail(command)

    def upload_many(self, sources, remote, *, permitted_names):
        for source in sources:
            assert source.name in permitted_names
            (self.remote / source.name).write_bytes(source.read_bytes())
        return {}

    def download_many(self, remote, output, names, *, permitted_names):
        manifest = {}
        for name in names:
            assert name in permitted_names
            (output / name).write_bytes((self.remote / name).read_bytes())
            manifest[name] = {"size": (output / name).stat().st_size, "sha256": script._digest(output / name)}
        return manifest

    def execute(self):
        name = next(iter(self.jobs))
        env = dict(os.environ, SLURM_JOB_ID="8101", SLURM_JOB_NAME=name)
        if self.scheduler == "pbs":
            env.update(PBS_JOBID=self.jobs[name], PBS_JOBNAME=get_scheduler("pbs").job_name(name))
        process = subprocess.run([sys.executable, "execute.py"], cwd=self.remote, env=env,
                                 capture_output=True, text=True, timeout=30)
        self.queue, self.exit_code = "", process.returncode
        self.accounting = "COMPLETED" if process.returncode == 0 else "FAILED"
        return process


def config(scheduler="slurm"):
    return ClusterConfig(engine="qe", scheduler=scheduler, host="cluster.invalid", user="user",
                         remote_root="/scratch/project-id/runs", partition="cpu", tasks=1, qe_pseudo_root="/unused")


@pytest.fixture(params=["slurm", "pbs"])
def prepared(tmp_path, request):
    source = tmp_path / "analysis.py"
    source.write_text("import json,numpy as np\nfrom pathlib import Path\n"
                      "Path('result.json').write_text(json.dumps({'sum':sum(range(10))}))\n"
                      "Path('data.csv').write_text('x,y\\n1,2\\n3,4\\n')\n"
                      "np.save('array.npy',np.arange(5.))\nnp.savez('arrays.npz',x=np.arange(2.))\nprint('Done')\n")
    root = tmp_path / "run"
    state = script.prepare_run(root, config(request.param), engine="python", files={"analysis.py": source},
                               entrypoint="analysis.py", command=shlex.quote(sys.executable),
                               outputs=[{"path": name, "kind": kind} for name, kind in
                                        (("result.json", "json"), ("data.csv", "csv"), ("array.npy", "npy"), ("arrays.npz", "npz"))])
    return root, Transport(tmp_path / "remote", request.param), state


def complete(prepared):
    root, transport, _ = prepared
    assert workflow.advance(root, transport)["status"] == "queued"
    process = transport.execute()
    assert process.returncode == 0, process.stderr
    state = workflow.advance(root, transport)
    assert state["status"] == "completed_unverified", state["last_error"]
    return root, transport, state


def test_python_lifecycle_and_typed_downloads_do_not_certify_science(prepared):
    root, transport, state = complete(prepared)
    assert len(transport.jobs) == 1
    output = root / "01_python/outputs"
    assert json.loads((output / "result.json").read_text())["sum"] == 45
    np.testing.assert_array_equal(np.load(output / "array.npy"), np.arange(5.))
    assert state["stages"][0]["result"]["process_success"]
    context = explanation.load_run_context(root)
    assert context["status"] == "completed_unverified"
    assert context["facts"]["run.process_success"]["value"]
    assert not context["facts"]["run.accepted"]["value"]
    result_file = next(item for item in context["artifacts"] if item["label"] == "result.json")
    assert result_file["kind"] == "json" and result_file["size"] > 0
    assert "cluster.invalid" not in json.dumps(context)
    assert str(root) not in json.dumps(context)
    archive = workflow.bundle_run(root)
    with zipfile.ZipFile(archive) as bundle:
        assert "01_python/outputs/result.json" in bundle.namelist()
        assert "config.json" not in bundle.namelist()
        assert "remote_root" not in bundle.read("run.json").decode()


@pytest.mark.parametrize("change", [None, "output", "scheduler", "state"])
def test_project_deletion_rechecks_completed_python_evidence(prepared, tmp_path, change):
    original, _, _ = complete(prepared)
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create_project("Analysis")
    root = workspace.project_paths(project["id"])["runs"] / "analysis"
    shutil.copytree(original, root)
    workspace.register_run(project["id"], root)
    if change == "output":
        (root / "01_python/outputs/result.json").write_text('{"sum": 46}')
    elif change == "scheduler":
        (root / "01_python/outputs/scheduler.txt").write_text("RUNNING")
    elif change == "state":
        state = script.read_state(root)
        state["stages"][0]["scheduler_state"] = "RUNNING"
        script._write(root / "run.json", state)
    if change:
        with pytest.raises(ValueError, match="reconciled"):
            workspace.delete_project(project["id"], confirmation=project["name"])
        assert not workspace.get_project(project["id"])["deleted"]
    else:
        assert workspace.delete_project(project["id"], confirmation=project["name"])["deleted"]
        assert root.is_dir()
        workspace.restore_project(project["id"])
        context = script.load_run_context(root)
        assert context["facts"]["run.process_success"]["value"]
        assert not context["facts"]["run.accepted"]["value"]


def test_reconnect_keeps_job_identity_and_cancels_lost_ack(prepared):
    root, transport, _ = prepared
    transport.lose_reply = True
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert workflow.cancel(root, transport)["status"] == "needs_attention"
    state = workflow.resume(root, transport)
    assert state["status"] == "queued"
    assert "--reconcile-only" in transport.commands[-1]
    assert len(transport.jobs) == 1
    workflow.advance(root, transport)
    assert transport.cancel_count == 1
    transport.queue, transport.accounting = "", "CANCELLED"
    assert workflow.advance(root, transport)["status"] == "cancelled"


def test_poll_only_never_submits_or_opens_transport(prepared, monkeypatch):
    root, transport, state = prepared
    monkeypatch.setattr(script, "SSHTransport", lambda *_: pytest.fail("Poll opened a connection"))
    assert workflow.advance(root, poll_only=True)["status"] == "planned"
    assert not transport.commands


@pytest.mark.parametrize("target", ["source", "plan", "remote_root"])
def test_changed_review_blocks_before_network(prepared, target):
    root, transport, state = prepared
    if target == "source":
        (root / "01_python/inputs/analysis.py").write_text("print('changed')")
    elif target == "plan":
        (root / "plan.json").write_text("{}")
    else:
        state["remote_root"] = "/scratch/another-project"
        script._write(root / "run.json", state)
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert not transport.commands


@pytest.mark.parametrize("target", ["result.json", "runtime.json", "execution.json", "analysis.py", "job.json"])
def test_output_tampering_revokes_process_proof(prepared, target):
    root, _, _ = complete(prepared)
    path = root / "01_python/outputs" / target
    path.write_bytes(path.read_bytes() + b"\nchanged")
    context = script.load_run_context(root)
    assert not context["facts"]["run.process_success"]["value"]
    assert context["status"] == "needs_attention"


def test_missing_output_is_distinct_from_failed_process(prepared):
    root, transport, _ = prepared
    workflow.advance(root, transport)
    assert transport.execute().returncode == 0
    (transport.remote / "result.json").unlink()
    state = workflow.advance(root, transport)
    assert state["status"] == "missing_outputs"
    assert not script.load_run_context(root)["facts"]["run.accepted"]["value"]


def test_remote_wrapper_uses_python36_compatible_apis(prepared, monkeypatch):
    root, transport, _ = prepared
    state = script.advance(root, transport)
    stage = state["stages"][0]
    monkeypatch.chdir(transport.remote)
    monkeypatch.setenv("SLURM_JOB_ID", stage["job_id"])
    monkeypatch.setenv("SLURM_JOB_NAME", stage["job_name"])
    monkeypatch.setenv("PBS_JOBID", stage["job_id"])
    monkeypatch.setenv("PBS_JOBNAME", get_scheduler("pbs").job_name(stage["job_name"]))
    native_run, native_reversed = subprocess.run, builtins.reversed
    calls = []

    def legacy_run(*args, **kwargs):
        assert not {"capture_output", "text"} & kwargs.keys()
        calls.append(kwargs)
        return native_run(*args, **kwargs)

    def legacy_reversed(value):
        if isinstance(value, dict):
            raise TypeError("dict object is not reversible in Python 3.6")
        return native_reversed(value)

    monkeypatch.setattr(subprocess, "run", legacy_run)
    monkeypatch.setattr(builtins, "reversed", legacy_reversed)
    with pytest.raises(SystemExit) as finished:
        exec(compile(script._EXECUTE, "execute.py", "exec"), {})
    assert finished.value.code == 0
    receipt = json.loads((transport.remote / "execution.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["environment"]["software_version"]
    assert len(calls) == 2 and calls[0]["universal_newlines"] is True


def test_missing_execution_receipt_has_a_short_actionable_message(prepared):
    root, transport, _ = prepared
    script.advance(root, transport)
    assert transport.execute().returncode == 0
    (transport.remote / "execution.json").unlink()
    state = script.advance(root, transport)
    assert state["status"] == "needs_attention"
    assert state["last_error"] == "Execution receipt is missing. Check stdout.log, stderr.log and the scheduler logs."
    assert str(root) not in state["last_error"]
    assert (root / "01_python/outputs/stdout.log").is_file()
    assert len(transport.jobs) == 1


def test_failed_process_and_no_resubmission(tmp_path):
    source = tmp_path / "fail.py"
    source.write_text("raise SystemExit(7)\n")
    root = tmp_path / "run"
    script.prepare_run(root, config(), engine="python", files={source.name: source}, entrypoint=source.name,
                       command=shlex.quote(sys.executable), outputs=[])
    transport = Transport(tmp_path / "remote")
    script.advance(root, transport)
    assert transport.execute().returncode == 7
    assert script.advance(root, transport)["status"] == "failed"
    assert script.resume(root, transport)["status"] == "failed"
    assert len(transport.jobs) == 1
    assert script.load_run_context(root)["status"] == "failed"


@pytest.mark.parametrize("payload,kind", [("NaN", "json"), ("1e999", "json"), ("x,y\n1\n", "csv"),
                                         ("x,y\n1,nan\n", "csv"), ("not a numpy file", "npy")])
def test_malformed_typed_output_is_not_verified(tmp_path, payload, kind):
    path = tmp_path / "output"
    path.write_text(payload)
    with pytest.raises(ValueError):
        script._typed_output(path, kind)


@pytest.mark.parametrize("command", ["python3; rm something", "python3 $(whoami)", "python3 | tee", "python3\nother"])
def test_shell_operators_rejected(tmp_path, command):
    source = tmp_path / "script.py"
    source.write_text("print('ok')")
    with pytest.raises(ValueError, match="shell"):
        script.prepare_run(tmp_path / "run", config(), engine="python", files={source.name: source},
                           entrypoint=source.name, command=command)
    assert not (tmp_path / "run").exists()


def wannier_files(tmp_path):
    win = tmp_path / "model.win"
    win.write_text("num_wann = 2\nwrite_hr = true\n")
    wout = "WANNIER90 fixture\n" + "".join(f"{i} 1e-9 1e-8 2.0 1.0 <-- CONV\n" for i in range(1, 6))
    wout += "Final State\nWF centre and spread 1 ( 0.0, 0.0, 0.0 ) 1.0\nWF centre and spread 2 ( 0.5, 0.0, 0.0 ) 1.0\nAll done: wannier90 exiting\n"
    hr = "Native two-orbital fixture\n2\n3\n1 1 1\n"
    for vector, matrix in [((0,0,0), [[1, .2j], [-.2j, 2]]), ((1,0,0), [[-.5, .1j], [0, -.2]]),
                           ((-1,0,0), [[-.5, 0], [-.1j, -.2]])]:
        for i in range(2):
            for j in range(2):
                value = complex(matrix[i][j])
                hr += f"{vector[0]} {vector[1]} {vector[2]} {i+1} {j+1} {value.real} {value.imag}\n"
    stub = tmp_path / "wannier_fixture.py"
    stub.write_text("from pathlib import Path\n" + f"Path('model.wout').write_text({wout!r})\nPath('model_hr.dat').write_text({hr!r})\n")
    return {"model.win": win, "wannier_fixture.py": stub}, wout, hr


def test_wannier_callable_with_scoped_validation_not_material_claim(tmp_path):
    files, _, _ = wannier_files(tmp_path)
    root = tmp_path / "run"
    script.prepare_run(root, config(), engine="wannier90", files=files, entrypoint="model.win",
                       command=shlex.quote(sys.executable) + " wannier_fixture.py", outputs=[])
    transport = Transport(tmp_path / "remote")
    assert workflow.advance(root, transport)["status"] == "queued"
    assert transport.execute().returncode == 0
    state = workflow.advance(root, transport)
    assert state["status"] == "succeeded", state["last_error"]
    context = explanation.load_run_context(root)
    assert context["facts"]["run.accepted"]["value"]
    assert context["facts"]["stage_1.num_wann"]["value"] == 2
    assert not context["facts"]["stage_1.material_model_validated"]["value"]
    assert "Band interpolation accuracy" in " ".join(context["limits"])


@pytest.mark.parametrize("status", ["unconverged", "suspect"])
def test_finished_wannier_validation_failure_can_be_hidden_without_certifying_it(tmp_path, status):
    files, wout, hr = wannier_files(tmp_path)
    wout = wout.replace("1e-9", "0.1") if status == "unconverged" else wout.replace("All done: wannier90 exiting", "")
    files["wannier_fixture.py"].write_text("from pathlib import Path\n" +
        f"Path('model.wout').write_text({wout!r})\nPath('model_hr.dat').write_text({hr!r})\n")
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create_project("Wannier")
    root = workspace.project_paths(project["id"])["runs"] / "localization"
    script.prepare_run(root, config(), engine="wannier90", files=files, entrypoint="model.win",
                       command=shlex.quote(sys.executable) + " wannier_fixture.py")
    workspace.register_run(project["id"], root)
    transport = Transport(tmp_path / "remote")
    script.advance(root, transport)
    assert transport.execute().returncode == 0
    state = script.advance(root, transport)
    assert state["status"] == status and state["last_error"]
    assert workspace.delete_project(project["id"], confirmation=project["name"])["deleted"]
    workspace.restore_project(project["id"])
    assert not script.load_run_context(root)["facts"]["run.accepted"]["value"]


@pytest.mark.parametrize("change", ["nonhermitian", "nonfinite", "missing_matrix", "duplicate_matrix", "wrong_num_wann", "unconverged", "incomplete", "disentanglement"])
def test_wannier_validator_rejects_bad_native_evidence(tmp_path, change):
    files, wout, hr = wannier_files(tmp_path)
    if change == "nonhermitian":
        hr = hr.replace("0 0 0 1 2 0.0 0.2", "0 0 0 1 2 0.0 0.7")
    elif change == "nonfinite":
        hr = hr.replace("1.0 0.0", "nan 0.0", 1)
    elif change == "missing_matrix":
        hr = "\n".join(hr.splitlines()[:-1]) + "\n"
    elif change == "duplicate_matrix":
        hr = hr.replace("0 0 0 2 1", "0 0 0 1 1")
    elif change == "wrong_num_wann":
        files["model.win"].write_text("num_wann=3\n")
    elif change == "unconverged":
        wout = wout.replace("1e-9", "0.1")
    elif change == "disentanglement":
        wout = "1 2 1 0.5 1 <-- DIS\n" + wout
    else:
        wout = wout.replace("All done: wannier90 exiting", "")
    (tmp_path / "model.wout").write_text(wout)
    (tmp_path / "model_hr.dat").write_text(hr)
    stage = {"entrypoint": "model.win", "validation": script._validation("wannier90", None)}
    with pytest.raises(ValueError):
        script._wannier(tmp_path, stage)
