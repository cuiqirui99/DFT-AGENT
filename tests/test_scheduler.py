"""Offline scheduler integration; fixtures model documented OpenPBS responses.

No PBS server or scientific calculation is used by these tests.
Sources: https://github.com/openpbs/openpbs/blob/master/doc/man1/qstat.1B
         https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.scheduler import get_scheduler
from vasp_slurm_agent.transport import Result, TransportError
from vasp_slurm_agent import workflow, qe_workflow, qe_setup


def config(**changes):
    return ClusterConfig(**(dict(host="cluster", user="scientist", remote_root="/work/runs",
                                partition="workq", vasp_command="mpiexec vasp_std", potcar_root="/pots") | changes))


def pbs_output(job="241.pbs.example.org", state="F", code=0, name="example"):
    value = {"Job_Name": name, "Job_Owner": "scientist@login", "job_state": state}
    if code is not None:
        value["Exit_status"] = code
    return json.dumps({"timestamp": 1479277336, "pbs_version": "23.06", "pbs_server": "pbs.example.org", "Jobs": {job: value}})


def test_default_serialization_preserves_legacy_frozen_config(tmp_path):
    cfg = config()
    old = asdict(cfg)
    for key in ("scheduler", "extra_pbs", "ssh_jump_host", "ssh_identity_file", "connection_id", "connection_home",
                "engine_command", "engine_library_root", "engine_library"):
        old.pop(key)
    source = json.dumps(old, indent=2) + "\n"
    path = tmp_path / "config.json"
    path.write_text(source)
    ClusterConfig.load(path).save(path)
    assert path.read_text() == source
    assert cfg.to_dict() == old


def test_pbs_config_roundtrip_resources_and_ssh_references(tmp_path):
    cfg = config(scheduler="pbs", extra_pbs=["#PBS -l mem=8gb", "-l place=scatter"],
                 ssh_jump_host="user@gateway:2222", ssh_identity_file=str(tmp_path / "key"),
                 connection_id="conn-1", connection_home=str(tmp_path))
    assert ClusterConfig.load(cfg.save(tmp_path / "cfg.json")) == cfg
    assert "scheduler" in cfg.to_dict()
    script = "\n".join(get_scheduler(cfg).header(replace(cfg, nodes=2, tasks=5, cpus_per_task=2, walltime="2-01:02:03")))
    assert "#PBS -l select=1:ncpus=6:mpiprocs=3:ompthreads=2+1:ncpus=4:mpiprocs=2:ompthreads=2" in script
    assert "#PBS -l walltime=49:02:03" in script
    assert "#PBS -l mem=8gb" in script and "#SBATCH" not in script


@pytest.mark.parametrize("changes", [dict(scheduler="unknown"), dict(scheduler="pbs", extra_sbatch=["--mem=8G"]),
    dict(extra_pbs=["-l mem=8gb"]), dict(scheduler="pbs", extra_pbs=["-l select=2"]),
    dict(scheduler="pbs", extra_pbs=["-l mem=8gb", "-l mem=4gb"]),
    dict(scheduler="pbs", extra_pbs=["-l mem=8gb\nqsub other.sh"]),
    dict(scheduler="pbs", compute_backend="gpu", tasks=1), dict(setup_commands=["#PBS -l mem=8gb"]),
    dict(ssh_jump_host="-oProxyCommand=bad"), dict(ssh_jump_host="host;command"), dict(ssh_identity_file="relative")])
def test_rejects_mixed_or_unsafe_scheduler_configuration(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize("job,valid", [("241.pbs.example.org", True), ("241", True), ("241.pbs-01", True),
    ("241[].pbs", False), ("241[1].pbs", False), ("241;touch", False), ("-241", False), ("241.pbs\n", False)])
def test_pbs_job_identity(job, valid):
    assert get_scheduler("pbs").valid_job_id(job) is valid


@pytest.mark.parametrize("phase,code,expected", [("Q", None, "PENDING"), ("H", None, "PENDING"),
    ("R", None, "RUNNING"), ("E", None, "COMPLETING"), ("F", 0, "COMPLETED"),
    ("F", "0", "COMPLETED"), ("F", 271, "FAILED"), ("F", -11, "FAILED"),
    ("F", None, "UNKNOWN"), ("M", 0, "UNKNOWN"), ("X", None, "UNKNOWN")])
def test_pbs_state_requires_matching_identity_and_terminal_exit_status(phase, code, expected):
    adapter = get_scheduler("pbs")
    payload = pbs_output(state=phase, code=code)
    assert adapter.parse_status(payload, "241.pbs.example.org") == expected
    assert adapter.parse_status(payload, "999.other") == "UNKNOWN"


class StatusTransport:
    def __init__(self, response):
        self.response = response
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append(command)
        return self.response


def test_workflow_pbs_poll_and_cancel_preserve_server_suffix():
    transport = StatusTransport(Result(0, pbs_output()))
    assert workflow._poll(transport, {"job_id": "241.pbs.example.org"}, config(scheduler="pbs")) == "COMPLETED"
    assert transport.commands == ["qstat -x -f -F json 241.pbs.example.org"]
    assert get_scheduler("pbs").cancel_command("241.pbs.example.org") == ["qdel", "241.pbs.example.org"]
    transport.response = Result(153, "", "qstat: Unknown Job Id 241.pbs.example.org")
    assert workflow._poll(transport, {"job_id": "241.pbs.example.org"}, config(scheduler="pbs")) == "UNKNOWN"
    transport.response = Result(1, "", "Cannot connect to server")
    with pytest.raises(TransportError):
        workflow._poll(transport, {"job_id": "241.pbs.example.org"}, config(scheduler="pbs"))


@pytest.mark.skipif(os.name != "posix", reason="Executes remote POSIX dispatch locally")
def test_pbs_dispatch_replays_and_recovers_lost_ack_without_resubmission(tmp_path, monkeypatch):
    from vasp_slurm_agent import dispatch
    adapter = get_scheduler("pbs")
    name = "qe-long-run-identity-17-electron-phonon"
    calls = []

    def recorded(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "qsub":
            assert argv == ["qsub", "-N", adapter.job_name(name), "submit.sh"]
            raise subprocess.TimeoutExpired(argv, 60)
        assert argv == ["qstat", "-x", "-f", "-F", "json"]
        return subprocess.CompletedProcess(argv, 0, pbs_output(name=adapter.job_name(name)), "")

    monkeypatch.setattr(dispatch.subprocess, "run", recorded)
    with pytest.raises(subprocess.TimeoutExpired):
        dispatch.dispatch(tmp_path, name, scheduler="pbs")
    recovered = dispatch.dispatch(tmp_path, name, scheduler="pbs", reconcile_only=True)
    assert recovered == {"job_id": "241.pbs.example.org", "job_name": name, "scheduler": "pbs", "reconciled": True}
    assert dispatch.dispatch(tmp_path, name, scheduler="pbs") == recovered
    assert [argv[0] for argv in calls] == ["qsub", "qstat"]
    with pytest.raises(RuntimeError, match="does not match"):
        dispatch.dispatch(tmp_path, name, scheduler="slurm")


@pytest.mark.skipif(os.name != "posix", reason="Executes remote POSIX dispatch locally")
@pytest.mark.parametrize("matches", [0, 2])
def test_pbs_ambiguous_history_never_resubmits(tmp_path, monkeypatch, matches):
    from vasp_slurm_agent import dispatch
    name = "qe-example"
    (tmp_path / "submit_intent.json").write_text(json.dumps({"job_name": name, "scheduler": "pbs", "date": "2026-01-01"}))
    jobs = {str(index) + ".pbs": {"Job_Name": get_scheduler("pbs").job_name(name)} for index in range(matches)}
    def query(argv, **kwargs):
        assert argv[0] == "qstat"
        return subprocess.CompletedProcess(argv, 0, json.dumps({"Jobs": jobs}), "")
    monkeypatch.setattr(dispatch.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="outcome uncertain"):
        dispatch.dispatch(tmp_path, name, scheduler="pbs")
    assert not (tmp_path / "job.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="Executes generated POSIX scripts locally")
def test_pbs_vasp_script_enters_workdir_and_records_native_job_id(tmp_path):
    pots = tmp_path / "pots" / "Si"
    pots.mkdir(parents=True)
    (pots / "POTCAR").write_text("TITEL = test fixture only\n")
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        (tmp_path / name).write_text("fixture\n")
    cfg = config(scheduler="pbs", vasp_command="true", potcar_root=str(pots.parent))
    stage = {"name": "scf", "metadata": {"potcar_labels": ["Si"]}}
    script = tmp_path / "submit.sh"
    script.write_text(workflow._script(cfg, stage, None))
    assert "#SBATCH" not in script.read_text()
    env = dict(os.environ, PBS_JOBID="241.pbs.example.org", PBS_O_WORKDIR=str(tmp_path), PBS_NP="8")
    result = subprocess.run(["bash", str(script)], cwd=tmp_path.parent, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert receipt["job_id"] == "241.pbs.example.org" and receipt["exit_code"] == 0


def test_qe_pbs_script_and_build_have_allocation_and_workdir():
    cfg = config(scheduler="pbs", engine="qe", qe_pseudo_root="/upfs", qe_command="mpiexec pw.x", extra_pbs=["-l mem=16gb"])
    script = qe_workflow._script(cfg, {"name": "scf", "metadata": {"input_file": "pw.in", "output_file": "pw.out"}})
    assert "#PBS -q workq" in script and "PBS_O_WORKDIR" in script and "mpiexec pw.x -in pw.in" in script
    plan = qe_setup.install_plan(cfg, build_jobs=6)
    build = qe_setup._build_script(plan)
    assert "#PBS -l select=1:ncpus=6:mpiprocs=1:ompthreads=6" in build
    assert "#PBS -l mem=16gb" in build and "#SBATCH" not in build
    assert plan["scheduler"] == "pbs"
    assert "scheduler" not in qe_setup.install_plan(config())


@pytest.mark.parametrize("attributes,expected", [({"substate": 91}, "CANCELLED"),
    ({"Exit_status": -26}, "OUT_OF_MEMORY"), ({"Exit_status": -27}, "OUT_OF_MEMORY"),
    ({"Exit_status": -28}, "TIMEOUT"), ({"Exit_status": -29}, "TIMEOUT")])
def test_pbs_native_terminal_causes(attributes, expected):
    payload = json.loads(pbs_output())
    payload["Jobs"]["241.pbs.example.org"].update(attributes)
    assert get_scheduler("pbs").parse_status(json.dumps(payload), "241.pbs.example.org") == expected
