"""Site resources, PBS discovery, and resumable installation planning."""
import json
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import discovery, engine_setup, qe_setup, setup_assistant, transport
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.connections import ConnectionRegistry


OPTIONS = ["--qos=required", "--mem=8G", "--constraint=zen4"]
PLATFORM = {"platform": "linux-64", "glibc": "2.34"}


def config(engine="qe", **changes):
    values = {"host": "cluster.example", "user": "researcher", "remote_root": "/scratch/researcher",
              "engine": engine, "partition": "cpu"}
    values.update({"qe_pseudo_root": "/pseudos"} if engine == "qe"
                  else {"engine_command": "mpiexec " + engine, "engine_library_root": "/library"})
    return ClusterConfig(**(values | changes))


@pytest.mark.parametrize("engine", ["qe", "abinit", "cp2k", "elk"])
def test_install_resources_are_preserved_and_change_plan_identity(engine):
    cfg = config(engine, account="project", extra_sbatch=OPTIONS)
    if engine == "qe":
        plan = qe_setup.install_plan(cfg)
        script = qe_setup._build_script(plan)
        other = qe_setup.install_plan(config(engine, account="project", extra_sbatch=["--mem=16G"]))
    else:
        plan = engine_setup.install_plan(cfg, engine, PLATFORM)
        script = engine_setup._job_script(plan)
        other = engine_setup.install_plan(config(engine, account="project", extra_sbatch=["--mem=16G"]), engine, PLATFORM)
    assert plan["extra_sbatch"] == OPTIONS
    assert all("#SBATCH " + option in script for option in OPTIONS)
    assert "#SBATCH --partition=cpu" in script and "#SBATCH --account=project" in script
    assert plan["id"] != other["id"] and plan["work_dir"] != other["work_dir"]


def test_build_owns_cpu_layout_without_losing_site_requirements():
    cfg = config(extra_sbatch=OPTIONS + ["--cpus-per-task=24", "--ntasks-per-node=32", "--gpus=4"])
    plan = qe_setup.install_plan(cfg, build_jobs=6)
    script = qe_setup._build_script(plan)
    assert plan["extra_sbatch"] == OPTIONS
    assert script.count("#SBATCH --cpus-per-task=") == 1
    assert "#SBATCH --cpus-per-task=6" in script and "#SBATCH --ntasks=1" in script
    assert "--gpus" not in script and "--ntasks-per-node" not in script


@pytest.mark.parametrize("engine", ["qe", "abinit", "cp2k", "elk"])
def test_pbs_install_resources_are_retained_in_plan_identity(engine):
    cfg = config(engine, scheduler="pbs", extra_pbs=["-l mem=8gb", "-l place=excl"])
    if engine == "qe":
        plan = qe_setup.install_plan(cfg)
        script = qe_setup._build_script(plan)
        other = qe_setup.install_plan(config(engine, scheduler="pbs", extra_pbs=["-l mem=16gb"]))
    else:
        plan = engine_setup.install_plan(cfg, engine, PLATFORM)
        script = engine_setup._job_script(plan)
        other = engine_setup.install_plan(config(engine, scheduler="pbs", extra_pbs=["-l mem=16gb"]), engine, PLATFORM)
    assert "#PBS -l mem=8gb" in script and "#PBS -l place=excl" in script
    assert "#SBATCH" not in script
    assert plan["id"] != other["id"]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    result = ConnectionRegistry(tmp_path / "workspace")
    cid = result.create("hpc", "Cluster", {"host": "cluster.example", "user": "researcher",
                        "remote_base": "/scratch/researcher", "scheduler": "slurm", "auth_mode": "agent"})["id"]
    result._save_probe(cid, result.get(cid)["revision"], {"auth_status": "available", "connection_status": "connected",
                      "schedulers": ["slurm"], "discovery": {"suggested": {"partition": "cpu"}}})
    monkeypatch.setattr(result, "_qe_transport", lambda item: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(qe_setup, "detect_toolchain", lambda config, transport=None: {
        "ready": True, "setup_commands": list(config.setup_commands), "mpi_fortran": "/usr/bin/mpifort",
        "notes": [], "modules_tried": []})
    monkeypatch.setattr(engine_setup, "detect_platform", lambda config, transport=None: PLATFORM)
    return result, cid


@pytest.mark.parametrize("engine", ["qe", "abinit", "cp2k", "elk"])
@pytest.mark.parametrize("scheduler", ["slurm", "pbs", "auto_pbs"])
def test_registry_plan_start_and_installed_environment_keep_resources(registry, monkeypatch, engine, scheduler):
    registry, cid = registry
    registry._patch_settings(cid, scheduler="auto" if scheduler == "auto_pbs" else scheduler)
    if scheduler == "auto_pbs":
        registry._store_probe(cid, "schedulers", ["pbs"])
    resources = {"extra_sbatch": OPTIONS} if scheduler == "slurm" else {"extra_pbs": ["-l mem=8gb"]}
    module = qe_setup if engine == "qe" else engine_setup
    seen = []
    def start(cfg, plan, **kwargs):
        script = qe_setup._build_script(plan) if engine == "qe" else engine_setup._job_script(plan)
        seen.append((cfg, plan, script))
        return {"status": "submitted", "job_id": "41", "id": plan["id"], "work_dir": plan["work_dir"]}
    monkeypatch.setattr(module, "start_install", start)
    monkeypatch.setattr(module, "poll_install", lambda *args, **kwargs: {"status": "installed", "scheduler_state": "COMPLETED"})
    if engine == "qe":
        planned = registry.qe_install_plan(cid, account="project", **resources)
        assert not seen
        registry.qe_install_start(cid, planned["plan_id"])
        result = registry.qe_install_status(cid)
    else:
        planned = registry.engine_install_plan(cid, engine, account="project", **resources)
        assert not seen
        registry.engine_install_start(cid, engine, planned["plan_id"])
        result = registry.engine_install_status(cid, engine)
    assert len(seen) == 1
    cfg, plan, script = seen[0]
    for field, options in resources.items():
        assert planned[field] == options and result[field] == options
        assert getattr(cfg, field) == options and plan[field] == options
        assert all(option in script for option in options)
        assert registry.environment(result["environment_id"])["config"][field] == options
        environment = registry.environment(result["environment_id"])
        registry.update_environment(environment["id"], environment["config"])
    assert cfg.account == "project"


def test_qe_failed_build_replans_to_new_folder_without_overwriting_old_receipt(registry):
    registry, cid = registry
    first = registry.qe_install_plan(cid, extra_sbatch=OPTIONS)
    previous = registry.get(cid)["probe"]["qe_install"]
    registry._store_probe(cid, "qe_install", {**previous, "status": "failed", "receipt": {"job_id": "41"}})
    second = registry.qe_install_plan(cid, extra_sbatch=OPTIONS)
    current = registry.get(cid)["probe"]["qe_install"]
    assert first["plan_id"] != second["plan_id"] and first["prefix"] != second["prefix"]
    assert current["plan"]["work_dir"] != previous["plan"]["work_dir"]
    assert current["attempt"] == 1 and "receipt" not in current
    assert registry.qe_install_plan(cid, extra_sbatch=OPTIONS)["plan_id"] == second["plan_id"]
    assert qe_setup.install_plan(registry._qe_base_config(registry.get(cid), partition="cpu", extra_sbatch=OPTIONS),
                                prefix=current["plan"]["prefix"], attempt=1) == current["plan"]


@pytest.mark.parametrize("status", ["submitted", "running", "submitting", "unknown"])
def test_qe_replanning_cannot_lose_active_or_uncertain_build(registry, status):
    registry, cid = registry
    registry.qe_install_plan(cid)
    record = {**registry.get(cid)["probe"]["qe_install"], "status": status, "receipt": {"job_id": "41"}}
    registry._store_probe(cid, "qe_install", record)
    with pytest.raises(ValueError, match="already tracked"):
        registry.qe_install_plan(cid, build_cpus=16)
    assert registry.get(cid)["probe"]["qe_install"] == record


@pytest.mark.parametrize("old,new", [("abinit", "cp2k"), ("cp2k", "elk"), ("elk", "qe"), ("abinit", "vasp")])
def test_switching_software_requires_new_program_and_library(old, new):
    state = setup_assistant.begin("software")
    state["settings"]["engine"] = old
    state = setup_assistant.advanced(state, {"engine_command": "mpiexec " + old,
                "engine_library_root": "/library/" + old, "engine_library": {"Si": "/library/Si"}})
    changed = setup_assistant.answer(state, "engine", new)
    assert not {"engine_command", "engine_library_root", "engine_library"} & changed["settings"]["config"].keys()
    assert "engine_command" in state["settings"]["config"]


def pbs_output():
    queues = {"Queue": {"workq": {"queue_type": "Execution", "enabled": True, "started": True,
                                   "resources_max": {"walltime": "24:00:00"}},
                         "disabled": {"queue_type": "Execution", "enabled": False, "started": True},
                         "routing": {"queue_type": "Route", "enabled": True, "started": True}}}
    return "\n".join([discovery.MARK + " pbs", "qsub", "qstat", "qdel",
        discovery.MARK + " pbs_queues", json.dumps(queues, indent=2), discovery.MARK + " slurm",
        "sbatch", "squeue", "sacct", "scancel", discovery.MARK + " partitions", "slurm-only*|up|1:00:00|8",
        discovery.MARK + " accounts", "slurm-account", discovery.MARK + " executables",
        "vasp_std=/opt/vasp_std", "vasp_ncl=/opt/vasp_ncl", discovery.MARK + " potcar", "/licensed/PBE",
        discovery.MARK + " codes", "exe pw.x=/opt/qe/pw.x"])


def test_pbs_probe_uses_execution_queues_and_never_slurm_launcher(registry, monkeypatch):
    registry, cid = registry
    registry._patch_settings(cid, scheduler="pbs")
    class Remote:
        def __init__(self, *args, **kwargs): pass
        def run(self, command, timeout=60):
            assert "qstat -Qf -F json" in command
            return transport.Result(0, pbs_output(), "")
        def close(self): pass
    monkeypatch.setattr(transport, "SSHTransport", Remote)
    monkeypatch.setattr(registry, "ssh_password", lambda identifier: None)
    report = registry.discover_environments(cid)
    assert report["suggested"]["partition"] == "workq" and "account" not in report["suggested"]
    environments = {e["engine"]: e["config"] for e in registry.environments(cid)}
    assert environments["qe"]["qe_command"] == "mpiexec pw.x"
    assert environments["vasp"]["vasp_command"] == "mpiexec vasp_std"
    assert environments["vasp"]["vasp_ncl_command"] == "mpiexec vasp_ncl"
    assert all(e["partition"] == "workq" for e in environments.values())
    assert not any("Missing Slurm" in note for note in report["notes"])


def test_slurm_suggestions_still_use_slurm_when_both_schedulers_exist():
    report = discovery.suggest(discovery.parse_probe(pbs_output()))
    assert report["suggested"]["partition"] == "slurm-only"
    assert report["suggested"]["account"] == "slurm-account"
    assert report["suggested"]["vasp_command"] == "srun vasp_std"


@pytest.mark.parametrize("payload", ["not JSON", '{"Queue":[]}', '{"Queue":{"route":{"queue_type":"Route"}}}'])
def test_invalid_pbs_queue_data_requires_manual_queue(payload):
    facts = discovery.parse_probe(discovery.MARK + " pbs_queues\n" + payload)
    report = discovery.suggest(facts, scheduler="pbs")
    assert "partition" not in report["suggested"]
    assert any("Enter a queue" in note for note in report["notes"])


def test_compact_pbs_json_with_long_attributes_is_not_truncated():
    payload = {"Queue": {"workq": {"queue_type": "Execution", "enabled": True, "started": True,
                                   "comment": "a" * 1000}}}
    facts = discovery.parse_probe(discovery.MARK + " pbs_queues\n" + json.dumps(payload))
    assert discovery.suggest(facts, scheduler="pbs")["suggested"]["partition"] == "workq"
    with pytest.raises(discovery.DiscoveryError, match="too large"):
        discovery.parse_probe("a" * 2_000_001)


def test_install_options_resolve_auto_scheduler_from_probe(monkeypatch):
    from vasp_slurm_agent import setup_app
    captured = []
    record = {"settings": {"scheduler": "auto"}, "probe": {"schedulers": ["pbs"]}}
    registry = SimpleNamespace(_scheduler_for=lambda item: ConnectionRegistry._scheduler_for(None, item))
    monkeypatch.setattr(setup_app.st, "text_area", lambda label, value, **kwargs: captured.append((label, value, kwargs)) or "-l mem=8gb")
    result = setup_app._installation_resources(registry, record, {"plan": {"extra_pbs": ["-l mem=4gb"]}}, "test")
    assert result == {"extra_pbs": ["-l mem=8gb"]}
    assert captured[0][1] == "-l mem=4gb" and captured[0][2]["placeholder"] == "-l mem=8gb"
