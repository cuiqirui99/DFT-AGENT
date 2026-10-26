"""The agent reviews requested resources from the frozen run configuration."""

import json

import pytest

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.mcp_server import ProjectTools, _review_resources
from test_project_execution import project, prepared, script_graph, SETTINGS


def config(**changes):
    return ClusterConfig(host="private-host.invalid", user="private-user", remote_root="/private/runs",
                         potcar_root="/private/potentials", vasp_command="/private/bin/vasp",
                         setup_commands=["source /private/setup.sh"], partition="cpu", **changes)


def test_allocation_review_excludes_private_config_and_uses_effective_cpu_count():
    result = _review_resources(config(nodes=2, tasks=16, walltime="00:12:00", account="allocation",
                                     extra_sbatch=["--mem=32G", "--cpus-per-task=4", "--licenses=/private/license"]))
    assert result["nodes"] == 2 and result["tasks"] == 16 and result["cpus_per_task"] == 4
    assert result["walltime"] == "00:12:00" and result["account"] == "allocation"
    assert result["options"] == {"mem": "32G", "cpus-per-task": "4"}
    assert result["hidden_options"] == ["licenses"]
    assert "private" not in json.dumps(result) and "gpus_per_node" not in result


def test_gpu_and_pbs_allocations():
    gpu = _review_resources(config(nodes=2, tasks=4, compute_backend="gpu", gpus_per_node=2, cpus_per_task=8))
    assert gpu["gpus_per_node"] == 2 and gpu["cpus_per_task"] == 8
    assert gpu["options"]["gpus-per-node"] == "2" and gpu["options"]["ntasks-per-node"] == "2"
    pbs = _review_resources(config(scheduler="pbs", extra_pbs=["-l mem=16gb"], cpus_per_task=2))
    assert pbs["scheduler"] == "pbs" and pbs["options"] == {"mem": "16gb"}
    assert pbs["cpus_per_task"] == 2


def test_tool_reads_frozen_allocation_after_environment_changes(project, monkeypatch):
    store, registry, execution, pid, wid, connection = project
    environment = registry.environments(connection)[0]
    original = {**environment["config"], "nodes": 2, "tasks": 16, "walltime": "00:12:00",
                "extra_sbatch": ["--mem=32G"]}
    registry.update_environment(environment["id"], original)
    prepared(project, monkeypatch)
    registry.update_environment(environment["id"], {**original, "tasks": 64, "walltime": "02:00:00"})
    result = ProjectTools(store, pid).review_inputs(wid, "scf")
    assert result["resources"]["tasks"] == 16 and result["resources"]["walltime"] == "00:12:00"
    assert result["resources"]["options"]["mem"] == "32G"
    assert "host" not in result["resources"] and "setup_commands" not in result["resources"]
    assert result["stages"] and result["contents"]


@pytest.mark.parametrize("engine", ["python", "wannier90"])
def test_tool_reviews_script_allocation(project, engine):
    store, registry, execution, pid, _, _ = project
    wid, environment_id = script_graph(project, engine)
    environment = registry.environment(environment_id)
    registry.update_environment(environment_id, {**environment["config"], "walltime": "00:07:00",
                                                "extra_sbatch": ["--mem=4G"]})
    execution.draft_step(pid, wid, "program", SETTINGS)
    execution.prepare_step(pid, wid, "program")
    result = ProjectTools(store, pid).review_inputs(wid, "program")
    assert result["resources"]["tasks"] == 1 and result["resources"]["walltime"] == "00:07:00"
    assert result["resources"]["options"] == {"mem": "4G"}
