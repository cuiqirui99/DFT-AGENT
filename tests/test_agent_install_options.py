"""MCP installation plans retain site resource choices; they never start jobs."""

from unittest.mock import Mock

import pytest

from vasp_slurm_agent.mcp_server import ProjectTools
from test_project_execution import project


def test_qe_install_plan_forwards_site_and_epw_settings(project, monkeypatch):
    store, _, _, pid, _, connection = project
    tools = ProjectTools(store, pid)
    planner = Mock(return_value={"status": "planned"})
    monkeypatch.setattr(tools.registry, "qe_install_plan", planner)
    options = {"build_cpus": 8, "partition": "build", "account": "allocation", "include_epw": True,
               "extra_sbatch": ["--mem=16G", "--qos=build"],
               "extra_pbs": []}
    result = tools.call("qe_install_plan", {"connection_id": connection, **options})
    planner.assert_called_once_with(connection, **{key: value for key, value in options.items() if key != "extra_pbs"})
    assert result["status"] == "planned" and "Only the user" in result["next"]


def test_model_cannot_supply_commands_to_installation_probe(project, monkeypatch):
    store, _, _, pid, _, connection = project
    tools = ProjectTools(store, pid)
    planner = Mock(side_effect=AssertionError("The probe must not be reached"))
    monkeypatch.setattr(tools.registry, "qe_install_plan", planner)
    with pytest.raises(ValueError, match="schema"):
        tools.call("qe_install_plan", {"connection_id": connection, "setup_commands": ["sbatch /tmp/job.sh"]})
    planner.assert_not_called()


def test_engine_install_plan_forwards_pbs_resources(project, monkeypatch):
    store, _, _, pid, _, connection = project
    tools = ProjectTools(store, pid)
    planner = Mock(return_value={"status": "planned"})
    monkeypatch.setattr(tools.registry, "engine_install_plan", planner)
    result = tools.call("engine_install_plan", {"connection_id": connection, "engine": "abinit",
                        "account": "allocation", "extra_pbs": ["-l mem=16gb"]})
    planner.assert_called_once_with(connection, "abinit", build_cpus=4, partition=None,
                                    account="allocation", extra_pbs=["-l mem=16gb"])
    assert result["status"] == "planned" and "Only the user" in result["next"]
