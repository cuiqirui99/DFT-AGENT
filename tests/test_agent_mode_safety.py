"""Regression checks for CLI tool isolation and reviewed submission boundaries."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_agent_mode import call, keychain, project, tools_for
from vasp_slurm_agent import agent, agent_runner, cli_tools, project_files
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_agent import _operation


def test_mcp_cannot_authorize_qe_build(project, monkeypatch):
    start = Mock(side_effect=AssertionError("The model cannot submit a build"))
    monkeypatch.setattr(ConnectionRegistry, "qe_install_start", start)
    tools = tools_for(project)
    assert "qe_install_start" not in {item["name"] for item in tools.definitions()}
    error, reason = call(tools, "qe_install_start", connection_id="any", plan_id="any", user_confirmed=True)
    assert error and "Unknown tool" in reason
    with pytest.raises(ValueError, match="model cannot authorize"):
        tools.qe_install_start("any", "any", user_confirmed=True)
    start.assert_not_called()


def test_mcp_rejects_connections_outside_project_and_revoked_access(project, monkeypatch):
    hpc = project.registry.create("hpc", "Other cluster", {"host": "other.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    environment = project.registry.create_environment(hpc["id"], "Other QE", "qe", {"partition": "cpu", "qe_command": "srun pw.x", "qe_pseudo_root": "/scratch/u/pseudos"})
    tools = tools_for(project)
    for name in ("discover_environments", "qe_install_plan", "qe_install_status", "install_pseudopotentials"):
        monkeypatch.setattr(ConnectionRegistry, name, Mock(side_effect=AssertionError("Foreign cluster must not be accessed")))
    calls = [("detect_software", {"connection_id": hpc["id"]}),
             ("qe_install_plan", {"connection_id": hpc["id"]}),
             ("qe_install_status", {"connection_id": hpc["id"]}),
             ("install_pseudopotentials", {"environment_id": environment["id"], "elements": ["Si"]})]
    for name, args in calls:
        error, reason = call(tools, name, **args)
        assert error and "bound to this project" in reason
    project.registry.bind(project.pid, hpc["id"])
    project.registry.revoke(hpc["id"])
    assert call(tools, "detect_software", connection_id=hpc["id"])[0]
    model = project.registry.create("model", "Other model", {"provider": "codex"})
    tools.model_connection_id = model["id"]
    with pytest.raises(ValueError, match="bound to this project"):
        tools._settings()


def test_mcp_checks_schema_before_side_effects(project, monkeypatch):
    tools = tools_for(project)
    plan = Mock(side_effect=AssertionError("Do not plan invalid requests"))
    monkeypatch.setattr(tools, "qe_install_plan", plan)
    for cpus in (True, "8", 0, 65):
        error, reason = call(tools, "qe_install_plan", connection_id="cluster", build_cpus=cpus)
        assert error and "schema" in reason
    assert call(tools, "explain_results", run_ids=[], question="energy?")[0]
    assert call(tools, "create_workflow", title="x", goal="x", steps=[])[0]
    assert call(tools, "project_overview", unexpected=True)[0]
    plan.assert_not_called()


def test_mcp_workflow_preserves_python_inputs_outputs_and_environment(project):
    hpc = project.registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    project.registry.bind(project.pid, hpc["id"])
    environment = project.registry.create_environment(hpc["id"], "Python", "python", {"partition": "cpu", "command": "python3"})
    project_files.write_file(project.workspace, project.pid, "files/analyze.py", "print('analysis')\n")
    project_files.write_file(project.workspace, project.pid, "files/data.csv", "value\n1\n")
    error, result = call(tools_for(project), "create_workflow", title="Analyze", goal="Inspect data", steps=[{
        "id": "analysis", "name": "Analysis", "engine": "python", "goal": "Inspect the supplied data",
        "source_path": "files/analyze.py", "environment_id": environment["id"], "inputs": ["files/data.csv"],
        "arguments": ["data.csv"], "outputs": [{"path": "summary.json", "kind": "json", "required": True}]}])
    assert not error, result
    spec = project.workspace.get_workflow(project.pid, result["workflow_id"])["steps"][0]["spec"]
    assert spec["environment_id"] == environment["id"] and spec["inputs"] == ["files/data.csv"]
    assert spec["outputs"][0]["path"] == "summary.json" and spec["arguments"] == ["data.csv"]


def test_mcp_imports_from_temporary_directory_with_relative_pythonpath(project, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "src")
    env = agent_runner.server_environment(project.workspace, project.pid, project.sid, project.model)
    assert all(Path(part).is_absolute() for part in env["PYTHONPATH"].split(__import__("os").pathsep))
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_examples", "arguments": {}}}
    result = subprocess.run(agent_runner.server_command(), input=json.dumps(request) + "\n", text=True,
                            capture_output=True, cwd=tmp_path, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["result"]["isError"] is False and "Si.cif" in response["result"]["content"][0]["text"]


def test_runner_marks_partial_answer_with_failed_terminal_event(project, monkeypatch, tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    project.registry._patch_settings(project.model, cli_path=str(executable))
    events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "Partial answer"}},
              {"type": "turn.failed", "error": {"message": "429 usage limit private-marker"}}]
    monkeypatch.setattr(agent_runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="\n".join(map(json.dumps, events)), stderr=""))
    result = agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "hello", project.model)
    assert result["error"] and "usage limit" in result["reply"] and "private-marker" not in result["reply"]
    assert result["session"]["messages"][-1]["metadata"]["error"] is True


def test_runner_shares_guided_conversation_lock(project):
    with _operation(project.workspace, project.pid, project.sid):
        with pytest.raises(agent.AgentError, match="busy"):
            agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "hello", project.model)
    assert project.workspace.get_session(project.pid, project.sid)["messages"] == []


def test_runner_rejects_foreign_result_before_saving_or_calling_model(project):
    with pytest.raises(ValueError, match="Run not found"):
        agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "compare these", project.model,
                              run_ids=["00000000-0000-0000-0000-000000000000"])
    assert project.workspace.get_session(project.pid, project.sid)["messages"] == []


def test_cli_login_details_never_echo_api_key_fragments():
    def run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Logged in using an API key - arbitrary-private-key-fragment", stderr="")
    assert cli_tools.login_status("codex", "codex", run=run) == {"logged_in": True, "detail": "API key"}
    def claude(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"loggedIn": True, "authMethod": "private-marker"}), stderr="")
    assert cli_tools.login_status("claude_cli", "claude", run=claude) == {"logged_in": True, "detail": "Signed in"}


def test_runner_passes_selected_workflow_and_results_without_changing_message(project, monkeypatch, tmp_path):
    created = tools_for(project).create_workflow('Relax', 'PBE relaxation', [{
        'id': 'relax', 'name': 'Relax', 'engine': 'vasp', 'goal': 'Relax Si', 'source_path': 'files/Si.cif'}])
    executable = tmp_path / 'codex'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    project.registry._patch_settings(project.model, cli_path=str(executable))
    selected = ['00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002']
    lookup = Mock(return_value={})
    monkeypatch.setattr(project.workspace, 'get_run', lookup)
    captured = {}
    def run(command, **kwargs):
        captured['prompt'] = kwargs['input']
        events = [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Selected context received.'}},
                  {'type': 'turn.completed', 'usage': {}}]
        return SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, events)), stderr='')
    monkeypatch.setattr(agent_runner.subprocess, 'run', run)
    result = agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, 'Compare these', project.model,
                                  workflow_id=created['workflow_id'], step_id='relax', run_ids=selected)
    assert result['error'] is None
    assert all(identifier in captured['prompt'] for identifier in [created['workflow_id'], *selected])
    assert lookup.call_count == 2 and lookup.call_args_list[0].args == (project.pid, selected[0])
    assert result['session']['messages'][0]['content'] == 'Compare these'


@pytest.mark.parametrize('provider,stdout', [
    ('codex', json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Only a partial answer'}})),
    ('claude_cli', json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Only a partial answer'}]}})),
])
def test_runner_rejects_stream_without_completion(project, monkeypatch, tmp_path, provider, stdout):
    executable = tmp_path / 'model-cli'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    model = project.registry.create('model', 'Incomplete CLI', {'provider': provider, 'cli_path': str(executable)})
    project.registry.bind(project.pid, model['id'])
    monkeypatch.setattr(agent_runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=''))
    result = agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, 'hello', model['id'])
    assert result['error'] and result['session']['messages'][-1]['metadata']['error']


def test_recovered_tool_error_does_not_mark_completed_turn_failed():
    stream = "\n".join(json.dumps(event) for event in [
        {"type": "item.completed", "item": {"type": "error", "message": "Tool unavailable"}},
        {"type": "error", "message": "Retrying stream"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Revised the same task."}},
        {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 8}},
    ])
    parsed = agent_runner.parse_codex(stream)
    assert parsed['completed'] and not parsed['terminal_failed']
    assert parsed['reply'] == 'Revised the same task.'
