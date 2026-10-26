"""Agent mode: the project's MCP tools, the CLI runner, and managed QE installation.

CLIs and clusters are simulated. The MCP server is exercised over real stdio.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import agent, agent_runner, credentials, mcp_server, project_files, qe_setup
from vasp_slurm_agent.agent import ModelSettings
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.transport import Result
from vasp_slurm_agent.workspace import Workspace
from importlib.resources import files


class Keychain:
    def __init__(self):
        self.values = {}

    def set_password(self, service, name, value):
        self.values[service, name] = value

    def get_password(self, service, name):
        return self.values.get((service, name))

    def delete_password(self, service, name):
        self.values.pop((service, name), None)


@pytest.fixture
def keychain(monkeypatch):
    store = Keychain()
    monkeypatch.setattr(credentials, "_keyring", lambda: store)
    return store


@pytest.fixture
def project(tmp_path, keychain):
    workspace = Workspace(tmp_path / "home")
    registry = ConnectionRegistry(workspace)
    pid = workspace.create_project("Silicon")["id"]
    sid = workspace.create_session(pid)["id"]
    model = registry.create("model", "Codex", {"provider": "codex"})
    registry.bind(pid, model["id"])
    project_files.write_file(workspace, pid, "files/Si.cif", files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    return SimpleNamespace(workspace=workspace, registry=registry, pid=pid, sid=sid, model=model["id"])


def tools_for(project):
    return mcp_server.ProjectTools(project.workspace, project.pid, project.sid, project.model)


def call(tools, name, **arguments):
    response = mcp_server.handle(tools, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    content = response["result"]["content"][0]["text"]
    return response["result"]["isError"], (json.loads(content) if not response["result"]["isError"] else content)


def test_prompt_preserves_long_requirements_and_user_revisions():
    goal = "Details. " * 1000 + "Use QE, PBE, 73/584 Ry, a 7x7x7 mesh and conv_thr=1e-9."
    history = [{"role": "user", "content": goal},
               {"role": "assistant", "content": "Ready."},
               {"role": "user", "content": "Change the mesh to 9x9x9; keep the other settings."}]
    prompt = agent_runner.render_prompt(history, "Use the silicon structure.")
    assert goal in prompt
    assert prompt.index("7x7x7") < prompt.index("Change the mesh to 9x9x9")
    assert "earlier messages are omitted" not in prompt
    assert prompt.endswith("User's new message:\nUse the silicon structure.")


def test_prompt_exposes_old_history_cursor_without_silently_dropping_requirements():
    history = [{"role": "user", "content": f"Requirement {index}"} for index in range(25)]
    prompt = agent_runner.render_prompt(history, "Continue.")
    assert "conversation_history(before=5)" in prompt
    assert "follow next_before" in prompt
    assert "[5] user: Requirement 5" in prompt
    assert "[24] user: Requirement 24" in prompt
    assert "[0] user:" not in prompt


def test_prompt_redacts_credentials_from_legacy_history():
    prompt = agent_runner.render_prompt([{"role": "user", "content": "password=old-test-secret"}], "Continue.")
    assert "old-test-secret" not in prompt
    assert "[secret omitted]" in prompt


# --- MCP server ----------------------------------------------------------------

def test_mcp_handshake_and_tool_listing(project):
    tools = tools_for(project)
    init = mcp_server.handle(tools, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert init["result"]["protocolVersion"] == "2024-11-05" and init["result"]["serverInfo"]["name"] == "dft-agent"
    assert mcp_server.handle(tools, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = mcp_server.handle(tools, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    names = {item["name"] for item in listed}
    assert {"project_overview", "read_file", "describe_structure", "create_workflow", "revise_workflow", "plan_step", "prepare_inputs",
            "review_inputs", "request_submission", "explain_results", "qe_install_plan", "install_pseudopotentials"} <= names
    assert all(item["inputSchema"]["type"] == "object" for item in listed)
    assert "submit" not in names and "shell" not in names
    assert mcp_server.handle(tools, {"jsonrpc": "2.0", "id": 2, "method": "resources/list"})["error"]["code"] == -32601
    assert mcp_server.handle(tools, {"jsonrpc": "2.0", "id": 3, "method": "ping"})["result"] == {}


def test_mcp_tools_read_describe_and_guard_private_files(project):
    tools = tools_for(project)
    error, overview = call(tools, "project_overview")
    assert not error and [item["path"] for item in overview["files"]] == ["files/Si.cif"]
    assert overview["connections"][0]["provider"] == "codex" and overview["workflows"] == []
    error, described = call(tools, "describe_structure", path="files/Si.cif")
    assert not error and described["formula"] == "Si" and described["spacegroup"]["number"] == 227 and described["elements"] == ["Si"]
    error, text = call(tools, "read_file", path="config/credentials.json")
    assert error and "Private" in text
    error, text = call(tools, "write_file", path="files/note.txt", content="OPENAI_API_KEY=sk-test-1234567890abcdef")
    assert error and "secure credential" in text
    error, written = call(tools, "write_file", path="files/note.txt", content="Relax first, then bands.")
    assert not error and written["created"]
    error, text = call(tools, "write_file", path="runs/x.txt", content="no")
    assert error
    error, text = call(tools, "nonexistent_tool")
    assert error and "Unknown tool" in text


def test_mcp_workflow_tools_plan_prepare_and_refuse_to_submit(project, monkeypatch):
    from test_conversation import provider as plan_provider, route, plan
    from vasp_slurm_agent import workflow

    monkeypatch.setattr(workflow, "start_worker", Mock(side_effect=AssertionError("Agent tools must not launch jobs")))
    hpc = project.registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    project.registry.bind(project.pid, hpc["id"])
    environment = project.registry.create_environment(hpc["id"], "VASP", "vasp", {"vasp_command": "srun vasp_std", "potcar_root": "/licensed", "partition": "cpu"})
    tools = tools_for(project)
    error, created = call(tools, "create_workflow", title="Relax silicon", goal="PBE relaxation",
                          steps=[{"name": "Relax", "engine": "vasp", "goal": "Relax silicon with PBE.", "source_path": "files/Si.cif"}])
    assert not error and created["steps"][0]["environment_id"] == environment["id"]
    workflow_id, step_id = created["workflow_id"], created["steps"][0]["id"]
    plan_provider(monkeypatch, [("dft_conversation", route()), ("dft_task", plan("vasp"))])
    error, planned = call(tools, "plan_step", workflow_id=workflow_id, step_id=step_id, instructions="Relax silicon with PBE.")
    assert not error and planned["state"] == "ready" and planned["plan"]
    error, prepared = call(tools, "prepare_inputs", workflow_id=workflow_id, step_id=step_id)
    assert not error and prepared["state"] == "prepared" and any(name.endswith("INCAR") for name in prepared["files"])
    error, review = call(tools, "review_inputs", workflow_id=workflow_id, step_id=step_id)
    assert not error and review["digest"] and any(name.endswith("INCAR") for name in review["contents"])
    error, request = call(tools, "request_submission", workflow_id=workflow_id, step_id=step_id)
    assert not error and request["status"] == "awaiting_user_approval"
    attempt = tools.execution.latest(project.pid, workflow_id, step_id)
    assert attempt["state"] == "prepared" and not attempt["approved_digest"]
    workflow.start_worker.assert_not_called()
    error, text = call(tools, "revise_workflow", workflow_id=workflow_id, title="Changed", goal="Changed",
                      steps=[{"name": "SCF", "engine": "vasp", "goal": "SCF", "source_path": "files/Si.cif"}])
    assert error and "follow-up" in text
    assert project.workspace.get_workflow(project.pid, workflow_id)["status"] != "superseded"
    assert tools.execution.latest(project.pid, workflow_id, step_id)["run_id"] == attempt["run_id"]
    error, text = call(tools, "create_workflow", title="Bad", goal="x", steps=[{"name": "A", "engine": "vasp", "goal": "g"}])
    assert error and "source_path" in text


def test_mcp_revises_downstream_intent_before_predecessors_finish(project, monkeypatch):
    from vasp_slurm_agent import workflow
    tools = tools_for(project)
    monkeypatch.setattr(workflow, "start_worker", Mock(side_effect=AssertionError("No draft may submit")))
    monkeypatch.setattr(agent, "_request_structured", Mock(side_effect=AssertionError("No unavailable structure may be planned")))
    steps = [
        {"id": "relax", "name": "Relax", "engine": "vasp", "goal": "Relax silicon", "source_path": "files/Si.cif"},
        {"id": "bands", "name": "Bands", "engine": "vasp", "goal": "Bands", "source_step": "relax", "source_stage": 0},
        {"id": "dos", "name": "DOS", "engine": "vasp", "goal": "DOS", "source_step": "relax", "depends_on": ["bands"]}]
    original = tools.create_workflow("Silicon", "Relax, bands and DOS", steps)
    old_id = original["workflow_id"]
    error, text = call(tools, "plan_step", workflow_id=old_id, step_id="bands", instructions="Use PBE and SOC")
    assert error and "preceding" in text
    revised = [dict(item) for item in steps]
    revised[0].update(name="PBE relaxation", goal="Relax silicon with PBE and no SOC.")
    for item in revised[1:]:
        item.update(name="PBE SOC " + item["name"], goal="Use PBE with SOC for " + item["goal"])
    error, result = call(tools, "revise_workflow", workflow_id=old_id, title="PBE + SOC", goal="SOC after relaxation", steps=revised)
    assert not error and result["workflow_id"] != old_id
    graph = project.workspace.get_workflow(project.pid, result["workflow_id"])
    assert [step["spec"]["goal"] for step in graph["steps"]] == [item["goal"] for item in revised]
    assert [step["name"] for step in graph["steps"]] == [item["name"] for item in revised]
    assert graph["steps"][1]["spec"]["source"] == {"kind": "step", "step_id": "relax", "stage": 0}
    assert set(graph["steps"][2]["depends_on"]) == {"relax", "bands"}
    assert project.workspace.get_workflow(project.pid, old_id)["status"] == "superseded"
    assert tools.execution.attempts(project.pid, old_id)[0]["last_error"] == "Complete the preceding workflow steps first."
    assert not tools.execution.attempts(project.pid, result["workflow_id"]) and not project.workspace.list_runs(project.pid)
    overview = tools.project_overview()["workflows"]
    assert len(overview) == 1 and overview[0]["id"] == result["workflow_id"] and overview[0]["title"] == "PBE + SOC"
    assert overview[0]["steps"][0]["source_path"] == "files/Si.cif"
    assert overview[0]["steps"][1]["source_step"] == "relax" and overview[0]["steps"][1]["source_stage"] == 0
    error, text = call(tools, "revise_workflow", workflow_id=old_id, title="Stale", goal="Stale", steps=revised)
    assert error and "revised workflow" in text and len(project.workspace.list_workflows(project.pid)) == 2
    error, text = call(tools, "plan_step", workflow_id=result["workflow_id"], step_id="bands")
    assert error and "preceding" in text
    workflow.start_worker.assert_not_called()
    agent._request_structured.assert_not_called()


def test_mcp_revision_uses_current_environment_and_preserves_program_inputs(project):
    tools = tools_for(project)
    project_files.write_file(project.workspace, project.pid, "files/analyze.py", "print('analysis')\n")
    cluster = project.registry.create("hpc", "Cluster", {"host": "fixture.invalid", "user": "researcher", "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    project.registry.bind(project.pid, cluster["id"])
    environments = [project.registry.create_environment(cluster["id"], name, "python", {"command": "python3", "partition": name})
                    for name in ("first", "second")]
    created = tools.create_workflow("Analysis", "Analyze", [{"id": "analysis", "name": "Analysis", "engine": "python", "goal": "Analyze",
        "source_path": "files/analyze.py", "environment_id": environments[0]["id"], "inputs": ["files/Si.cif"],
        "arguments": ["--check"], "outputs": [{"path": "result.json", "kind": "json", "required": True}]}])
    tools.execution.configure_step(project.pid, created["workflow_id"], "analysis", environments[1]["id"])
    visible = tools.project_overview()["workflows"][0]["steps"][0]
    assert visible["environment_id"] == environments[1]["id"]
    fields = next(item for item in tools.definitions() if item["name"] == "revise_workflow")["inputSchema"]["properties"]["steps"]["items"]["properties"]
    revised = {key: value for key, value in visible.items() if key in fields}
    revised["goal"] = "Analyze and report results"
    result = tools.revise_workflow(created["workflow_id"], "Analysis", revised["goal"], [revised])
    saved = project.workspace.get_workflow(project.pid, result["workflow_id"])["steps"][0]["spec"]
    assert saved["environment_id"] == environments[1]["id"]
    for name in ("inputs", "input_artifacts", "arguments", "outputs"):
        assert saved[name] == revised[name]
    assert saved["source"] == {"kind": "file", "path": "files/analyze.py"}
    assert not project.workspace.list_runs(project.pid)


def test_mcp_server_runs_over_stdio(project):
    env = {**os.environ, "DFT_AGENT_HOME": str(project.workspace.home), "DFT_AGENT_PROJECT": project.pid,
           "DFT_AGENT_SESSION": project.sid, "DFT_AGENT_MODEL_CONNECTION": project.model}
    env.pop("DFT_AGENT_UI", None)
    with subprocess.Popen(agent_runner.server_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env) as process:
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_examples", "arguments": {}}},
                    "not json"]
        stdout, stderr = process.communicate("\n".join(json.dumps(m) if not isinstance(m, str) else m for m in messages) + "\n", timeout=120)
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    assert [item.get("id") for item in responses] == [1, 2, None]
    assert "Si.cif" in json.loads(responses[1]["result"]["content"][0]["text"])["examples"]
    assert responses[2]["error"]["code"] == -32700
    assert all(line.startswith("{") for line in stdout.splitlines() if line.strip())


# --- CLI runner ------------------------------------------------------------------

def test_codex_command_offers_only_project_tools_and_no_shell(tmp_path):
    settings = ModelSettings("codex", "gpt-test")
    command = agent_runner.codex_command("/bin/codex", settings, ["/py", "-m", "vasp_slurm_agent.mcp_server"],
                                         {"DFT_AGENT_PROJECT": "p1", "PATH": "/usr/bin"}, str(tmp_path), str(tmp_path / "reply.md"))
    options = [command[index + 1] for index, item in enumerate(command) if item == "-c"]
    assert 'mcp_servers.dft.command="/py"' in options
    assert 'mcp_servers.dft.args=["-m", "vasp_slurm_agent.mcp_server"]' in options
    assert any(option.startswith("mcp_servers.dft.env={") and 'DFT_AGENT_PROJECT = "p1"' in option for option in options)
    assert "features.shell_tool=false" in options and 'web_search="disabled"' in options and 'approval_policy="never"' in options
    assert "features.image_generation=false" in options and "features.in_app_browser=false" in options
    assert "features.browser_use=false" in options and "features.browser_use_external=false" in options
    assert "tools.view_image=false" not in options
    # Codex rejects every MCP call under approval_policy=never unless the server's tools are pre-approved.
    assert 'mcp_servers.dft.default_tools_approval_mode="approve"' in options
    assert "--ignore-user-config" in command and "--ephemeral" in command and command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("-m") + 1] == "gpt-test" and command[-1] == "-"


def test_claude_command_disables_builtin_tools_and_allows_only_dft_server():
    command = agent_runner.claude_command("/bin/claude", ModelSettings("claude_cli"), ["/py", "-m", "vasp_slurm_agent.mcp_server"], {"DFT_AGENT_PROJECT": "p1"})
    assert command[command.index("--tools") + 1] == "" and command[command.index("--allowedTools") + 1] == "mcp__dft__*"
    config = json.loads(command[command.index("--mcp-config") + 1])
    assert config == {"mcpServers": {"dft": {"type": "stdio", "command": "/py", "args": ["-m", "vasp_slurm_agent.mcp_server"], "env": {"DFT_AGENT_PROJECT": "p1"}}}}
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands", "-p"):
        assert flag in command
    assert "--model" not in command and command[command.index("--output-format") + 1] == "stream-json"


def test_parse_codex_events_collects_tool_calls_reply_and_errors(tmp_path):
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call", "server": "dft", "tool": "project_overview", "arguments": {}, "status": "completed", "result": {"content": [{"type": "text", "text": "{\"files\": []}"}]}}},
        {"type": "item.completed", "item": {"id": "i2", "type": "mcp_tool_call", "server": "dft", "tool": "read_file", "arguments": {"path": "files/x"}, "status": "failed", "error": {"message": "Private files are excluded"}}},
        {"type": "item.completed", "item": {"id": "i3", "type": "agent_message", "text": "Done. password: hunter2"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    parsed = agent_runner.parse_codex("\n".join(json.dumps(event) for event in events) + "\nnot json\n")
    assert [call["tool"] for call in parsed["tool_calls"]] == ["project_overview", "read_file"]
    assert parsed["tool_calls"][1]["status"] == "failed" and "Private" in parsed["tool_calls"][1]["result"]
    assert parsed["reply"] == "Done. password: hunter2" and parsed["usage"] == {"input_tokens": 10, "output_tokens": 2}
    last = tmp_path / "reply.md"
    last.write_text("Final answer from file")
    assert agent_runner.parse_codex("", str(last))["reply"] == "Final answer from file"
    failed = agent_runner.parse_codex(json.dumps({"type": "turn.failed", "error": {"message": "401 unauthorized"}}))
    assert failed["errors"] and not failed["reply"]


def test_parse_claude_stream_pairs_tool_results():
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u1", "name": "mcp__dft__describe_structure", "input": {"path": "files/Si.cif"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": "{\"formula\": \"Si\"}"}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Silicon, diamond structure."}]}},
        {"type": "result", "subtype": "success", "result": "Silicon, diamond structure.", "usage": {"input_tokens": 5, "output_tokens": 3}},
    ]
    parsed = agent_runner.parse_claude("\n".join(json.dumps(event) for event in events))
    assert parsed["tool_calls"] == [{"tool": "describe_structure", "arguments": '{"path": "files/Si.cif"}', "status": "completed", "result": '{"formula": "Si"}'}]
    assert parsed["reply"] == "Silicon, diamond structure." and not parsed["errors"]


def test_run_turn_records_both_messages_and_keeps_secrets_out(project, monkeypatch, tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    project.registry._patch_settings(project.model, cli_path=str(executable))
    seen = {}

    def run(command, **kwargs):
        seen["command"], seen["env"], seen["input"] = command, kwargs["env"], kwargs["input"]
        last = command[command.index("-o") + 1]
        Path(last).write_text("I read the project. 工作流尚未创建。", encoding="utf-8")
        events = [{"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "dft", "tool": "project_overview", "arguments": {}, "status": "completed", "result": "ok"}},
                  {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 5}}]
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(e) for e in events), stderr="")
    monkeypatch.setattr(agent_runner.subprocess, "run", run)
    monkeypatch.setenv("OPENAI_API_KEY", "never-forward-this")
    result = agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "项目里有什么？", project.model)
    assert result["error"] is None and result["reply"].startswith("I read the project")
    assert seen["command"][0] == str(executable) and "never-forward-this" not in json.dumps(seen["env"])
    assert "User's new message:\n项目里有什么？" in seen["input"] and agent_runner.SYSTEM in seen["input"]
    env_option = next(item for item in seen["command"] if item.startswith("mcp_servers.dft.env="))
    assert f'DFT_AGENT_PROJECT = "{project.pid}"' in env_option and "OPENAI_API_KEY" not in env_option
    messages = project.workspace.get_session(project.pid, project.sid)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["metadata"]["mode"] == "agent" and messages[1]["metadata"]["tool_calls"][0]["tool"] == "project_overview"
    assert messages[1]["metadata"]["model_usage"][0]["input_tokens"] == 20
    with pytest.raises(ValueError, match="secure credential"):
        agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "my password is hunter2", project.model)
    api = project.registry.create("model", "API", {"provider": "responses", "model": "m"})
    project.registry.bind(project.pid, api["id"])
    with pytest.raises(ValueError, match="Agent mode"):
        agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "hi", api["id"])


def test_run_turn_reports_cli_failure_without_echoing_output(project, monkeypatch, tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o700)
    project.registry._patch_settings(project.model, cli_path=str(executable))
    monkeypatch.setattr(agent_runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=json.dumps({"type": "turn.failed", "error": {"message": "429 rate limit private-marker"}}), stderr=""))
    result = agent_runner.run_turn(project.workspace, project.registry, project.pid, project.sid, "hello", project.model)
    assert result["error"] and "rate or usage limit" in result["reply"] and "private-marker" not in result["reply"]
    assert project.workspace.get_session(project.pid, project.sid)["messages"][-1]["metadata"]["error"] is True


# --- managed QE installation ----------------------------------------------------

class ScriptedTransport:
    def __init__(self, answers):
        self.answers, self.commands = list(answers), []

    def run(self, command, timeout=60):
        self.commands.append(command)
        return Result(0, self.answers.pop(0) if self.answers else "")

    def close(self):
        pass


def test_detect_toolchain_prefers_path_then_tries_module_combinations(tmp_path):
    config = SimpleNamespace(setup_commands=[], remote_root="/scratch/u", partition="cpu", account="")
    ready = ScriptedTransport(["__TC_PATH__\nmpif90=/usr/bin/mpif90\nmake=/usr/bin/make\ngit=/usr/bin/git\n__TC_MODULES__\n__TC_END__\n"])
    result = qe_setup.detect_toolchain(config, ready)
    assert result["ready"] and result["setup_commands"] == [] and result["mpi_fortran"] == "/usr/bin/mpif90"
    listing = "__TC_PATH__\nmake=/usr/bin/make\ngit=/usr/bin/git\n__TC_MODULES__\ngcc/12.2\nintel/2023\nopenmpi/4.1.5\nimpi/2021\n__TC_END__\n"
    tried = ScriptedTransport([listing, "LOAD=failed\n", "MPI=/opt/impi/bin/mpiifort\nMAKE=yes\nGIT=yes\n"])
    result = qe_setup.detect_toolchain(config, tried)
    assert result["ready"] and result["setup_commands"] == ["module load impi/2021"] and result["mpi_fortran"] == "/opt/impi/bin/mpiifort"
    assert [item["modules"] for item in result["modules_tried"]] == [["openmpi/4.1.5"], ["impi/2021"]]
    assert "module purge" not in tried.commands[1] and "module load openmpi/4.1.5" in tried.commands[1]
    nothing = ScriptedTransport([listing] + ["LOAD=failed\n"] * 20)
    result = qe_setup.detect_toolchain(config, nothing)
    assert not result["ready"] and "No MPI Fortran compiler" in result["notes"][-1]
    kept = SimpleNamespace(setup_commands=["module load site"], remote_root="/scratch/u", partition="cpu", account="")
    result = qe_setup.detect_toolchain(kept, ScriptedTransport(["__TC_PATH__\nmpifort=/x/mpifort\nmake=/usr/bin/make\n__TC_MODULES__\n__TC_END__\n"]))
    assert result["setup_commands"] == ["module load site"] and any("Git was not found" in note for note in result["notes"])


@pytest.fixture
def connected_cluster(project, monkeypatch):
    registry = project.registry
    hpc = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    registry.bind(project.pid, hpc["id"])
    probe = {"auth_status": "available", "connection_status": "connected", "detail": "ok", "schedulers": ["slurm"],
             "discovery": {"suggested": {"partition": "cpu", "account": "proj1"}, "notes": [], "vasp_found": False, "qe_found": False}}
    registry._save_probe(hpc["id"], registry.get(hpc["id"])["revision"], probe)
    monkeypatch.setattr(ConnectionRegistry, "_qe_transport", lambda self, item: SimpleNamespace(close=lambda: None))
    return hpc["id"]


def test_qe_install_plan_start_status_create_environment(project, connected_cluster, monkeypatch):
    registry = project.registry
    monkeypatch.setattr(qe_setup, "detect_toolchain", lambda config, transport=None, **kwargs: {
        "ready": True, "setup_commands": ["module load gcc openmpi"], "mpi_fortran": "/opt/bin/mpif90", "notes": ["found"], "modules_tried": [], "tools": {}})
    planned = registry.qe_install_plan(connected_cluster, build_cpus=4)
    assert planned["status"] == "planned" and planned["partition"] == "cpu" and planned["account"] == "proj1"
    assert planned["setup_commands"] == ["module load gcc openmpi"] and planned["prefix"].startswith("/scratch/researcher/software/qe-")
    stored = registry.get(connected_cluster)["probe"]["qe_install"]
    assert stored["plan"]["build_jobs"] == 4 and stored["plan"]["setup_commands"] == ["module load gcc openmpi"]
    with pytest.raises(ValueError, match="Plan the installation"):
        registry.qe_install_start(connected_cluster, "other-plan")
    started = {}

    def start_install(config, plan, transport=None, **kwargs):
        started["config"], started["plan"] = config, plan
        return {"status": "submitted", "job_id": "4242", "work_dir": plan["work_dir"], "id": plan["id"], "scheduler": "slurm"}
    monkeypatch.setattr(qe_setup, "start_install", start_install)
    result = registry.qe_install_start(connected_cluster, planned["plan_id"])
    assert result["job_id"] == "4242" and started["config"].setup_commands == ["module load gcc openmpi"] and started["config"].engine == "qe"
    assert started["config"].connection_id == connected_cluster and started["config"].remote_root == "/scratch/researcher"
    monkeypatch.setattr(qe_setup, "poll_install", lambda config, receipt, transport=None: {"status": "running", "scheduler_state": "RUNNING"})
    assert registry.qe_install_status(connected_cluster)["status"] == "running"
    assert registry.environments(connected_cluster) == []
    monkeypatch.setattr(qe_setup, "poll_install", lambda config, receipt, transport=None: {"status": "installed", "bin_dir": planned["prefix"] + "/bin"})
    status = registry.qe_install_status(connected_cluster)
    assert status["status"] == "installed" and status["environment_id"]
    environment = registry.environment(status["environment_id"])
    assert environment["engine"] == "qe" and environment["config"]["qe_bin_dir"] == planned["prefix"] + "/bin"
    assert environment["config"]["setup_commands"] == ["module load gcc openmpi"] and environment["config"]["qe_pseudo_root"] == "/scratch/researcher/pseudos"
    # A later status call does not create a second environment.
    assert registry.qe_install_status(connected_cluster)["environment_id"] == status["environment_id"]
    assert len(registry.environments(connected_cluster)) == 1


def test_qe_install_plan_requires_toolchain_and_partition(project, connected_cluster, monkeypatch):
    registry = project.registry
    monkeypatch.setattr(qe_setup, "detect_toolchain", lambda config, transport=None, **kwargs: {
        "ready": False, "setup_commands": list(config.setup_commands), "mpi_fortran": None, "notes": ["No MPI Fortran compiler was found"], "modules_tried": [], "tools": {}})
    planned = registry.qe_install_plan(connected_cluster, setup_commands=["module load mystery"])
    assert not planned["toolchain_ready"] and planned["setup_commands"] == ["module load mystery"]
    with pytest.raises(ValueError, match="MPI Fortran"):
        registry.qe_install_start(connected_cluster, planned["plan_id"])
    with pytest.raises(ValueError, match="single lines"):
        registry.qe_install_plan(connected_cluster, setup_commands=["module load a\nrm -rf /"])
    with pytest.raises(ValueError, match="secure credential"):
        registry.qe_install_plan(connected_cluster, setup_commands=["export API_KEY=hidden"])
    registry._store_probe(connected_cluster, "discovery", {"suggested": {}})
    with pytest.raises(ValueError, match="partition"):
        registry.qe_install_plan(connected_cluster)


def test_install_pseudopotentials_records_mapping(project, connected_cluster, monkeypatch):
    registry = project.registry
    environment = registry.create_environment(connected_cluster, "QE", "qe", {"partition": "cpu", "qe_command": "srun pw.x", "qe_pseudo_root": "/scratch/researcher/pseudos"})
    seen = {}

    def ensure(config, elements, transport=None, **kwargs):
        seen["elements"], seen["library"], seen["root"] = elements, kwargs.get("library"), config.qe_pseudo_root
        return {"library": "SSSP 1.3.0 PBE efficiency", "elements": {element: {"filename": f"{element}.upf", "sha256": "0" * 64, "cutoff_wfc": 30, "cutoff_rho": 240}
                for element in elements},
                "config_pseudopotentials": {element: {"filename": f"{element}.upf", "sha256": "0" * 64, "ecutwfc": 30, "ecutrho": 240, "element": element,
                                                      "functional": "PBE", "pseudo_type": "US", "relativistic": "scalar", "fully_relativistic": False,
                                                      "valence_electrons": {"Si": 4.0, "O": 6.0}[element]} for element in elements}}
    monkeypatch.setattr(qe_setup, "ensure_pseudos", ensure)
    result = registry.install_pseudopotentials(environment["id"], ["Si", "O", "Si"])
    assert seen["elements"] == ["Si", "O"] and seen["library"] == "sssp" and seen["root"] == "/scratch/researcher/pseudos"
    mapping = registry.environment(environment["id"])["config"]["qe_pseudopotentials"]
    assert mapping["Si"]["filename"] == "Si.upf" and mapping["O"]["ecutwfc"] == 30
    # The UPF valence is recorded so band and NSCF stages can add empty bands.
    assert mapping["O"]["valence_electrons"] == 6.0 and mapping["Si"]["fully_relativistic"] is False
    assert result["elements"]["O"]["filename"] == "O.upf"
    with pytest.raises(ValueError, match="element symbols"):
        registry.install_pseudopotentials(environment["id"], ["si"])
    vasp = registry.create_environment(connected_cluster, "VASP", "vasp", {"partition": "cpu", "vasp_command": "srun vasp_std", "potcar_root": "/licensed"})
    with pytest.raises(ValueError, match="Quantum ESPRESSO environment"):
        registry.install_pseudopotentials(vasp["id"], ["Si"])
    with pytest.raises(ValueError, match="Keep host"):
        registry.update_environment(environment["id"], {**environment["config"], "host": "other"})


# --- interface ------------------------------------------------------------------

def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_agent_mode_switch_runs_cli_turn_and_shows_actions(monkeypatch):
    monkeypatch.delenv("DFT_AGENT_UI")
    workspace = Workspace()
    registry = ConnectionRegistry(workspace)
    pid = workspace.create_project("Silicon")["id"]
    sid = workspace.create_session(pid)["id"]
    model = registry.create("model", "Codex", {"provider": "codex"})
    registry.bind(pid, model["id"])
    calls = []

    def run_turn(workspace_, registry_, project_id, session_id, message, model_id, **kwargs):
        calls.append(message)
        session = workspace_.get_session(project_id, session_id)
        session = workspace_.append_message(project_id, session_id, "user", message, metadata={"mode": "agent"}, expected_revision=session["revision"])
        session = workspace_.append_message(project_id, session_id, "assistant", "Silicon has two atoms.", metadata={
            "mode": "agent", "tool_calls": [{"tool": "describe_structure", "arguments": '{"path": "files/Si.cif"}', "status": "completed", "result": "Si"}]},
            expected_revision=session["revision"])
        return {"reply": "Silicon has two atoms.", "tool_calls": [], "usage": None, "error": None, "session": session}
    monkeypatch.setattr(agent_runner, "run_turn", run_turn)
    monkeypatch.setattr(project_files, "write_file", project_files.write_file)
    ui = AppTest.from_string("from vasp_slurm_agent.app import main\nmain()", default_timeout=30).run()
    assert widget(ui.pills, "Mode").value == "agent"
    widget(ui.pills, "Mode").set_value("guided").run()
    widget(ui.pills, "Mode").set_value("agent").run()
    ui.chat_input[0].set_value("Describe the structure").run()
    assert not ui.exception and calls == ["Describe the structure"]
    assert any(item.label.startswith("Agent actions") for item in ui.expander)
    assert "Silicon has two atoms." in [item.value for item in ui.markdown]
    api = registry.create("model", "API", {"provider": "responses", "model": "m"})
    other = workspace.create_project("Other")["id"]
    registry.bind(other, api["id"])
    ui.run()
    ui.button(key="project_open_" + other).click().run()
    assert not ui.exception and not any(item.label == "Mode" for item in ui.pills)


def test_cluster_setup_offers_qe_installation_when_qe_is_missing(monkeypatch, keychain):
    registry = ConnectionRegistry()
    hpc = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    registry._save_probe(hpc["id"], registry.get(hpc["id"])["revision"], {
        "auth_status": "available", "connection_status": "connected", "detail": "ok", "schedulers": ["slurm"],
        "discovery": {"suggested": {"partition": "cpu"}, "notes": [], "vasp_found": False, "qe_found": False}})
    planned = {"plan_id": "abc", "version": "7.4.1", "prefix": "/scratch/researcher/software/qe-7.4.1-x", "partition": "cpu", "account": "",
               "build_cpus": 8, "walltime": "02:00:00", "setup_commands": ["module load gcc openmpi"], "toolchain_ready": True, "notes": ["found"], "status": "planned"}

    def plan(self, identifier, **kwargs):
        self._store_probe(identifier, "qe_install", {"status": "planned", "plan": {"version": "7.4.1", "build_jobs": 8, "walltime": "02:00:00", "prefix": planned["prefix"], "id": "abc"},
                                                     "plan_id": "abc", "partition": kwargs.get("partition") or "cpu", "account": "",
                                                     "setup_commands": ["module load gcc openmpi"], "toolchain": {"ready": True, "notes": ["found"], "mpi_fortran": "/x", "modules_tried": []}})
        return planned
    started = Mock(return_value={"status": "submitted", "job_id": "77"})
    monkeypatch.setattr(ConnectionRegistry, "qe_install_plan", plan)
    monkeypatch.setattr(ConnectionRegistry, "qe_install_start", started)
    view = AppTest.from_string(f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import chat_panel
st.session_state.setdefault('setup_chat_global_kind', 'hpc')
st.session_state.setdefault('setup_chat_global_hpc_state', {{'schema_version': 2, 'kind': 'hpc', 'record_id': {hpc['id']!r}, 'detect': None, 'needs_code': None, 'attempted': True, 'discovered': True, 'pending_host': None, 'history': []}})
chat_panel(ConnectionRegistry(), None)
""", default_timeout=20).run()
    assert not view.exception
    widget(view.button, "Plan the build").click().run()
    assert not view.exception and any("module load gcc openmpi" in item.value for item in view.code)
    assert widget(view.button, "Start the build").disabled
    widget(view.checkbox, "Use this account and allocation to build QE.").check().run()
    widget(view.button, "Start the build").click().run()
    assert not view.exception
    started.assert_called_once()
    assert started.call_args.args == (hpc["id"], "abc")
