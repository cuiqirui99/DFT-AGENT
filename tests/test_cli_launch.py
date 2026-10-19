"""Windows npm launchers preserve structured arguments without a command shell."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vasp_slurm_agent import agent, cli_tools


def npm_cli(tmp_path, name, entry="bin/cli.js", *, local=False):
    package = "@openai/codex" if name == "codex" else "@anthropic-ai/claude-code"
    prefix = tmp_path / "CLI tools 中文"
    launcher_folder = prefix / "node_modules" / ".bin" if local else prefix
    launcher_folder.mkdir(parents=True)
    root = prefix / "node_modules" / package
    root.mkdir(parents=True)
    (root / "package.json").write_text(json.dumps({"name": package, "bin": {name: entry}}), encoding="utf-8")
    target = root / entry
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    launcher = launcher_folder / (name + ".cmd")
    launcher.write_text("@echo off\r\nexit /b 9\r\n")  # It must never execute.
    launcher.chmod(0o700)
    return launcher, target


@pytest.mark.parametrize("name,local", [("codex", False), ("codex", True), ("claude", False), ("claude", True)])
def test_known_npm_entrypoint_resolves_without_a_shell(tmp_path, monkeypatch, name, local):
    launcher, target = npm_cli(tmp_path, name, local=local)
    monkeypatch.setattr(cli_tools.shutil, "which", lambda name: "/tools/node.exe" if name == "node" else None)
    args = [str(launcher), "--settings", '{"value":"%PATH% & 中文"}', ""]
    assert cli_tools.command(args, windows=True) == ["/tools/node.exe", str(target.resolve()), *args[1:]]
    assert cli_tools.command(args, windows=False) == args


def test_native_cli_and_current_claude_npm_executable_remain_native(tmp_path, monkeypatch):
    launcher, target = npm_cli(tmp_path, "claude", "bin/claude.exe")
    monkeypatch.setattr(cli_tools.shutil, "which", lambda name: pytest.fail("Native Claude does not need Node"))
    assert cli_tools.command([str(launcher), "auth", "status"], windows=True) == [str(target.resolve()), "auth", "status"]
    direct = [str(target), "--version"]
    assert cli_tools.command(direct, windows=True) == direct


@pytest.mark.parametrize("invalid", ["missing", "wrong_package", "outside"])
def test_known_shim_cannot_fall_back_to_shell_on_broken_install(tmp_path, invalid):
    launcher, target = npm_cli(tmp_path, "codex")
    metadata = target.parent.parent / "package.json"
    if invalid == "missing":
        metadata.unlink()
    elif invalid == "wrong_package":
        metadata.write_text(json.dumps({"name": "unrelated", "bin": {"codex": "bin/cli.js"}}))
    else:
        metadata.write_text(json.dumps({"name": "@openai/codex", "bin": {"codex": "../../outside.js"}}))
        (metadata.parent / "../../outside.js").write_text("")
    with pytest.raises(OSError, match="incomplete"):
        cli_tools.command([str(launcher), "--version"], windows=True)


def test_windows_npm_arguments_and_unicode_prompt_round_trip(tmp_path, monkeypatch):
    # Node is part of npm CLI installations and is available on the CI runners.
    assert shutil.which("node"), "This npm integration test needs Node.js on PATH."
    launcher, target = npm_cli(tmp_path, "claude")
    target.write_text("""
const fs = require('fs');
const args = process.argv.slice(2);
const schema = JSON.parse(args[args.indexOf('--json-schema') + 1]);
if (schema.type !== 'object') process.exit(2);
const input = fs.readFileSync(0, 'utf8');
console.log(JSON.stringify({type:'result', subtype:'success', is_error:false,
  structured_output:{answer:input}, usage:{input_tokens:1, output_tokens:1}}));
""", encoding="utf-8")
    command = cli_tools.command
    monkeypatch.setattr(cli_tools, "command", lambda args: command(args, windows=True))
    payload = '计算含 SOC 的能带。 Preserve "quotes", %PATH%, !name!, & | < > ^ and C:\\Users\\研究者\\file.'
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    result = agent._claude_cli_request(payload, schema, agent.ModelSettings("claude_cli", cli_path=str(launcher)),
                                       instructions='Return the answer; keep "quoted" values, %PATH% and 中文.')
    assert json.loads(result) == {"answer": payload}
    # Test the same direct invocation with large MCP/TOML arguments and empty values.
    target.write_text("process.stdout.write(JSON.stringify(process.argv.slice(2)));", encoding="utf-8")
    arguments = ['{"path":"C:\\\\Users\\\\研究者","items":[]}', '', 'approval_policy="never"', '%PATH% & !x!', 'first line\nsecond line', 'x' * 10000]
    response = subprocess.run(command([str(launcher), *arguments], windows=True), capture_output=True,
                              text=True, encoding="utf-8", check=True, timeout=15)
    assert json.loads(response.stdout) == arguments


@pytest.mark.parametrize("provider", ["codex", "claude_cli"])
def test_cli_keeps_windows_system_root_and_utf8_but_not_credentials(tmp_path, monkeypatch, provider):
    from types import SimpleNamespace
    from vasp_slurm_agent import agent_runner
    executable = tmp_path / ("codex.exe" if provider == "codex" else "claude.exe")
    executable.write_bytes(b"fixture")
    executable.chmod(0o700)
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-forward")
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "must-not-forward")
    def run(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["env"]["SYSTEMROOT"] == "C:\\Windows"
        assert "must-not-forward" not in kwargs["env"].values()
        if provider == "codex":
            Path(command[command.index("-o") + 1]).write_text('{"answer":"Ready"}', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"type":"result", "subtype":"success",
            "is_error":False, "structured_output":{"answer":"Ready"}}))
    monkeypatch.setattr(agent.subprocess, "run", run)
    schema = {"type":"object", "properties":{"answer":{"type":"string"}}, "required":["answer"], "additionalProperties":False}
    result = agent._request_structured('{"question":"Describe this structure"}', schema,
        agent.ModelSettings(provider, cli_path=str(executable)), instructions="Return JSON")
    assert json.loads(result) == {"answer":"Ready"}
    assert "SYSTEMROOT" in agent_runner._PASSTHROUGH
