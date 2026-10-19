"""Native terminal launches are mocked; these tests never initiate SSH login."""

import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vasp_slurm_agent import ssh_auth
from vasp_slurm_agent.connections import ConnectionRegistry


@pytest.fixture
def connection(tmp_path):
    registry = ConnectionRegistry(tmp_path / "home")
    local = Path(tmp_path.anchor) / "tmp"
    record = registry.create("hpc", "HPC", {"host": "hpc.invalid", "user": "user@group", "port": 2222,
                                           "auth_mode": "interactive", "remote_base": "/scratch/user",
                                           "ssh_control_path": (local / 'dft "socket".sock').as_posix(),
                                           "identity_file": (local / "key's file $(unused)").as_posix(),
                                           "jump_host": "jump@gateway.invalid:2200"})
    return registry, record


def binaries(monkeypatch, names):
    monkeypatch.setattr(ssh_auth.shutil, "which", lambda name: "/mock/" + name if name in names else None)


def test_macos_command_quotes_paths_and_keeps_prompts_in_terminal(connection, monkeypatch):
    registry, record = connection
    binaries(monkeypatch, {"ssh", "osascript"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("SSH_ASKPASS", "private-helper")
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "not-for-terminal")
    monkeypatch.setenv("_DFT_AGENT_ASKPASS_CODE", "private-code")
    launched = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(ssh_auth.subprocess, "run", launched)
    result = ssh_auth.launch(registry, record["id"])
    assert result["status"] == "launched" and result["reusable"]
    argv, = launched.call_args.args
    literal = argv[2].removeprefix('tell application "Terminal" to do script ')
    visible = shlex.split(json.loads(literal))
    assert visible[:5] == ["/usr/bin/env", "-u", "SSH_ASKPASS", "-u", "SSH_ASKPASS_REQUIRE"]
    assert visible[15:] == ssh_auth.command(registry, record["id"], system="Darwin")
    assert record["settings"]["identity_file"] in visible
    assert "jump@gateway.invalid:2200" in visible
    control = record["settings"]["ssh_control_path"].replace('"', '\\"')
    assert f'ControlPath="{control}"' in visible
    assert "SSH_ASKPASS" not in launched.call_args.kwargs["env"]
    assert "DFT_AGENT_SSH_PASSWORD" not in launched.call_args.kwargs["env"]
    assert "_DFT_AGENT_ASKPASS_CODE" not in launched.call_args.kwargs["env"]
    assert "_DFT_AGENT_ASKPASS_CODE" in visible[1:15]
    assert registry.get(record["id"])["connection_status"] == "untested"
    assert "not-for-terminal" not in json.dumps(result)


def test_windows_terminal_is_visible_and_does_not_claim_session_reuse(connection, monkeypatch):
    registry, record = connection
    binaries(monkeypatch, {"ssh", "powershell"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: "Windows")
    launched = Mock()
    monkeypatch.setattr(ssh_auth.subprocess, "Popen", launched)
    result = ssh_auth.launch(registry, record["id"])
    assert result["status"] == "launched" and not result["reusable"]
    argv, = launched.call_args.args
    assert argv[:5] == ["/mock/powershell", "-NoLogo", "-NoProfile", "-NoExit", "-Command"]
    identity = record["settings"]["identity_file"].replace("'", "''")
    assert f"'{identity}'" in argv[-1]
    assert "ControlPath" not in argv[-1]
    assert launched.call_args.kwargs["creationflags"]
    assert not {"stdin", "stdout", "stderr"} & launched.call_args.kwargs.keys()
    assert "Connections" in result["message"]


def test_linux_terminal_receives_argv_without_shell(connection, monkeypatch):
    registry, record = connection
    binaries(monkeypatch, {"ssh", "xterm"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DISPLAY", ":99")
    launched = Mock()
    monkeypatch.setattr(ssh_auth.subprocess, "Popen", launched)
    assert ssh_auth.launch(registry, record["id"])["status"] == "launched"
    argv, = launched.call_args.args
    assert argv[:7] == ["/mock/xterm", "-e", "env", "-u", "SSH_ASKPASS", "-u", "SSH_ASKPASS_REQUIRE"]
    assert argv[17:] == ssh_auth.command(registry, record["id"], system="Linux")
    assert not launched.call_args.kwargs.get("shell")


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows", "Other"])
def test_missing_terminal_returns_manual_command(connection, monkeypatch, system):
    registry, record = connection
    binaries(monkeypatch, {"ssh"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: system)
    launch = Mock(side_effect=AssertionError("No launcher"))
    monkeypatch.setattr(ssh_auth.subprocess, "Popen", launch)
    monkeypatch.setattr(ssh_auth.subprocess, "run", launch)
    result = ssh_auth.launch(registry, record["id"])
    assert result["status"] == "manual" and "ssh" in result["command"]
    launch.assert_not_called()


def test_denied_macos_terminal_permission_returns_manual_fallback(connection, monkeypatch):
    registry, record = connection
    binaries(monkeypatch, {"ssh", "osascript"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ssh_auth.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1)))
    assert ssh_auth.launch(registry, record["id"])["status"] == "manual"


def test_generated_socket_is_private_and_reused(tmp_path, monkeypatch):
    if os.name == "nt":
        from vasp_slurm_agent import connections
        mkdtemp = connections.tempfile.mkdtemp
        monkeypatch.setattr(connections.tempfile, "mkdtemp", lambda **kwargs: mkdtemp(prefix=kwargs["prefix"]))
    registry = ConnectionRegistry(tmp_path / "home")
    record = registry.create("hpc", "HPC", {"host": "hpc.invalid", "user": "user", "auth_mode": "interactive"})
    binaries(monkeypatch, {"ssh"})
    first = ssh_auth.command(registry, record["id"], system="Darwin")
    socket = Path(registry.get(record["id"])["settings"]["ssh_control_path"])
    try:
        assert socket.parent.is_dir()
        if os.name != "nt":
            assert socket.parent.stat().st_mode & 0o777 == 0o700
        assert ssh_auth.command(registry, record["id"], system="Darwin") == first
    finally:
        if socket.parent.name.startswith("dft-login-"):
            socket.parent.rmdir()


def test_revoked_connection_cannot_launch_even_after_build(connection, monkeypatch):
    registry, record = connection
    binaries(monkeypatch, {"ssh", "xterm"})
    monkeypatch.setattr(ssh_auth.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DISPLAY", ":99")
    launch = Mock()
    monkeypatch.setattr(ssh_auth.subprocess, "Popen", launch)
    original = ssh_auth.command

    def command_then_revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        registry.revoke(record["id"])
        return result

    monkeypatch.setattr(ssh_auth, "command", command_then_revoke)
    with pytest.raises(ValueError, match="active HPC"):
        ssh_auth.launch(registry, record["id"])
    launch.assert_not_called()


@pytest.mark.parametrize("field,value", [("host", "hpc;touch${IFS}/tmp/file"), ("user", "user`id`"), ("jump_host", "jump$(id)")])
def test_shell_syntax_in_endpoint_names_is_rejected(connection, field, value):
    registry, record = connection
    registry.update(record["id"], settings={field: value})
    with pytest.raises(ValueError, match="shell characters"):
        ssh_auth.command(registry, record["id"], system="Darwin")
