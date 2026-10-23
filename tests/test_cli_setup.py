"""Installing and signing in to the CLIs through a terminal window the app opens."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

from vasp_slurm_agent import cli_setup, cli_tools

MESSAGES = {key: f"<{key}>" for key in cli_setup.MESSAGES}
MISSING = {"provider": "codex", "installed": False, "path": None, "logged_in": None}
SIGNED_OUT = {"provider": "claude_cli", "installed": True, "path": "/opt/tools/claude", "logged_in": False}
SIGNED_IN = {"provider": "codex", "installed": True, "path": "/opt/tools/codex", "logged_in": True}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_INSTALL_DIR", raising=False)
    return tmp_path


def test_next_step_follows_install_sign_in_connect():
    assert [cli_setup.next_step(item) for item in (MISSING, SIGNED_OUT, SIGNED_IN)] == ["install", "sign_in", "connect"]
    assert cli_setup.next_step({**SIGNED_OUT, "logged_in": None}) == "sign_in"
    assert cli_setup.next_step({**SIGNED_OUT, "version": None}) == "install"


def test_commands_use_the_official_installers_without_node(home):
    unix = cli_setup.commands("codex", MISSING, "macos", which=lambda name: None)
    assert unix == ["curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh", "~/.local/bin/codex login"]
    claude = cli_setup.commands("claude_cli", {**MISSING, "provider": "claude_cli"}, "linux", which=lambda name: None)
    assert claude == ["curl -fsSL https://claude.ai/install.sh | bash", "~/.local/bin/claude auth login"]
    assert not any("npm" in line for line in unix + claude)
    # A signed-out CLI on PATH needs only its sign-in, by its plain name.
    assert cli_setup.commands("claude_cli", SIGNED_OUT, "macos", which=lambda name: "/opt/tools/claude") == ["claude auth login"]
    assert cli_setup.commands("claude_cli", SIGNED_OUT, "macos", which=lambda name: None) == ["/opt/tools/claude auth login"]
    # An update reinstalls and keeps the existing sign-in.
    assert cli_setup.commands("codex", SIGNED_IN, "macos", update=True) == [cli_setup.INSTALL[("codex", "unix")]]


def test_windows_commands_point_at_the_installers_folders(home, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Ada Lovelace\AppData\Local")
    lines = cli_setup.commands("codex", MISSING, "windows", which=lambda name: None)
    assert lines[0] == "$env:CODEX_NON_INTERACTIVE = '1'; irm https://chatgpt.com/codex/install.ps1 | iex"
    # The path holds a space, so PowerShell needs the call operator and quotes.
    assert lines[1].startswith("& 'C:") and "Ada Lovelace" in lines[1] and lines[1].endswith("codex.exe' login")
    assert cli_setup.commands("claude_cli", {**MISSING, "provider": "claude_cli"}, "windows", which=lambda name: None)[0] == \
        "irm https://claude.ai/install.ps1 | iex"


def test_macos_opens_terminal_with_a_script_that_installs_then_signs_in(home):
    launched = []
    script = cli_setup.open_terminal("codex", MISSING, MESSAGES, name="macos", launch=lambda arguments, **kwargs: launched.append(arguments))
    assert launched == [["open", "-a", "Terminal", str(script)]] and script.suffix == ".command"
    if os.name == "posix":
        assert script.stat().st_mode & 0o777 == 0o700
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[:3] == ["#!/bin/bash", 'rm -f -- "$0"; rmdir -- "$(dirname -- "$0")" 2>/dev/null', "set -o pipefail"]
    assert "if ! curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh; then" in lines
    assert f"if ! {shlex.quote(str(home / '.local' / 'bin' / 'codex'))} login; then" in lines
    # Signing in starts only after the install worked and the program is where the installer puts it.
    assert text.index("install.sh") < text.index("[ ! -x") < text.index(shlex.quote(str(home / ".local" / "bin" / "codex")) + " login")
    for key in cli_setup.MESSAGES:
        assert f"<{key}>" in text


def test_signed_out_cli_only_signs_in_and_quotes_its_path(home):
    launched = []
    item = {**SIGNED_OUT, "path": "/Applications/My Tools/claude"}
    script = cli_setup.open_terminal("claude_cli", item, MESSAGES, name="macos", launch=lambda arguments, **kwargs: launched.append(arguments))
    text = script.read_text(encoding="utf-8")
    assert "install.sh" not in text and "'/Applications/My Tools/claude' auth login" in text


def test_update_reinstalls_without_signing_in_again(home):
    script = cli_setup.open_terminal("codex", SIGNED_IN, MESSAGES, update=True, name="linux",
                                     launch=lambda arguments, **kwargs: None, which=lambda name: "/usr/bin/" + name)
    text = script.read_text(encoding="utf-8")
    assert "install.sh" in text and "login" not in text


@pytest.mark.skipif(os.name != "posix", reason="Runs the Unix installer script in bash.")
def test_the_script_runs_and_stops_when_the_install_fails(home, tmp_path):
    # A real bash run: a failing install leaves the sign-in out.
    item = {**MISSING}
    script = cli_setup.open_terminal("codex", item, MESSAGES, name="linux", launch=lambda arguments, **kwargs: None,
                                     which=lambda name: "/usr/bin/xterm" if name == "xterm" else None)
    text = script.read_text(encoding="utf-8").replace(cli_setup.INSTALL[("codex", "unix")], "false | cat")
    script.write_text(text, encoding="utf-8")
    result = subprocess.run(["bash", str(script)], input="\n", capture_output=True, text=True, timeout=30,
                            env={"PATH": "/usr/bin:/bin", "TERM": "dumb"})
    assert result.returncode == 1 and "<failed>" in result.stdout and "<signing_in>" not in result.stdout
    assert not script.exists()


def test_windows_opens_powershell_in_a_new_console(home, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(home / "Local"))
    launched = []
    script = cli_setup.open_terminal("codex", MISSING, MESSAGES, name="windows",
                                     launch=lambda arguments, **kwargs: launched.append((arguments, kwargs)))
    arguments, options = launched[0]
    assert arguments == ["powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    assert options == {"creationflags": getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)}
    data = script.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and b"\r\n" in data and b"\r\r\n" not in data
    text = data.decode("utf-8-sig")
    assert "$ErrorActionPreference = 'Stop'" in text
    assert cli_setup.INSTALL[("codex", "windows")] in text
    assert "& $program login" in text and str(home / "Local" / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe") in text


def test_linux_uses_the_first_terminal_found_or_explains(home):
    launched = []
    available = {"konsole", "xterm"}
    script = cli_setup.open_terminal("claude_cli", SIGNED_OUT, MESSAGES, name="linux", launch=lambda arguments, **kwargs: launched.append(arguments),
                                     which=lambda name: f"/usr/bin/{name}" if name in available else None)
    assert launched == [["konsole", "-e", "bash", str(script)]]
    with pytest.raises(OSError, match="No terminal program"):
        cli_setup.open_terminal("claude_cli", SIGNED_OUT, MESSAGES, name="linux", launch=lambda arguments, **kwargs: None, which=lambda name: None)


def test_detection_looks_where_the_windows_installer_puts_codex(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_tools.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    locations = [str(path) for path in cli_tools._known_locations("codex")]
    assert str(tmp_path / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe") in locations


@pytest.mark.parametrize("name,binary", [("macos", "codex"), ("windows", "codex.exe")])
def test_custom_codex_install_folder_is_used_and_discovered(home, monkeypatch, name, binary):
    folder = home / "My CLI's" / "bin"
    monkeypatch.setenv("CODEX_INSTALL_DIR", str(folder))
    monkeypatch.setattr(cli_tools.sys, "platform", "win32" if name == "windows" else "darwin")
    assert cli_setup.installed_path("codex", name) == str(folder / binary)
    assert folder / binary in cli_tools._known_locations("codex")
    # Claude's installer has a separate fixed destination.
    assert cli_setup.installed_path("claude_cli", name) != str(folder / binary)
    script = cli_setup.open_terminal("codex", MISSING, MESSAGES, name=name, launch=lambda *args, **kwargs: None)
    content = script.read_text(encoding="utf-8-sig")
    assignment = (f"$env:CODEX_INSTALL_DIR = {cli_setup._quoted(str(folder))}" if name == "windows"
                  else f"export CODEX_INSTALL_DIR={shlex.quote(str(folder))}")
    assert assignment in content


def test_unusable_cli_gets_a_native_reinstall(home):
    item = {**SIGNED_OUT, "version": None}
    lines = cli_setup.commands("claude_cli", item, "macos", which=lambda name: item["path"])
    assert lines == [cli_setup.INSTALL[("claude_cli", "unix")], "~/.local/bin/claude auth login"]


@pytest.mark.parametrize("name", ["macos", "windows", "linux"])
def test_launch_failure_removes_the_unused_script(home, monkeypatch, name):
    folder = home / "launch"
    folder.mkdir()
    monkeypatch.setattr(cli_setup.tempfile, "mkdtemp", lambda **kwargs: str(folder))

    def fail(*args, **kwargs):
        raise OSError("Terminal could not start")

    with pytest.raises(OSError, match="Terminal could not start"):
        cli_setup.open_terminal("codex", MISSING, MESSAGES, name=name, launch=fail, which=lambda _: "terminal")
    assert not folder.exists()


@pytest.mark.skipif(os.name != "posix", reason="Runs native Unix sign-in scripts in bash.")
@pytest.mark.parametrize("provider", cli_setup.PROVIDERS)
def test_unix_sign_in_failure_cancel_and_retry(home, provider):
    program = home / "Ada's tools" / cli_setup.PROGRAMS[provider]
    program.parent.mkdir()
    item = {**SIGNED_OUT, "path": str(program)}
    # Both a rejected login and a cancelled one must stop before Done. A new attempt can succeed.
    for exit_code in (1, 130, 0):
        program.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$*\"\nexit {exit_code}\n", encoding="utf-8")
        program.chmod(0o700)
        script = cli_setup.open_terminal(provider, item, MESSAGES, name="linux", launch=lambda *args, **kwargs: None,
                                         which=lambda _: "terminal")
        result = subprocess.run(["bash", str(script)], input="\n", capture_output=True, text=True,
                                encoding="utf-8", timeout=30, env={"PATH": "/usr/bin:/bin", "TERM": "dumb"})
        assert cli_setup.SIGN_IN[provider] in result.stdout
        assert result.returncode == (1 if exit_code else 0)
        assert ("<sign_in_failed>" in result.stdout) == bool(exit_code)
        assert ("<done>" in result.stdout) == (not exit_code)
        assert not script.parent.exists()


@pytest.mark.skipif(os.name != "nt", reason="Runs native Windows PowerShell sign-in scripts.")
@pytest.mark.parametrize("provider", cli_setup.PROVIDERS)
def test_windows_sign_in_failure_cancel_and_retry(home, provider):
    program = home / "Ada's tools" / f"{cli_setup.PROGRAMS[provider]}.cmd"
    program.parent.mkdir()
    item = {**SIGNED_OUT, "path": str(program)}
    for exit_code in (1, 130, 0):
        program.write_text(f"@echo off\necho %*\nexit /b {exit_code}\n", encoding="utf-8")
        script = cli_setup.open_terminal(provider, item, MESSAGES, name="windows", launch=lambda *args, **kwargs: None)
        result = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                                input="\n", capture_output=True, text=True, timeout=30)
        assert cli_setup.SIGN_IN[provider] in result.stdout
        assert result.returncode == (1 if exit_code else 0)
        assert ("<sign_in_failed>" in result.stdout) == bool(exit_code)
        assert ("<done>" in result.stdout) == (not exit_code)
        assert not script.parent.exists()


@pytest.mark.skipif(os.name != "nt", reason="Runs native Windows PowerShell installer scripts.")
@pytest.mark.parametrize("installer", ["Write-Error 'private-install-error'", "& $env:ComSpec /c exit 7"])
def test_windows_failed_update_cannot_accept_an_old_executable(home, monkeypatch, installer):
    program = home / "old.cmd"
    program.write_text("@exit /b 0\n", encoding="utf-8")
    monkeypatch.setitem(cli_setup.INSTALL, ("codex", "windows"), installer)
    monkeypatch.setattr(cli_setup, "installed_path", lambda *args: str(program))
    script = cli_setup.open_terminal("codex", SIGNED_IN, MESSAGES, name="windows", update=True,
                                     launch=lambda *args, **kwargs: None)
    result = subprocess.run([shutil.which("powershell.exe"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            input="\n", capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and "<failed>" in result.stdout and "<done>" not in result.stdout
    assert "private-install-error" not in result.stdout + result.stderr
    assert program.exists() and not script.parent.exists()
