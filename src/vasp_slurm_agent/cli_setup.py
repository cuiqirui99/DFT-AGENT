"""Install and sign in to the Codex or Claude Code CLI for someone who has never used a terminal.

The official installers need neither Node.js nor administrator rights. DFT Agent opens the
computer's own terminal with a short script that installs the CLI when it is missing and then
starts its sign-in, so the person only signs in on the page the CLI opens in the browser.
Nothing is installed or signed in without that visible terminal window, and no credential
passes through DFT Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

PROVIDERS = ("codex", "claude_cli")
PROGRAMS = {"codex": "codex", "claude_cli": "claude"}
# Official installers: https://github.com/openai/codex#installing-and-running-codex-cli and
# https://code.claude.com/docs/en/setup. Both also update an existing installation.
INSTALL = {
    ("codex", "unix"): "curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh",
    ("codex", "windows"): "$env:CODEX_NON_INTERACTIVE = '1'; irm https://chatgpt.com/codex/install.ps1 | iex",
    ("claude_cli", "unix"): "curl -fsSL https://claude.ai/install.sh | bash",
    ("claude_cli", "windows"): "irm https://claude.ai/install.ps1 | iex",
}
SIGN_IN = {"codex": "login", "claude_cli": "auth login"}
ACCOUNTS = {"codex": "Uses a ChatGPT Plus, Pro, Business, Edu or Enterprise plan.",
            "claude_cli": "Uses a Claude Pro, Max, Team or Enterprise plan, or an Anthropic Console account."}
# Lines the terminal window shows, translated by the caller.
MESSAGES = ("title", "installing", "failed", "missing", "signing_in", "sign_in_failed", "done", "close")


def system():
    """macos, windows or linux."""
    return "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux"


def _family(name):
    return "windows" if name == "windows" else "unix"


def installed_path(provider, name=None):
    """Where the official installer puts the program, which a new terminal may not find on PATH yet."""
    if provider == "codex" and os.environ.get("CODEX_INSTALL_DIR"):
        binary = "codex.exe" if _family(name or system()) == "windows" else "codex"
        return str(Path(os.environ["CODEX_INSTALL_DIR"]).expanduser() / binary)
    if _family(name or system()) == "windows":
        if provider == "codex":
            local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
            return str(local / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe")
        return str(Path.home() / ".local" / "bin" / "claude.exe")
    return str(Path.home() / ".local" / "bin" / PROGRAMS[provider])


def next_step(item):
    """install, sign_in or connect for a detected CLI (an entry of cli_tools.detect)."""
    if not item.get("installed") or ("version" in item and item["version"] is None):
        return "install"
    return "connect" if item.get("logged_in") else "sign_in"


def _plan(provider, item, name, update):
    """What the window does: (install command or None, program path, sign in?)."""
    install = INSTALL[(provider, _family(name))] if update or next_step(item) == "install" else None
    path = installed_path(provider, name) if install else item["path"]
    return install, path, not item.get("logged_in")


def commands(provider, item, name=None, *, update=False, which=None):
    """The same steps as lines to paste into a terminal by hand."""
    name = name or system()
    install, path, sign_in = _plan(provider, item, name, update)
    lines = [install] if install else []
    if sign_in:
        located = (which or shutil.which)(PROGRAMS[provider])
        if not install and located and Path(located).resolve() == Path(path).resolve():
            program = PROGRAMS[provider]
        elif _family(name) == "windows":
            program = "& '" + path.replace("'", "''") + "'"
        elif Path(path).is_relative_to(Path.home()):
            program = "~/" + shlex.quote(Path(path).relative_to(Path.home()).as_posix())
        else:
            program = shlex.quote(path)
        lines.append(f"{program} {SIGN_IN[provider]}")
    return lines


def _unix_script(provider, item, messages, update):
    install, path, sign_in = _plan(provider, item, name="unix", update=update)

    def say(key):
        return f"echo {shlex.quote(messages[key])}"
    wait = f"read -r -p {shlex.quote(messages['close'] + ' ')} _"
    lines = ["#!/bin/bash", 'rm -f -- "$0"; rmdir -- "$(dirname -- "$0")" 2>/dev/null', "set -o pipefail", "clear", say("title"), "echo"]
    if install:
        if provider == "codex" and os.environ.get("CODEX_INSTALL_DIR"):
            lines += [f"export CODEX_INSTALL_DIR={shlex.quote(str(Path(path).parent))}"]
        lines += [say("installing"), f"if ! {install}; then", "  echo", "  " + say("failed"), "  " + wait, "  exit 1", "fi", "echo"]
    lines += [f"if [ ! -x {shlex.quote(path)} ]; then", "  " + say("missing"), "  " + wait, "  exit 1", "fi"]
    if sign_in:
        lines += [say("signing_in"), f"if ! {shlex.quote(path)} {SIGN_IN[provider]}; then",
                  "  " + say("sign_in_failed"), "  " + wait, "  exit 1", "fi", "echo"]
    lines += [say("done"), wait]
    return "\n".join(lines) + "\n"


def _quoted(text):
    return "'" + text.replace("'", "''") + "'"


def _windows_script(provider, item, messages, update):
    install, path, sign_in = _plan(provider, item, name="windows", update=update)

    def say(key):
        return f"Write-Host {_quoted(messages[key])}"
    def stop(key):
        return f"{say(key)}; Read-Host {_quoted(messages['close'])} | Out-Null; exit 1"
    lines = ["Remove-Item -LiteralPath $PSCommandPath -ErrorAction SilentlyContinue",
             "Remove-Item -LiteralPath (Split-Path -Parent $PSCommandPath) -ErrorAction SilentlyContinue",
             "$ErrorActionPreference = 'Stop'",
             "Clear-Host", say("title"), "Write-Host ''"]
    if install:
        if provider == "codex" and os.environ.get("CODEX_INSTALL_DIR"):
            lines += [f"$env:CODEX_INSTALL_DIR = {_quoted(str(Path(path).parent))}"]
        lines += [say("installing"), "$global:LASTEXITCODE = 0", "try {", f"  {install}",
                  "  if ($LASTEXITCODE -ne 0) { throw 'Installer failed.' }",
                  f"}} catch {{ Write-Host ''; {stop('failed')} }}", "Write-Host ''"]
    lines += [f"$program = {_quoted(path)}", f"if (-not (Test-Path -LiteralPath $program -PathType Leaf)) {{ {stop('missing')} }}"]
    if sign_in:
        lines += [say("signing_in"), "$global:LASTEXITCODE = 0", "try {",
                  f"  & $program {SIGN_IN[provider]}",
                  "  if ($LASTEXITCODE -ne 0) { throw 'Sign-in failed.' }",
                  f"}} catch {{ {stop('sign_in_failed')} }}", "Write-Host ''"]
    lines.append(say("done"))
    return "\r\n".join(lines) + "\r\n"


# Terminal programs of Linux desktops and how each runs a command.
_LINUX_TERMINALS = (("x-terminal-emulator", ["-e"]), ("gnome-terminal", ["--"]), ("konsole", ["-e"]),
                    ("xfce4-terminal", ["-x"]), ("kitty", []), ("alacritty", ["-e"]), ("xterm", ["-e"]))


def open_terminal(provider, item, messages, *, update=False, name=None, launch=None, which=None):
    """Open a terminal window that installs (when needed) and signs in; returns the script's path.

    item is the CLI's entry from cli_tools.detect; messages maps each of MESSAGES to the line
    the window shows."""
    name = name or system()
    launch = launch or subprocess.Popen
    folder = Path(tempfile.mkdtemp(prefix="dft-agent-"))
    try:
        return _open_terminal(provider, item, messages, update, name, launch, which, folder)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _open_terminal(provider, item, messages, update, name, launch, which, folder):
    if name == "windows":
        script = folder / f"{provider}.ps1"
        # Windows PowerShell 5 reads a script without a byte-order mark in the legacy code page.
        script.write_text(_windows_script(provider, item, messages, update), encoding="utf-8-sig", newline="")
        launch(["powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
               creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10))
        return script
    script = folder / f"{provider}.command"
    script.write_text(_unix_script(provider, item, messages, update), encoding="utf-8")
    script.chmod(0o700)
    if name == "macos":
        # Terminal runs a .command file without asking for permission to be controlled.
        launch(["open", "-a", "Terminal", str(script)])
        return script
    for program, flags in _LINUX_TERMINALS:
        if (which or shutil.which)(program):
            launch([program, *flags, "bash", str(script)])
            return script
    script.unlink(missing_ok=True)
    folder.rmdir()
    raise OSError("No terminal program was found. Open a terminal and paste the commands below.")
