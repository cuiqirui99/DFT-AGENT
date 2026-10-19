"""Open visible native SSH sign-in. Login prompts never pass through the app."""

import os
import platform
import re
import shlex
import shutil
import subprocess


def _active_hpc(registry, identifier):
    from .connections import _settings

    record = registry.get(identifier)
    if record["kind"] != "hpc" or record.get("revoked") or record.get("deleted"):
        raise ValueError("Choose an active HPC connection.")
    settings = _settings("hpc", record["settings"])
    # ProxyJump can construct an internal proxy command. Reject shell syntax in
    # endpoint names as well as quoting every argument in the terminal command.
    for field in ("host", "user", "jump_host"):
        if settings.get(field) and not re.fullmatch(r"[A-Za-z0-9_.@:+,\[\]%\\-]+", settings[field]):
            raise ValueError("Use an SSH host, user or jump-host name without shell characters.")
    return {**record, "settings": settings}


def _control_path(registry, record):
    existing = record["settings"].get("ssh_control_path")
    if existing:
        return existing
    return registry.ensure_control_path(record["id"])


def _ssh_option(value):
    # OpenSSH parses -o values again as SSH configuration, even with shell=False.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def command(registry, identifier, *, system=None):
    """Build argv for a validated connection; no credential is read."""
    system = system or platform.system()
    record = _active_hpc(registry, identifier)
    settings = record["settings"]
    ssh = shutil.which("ssh") or "ssh"
    argv = [ssh, "-tt", "-o", "BatchMode=no", "-o", f"ConnectTimeout={settings['connect_timeout']}",
            "-p", str(settings["port"]), "-l", settings["user"]]
    if settings["jump_host"]:
        argv += ["-J", settings["jump_host"]]
    if settings["identity_file"]:
        argv += ["-i", settings["identity_file"]]
    if system != "Windows":
        control = _control_path(registry, record)
        if len(os.fsencode(control)) > 100:
            raise ValueError("Use a shorter local SSH control socket path.")
        from .transport import LOGIN_PERSIST_SECONDS
        argv += ["-o", "ControlMaster=auto", "-o", f"ControlPersist={LOGIN_PERSIST_SECONDS}", "-o", "ControlPath=" + _ssh_option(control)]
    argv += ["--", settings["host"]]
    return argv


def _powershell(argv):
    return "& " + " ".join("'" + item.replace("'", "''") + "'" for item in argv)


def _apple_string(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def launch(registry, identifier):
    """Launch sign-in, or return a manual command when no terminal is available.

    A launched terminal is not evidence that SSH authenticated. The caller must
    test the connection separately after the user completes terminal prompts.
    """
    system = platform.system()
    argv = command(registry, identifier, system=system)
    reusable = system != "Windows"
    manual = _powershell(argv) if system == "Windows" else shlex.join(argv)
    result = {"status": "manual", "command": manual, "reusable": reusable,
              "message": "Run this command in your terminal, then test the connection."}
    if system == "Windows":
        result["message"] = "Sign in in the terminal. Windows cannot reuse this session; automated access needs a working SSH key or agent."
    if not shutil.which("ssh"):
        result["message"] = "Install OpenSSH, then run this command in your terminal."
        return result
    environment = dict(os.environ)
    private_variables = ("SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "_DFT_AGENT_ASKPASS_PASSWORD",
                         "DFT_AGENT_SSH_PASSWORD", "_DFT_AGENT_WORKER_PASSWORD",
                         "_DFT_AGENT_ASKPASS_CODE", "_DFT_AGENT_ASKPASS_PROMPT_FILE")
    unset = [arg for name in private_variables for arg in ("-u", name)]
    for name in private_variables:
        environment.pop(name, None)
    options = {"env": environment, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if system == "Darwin":
        executable = shutil.which("osascript")
        if not executable:
            return result
        # Terminal can already be running with its own environment. Disable
        # askpass explicitly so MFA prompts stay in the visible terminal.
        visible = shlex.join(["/usr/bin/env", *unset, *argv])
        launcher = [executable, "-e", "tell application \"Terminal\" to do script " + _apple_string(visible),
                    "-e", 'tell application "Terminal" to activate']
    elif system == "Windows":
        executable = shutil.which("pwsh") or shutil.which("powershell")
        if not executable:
            return result
        launcher = [executable, "-NoLogo", "-NoProfile", "-NoExit", "-Command", manual]
        options["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        for key in ("stdin", "stdout", "stderr"):
            options.pop(key)
    elif system == "Linux":
        if not (environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")):
            result["message"] = "No graphical terminal is available. Run this command in your terminal."
            return result
        launcher = None
        for executable, separator in (("gnome-terminal", "--"), ("konsole", "-e"), ("xterm", "-e"), ("kitty", "--")):
            if path := shutil.which(executable):
                launcher = [path, separator, "env", *unset, *argv]
                break
        if launcher is None:
            return result
    else:
        return result
    # Reject revocation or a changed destination after command construction.
    _active_hpc(registry, identifier)
    if command(registry, identifier, system=system) != argv:
        raise ValueError("The connection changed. Open sign-in again.")
    try:
        if system == "Darwin":
            if subprocess.run(launcher, timeout=10, check=False, **options).returncode:
                return result
        else:
            subprocess.Popen(launcher, **options)
    except (OSError, subprocess.TimeoutExpired):
        return result
    result["status"] = "launched"
    if reusable:
        from .transport import LOGIN_PERSIST_SECONDS
        hours = LOGIN_PERSIST_SECONDS // 3600
        result["message"] = (f"Complete sign-in in the terminal, then test this connection. The signed-in session is reused "
                             f"for about {hours} hours after its last use.")
    return result
