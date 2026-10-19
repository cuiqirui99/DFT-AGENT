"""Find the newest usable Codex or Claude Code CLI and read its sign-in state.

Several copies of a CLI can coexist: a standalone install on PATH, an npm
global install and the copy bundled with a desktop app. Only the newest one
may support the models the user has access to, so connections remember the
executable they were tested with instead of trusting PATH order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CLIS = {"codex": "codex", "claude_cli": "claude"}
OVERRIDES = {"codex": "CODEX_CLI_PATH", "claude_cli": "CLAUDE_CLI_PATH"}
LABELS = {"codex": "Codex CLI", "claude_cli": "Claude Code CLI"}
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.]+))?")


def _known_locations(provider: str) -> list[Path]:
    home = Path.home()
    name = CLIS[provider]
    folders = [home / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"),
               home / ".npm-global" / "bin", home / "bin", home / ".volta" / "bin"]
    if provider == "codex":
        folders += [home / ".codex" / "packages" / "standalone" / "current" / "bin",
                    Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin"),
                    home / "Applications" / "ChatGPT.app" / "Contents" / "Resources" / "codex-cli" / "bin",
                    Path("/Applications/Codex.app/Contents/Resources/codex-cli/bin")]
    else:
        folders += [home / ".claude" / "local", home / ".claude" / "local" / "node_modules" / ".bin"]
    names = [name]
    if sys.platform == "win32":
        names += [name + ".cmd", name + ".exe"]
        for variable in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(variable)
            if base:
                folders += [Path(base) / "npm", Path(base) / "Programs" / name]
    return [folder / item for folder in folders for item in names]


def version_of(path: str | Path, timeout: float = 10, *, run=None) -> tuple[int, ...] | None:
    """Parse `<cli> --version`; None when the executable cannot report one."""
    try:
        result = (run or subprocess.run)([str(path), "--version"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    match = _VERSION.search((getattr(result, "stdout", "") or "") + " " + (getattr(result, "stderr", "") or ""))
    if getattr(result, "returncode", 1) or not match:
        return None
    # A pre-release sorts below the same release; the numbers decide otherwise.
    return (int(match[1]), int(match[2]), int(match[3]), 0 if match[4] else 1)


def version_text(version: tuple[int, ...] | None) -> str:
    return ".".join(str(part) for part in version[:3]) if version else "unknown version"


def locate(provider: str, *, which=None, run=None) -> dict | None:
    """The CLI to use: an explicit override, else the newest copy found.

    Returns {"path", "version", "source"} or None. The PATH copy wins ties so a
    deliberate install is not replaced by an identical bundled copy.
    """
    if provider not in CLIS:
        raise ValueError("Choose codex or claude_cli.")
    which = which or shutil.which
    override = os.environ.get(OVERRIDES[provider])
    if override:
        resolved = which(override) or (override if Path(override).is_file() else None)
        return {"path": resolved, "version": version_of(resolved, run=run) if resolved else None, "source": "override"} if resolved else None
    candidates: dict[str, str] = {}
    primary = which(CLIS[provider])
    if primary:
        candidates[str(Path(primary).resolve())] = "path"
    for item in _known_locations(provider):
        try:
            if item.is_file() and os.access(item, os.X_OK):
                candidates.setdefault(str(item.resolve()), "known")
        except OSError:
            continue
    if not candidates:
        return None
    if len(candidates) == 1:
        path = next(iter(candidates))
        return {"path": primary or path, "version": version_of(path, run=run), "source": candidates[path]}
    best = None
    for path, source in candidates.items():
        version = version_of(path, run=run)
        if version is None:
            continue
        order = (version, source == "path")
        if best is None or order > best[0]:
            best = (order, path, version, source)
    if best is None:
        path = primary or next(iter(candidates))
        return {"path": path, "version": None, "source": candidates.get(str(Path(path).resolve()), "path")}
    _, path, version, source = best
    return {"path": primary if source == "path" and primary else path, "version": version, "source": source}


def login_status(provider: str, path: str, timeout: float = 20, *, run=None) -> dict:
    """Ask the CLI whether it is signed in; never prints or stores its tokens."""
    arguments = [path, "login", "status"] if provider == "codex" else [path, "auth", "status"]
    try:
        result = (run or subprocess.run)(arguments, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {"logged_in": None, "detail": "The sign-in check did not finish."}
    stdout, stderr = getattr(result, "stdout", "") or "", getattr(result, "stderr", "") or ""
    if provider == "codex":
        text = stdout + stderr
        if result.returncode == 0 and "logged in" in text.lower() and "not logged in" not in text.lower():
            # Some versions append an API key (or a fragment) to this line.
            detail = "ChatGPT" if "chatgpt" in text.lower() else "API key" if re.search(r"api[ _-]?key", text, re.I) else "Signed in"
            return {"logged_in": True, "detail": detail}
        return {"logged_in": False, "detail": "Not signed in. Run `codex login` in a terminal."}
    try:
        data = json.loads(stdout or "{}")
    except ValueError:
        data = {}
    if result.returncode == 0 and isinstance(data, dict) and data.get("loggedIn"):
        method = data.get("authMethod")
        detail = method if isinstance(method, str) and method in {"claude.ai", "api_key", "apiKey", "bedrock", "vertex", "foundry", "oauth"} else "Signed in"
        return {"logged_in": True, "detail": detail}
    return {"logged_in": False, "detail": "Not signed in. Run `claude` in a terminal and sign in."}


def detect(providers=("codex", "claude_cli"), *, which=None) -> list[dict]:
    """Installed CLIs with version and sign-in state, for the setup screen."""
    found = []
    for provider in providers:
        located = locate(provider, which=which)
        if not located or not located["path"]:
            found.append({"provider": provider, "label": LABELS[provider], "installed": False, "path": None,
                          "version": None, "logged_in": None, "detail": "Not installed."})
            continue
        status = login_status(provider, located["path"])
        found.append({"provider": provider, "label": LABELS[provider], "installed": True, "path": located["path"],
                      "version": located["version"], "logged_in": status["logged_in"], "detail": status["detail"]})
    return found
