"""Reusable connections, private credential references and real connection probes."""

from __future__ import annotations

from dataclasses import fields
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from urllib.parse import urlsplit
import uuid

from . import credentials
from .workspace import Workspace, _id, _now, _object, _text


_MODEL_FIELDS = {"provider", "model", "base_url", "cli_path"}
_HPC_FIELDS = {"host", "user", "port", "auth_mode", "ssh_control_path", "jump_host",
               "identity_file", "remote_base", "scheduler", "connect_timeout"}
_CLI = {"codex": "codex", "claude_cli": "claude"}
# Passwords stay in the keychain. "code" can follow a password or SSH key;
# it keeps an authenticated session for later commands and detached workers.
PASSWORD_MODES = {"password", "code"}
AUTH_MODES = {"key", "agent", "password", "code", "interactive"}
# Settings a successful test may fill in without resetting the test result.
_PATCHABLE = {"remote_base", "ssh_control_path", "scheduler", "cli_path", "auth_mode"}
_CODE_LABELS = {"abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk"}
_CODE_PROGRAMS = {"abinit": ("abinit",), "cp2k": ("cp2k.psmp", "cp2k.popt", "cp2k"), "elk": ("elk",)}


def site_library_root(engine, remote_base):
    """Where the pinned library lives for a program the cluster already provides."""
    base = PurePosixPath(remote_base or "/tmp")
    return str({"abinit": base / "pseudos" / "abinit", "cp2k": base / "libraries" / "cp2k-2026.2",
                "elk": base / "libraries" / "elk-10.2.4" / "species"}[engine])


_TRANSPORT_FIELDS = {"host", "user", "port", "remote_root", "ssh_control_path",
                     "ssh_jump_host", "ssh_identity_file", "connection_id", "connection_home"}


def _settings(kind, value):
    from .providers import PROVIDERS

    if not isinstance(value, dict):
        raise ValueError("Connection settings must be an object.")
    _object(value)
    allowed = _MODEL_FIELDS if kind == "model" else _HPC_FIELDS
    if set(value) - allowed:
        raise ValueError("Unknown connection settings. Enter credentials in the secure field.")
    result = dict(value)
    if kind == "model":
        provider = value.get("provider")
        if provider not in {*PROVIDERS, "claude_cli", "local"}:
            raise ValueError("Choose an available model provider.")
        result["model"] = _text(value.get("model", ""), "model", 200, empty=True).strip()
        if provider not in _CLI and not result["model"]:
            raise ValueError("Enter the model name offered by this provider.")
        url = value.get("base_url") or ""
        if url:
            if not isinstance(url, str) or any(c.isspace() for c in url):
                raise ValueError("Enter a valid API URL.")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Use an API URL without credentials or query parameters.")
        if provider in {"local", "chat_completions", "qwen"} and not url:
            raise ValueError("Enter the API base URL.")
        if provider == "local" and urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Local models use a loopback URL. Use Compatible API for a remote server.")
        if provider in _CLI and url:
            raise ValueError("CLI connections use the CLI's own login, without an API URL.")
        result["base_url"] = url or None
        cli_path = value.get("cli_path") or None
        if cli_path is not None:
            if provider not in _CLI or not isinstance(cli_path, str) or re.search(r"[\x00-\x1f]", cli_path) \
                    or not Path(cli_path).is_absolute():
                raise ValueError("The CLI path must be an absolute path of a CLI connection.")
        result["cli_path"] = cli_path
    else:
        for name in ("host", "user"):
            item = value.get(name)
            if not isinstance(item, str) or not item or item.startswith("-") or re.search(r"[\s\x00-\x1f]", item):
                raise ValueError(f"Enter a valid SSH {name}.")
        for name, default, lower, upper in (("port", 22, 1, 65535), ("connect_timeout", 15, 1, 120)):
            item = value.get(name, default)
            if type(item) is not int or not lower <= item <= upper:
                raise ValueError(f"Invalid {name}.")
            result[name] = item
        result["auth_mode"] = value.get("auth_mode", "agent")
        if result["auth_mode"] not in AUTH_MODES:
            raise ValueError("Choose a supported SSH authentication method.")
        result["scheduler"] = value.get("scheduler", "auto")
        if result["scheduler"] not in {"auto", "slurm", "pbs"}:
            raise ValueError("Choose automatic detection, Slurm or PBS.")
        for name in ("ssh_control_path", "identity_file", "remote_base", "jump_host"):
            item = value.get(name, "")
            if not isinstance(item, str) or re.search(r"[\x00-\x1f]", item):
                raise ValueError(f"Invalid {name}.")
            if name in {"ssh_control_path", "identity_file"} and item:
                path = Path(item).expanduser()
                if not path.is_absolute():
                    raise ValueError(f"Use an absolute {name} path.")
                item = str(path)
            if name == "remote_base" and item and (not item.startswith("/") or ".." in PurePosixPath(item).parts):
                raise ValueError("Use an absolute remote work folder.")
            if name == "jump_host" and item and (item.startswith("-") or re.search(r"\s", item)):
                raise ValueError("Enter a valid SSH jump host.")
            result[name] = item
    return result


class ConnectionRegistry:
    """Credentials are resolved from the keychain for each action, never cached."""

    def __init__(self, home=None):
        self.workspace = home if isinstance(home, Workspace) else Workspace(home)
        self.home = self.workspace.home

    @staticmethod
    def _record(row):
        item = dict(row)
        item["settings"] = json.loads(item["settings"])
        item["probe"] = json.loads(item["probe"]) if item["probe"] else None
        item["has_secret"] = item["secret_version"] > 0 and not item["revoked_at"]
        item["revoked"], item["deleted"] = bool(item["revoked_at"]), bool(item["deleted_at"])
        return item

    def _row(self, db, identifier, *, active=False):
        row = db.execute("SELECT * FROM connections WHERE id=?", (_id(identifier),)).fetchone()
        if row is None:
            raise ValueError("Connection not found.")
        if active and (row["revoked_at"] or row["deleted_at"]):
            raise ValueError("This connection has been revoked. Create a new connection to use it again.")
        return row

    @staticmethod
    def _event(db, identifier, kind, details=None):
        db.execute("INSERT INTO connection_events(connection_id,kind,details,created_at) VALUES (?,?,?,?)",
                   (identifier, kind, _object(details or {}), _now()))

    def create(self, kind, name, settings):
        if kind not in {"model", "hpc"}:
            raise ValueError("Choose a model or HPC connection.")
        name, settings = _text(name, "connection name"), _settings(kind, settings)
        identifier, now = str(uuid.uuid4()), _now()
        with self.workspace._transaction() as db:
            db.execute("INSERT INTO connections(id,kind,name,settings,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                       (identifier, kind, name, _object(settings), now, now))
            self._event(db, identifier, "created")
            return self._record(self._row(db, identifier))

    def get(self, identifier):
        with self.workspace._transaction(write=False) as db:
            return self._record(self._row(db, identifier))

    def list(self, kind=None):
        if kind is not None and kind not in {"model", "hpc"}:
            raise ValueError("Unknown connection type.")
        with self.workspace._transaction(write=False) as db:
            return [self._record(row) for row in db.execute(
                "SELECT * FROM connections WHERE deleted_at IS NULL AND (? IS NULL OR kind=?) ORDER BY updated_at DESC,id",
                (kind, kind))]

    def update(self, identifier, settings=None, name=None):
        with self.workspace._transaction() as db:
            row = self._row(db, identifier, active=True)
            merged = json.loads(row["settings"])
            if settings is not None:
                if not isinstance(settings, dict):
                    raise ValueError("Connection settings must be an object.")
                merged.update(settings)
            merged = _settings(row["kind"], merged)
            prior = json.loads(row["settings"])
            if merged == prior:
                db.execute("UPDATE connections SET name=?,updated_at=? WHERE id=?",
                           (_text(name, "connection name") if name is not None else row["name"], _now(), identifier))
                self._event(db, identifier, "renamed")
                return self._record(self._row(db, identifier))
            identity_fields = {"provider", "base_url"} if row["kind"] == "model" else {"host", "user", "port", "auth_mode", "jump_host"}
            if row["secret_version"] and any(merged.get(key) != prior.get(key) for key in identity_fields):
                # A stored key must never follow an edited endpoint or account.
                raise ValueError("Create a new connection for changed settings, then enter its credential.")
            db.execute("UPDATE connections SET name=?,settings=?,revision=revision+1,auth_status='unknown',"
                       "connection_status='untested',probe=NULL,last_checked=NULL,updated_at=? WHERE id=?",
                       (_text(name, "connection name") if name is not None else row["name"], _object(merged), _now(), identifier))
            self._event(db, identifier, "updated")
            result = self._record(self._row(db, identifier))
        session_fields = {"host", "user", "port", "auth_mode", "jump_host", "identity_file", "ssh_control_path"}
        if platform.system() == "Windows" and result["kind"] == "hpc" and any(merged.get(key) != prior.get(key) for key in session_fields):
            result.update(self._retire_session(identifier))
        return result

    def _patch_settings(self, identifier, **changes):
        """Fill in non-identity settings learned from a test without resetting its result.

        The password sign-in may switch to "code" (same account, same stored
        password) when the cluster also asks for a login code.
        """
        if set(changes) - _PATCHABLE:
            raise ValueError("Only test results may be recorded this way.")
        with self.workspace._transaction() as db:
            row = self._row(db, identifier, active=True)
            current = json.loads(row["settings"])
            if "auth_mode" in changes:
                modes = {current.get("auth_mode"), changes["auth_mode"]}
                adds_code = current.get("auth_mode") in {"key", "agent"} and changes["auth_mode"] == "code"
                if modes - PASSWORD_MODES and not adds_code:
                    raise ValueError("Create a new connection for a different sign-in method.")
            merged = _settings(row["kind"], {**current, **changes})
            if merged == current:
                return self._record(row)
            db.execute("UPDATE connections SET settings=?,revision=revision+1,updated_at=? WHERE id=?",
                       (_object(merged), _now(), identifier))
            self._event(db, identifier, "updated", {"fields": sorted(changes)})
            return self._record(self._row(db, identifier))

    def record_settings(self, identifier, **changes):
        """Change the work folder, scheduler or CLI path; the sign-in method stays."""
        if "auth_mode" in changes:
            raise ValueError("Create a new connection for a different sign-in method.")
        return self._patch_settings(identifier, **changes)

    def disconnect(self, identifier):
        """Close this connection's signed-in multiplexing session, if one is open."""
        record = self.get(identifier)
        if platform.system() == "Windows" and record["kind"] == "hpc":
            from . import ssh_broker
            return ssh_broker.disconnect(self.home, identifier)
        socket = record["settings"].get("ssh_control_path") if record["kind"] == "hpc" else None
        if not socket or not Path(socket).exists():
            return False
        argv = [shutil.which("ssh") or "ssh", "-o", "ControlPath=" + socket, "-O", "exit", "--", record["settings"]["host"]]
        try:
            subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return not Path(socket).exists()

    def _retire_session(self, identifier):
        """Cleanup cannot undo a connection change already committed to the registry."""
        try:
            self.disconnect(identifier)
        except Exception:
            return {"session_cleanup_pending": True}
        return {}

    def ensure_control_path(self, identifier):
        """A private multiplexing socket path for this connection's sign-in session."""
        record = self.get(identifier)
        existing = record["settings"].get("ssh_control_path")
        if existing:
            parent = Path(existing).parent
            if not parent.exists():
                # Temporary folders are swept by the OS; the socket itself is recreated on sign-in.
                parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            return existing
        folder = Path(self.home) / "ssh-sockets"
        candidate = folder / (record["id"][:12] + ".sock")
        if len(os.fsencode(candidate)) > 96:
            # macOS temporary paths can also exceed the Unix socket limit.
            folder = Path(tempfile.mkdtemp(prefix="dft-login-", dir="/tmp"))
            candidate = folder / "socket"
        else:
            if folder.is_symlink():
                raise ValueError("The SSH socket folder cannot be a symbolic link.")
            folder.mkdir(mode=0o700, exist_ok=True)
            folder.chmod(0o700)
        self._patch_settings(identifier, ssh_control_path=str(candidate))
        return str(candidate)

    @staticmethod
    def _secret_name(identifier, version):
        return f"connection:{identifier}:{version}"

    def set_secret(self, identifier, value):
        if not isinstance(value, str) or not value or len(value) > 16384 or re.search(r"[\x00\r\n]", value):
            raise ValueError("Enter the credential in the secure field on one line.")
        backend = credentials._keyring()
        secret_name = None
        # Reserve the slot durably before touching the OS keychain. A failed
        # database commit and locked keychain must not leave untracked secrets.
        with self.workspace._transaction() as db:
            original = self._row(db, identifier, active=True)
            settings = json.loads(original["settings"])
            if (original["kind"] == "model" and settings["provider"] in {*_CLI, "local"}) or (original["kind"] == "hpc" and settings["auth_mode"] not in PASSWORD_MODES):
                raise ValueError("This connection does not store a password or API key.")
            reserved = db.execute("SELECT MAX(version) FROM credential_slots WHERE connection_id=?", (identifier,)).fetchone()[0] or 0
            version = max(original["secret_version"], reserved) + 1
            db.execute("INSERT INTO credential_slots VALUES (?,?,?)", (identifier, version, _now()))
        try:
            with self.workspace._transaction() as db:
                row = self._row(db, identifier, active=True)
                if row["revision"] != original["revision"]:
                    raise ValueError("The connection changed. Reload it before updating its credential.")
                secret_name = self._secret_name(identifier, version)
                backend.set_password(credentials.SERVICE, secret_name, value)
                if backend.get_password(credentials.SERVICE, secret_name) != value:
                    raise credentials.CredentialError("The keychain did not retain the credential.")
                prior_probe = json.loads(row["probe"]) if row["probe"] else {}
                progress = {key: prior_probe[key] for key in ("discovery", "qe_install") if key in prior_probe}
                db.execute("UPDATE connections SET secret_version=?,revision=revision+1,auth_status='unknown',"
                           "connection_status='untested',probe=?,last_checked=NULL,updated_at=? WHERE id=?",
                           (version, _object(progress) if progress else None, _now(), identifier))
                self._event(db, identifier, "credential.updated")
        except Exception as error:
            if secret_name is not None:
                try:
                    backend.delete_password(credentials.SERVICE, secret_name)
                except Exception:
                    pass
            if isinstance(error, ValueError):
                raise
            raise credentials.CredentialError("The credential could not be saved securely.") from None
        if row["secret_version"]:
            try:
                backend.delete_password(credentials.SERVICE, self._secret_name(identifier, row["secret_version"]))
            except Exception:
                # The old version is unreachable; revocation retries cleanup.
                pass
        cleanup = self._retire_session(identifier) if platform.system() == "Windows" and row["kind"] == "hpc" else {}
        return {**self.get(identifier), **cleanup}

    def _secret(self, identifier):
        with self.workspace._transaction(write=False) as db:
            row = self._row(db, identifier, active=True)
            version = row["secret_version"]
            if not version:
                raise credentials.CredentialError("Add this connection's credential in the secure field.")
            try:
                value = credentials._keyring().get_password(credentials.SERVICE, self._secret_name(identifier, version))
            except Exception:
                raise credentials.CredentialError("The system keychain could not provide this credential.") from None
            if not value:
                raise credentials.CredentialError("The saved credential is unavailable. Enter it again securely.")
            return value

    def revoke(self, identifier):
        # Deny new requests before waiting for a busy transport or locked keychain.
        with self.workspace._transaction() as db:
            row = self._row(db, identifier)
            db.execute("UPDATE connections SET revoked_at=?,revision=revision+1,auth_status='revoked',"
                       "connection_status='revoked',updated_at=? WHERE id=?", (_now(), _now(), identifier))
            self._event(db, identifier, "revoked")
            versions = set(range(1, row["secret_version"] + 1))
            versions.update(item[0] for item in db.execute("SELECT version FROM credential_slots WHERE connection_id=?", (identifier,)))
        cleanup = self._retire_session(identifier)
        if versions:
            try:
                backend = credentials._keyring()
                for version in sorted(versions):
                    name = self._secret_name(identifier, version)
                    if backend.get_password(credentials.SERVICE, name):
                        backend.delete_password(credentials.SERVICE, name)
                    if backend.get_password(credentials.SERVICE, name):
                        return {**self.get(identifier), **cleanup, "credential_cleanup_pending": True}
            except Exception:
                return {**self.get(identifier), **cleanup, "credential_cleanup_pending": True}
        return {**self.get(identifier), **cleanup}

    def delete(self, identifier):
        result = self.revoke(identifier)
        with self.workspace._transaction() as db:
            db.execute("UPDATE connections SET deleted_at=?,revision=revision+1,updated_at=? WHERE id=?", (_now(), _now(), identifier))
            self._event(db, identifier, "deleted")
        return {**result, **self.get(identifier)}

    def model_settings(self, identifier, *, resolve_secret=False):
        from .agent import ModelSettings

        item = self.get(identifier)
        if item["kind"] != "model" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active model connection.")
        values = item["settings"]
        args = {"provider": values["provider"], "model": values["model"],
                "base_url": values.get("base_url"), "allow_env": False, "cli_path": values.get("cli_path")}
        if resolve_secret:
            if values["provider"] not in {*_CLI, "local"}:
                args["api_key"] = self._secret(identifier)
        else:
            args.update(connection_id=identifier, connection_home=str(self.home))
        return ModelSettings(**args)

    def ssh_password(self, identifier):
        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        mode = item["settings"]["auth_mode"]
        if mode == "code" and not item["has_secret"]:
            return ""
        return self._secret(identifier) if mode in PASSWORD_MODES else ""

    def bind(self, project_id, identifier, role=None):
        with self.workspace._transaction() as db:
            self.workspace._project_row(db, project_id, writable=True)
            row = self._row(db, identifier, active=True)
            role = role or row["kind"]
            if role != row["kind"]:
                raise ValueError("The connection does not match this project role.")
            db.execute("INSERT OR IGNORE INTO project_connections VALUES (?,?,?,?)", (project_id, identifier, role, _now()))
            self.workspace._event(db, project_id, "connection.bound", {"connection_id": identifier, "role": role})
        return self.get(identifier)

    def project_connections(self, project_id):
        with self.workspace._transaction(write=False) as db:
            self.workspace._project_row(db, project_id)
            return [self._record(row) for row in db.execute("SELECT connections.* FROM connections JOIN project_connections "
                    "ON connections.id=project_connections.connection_id WHERE project_connections.project_id=? "
                    "ORDER BY connections.name,connections.id", (project_id,))]

    def _transport_config(self, item):
        settings = item["settings"]
        return SimpleNamespace(host=settings["host"], user=settings["user"], port=settings["port"],
            connect_timeout=settings["connect_timeout"], ssh_control_path=settings["ssh_control_path"],
            ssh_jump_host=settings["jump_host"], ssh_identity_file=settings["identity_file"],
            connection_id=item["id"], connection_home=str(self.home))

    def _save_probe(self, identifier, revision, report):
        with self.workspace._transaction() as db:
            row = self._row(db, identifier, active=True)
            if row["revision"] != revision:
                raise ValueError("The connection changed during its test. Test its current settings again.")
            previous = json.loads(row["probe"]) if row["probe"] else {}
            report = {**{key: previous[key] for key in ("discovery", "qe_install") if key in previous}, **report}
            db.execute("UPDATE connections SET probe=?,auth_status=?,connection_status=?,last_checked=?,updated_at=? WHERE id=?",
                       (_object(report), report["auth_status"], report["connection_status"], _now(), _now(), identifier))
            self._event(db, identifier, "probed", {"auth_status": report["auth_status"], "connection_status": report["connection_status"]})
        return self.get(identifier)

    def probe(self, identifier):
        """Make a real model request or read-only SSH probe. Never submit a job."""
        item = self.get(identifier)
        if item["revoked"] or item["deleted"]:
            raise ValueError("This connection is revoked.")
        if item["kind"] == "model" and item["settings"]["provider"] in _CLI:
            item = self._choose_cli(item)
        report = self._probe_model(item) if item["kind"] == "model" else self._probe_hpc(item)
        # A test may record what it learned (work folder, scheduler, sign-in
        # mode); only changes made elsewhere invalidate the result.
        revision = report.pop("revision", item["revision"])
        return self._save_probe(identifier, revision, report)

    def _choose_cli(self, item):
        """Remember the newest working CLI so requests do not depend on PATH order."""
        from . import cli_tools

        located = cli_tools.locate(item["settings"]["provider"], which=shutil.which, run=subprocess.run)
        if located and located["path"] and located["path"] != item["settings"].get("cli_path"):
            return self._patch_settings(item["id"], cli_path=located["path"])
        return item

    def _probe_model(self, item):
        from . import agent

        provider = item["settings"]["provider"]
        details = {}
        if provider in _CLI:
            from . import cli_tools

            override = "CODEX_CLI_PATH" if provider == "codex" else "CLAUDE_CLI_PATH"
            executable = item["settings"].get("cli_path") or shutil.which(os.environ.get(override) or _CLI[provider])
            if not executable:
                return {"auth_status": "unavailable", "connection_status": "unavailable",
                        "detail": f"{cli_tools.LABELS[provider]} is not installed. Install it and sign in, or choose an API provider."}
            details["cli_path"] = executable
            details["cli_version"] = cli_tools.version_text(cli_tools.version_of(executable, run=subprocess.run))
            status = cli_tools.login_status(provider, executable, run=subprocess.run)
            if status["logged_in"] is None:
                return {"auth_status": "unknown", "connection_status": "error",
                        "detail": "The CLI login check did not finish successfully.", **details}
            if not status["logged_in"]:
                return {"auth_status": "required", "connection_status": "untested",
                        "detail": f"Sign in with {cli_tools.LABELS[provider]} in your terminal, then test again.", **details}
            details["cli_login_checked"] = True
            details["cli_account"] = status["detail"]
        try:
            schema = agent._object({"status": {"type": "string", "enum": ["connected"]}})
            answer = agent._request_structured('{"task":"Return connected to test this connection."}', schema,
                agent._settings(self.model_settings(item["id"])), instructions="Return the requested JSON. Do not use tools.", name="dft_connection")
            agent._validate_json(answer, schema)
            return {"auth_status": "available", "connection_status": "connected", "detail": "The selected model answered the test request.",
                    "model_usage": getattr(answer, "usage", None), **details}
        except (agent.AgentError, credentials.CredentialError, ValueError) as error:
            # SDK errors are already classified without raw response bodies.
            if isinstance(error, agent.AgentError):
                detail = str(error)
                if provider in _CLI and "too old" in detail:
                    detail += f" Installed: {details.get('cli_version', 'unknown version')} at {details.get('cli_path', '')}."
                if error.code:
                    details["error_code"] = error.code
            elif isinstance(error, credentials.CredentialError):
                detail = str(error)
            else:
                detail = "Check the saved connection and its credential."
            return {"auth_status": "unknown", "connection_status": "error", "detail": detail, **details}

    def detect_auth(self, identifier):
        """Try the user's keys and agent first; report what the cluster accepts instead.

        No password is read. The result tells the setup flow whether to ask for one.
        """
        from .transport import AuthenticationError, ConnectionFailed, HostKeyError, SSHTransport, TransportError, VerificationRequired

        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        remote = None
        try:
            remote = SSHTransport(self._transport_config(item), password="")
            remote.verify(timeout=max(10, item["settings"]["connect_timeout"] + 5))
            return {"status": "connected", "methods": (), "detail": "Your SSH key or agent signed in."}
        except VerificationRequired as error:
            if item["settings"]["auth_mode"] in {"agent", "key", "password"}:
                self._patch_settings(identifier, auth_mode="code")
            if platform.system() != "Windows":
                self.ensure_control_path(identifier)
            return {"status": "code", "methods": ("keyboard-interactive",), "prompt": error.prompt,
                    "detail": "The cluster also asks for a login code."}
        except AuthenticationError as error:
            methods = tuple(error.methods)
            if "password" in methods or "keyboard-interactive" in methods or not methods:
                return {"status": "password", "methods": methods, "detail": "The cluster asks for a password."}
            return {"status": "key", "methods": methods,
                    "detail": "The cluster only accepts SSH keys. Add your key to an SSH agent or set its file in the options."}
        except HostKeyError as error:
            return {"status": "host_key", "methods": (), "detail": str(error)}
        except ConnectionFailed as error:
            return {"status": "unreachable", "methods": (), "detail": str(error)}
        except (TransportError, ValueError, OSError) as error:
            return {"status": "error", "methods": (), "detail": "SSH did not complete. " + str(error)[:300]}
        finally:
            if remote is not None:
                remote.close()

    def connect(self, identifier, code=""):
        """Sign in with the saved method. Raises VerificationRequired when a login code is needed.

        For "code" connections the authenticated session stays in the Windows
        broker or Unix multiplexing socket, so later workers need no saved code.
        """
        from .transport import SSHTransport, VerificationRequired

        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        mode = item["settings"]["auth_mode"]
        if code and not re.fullmatch(r"[A-Za-z0-9 .\-]{1,64}", code):
            raise ValueError("Enter the login code exactly as shown by your authenticator.")
        if code and mode == "password":
            item = self._patch_settings(identifier, auth_mode="code")
            mode = "code"
        if mode == "code" and platform.system() != "Windows":
            self.ensure_control_path(identifier)
            item = self.get(identifier)
        remote = None
        try:
            if mode in PASSWORD_MODES:
                remote = SSHTransport(self._transport_config(item), password=self.ssh_password(identifier),
                                      code=code, master=(mode == "code"))
            else:
                remote = SSHTransport(self._transport_config(item), password="")
            remote.verify(timeout=max(20, item["settings"]["connect_timeout"] + 10))
            return {"status": "connected", "auth_mode": mode, "detail": "Signed in."}
        except VerificationRequired as error:
            if mode == "password":
                # Same account and password; the cluster additionally wants a code.
                self._patch_settings(identifier, auth_mode="code")
                if platform.system() != "Windows":
                    self.ensure_control_path(identifier)
            raise
        finally:
            if remote is not None:
                remote.close()

    def _probe_hpc(self, item):
        from .transport import (AuthenticationError, ConnectionFailed, HostKeyError, SSHTransport, TransportError,
                                VerificationRequired, classify_failure)

        root = item["settings"]["remote_base"]
        revision = item["revision"]

        def patch(**changes):
            nonlocal revision
            revision = self._patch_settings(item["id"], **changes)["revision"]

        command = "\n".join([
            "printf '__DFT_USER__\\n'; id -un",
            "printf '__DFT_HOME__\\n'; printf '%s\\n' \"$HOME\"",
            "printf '__DFT_TOOLS__\\n'",
            "for tool in python3 sbatch squeue sacct scancel sinfo qsub qstat qdel vasp_std vasp_ncl pw.x ph.x wannier90.x python conda; do command -v \"$tool\" 2>/dev/null; done",
            "printf '__DFT_MODULES__\\n'; module -t list 2>&1 | head -40",
            "printf '__DFT_QUEUES__\\n'",
            "if command -v sinfo >/dev/null 2>&1; then sinfo -h -o '%P|%a|%l|%D|%G' 2>/dev/null | head -50; fi",
            "if command -v qstat >/dev/null 2>&1; then qstat -Q 2>/dev/null | head -50; fi",
            "printf '__DFT_WORKDIR__\\n'",
            ("test -d " + shlex.quote(root) + " && test -w " + shlex.quote(root) + " && printf 'writable\\n'") if root else "printf 'not configured\\n'",
        ])
        remote = None
        try:
            remote = SSHTransport(self._transport_config(item), password=self.ssh_password(item["id"]))
            response = remote.run(command)
            failure = classify_failure(response)
            if failure is not None:
                raise failure
            sections, active = {}, None
            for line in response.stdout.splitlines():
                match = re.fullmatch(r"__DFT_(USER|HOME|TOOLS|MODULES|QUEUES|WORKDIR)__", line)
                if match:
                    active = match[1].lower()
                    sections[active] = []
                elif active and len(sections[active]) < 100:
                    sections[active].append(line[:300])
            if not sections.get("user"):
                raise TransportError("The remote identity could not be checked.")
            tools = {Path(path).name for path in sections.get("tools", [])}
            schedulers = [name for name, required in (("slurm", {"sbatch", "squeue", "sacct", "scancel"}), ("pbs", {"qsub", "qstat", "qdel"})) if required <= tools]
            home = sections.get("home", [])
            home = home[0] if len(home) == 1 else ""
            if not home.startswith("/") or ".." in PurePosixPath(home).parts or re.search(r"[\x00-\x1f]", home):
                home = ""
            suggested = str(PurePosixPath(home) / "dft-agent") if home else None
            if not root and suggested:
                # The default work folder is the user's home; it can be changed later.
                patch(remote_base=suggested)
                root = suggested
            if item["settings"]["scheduler"] == "auto" and len(schedulers) == 1:
                patch(scheduler=schedulers[0])
            return {"auth_status": "available", "connection_status": "connected", "detail": "SSH connected. Submission permissions have not been tested.",
                    "schedulers": schedulers, "submission_verified": False, "probe_sections": sections,
                    "home_directory": home or None, "remote_base": root or None,
                    "suggested_remote_base": suggested, "revision": revision,
                    "workdir_writable": "writable" in sections.get("workdir", [])}
        except VerificationRequired as error:
            if item["settings"]["auth_mode"] in {"password", "key", "agent"}:
                patch(auth_mode="code")
                if platform.system() != "Windows":
                    self.ensure_control_path(item["id"])
                revision = self.get(item["id"])["revision"]
            return {"auth_status": "required", "connection_status": "untested", "needs": "code", "revision": revision,
                    "prompt": error.prompt, "detail": "The cluster asks for a login code. Enter the code to sign in."}
        except AuthenticationError as error:
            needs = "password" if ("password" in error.methods or "keyboard-interactive" in error.methods
                                   or item["settings"]["auth_mode"] in PASSWORD_MODES) else "key"
            return {"auth_status": "required", "connection_status": "untested", "needs": needs, "methods": list(error.methods),
                    "detail": str(error)}
        except HostKeyError as error:
            return {"auth_status": "unknown", "connection_status": "error", "needs": "host_key", "detail": str(error)}
        except ConnectionFailed as error:
            return {"auth_status": "unknown", "connection_status": "error", "needs": "network", "detail": str(error)}
        except credentials.CredentialError as error:
            return {"auth_status": "required", "connection_status": "untested", "needs": "password", "detail": str(error)}
        except (TransportError, ValueError, OSError):
            return {"auth_status": "unknown", "connection_status": "error", "detail": "SSH did not complete. Check authentication, VPN, host access and the selected work folder."}
        finally:
            if remote is not None:
                remote.close()

    def discover_environments(self, identifier, *, potcar_root=None):
        """Read the cluster once and create default VASP/QE environments.

        Nothing is submitted. Values come from the cluster's own sinfo, module
        and PATH answers; the user reviews them in each step before a run.
        """
        from . import discovery
        from .transport import SSHTransport, TransportError

        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        if item["connection_status"] != "connected":
            raise ValueError("Test the cluster connection first.")
        settings = item["settings"]
        scheduler = self._scheduler_for(item)
        launcher = "mpiexec" if scheduler == "pbs" else "srun"
        config = discovery.probe_config(settings["host"], settings["user"], settings["port"],
                                        connect_timeout=settings["connect_timeout"], ssh_control_path=settings["ssh_control_path"],
                                        scheduler=scheduler)
        remote = SSHTransport(self._transport_config(item), password=self.ssh_password(identifier))
        try:
            report = discovery.probe_cluster(config, transport=remote)
        except (discovery.DiscoveryError, TransportError) as error:
            raise ValueError("The cluster could not be read: " + str(error)[:300]) from None
        finally:
            remote.close()
        facts, suggested, notes = report["facts"], dict(report["suggested"]), list(report["notes"])
        tools = {Path(path).name for path in (item["probe"] or {}).get("probe_sections", {}).get("tools", [])}
        existing = {environment["engine"] for environment in self.environments(identifier)}
        created, missing = [], {}
        base = {"partition": suggested.get("partition", ""), "account": suggested.get("account", "")}
        if potcar_root:
            suggested["potcar_root"] = potcar_root
        vasp_missing = [name for name, present in (("partition", base["partition"]), ("potcar_root", suggested.get("potcar_root"))) if not present]
        vasp_seen = bool(facts.get("executables") or facts.get("modules") or suggested.get("potcar_root"))
        if "vasp" not in existing and vasp_seen:
            if vasp_missing:
                missing["vasp"] = vasp_missing
            else:
                values = {**base, "potcar_root": suggested["potcar_root"],
                          "vasp_command": suggested.get("vasp_command") or f"{launcher} vasp_std",
                          "vasp_ncl_command": suggested.get("vasp_ncl_command", ""),
                          "setup_commands": suggested.get("setup_commands", []), "walltime": "02:00:00"}
                created.append(self.create_environment(identifier, "VASP · " + base["partition"], "vasp",
                                                       {key: value for key, value in values.items() if value not in ("", [], None)}))
        qe_software = (facts.get("codes") or {}).get("qe") or {}
        qe_executables, qe_modules = qe_software.get("executables") or {}, qe_software.get("modules") or []
        qe_seen = bool("pw.x" in tools or qe_executables or qe_modules)
        if "qe" not in existing and qe_seen:
            if not base["partition"]:
                missing["qe"] = ["partition"]
            else:
                values = {**base, "qe_command": f"{launcher} pw.x",
                          "qe_pseudo_root": str(PurePosixPath(settings["remote_base"] or "/tmp") / "pseudos"), "walltime": "02:00:00"}
                if qe_modules and not qe_executables and "pw.x" not in tools:
                    values["setup_commands"] = [f"module load {discovery.preferred_module(qe_modules)}"]
                if qe_executables.get("pw.x"):
                    values["qe_bin_dir"] = str(PurePosixPath(qe_executables["pw.x"]).parent)
                created.append(self.create_environment(identifier, "Quantum ESPRESSO · " + base["partition"], "qe",
                                                       {key: value for key, value in values.items() if value not in ("", [], None)}))
        codes_found = []
        for engine, entry in sorted((facts.get("codes") or {}).items()):
            if engine == "qe":
                continue
            executables, modules = entry.get("executables") or {}, entry.get("modules") or []
            if not executables and not modules:
                continue
            codes_found.append(engine)
            if engine in existing:
                continue
            if not base["partition"]:
                missing[engine] = ["partition"]
                continue
            program = next((name for name in _CODE_PROGRAMS[engine] if name in executables), _CODE_PROGRAMS[engine][0])
            values = {**base, "engine_command": f"{launcher} {program}",
                      "engine_library_root": site_library_root(engine, settings["remote_base"]),
                      "setup_commands": [] if executables else [f"module load {discovery.preferred_module(modules)}"], "walltime": "02:00:00"}
            if engine == "elk":
                values.update(tasks=1, cpus_per_task=8)
            created.append(self.create_environment(identifier, f"{_CODE_LABELS[engine]} · {base['partition']}", engine,
                                                   {key: value for key, value in values.items() if value not in ("", [], None)}))
        if not vasp_seen and not qe_seen and not codes_found:
            notes.append("No VASP, Quantum ESPRESSO, ABINIT, CP2K or Elk was found on the login node. Install an open-source code from the app, or add a software environment manually.")
        result = {"facts": facts, "suggested": suggested, "notes": notes, "created": created, "missing": missing,
                  "vasp_found": vasp_seen, "qe_found": qe_seen, "codes_found": codes_found}
        with self.workspace._transaction() as db:
            row = self._row(db, identifier, active=True)
            probe = json.loads(row["probe"]) if row["probe"] else {}
            probe["discovery"] = {"suggested": suggested, "notes": notes, "partitions": facts.get("partitions", [])[:20],
                                  "vasp_found": vasp_seen, "qe_found": qe_seen, "codes_found": codes_found, "checked_at": _now()}
            db.execute("UPDATE connections SET probe=?,updated_at=? WHERE id=?", (_object(probe), _now(), identifier))
            self._event(db, identifier, "discovered", {"created": [environment["id"] for environment in created]})
        return result

    def _store_probe(self, identifier, key, value):
        """Keep a sub-record (discovery, QE installation) in the probe JSON without touching the test result."""
        with self.workspace._transaction() as db:
            row = self._row(db, identifier, active=True)
            probe = json.loads(row["probe"]) if row["probe"] else {}
            probe[key] = value
            db.execute("UPDATE connections SET probe=?,updated_at=? WHERE id=?", (_object(probe), _now(), identifier))
            self._event(db, identifier, key + ".updated", {})

    def _scheduler_for(self, item, *, allow_unresolved=False):
        scheduler = item["settings"]["scheduler"]
        if scheduler == "auto":
            detected = (item["probe"] or {}).get("schedulers", [])
            if len(detected) != 1:
                if allow_unresolved:
                    # Syntax validation may precede login; execution may not.
                    return "slurm"
                raise ValueError("Choose Slurm or PBS in the cluster settings before continuing.")
            scheduler = detected[0]
        return scheduler

    def _qe_base_config(self, item, *, partition, account="", setup_commands=(), qe_bin_dir="", extra_sbatch=(), extra_pbs=()):
        """A QE cluster profile for installation and pseudopotentials; never for a project run."""
        from .config import ClusterConfig

        settings = item["settings"]
        if not settings["remote_base"]:
            raise ValueError("Set the connection's work folder first.")
        return ClusterConfig(engine="qe", host=settings["host"], user=settings["user"], port=settings["port"],
                             connect_timeout=settings["connect_timeout"], remote_root=settings["remote_base"],
                             partition=partition, account=account or "", setup_commands=list(setup_commands),
                             extra_sbatch=list(extra_sbatch), extra_pbs=list(extra_pbs),
                             qe_command=("mpiexec pw.x" if self._scheduler_for(item) == "pbs" else "srun pw.x"), qe_bin_dir=qe_bin_dir or "",
                             qe_pseudo_root=str(PurePosixPath(settings["remote_base"]) / "pseudos"),
                             scheduler=self._scheduler_for(item), ssh_control_path=settings["ssh_control_path"],
                             ssh_jump_host=settings["jump_host"], ssh_identity_file=settings["identity_file"],
                             connection_id=item["id"], connection_home=str(self.home))

    def _qe_transport(self, item):
        from .transport import SSHTransport

        return SSHTransport(self._transport_config(item), password=self.ssh_password(item["id"]))

    def qe_install_plan(self, identifier, *, build_cpus=8, include_epw=False, partition=None, account=None, setup_commands=None,
                        extra_sbatch=(), extra_pbs=()):
        """Detect a toolchain and plan a managed QE build. Nothing is downloaded or submitted."""
        from . import qe_setup

        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        if item["connection_status"] != "connected":
            raise ValueError("Test the cluster connection first.")
        probe = item["probe"] or {}
        previous = probe.get("qe_install") or {}
        if previous.get("receipt") and previous.get("status") not in {"failed", "installed", "cancelled"}:
            raise ValueError("A QE build is already tracked. Check its status before planning another installation.")
        attempt = previous.get("attempt", 0) + 1 if previous.get("status") == "failed" else previous.get("attempt", 0)
        suggested = (probe.get("discovery") or {}).get("suggested", {})
        partition = (partition or suggested.get("partition") or "").strip()
        if not partition:
            raise ValueError("Choose a partition for the build job; the cluster listed none.")
        account = suggested.get("account", "") if account is None else account.strip()
        if setup_commands is not None:
            if not isinstance(setup_commands, list) or any(not isinstance(line, str) or re.search(r"[\x00\r\n]", line) for line in setup_commands):
                raise ValueError("Give module commands as a list of single lines.")
            from .secret_guard import reject_ordinary_text
            reject_ordinary_text("\n".join(setup_commands))
        remote = self._qe_transport(item)
        try:
            base = self._qe_base_config(item, partition=partition, account=account, setup_commands=setup_commands or (),
                                        extra_sbatch=extra_sbatch, extra_pbs=extra_pbs)
            toolchain = qe_setup.detect_toolchain(base, transport=remote)
            config = self._qe_base_config(item, partition=partition, account=account, setup_commands=toolchain["setup_commands"],
                                          extra_sbatch=extra_sbatch, extra_pbs=extra_pbs)
            plan = qe_setup.install_plan(config, build_jobs=build_cpus, include_epw=include_epw, attempt=attempt)
        finally:
            remote.close()
        record = {"status": "planned", "plan": plan, "plan_id": plan["id"], "partition": partition, "account": account,
                  "setup_commands": toolchain["setup_commands"], "toolchain": {key: toolchain[key] for key in ("ready", "mpi_fortran", "notes", "modules_tried")},
                  "planned_at": _now(), "attempt": attempt}
        self._store_probe(identifier, "qe_install", record)
        return {"plan_id": plan["id"], "version": plan["version"], "prefix": plan["prefix"], "partition": partition, "account": account,
                "build_cpus": build_cpus, "walltime": plan["walltime"], "setup_commands": toolchain["setup_commands"],
                "extra_sbatch": plan.get("extra_sbatch", []), "extra_pbs": plan.get("extra_pbs", []),
                "toolchain_ready": toolchain["ready"], "notes": toolchain["notes"], "status": "planned",
                "next": "Review the partition, account and module commands, then start the build." if toolchain["ready"]
                        else "Add the module commands that provide an MPI Fortran compiler, then plan again."}

    def qe_install_start(self, identifier, plan_id):
        """Submit the reviewed build job. Repeated calls reuse the same receipt."""
        from . import qe_setup

        item = self.get(identifier)
        record = (item["probe"] or {}).get("qe_install")
        if not record or record.get("plan_id") != plan_id:
            raise ValueError("Plan the installation first, then start the plan you reviewed.")
        if not record["toolchain"]["ready"]:
            raise ValueError("No MPI Fortran compiler was found. Add the cluster's module commands and plan again.")
        config = self._qe_base_config(item, partition=record["partition"], account=record["account"], setup_commands=record["setup_commands"],
                                      extra_sbatch=record["plan"].get("extra_sbatch", []), extra_pbs=record["plan"].get("extra_pbs", []))
        remote = self._qe_transport(item)
        try:
            receipt = qe_setup.start_install(config, record["plan"], transport=remote)
        finally:
            remote.close()
        record.update(receipt=receipt, status=receipt.get("status", "submitted"), job_id=receipt.get("job_id"), started_at=_now())
        record.pop("error", None)
        self._store_probe(identifier, "qe_install", record)
        return {"status": record["status"], "job_id": record.get("job_id"), "plan_id": plan_id, "work_dir": receipt.get("work_dir"),
                "prefix": record["plan"]["prefix"], "extra_sbatch": record["plan"].get("extra_sbatch", []),
                "extra_pbs": record["plan"].get("extra_pbs", [])}

    def qe_install_status(self, identifier):
        """Read the build's state; when it finished, create the QE environment."""
        from . import qe_setup

        item = self.get(identifier)
        record = (item["probe"] or {}).get("qe_install")
        if not record:
            return {"status": "not_planned"}
        summary_keys = ("status", "plan_id", "job_id", "prefix", "bin_dir", "error", "environment_id", "scheduler_state")
        resources = {field: record["plan"].get(field, []) for field in ("extra_sbatch", "extra_pbs")}
        if record.get("status") == "installed" or not record.get("receipt"):
            return {**{key: record.get(key) or (record["plan"]["prefix"] if key == "prefix" else None) for key in summary_keys}, **resources}
        config = self._qe_base_config(item, partition=record["partition"], account=record["account"], setup_commands=record["setup_commands"],
                                      extra_sbatch=record["plan"].get("extra_sbatch", []), extra_pbs=record["plan"].get("extra_pbs", []))
        remote = self._qe_transport(item)
        try:
            status = qe_setup.poll_install(config, record["receipt"], transport=remote)
        finally:
            remote.close()
        record.update(status=status.get("status", "unknown"), scheduler_state=status.get("scheduler_state"),
                      error=status.get("error"), checked_at=_now(), logs=status.get("logs"))
        if record["status"] == "installed":
            bin_dir = status.get("bin_dir") or record["plan"]["bin_dir"]
            record["bin_dir"] = bin_dir
            existing = [environment for environment in self.environments(identifier)
                        if environment["engine"] == "qe" and environment["config"].get("qe_bin_dir") == bin_dir]
            if existing and not record.get("environment_id"):
                record["environment_id"] = existing[0]["id"]
            elif not record.get("environment_id"):
                environment = self.create_environment(identifier, f"Quantum ESPRESSO {record['plan']['version']} · {record['partition']}", "qe", {
                    key: value for key, value in {"partition": record["partition"], "account": record["account"],
                        "setup_commands": record["setup_commands"], "qe_command": config.qe_command, "qe_bin_dir": bin_dir,
                        "qe_pseudo_root": config.qe_pseudo_root, "walltime": "02:00:00",
                        "extra_sbatch": config.extra_sbatch, "extra_pbs": config.extra_pbs}.items() if value not in ("", [], None)})
                record["environment_id"] = environment["id"]
        self._store_probe(identifier, "qe_install", record)
        return {**{key: record.get(key) or (record["plan"]["prefix"] if key == "prefix" else None) for key in summary_keys}, **resources}

    # --- ABINIT, CP2K and Elk: managed installs and pinned libraries ---------

    def _engine_base_config(self, item, engine, *, partition, account="", extra_sbatch=(), extra_pbs=()):
        """A cluster profile for an installation; never used for a project run."""
        from .config import ClusterConfig

        settings = item["settings"]
        if not settings["remote_base"]:
            raise ValueError("Set the connection's work folder first.")
        return ClusterConfig(engine=engine, host=settings["host"], user=settings["user"], port=settings["port"],
                             connect_timeout=settings["connect_timeout"], remote_root=settings["remote_base"],
                             partition=partition, account=account or "", engine_command=_CODE_PROGRAMS[engine][0],
                             extra_sbatch=list(extra_sbatch), extra_pbs=list(extra_pbs),
                             engine_library_root=site_library_root(engine, settings["remote_base"]),
                             scheduler=self._scheduler_for(item), ssh_control_path=settings["ssh_control_path"],
                             ssh_jump_host=settings["jump_host"], ssh_identity_file=settings["identity_file"],
                             connection_id=item["id"], connection_home=str(self.home))

    def _connected_hpc(self, identifier):
        item = self.get(identifier)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Select an active HPC connection.")
        if item["connection_status"] != "connected":
            raise ValueError("Test the cluster connection first.")
        return item

    @staticmethod
    def _install_summary(engine, record):
        plan = record.get("plan") or {}
        summary = {key: record.get(key) for key in ("status", "plan_id", "job_id", "error", "environment_id", "scheduler_state")}
        summary.update(engine=engine, prefix=plan.get("prefix"), version=plan.get("version"), platform=plan.get("platform"),
                       extra_sbatch=plan.get("extra_sbatch", []), extra_pbs=plan.get("extra_pbs", []),
                       glibc=plan.get("glibc"), mpi=plan.get("mpi"), partition=record.get("partition"), account=record.get("account"),
                       download_mb=round(plan.get("download_size", 0) / 1e6) if plan else None, progress=record.get("progress"),
                       smoke=record.get("smoke"), retained_prefix=record.get("retained_prefix"))
        return summary

    def engine_install_plan(self, identifier, engine, *, build_cpus=4, partition=None, account=None, extra_sbatch=(), extra_pbs=()):
        """Read the cluster's architecture and plan a pinned installation. Nothing is downloaded or submitted."""
        from . import engine_setup

        if engine not in _CODE_LABELS:
            raise ValueError("Choose ABINIT, CP2K or Elk.")
        item = self._connected_hpc(identifier)
        suggested = ((item["probe"] or {}).get("discovery") or {}).get("suggested", {})
        partition = (partition or suggested.get("partition") or "").strip()
        if not partition:
            raise ValueError("Choose a partition for the installation job; the cluster listed none.")
        account = suggested.get("account", "") if account is None else account.strip()
        base = self._engine_base_config(item, engine, partition=partition, account=account,
                                        extra_sbatch=extra_sbatch, extra_pbs=extra_pbs)
        remote = self._qe_transport(item)
        try:
            platform = engine_setup.detect_platform(base, transport=remote)
        finally:
            remote.close()
        previous = (item["probe"] or {}).get(f"{engine}_install") or {}
        # After a failed attempt, plan a new one; it reuses an environment the old one created.
        attempt = previous.get("attempt", 0) + 1 if previous.get("status") == "failed" else previous.get("attempt", 0)
        plan = engine_setup.install_plan(base, engine, platform, build_cpus=build_cpus, attempt=attempt)
        if previous.get("status") in {"staging", "submitted", "running", "installed"}:
            if previous.get("plan_id") != plan["id"]:
                raise ValueError("An installation already exists. Check its status before changing the plan.")
            return {**self._install_summary(engine, previous), "walltime": plan["walltime"], "build_cpus": build_cpus,
                    "command": plan["command"], "library_root": plan["library_root"],
                    "next": "Continue the existing installation."}
        record = {"status": "planned", "plan": plan, "plan_id": plan["id"], "partition": partition, "account": account,
                  "platform": platform, "planned_at": _now(), "attempt": attempt}
        self._store_probe(identifier, f"{engine}_install", record)
        return {**self._install_summary(engine, record), "walltime": plan["walltime"], "build_cpus": build_cpus,
                "command": plan["command"], "library_root": plan["library_root"],
                "next": "Review the partition and account, then start the installation. Only the user can start it."}

    def engine_install_start(self, identifier, engine, plan_id, *, wait=900):
        """Download the pinned packages and submit the reviewed installation job."""
        from . import engine_setup

        item = self.get(identifier)
        record = (item["probe"] or {}).get(f"{engine}_install")
        if not record or record.get("plan_id") != plan_id:
            raise ValueError("Plan the installation first, then start the plan you reviewed.")
        base = self._engine_base_config(item, engine, partition=record["partition"], account=record["account"],
                                        extra_sbatch=record["plan"].get("extra_sbatch", []), extra_pbs=record["plan"].get("extra_pbs", []))
        remote = self._qe_transport(item)
        try:
            receipt = engine_setup.start_install(base, record["plan"], transport=remote, wait=wait)
        finally:
            remote.close()
        record.update(receipt=receipt, status=receipt.get("status", "submitted"), job_id=receipt.get("job_id"),
                      progress=receipt.get("progress"), started_at=_now())
        record.pop("error", None)
        self._store_probe(identifier, f"{engine}_install", record)
        return self._install_summary(engine, record)

    def engine_install_status(self, identifier, engine):
        """Read the installation; when it passed its silicon test, create the environment."""
        from . import engine_setup

        if engine not in _CODE_LABELS:
            raise ValueError("Choose ABINIT, CP2K or Elk.")
        item = self.get(identifier)
        record = (item["probe"] or {}).get(f"{engine}_install")
        if not record:
            return {"engine": engine, "status": "not_planned"}
        receipt = record.get("receipt")
        if record.get("status") == "installed" or not receipt:
            return self._install_summary(engine, record)
        if receipt.get("status") == "staging":
            # Continue the same reviewed plan; report progress instead of waiting long.
            return self.engine_install_start(identifier, engine, record["plan_id"], wait=45)
        base = self._engine_base_config(item, engine, partition=record["partition"], account=record["account"],
                                        extra_sbatch=record["plan"].get("extra_sbatch", []), extra_pbs=record["plan"].get("extra_pbs", []))
        remote = self._qe_transport(item)
        try:
            status = engine_setup.poll_install(base, record["plan"], receipt, transport=remote)
        finally:
            remote.close()
        record.update(status=status.get("status", "unknown"), scheduler_state=status.get("scheduler_state"),
                      error=status.get("error"), checked_at=_now(), logs=status.get("logs"), smoke=status.get("smoke"),
                      retained_prefix=status.get("retained_prefix"))
        if record["status"] == "installed" and not record.get("environment_id"):
            plan = record["plan"]
            library = {}
            if engine == "abinit":
                library = {download["name"]: {"sha256": download["sha256"], "element": "Si", "library": "pbe_sr"}
                           for download in plan["downloads"] if download["role"] == "library"}
            else:
                library = {entry["name"]: {"sha256": entry["sha256"], "library": entry["library"]}
                           for entry in engine_setup.plan_library(engine, [])}
            values = {"partition": record["partition"], "account": record["account"], "setup_commands": plan["setup_commands"],
                      "engine_command": plan["command"], "engine_library_root": plan["library_root"], "engine_library": library,
                      "walltime": "02:00:00", "extra_sbatch": plan.get("extra_sbatch", []), "extra_pbs": plan.get("extra_pbs", [])}
            values.update({"tasks": 1, "cpus_per_task": 8} if engine == "elk" else {"tasks": 8})
            environment = self.create_environment(identifier, f"{_CODE_LABELS[engine]} {plan['version']} · {record['partition']}", engine,
                                                  {key: value for key, value in values.items() if value not in ("", [], None)})
            record["environment_id"] = environment["id"]
        self._store_probe(identifier, f"{engine}_install", record)
        return self._install_summary(engine, record)

    def build_reference(self, environment_id):
        """The official manual and examples for a QE, Elk, ABINIT or CP2K environment; CP2K's keywords come from its program."""
        from . import references

        environment = self.environment(environment_id)
        engine = environment["engine"]
        if engine not in references.ENGINES:
            raise ValueError("References are available for QE, Elk, ABINIT and CP2K environments.")
        config = self._environment_config(environment)
        if engine != "cp2k" and references.environment_version(engine, config):
            built = references.build_for_environment(config)
        else:
            remote = self._qe_transport(self.get(environment["connection_id"]))
            try:
                built = references.build_for_environment(config, transport=remote)
            finally:
                remote.close()
        return {"environment_id": environment_id, "built": built, **references.status(engine, config)}

    def library_status(self, environment_id, elements, *, soc=False, functional="PBE"):
        """Library files a calculation on these elements needs that the environment has not recorded."""
        from . import engine_setup

        environment = self.environment(environment_id)
        if environment["engine"] not in _CODE_LABELS:
            raise ValueError("Libraries belong to ABINIT, CP2K or Elk environments.")
        recorded = environment["config"].get("engine_library") or {}
        needed = engine_setup.plan_library(environment["engine"], list(elements), soc=soc, functional=functional)
        return {"missing": sorted({item.get("element") or item["name"] for item in needed
                                   if (recorded.get(item["name"]) or {}).get("sha256") != item["sha256"]}),
                "files": [item["name"] for item in needed]}

    def install_engine_library(self, environment_id, elements, *, soc=False, functional="PBE"):
        """Install pinned pseudopotentials, basis files or species and record their hashes."""
        from . import engine_setup

        environment = self.environment(environment_id)
        if environment["engine"] not in _CODE_LABELS:
            raise ValueError("Libraries belong to ABINIT, CP2K or Elk environments.")
        symbols = []
        for element in elements:
            if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
                raise ValueError("Give element symbols such as Si or O.")
            if element not in symbols:
                symbols.append(element)
        if not symbols and environment["engine"] != "cp2k":
            raise ValueError("Name at least one element.")
        config = self._environment_config(environment)
        item = self.get(environment["connection_id"])
        remote = self._qe_transport(item)
        try:
            result = engine_setup.ensure_library(config, symbols, transport=remote, soc=bool(soc), functional=functional)
        finally:
            remote.close()
        mapping = dict(environment["config"].get("engine_library") or {})
        mapping.update(result["files"])
        self.update_environment(environment_id, {**environment["config"], "engine_library": mapping})
        return {"environment_id": environment_id, "library_root": result["library_root"], "installed": sorted(result["files"])}

    def _environment_config(self, environment):
        """ClusterConfig for an environment outside a project, for pseudopotential installation."""
        from .config import ClusterConfig

        item = self.get(environment["connection_id"])
        if item["revoked"] or item["deleted"]:
            raise ValueError("The environment's HPC connection is revoked.")
        settings = item["settings"]
        values = {key: value for key, value in environment["config"].items() if key != "command"}
        values.update(engine=environment["engine"], host=settings["host"], user=settings["user"], port=settings["port"],
                      connect_timeout=settings["connect_timeout"], remote_root=settings["remote_base"] or "/tmp",
                      ssh_control_path=settings["ssh_control_path"], ssh_jump_host=settings["jump_host"],
                      ssh_identity_file=settings["identity_file"], scheduler=self._scheduler_for(item),
                      connection_id=item["id"], connection_home=str(self.home))
        return ClusterConfig(**values)

    def update_environment(self, environment_id, config):
        """Replace an environment's reviewed settings; host and sign-in stay on the connection."""
        environment = self.environment(environment_id)
        item = self.get(environment["connection_id"])
        checked = self._check_environment(item, environment["engine"], config)
        with self.workspace._transaction() as db:
            db.execute("UPDATE environments SET config=?,revision=revision+1,updated_at=? WHERE id=?",
                       (_object(checked), _now(), environment_id))
            self._event(db, environment["connection_id"], "environment.updated", {"environment_id": environment_id})
        return self.environment(environment_id)

    def install_pseudopotentials(self, environment_id, elements, *, library="sssp"):
        """Install public PBE UPFs for the elements into the environment's UPF folder and record them."""
        from . import qe_setup

        environment = self.environment(environment_id)
        if environment["engine"] != "qe":
            raise ValueError("Pseudopotentials belong to a Quantum ESPRESSO environment.")
        symbols = []
        for element in elements:
            if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
                raise ValueError("Give element symbols such as Si or O.")
            if element not in symbols:
                symbols.append(element)
        if not symbols:
            raise ValueError("Name at least one element.")
        config = self._environment_config(environment)
        if not config.qe_pseudo_root or config.qe_pseudo_root == "/":
            raise ValueError("Set the environment's UPF folder first.")
        item = self.get(environment["connection_id"])
        remote = self._qe_transport(item)
        try:
            result = qe_setup.ensure_pseudos(config, symbols, transport=remote, library=library)
        finally:
            remote.close()
        mapping = dict(environment["config"].get("qe_pseudopotentials") or {})
        mapping.update(result["config_pseudopotentials"])
        self.update_environment(environment_id, {**environment["config"], "qe_pseudopotentials": mapping})
        return {"environment_id": environment_id, "library": result.get("library", library), "pseudo_root": config.qe_pseudo_root,
                "elements": {element: {"filename": entry["filename"], "ecutwfc": entry["cutoff_wfc"], "ecutrho": entry["cutoff_rho"]}
                             for element, entry in result["elements"].items()}}

    def _check_environment(self, item, engine, config):
        from .config import ClusterConfig

        if engine not in {"vasp", "qe", "abinit", "cp2k", "elk", "wannier90", "python"}:
            raise ValueError("Choose a supported scientific program.")
        allowed = {field.name for field in fields(ClusterConfig)} - _TRANSPORT_FIELDS
        if engine in {"wannier90", "python"}:
            allowed |= {"command", "setup_commands"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("Keep host and authentication settings on the HPC connection.")
        _object(config)
        from .secret_guard import reject_ordinary_text
        reject_ordinary_text(json.dumps(config))
        if config.get("engine", engine) != engine:
            raise ValueError("The environment engine does not match its configuration.")
        connection = item["settings"]
        if engine in {"vasp", "qe", "abinit", "cp2k", "elk"}:
            ClusterConfig(**{**config, "engine": engine, "host": connection["host"],
                "user": connection["user"], "port": connection["port"],
                "remote_root": connection["remote_base"] or "/tmp",
                "scheduler": self._scheduler_for(item, allow_unresolved=True)})
        elif not isinstance(config.get("command"), str) or not config["command"].strip() or re.search(r"[\r\n\x00]", config["command"]):
            raise ValueError("Enter the program command on one line.")
        else:
            tokens = shlex.split(config['command'])
            if not tokens or any(token in {';', '|', '&&', '||', '>', '<'} for token in tokens):
                raise ValueError('Use an executable and its arguments for the program command.')
            ClusterConfig(**{**{k: v for k, v in config.items() if k not in {'command', 'engine'}},
                'engine': 'qe', 'qe_pseudo_root': '/', 'qe_command': config['command'],
                'host': connection['host'], 'user': connection['user'], 'port': connection['port'],
                'remote_root': connection['remote_base'] or '/tmp',
                'scheduler': self._scheduler_for(item, allow_unresolved=True)})
        return config

    def create_environment(self, connection_id, name, engine, config):
        from .config import ClusterConfig

        item = self.get(connection_id)
        if item["kind"] != "hpc" or item["revoked"] or item["deleted"]:
            raise ValueError("Choose an active HPC connection.")
        if engine not in {"vasp", "qe", "abinit", "cp2k", "elk", "wannier90", "python"}:
            raise ValueError("Choose a supported scientific program.")
        allowed = {field.name for field in fields(ClusterConfig)} - _TRANSPORT_FIELDS
        if engine in {"wannier90", "python"}:
            allowed |= {"command", "setup_commands"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("Keep host and authentication settings on the HPC connection.")
        _object(config)
        from .secret_guard import reject_ordinary_text
        reject_ordinary_text(json.dumps(config))
        if config.get("engine", engine) != engine:
            raise ValueError("The environment engine does not match its configuration.")
        if engine in {"vasp", "qe", "abinit", "cp2k", "elk"}:
            connection = item["settings"]
            ClusterConfig(**{**config, "engine": engine, "host": connection["host"],
                "user": connection["user"], "port": connection["port"],
                "remote_root": connection["remote_base"] or "/tmp",
                "scheduler": self._scheduler_for(item, allow_unresolved=True)})
        elif not isinstance(config.get("command"), str) or not config["command"].strip() or re.search(r"[\r\n\x00]", config["command"]):
            raise ValueError("Enter the program command on one line.")
        else:
            tokens = shlex.split(config['command'])
            if not tokens or any(token in {';', '|', '&&', '||', '>', '<'} for token in tokens):
                raise ValueError('Use an executable and its arguments for the program command.')
            connection = item['settings']
            # Reuse the validated scheduler/SSH resource profile. The script
            # adapter records its own engine; these QE fields are never used
            # to construct scientific inputs or launch pw.x.
            ClusterConfig(**{**{k: v for k, v in config.items() if k not in {'command', 'engine'}},
                'engine': 'qe', 'qe_pseudo_root': '/', 'qe_command': config['command'],
                'host': connection['host'], 'user': connection['user'], 'port': connection['port'],
                'remote_root': connection['remote_base'] or '/tmp',
                'scheduler': self._scheduler_for(item, allow_unresolved=True)})
        identifier, now = str(uuid.uuid4()), _now()
        with self.workspace._transaction() as db:
            self._row(db, connection_id, active=True)
            db.execute("INSERT INTO environments(id,connection_id,name,engine,config,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                       (identifier, connection_id, _text(name, "environment name"), engine, _object(config), now, now))
            self._event(db, connection_id, "environment.created", {"environment_id": identifier})
        return self.environment(identifier)

    def environment(self, identifier):
        with self.workspace._transaction(write=False) as db:
            row = db.execute("SELECT * FROM environments WHERE id=?", (_id(identifier),)).fetchone()
            if row is None:
                raise ValueError("Environment not found.")
            return {**dict(row), "config": json.loads(row["config"])}

    def environments(self, connection_id=None):
        with self.workspace._transaction(write=False) as db:
            if connection_id is not None:
                self._row(db, connection_id)
            return [{**dict(row), "config": json.loads(row["config"])} for row in db.execute(
                "SELECT * FROM environments WHERE (? IS NULL OR connection_id=?) ORDER BY name,id", (connection_id, connection_id))]

    def resolve_cluster(self, project_id, environment_id):
        resolved = self.resolve_environment(project_id, environment_id)
        if resolved['environment']['engine'] not in {'vasp', 'qe', 'abinit', 'cp2k', 'elk'}:
            raise ValueError('Use the reviewed script adapter for this environment.')
        return resolved['config']

    def resolve_environment(self, project_id, environment_id):
        """Resolve a project-bound resource profile without loading credentials."""
        from .config import ClusterConfig

        project = self.workspace.get_project(project_id)
        if project["archived"] or project["deleted"]:
            raise ValueError("Open an active project first.")
        environment = self.environment(environment_id)
        connections = {item["id"]: item for item in self.project_connections(project_id)}
        item = connections.get(environment["connection_id"])
        if item is None or item["revoked"] or item["deleted"]:
            raise ValueError("Connect this project to the environment's active HPC connection first.")
        settings = item["settings"]
        if not settings["remote_base"]:
            raise ValueError("Set the connection's remote work folder first.")
        scheduler = settings["scheduler"]
        if scheduler == "auto":
            detected = (item["probe"] or {}).get("schedulers", [])
            if len(detected) != 1:
                raise ValueError("Test this connection and select its scheduler.")
            scheduler = detected[0]
        settings_values = {k: v for k, v in environment['config'].items() if k != 'command'}
        if environment['engine'] in {'python', 'wannier90'}:
            settings_values.update(engine='qe', qe_pseudo_root='/', qe_command=environment['config']['command'])
        else:
            settings_values['engine'] = environment['engine']
        values = {**settings_values,
                  "host": settings["host"], "user": settings["user"], "port": settings["port"],
                  "connect_timeout": settings["connect_timeout"],
                  "remote_root": str(PurePosixPath(settings["remote_base"]) / "projects" / project_id),
                  "ssh_control_path": settings["ssh_control_path"],
                  "ssh_jump_host": settings["jump_host"], "ssh_identity_file": settings["identity_file"],
                  "scheduler": scheduler, "connection_id": item["id"], "connection_home": str(self.home)}
        return {'environment': environment, 'config': ClusterConfig(**values)}
