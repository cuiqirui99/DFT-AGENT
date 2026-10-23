"""Guided first-run setup: CLI discovery, typed SSH failures, in-app login codes.

Cluster access is simulated with fake transports; the OS keychain is a dict.
No test opens a network connection or runs a real CLI.
"""

import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import agent, cli_tools, connections, credentials, transport
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import parse_address
from vasp_slurm_agent.transport import (AuthenticationError, ConnectionFailed, HostKeyError, Result, SSHTransport,
                                        VerificationRequired, askpass_main, classify_failure, classify_prompt)


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


@pytest.fixture(autouse=True)
def _unix_backend_temp_paths_on_windows(monkeypatch):
    # The Unix-backend unit cases still use real files on the host test system.
    if os.name == "nt":
        original = connections.tempfile.mkdtemp
        def temporary(suffix=None, prefix=None, dir=None):
            return original(suffix=suffix, prefix=prefix, dir=None if dir == "/tmp" else dir)
        monkeypatch.setattr(connections.tempfile, "mkdtemp", temporary)


@pytest.fixture
def registry(tmp_path, keychain):
    return ConnectionRegistry(tmp_path / "home")


def widget(elements, label):
    return next(element for element in elements if element.label == label)


# --- prompts, askpass and failure classification ---------------------------

@pytest.mark.parametrize("prompt,kind", [
    ("researcher@hpc.invalid's password: ", "password"), ("Password:", "password"), ("请输入密码：", "password"),
    ("Verification code: ", "code"), ("One-time password (OATH) for researcher: ", "code"), ("Enter OTP:", "code"),
    ("Duo two-factor login", "code"), ("请输入动态口令:", "code"), ("验证码：", "code"), ("Passcode or option (1-3): ", "code"),
    ("Are you sure you want to continue connecting (yes/no/[fingerprint])? ", "other"),
])
def test_prompt_classification(prompt, kind):
    assert classify_prompt(prompt) == kind


def test_askpass_answers_only_matching_prompts(tmp_path):
    record = tmp_path / "prompt"
    environ = {"_DFT_AGENT_ASKPASS_PASSWORD": "pw-secret", "_DFT_AGENT_ASKPASS_PROMPT_FILE": str(record)}
    out = io.StringIO()
    assert askpass_main(["askpass", "hpc password: "], environ, out) == 0 and out.getvalue() == "pw-secret\n"
    out = io.StringIO()
    assert askpass_main(["askpass", "Verification code: "], environ, out) == 1 and out.getvalue() == ""
    assert json.loads(record.read_text()) == {"kind": "code", "prompt": "Verification code: "}
    record.unlink()
    out = io.StringIO()
    assert askpass_main(["askpass", "Verification code: "], {**environ, "_DFT_AGENT_ASKPASS_CODE": "123456"}, out) == 0
    assert out.getvalue() == "123456\n" and not record.exists()
    out = io.StringIO()
    assert askpass_main(["askpass", "continue connecting (yes/no)?"], environ, out) == 1 and out.getvalue() == ""
    assert json.loads(record.read_text())["kind"] == "other"
    # A code is never sent to a password prompt, even when no password is known.
    out = io.StringIO()
    assert askpass_main(["askpass", "Password: "], {"_DFT_AGENT_ASKPASS_CODE": "123456"}, out) == 1 and out.getvalue() == ""


@pytest.mark.parametrize("result,expected,methods", [
    (Result(255, "", "researcher@hpc.invalid: Permission denied (publickey,password,keyboard-interactive)."), AuthenticationError,
     ("publickey", "password", "keyboard-interactive")),
    (Result(255, "", "Permission denied, please try again.\nPermission denied (keyboard-interactive)."), AuthenticationError, ("keyboard-interactive",)),
    (Result(255, "", "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key verification failed."), HostKeyError, None),
    (Result(255, "", "ssh: connect to host hpc.invalid port 22: Connection refused"), ConnectionFailed, None),
    (Result(255, "", "ssh: Could not resolve hostname hpc.invalid: nodename nor servname provided"), ConnectionFailed, None),
    (Result(255, "", "", "Verification code: "), VerificationRequired, None),
    (Result(255, "", "", "Password: "), AuthenticationError, ("password",)),
    (Result(255, "", "kex_exchange_identification: read: Connection reset by peer"), ConnectionFailed, None),
])
def test_failure_classification(result, expected, methods):
    error = classify_failure(result)
    assert isinstance(error, expected)
    if methods is not None:
        assert error.methods == methods


def test_ordinary_command_failures_are_not_connection_errors():
    assert classify_failure(Result(0)) is None
    assert classify_failure(Result(1, "", "command not found")) is None
    assert isinstance(classify_failure(Result(255, "", "some other ssh failure")), transport.TransportError)


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(host="hpc.invalid", user="researcher", port=22, connect_timeout=5,
                           ssh_control_path="", ssh_jump_host="", ssh_identity_file="")


def test_master_mode_uses_persistent_socket_and_never_closes_it(config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transport.platform, "system", lambda: "Darwin")

    def run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(transport, "_run_process", run)
    with pytest.raises(transport.TransportError, match="control socket"):
        SSHTransport(config, password="pw", master=True)
    config.ssh_control_path = str(tmp_path / "login.sock")
    with SSHTransport(config, password="pw", code="654321", master=True) as remote:
        remote.verify()
        options = remote._options()
    assert "ControlMaster=auto" in options and f"ControlPersist={transport.LOGIN_PERSIST_SECONDS}" in options
    assert "BatchMode=no" in options and "ControlPersist=60" not in options
    argv, environment = calls[0]
    assert environment["_DFT_AGENT_ASKPASS_CODE"] == "654321" and environment["_DFT_AGENT_ASKPASS_PASSWORD"] == "pw"
    assert environment["_DFT_AGENT_ASKPASS_PROMPT_FILE"]
    assert "654321" not in " ".join(argv) and "pw" not in " ".join(argv)
    # Closing the master transport must not send -O exit: later commands and workers reuse the socket.
    assert not any("-O" in argv for argv, _ in calls)
    attached = SSHTransport(config, password="")
    assert attached._external_control and "ControlMaster=no" in attached._options()
    attached.close()


def test_master_mode_recreates_a_swept_socket_folder(config, tmp_path, monkeypatch):
    monkeypatch.setattr(transport.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(transport, "_run_process", lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    config.ssh_control_path = str(tmp_path / "swept" / "login.sock")
    with SSHTransport(config, password="pw", code="1", master=True) as remote:
        remote.verify()
    folder = tmp_path / "swept"
    assert folder.is_dir()
    assert folder.stat().st_mode & stat.S_IWRITE
    if os.name != "nt":
        assert folder.stat().st_mode & 0o777 == 0o700


def test_unanswered_prompt_becomes_verification_error(config, tmp_path, monkeypatch):
    monkeypatch.setattr(transport.platform, "system", lambda: "Darwin")

    def run(argv, **kwargs):
        Path(kwargs["env"]["_DFT_AGENT_ASKPASS_PROMPT_FILE"]).write_text(json.dumps({"kind": "code", "prompt": "Verification code: "}))
        return SimpleNamespace(returncode=255, stdout="", stderr="Permission denied (keyboard-interactive).")
    monkeypatch.setattr(transport, "_run_process", run)
    with SSHTransport(config, password="pw") as remote:
        result = remote.run("true")
        assert result.prompt == "Verification code: "
        with pytest.raises(VerificationRequired) as caught:
            remote.verify()
        assert caught.value.prompt == "Verification code: "
        # The prompt file is consumed; the next failure without it is an ordinary rejection.
        monkeypatch.setattr(transport, "_run_process", lambda argv, **kwargs: SimpleNamespace(
            returncode=255, stdout="", stderr="Permission denied (keyboard-interactive)."))
        with pytest.raises(AuthenticationError):
            remote.verify()


def test_askpass_helper_script_forwards_prompt(config, monkeypatch):
    monkeypatch.setattr(transport.platform, "system", lambda: "Darwin")
    with SSHTransport(config, password="pw") as remote:
        helper = remote._askpass_path.read_text()
        assert helper.startswith("#!/bin/sh\nexec ") and helper.rstrip().endswith('"$@"')
        assert "askpass_main" in helper and "pw" not in helper
        assert remote._askpass_path.stat().st_mode & stat.S_IWRITE
        if os.name != "nt":
            assert remote._askpass_path.stat().st_mode & 0o777 == 0o700


# --- registry: detection, sign-in and code mode -----------------------------

class FakeTransport:
    """Records how it was opened and answers verify() from a scripted plan."""
    opened = []
    plan = {}

    def __init__(self, config, password=None, *, code="", master=False):
        self.config, self.password, self.code, self.master = config, password, code, master
        FakeTransport.opened.append(self)
        if master and not config.ssh_control_path:
            raise transport.TransportError("A persistent sign-in needs the connection's SSH control socket path.")

    def verify(self, timeout=45):
        outcome = FakeTransport.plan.get((bool(self.password), self.code))
        if isinstance(outcome, Exception):
            raise outcome
        return Result(0)

    def run(self, command, timeout=60):
        return Result(0, "\n".join(["__DFT_USER__", "researcher", "__DFT_HOME__", "/home/researcher", "__DFT_TOOLS__",
                                    "/bin/python3", "/bin/sbatch", "/bin/squeue", "/bin/sacct", "/bin/scancel", "/opt/qe/pw.x",
                                    "__DFT_QUEUES__", "cpu*|up|2-00:00:00|40|(null)", "__DFT_WORKDIR__", "not configured"]))

    def close(self):
        pass


@pytest.fixture
def fake_transport(monkeypatch):
    # These cases test the Unix multiplexing flow; native Windows MFA is tested below.
    monkeypatch.setattr(connections.platform, "system", lambda: "Darwin")
    FakeTransport.opened = []
    FakeTransport.plan = {}
    monkeypatch.setattr(transport, "SSHTransport", FakeTransport)
    return FakeTransport


def test_detect_then_password_then_code_switches_mode_and_reuses_socket(registry, keychain, fake_transport):
    record = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "agent"})
    cid = record["id"]
    fake_transport.plan[(False, "")] = AuthenticationError("SSH authentication failed.", ("password", "keyboard-interactive"))
    assert registry.detect_auth(cid)["status"] == "password"
    assert fake_transport.opened[-1].password == "" and not fake_transport.opened[-1].master
    registry.update(cid, settings={"auth_mode": "password"})
    registry.set_secret(cid, "cluster-password")
    fake_transport.plan[(True, "")] = VerificationRequired("code needed", "Verification code: ")
    with pytest.raises(VerificationRequired) as caught:
        registry.connect(cid)
    assert caught.value.prompt == "Verification code: "
    record = registry.get(cid)
    assert record["settings"]["auth_mode"] == "code" and record["settings"]["ssh_control_path"]
    assert record["has_secret"] and keychain.values  # the same password keeps serving the connection
    fake_transport.plan[(True, "000000")] = AuthenticationError("SSH authentication failed.", ("keyboard-interactive",))
    with pytest.raises(AuthenticationError):
        registry.connect(cid, code="000000")
    with pytest.raises(ValueError, match="login code"):
        registry.connect(cid, code="rm -rf /")
    assert registry.connect(cid, code="123456")["status"] == "connected"
    opened = fake_transport.opened[-1]
    assert opened.master and opened.code == "123456" and opened.password == "cluster-password"
    assert opened.config.ssh_control_path == record["settings"]["ssh_control_path"]
    # The probe reuses the saved password and records what it learned without a revision conflict.
    probed = registry.probe(cid)
    assert probed["connection_status"] == "connected"
    assert probed["settings"]["remote_base"] == "/home/researcher/dft-agent"
    assert probed["settings"]["scheduler"] == "slurm" and probed["probe"]["schedulers"] == ["slurm"]
    assert "cluster-password" not in json.dumps({k: v for k, v in probed.items()})


def test_probe_reports_what_the_cluster_needs(registry, keychain, fake_transport):
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "password"})["id"]
    registry.set_secret(cid, "cluster-password")
    for outcome, needs, status in [
        (VerificationRequired("code", "Verification code: "), "code", "untested"),
        (AuthenticationError("SSH authentication failed.", ("password",)), "password", "untested"),
        (HostKeyError("changed"), "host_key", "error"),
        (ConnectionFailed("unreachable"), "network", "error"),
    ]:
        class Remote(FakeTransport):
            def run(self, command, timeout=60):
                raise outcome
        fake_transport.plan.clear()
        import vasp_slurm_agent.transport as module
        module.SSHTransport = Remote
        result = registry.probe(cid)
        assert result["probe"]["needs"] == needs and result["connection_status"] == status
        if needs == "code":
            assert result["settings"]["auth_mode"] == "code" and result["probe"]["prompt"] == "Verification code: "
            registry._patch_settings(cid, auth_mode="password")


def test_revocation_closes_login_socket_and_settings_patch_is_bounded(registry, keychain, tmp_path, monkeypatch):
    monkeypatch.setattr(connections.platform, "system", lambda: "Darwin")
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "code",
                                              "ssh_control_path": str(tmp_path / "login.sock")})["id"]
    (tmp_path / "login.sock").write_text("")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        (tmp_path / "login.sock").unlink()
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(connections.subprocess, "run", run)
    assert registry.disconnect(cid) is True
    assert calls and calls[0][-1] == "hpc.invalid" and "-O" in calls[0] and "exit" in calls[0]
    assert registry.disconnect(cid) is False
    registry.set_secret(cid, "cluster-password")
    with pytest.raises(ValueError, match="different sign-in"):
        registry.record_settings(cid, auth_mode="agent")
    with pytest.raises(ValueError, match="test results"):
        registry._patch_settings(cid, host="other.invalid")
    with pytest.raises(ValueError, match="different sign-in"):
        registry._patch_settings(cid, auth_mode="interactive")
    before = registry.get(cid)
    patched = registry.record_settings(cid, remote_base="/scratch/researcher")
    assert patched["settings"]["remote_base"] == "/scratch/researcher"
    assert patched["connection_status"] == before["connection_status"] and patched["has_secret"]
    registry.revoke(cid)
    assert registry.get(cid)["revoked"] and not keychain.values


def test_discovery_creates_default_environments_from_cluster_answers(registry, keychain, fake_transport, monkeypatch):
    from vasp_slurm_agent import discovery

    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "agent",
                                              "remote_base": "/home/researcher/dft-agent"})["id"]
    registry.probe(cid)
    facts = {"partitions": [{"name": "cpu", "default": True, "timelimit": "2-00:00:00", "gpus_per_node": 0}],
             "accounts": ["naiss2026-1-1"], "modules": ["VASP/6.4.3"], "executables": {}, "potcar_roots": [],
             "scratch_dirs": ["/scratch/researcher"], "slurm": ["sbatch", "squeue", "sacct", "scancel"]}
    monkeypatch.setattr(discovery, "probe_cluster", lambda config, transport=None, timeout=90: {"facts": facts, **discovery.suggest(facts)})
    result = registry.discover_environments(cid)
    assert result["vasp_found"] and result["qe_found"] and result["missing"] == {"vasp": ["potcar_root"]}
    assert [environment["engine"] for environment in result["created"]] == ["qe"]
    qe = result["created"][0]["config"]
    assert qe["partition"] == "cpu" and qe["qe_pseudo_root"] == "/home/researcher/dft-agent/pseudos" and qe["qe_command"] == "srun pw.x"
    completed = registry.discover_environments(cid, potcar_root="/licensed/potpaw_PBE")
    vasp = next(environment for environment in completed["created"] if environment["engine"] == "vasp")
    assert vasp["config"]["potcar_root"] == "/licensed/potpaw_PBE" and vasp["config"]["setup_commands"] == ["module load VASP/6.4.3"]
    assert vasp["config"]["account"] == "naiss2026-1-1" and vasp["config"]["vasp_command"] == "srun vasp_std"
    assert {environment["engine"] for environment in registry.environments(cid)} == {"vasp", "qe"}
    assert registry.get(cid)["probe"]["discovery"]["vasp_found"] is True
    # Discovery never runs twice for the same engine.
    assert registry.discover_environments(cid)["created"] == []


# --- CLI discovery -----------------------------------------------------------

def fake_cli(folder, name, version, *, login="Logged in using ChatGPT", status=0):
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # Match npm's Windows layout; production discovery launches its entrypoint.
        package = "@openai/codex" if name == "codex" else "@anthropic-ai/claude-code"
        root = folder / "node_modules" / package
        root.mkdir(parents=True)
        (root / "package.json").write_text(json.dumps({"name": package, "bin": {name: "cli.js"}}))
        (root / "cli.js").write_text(
            "if(process.argv[2] === '--version') { console.log(" + json.dumps("codex-cli " + version) + "); } "
            "else { console.log(" + json.dumps(login) + "); process.exit(" + str(status) + "); }",
            encoding="utf-8")
        path = folder / (name + ".cmd")
        path.write_text('@echo off\r\nnode "%~dp0node_modules\\' + package.replace('/', '\\') + '\\cli.js" %*\r\n')
    else:
        path = folder / name
        path.write_text("#!/bin/sh\n"
                        f"case \"$1\" in --version) echo 'codex-cli {version}';; "
                        f"login) echo '{login}'; exit {status};; "
                        f"auth) echo '{login}'; exit {status};; esac\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_locate_prefers_newest_installed_cli_and_honours_override(tmp_path, monkeypatch):
    older = fake_cli(tmp_path / "path", "codex", "0.139.0") if (tmp_path / "path").mkdir() is None else None
    bundled = fake_cli(tmp_path / "bundle", "codex", "0.162.0-alpha.17") if (tmp_path / "bundle").mkdir() is None else None
    monkeypatch.setattr(cli_tools, "_known_locations", lambda provider: [bundled])
    monkeypatch.delenv("CODEX_CLI_PATH", raising=False)
    located = cli_tools.locate("codex", which=lambda name: str(older) if name == "codex" else None)
    assert located["path"] == str(bundled) and located["version"][:3] == (0, 162, 0) and located["source"] == "known"
    assert cli_tools.version_text(located["version"]) == "0.162.0"
    # A release outranks a pre-release of the same number; PATH wins ties.
    release = fake_cli(tmp_path / "release", "codex", "0.162.0") if (tmp_path / "release").mkdir() is None else None
    monkeypatch.setattr(cli_tools, "_known_locations", lambda provider: [bundled, release])
    assert cli_tools.locate("codex", which=lambda name: str(older))["path"] == str(release)
    same = fake_cli(tmp_path / "same", "codex", "0.162.0") if (tmp_path / "same").mkdir() is None else None
    monkeypatch.setattr(cli_tools, "_known_locations", lambda provider: [same])
    assert cli_tools.locate("codex", which=lambda name: str(release))["path"] == str(release)
    monkeypatch.setenv("CODEX_CLI_PATH", str(older))
    assert cli_tools.locate("codex", which=lambda name: name)["source"] == "override"
    assert cli_tools.locate("codex", which=lambda name: None)["path"] == str(older)
    monkeypatch.delenv("CODEX_CLI_PATH")
    monkeypatch.setattr(cli_tools, "_known_locations", lambda provider: [])
    assert cli_tools.locate("claude_cli", which=lambda name: None) is None


def test_login_status_and_detection_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_CLI_PATH", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
    signed = fake_cli(tmp_path, "codex", "0.162.0")
    unsigned = fake_cli(tmp_path, "claude", "2.1.0", login="not logged in", status=1)
    assert cli_tools.login_status("codex", str(signed)) == {"logged_in": True, "detail": "ChatGPT"}
    assert cli_tools.login_status("claude_cli", str(unsigned))["logged_in"] is False
    json_cli = fake_cli(tmp_path / "json-cli", "claude", "2.1.0", login='{"loggedIn": true, "authMethod": "claude.ai"}')
    assert cli_tools.login_status("claude_cli", str(json_cli)) == {"logged_in": True, "detail": "claude.ai"}
    monkeypatch.setattr(cli_tools, "_known_locations", lambda provider: [])
    found = cli_tools.detect(which=lambda name: {"codex": str(signed), "claude": str(unsigned)}.get(name))
    assert [(item["provider"], item["installed"], item["logged_in"]) for item in found] == [("codex", True, True), ("claude_cli", True, False)]
    assert cli_tools.detect(which=lambda name: None)[0]["installed"] is False
    assert cli_tools.login_status("codex", str(tmp_path / "missing"))["logged_in"] is None


def test_model_probe_records_the_tested_cli_and_requests_use_it(registry, monkeypatch, tmp_path):
    cid = registry.create("model", "Codex", {"provider": "codex"})["id"]
    chosen = fake_cli(tmp_path, "codex", "0.162.0")
    monkeypatch.setattr(cli_tools, "locate", lambda provider, **kwargs: {"path": str(chosen), "version": (0, 162, 0, 1), "source": "known"})
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: '{"status":"connected"}')
    result = registry.probe(cid)
    assert result["connection_status"] == "connected" and result["settings"]["cli_path"] == str(chosen)
    assert result["probe"]["cli_version"] == "0.162.0" and result["probe"]["cli_account"] == "ChatGPT"
    settings = registry.model_settings(cid)
    assert settings.cli_path == str(chosen)
    assert agent._cli_executable(settings, "codex", "CODEX_CLI_PATH") == str(chosen)
    missing = agent.ModelSettings("codex", cli_path=str(tmp_path / "gone"))
    monkeypatch.setattr(agent.shutil, "which", lambda name: "/fallback/codex")
    assert agent._cli_executable(missing, "codex", "CODEX_CLI_PATH") == "/fallback/codex"
    with pytest.raises(ValueError, match="absolute"):
        registry.create("model", "Bad", {"provider": "codex", "cli_path": "relative/codex"})
    with pytest.raises(ValueError, match="CLI connection"):
        registry.create("model", "Bad", {"provider": "responses", "model": "m", "cli_path": "/usr/bin/codex"})


def test_model_probe_explains_an_outdated_cli(registry, monkeypatch, tmp_path):
    cid = registry.create("model", "Codex", {"provider": "codex"})["id"]
    chosen = fake_cli(tmp_path, "codex", "0.139.0")
    monkeypatch.setattr(cli_tools, "locate", lambda provider, **kwargs: {"path": str(chosen), "version": (0, 139, 0, 1), "source": "path"})

    def too_old(*args, **kwargs):
        raise agent.AgentError("Codex CLI is too old for this model. Update it or choose another model.")
    monkeypatch.setattr(agent, "_request_structured", too_old)
    result = registry.probe(cid)
    assert result["connection_status"] == "error"
    assert "too old" in result["probe"]["detail"] and "0.139.0" in result["probe"]["detail"] and str(chosen) in result["probe"]["detail"]


# --- address parsing ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("ssh researcher@hpc.invalid", {"host": "hpc.invalid", "user": "researcher", "port": 22}),
    ("researcher@hpc.invalid:2222", {"host": "hpc.invalid", "user": "researcher", "port": 2222}),
    ("ssh -p 2200 researcher@login.hpc.invalid", {"host": "login.hpc.invalid", "user": "researcher", "port": 2200}),
    ("ssh -l researcher hpc.invalid -p 2200", {"host": "hpc.invalid", "user": "researcher", "port": 2200}),
    ("hpc.invalid", {"host": "hpc.invalid", "user": None, "port": 22}),
    ("name@group@gateway.invalid", {"host": "gateway.invalid", "user": "name@group", "port": 22}),
])
def test_parse_address_variants(text, expected):
    assert parse_address(text) == expected


@pytest.mark.parametrize("text", ["", "ssh", "researcher@", "ssh -p abc host", "host:99999", "-oProxyCommand=evil host", "password: hidden"])
def test_parse_address_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        parse_address(text)


# --- guided setup screens ----------------------------------------------------

def chat_app(project_id=None):
    return AppTest.from_string(f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import chat_panel
chat_panel(ConnectionRegistry(), {project_id!r})
""", default_timeout=20).run()


def address(view, text):
    """Type into the cluster form's address (or username) field and press Next."""
    field = next(item for item in view.text_input if item.label in {"SSH address", "Username on this cluster"})
    field.set_value(text)
    widget(view.button, "Next").click().run()


def address_form(view):
    return any(item.label in {"SSH address", "Username on this cluster"} for item in view.text_input)


def detected(monkeypatch, codex_signed_in=True):
    items = [{"provider": "codex", "label": "Codex CLI", "installed": True, "path": "/mock/codex", "version": (0, 162, 0, 1),
              "logged_in": codex_signed_in, "detail": "ChatGPT" if codex_signed_in else "Not signed in."},
             {"provider": "claude_cli", "label": "Claude Code CLI", "installed": False, "path": None, "version": None,
              "logged_in": None, "detail": "Not installed."}]
    monkeypatch.setattr(cli_tools, "detect", lambda *args, **kwargs: items)
    return items


def test_signed_in_cli_connects_with_one_click(monkeypatch):
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Study")
    detected(monkeypatch)
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "Mock model probe."})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    view = chat_app(project["id"])
    assert not view.text_input and not view.chat_input
    # Each CLI shows its next step: the missing one offers to install itself.
    assert widget(view.button, "Install and sign in") and not widget(view.button, "Install and sign in").disabled
    assert "npm" not in " ".join(item.value for item in view.code)
    widget(view.button, "Connect Codex CLI").click().run()
    assert not view.exception and not view.error
    record = registry.project_connections(project["id"])[0]
    assert record["settings"]["provider"] == "codex" and record["connection_status"] == "connected"
    probe.assert_called_once()
    assert any("Model connected" in item.value for item in view.success)
    assert widget(view.button, "Back to chat")


def test_signed_out_cli_signs_in_through_a_terminal_and_the_card_follows(monkeypatch):
    from vasp_slurm_agent import cli_setup
    registry = ConnectionRegistry()
    items = detected(monkeypatch, codex_signed_in=False)
    opened = []
    monkeypatch.setattr(cli_setup, "open_terminal", lambda provider, item, messages, **kwargs: opened.append((provider, item, messages, kwargs)))
    view = chat_app()
    assert any(item.value == "/mock/codex login" for item in view.code)   # the same step by hand
    widget(view.button, "Sign in").click().run()
    assert not view.exception and not registry.list(kind="model")
    provider, item, messages, options = opened[0]
    assert (provider, item["path"], options) == ("codex", "/mock/codex", {"update": False})
    assert set(messages) == set(cli_setup.MESSAGES) and "ChatGPT" in messages["signing_in"]
    assert any("this page updates by itself" in item.value for item in view.caption)
    # Signing in finishes in the terminal; the next check finds it and offers to connect.
    items[0].update(logged_in=True, detail="ChatGPT")
    view.run()
    assert not view.exception and widget(view.button, "Connect Codex CLI")
    assert not any("this page updates by itself" in item.value for item in view.caption)


def test_saved_cli_connection_that_signed_out_offers_the_same_steps(monkeypatch):
    from vasp_slurm_agent import cli_setup
    registry = ConnectionRegistry()
    record = registry.create("model", "Codex CLI", {"provider": "codex"})
    detected(monkeypatch, codex_signed_in=False)
    probe = Mock(return_value={"auth_status": "required", "connection_status": "untested", "detail": "Sign in with Codex CLI in your terminal, then test again."})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    opened = []
    monkeypatch.setattr(cli_setup, "open_terminal", lambda provider, item, messages, **kwargs: opened.append(provider))
    view = chat_app()
    widget(view.button, "Codex CLI").click().run()
    assert any(item.value == "/mock/codex login" for item in view.code)
    widget(view.button, "Sign in").click().run()
    assert opened == ["codex"] and any("then click Test connection" in item.value for item in view.caption)
    widget(view.button, "Test connection").click().run()
    probe.assert_called_once()
    assert registry.get(record["id"])["connection_status"] == "untested"


def test_cli_monitor_can_be_stopped_and_sign_in_retried(monkeypatch):
    from vasp_slurm_agent import cli_setup

    detected(monkeypatch, codex_signed_in=False)
    opened = Mock()
    monkeypatch.setattr(cli_setup, "open_terminal", opened)
    view = chat_app()
    widget(view.button, "Sign in").click().run()
    assert any("Close the terminal window" in item.value for item in view.caption)
    widget(view.button, "Stop checking").click().run()
    assert not view.exception
    assert "setup_chat_global_model_cli_watch" not in view.session_state
    assert not any(item.label == "Stop checking" for item in view.button)
    widget(view.button, "Sign in").click().run()
    assert opened.call_count == 2
    assert "setup_chat_global_model_cli_watch" in view.session_state


def test_cli_without_a_working_runtime_offers_install_instead_of_sign_in(monkeypatch):
    from vasp_slurm_agent import cli_setup

    items = detected(monkeypatch, codex_signed_in=False)
    items[0]["version"] = None
    opened = Mock()
    monkeypatch.setattr(cli_setup, "open_terminal", opened)
    view = chat_app()
    button = view.button(key="setup_chat_global_model_terminal_codex")
    assert button.label == "Install and sign in"
    button.click().run()
    assert not view.exception
    opened.assert_called_once()


def test_saved_cli_with_a_broken_runtime_can_be_reinstalled(monkeypatch):
    from vasp_slurm_agent import cli_setup

    registry = ConnectionRegistry()
    registry.create("model", "Codex CLI", {"provider": "codex"})
    items = detected(monkeypatch)
    items[0].update(version=None, logged_in=None, detail="Could not run the CLI.")
    opened = Mock()
    monkeypatch.setattr(cli_setup, "open_terminal", opened)
    view = chat_app()
    widget(view.button, "Codex CLI").click().run()
    widget(view.button, "Install and sign in").click().run()
    assert not view.exception
    opened.assert_called_once()


def test_cli_fragment_stops_after_its_deadline_without_a_full_page_rerun(monkeypatch):
    from vasp_slurm_agent import setup_app

    class Rerun(Exception):
        pass

    now = [100]
    state = {"review_cli_watch": {"provider": "codex", "until": 110}}

    def fragment(**kwargs):
        assert kwargs["run_every"] == 4
        def decorate(callback):
            def expired_call():
                now[0] = 111
                callback()
            return expired_call
        return decorate

    monkeypatch.setattr(setup_app, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(setup_app, "st", SimpleNamespace(
        session_state=state, fragment=fragment, rerun=Mock(side_effect=Rerun)))
    detection = Mock()
    monkeypatch.setattr(setup_app, "_cli_detection", detection)
    with pytest.raises(Rerun):
        setup_app._cli_cards(None, None, "review_", {})
    assert not state
    detection.assert_not_called()


def test_model_setup_defaults_to_english_and_translates_sign_in_actions(monkeypatch):
    from vasp_slurm_agent.i18n import current_language, set_language

    detected(monkeypatch, codex_signed_in=False)
    assert current_language() == "en"
    view = chat_app()
    assert widget(view.button, "Sign in") and widget(view.button, "Install and sign in")
    set_language("zh")
    view.run()
    assert not view.exception
    assert widget(view.button, "登录") and widget(view.button, "安装并登录")


def models_listed(monkeypatch, answers):
    """The free model-list check, answered per address: (status, models) or a callable."""
    from vasp_slurm_agent import model_connect
    asked = []

    def list_models(provider, key, base_url, **kwargs):
        asked.append(base_url)
        answer = answers.get(base_url, answers.get("*", ("ok", [])))
        return answer(key) if callable(answer) else answer
    monkeypatch.setattr(model_connect, "list_models", list_models)
    return asked


def test_api_key_provider_connects_in_one_step_and_keeps_the_key_out_of_state(monkeypatch, keychain):
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Study")
    detected(monkeypatch, codex_signed_in=False)
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "Mock model probe."})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    asked = models_listed(monkeypatch, {"https://api.openai.com/v1": ("ok", ["gpt-6.1-sol", "gpt-5", "text-embedding-3-large"])})
    view = chat_app(project["id"])
    widget(view.pills, "Provider").set_value("responses").run()
    assert widget(view.text_input, "Model name").value == "gpt-6.1-sol"
    secure = widget(view.text_input, "API key")
    assert secure.proto.type == secure.proto.PASSWORD
    assert any(button.label == "Open the OpenAI key page" for button in view.get("link_button"))
    widget(view.button, "Connect").click().run()
    assert not view.exception and any("Paste the API key first" in item.value for item in view.error)
    assert not registry.project_connections(project["id"]) and not asked  # nothing checked or bound without a key
    view = chat_app(project["id"])
    widget(view.pills, "Provider").set_value("responses").run()
    # Copying often adds spaces or a line break; they are removed.
    widget(view.text_input, "API key").set_value("  sk-test-1234567890abcdef\n")
    widget(view.button, "Connect").click().run()
    assert not view.exception and not view.error
    record = next(item for item in registry.project_connections(project["id"]) if item["has_secret"])
    assert record["settings"] == {"provider": "responses", "model": "gpt-6.1-sol", "base_url": None, "cli_path": None}
    assert keychain.values and "sk-test-1234567890abcdef" in keychain.values.values()
    assert "sk-test" not in json.dumps({key: str(value) for key, value in view.session_state.items()})
    assert record["connection_status"] == "connected"
    probe.assert_called_once()
    assert any("Model: gpt-6.1-sol" in item.value for item in view.caption)


def test_keychain_failure_keeps_one_recoverable_model_without_binding_it(monkeypatch, keychain):
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Study")["id"]
    detected(monkeypatch)
    models_listed(monkeypatch, {"*": ("ok", ["gpt-6.1-sol"])})
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    store = keychain.set_password
    monkeypatch.setattr(keychain, "set_password", Mock(side_effect=OSError("Keychain locked")))
    view = chat_app(project)
    widget(view.pills, "Provider").set_value("responses").run()
    widget(view.text_input, "API key").set_value("sk-recoverable-test")
    widget(view.button, "Connect").click().run()
    assert not view.exception and any("Check the system keychain" in item.value for item in view.error)
    assert not registry.project_connections(project) and len(registry.list(kind="model")) == 1
    probe.assert_not_called()
    assert "sk-recoverable-test" not in str(dict(view.session_state.items()))
    assert widget(view.text_input, "API key").proto.type == widget(view.text_input, "API key").proto.PASSWORD

    monkeypatch.setattr(keychain, "set_password", store)
    widget(view.text_input, "API key").set_value("  sk-recovered-test\n")
    widget(view.button, "Save credential").click().run()
    widget(view.button, "Test connection").click().run()
    widget(view.button, "Use in this project").click().run()
    assert not view.exception and any("Model connected" in item.value for item in view.success)
    assert len(registry.list(kind="model")) == len(registry.project_connections(project)) == 1
    assert "sk-recovered-test" in keychain.values.values()
    assert "sk-recovered-test" not in str(dict(view.session_state.items()))


def test_saved_model_edit_failure_is_visible_and_retry_survives_restart(monkeypatch, keychain):
    registry = ConnectionRegistry()
    record = registry.create("model", "My API", {"provider": "responses", "model": "first-model"})
    registry.set_secret(record["id"], "sk-saved-test")
    detected(monkeypatch)
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    view = chat_app()
    widget(view.button, "My API").click().run()
    widget(view.text_input, "Model name").set_value("m" * 201).run()
    widget(view.button, "Save and test").click().run()
    assert not view.exception and view.error
    assert registry.get(record["id"])["settings"]["model"] == "first-model"
    probe.assert_not_called()
    widget(view.text_input, "Model name").set_value("second-model").run()
    widget(view.button, "Save and test").click().run()
    assert not view.exception and not view.error
    assert registry.get(record["id"])["settings"]["model"] == "second-model"
    probe.assert_called_once()
    reopened = chat_app()
    widget(reopened.button, "My API").click().run()
    assert widget(reopened.text_input, "Model name").value == "second-model"
    assert widget(reopened.text_input, "API key").value == ""
    assert "sk-saved-test" not in str(dict(reopened.session_state.items()))


def test_model_setup_keeps_project_selection_and_credentials_separate(monkeypatch, keychain):
    registry = ConnectionRegistry()
    first = registry.workspace.create_project("First")["id"]
    second = registry.workspace.create_project("Second")["id"]
    detected(monkeypatch)
    models_listed(monkeypatch, {"*": ("ok", ["gpt-6.1-sol", "deepseek-flash"])})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", Mock(return_value={
        "auth_status": "available", "connection_status": "connected", "detail": "ok"}))
    view = AppTest.from_string(f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import chat_panel
project = st.selectbox('Project', {[first, second]!r})
chat_panel(ConnectionRegistry(), project)
""").run()
    widget(view.pills, "Provider").set_value("responses").run()
    widget(view.text_input, "API key").set_value("sk-first-project")
    widget(view.button, "Connect").click().run()
    first_model = registry.project_connections(first)[0]
    widget(view.selectbox, "Project").select(second).run()
    assert not registry.project_connections(second)
    assert not any("Model connected" in item.value for item in view.success)
    widget(view.pills, "Provider").set_value("deepseek").run()
    assert widget(view.text_input, "API key").value == ""
    widget(view.text_input, "API key").set_value("sk-second-project")
    widget(view.button, "Connect").click().run()
    second_model = registry.project_connections(second)[0]
    assert first_model["id"] != second_model["id"]
    widget(view.selectbox, "Project").select(first).run()
    assert not view.exception and any("Model: gpt-6.1-sol" in item.value for item in view.caption)
    assert not any("deepseek-flash" in item.value for item in view.caption)
    assert "sk-first-project" not in str(dict(view.session_state.items()))
    assert "sk-second-project" not in str(dict(view.session_state.items()))
    assert registry.project_connections(first)[0]["id"] == first_model["id"]


def test_a_key_typed_for_one_provider_is_not_offered_to_another(monkeypatch, keychain):
    detected(monkeypatch)
    asked = models_listed(monkeypatch, {"*": ("rejected", [])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("deepseek").run()
    widget(view.text_input, "API key").set_value("sk-deepseek-only")
    widget(view.pills, "Provider").set_value("qwen").run()
    assert widget(view.text_input, "API key").value == ""
    widget(view.button, "Connect").click().run()
    assert not asked and any("Paste the API key first" in item.value for item in view.error)


def test_qwen_key_finds_its_region_by_itself(monkeypatch, keychain):
    from vasp_slurm_agent.model_connect import REGIONS
    registry = ConnectionRegistry()
    detected(monkeypatch)
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"}))
    beijing, singapore = REGIONS["qwen"][0][1], REGIONS["qwen"][1][1]
    asked = models_listed(monkeypatch, {"*": ("rejected", []), singapore: ("ok", ["qwen-max", "qwen-plus", "qwen-vl-plus"])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("qwen").run()
    assert not any(item.label == "API address" and item.value for item in view.text_input)
    widget(view.text_input, "API key").set_value("sk-qwen-test-key")
    widget(view.button, "Connect").click().run()
    # An English interface tries the international regions first and stops at the one that accepts the key.
    assert not view.exception and not view.error and asked == [singapore] and beijing not in asked
    record = registry.list(kind="model")[0]
    assert record["settings"]["base_url"] == singapore and record["settings"]["model"] == "qwen-plus"
    assert any("Region: Singapore" in item.value for item in view.caption)


def test_a_rejected_key_is_explained_and_nothing_is_saved(monkeypatch, keychain):
    registry = ConnectionRegistry()
    detected(monkeypatch)
    models_listed(monkeypatch, {"*": ("rejected", [])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("responses").run()
    widget(view.text_input, "API key").set_value("sk-ant-api03-wrong-provider")
    widget(view.button, "Connect").click().run()
    assert not view.exception and not registry.list(kind="model") and not keychain.values
    message = " ".join(item.value for item in view.error)
    assert "did not accept this key" in message and "It looks like a Claude key: choose Claude above." in message
    assert any(button.label == "Create a new key" for button in view.get("link_button"))


def test_no_credit_after_a_valid_key_points_to_top_up(monkeypatch, keychain):
    from vasp_slurm_agent import agent
    registry = ConnectionRegistry()
    detected(monkeypatch)
    models_listed(monkeypatch, {"*": ("ok", ["deepseek-flash", "deepseek-v4-pro"])})

    def broke(*args, **kwargs):
        raise agent.AgentError("DeepSeek API credit or spending limit was reached.", code="credit")
    monkeypatch.setattr(agent, "_request_structured", broke)
    view = chat_app()
    widget(view.pills, "Provider").set_value("deepseek").run()
    widget(view.text_input, "API key").set_value("sk-deepseek-test")
    widget(view.button, "Connect").click().run()
    assert not view.exception
    record = registry.list(kind="model")[0]
    assert record["connection_status"] == "error" and record["probe"]["error_code"] == "credit"
    assert any("no API credit" in item.value for item in view.error)
    assert [button.proto.url for button in view.get("link_button") if button.label == "Top up"] == ["https://platform.deepseek.com/top_up"]


def test_a_model_that_cannot_answer_in_the_format_gives_way_to_the_next(monkeypatch, keychain):
    from vasp_slurm_agent import agent
    registry = ConnectionRegistry()
    detected(monkeypatch)
    models_listed(monkeypatch, {"*": ("ok", ["glm-5.3", "glm-5.2", "glm-4.6"])})
    tried = []

    def answer(payload, schema, settings, **kwargs):
        tried.append(settings.model if hasattr(settings, "model") else None)
        if len(tried) == 1:
            raise agent.AgentError("GLM rejected the request format.", code="format")
        return agent.ModelResponse('{"status":"connected"}', {})
    monkeypatch.setattr(agent, "_request_structured", answer)
    view = chat_app()
    widget(view.pills, "Provider").set_value("glm").run()
    widget(view.text_input, "API key").set_value("0123456789abcdef0123456789abcdef.ABCDEFGHIJKLMNOP")
    widget(view.button, "Connect").click().run()
    assert not view.exception and not view.error
    record = registry.list(kind="model")[0]
    assert record["connection_status"] == "connected" and record["settings"]["model"] == "glm-5.2" and len(tried) == 2


def test_local_model_servers_are_found_and_connected(monkeypatch):
    from vasp_slurm_agent import model_connect
    registry = ConnectionRegistry()
    detected(monkeypatch)
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"}))
    servers = []
    monkeypatch.setattr(model_connect, "local_servers", lambda **kwargs: list(servers))
    models_listed(monkeypatch, {"http://127.0.0.1:11434/v1": ("ok", ["qwen3:8b"])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("local").run()
    assert widget(view.button, "Connect").disabled and any("No model server is running" in item.value for item in view.info)
    servers.append({"name": "Ollama", "base_url": "http://127.0.0.1:11434/v1", "models": ["qwen3:8b"]})
    widget(view.button, "Look for local servers again").click().run()
    widget(view.button, "Connect").click().run()
    assert not view.exception
    record = registry.list(kind="model")[0]
    assert record["settings"] == {"provider": "local", "model": "qwen3:8b", "base_url": "http://127.0.0.1:11434/v1", "cli_path": None}


def test_local_model_on_a_custom_port_can_be_entered_manually(monkeypatch):
    from vasp_slurm_agent import model_connect

    registry = ConnectionRegistry()
    detected(monkeypatch)
    monkeypatch.setattr(model_connect, "local_servers", lambda **kwargs: [])
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", Mock(return_value={
        "auth_status": "available", "connection_status": "connected", "detail": "ok"}))
    asked = models_listed(monkeypatch, {"http://127.0.0.1:19000/v1": ("ok", ["custom-model"])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("local").run()
    widget(view.button, "Connect local server").click().run()
    assert not asked and not registry.list(kind="model")
    assert any("Enter the local server's API address" in item.value for item in view.error)
    widget(view.text_input, "API address").set_value("http://127.0.0.1:19000/v1")
    widget(view.text_input, "Model name").set_value("custom-model")
    widget(view.button, "Connect local server").click().run()
    assert not view.exception and not view.error
    record = registry.list(kind="model")[0]
    assert record["connection_status"] == "connected" and not record["has_secret"]
    assert record["settings"]["base_url"] == "http://127.0.0.1:19000/v1"
    assert record["settings"]["model"] == "custom-model"


def test_compatible_api_tries_the_v1_address_too(monkeypatch, keychain):
    registry = ConnectionRegistry()
    detected(monkeypatch)
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"}))
    asked = models_listed(monkeypatch, {"https://llm.example.org": ("unsupported", []), "https://llm.example.org/v1": ("ok", ["kimi-k3"])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("chat_completions").run()
    widget(view.text_input, "API address").set_value("https://llm.example.org")
    widget(view.text_input, "API key").set_value("sk-compatible")
    widget(view.button, "Connect").click().run()
    assert not view.exception and not view.error and asked == ["https://llm.example.org", "https://llm.example.org/v1"]
    record = registry.list(kind="model")[0]
    assert record["settings"]["model"] == "kimi-k3" and record["settings"]["base_url"] == "https://llm.example.org/v1"


@pytest.mark.parametrize("provider,address", [
    ("local", "https://remote.example.org/v1"),
    ("chat_completions", "https://user:private-value@api.example.org/v1"),
])
def test_invalid_api_address_shows_guidance_before_any_request(monkeypatch, provider, address):
    from vasp_slurm_agent import model_connect

    registry = ConnectionRegistry()
    detected(monkeypatch)
    monkeypatch.setattr(model_connect, "local_servers", lambda **kwargs: [])
    get = Mock(side_effect=AssertionError("Invalid addresses must not send credentials."))
    monkeypatch.setattr(model_connect, "_get", get)
    view = chat_app()
    widget(view.pills, "Provider").set_value(provider).run()
    widget(view.text_input, "API address").set_value(address)
    widget(view.text_input, "Model name").set_value("test-model")
    if provider != "local":
        widget(view.text_input, "API key").set_value("private-api-key")
    widget(view.button, "Connect local server" if provider == "local" else "Connect").click().run()
    assert not view.exception and not registry.list(kind="model")
    assert any("Check the API address." in item.value for item in view.error)
    get.assert_not_called()
    assert "private-api-key" not in str(dict(view.session_state.items()))


def test_invalid_model_name_keeps_the_form_available_for_retry(monkeypatch, keychain):
    registry = ConnectionRegistry()
    detected(monkeypatch)
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "ok"})
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", probe)
    models_listed(monkeypatch, {"*": ("ok", [])})
    view = chat_app()
    widget(view.pills, "Provider").set_value("chat_completions").run()
    widget(view.text_input, "API address").set_value("https://api.example.org/v1")
    widget(view.text_input, "Model name").set_value("x" * 201)
    widget(view.text_input, "API key").set_value("private-api-key")
    widget(view.button, "Connect").click().run()
    assert not view.exception and view.error and not registry.list(kind="model")
    probe.assert_not_called()
    assert "private-api-key" not in str(dict(view.session_state.items()))
    widget(view.text_input, "Model name").set_value("valid-model")
    widget(view.text_input, "API key").set_value("private-api-key")
    widget(view.button, "Connect").click().run()
    assert not view.exception and not view.error
    assert registry.list(kind="model")[0]["settings"]["model"] == "valid-model"
    probe.assert_called_once()


def test_cluster_setup_asks_only_for_what_the_cluster_needs(monkeypatch, keychain):
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Study")
    monkeypatch.setattr(ConnectionRegistry, "detect_auth", lambda self, identifier: {"status": "password", "methods": ("keyboard-interactive",), "detail": "password"})
    attempts = []

    def connect(self, identifier, code=""):
        attempts.append(code)
        if not code:
            raise VerificationRequired("code", "Verification code: ")
        if code != "246810":
            raise AuthenticationError("SSH authentication failed.", ("keyboard-interactive",))
        return {"status": "connected", "auth_mode": "code", "detail": "Signed in."}
    monkeypatch.setattr(ConnectionRegistry, "connect", connect)
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "SSH connected.",
                               "schedulers": ["slurm"], "home_directory": "/home/researcher",
                               "suggested_remote_base": "/home/researcher/dft-agent", "remote_base": "/home/researcher/dft-agent"})
    monkeypatch.setattr(ConnectionRegistry, "_probe_hpc", probe)
    discover = Mock(return_value={"created": [], "missing": {}, "notes": [], "vasp_found": False, "qe_found": False})
    monkeypatch.setattr(ConnectionRegistry, "discover_environments", discover)
    view = chat_app(project["id"])
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "ssh researcher@cluster.invalid:2222")
    assert not view.exception and not address_form(view)
    record = registry.project_connections(project["id"])[0]
    assert record["settings"]["host"] == "cluster.invalid" and record["settings"]["port"] == 2222
    password_field = widget(view.text_input, "SSH password")
    assert password_field.proto.type == password_field.proto.PASSWORD
    assert not any(button.label == "Sign in" and button.disabled for button in view.button)
    password_field.set_value("cluster-password")
    widget(view.button, "Sign in").click().run()
    assert not view.exception
    assert registry.get(record["id"])["has_secret"] and attempts == [""]
    assert "cluster-password" not in json.dumps({key: str(value) for key, value in view.session_state.items()})
    assert any(item.value.strip() == "Verification code:" for item in view.code)
    assert widget(view.text_input, "Login code").proto.type == widget(view.text_input, "Login code").proto.PASSWORD
    widget(view.text_input, "Login code").set_value("000000")
    widget(view.button, "Submit code").click().run()
    assert any("rejected the login code" in item.value for item in view.error) and attempts == ["", "000000"]
    widget(view.text_input, "Login code").set_value("246810")
    widget(view.button, "Submit code").click().run()
    assert not view.exception and not view.error
    assert attempts == ["", "000000", "246810"]
    probe.assert_called_once()
    discover.assert_called_once()
    assert registry.get(record["id"])["connection_status"] == "connected"
    assert any("Connected to the cluster" in item.value for item in view.success)
    assert not any(field.label in {"SSH password", "Login code"} for field in view.text_input)


def test_cluster_setup_uses_keys_when_they_work(monkeypatch, keychain):
    registry = ConnectionRegistry()
    monkeypatch.setattr(ConnectionRegistry, "detect_auth", lambda self, identifier: {"status": "connected", "methods": (), "detail": "key"})
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "detail": "SSH connected.", "schedulers": ["slurm"]})
    monkeypatch.setattr(ConnectionRegistry, "_probe_hpc", probe)
    monkeypatch.setattr(ConnectionRegistry, "discover_environments", Mock(return_value={"created": [], "missing": {}, "notes": []}))
    view = chat_app()
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "hpc.invalid")
    assert any(item.label == "Username on this cluster" for item in view.text_input)
    assert widget(view.text_input, "Username on this cluster").value == ""
    address(view, "researcher")
    assert not view.exception and not view.error
    record = registry.list(kind="hpc")[0]
    assert record["settings"] == {**record["settings"], "host": "hpc.invalid", "user": "researcher", "auth_mode": "agent"}
    assert record["connection_status"] == "connected" and not record["has_secret"]
    assert not any(field.label == "SSH password" for field in view.text_input)
    probe.assert_called_once()


def test_cluster_setup_explains_unreachable_hosts_and_keeps_terminal_fallback(monkeypatch):
    registry = ConnectionRegistry()
    monkeypatch.setattr(ConnectionRegistry, "detect_auth", lambda self, identifier: {"status": "unreachable", "methods": (), "detail": "The cluster could not be reached."})
    view = chat_app()
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "researcher@nowhere.invalid")
    assert not view.exception and any("could not be reached" in item.value for item in view.error)
    assert widget(view.button, "Try again") and widget(view.button, "Open terminal sign-in")
    assert not any(field.label == "SSH password" for field in view.text_input)
    assert registry.list(kind="hpc")[0]["connection_status"] == "untested"


def test_setup_is_isolated_by_project_and_blocked_for_archived_projects(monkeypatch):
    registry = ConnectionRegistry()
    first = registry.workspace.create_project("First")["id"]
    second = registry.workspace.create_project("Second")["id"]
    monkeypatch.setattr(ConnectionRegistry, "detect_auth", lambda self, identifier: {"status": "password", "methods": ("password",), "detail": "password"})
    view = AppTest.from_string(f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import chat_panel
project = st.selectbox('Project', {[first, second]!r})
chat_panel(ConnectionRegistry(), project)
""").run()
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "researcher@first.invalid")
    assert any(field.label == "SSH password" for field in view.text_input)
    widget(view.selectbox, "Project").select(second).run()
    widget(view.pills, "Connect").set_value("hpc").run()
    assert address_form(view) and not any(field.label == "SSH password" for field in view.text_input)
    assert not registry.project_connections(second)
    widget(view.selectbox, "Project").select(first).run()
    # Streamlit forgets widgets that were not drawn; the project's own setup progress survives.
    widget(view.pills, "Connect").set_value("hpc").run()
    assert any(field.label == "SSH password" for field in view.text_input)
    assert len(registry.project_connections(first)) == 1
    registry.workspace.archive_project(first)
    view.run()
    assert not view.button and not view.chat_input and not view.text_input


def test_setup_rejects_secrets_typed_as_addresses_or_answers(monkeypatch):
    registry = ConnectionRegistry()
    view = chat_app()
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "password: hidden")
    assert not view.exception and view.error and not registry.list()
    assert "hidden" not in json.dumps({key: str(value) for key, value in view.session_state.items() if "state" in key})


def test_terminal_fallback_keeps_saved_password_connection(monkeypatch, keychain):
    from vasp_slurm_agent import ssh_auth
    registry = ConnectionRegistry()
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "password"})["id"]
    registry.set_secret(cid, "saved-password")
    launched = []
    monkeypatch.setattr(ssh_auth, "launch", lambda registry, identifier: launched.append(identifier) or {
        "status": "launched", "reusable": True, "message": "Complete sign-in in the terminal."})
    view = AppTest.from_string(f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import _terminal_sign_in
registry = ConnectionRegistry()
_terminal_sign_in(registry, registry.get({cid!r}), 'terminal_')
""").run()
    widget(view.button, "Open terminal sign-in").click().run()
    assert not view.exception and not view.error and launched == [cid]
    assert registry.get(cid)["settings"]["auth_mode"] == "password"
    assert registry.ssh_password(cid) == "saved-password"
    assert not any("saved-password" in str(item.value) for item in view.info)


def test_qe_build_controls_keep_modules_account_and_existing_build_visible(monkeypatch):
    registry = ConnectionRegistry()
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "remote_base": "/scratch/researcher", "scheduler": "slurm"})["id"]
    registry.create_environment(cid, "Older QE", "qe", {"partition": "cpu", "qe_pseudo_root": "/pseudo", "qe_bin_dir": "/old/bin"})
    install = {"status": "planned", "plan_id": "reviewed", "partition": "cpu", "account": "my-allocation",
        "plan": {"version": "7.4.1", "build_jobs": 4, "walltime": "02:00:00", "prefix": "/new"},
        "setup_commands": ["module purge", "module load gcc openmpi"], "toolchain": {"ready": True, "notes": []}}
    registry._save_probe(cid, registry.get(cid)["revision"], {"auth_status": "available", "connection_status": "connected", "schedulers": ["slurm"],
        "discovery": {"qe_found": True, "vasp_found": False}, "qe_install": install})
    planned = []
    monkeypatch.setattr(ConnectionRegistry, "qe_install_plan", lambda self, identifier, **kwargs: planned.append((identifier, kwargs)))
    view = AppTest.from_string(f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import _environments_panel
registry = ConnectionRegistry()
_environments_panel(registry, registry.get({cid!r}), 'build_', {{'discovered': True}})
""").run()
    assert not view.exception and widget(view.button, "Start the build").disabled
    assert widget(view.text_area, "Module commands (optional, one per line)").value == "module purge\nmodule load gcc openmpi"
    widget(view.text_input, "Account (optional)").set_value("my-other-allocation")
    widget(view.button, "Plan the build").click().run()
    assert not view.exception and planned[0][1]["account"] == "my-other-allocation"
    assert planned[0][1]["setup_commands"] == ["module purge", "module load gcc openmpi"]
    registry._store_probe(cid, "qe_install", {**install, "status": "failed", "error": "Compiler could not link MPI.", "job_id": "123"})
    view.run()
    assert any("Compiler could not link MPI" in item.value for item in view.error)
    assert widget(view.button, "Check the build") and widget(view.button, "Plan again")



def test_windows_mfa_uses_persistent_transport_without_control_socket(registry, keychain, monkeypatch):
    import sys
    monkeypatch.setattr(connections.platform, "system", lambda: "Windows")
    import vasp_slurm_agent
    broker = SimpleNamespace(disconnect=lambda *args: False)
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.ssh_broker", broker)
    monkeypatch.setattr(vasp_slurm_agent, "ssh_broker", broker, raising=False)
    class WindowsRemote(FakeTransport):
        def __init__(self, config, password=None, *, code="", master=False):
            self.config, self.password, self.code, self.master = config, password, code, master
            FakeTransport.opened.append(self)
    monkeypatch.setattr(transport, "SSHTransport", WindowsRemote)
    FakeTransport.opened, FakeTransport.plan = [], {}
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "password"})["id"]
    registry.set_secret(cid, "password")
    FakeTransport.plan[(True, "")] = VerificationRequired("code needed", "Verification code:")
    with pytest.raises(VerificationRequired):
        registry.connect(cid)
    assert registry.get(cid)["settings"]["ssh_control_path"] == ""
    assert registry.get(cid)["settings"]["auth_mode"] == "code"
    assert registry.connect(cid, code="123456")["status"] == "connected"
    signed_in = FakeTransport.opened[-1]
    assert signed_in.master and signed_in.code == "123456"
    assert signed_in.config.connection_id == cid and signed_in.config.connection_home == str(registry.home)
    assert signed_in.config.ssh_control_path == ""
    tested = registry.probe(cid)
    assert tested["connection_status"] == "connected"
    assert not FakeTransport.opened[-1].master and FakeTransport.opened[-1].code == ""
    assert "123456" not in json.dumps(registry.get(cid))
    monkeypatch.setattr(WindowsRemote, "run", lambda self, *args, **kwargs: (_ for _ in ()).throw(VerificationRequired("Session expired", "Verification code:")))
    expired = registry.probe(cid)
    assert expired["connection_status"] == "untested" and expired["probe"]["needs"] == "code"
    assert expired["settings"]["ssh_control_path"] == ""


def test_windows_registry_invalidates_broker_after_credential_or_ssh_changes(registry, monkeypatch):
    import sys
    monkeypatch.setattr(connections.platform, "system", lambda: "Windows")
    closed = []
    import vasp_slurm_agent
    broker = SimpleNamespace(disconnect=lambda home, identifier: closed.append(identifier) or True)
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.ssh_broker", broker)
    monkeypatch.setattr(vasp_slurm_agent, "ssh_broker", broker, raising=False)
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "password"})["id"]
    registry.set_secret(cid, "first-password")
    registry.set_secret(cid, "second-password")
    assert closed == [cid, cid]
    registry.update(cid, name="My cluster")
    registry.record_settings(cid, remote_base="/scratch/researcher")
    assert closed == [cid, cid]
    registry.update(cid, settings={"identity_file": str(registry.home / "id_ed25519")})
    assert closed == [cid, cid, cid]
    registry.revoke(cid)
    assert closed == [cid, cid, cid, cid] and registry.get(cid)["revoked"]


def test_windows_setup_prompts_again_after_session_expiry(monkeypatch, keychain):
    import sys
    monkeypatch.setattr(connections.platform, "system", lambda: "Windows")
    import vasp_slurm_agent
    broker = SimpleNamespace(disconnect=lambda *args: False)
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.ssh_broker", broker)
    monkeypatch.setattr(vasp_slurm_agent, "ssh_broker", broker, raising=False)
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Windows study")["id"]
    cid = registry.create("hpc", "Cluster", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "code"})["id"]
    registry.set_secret(cid, "password")
    registry.bind(project, cid)
    registry._save_probe(cid, registry.get(cid)["revision"], {"auth_status": "required", "connection_status": "untested", "needs": "code", "prompt": "Verification code:", "detail": "Session expired"})
    entered = []
    monkeypatch.setattr(ConnectionRegistry, "connect", lambda self, identifier, code="": entered.append(code) or {"status": "connected"})
    monkeypatch.setattr(ConnectionRegistry, "_probe_hpc", lambda self, item: {"auth_status": "available", "connection_status": "connected", "schedulers": ["slurm"], "detail": "SSH connected."})
    monkeypatch.setattr(ConnectionRegistry, "discover_environments", lambda *args, **kwargs: {})
    view = AppTest.from_string(f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import chat_panel
st.session_state.setdefault('setup_chat_{project}_kind', 'hpc')
st.session_state.setdefault('setup_chat_{project}_hpc_state', {{'schema_version': 2, 'kind': 'hpc', 'record_id': {cid!r}, 'detect': None, 'needs_code': None, 'attempted': True, 'discovered': True, 'pending_host': None, 'history': []}})
chat_panel(ConnectionRegistry(), {project!r})
""").run()
    assert not view.exception and widget(view.text_input, "Login code").proto.type == widget(view.text_input, "Login code").proto.PASSWORD
    widget(view.text_input, "Login code").set_value("246810")
    widget(view.button, "Submit code").click().run()
    assert not view.exception and not view.error and entered == ["246810"]
    assert any("Connected to the cluster" in item.value for item in view.success)
    assert not any(item.label == "Login code" for item in view.text_input)
    assert "246810" not in json.dumps({key: str(value) for key, value in view.session_state.items()})
    assert "246810" not in json.dumps(registry.get(cid))


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_key_plus_code_login_does_not_require_or_store_password(registry, monkeypatch, system):
    monkeypatch.setattr(connections.platform, "system", lambda: system)
    class Remote(FakeTransport):
        def __init__(self, config, password=None, *, code="", master=False):
            self.config, self.password, self.code, self.master = config, password, code, master
            FakeTransport.opened.append(self)
    monkeypatch.setattr(transport, "SSHTransport", Remote)
    FakeTransport.opened, FakeTransport.plan = [], {(False, ""): VerificationRequired("code", "Verification code:")}
    cid = registry.create("hpc", "Key and code", {"host": "hpc.invalid", "user": "researcher", "auth_mode": "key",
        "identity_file": str(registry.home / "id_ed25519")})["id"]
    result = registry.detect_auth(cid)
    assert result["status"] == "code" and result["prompt"] == "Verification code:"
    record = registry.get(cid)
    assert record["settings"]["auth_mode"] == "code" and not record["has_secret"]
    assert bool(record["settings"]["ssh_control_path"]) == (system != "Windows")
    assert registry.ssh_password(cid) == ""
    assert registry.connect(cid, code="246810")["status"] == "connected"
    opened = FakeTransport.opened[-1]
    assert opened.master and opened.code == "246810" and opened.password == ""
    assert opened.config.ssh_identity_file == str(registry.home / "id_ed25519")
    assert not registry.fake_keychain.values if hasattr(registry, "fake_keychain") else not registry.get(cid)["has_secret"]
    assert "246810" not in json.dumps(registry.get(cid))


def test_setup_recognizes_key_plus_code_without_password_prompt(monkeypatch):
    def detect(self, identifier):
        self._patch_settings(identifier, auth_mode="code")
        return {"status": "code", "prompt": "Verification code:", "detail": "Login code required."}
    monkeypatch.setattr(ConnectionRegistry, "detect_auth", detect)
    view = chat_app()
    widget(view.pills, "Connect").set_value("hpc").run()
    address(view, "researcher@hpc.invalid")
    assert not view.exception and widget(view.text_input, "Login code")
    assert not any(field.label == "SSH password" for field in view.text_input)
    assert not ConnectionRegistry().list(kind="hpc")[0]["has_secret"]
