"""Connection persistence and credential boundaries with a fake OS keychain."""

import json
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import agent, connections, credentials
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.workspace import Workspace


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
def registry(tmp_path, monkeypatch):
    keychain = Keychain()
    monkeypatch.setattr(credentials, "_keyring", lambda: keychain)
    result = ConnectionRegistry(tmp_path / "workspace")
    result.fake_keychain = keychain
    return result


def model(registry, name="Model"):
    return registry.create("model", name, {"provider": "responses", "model": "chosen-model"})["id"]


def hpc(registry, **extra):
    return registry.create("hpc", "Cluster", {"host": "cluster.example", "user": "researcher",
        "remote_base": "/scratch/researcher", "scheduler": "slurm", **extra})["id"]


def test_secrets_are_per_connection_and_never_enter_database(registry, monkeypatch):
    a, b = model(registry, "Account A"), model(registry, "Account B")
    registry.set_secret(a, "account-a-only")
    registry.set_secret(b, "account-b-only")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-override")
    settings = registry.model_settings(a)
    assert settings.api_key is None and settings.connection_id == a and not settings.allow_env
    assert registry.model_settings(a, resolve_secret=True).api_key == "account-a-only"
    assert registry.model_settings(b, resolve_secret=True).api_key == "account-b-only"
    restarted = ConnectionRegistry(registry.home)
    assert restarted.model_settings(a, resolve_secret=True).api_key == "account-a-only"
    with registry.workspace._connection() as db:
        contents = json.dumps([tuple(row) for row in db.execute("SELECT * FROM connections")])
        contents += json.dumps([tuple(row) for row in db.execute("SELECT * FROM connection_events")])
    assert "account-a-only" not in contents and "account-b-only" not in contents


def test_rotation_and_revocation_apply_to_old_model_references(registry, monkeypatch):
    identifier = model(registry)
    registry.set_secret(identifier, "first-secret")
    reference = registry.model_settings(identifier)
    registry.set_secret(identifier, "second-secret")
    assert agent._resolve_connection(reference).api_key == "second-secret"
    assert "first-secret" not in registry.fake_keychain.values.values()
    registry.revoke(identifier)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-is-forbidden")
    with pytest.raises(agent.AgentError, match="revoked"):
        agent._request_structured("{}", agent._object({"ok": {"type": "boolean"}}), reference,
                                  instructions="Return JSON")
    assert registry.fake_keychain.values == {}
    with pytest.raises(ValueError):
        registry.set_secret(identifier, "third-secret")


def test_revocation_still_blocks_requests_when_keychain_delete_fails(registry, monkeypatch):
    identifier = model(registry)
    registry.set_secret(identifier, "saved-secret")
    monkeypatch.setattr(registry.fake_keychain, "delete_password", lambda *args: None)
    result = registry.revoke(identifier)
    assert result["revoked"] and result["credential_cleanup_pending"]
    assert not result["has_secret"]
    with pytest.raises(ValueError):
        registry.model_settings(identifier, resolve_secret=True)


def test_keychain_write_and_database_failure_keep_old_credential(registry, monkeypatch):
    identifier = model(registry)
    registry.set_secret(identifier, "old-value")
    original = registry._event

    def fail(db, identifier, kind, details=None):
        if kind == "credential.updated":
            raise OSError("Simulated storage failure")
        return original(db, identifier, kind, details)

    monkeypatch.setattr(registry, "_event", fail)
    with pytest.raises(credentials.CredentialError):
        registry.set_secret(identifier, "new-value")
    assert registry.model_settings(identifier, resolve_secret=True).api_key == "old-value"
    assert "new-value" not in registry.fake_keychain.values.values()


def test_secret_cannot_follow_another_provider_or_endpoint(registry):
    identifier = model(registry)
    registry.set_secret(identifier, "private-value")
    for settings in ({"provider": "anthropic"}, {"base_url": "https://different.example/v1"}):
        with pytest.raises(ValueError, match="new connection"):
            registry.update(identifier, settings=settings)
    assert registry.update(identifier, settings={"model": "another-authorized-model"})["settings"]["model"] == "another-authorized-model"


@pytest.mark.parametrize("settings", [
    {"provider": "responses", "model": "model", "api_key": "value"},
    {"provider": "responses", "model": "model", "base_url": "https://user:pass@api.example/v1"},
    {"provider": "responses", "model": "model", "base_url": "https://api.example/v1?key=value"},
    {"provider": "local", "model": "model", "base_url": "https://remote.example/v1"},
    {"provider": "codex", "base_url": "https://api.example/v1"},
])
def test_invalid_model_settings_never_saved(registry, settings):
    with pytest.raises(ValueError):
        registry.create("model", "Invalid", settings)
    assert registry.list() == []


def test_shared_hpc_has_different_remote_roots_and_private_credentials(registry):
    identifier = hpc(registry, auth_mode="password")
    registry.set_secret(identifier, "ssh-private-value")
    a, b = [registry.workspace.create_project(name)["id"] for name in ("A", "B")]
    environment = registry.create_environment(identifier, "VASP", "vasp", {
        "vasp_command": "srun vasp_std", "potcar_root": "/licensed/potentials", "partition": "compute"})
    with pytest.raises(ValueError, match="Connect"):
        registry.resolve_cluster(a, environment["id"])
    registry.bind(a, identifier)
    registry.bind(b, identifier)
    ca, cb = [registry.resolve_cluster(p, environment["id"]) for p in (a, b)]
    assert ca.remote_root != cb.remote_root and ca.remote_root.endswith(a) and cb.remote_root.endswith(b)
    assert ca.connection_id == cb.connection_id == identifier
    assert "ssh-private-value" not in json.dumps(ca.to_dict())
    registry.revoke(identifier)
    with pytest.raises(ValueError):
        registry.ssh_password(identifier)
    with pytest.raises(ValueError):
        registry.resolve_cluster(a, environment["id"])


def test_environment_values_are_validated_before_saving(registry):
    identifier = hpc(registry)
    for config in ({"host": "another.example"}, {"partition": "compute", "tasks": 0},
                   {"setup_commands": ["export API_KEY=never-store-this"]}):
        with pytest.raises(ValueError):
            registry.create_environment(identifier, "Bad", "vasp", config)
    assert registry.environments() == []


def test_model_probe_saves_only_actual_request_result(registry, monkeypatch):
    identifier = model(registry)
    calls = []

    def request(payload, schema, settings, **kwargs):
        calls.append(settings.connection_id)
        raise agent.AgentError("Rate limit reached. Try again later.")

    monkeypatch.setattr(agent, "_request_structured", request)
    result = registry.probe(identifier)
    assert result["connection_status"] == "error" and result["auth_status"] == "unknown"
    assert "Rate limit" in result["probe"]["detail"] and calls == [identifier]
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: '{"status":"connected"}')
    assert registry.probe(identifier)["connection_status"] == "connected"


def test_revocation_during_probe_cannot_restore_connected_status(registry, monkeypatch):
    identifier = model(registry)

    def probe(item):
        registry.revoke(identifier)
        return {"auth_status": "available", "connection_status": "connected"}

    monkeypatch.setattr(registry, "_probe_model", probe)
    with pytest.raises(ValueError, match="revoked"):
        registry.probe(identifier)
    assert registry.get(identifier)["connection_status"] == "revoked"


def test_cli_probe_checks_selected_install_and_does_not_claim_inference_from_login(registry, monkeypatch, tmp_path):
    identifier = registry.create("model", "Claude Code", {"provider": "claude_cli"})["id"]
    selected = str(tmp_path / "claude.exe")
    monkeypatch.setenv("CLAUDE_CLI_PATH", selected)
    seen = []
    monkeypatch.setattr(connections.shutil, "which", lambda value: seen.append(value) or value)
    monkeypatch.setattr(connections.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='{"loggedIn":false}'))
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: pytest.fail("Must log in first"))
    result = registry.probe(identifier)
    assert result["auth_status"] == "required" and result["connection_status"] == "untested"
    assert seen == [selected]


@pytest.mark.parametrize("remote_home,expected", [("/home/researcher", "/home/researcher/dft-agent"), ("", None), ("relative/home", None), ("/home/../other", None)])
def test_hpc_probe_reports_real_sections_without_claiming_submission(registry, monkeypatch, remote_home, expected):
    from vasp_slurm_agent import transport

    identifier = hpc(registry)
    response = "\n".join(["__DFT_USER__", "researcher", "__DFT_HOME__", remote_home, "__DFT_TOOLS__", "/bin/python3", "/bin/qsub",
                          "/bin/qstat", "/bin/qdel", "/opt/qe/pw.x", "__DFT_QUEUES__", "workq", "__DFT_WORKDIR__", "writable"])

    class Remote:
        def __init__(self, config, password):
            assert password == "" and config.connection_id == identifier
        def run(self, command):
            # Discovering executables is allowed; invoking submission is not.
            import re
            assert not re.search(r"(?:^|[;\n])\s*(?:sbatch|qsub)\s", command)
            return transport.Result(0, response)
        def close(self):
            pass

    monkeypatch.setattr(transport, "SSHTransport", Remote)
    result = registry.probe(identifier)
    assert result["connection_status"] == "connected"
    assert result["probe"]["schedulers"] == ["pbs"]
    assert result["probe"]["submission_verified"] is False
    assert result["probe"]["suggested_remote_base"] == expected


def test_upgrade_from_project_only_database_keeps_project_and_history(tmp_path, monkeypatch):
    from vasp_slurm_agent import workspace as module

    migrations = module._MIGRATIONS
    monkeypatch.setattr(module, "_MIGRATIONS", migrations[:1])
    old = Workspace(tmp_path)
    project = old.create_project("Existing")
    session = old.create_session(project["id"])
    old.append_message(project["id"], session["id"], "user", "Preserve this goal")
    monkeypatch.setattr(module, "_MIGRATIONS", migrations)
    upgraded = ConnectionRegistry(tmp_path)
    assert upgraded.workspace.get_session(project["id"], session["id"])["messages"][0]["content"] == "Preserve this goal"
    assert upgraded.create("model", "Codex", {"provider": "codex"})["connection_status"] == "untested"


def test_failed_commit_and_keychain_cleanup_remain_tracked_for_revocation(registry, monkeypatch):
    identifier = model(registry)
    registry.set_secret(identifier, 'old-value')
    original_event, original_delete = registry._event, registry.fake_keychain.delete_password
    def failed_event(db, identifier, kind, details=None):
        if kind == 'credential.updated':
            raise OSError('Simulated storage failure')
        return original_event(db, identifier, kind, details)
    monkeypatch.setattr(registry, '_event', failed_event)
    monkeypatch.setattr(registry.fake_keychain, 'delete_password', lambda *args: None)
    with pytest.raises(credentials.CredentialError):
        registry.set_secret(identifier, 'orphan-attempt')
    assert registry.model_settings(identifier, resolve_secret=True).api_key == 'old-value'
    assert 'orphan-attempt' in registry.fake_keychain.values.values()
    assert registry.revoke(identifier)['credential_cleanup_pending']
    monkeypatch.setattr(registry.fake_keychain, 'delete_password', original_delete)
    assert not registry.revoke(identifier).get('credential_cleanup_pending')
    assert registry.fake_keychain.values == {}


def test_noop_keychain_write_does_not_replace_existing_credential(registry, monkeypatch):
    identifier = model(registry)
    registry.set_secret(identifier, 'old-value')
    monkeypatch.setattr(registry.fake_keychain, 'set_password', lambda *args: None)
    with pytest.raises(credentials.CredentialError):
        registry.set_secret(identifier, 'new-value')
    assert registry.model_settings(identifier, resolve_secret=True).api_key == 'old-value'


def test_generic_program_profile_has_project_bound_resources(registry):
    identifier = hpc(registry)
    pid = registry.workspace.create_project('Analysis')['id']
    registry.bind(pid, identifier)
    env = registry.create_environment(identifier, 'Python', 'python', {'command':'python3', 'partition':'cpu', 'tasks':1})
    resolved = registry.resolve_environment(pid, env['id'])
    assert resolved['environment']['engine'] == 'python'
    assert resolved['config'].remote_root.endswith(pid)
    assert resolved['config'].tasks == 1
    with pytest.raises(ValueError, match='script adapter'):
        registry.resolve_cluster(pid, env['id'])
    with pytest.raises(ValueError):
        registry.create_environment(identifier, 'Broken', 'python', {'command':'python3', 'partition':'cpu', 'tasks':0})


def test_connection_probe_and_rename_preserve_running_build_receipt(registry, monkeypatch):
    cid = hpc(registry, auth_mode="password")
    registry.set_secret(cid, "first-password")
    receipt = {"status": "running", "plan_id": "reviewed-plan", "job_id": "123", "receipt": {"work_dir": "/scratch/build"}}
    discovery = {"suggested": {"partition": "cpu"}}
    registry._store_probe(cid, "qe_install", receipt)
    registry._store_probe(cid, "discovery", discovery)
    monkeypatch.setattr(registry, "_probe_hpc", lambda item: {
        "auth_status": "available", "connection_status": "connected", "schedulers": ["slurm"], "detail": "SSH connected."})
    tested = registry.probe(cid)
    assert tested["probe"]["qe_install"] == receipt
    assert tested["probe"]["discovery"] == discovery
    renamed = registry.update(cid, name="My cluster")
    assert renamed["probe"] == tested["probe"]
    assert renamed["revision"] == tested["revision"]
    assert renamed["connection_status"] == "connected"
    assert ConnectionRegistry(registry.home).get(cid)["probe"]["qe_install"] == receipt
    rotated = registry.set_secret(cid, "new-password")
    assert rotated["connection_status"] == "untested"
    assert rotated["probe"]["qe_install"] == receipt
    assert rotated["probe"]["discovery"] == discovery
    assert "first-password" not in registry.fake_keychain.values.values()


def test_install_never_guesses_an_ambiguous_scheduler(registry):
    cid = hpc(registry, scheduler="auto")
    registry._store_probe(cid, "schedulers", ["slurm", "pbs"])
    with pytest.raises(ValueError, match="Choose Slurm or PBS"):
        registry._qe_base_config(registry.get(cid), partition="cpu")
    registry.record_settings(cid, scheduler="pbs")
    config = registry._qe_base_config(registry.get(cid), partition="workq")
    assert config.scheduler == "pbs" and config.qe_command == "mpiexec pw.x"


def test_new_qe_install_registers_its_own_binary_directory(registry, monkeypatch):
    from vasp_slurm_agent import qe_setup
    cid = hpc(registry)
    old = registry.create_environment(cid, "Older QE", "qe", {"partition": "cpu", "qe_bin_dir": "/old/bin", "qe_pseudo_root": "/pseudo"})
    registry._store_probe(cid, "qe_install", {"status": "running", "plan_id": "new", "partition": "cpu", "account": "allocation",
        "setup_commands": ["module load compiler"], "plan": {"prefix": "/new", "bin_dir": "/new/bin", "version": "7.4.1"},
        "receipt": {"job_id": "12"}})
    class Remote:
        def close(self):
            pass
    monkeypatch.setattr(registry, "_qe_transport", lambda item: Remote())
    monkeypatch.setattr(qe_setup, "poll_install", lambda *args, **kwargs: {"status": "installed", "bin_dir": "/new/bin"})
    result = registry.qe_install_status(cid)
    assert result["environment_id"] != old["id"]
    assert registry.environment(result["environment_id"])["config"]["qe_bin_dir"] == "/new/bin"
    assert registry.environment(old["id"])["config"]["qe_bin_dir"] == "/old/bin"
    assert registry.qe_install_status(cid)["environment_id"] == result["environment_id"]


@pytest.mark.parametrize("operation", ["revoke", "delete"])
@pytest.mark.parametrize("failure", [TimeoutError, OSError])
def test_revocation_is_durable_before_busy_disconnect(registry, monkeypatch, operation, failure):
    cid = hpc(registry, auth_mode="password")
    registry.set_secret(cid, "saved-password")
    observed = []

    def disconnect(identifier):
        persisted = ConnectionRegistry(registry.home)
        observed.append(persisted.get(identifier)["revoked"])
        with pytest.raises(ValueError, match="active HPC"):
            persisted.ssh_password(identifier)
        raise failure("The previous SSH command has not released its session.")

    monkeypatch.setattr(registry, "disconnect", disconnect)
    result = getattr(registry, operation)(cid)
    assert observed == [True]
    assert result["revoked"] and not result["has_secret"] and not registry.fake_keychain.values
    assert result["session_cleanup_pending"]
    assert result["deleted"] == (operation == "delete")


def test_windows_rotation_and_settings_remain_saved_when_session_cleanup_fails(registry, monkeypatch):
    cid = hpc(registry, auth_mode="password")
    registry.set_secret(cid, "first-password")
    monkeypatch.setattr(connections.platform, "system", lambda: "Windows")
    seen = []

    def disconnect(identifier):
        seen.append(ConnectionRegistry(registry.home).get(identifier))
        raise TimeoutError("A busy transport timed out while closing.")

    monkeypatch.setattr(registry, "disconnect", disconnect)
    rotated = registry.set_secret(cid, "second-password")
    assert rotated["session_cleanup_pending"] and rotated["secret_version"] == 2
    assert seen[-1]["secret_version"] == 2
    assert registry.ssh_password(cid) == "second-password"
    assert "first-password" not in registry.fake_keychain.values.values()
    changed = registry.update(cid, settings={"identity_file": str(registry.home / "new-key")})
    assert changed["session_cleanup_pending"]
    assert seen[-1]["settings"]["identity_file"] == changed["settings"]["identity_file"]
    assert changed["auth_status"] == "unknown" and changed["connection_status"] == "untested"
