"""Guided setup uses real storage; probes and the OS keychain are mocked here."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import credentials, setup_assistant as setup
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.secret_guard import redact, reject_ordinary_text


@pytest.mark.parametrize("secret", [
    "sk-proj-1234567890abcdef", "sk-ant-1234567890abcdef", "xai-1234567890abcdef",
    "password: hidden", '"api_key": "hidden"', "OPENAI_API_KEY=hidden", "OTP 123456",
    "device code ABCD-EFGHI", "device_code: ABCD-EFGHI", "Bearer abc123",
    "my password is hidden", "OTP123456", "密码：hidden", "验证码是123456", "验证码123456", "API密钥=hidden",
    "https://user:hidden@api.invalid/v1",
    "-----BEGIN OPENSSH PRIVATE KEY-----\nhidden\n-----END OPENSSH PRIVATE KEY-----",
])
def test_secret_guard_rejects_credentials_before_history(secret):
    state = setup.begin("model")
    original = deepcopy(state)
    with pytest.raises(ValueError, match="secure credential"):
        setup.answer(state, "name", secret)
    assert state == original
    assert "hidden" not in redact(secret) and "1234567890abcdef" not in redact(secret)


@pytest.mark.parametrize("text", ["Use 256000 k points.", "How do I add an API key?", "SSH key file /home/me/.ssh/id_ed25519", "Relax silicon.",
                                 "API key is required", "password is optional", "The API key is missing.", "密码是可选的"])
def test_secret_guard_allows_scientific_text_and_setup_questions(text):
    assert reject_ordinary_text(text) is None
    assert redact(text) == text


@pytest.mark.parametrize("text", ["API_KEY=required", "password: optional", "API key is required; OTP123456", "password is optional; sk-test-1234567890abcdef"])
def test_setup_prose_exception_does_not_allow_credentials(text):
    with pytest.raises(ValueError):
        reject_ordinary_text(text)


def complete(kind, values):
    state = setup.begin(kind)
    for field, value in values.items():
        state = setup.answer(state, field, value)
    assert setup.next_question(state) is None
    return state


def test_model_setup_saves_and_binds_without_model_call_or_secret(tmp_path, monkeypatch):
    registry = ConnectionRegistry(tmp_path / "home")
    project = registry.workspace.create_project("Study")
    monkeypatch.setattr(registry, "probe", Mock(side_effect=AssertionError("No implicit probe")))
    state = complete("model", {"name": "Local model", "provider": "local", "model": "test-model", "base_url": "http://localhost:11434/v1"})
    result = setup.save(state, registry, project["id"])
    assert result["settings"]["provider"] == "local" and not result["has_secret"]
    assert state["record_id"] == result["id"]
    assert [item["id"] for item in registry.project_connections(project["id"])] == [result["id"]]
    assert "secret" not in json.dumps(state).lower()
    registry.probe.assert_not_called()


def test_missing_settings_and_provider_changes_require_answers(tmp_path):
    registry = ConnectionRegistry(tmp_path / "home")
    state = setup.begin("model")
    with pytest.raises(ValueError, match="Complete"):
        setup.save(state, registry)
    assert not registry.list()
    state = complete("model", {"name": "Model", "provider": "responses", "model": "test-model"})
    state = setup.answer(state, "provider", "local")
    assert setup.next_question(state)["field"] == "model"
    state = setup.answer(state, "model", "local-model")
    assert setup.next_question(state)["field"] == "base_url"
    state = setup.answer(state, "base_url", "https://remote.invalid/v1")
    with pytest.raises(ValueError, match="loopback"):
        setup.save(state, registry)
    assert not registry.list()


@pytest.mark.parametrize("engine,command", [("python", "python3"), ("wannier90", "srun wannier90.x")])
def test_script_environment_setup_uses_real_registry_and_typed_resources(tmp_path, engine, command):
    registry = ConnectionRegistry(tmp_path / "home")
    connection = registry.create("hpc", "Cluster", {"host": "fixture.invalid", "user": "researcher",
                                 "remote_base": "/scratch/test", "scheduler": "slurm"})
    state = complete("software", {"name": engine, "connection_id": connection["id"], "engine": engine,
                      "config.partition": "cpu", "config.command": command, "config.tasks": "2", "config.nodes": "1"})
    result = setup.save(state, registry)
    assert result["engine"] == engine
    assert result["config"]["command"] == command
    assert result["config"]["tasks"] == 2 and result["config"]["nodes"] == 1
    with pytest.raises(ValueError, match="positive integer"):
        setup.answer(state, "config.tasks", "0")


@pytest.mark.parametrize("metadata", [{"password": "plain"}, {"OPENAI_API_KEY": "plain"}, {"setup_commands": ["export API_KEY=hidden"]}])
def test_advanced_settings_cannot_store_credential_fields(metadata):
    state = setup.begin("hpc")
    original = deepcopy(state)
    with pytest.raises(ValueError, match="secure credential"):
        setup.advanced(state, metadata)
    assert state == original


def test_hpc_and_software_setup_keep_auth_separate(tmp_path):
    registry = ConnectionRegistry(tmp_path / "home")
    project = registry.workspace.create_project("Study")
    state = complete("hpc", {"name": "Cluster", "host": "hpc.invalid", "user": "researcher",
                              "auth_mode": "interactive", "remote_base": "/scratch/researcher", "scheduler": "slurm",
                              "ssh_control_path": "/tmp/existing-socket"})
    hpc = setup.save(state, registry, project["id"])
    assert not hpc["has_secret"] and hpc["settings"]["auth_mode"] == "interactive"
    software = complete("software", {"name": "QE", "connection_id": hpc["id"], "engine": "qe",
                                      "config.partition": "cpu", "config.qe_command": "srun pw.x",
                                      "config.qe_pseudo_root": "/scratch/pseudos"})
    environment = setup.save(software, registry, project["id"])
    assert environment["engine"] == "qe" and "host" not in environment["config"]
    assert environment["config"]["tasks"] == 8
    with pytest.raises(ValueError, match="already saved"):
        setup.save(software, registry, project["id"])
    assert len(registry.environments()) == 1


def test_failed_binding_retry_does_not_duplicate_connection(tmp_path):
    registry = ConnectionRegistry(tmp_path / "home")
    state = complete("model", {"name": "CLI", "provider": "codex"})
    with pytest.raises(ValueError):
        setup.save(state, registry, "not-a-project")
    identifier = state["record_id"]
    assert identifier and len(registry.list()) == 1
    project = registry.workspace.create_project("Study")
    assert setup.save(state, registry, project["id"])["id"] == identifier
    assert len(registry.list()) == 1


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def app():
    return AppTest.from_string("""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import panel
panel(ConnectionRegistry())
""", default_timeout=20).run()


def test_setup_panel_creates_cli_connection_and_resets_new_answers(monkeypatch):
    probe = Mock(return_value={"auth_status": "available", "connection_status": "connected", "mock_probe": True})
    monkeypatch.setattr(ConnectionRegistry, "probe", probe)
    view = app()
    widget(view.text_input, "Name").set_value("My CLI")
    widget(view.button, "Continue").click().run()
    widget(view.selectbox, "Model provider").select("codex")
    widget(view.button, "Continue").click().run()
    widget(view.button, "Save connection").click().run()
    assert not view.exception
    assert len(ConnectionRegistry().list()) == 1
    assert not any(item.label == "API key" for item in view.text_input)
    probe.assert_not_called()
    widget(view.button, "Test connection").click().run()
    probe.assert_called_once()
    widget(view.selectbox, "Connection").select(None).run()
    assert widget(view.text_input, "Name").value == ""


def test_secure_widget_clears_secret_and_revoke_requires_explicit_action(monkeypatch):
    stored = {}

    class Keychain:
        def set_password(self, service, name, value):
            stored[(service, name)] = value

        def get_password(self, service, name):
            return stored.get((service, name))

        def delete_password(self, service, name):
            stored.pop((service, name), None)

    monkeypatch.setattr(credentials, "_keyring", lambda: Keychain())
    registry = ConnectionRegistry()
    connection = registry.create("model", "API", {"provider": "responses", "model": "test-model"})
    view = app()
    widget(view.selectbox, "Connection").select(connection["id"]).run()
    secret = "sk-test-1234567890abcdef"
    widget(view.text_input, "API key").set_value(secret)
    widget(view.button, "Save credential").click().run()
    assert not view.exception and registry.get(connection["id"])["has_secret"]
    assert widget(view.text_input, "API key").value == ""
    assert secret not in json.dumps(view.session_state["setup_global_state"])
    assert widget(view.button, "Revoke").disabled
    widget(view.checkbox, "Revoke this connection").check().run()
    widget(view.button, "Revoke").click().run()
    assert not view.exception and registry.get(connection["id"])["revoked"]
    assert not stored


def test_project_switch_preserves_only_its_own_setup_answers():
    registry = ConnectionRegistry()
    first = registry.workspace.create_project("First")
    second = registry.workspace.create_project("Second")
    view = AppTest.from_string(f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import panel
project = st.selectbox('Project', {[first['id'], second['id']]!r})
panel(ConnectionRegistry(), project)
""").run()
    widget(view.text_input, "Name").set_value("First model")
    widget(view.button, "Continue").click().run()
    widget(view.selectbox, "Project").select(second["id"]).run()
    assert widget(view.text_input, "Name").value == ""
    assert view.session_state[f"setup_{second['id']}_state"]["history"] == []
    widget(view.selectbox, "Project").select(first["id"]).run()
    assert widget(view.selectbox, "Model provider")
    assert view.session_state[f"setup_{first['id']}_state"]["name"] == "First model"
    registry.workspace.archive_project(first["id"])
    view.run()
    assert not view.button and not view.text_input
    assert any("Restore the project" in item.value for item in view.info)


def test_external_connection_update_reloads_draft_and_revocation_hides_actions():
    registry = ConnectionRegistry()
    record = registry.create("model", "Original", {"provider": "responses", "model": "first-model"})
    view = app()
    widget(view.selectbox, "Connection").select(record["id"]).run()
    registry.update(record["id"], settings={"model": "second-model"}, name="Updated")
    view.run()
    assert view.session_state["setup_global_state"]["settings"]["model"] == "second-model"
    assert view.session_state["setup_global_state"]["name"] == "Updated"
    registry.revoke(record["id"])
    view.run()
    assert not view.button and not view.text_input
    assert any("revoked" in item.value for item in view.info)


@pytest.mark.parametrize("invalidate", ["archive", "revoke"])
def test_stale_secure_callback_rechecks_project_and_connection(monkeypatch, invalidate):
    registry = ConnectionRegistry()
    project = registry.workspace.create_project("Study")
    record = registry.create("model", "API", {"provider": "responses", "model": "test-model"})
    view = AppTest.from_string(f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.setup_app import panel
panel(ConnectionRegistry(), {project['id']!r})
""").run()
    widget(view.selectbox, "Connection").select(record["id"]).run()
    widget(view.text_input, "API key").set_value("sk-test-1234567890abcdef")
    if invalidate == "archive":
        registry.workspace.archive_project(project["id"])
    else:
        registry.revoke(record["id"])
    backend = Mock()
    monkeypatch.setattr(credentials, "_keyring", lambda: backend)
    widget(view.button, "Save credential").click().run()
    assert not view.exception and not registry.get(record["id"])["has_secret"]
    backend.set_password.assert_not_called()


def test_ssh_signin_button_launches_only_on_explicit_action(monkeypatch):
    from vasp_slurm_agent import ssh_auth

    registry = ConnectionRegistry()
    record = registry.create("hpc", "HPC", {"host": "hpc.invalid", "user": "user", "auth_mode": "interactive"})
    launch = Mock(return_value={"status": "manual", "command": "ssh hpc.invalid", "reusable": True,
                               "message": "Use your terminal."})
    probe = Mock(side_effect=AssertionError("No automatic probe"))
    monkeypatch.setattr(ssh_auth, "launch", launch)
    monkeypatch.setattr(ConnectionRegistry, "probe", probe)
    view = app()
    widget(view.selectbox, "Set up").select("hpc").run()
    widget(view.selectbox, "Connection").select(record["id"]).run()
    assert not any(item.label in {"SSH password", "OTP", "Device code"} for item in view.text_input)
    launch.assert_not_called()
    widget(view.button, "Open SSH sign-in").click().run()
    assert not view.exception and launch.call_args.args[1] == record["id"]
    probe.assert_not_called()
