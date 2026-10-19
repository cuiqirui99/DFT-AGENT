"""Opt-in keychain storage never touches a real keyring in tests."""

import sys
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import cli, credentials, settings


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.store:
            raise RuntimeError("not found")
        del self.store[(service, name)]


@pytest.fixture
def keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: fake)
    return fake


def test_save_load_delete(keyring):
    assert credentials.keyring_available()
    assert credentials.stored_key("anthropic") is None
    credentials.save_key("anthropic", " key-123 ")
    assert keyring.store == {("dft-agent", "anthropic"): "key-123"}
    assert settings.load_settings()["keychain"] is True
    assert credentials.stored_key("anthropic") == "key-123"
    assert credentials.load_key("qwen") is None
    credentials.delete_key("anthropic")
    credentials.delete_key("anthropic")
    assert credentials.stored_key("anthropic") is None


def test_rejects_bad_input(keyring):
    with pytest.raises(credentials.CredentialError):
        credentials.save_key("codex", "x")
    with pytest.raises(credentials.CredentialError):
        credentials.save_key("anthropic", "two\nlines")
    with pytest.raises(credentials.CredentialError):
        credentials.load_key("unknown")


def test_unavailable_keyring_is_reported(monkeypatch):
    def broken():
        raise credentials.CredentialError("No system keychain is available on this computer.")

    monkeypatch.setattr(credentials, "_keyring", broken)
    assert not credentials.keyring_available()
    settings.save_settings({"keychain": True})
    assert credentials.stored_key("anthropic") is None


def test_cli_key_commands(keyring, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "status"])
    assert cli.main() == 0
    assert "No API key" in capsys.readouterr().out
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "secret-key")
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "set", "deepseek"])
    assert cli.main() == 0
    assert keyring.store[("dft-agent", "deepseek")] == "secret-key"
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "status"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DeepSeek" in out and "secret-key" not in out
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "clear", "deepseek"])
    assert cli.main() == 0
    assert keyring.store == {}


@pytest.mark.parametrize("error_name,message", [("PasswordDeleteError", "access denied"), ("RuntimeError", "service not found")])
def test_failed_delete_keeps_key_and_reports_failure(keyring, monkeypatch, capsys, error_name, message):
    credentials.save_key("anthropic", "test-secret")

    def denied(*args):
        raise type(error_name, (Exception,), {})(message)

    monkeypatch.setattr(keyring, "delete_password", denied)
    with pytest.raises(credentials.CredentialError, match="Cannot remove"):
        credentials.delete_key("anthropic")
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "clear", "anthropic"])
    assert cli.main() == 1
    assert credentials.load_key("anthropic") == "test-secret"
    captured = capsys.readouterr()
    assert "Cannot remove" in captured.err
    assert "test-secret" not in captured.out + captured.err


def test_delete_verifies_backend_did_remove_key(keyring, monkeypatch):
    credentials.save_key("anthropic", "test-secret")
    monkeypatch.setattr(keyring, "delete_password", lambda *args: None)
    with pytest.raises(credentials.CredentialError, match="still contains"):
        credentials.delete_key("anthropic")


def backend_type(module, name):
    return type(name, (FakeKeyring,), {"__module__": module})()


def select_backend(monkeypatch, backend):
    # No OS keychain is opened by these backend-selection tests.
    monkeypatch.setitem(sys.modules, "keyring", SimpleNamespace(
        get_keyring=lambda: backend, errors=SimpleNamespace()))


@pytest.mark.parametrize("module,name", [
    ("keyring.backends.macOS", "Keyring"),
    ("keyring.backends.Windows", "WinVaultKeyring"),
    ("keyring.backends.SecretService", "Keyring"),
    ("keyring.backends.libsecret", "Keyring"),
    ("keyring.backends.kwallet", "DBusKeyring"),
    ("keyring.backends.kwallet", "DBusKeyringKWallet4"),
])
def test_native_keychains_remain_supported(monkeypatch, module, name):
    backend = backend_type(module, name)
    select_backend(monkeypatch, backend)
    assert credentials._keyring() is backend
    credentials.save_key("anthropic", "placeholder-native-key")
    assert credentials.load_key("anthropic") == "placeholder-native-key"
    credentials.delete_key("anthropic")
    assert not backend.store


@pytest.mark.parametrize("module,name", [
    ("keyrings.alt.file", "PlaintextKeyring"),
    ("keyrings.alt.file", "EncryptedKeyring"),
    ("custom_file_store", "Keyring"),
    ("keyring.backends.fail", "Keyring"),
    ("keyring.backends.null", "Keyring"),
])
def test_unverified_stores_are_rejected_before_access(monkeypatch, module, name):
    backend = backend_type(module, name)
    backend.store[credentials.SERVICE, "anthropic"] = "must-not-read"
    select_backend(monkeypatch, backend)
    assert not credentials.keyring_available()
    for operation in (lambda: credentials.save_key("anthropic", "must-not-write"),
                      lambda: credentials.load_key("anthropic"),
                      lambda: credentials.delete_key("anthropic")):
        with pytest.raises(credentials.CredentialError, match="system keychain"):
            operation()
    assert list(backend.store.values()) == ["must-not-read"]


def test_chain_rejects_file_fallback_and_pins_safe_backend(monkeypatch):
    native = backend_type("keyring.backends.SecretService", "Keyring")
    other_native = backend_type("keyring.backends.libsecret", "Keyring")
    file_store = backend_type("keyrings.alt.file", "PlaintextKeyring")
    chain = backend_type("keyring.backends.chainer", "ChainerBackend")
    select_backend(monkeypatch, chain)
    for children in ([], [native, file_store], [file_store, native]):
        chain.backends = children
        with pytest.raises(credentials.CredentialError):
            credentials._keyring()
    chain.backends = [native, other_native]
    selected = credentials._keyring()
    assert selected is native
    chain.backends = [file_store]
    selected.set_password(credentials.SERVICE, "test", "native-only")
    assert not file_store.store


def test_keychain_discovery_error_does_not_expose_backend_details(monkeypatch):
    def fail():
        raise RuntimeError("private-backend-detail")
    monkeypatch.setitem(sys.modules, "keyring", SimpleNamespace(
        get_keyring=fail, errors=SimpleNamespace()))
    with pytest.raises(credentials.CredentialError, match="could not be opened") as caught:
        credentials._keyring()
    assert "private-backend-detail" not in str(caught.value)
