"""Broker authentication and lifecycle checks using a real loopback SSH server."""
from dataclasses import replace
from pathlib import Path
import secrets
import threading
import time

import pytest

paramiko = pytest.importorskip('paramiko')
from test_windows_transport import native_home, ssh_server  # noqa: F401
from vasp_slurm_agent import ssh_broker
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.transport import AuthenticationError, SSHTransport, TransportError, VerificationRequired
from vasp_slurm_agent.workspace import Workspace


@pytest.fixture
def broker(native_home, ssh_server, tmp_path, monkeypatch):
    home = Workspace(tmp_path / 'workspace').home
    registry = ConnectionRegistry(home)
    record = registry.create('hpc', 'Local MFA', {
        'host': '127.0.0.1', 'user': 'test-user', 'port': ssh_server.config.port,
        'auth_mode': 'code', 'scheduler': 'slurm', 'remote_base': '/cluster'})
    config = ClusterConfig(host='127.0.0.1', user='test-user', port=ssh_server.config.port,
                           remote_root='/cluster', partition='test', engine='qe', qe_pseudo_root='/cluster/upf',
                           connection_id=record['id'], connection_home=str(home), connect_timeout=3)
    capability = secrets.token_bytes(32)
    monkeypatch.setattr(ssh_broker, '_capability', lambda home, create=False: capability)
    server = ssh_broker.BrokerServer(home, capability)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(5)
    ssh_server.authentication.mfa = True
    yield server, registry, config, capability
    server.stop()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_sign_in_clears_credentials_and_proxy_close_keeps_session(broker, ssh_server, monkeypatch):
    server, registry, config, _ = broker
    from test_connections import Keychain
    from vasp_slurm_agent import credentials
    keychain = Keychain()
    monkeypatch.setattr(credentials, '_keyring', lambda: keychain)
    registry.set_secret(config.connection_id, 'test-password')
    with SSHTransport(config, password='test-password', code='123456', master=True) as transport:
        assert transport.run('true').returncode == 0
    session = server.sessions[config.connection_id]
    assert session.transport._password == session.transport._code == ''
    assert ssh_broker.status(registry.home, config.connection_id) == {'connected': True}
    with SSHTransport(config) as transport:
        assert transport.run('true').returncode == 0
    assert ssh_server.authentication.accepted == 1
    for path in Path(config.connection_home).rglob('*'):
        if path.is_file():
            assert b'test-password' not in path.read_bytes()
            assert b'123456' not in path.read_bytes()


def test_wrong_code_does_not_establish_session(broker, ssh_server):
    server, registry, config, _ = broker
    with pytest.raises(AuthenticationError):
        ssh_broker.sign_in(config, 'test-password', '999999')
    assert ssh_broker.status(registry.home, config.connection_id) == {'connected': False}
    assert not server.sessions
    assert ssh_server.authentication.accepted == 0


def test_wrong_ipc_capability_cannot_use_existing_session(broker, monkeypatch, ssh_server):
    server, registry, config, capability = broker
    ssh_broker.sign_in(config, 'test-password', '123456')
    monkeypatch.setattr(ssh_broker, '_capability', lambda home, create=False: secrets.token_bytes(32))
    with pytest.raises(TransportError):
        ssh_broker._request(registry.home, {'operation': 'run', 'config': ssh_broker._configuration(config), 'command': 'true'})
    assert not ssh_server.calls
    monkeypatch.setattr(ssh_broker, '_capability', lambda home, create=False: capability)
    assert ssh_broker.status(registry.home, config.connection_id)['connected']


@pytest.mark.parametrize('action', ['disconnect', 'expiry', 'revoke', 'destination'])
def test_sessions_are_invalidated_without_reauthentication(broker, ssh_server, action):
    server, registry, config, _ = broker
    ssh_broker.sign_in(config, 'test-password', '123456')
    native = server.sessions[config.connection_id].transport
    if action == 'disconnect':
        ssh_broker.disconnect(registry.home, config.connection_id)
    elif action == 'expiry':
        server.sessions[config.connection_id].used -= server.idle_timeout + 1
    elif action == 'revoke':
        registry.revoke(config.connection_id)
    else:
        registry.update(config.connection_id, settings={'host': 'changed.invalid'})
    assert not ssh_broker.status(registry.home, config.connection_id)['connected']
    with pytest.raises(TransportError):
        ssh_broker.BrokerSSH(config).run('true', 5)
    assert native._closed
    assert ssh_server.authentication.accepted == ssh_server.authentication.code_attempts == 1


def test_config_cannot_redirect_existing_session(broker, ssh_server):
    server, registry, config, _ = broker
    ssh_broker.sign_in(config, 'test-password', '123456')
    with pytest.raises(VerificationRequired, match='destination'):
        ssh_broker.BrokerSSH(replace(config, host='other.invalid')).run('true', 5)
    assert not ssh_server.calls


def test_status_and_disconnect_do_not_start_a_broker(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh_broker, 'start', lambda *args: pytest.fail('No daemon should start'))
    monkeypatch.setattr(ssh_broker, '_capability', lambda *args, **kwargs: pytest.fail('No keychain access needed'))
    assert ssh_broker.status(tmp_path, 'missing') == {'connected': False}
    assert ssh_broker.disconnect(tmp_path, 'missing') == {'connected': False}
    assert not (tmp_path / 'runtime').exists()


def test_idle_daemon_stops_and_removes_endpoint(tmp_path):
    server = ssh_broker.BrokerServer(tmp_path, secrets.token_bytes(32), idle_timeout=0.1)
    thread = threading.Thread(target=server.serve)
    thread.start()
    assert server.ready.wait(5)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not (tmp_path / 'runtime' / 'ssh-broker.json').exists()


def test_key_and_code_use_secure_prompt_then_reuse_session(broker, ssh_server, native_home, monkeypatch):
    server, registry, config, _ = broker
    key = paramiko.RSAKey.generate(2048)
    key_path = native_home / '.ssh' / 'mfa_key'
    key_path.parent.mkdir(exist_ok=True)
    key.write_private_key_file(str(key_path))
    ssh_server.authentication.public_key = key
    registry.update(config.connection_id, settings={'identity_file': str(key_path)})
    config = replace(config, ssh_identity_file=str(key_path))
    monkeypatch.setattr('builtins.input', lambda *args: pytest.fail('No terminal prompt'))
    # First-factor discovery asks the GUI for the code instead of reading stdin.
    from vasp_slurm_agent.windows_ssh import WindowsSSH
    native = WindowsSSH(config, '')
    with pytest.raises(VerificationRequired, match='login code'):
        native.run('true', 5)
    native.close()
    assert ssh_server.authentication.accepted == 0
    ssh_broker.sign_in(config, code='123456')
    with SSHTransport(config) as transport:
        assert transport.run('true').returncode == 0
    assert ssh_server.authentication.password_attempts == 0
    assert ssh_server.authentication.accepted == 1
    assert ssh_server.authentication.code_attempts == 1
    assert server.sessions[config.connection_id].transport._code == ''
