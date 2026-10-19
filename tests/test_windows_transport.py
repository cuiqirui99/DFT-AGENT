"""Native SSH/SFTP contract tests against an isolated loopback SSH server.

The server translates the test-only /cluster namespace to a temporary folder;
no real key files, cluster connections, or external services are used.
"""
from __future__ import annotations

import ast
import builtins
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

paramiko = pytest.importorskip("paramiko")
from vasp_slurm_agent import transport as transport_module
from vasp_slurm_agent import windows_ssh
from vasp_slurm_agent.transport import SSHTransport, TransportError
from vasp_slurm_agent.windows_ssh import _accept_new_policy


@pytest.fixture
def native_home(tmp_path, monkeypatch):
    home = tmp_path / "windows user"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(transport_module.platform, "system", lambda: "Windows")
    # Paramiko's default identity lookup must not inspect the real user's .ssh.
    monkeypatch.setattr(os.path, "expanduser", lambda value: str(home) + value[1:] if value.startswith("~") else value)
    class EmptyAgent:
        def get_keys(self): return []
        def close(self): pass
    monkeypatch.setattr(paramiko.client, "Agent", EmptyAgent)
    return home


@pytest.fixture
def ssh_server(tmp_path):
    root = tmp_path / "server files"
    root.mkdir()
    key = paramiko.RSAKey.generate(2048)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    stopped = threading.Event()
    transports = []
    workers = []
    calls = []
    authentication = SimpleNamespace(public_key=None, mfa=False, password_attempts=0, code_attempts=0, accepted=0)

    def path_for(value):
        assert value == "/cluster" or value.startswith("/cluster/")
        path = (root / value.removeprefix("/cluster").lstrip("/")).resolve()
        assert path == root.resolve() or root.resolve() in path.parents
        return path

    class SFTP(paramiko.SFTPServerInterface):
        def stat(self, path):
            try:
                return paramiko.SFTPAttributes.from_stat(path_for(path).stat())
            except OSError as exc:
                return paramiko.SFTPServer.convert_errno(exc.errno)
        lstat = stat
        def open(self, path, flags, attr):
            try:
                descriptor = os.open(path_for(path), flags, 0o600)
                mode = "r+b" if flags & os.O_RDWR else "wb" if flags & os.O_WRONLY else "rb"
                stream = os.fdopen(descriptor, mode)
                handle = paramiko.SFTPHandle(flags)
                handle.readfile = stream
                handle.writefile = stream
                return handle
            except OSError as exc:
                return paramiko.SFTPServer.convert_errno(exc.errno)

    class RemotePaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and (node.value == "/cluster" or node.value.startswith("/cluster/")):
                return ast.copy_location(ast.Constant(str(path_for(node.value))), node)
            return node

    def execute(channel, command):
        # Wait until Paramiko acknowledges exec; closing before the ACK loses
        # fast simulated scheduler replies on Windows.
        if not channel._test_exec_ack.wait(5):
            channel.close()
            return
        try:
            pieces = shlex.split(command.decode())
            assert pieces[:2] == ["bash", "-lc"]
            command = pieces[2]
            calls.append(command)
            if command == "true":
                status = 0
            elif command == "sbatch --parsable submit.sh":
                channel.sendall(b"12345\n")
                status = 0
            elif command == "squeue -h -j 12345 -o %T":
                channel.sendall(b"COMPLETED\n")
                status = 0
            elif command == "test-large-output":
                # Exceed the default SSH window on stderr, then on stdout.
                channel.sendall_stderr(b"e" * 3000000)
                channel.sendall(b"x" * 3000000)
                status = 7
            elif command == "test-timeout":
                channel.sendall(b"partial output")
                stopped.wait(2)
                status = 0
            else:
                args = shlex.split(command)
                assert args[:2] == ["python3", "-c"]
                code = ast.unparse(RemotePaths().visit(ast.parse(args[2])))
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=15)
                channel.sendall(result.stdout)
                channel.sendall_stderr(result.stderr)
                status = result.returncode
            channel.send_exit_status(status)
        except (OSError, EOFError):
            pass  # Expected when the client times out or rejects a host key.
        finally:
            channel.close()

    class ServerTransport(paramiko.Transport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exec_events = {}
        def _send_user_message(self, message):
            super()._send_user_message(message)
            data = message.asbytes()
            if data[:1] == paramiko.common.cMSG_CHANNEL_SUCCESS:
                event = self.exec_events.get(int.from_bytes(data[1:5], 'big'))
                if event is not None:
                    event.set()

    class Server(paramiko.ServerInterface):
        def __init__(self, transport):
            self.transport = transport
            self.password_ok = False
        def check_auth_password(self, username, password):
            authentication.password_attempts += 1
            self.password_ok = username == "test-user" and password == "test-password"
            if not self.password_ok:
                return paramiko.AUTH_FAILED
            if authentication.mfa:
                return paramiko.AUTH_PARTIALLY_SUCCESSFUL
            authentication.accepted += 1
            return paramiko.AUTH_SUCCESSFUL
        def check_auth_interactive(self, username, submethods):
            if authentication.mfa and self.password_ok:
                return paramiko.InteractiveQuery("Verification", "", ("Verification code:", False))
            return paramiko.AUTH_FAILED
        def check_auth_interactive_response(self, responses):
            authentication.code_attempts += 1
            if responses == ["123456"] and self.password_ok:
                authentication.accepted += 1
                return paramiko.AUTH_SUCCESSFUL
            return paramiko.AUTH_FAILED
        def check_auth_publickey(self, username, key):
            if username != "test-user" or key != authentication.public_key:
                return paramiko.AUTH_FAILED
            self.password_ok = True
            if authentication.mfa:
                return paramiko.AUTH_PARTIALLY_SUCCESSFUL
            authentication.accepted += 1
            return paramiko.AUTH_SUCCESSFUL
        def get_allowed_auths(self, username):
            return "keyboard-interactive" if authentication.mfa and self.password_ok else "publickey,password"
        def check_channel_request(self, kind, chanid):
            return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
        def check_channel_exec_request(self, channel, command):
            channel._test_exec_ack = threading.Event()
            self.transport.exec_events[channel.remote_chanid] = channel._test_exec_ack
            worker = threading.Thread(target=execute, args=(channel, command), daemon=True)
            workers.append(worker)
            worker.start()
            return True

    def serve_connection(connection):
        transport = ServerTransport(connection)
        transports.append(transport)
        transport.add_server_key(key)
        transport.set_subsystem_handler("sftp", paramiko.SFTPServer, SFTP)
        try:
            transport.start_server(server=Server(transport))
            while transport.is_active() and not stopped.wait(0.05):
                pass
        except (EOFError, OSError, paramiko.SSHException):
            pass
        finally:
            transport.close()

    def serve():
        while not stopped.is_set():
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            worker = threading.Thread(target=serve_connection, args=(connection,), daemon=True)
            workers.append(worker)
            worker.start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    config = SimpleNamespace(host="127.0.0.1", user="test-user", port=listener.getsockname()[1], connect_timeout=3)
    yield SimpleNamespace(config=config, root=root, key=key, calls=calls, authentication=authentication, transports=transports)
    stopped.set()
    listener.close()
    for transport in transports:
        transport.close()
    thread.join(timeout=3)
    for worker in workers:
        worker.join(timeout=3)


def test_native_real_ssh_and_sftp_roundtrip(native_home, ssh_server, tmp_path):
    content = bytes(range(256)) * 10001 + b"\x00\xff\r\n"
    source = tmp_path / "local input ' 文件.bin"
    source.write_bytes(content)
    output = tmp_path / "download 文件.bin"
    remote = "/cluster/folder ' quote/input.bin"
    expected = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        assert not connection._askpass_path.exists()
        assert connection.upload(source, remote) == expected
        assert connection.download(remote, output) == expected
        assert output.read_bytes() == content
    known = paramiko.HostKeys(str(native_home / ".ssh" / "known_hosts"))
    host = f"[127.0.0.1]:{ssh_server.config.port}"
    assert known.check(host, ssh_server.key)
    assert not list(ssh_server.root.rglob("*.part-*"))
    with pytest.raises(TransportError, match="closed"):
        connection.run("true")


@pytest.mark.parametrize("unix_apis", [True, False])
def test_native_bulk_roundtrip(native_home, ssh_server, tmp_path, monkeypatch, unix_apis):
    if not unix_apis:
        for name in ("fchmod", "killpg", "setsid"):
            monkeypatch.delattr(os, name, raising=False)
        monkeypatch.delattr(signal, "SIGKILL", raising=False)
        monkeypatch.setattr(transport_module, "_run_process", lambda *a, **kw: pytest.fail("Native SSH must not call Unix OpenSSH helpers"))
    files = []
    for name in ("INCAR", "POSCAR", "KPOINTS"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode() * 3000)
        files.append(path)
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        manifest = connection.upload_many(files, "/cluster/bulk")
        assert connection.download_many("/cluster/bulk", tmp_path / "results", [p.name for p in files]) == manifest
    for path in files:
        assert (tmp_path / "results" / path.name).read_bytes() == path.read_bytes()


def test_native_authenticates_with_encrypted_default_private_key(native_home, ssh_server):
    key = paramiko.RSAKey.generate(2048)
    key_path = native_home / ".ssh" / "id_rsa"
    key_path.parent.mkdir()
    key.write_private_key_file(str(key_path), password="test-key-passphrase")
    ssh_server.authentication.public_key = key
    with SSHTransport(ssh_server.config, password="test-key-passphrase") as connection:
        assert connection.run("python3 -c 'print(42)'").stdout.strip() == "42"


def test_native_authenticates_with_agent_key(native_home, ssh_server, monkeypatch):
    key = paramiko.RSAKey.generate(2048)
    ssh_server.authentication.public_key = key
    class TestAgent:
        def get_keys(self): return [key]
        def close(self): pass
    monkeypatch.setattr(paramiko.client, "Agent", TestAgent)
    with SSHTransport(ssh_server.config, password="") as connection:
        assert connection.run("python3 -c 'print(43)'").stdout.strip() == "43"


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_native_checksum_failure_keeps_existing_result(native_home, ssh_server, tmp_path, direction):
    local = tmp_path / "existing-local.bin"
    remote = ssh_server.root / "existing-remote.bin"
    local.write_bytes(b"local original")
    remote.write_bytes(b"remote original")
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        transfer = connection._native.transfer
        def corrupt(local_name, remote_name, *, upload):
            transfer(local_name, remote_name, upload=upload)
            path = (ssh_server.root / remote_name.removeprefix("/cluster/")) if upload else Path(local_name)
            path.write_bytes(b"corrupt transfer")
        connection._native.transfer = corrupt
        with pytest.raises(TransportError, match="mismatch"):
            if direction == "upload":
                connection.upload(local, "/cluster/existing-remote.bin")
            else:
                connection.download("/cluster/existing-remote.bin", local)
    assert local.read_bytes() == b"local original"
    assert remote.read_bytes() == b"remote original"
    assert not list(ssh_server.root.glob("*.part-*"))


def test_native_drains_both_streams_and_preserves_exit_status(native_home, ssh_server):
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        result = connection.run("test-large-output", timeout=10)
    assert result.returncode == 7
    assert result.stdout == "x" * 3000000
    assert result.stderr == "e" * 3000000


def test_native_timeout_and_auth_failure(native_home, ssh_server):
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        result = connection.run("test-timeout", timeout=0.1)
    assert result.returncode == 124 and "partial output" in result.stdout
    with SSHTransport(ssh_server.config, password="wrong-password") as connection:
        with pytest.raises(TransportError, match="authentication failed"):
            connection.run("true")


def test_native_changed_host_key_is_rejected(native_home, ssh_server):
    target = native_home / ".ssh" / "known_hosts"
    target.parent.mkdir()
    keys = paramiko.HostKeys()
    keys.add(f"[127.0.0.1]:{ssh_server.config.port}", "ssh-rsa", paramiko.RSAKey.generate(2048))
    keys.save(str(target))
    original = target.read_bytes()
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        with pytest.raises(TransportError, match="host key has changed"):
            connection.run("true")
    assert target.read_bytes() == original
    assert not ssh_server.calls


def test_accept_new_preserves_concurrent_entries_and_rechecks_races(tmp_path):
    target = tmp_path / ".ssh" / "known_hosts"
    target.parent.mkdir()
    target.write_text("# Existing user comment\n")
    key = paramiko.RSAKey.generate(2048)
    policy = _accept_new_policy(paramiko, target)
    def add(number):
        policy.missing_host_key(paramiko.SSHClient(), f"host-{number}", key)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(12)))
    assert len(paramiko.HostKeys(str(target))) == 12
    assert target.read_text().startswith("# Existing user comment\n")
    with pytest.raises(paramiko.BadHostKeyException):
        policy.missing_host_key(paramiko.SSHClient(), "host-0", paramiko.RSAKey.generate(2048))


def test_native_rejects_unix_control_socket(native_home, ssh_server):
    ssh_server.config.ssh_control_path = "/tmp/control"
    with pytest.raises(TransportError, match="Clear the SSH control"):
        SSHTransport(ssh_server.config, password="")


def test_native_known_hosts_is_utf8_under_ansi_locale(native_home, ssh_server, monkeypatch):
    target = native_home / ".ssh" / "known_hosts"
    target.parent.mkdir()
    target.write_text("# 服务器注释\n", encoding="utf-8")
    def ansi_open(path, mode="r", *args, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return builtins.open(path, mode, *args, **kwargs)
    monkeypatch.setattr(paramiko.hostkeys, "open", ansi_open, raising=False)
    # Demonstrate the default-locale behavior this regression test excludes.
    with pytest.raises(UnicodeDecodeError):
        paramiko.HostKeys(str(target))
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        assert connection.run("python3 -c 'print(44)'").stdout.strip() == "44"
    # The second connection must also load and verify the saved UTF-8 record.
    with SSHTransport(ssh_server.config, password="test-password") as connection:
        assert connection.run("python3 -c 'print(45)'").stdout.strip() == "45"
    assert target.read_text(encoding="utf-8").startswith("# 服务器注释\n")


def test_windows_host_key_lock_uses_one_byte_and_releases_it(tmp_path, monkeypatch):
    calls = []
    def locking(descriptor, operation, length):
        calls.append((operation, length, os.lseek(descriptor, 0, os.SEEK_CUR)))
        if len(calls) == 1:
            raise BlockingIOError("A monitor owns the first attempt")
    monkeypatch.setattr(windows_ssh, "_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2))
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(ValueError, match="test body"):
        with windows_ssh._known_hosts_lock(tmp_path / "known_hosts"):
            raise ValueError("test body")
    assert calls == [(1, 1, 0), (1, 1, 0), (2, 1, 0)]


def test_host_key_replace_closes_file_and_fsyncs_only_regular_file(tmp_path, monkeypatch):
    target = tmp_path / "known_hosts"
    target.write_text("# Keep this comment\n")
    descriptor = None
    original_mkstemp = windows_ssh.tempfile.mkstemp
    original_fsync = os.fsync
    original_replace = os.replace
    def create(*args, **kwargs):
        nonlocal descriptor
        descriptor, name = original_mkstemp(*args, **kwargs)
        return descriptor, name
    def sync(fd):
        assert stat.S_ISREG(os.fstat(fd).st_mode), "Windows does not fsync directories"
        original_fsync(fd)
    def replace(source, destination):
        with pytest.raises(OSError):
            os.fstat(descriptor)
        original_replace(source, destination)
    monkeypatch.setattr(windows_ssh.tempfile, "mkstemp", create)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    windows_ssh._accept_new_policy(paramiko, target).missing_host_key(
        paramiko.SSHClient(), "test-host", paramiko.RSAKey.generate(2048))
    assert target.read_text().startswith("# Keep this comment\n")


def test_native_explicit_identity_and_ssh_alias(native_home, ssh_server):
    key = paramiko.RSAKey.generate(2048)
    key_path = native_home / ".ssh" / "custom identity"
    key_path.parent.mkdir()
    key.write_private_key_file(str(key_path), password="test-key-passphrase")
    (key_path.parent / "config").write_text("Host cluster-alias\n  HostName 127.0.0.1\n")
    ssh_server.authentication.public_key = key
    cfg = SimpleNamespace(**vars(ssh_server.config))
    cfg.host, cfg.ssh_identity_file = "cluster-alias", str(key_path)
    with SSHTransport(cfg, password="test-key-passphrase") as connection:
        assert connection.run("python3 -c 'print(44)'").stdout.strip() == "44"


def test_native_jump_chain_routes_channels_without_target_password(native_home, monkeypatch):
    """Exercise Paramiko routing contracts without opening an external network."""
    folder = native_home / ".ssh"
    folder.mkdir()
    (folder / "config").write_text("Host gateway\n  HostName gateway.internal\n  User gateway-user\n  IdentityFile ~/.ssh/gateway-key\n")
    calls, channels, clients, passwords = [], [], [], []
    class Client:
        def __init__(self):
            self.closed = False
            clients.append(self)
        def connect(self, **kwargs):
            calls.append(kwargs)
            if kwargs["hostname"] == "cluster.internal":
                # The target refuses keys; the password must then go only to this transport.
                raise paramiko.AuthenticationException("Authentication failed.")
        def get_transport(self): return self
        def get_host_keys(self): return paramiko.HostKeys()
        def set_missing_host_key_policy(self, policy): pass
        def set_keepalive(self, interval): pass
        def auth_none(self, username): raise paramiko.BadAuthenticationType("Bad auth type", ["password"])
        def auth_password(self, username, password, fallback=True):
            passwords.append((self, username, password, fallback))
            return []
        def open_channel(self, kind, destination, source, **kwargs):
            channel = object()
            channels.append((kind, destination, channel))
            return channel
        def close(self): self.closed = True
        def is_active(self): return not self.closed
        def is_authenticated(self): return bool(passwords)
    monkeypatch.setattr(paramiko, "SSHClient", Client)
    cfg = SimpleNamespace(host="cluster.internal", user="target-user", port=2222,
                          connect_timeout=3, ssh_jump_host="gateway")
    native = windows_ssh.WindowsSSH(cfg, "target-secret")
    native._connect()
    assert len(calls) == 2
    assert calls[0]["hostname"] == "gateway.internal" and calls[0]["username"] == "gateway-user"
    assert calls[0]["password"] is None
    assert calls[0]["key_filename"] == [str(folder / "gateway-key")]
    # Paramiko's connect() would answer every interactive prompt with the password;
    # the password is sent explicitly, to the target transport only, without fallback.
    assert calls[1]["password"] is None and calls[1]["passphrase"] == "target-secret"
    assert passwords == [(clients[-1], "target-user", "target-secret", False)]
    assert calls[1]["sock"] is channels[0][2]
    assert channels[0][:2] == ("direct-tcpip", ("cluster.internal", 2222))
    native.close()
    assert all(client.closed for client in clients)
