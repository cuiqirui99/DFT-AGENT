"""Native Windows SSH and SFTP; the shared transport still verifies every file."""

from __future__ import annotations

from contextlib import contextmanager
import os
import getpass
from urllib.parse import urlsplit
from pathlib import Path
import socket
import sys
import tempfile
import time

from .transport import (AuthenticationError, HostKeyError, Result, TransportError, VerificationRequired,
                        classify_prompt)

_WINDOWS = sys.platform == "win32"


def _read_host_keys(paramiko, path: Path):
    """Read our UTF-8 file independently of the Windows ANSI code page."""
    keys = paramiko.HostKeys()
    if not path.exists():
        return keys
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = paramiko.hostkeys.HostKeyEntry.from_line(line, number)
        if entry is not None:
            for hostname in entry.hostnames:
                # Retain the first key for duplicate host/algorithm entries.
                existing = keys.lookup(hostname)
                if existing is None or entry.key.get_name() not in existing:
                    keys.add(hostname, entry.key.get_name(), entry.key)
    return keys


@contextmanager
def _known_hosts_lock(path: Path):
    """Serialize first-use host-key writes across app and monitoring processes."""
    with path.with_name(path.name + ".dft-agent.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + 10
        while True:
            stream.seek(0)
            try:
                if _WINDOWS:
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # The backend is also exercised by local integration tests.
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TransportError("The SSH known_hosts file is busy. Try again.") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if _WINDOWS:
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _accept_new_policy(paramiko, path: Path):
    class AcceptNew(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            path.parent.mkdir(parents=True, exist_ok=True)
            with _known_hosts_lock(path):
                known = _read_host_keys(paramiko, path)
                previous = known.lookup(hostname)
                if previous:
                    expected = previous.get(key.get_name()) or next(iter(previous.values()))
                    if expected != key:
                        raise paramiko.BadHostKeyException(hostname, key, expected)
                else:
                    # Preserve comments and existing entries, with atomic replacement.
                    original = path.read_text(encoding="utf-8") if path.exists() else ""
                    entry = f"{hostname} {key.get_name()} {key.get_base64()}\n"
                    descriptor, name = tempfile.mkstemp(prefix=".known-hosts-", dir=path.parent)
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                            stream.write(original + ("\n" if original and not original.endswith("\n") else "") + entry)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(name, path)
                    finally:
                        Path(name).unlink(missing_ok=True)
                client.get_host_keys().add(hostname, key.get_name(), key)

    return AcceptNew()


class WindowsSSH:
    def __init__(self, config, password, code=""):
        try:
            import paramiko
        except ImportError:
            raise TransportError("The Windows SSH component is missing. Reinstall DFT Agent.") from None
        self._paramiko = paramiko
        self.config = config
        self._password = password
        self._code = code or ""
        self._client = None
        self._chain = []
        self._proxy = None
        self._closed = False

    def _close_clients(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        for client in reversed(self._chain):
            client.close()
        self._chain = []
        if self._proxy is not None:
            self._proxy.close()
            self._proxy = None

    def _new_client(self):
        client = self._paramiko.SSHClient()
        known_hosts = Path.home() / ".ssh" / "known_hosts"
        if known_hosts.exists():
            known = _read_host_keys(self._paramiko, known_hosts)
            for hostname, entries in known.items():
                for kind, key in entries.items():
                    client.get_host_keys().add(hostname, kind, key)
        client.set_missing_host_key_policy(_accept_new_policy(self._paramiko, known_hosts))
        return client

    def _connect(self):
        if self._closed:
            raise TransportError("SSH transport has been closed.")
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active() and transport.is_authenticated():
                return self._client
        self._close_clients()
        client = None
        try:
            path = Path.home() / ".ssh" / "config"
            settings = self._paramiko.SSHConfig.from_path(str(path)) if path.is_file() else self._paramiko.SSHConfig()
            target = settings.lookup(str(self.config.host))
            hostname = target.get("hostname", str(self.config.host)).removeprefix("[").removesuffix("]")
            port = int(self.config.port)
            timeout = max(1, int(getattr(self.config, "connect_timeout", 15)))
            common = dict(timeout=timeout, banner_timeout=timeout, auth_timeout=timeout, channel_timeout=timeout)
            jump = getattr(self.config, "ssh_jump_host", "") or target.get("proxyjump", "")
            sock = None
            if jump and jump.lower() != "none":
                hops = jump.split(",")
                if len(hops) > 8:
                    raise ValueError("Use at most eight SSH jump hosts.")
                for entry in hops:
                    address = urlsplit("ssh://" + entry)
                    if not address.hostname or address.password or address.path or address.query or address.fragment:
                        raise ValueError("Invalid SSH jump host.")
                    options = settings.lookup(address.hostname)
                    if options.get("proxyjump", "none").lower() != "none":
                        raise ValueError("List nested jump hosts explicitly in the connection.")
                    hop_host = options.get("hostname", address.hostname)
                    hop_port = address.port or int(options.get("port", 22))
                    if self._chain:
                        sock = self._chain[-1].get_transport().open_channel("direct-tcpip", (hop_host, hop_port), ("127.0.0.1", 0), timeout=timeout)
                    hop = self._new_client()
                    self._chain.append(hop)
                    # A target password must never be sent to a different host.
                    hop.connect(hostname=hop_host, port=hop_port,
                                username=address.username or options.get("user", getpass.getuser()),
                                key_filename=options.get("identityfile"), password=None,
                                allow_agent=True, look_for_keys=True, sock=sock, **common)
                    hop.get_transport().set_keepalive(30)
                sock = self._chain[-1].get_transport().open_channel("direct-tcpip", (hostname, port), ("127.0.0.1", 0), timeout=timeout)
            elif target.get("proxycommand"):
                # Honor the user's existing SSH config, as OpenSSH does.
                self._proxy = self._paramiko.ProxyCommand(target["proxycommand"])
                sock = self._proxy
            identity = getattr(self.config, "ssh_identity_file", "") or target.get("identityfile")
            client = self._new_client()
            self._authenticate(client, hostname, port, identity, sock, common)
            client.get_transport().set_keepalive(30)
        except self._paramiko.BadHostKeyException:
            if client is not None: client.close()
            self._close_clients()
            raise HostKeyError("SSH host key has changed. Verify it with your cluster administrator before updating known_hosts.") from None
        except TransportError:
            if client is not None: client.close()
            self._close_clients()
            raise
        except self._paramiko.AuthenticationException:
            if client is not None: client.close()
            self._close_clients()
            raise AuthenticationError("SSH authentication failed. Check your username, password or SSH key.") from None
        except Exception as exc:
            if client is not None: client.close()
            self._close_clients()
            raise TransportError(f"Cannot connect with SSH: {exc}") from None
        self._client = client
        return client

    def _allowed_methods(self, transport, username):
        try:
            transport.auth_none(username)
            return set()
        except self._paramiko.BadAuthenticationType as error:
            return set(error.allowed_types or [])
        except self._paramiko.SSHException:
            return set()

    def _authenticate(self, client, hostname, port, identity, sock, common):
        """Keys and agent first; then a password and, when the server asks, a login code.

        Paramiko's own password fallback answers every keyboard-interactive prompt
        with the password, which fails on clusters that also ask for a code.
        """
        username = str(self.config.user)
        pinned = bool(getattr(self.config, "ssh_identity_file", ""))
        answer_prompts = self._answer_prompts

        class InteractiveTransport(self._paramiko.Transport):
            def auth_interactive_dumb(self, username, handler=None, submethods=""):
                # Paramiko calls this after a key succeeds as the first factor.
                # Desktop authentication must never read a terminal or print prompts.
                return self.auth_interactive(username, answer_prompts, submethods)

        try:
            # The password also unlocks an encrypted default key, as OpenSSH asks for it.
            client.connect(hostname=hostname, port=port, username=username, key_filename=identity or None,
                           password=None, passphrase=self._password or None,
                           allow_agent=not pinned, look_for_keys=not pinned, sock=sock,
                           transport_factory=InteractiveTransport, **common)
            transport = client.get_transport()
            if transport is not None and transport.is_authenticated():
                return
        except (self._paramiko.AuthenticationException, self._paramiko.SSHException) as error:
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise
            if isinstance(error, self._paramiko.SSHException) and not isinstance(error, self._paramiko.AuthenticationException) \
                    and "authentication methods" not in str(error).lower():
                raise
        allowed = self._allowed_methods(transport, username)
        if not (self._password or self._code) and "keyboard-interactive" not in allowed:
            raise AuthenticationError("SSH authentication failed. The cluster rejected the offered keys.", tuple(sorted(allowed)))
        remaining = None
        if "password" in allowed and self._password:
            try:
                remaining = transport.auth_password(username, self._password, fallback=False)
            except self._paramiko.AuthenticationException:
                remaining = ["keyboard-interactive"] if "keyboard-interactive" in allowed else None
                if remaining is None:
                    raise AuthenticationError("SSH authentication failed. The cluster rejected the password.", tuple(sorted(allowed))) from None
            if not remaining and transport.is_authenticated():
                return
        if "keyboard-interactive" in allowed or (remaining and "keyboard-interactive" in remaining):
            remaining = transport.auth_interactive(username, self._answer_prompts)
            if not remaining and transport.is_authenticated():
                return
            raise AuthenticationError("SSH authentication failed. The cluster requires another sign-in method.", tuple(remaining))
        raise AuthenticationError("SSH authentication failed. The cluster does not accept password sign-in.", tuple(sorted(allowed)))

    def _answer_prompts(self, title, instructions, prompts):
        answers = []
        for prompt, _echo in prompts:
            kind = classify_prompt(prompt)
            answer = {"password": self._password, "code": self._code}.get(kind, "")
            if not answer:
                if kind == "password":
                    raise AuthenticationError("The cluster asks for a password.", ("password",))
                raise VerificationRequired("The cluster asks for a login code before it accepts the connection."
                                           if kind == "code" else "The cluster asked a question the app could not answer.",
                                           str(prompt)[:500])
            answers.append(answer)
        return answers

    def run(self, command: str, timeout: float) -> Result:
        channel = None
        stdout, stderr = bytearray(), bytearray()
        try:
            client = self._connect()
            deadline = time.monotonic() + timeout
            channel = client.get_transport().open_session(timeout=timeout)
            channel.settimeout(timeout)
            channel.exec_command(command)
            channel.shutdown_write()
            while True:
                # Drain both streams before requesting status: either can fill the SSH window.
                if channel.recv_ready():
                    stdout.extend(channel.recv(65536))
                if channel.recv_stderr_ready():
                    stderr.extend(channel.recv_stderr(65536))
                if (channel.exit_status_ready() and (channel.closed or channel.eof_received)
                        and not channel.recv_ready() and not channel.recv_stderr_ready()):
                    status = channel.recv_exit_status()
                    return Result(status if status >= 0 else 255,
                                  stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"))
                if time.monotonic() >= deadline:
                    raise socket.timeout()
                time.sleep(0.01)
        except (socket.timeout, TimeoutError):
            return Result(124, stdout.decode("utf-8", errors="replace"),
                          stderr.decode("utf-8", errors="replace") + "\nSSH operation timed out.")
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(f"SSH command failed: {exc}") from None
        finally:
            if channel is not None:
                channel.close()

    def transfer(self, local: str, remote: str, *, upload: bool):
        sftp = None
        try:
            sftp = self._connect().open_sftp()
            deadline = time.monotonic() + 600
            sftp.get_channel().settimeout(600)

            def progress(_transferred, _total):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise socket.timeout("SFTP transfer timed out")
                sftp.get_channel().settimeout(remaining)

            if upload:
                sftp.put(local, remote, callback=progress)
            else:
                # Avoid a prefetch thread outliving its channel on timeout/close.
                sftp.get(remote, local, callback=progress, prefetch=False)
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(f"SFTP transfer failed: {exc}") from None
        finally:
            if sftp is not None:
                sftp.close()

    def close(self):
        self._closed = True
        self._password = ""
        self._code = ""
        self._close_clients()
