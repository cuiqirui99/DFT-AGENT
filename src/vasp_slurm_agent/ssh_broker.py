"""Per-user, authenticated local broker for reusable Windows MFA SSH sessions.

Only the OS keychain stores the IPC capability. Passwords and one-time codes
cross authenticated loopback IPC during sign-in and are discarded after it.
Endpoint files contain no credentials. Workers attach to the live SSH session.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from . import credentials
from .secret_guard import redact
from .transport import (AuthenticationError, ConnectionFailed, HostKeyError, LOGIN_PERSIST_SECONDS,
                        Result, TransportError, VerificationRequired)

_SERVICE = credentials.SERVICE + '-ssh-broker'
_FIELDS = {'host': 'host', 'user': 'user', 'port': 'port', 'jump_host': 'ssh_jump_host',
           'identity_file': 'ssh_identity_file', 'ssh_control_path': 'ssh_control_path'}
_MAX_MESSAGE = 128 * 1024 * 1024
_ERRORS = {kind.__name__: kind for kind in (TransportError, AuthenticationError, ConnectionFailed, HostKeyError, VerificationRequired)}


def _home(home):
    path = Path(home).expanduser().absolute()
    if path.is_symlink():
        raise TransportError('The workspace cannot be a symbolic link.')
    return path.resolve()


def _runtime(home, create=False):
    path = _home(home) / 'runtime'
    if path.is_symlink():
        raise TransportError('The session folder cannot be a symbolic link.')
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    return path


def _capability(home, create=False):
    name = hashlib.sha256(str(_home(home)).encode()).hexdigest()
    backend = credentials._keyring()
    try:
        value = backend.get_password(_SERVICE, name)
        if value is None and create:
            value = secrets.token_hex(32)
            backend.set_password(_SERVICE, name, value)
            if backend.get_password(_SERVICE, name) != value:
                raise credentials.CredentialError('The keychain did not retain the local session credential.')
        if value is None:
            return None
        decoded = bytes.fromhex(value)
        if len(decoded) != 32:
            raise ValueError()
        return decoded
    except credentials.CredentialError:
        raise
    except Exception:
        raise credentials.CredentialError('The system keychain could not open the local SSH session.') from None


def _endpoint(home):
    path = _runtime(home) / 'ssh-broker.json'
    if path.is_symlink():
        raise TransportError('Invalid local session endpoint.')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if (type(value.get('port')) is not int or not 1 <= value['port'] <= 65535
                or not isinstance(value.get('instance'), str)):
            raise ValueError()
        return value
    except (FileNotFoundError, ValueError, TypeError, AttributeError):
        return None


def _read_exact(stream, length):
    result = bytearray()
    while len(result) < length:
        data = stream.recv(min(65536, length - len(result)))
        if not data:
            raise ConnectionError('The local session ended.')
        result.extend(data)
    return bytes(result)


def _receive(stream, maximum=_MAX_MESSAGE):
    size, = struct.unpack('!I', _read_exact(stream, 4))
    if not 0 < size <= maximum:
        raise TransportError('Invalid local session message.')
    value = json.loads(_read_exact(stream, size))
    if not isinstance(value, dict):
        raise TransportError('Invalid local session message.')
    return value


def _send(stream, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(data) > _MAX_MESSAGE:
        raise TransportError('The command returned too much output. Read a smaller part of the file.')
    stream.sendall(struct.pack('!I', len(data)) + data)


def _proof(capability, role, challenge):
    return hmac.new(capability, b'dft-ssh-v1:' + role + bytes.fromhex(challenge), hashlib.sha256).hexdigest()


def _request(home, message, timeout=60):
    endpoint = _endpoint(home)
    if endpoint is None:
        raise VerificationRequired('Sign in with a fresh login code to reconnect.', 'Verification code')
    capability = _capability(home)
    if capability is None:
        raise VerificationRequired('Sign in with a fresh login code to reconnect.', 'Verification code')
    try:
        with socket.create_connection(('127.0.0.1', endpoint['port']), timeout=min(5, timeout)) as stream:
            stream.settimeout(5)
            greeting = _receive(stream, 4096)
            challenge = greeting.get('challenge', '')
            if greeting.get('instance') != endpoint['instance'] or not isinstance(challenge, str) or len(challenge) != 64:
                raise TransportError('The local session identity changed. Sign in again.')
            _send(stream, {'proof': _proof(capability, b'client:', challenge)})
            reply = _receive(stream, 4096)
            if not hmac.compare_digest(str(reply.get('proof', '')), _proof(capability, b'server:', challenge)):
                raise TransportError('The local session could not be authenticated. Sign in again.')
            stream.settimeout(timeout + 10)
            _send(stream, message)
            response = _receive(stream)
    except (OSError, EOFError, ValueError) as error:
        raise VerificationRequired('The saved SSH session is no longer available. Sign in again.', 'Verification code') from None
    if 'error' in response:
        value = response['error']
        kind = _ERRORS.get(value.get('type'), TransportError)
        detail = str(value.get('message') or 'The SSH operation failed.')
        if kind is VerificationRequired:
            raise kind(detail, str(value.get('prompt') or ''))
        if kind is AuthenticationError:
            raise kind(detail, tuple(value.get('methods') or ()))
        raise kind(detail)
    return response.get('result', {})


@contextmanager
def _start_lock(home):
    from .workflow import _acquire, _release
    path = _runtime(home, True) / 'ssh-broker.lock'
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'r+b') as stream:
        if not os.fstat(stream.fileno()).st_size:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        _acquire(stream)
        try:
            yield
        finally:
            stream.seek(0)
            _release(stream)


def start(home):
    """Start the local broker if needed; no SSH connection is opened here."""
    home = _home(home)
    with _start_lock(home):
        try:
            return _request(home, {'operation': 'ping'}, timeout=3)
        except (VerificationRequired, TransportError):
            pass
        _capability(home, create=True)
        from .runtime import cli_command
        environment = dict(os.environ)
        for name in ('DFT_AGENT_SSH_PASSWORD', '_DFT_AGENT_WORKER_PASSWORD', '_DFT_AGENT_ASKPASS_PASSWORD',
                     '_DFT_AGENT_ASKPASS_CODE', '_DFT_AGENT_ASKPASS_PROMPT_FILE', 'SSH_ASKPASS', 'SSH_ASKPASS_REQUIRE'):
            environment.pop(name, None)
        if not getattr(sys, 'frozen', False):
            roots = [str(Path(__file__).resolve().parent.parent)]
            roots.extend(str(Path(value or '.').resolve()) for value in environment.get('PYTHONPATH', '').split(os.pathsep))
            environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(roots))
        options = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS} if sys.platform == 'win32' else {'start_new_session': True}
        process = subprocess.Popen(cli_command('ssh-broker', 'serve', '--home', home), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise TransportError('The local SSH session service could not start. Check that the system keychain is unlocked.')
            try:
                return _request(home, {'operation': 'ping'}, timeout=1)
            except (VerificationRequired, TransportError):
                time.sleep(0.1)
        process.terminate()
        raise TransportError('The local SSH session service did not start in time.')


def _configuration(config):
    if not getattr(config, 'connection_id', '') or not getattr(config, 'connection_home', ''):
        raise TransportError('A reusable SSH session needs a saved connection.')
    return {**{name: getattr(config, name, '') for name in _FIELDS.values()},
            'connect_timeout': getattr(config, 'connect_timeout', 15), 'connection_id': config.connection_id}


def sign_in(config, password='', code=''):
    """Authenticate once. No credential is placed in a file, command or environment."""
    data = _configuration(config)
    home = config.connection_home
    start(home)
    return _request(home, {'operation': 'sign_in', 'config': data, 'password': password, 'code': code},
                    timeout=max(30, int(data['connect_timeout']) + 15))


def status(home, connection_id):
    if _endpoint(home) is None:
        return {'connected': False}
    try:
        return _request(home, {'operation': 'status', 'connection_id': connection_id}, timeout=5)
    except VerificationRequired:
        return {'connected': False}


def disconnect(home, connection_id):
    if _endpoint(home) is None:
        return {'connected': False}
    try:
        return _request(home, {'operation': 'disconnect', 'connection_id': connection_id}, timeout=5)
    except VerificationRequired:
        return {'connected': False}


def stop(home):
    if _endpoint(home) is None:
        return {'stopped': True}
    try:
        return _request(home, {'operation': 'stop'}, timeout=5)
    except VerificationRequired:
        return {'stopped': True}


class BrokerSSH:
    """Transport-shaped client; closing it does not close the authenticated session."""
    def __init__(self, config, password='', code='', master=False):
        self.config = config
        self._password, self._code, self._master = password, code, master
        self._closed = False
        self._signed_in = False

    def _call(self, operation, **values):
        if self._closed:
            raise TransportError('SSH transport has been closed.')
        if self._master and not self._signed_in:
            if self._code:
                sign_in(self.config, self._password, self._code)
            elif not status(self.config.connection_home, self.config.connection_id).get('connected'):
                raise VerificationRequired('Enter a fresh login code to connect.', 'Verification code')
            self._signed_in = True
            self._password = self._code = ''
        timeout = values.get('timeout', 600 if operation == 'transfer' else 60)
        return _request(self.config.connection_home, {'operation': operation, 'config': _configuration(self.config), **values}, timeout=timeout)

    def run(self, command, timeout):
        value = self._call('run', command=command, timeout=timeout)
        return Result(**value)

    def transfer(self, local, remote, *, upload):
        self._call('transfer', local=local, remote=remote, upload=upload)

    def close(self):
        self._closed = True
        self._password = self._code = ''


@dataclass
class _Session:
    transport: object
    fingerprint: str
    used: float = field(default_factory=time.monotonic)


class BrokerServer:
    """One loopback daemon. A lock serializes operations on each SSH connection."""
    def __init__(self, home, capability, idle_timeout=LOGIN_PERSIST_SECONDS):
        if not isinstance(capability, bytes) or len(capability) != 32:
            raise ValueError('Invalid local session capability.')
        self.home, self.capability = _home(home), capability
        self.idle_timeout = idle_timeout
        self.instance = secrets.token_hex(16)
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.sessions, self.locks = {}, {}
        self.generations = {}
        self.guard = threading.Lock()
        self.listener = None
        self.last_activity = time.monotonic()
        self.active_requests = 0

    def _lock(self, identifier):
        with self.guard:
            return self.locks.setdefault(identifier, threading.RLock())

    def _current(self, identifier, config=None):
        from .connections import ConnectionRegistry
        item = ConnectionRegistry(self.home).get(identifier)
        if item['kind'] != 'hpc' or item['revoked'] or item['deleted'] or item['settings']['auth_mode'] != 'code':
            raise VerificationRequired('This SSH connection changed or was revoked. Sign in again.', 'Verification code')
        settings = item['settings']
        if config is not None and any(config.get(field, '') != settings.get(key, '') for key, field in _FIELDS.items()):
            raise VerificationRequired('The SSH destination changed. Sign in with its current settings.', 'Verification code')
        fingerprint = hashlib.sha256(json.dumps({**{key: settings.get(key) for key in _FIELDS},
                                                'auth_mode': settings['auth_mode'], 'secret_version': item['secret_version']}, sort_keys=True).encode()).hexdigest()
        return fingerprint

    def _drop(self, identifier):
        with self.guard:
            item = self.sessions.pop(identifier, None)
        if item:
            item.transport.close()

    def _session(self, identifier, config=None):
        try:
            fingerprint = self._current(identifier, config)
            session = self.sessions.get(identifier)
            client = session.transport._client if session else None
            transport = client.get_transport() if client else None
            if (not session or session.fingerprint != fingerprint or time.monotonic() - session.used > self.idle_timeout
                    or transport is None or not transport.is_active() or not transport.is_authenticated()):
                raise VerificationRequired('The SSH session expired or disconnected. Enter a fresh login code.', 'Verification code')
            return session
        except Exception:
            self._drop(identifier)
            raise

    def _dispatch(self, request):
        operation = request.get('operation')
        if operation == 'ping':
            return {'running': True, 'instance': self.instance}
        if operation == 'stop':
            self.stopped.set()
            return {'stopped': True}
        config = request.get('config') or {}
        identifier = config.get('connection_id') or request.get('connection_id')
        if not isinstance(identifier, str):
            raise TransportError('Choose a saved SSH connection.')
        if operation == 'disconnect':
            with self.guard:
                self.generations[identifier] = self.generations.get(identifier, 0) + 1
            self._drop(identifier)
            return {'connected': False}
        with self._lock(identifier):
            if operation == 'status':
                try:
                    self._session(identifier)
                    return {'connected': True}
                except (ValueError, TransportError):
                    return {'connected': False}
            if operation == 'sign_in':
                fingerprint = self._current(identifier, config)
                with self.guard:
                    generation = self.generations.get(identifier, 0)
                self._drop(identifier)
                password, code = request.get('password', ''), request.get('code', '')
                if (not isinstance(password, str) or not isinstance(code, str) or len(password) > 16384 or len(code) > 64
                        or any(character in password + code for character in '\0\r\n')):
                    raise TransportError('Enter the credential in its secure field.')
                from .windows_ssh import WindowsSSH
                native = WindowsSSH(SimpleNamespace(**config), password, code)
                try:
                    client = native._connect()
                    if not client.get_transport().is_authenticated():
                        raise AuthenticationError('SSH authentication did not complete.')
                    # Credentials are never used to reconnect automatically.
                    native._password = native._code = ''
                    if self._current(identifier, config) != fingerprint:
                        raise VerificationRequired('The connection changed during sign-in. Sign in again.', 'Verification code')
                    with self.guard:
                        if generation != self.generations.get(identifier, 0) or self.stopped.is_set():
                            raise VerificationRequired('Sign-in was cancelled. Enter a fresh login code to reconnect.', 'Verification code')
                        self.sessions[identifier] = _Session(native, fingerprint)
                    return {'connected': True}
                except Exception:
                    native.close()
                    raise
                finally:
                    request.pop('password', None)
                    request.pop('code', None)
            session = self._session(identifier, config)
            session.used = time.monotonic()
            try:
                if operation == 'run':
                    command, timeout = request.get('command'), request.get('timeout', 60)
                    if not isinstance(command, str) or len(command) > 4 * 1024 * 1024 or type(timeout) not in (int, float) or not 0 < timeout <= 86400:
                        raise TransportError('Invalid SSH command request.')
                    value = session.transport.run(command, timeout)
                    return {'returncode': value.returncode, 'stdout': value.stdout, 'stderr': value.stderr}
                if operation == 'transfer':
                    local, remote = request.get('local'), request.get('remote')
                    if not isinstance(local, str) or not isinstance(remote, str) or type(request.get('upload')) is not bool:
                        raise TransportError('Invalid transfer request.')
                    session.transport.transfer(local, remote, upload=request['upload'])
                    return {'transferred': True}
                raise TransportError('Unknown local SSH operation.')
            finally:
                session.used = time.monotonic()

    def _handle(self, stream):
        with stream:
            try:
                stream.settimeout(5)
                challenge = secrets.token_hex(32)
                _send(stream, {'instance': self.instance, 'challenge': challenge})
                response = _receive(stream, 4096)
                if not hmac.compare_digest(str(response.get('proof', '')), _proof(self.capability, b'client:', challenge)):
                    return
                _send(stream, {'proof': _proof(self.capability, b'server:', challenge)})
                stream.settimeout(15)
                request = _receive(stream, 5 * 1024 * 1024)
                with self.guard:
                    self.active_requests += 1
                    self.last_activity = time.monotonic()
                # Retained only for redacting errors from the authentication call.
                protected = [value for value in (request.get('password'), request.get('code')) if isinstance(value, str) and value]
                try:
                    result = self._dispatch(request)
                    response = {'result': result}
                except Exception as error:
                    detail = str(error) if isinstance(error, (TransportError, ValueError, credentials.CredentialError)) else 'The local SSH operation failed.'
                    for value in sorted(protected, key=len, reverse=True):
                        detail = detail.replace(value, '[REDACTED]')
                    response = {'error': {'type': type(error).__name__, 'message': redact(detail)}}
                    if isinstance(error, VerificationRequired):
                        prompt = error.prompt
                        for value in protected:
                            prompt = prompt.replace(value, '[REDACTED]')
                        response['error']['prompt'] = redact(prompt)
                    if isinstance(error, AuthenticationError):
                        response['error']['methods'] = list(error.methods)
                finally:
                    with self.guard:
                        self.active_requests -= 1
                        self.last_activity = time.monotonic()
                protected.clear()
                request.clear()
                _send(stream, response)
            except (OSError, ValueError, TransportError):
                pass

    def _sweep(self):
        with self.guard:
            identifiers = list(self.locks)
        for identifier in identifiers:
            lock = self._lock(identifier)
            if lock.acquire(blocking=False):
                try:
                    if identifier in self.sessions:
                        try:
                            self._session(identifier)
                        except Exception:
                            pass
                finally:
                    lock.release()

    def stop(self):
        self.stopped.set()

    def serve(self):
        folder = _runtime(self.home, True)
        endpoint = folder / 'ssh-broker.json'
        with socket.socket() as listener:
            self.listener = listener
            listener.bind(('127.0.0.1', 0))
            listener.listen(16)
            listener.settimeout(0.5)
            descriptor, temporary = tempfile.mkstemp(prefix='.ssh-broker-', dir=folder)
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                    json.dump({'port': listener.getsockname()[1], 'pid': os.getpid(), 'instance': self.instance}, stream)
                os.replace(temporary, endpoint)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.ready.set()
            try:
                while not self.stopped.is_set():
                    try:
                        connection, _ = listener.accept()
                        threading.Thread(target=self._handle, args=(connection,), daemon=True).start()
                    except socket.timeout:
                        pass
                    self._sweep()
                    with self.guard:
                        if not self.sessions and not self.active_requests and time.monotonic() - self.last_activity > self.idle_timeout:
                            self.stopped.set()
            finally:
                self.stopped.set()
                with self.guard:
                    identifiers = list(self.locks)
                for identifier in identifiers:
                    with self._lock(identifier):
                        self._drop(identifier)
                current = _endpoint(self.home)
                if current and current.get('instance') == self.instance:
                    endpoint.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Keep a private Windows SSH session for DFT Agent.')
    parser.add_argument('action', choices=['serve', 'start', 'stop', 'status'])
    parser.add_argument('--home', required=True)
    parser.add_argument('--connection')
    arguments = parser.parse_args(argv)
    if arguments.action == 'serve':
        capability = _capability(arguments.home)
        if capability is None:
            raise TransportError('Initialize the local session from DFT Agent first.')
        BrokerServer(arguments.home, capability).serve()
        return 0
    value = start(arguments.home) if arguments.action == 'start' else stop(arguments.home) if arguments.action == 'stop' else status(arguments.home, arguments.connection)
    print(json.dumps(value))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
