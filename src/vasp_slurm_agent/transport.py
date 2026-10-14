"""SSH with full output and verified transfers. Requires remote Python 3.

Uses OpenSSH on Unix and Paramiko on Windows. Passwords never enter files or arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


class TransportError(RuntimeError):
    """An SSH operation or verified file transfer could not be completed."""


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


_META_PREFIX = "__DFT_AGENT_FILE_META__"
_BULK_META_PREFIX = "__DFT_AGENT_BULK_META__"
UPLOAD_ALLOWLIST = frozenset({"POSCAR", "INCAR", "KPOINTS", "submit.sh", "dispatch.py", "restart.py",
                              "seed.INCAR", "seed.metadata.json", "warm_start.spec.json"})
DOWNLOAD_ALLOWLIST = frozenset({
    "INCAR", "KPOINTS", "POSCAR", "OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR",
    "EIGENVAL", "DOSCAR", "slurm.out", "slurm.err", "execution.json", "input_hashes.sha256",
    "potcar_hash.sha256", "potcar_titles.txt",
    "seed.INCAR", "seed.metadata.json", "warm_start.spec.json", "seed.vasprun.xml", "seed.OUTCAR",
    "seed.IBZKPT", "seed.stdout", "seed.stderr", "hybrid.stdout", "hybrid.stderr", "warm_start.json", "IBZKPT",
    "PROCAR", "XDATCAR", "REPORT", "CHGCAR", "CHG", "WAVECAR", "LOCPOT", "ELFCAR",
    "PARCHG", "AECCAR0", "AECCAR1", "AECCAR2", "vaspout.h5",
})
_HASH_CODE = """\
def metadata(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return {'sha256': digest.hexdigest(), 'size': size}
"""


def _local_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


# This same Python 3.6-compatible helper is sent over SSH and works in frozen apps.
_CHECKED_ARCHIVE_CODE = r'''def _extract_checked_archive(archive_path, folder, expected):
    """Extract plain, named files only; this function also runs on the host."""
    import hashlib
    import os
    from pathlib import Path, PurePosixPath
    import tarfile

    folder = Path(folder)
    actual = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name
            relative = PurePosixPath(name)
            if (name not in expected or name in actual or not name or name in {".", ".."} or relative.is_absolute()
                    or relative.as_posix() != name or any(part in {".", ".."} for part in relative.parts)
                    or any(c in name for c in "\\\r\n\x00:") or not member.isfile()):
                raise ValueError("Archive contains an unexpected, duplicate, linked, or unsafe member")
            if member.size != expected[name]["size"]:
                raise ValueError("Archive member size mismatch")
            digest = hashlib.sha256()
            size = 0
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Archive member is not a regular readable file")
            destination = folder / name
            if folder.resolve() not in destination.resolve().parents:
                raise ValueError("Archive member escapes its extraction folder")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("xb") as target:
                if hasattr(os, "fchmod"):
                    os.fchmod(target.fileno(), 0o600)
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(block)
                    if size > expected[name]["size"]:
                        raise ValueError("Archive member exceeds its declared size")
                    target.write(block)
                    digest.update(block)
            actual[name] = {"sha256": digest.hexdigest(), "size": size}
            if actual[name] != expected[name]:
                raise ValueError("Archive member SHA-256 or size mismatch")
    if set(actual) != set(expected):
        raise ValueError("Archive is missing required files")
    return actual

'''
_ARCHIVE_HELPERS: dict[str, Any] = {}
exec(_CHECKED_ARCHIVE_CODE, _ARCHIVE_HELPERS)


def _extract_checked_archive(archive_path, folder, expected):
    return _ARCHIVE_HELPERS["_extract_checked_archive"](archive_path, folder, expected)


def _kill_group(process: subprocess.Popen, sig: int) -> None:
    """Signal only the fresh session created for this local SSH/SCP call."""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some macOS hosts deny process-group signals but allow the child.
        # Popen checks its own child status before sending the signal.
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _run_process(argv: list[str], *, timeout: float, **kwargs: Any) -> subprocess.CompletedProcess:
    """Kill the SSH/SCP process group on timeout.

    The separate master stays open until SSHTransport.close().
    """
    kwargs.pop("check", None)
    if kwargs.pop("capture_output", False):
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with subprocess.Popen(argv, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as original:
            _kill_group(process, signal.SIGTERM)
            try:
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    _kill_group(process, signal.SIGKILL)
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired as partial:
                        # A detached master may hold a pipe; keep the wait bounded.
                        stdout, stderr = partial.stdout, partial.stderr
                        for stream in (process.stdout, process.stderr):
                            if stream is not None:
                                stream.close()
                        process.wait(timeout=2)
            finally:
                # EOF does not prove that all descendants exited.
                _kill_group(process, signal.SIGKILL)
            raise subprocess.TimeoutExpired(
                argv, original.timeout, output=stdout, stderr=stderr
            ) from None
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


class SSHTransport:
    """Config requires host, user, port and connect_timeout.

    Close the private SSH master after use. Legacy SCP uses quoted absolute paths.
    """

    def __init__(self, config: Any, password: str | None = None):
        self.config = config
        self._password = (
            os.environ.get("DFT_AGENT_SSH_PASSWORD", "") if password is None else password
        )
        self._closed = False
        self._host = str(config.host)
        self._user = str(config.user)
        if (
            not self._host
            or self._host.startswith("-")
            or any(c.isspace() or c == "\x00" for c in self._host)
            or not self._user
            or any(c in self._user for c in "\r\n\x00")
        ):
            raise ValueError("SSH host and user must be valid nonempty names.")
        if not 1 <= int(config.port) <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        self._windows = platform.system() == "Windows"
        external_control = getattr(config, "ssh_control_path", "")
        if self._windows and external_control:
            raise TransportError("SSH control sockets are not used on Windows. Clear the SSH control socket setting.")
        self._native = None
        if self._windows:
            from .windows_ssh import WindowsSSH
            self._native = WindowsSSH(config, self._password)
        # macOS's normal temporary directory can exceed Unix socket path limits.
        temp_parent = "/tmp" if not self._windows and Path("/tmp").is_dir() else None
        self._temporary = tempfile.TemporaryDirectory(prefix="dft-ssh-", dir=temp_parent)
        self._external_control = bool(external_control)
        self._control_path = str(Path(external_control).expanduser()) if external_control else str(Path(self._temporary.name) / "socket")
        self._askpass_path = Path(self._temporary.name) / "askpass"
        if self._password and not self._windows:
            # Use a shell wrapper so Python paths containing spaces also work.
            code = "import os, sys; sys.stdout.write(os.environ['_DFT_AGENT_ASKPASS_PASSWORD'] + '\\n')"
            arguments = [sys.executable, "--askpass"] if getattr(sys, "frozen", False) else [sys.executable, "-c", code]
            self._askpass_path.write_text(
                "#!/bin/sh\nexec " + shlex.join(arguments) + "\n",
                encoding="utf-8", newline="\n",
            )
            self._askpass_path.chmod(0o700)

    def _options(self) -> list[str]:
        control_path = self._control_path
        if any(character.isspace() or character in "\\\"'" for character in control_path):
            # OpenSSH parses -o values as config text, even in a single argv item.
            control_path = '"' + control_path.replace("\\", "\\\\").replace('"', '\\"') + '"'
        options = [
            "StrictHostKeyChecking=accept-new",
            f"ConnectTimeout={max(1, int(getattr(self.config, 'connect_timeout', 15)))}",
            "ServerAliveInterval=30",
            "ServerAliveCountMax=3",
            "ControlMaster=no" if self._external_control else "ControlMaster=auto",
            "ControlPersist=no" if self._external_control else "ControlPersist=60",
            f"ControlPath={control_path}",
            "LogLevel=ERROR",
            "NumberOfPasswordPrompts=1",
            f"BatchMode={'no' if self._password else 'yes'}",
        ]
        return [arg for option in options for arg in ("-o", option)]

    def _ssh(self) -> list[str]:
        # -l is necessary when the gateway user itself contains an @ character.
        return ["ssh", "-p", str(self.config.port), *self._options(), "-l", self._user]

    def _redact(self, value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = value or ""
        return text.replace(self._password, "[REDACTED]") if self._password else text

    def _execute(self, argv: list[str], timeout: float) -> Result:
        if self._closed:
            raise TransportError("SSH transport has been closed.")
        environment = os.environ.copy()
        # Avoid passing the public password variable on to unrelated children.
        environment.pop("DFT_AGENT_SSH_PASSWORD", None)
        if self._password:
            environment.update(
                SSH_ASKPASS=str(self._askpass_path),
                SSH_ASKPASS_REQUIRE="force",
                DISPLAY=environment.get("DISPLAY") or "dft-agent:0",
                _DFT_AGENT_ASKPASS_PASSWORD=self._password,
            )
        try:
            result = _run_process(
                argv,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            return Result(
                124,
                self._redact(exc.stdout),
                self._redact(exc.stderr) + "\nSSH operation timed out.",
            )
        except OSError as exc:
            raise TransportError(self._redact(str(exc))) from None
        return Result(result.returncode, self._redact(result.stdout), self._redact(result.stderr))

    def run(self, command: str, timeout: float = 60) -> Result:
        """Run a remote shell command, preserving complete stdout and stderr."""
        if self._closed:
            raise TransportError("SSH transport has been closed.")
        if self._native is not None:
            try:
                result = self._native.run("bash -lc " + shlex.quote(command), timeout)
                return Result(result.returncode, self._redact(result.stdout), self._redact(result.stderr))
            except TransportError as exc:
                raise TransportError(self._redact(str(exc))) from None
        return self._execute(
            [*self._ssh(), "--", self._host, "bash -lc " + shlex.quote(command)], timeout
        )

    def _require_success(self, result: Result, operation: str) -> None:
        if result.returncode:
            detail = self._redact(result.stderr or result.stdout).strip()[-2000:]
            raise TransportError(f"{operation} failed (exit {result.returncode}): {detail}")

    @staticmethod
    def _remote_path(path: str) -> str:
        value = str(path)
        if not value.startswith("/") or any(c in value for c in "\x00\r\n"):
            raise TransportError("Remote file paths must be absolute and contain no line breaks.")
        return value

    def _remote_python(self, code: str, timeout: float = 120) -> Result:
        return self.run("python3 -c " + shlex.quote(code), timeout=timeout)

    def _remote_metadata(self, path: str) -> dict[str, Any]:
        code = (
            "import hashlib, json\n"
            + _HASH_CODE
            + f"print({_META_PREFIX!r} + json.dumps(metadata({path!r})))\n"
        )
        result = self._remote_python(code)
        self._require_success(result, "Remote checksum")
        return self._parse_metadata(result.stdout)

    def _parse_metadata(self, stdout: str) -> dict[str, Any]:
        try:
            line = next(line for line in reversed(stdout.splitlines()) if line.startswith(_META_PREFIX))
            value = json.loads(line[len(_META_PREFIX) :])
            if (
                not isinstance(value["size"], int)
                or value["size"] < 0
                or len(value["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in value["sha256"])
            ):
                raise ValueError
            return {"sha256": value["sha256"], "size": value["size"]}
        except (StopIteration, KeyError, TypeError, ValueError):
            raise TransportError("Remote checksum response was missing or invalid.") from None

    def _remote_operand(self, path: str) -> str:
        host = self._host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return host + ":" + shlex.quote(path)

    def _scp(self, source: str, destination: str) -> Result:
        return self._execute(
            [
                "scp", "-O", "-q", "-P", str(self.config.port), *self._options(),
                "-o", f"User={self._user}", "--", source, destination,
            ],
            timeout=600,
        )

    def _upload_file(self, source: Path, destination: str) -> Result:
        if self._native is not None:
            self._native.transfer(str(source), destination, upload=True)
            return Result(0)
        return self._scp(str(source), self._remote_operand(destination))

    def _download_file(self, source: str, destination: Path) -> Result:
        if self._native is not None:
            self._native.transfer(str(destination), source, upload=False)
            return Result(0)
        return self._scp(self._remote_operand(source), str(destination))

    def upload(self, local_path: str | Path, remote_path: str) -> dict[str, Any]:
        """Upload to a temporary remote file, verify, then atomically replace."""
        destination = self._remote_path(remote_path)
        source = Path(local_path).expanduser().absolute()
        temporary = destination + f".part-{uuid.uuid4().hex}"
        transfer_attempted = False
        try:
            expected = _local_metadata(source)
            parent = str(PurePosixPath(destination).parent)
            prepared = self._remote_python(f"import os\nos.makedirs({parent!r}, exist_ok=True)\n")
            self._require_success(prepared, "Upload directory preparation")
            transfer_attempted = True
            copied = self._upload_file(source, temporary)
            self._require_success(copied, "Upload")
            code = (
                "import hashlib, json, os\n"
                + _HASH_CODE
                + f"actual = metadata({temporary!r})\n"
                + f"if actual != {expected!r}:\n    raise RuntimeError('Upload size or SHA-256 mismatch')\n"
                + f"os.replace({temporary!r}, {destination!r})\n"
                + f"print({_META_PREFIX!r} + json.dumps(actual))\n"
            )
            finalized = self._remote_python(code)
            self._require_success(finalized, "Upload verification")
            return self._parse_metadata(finalized.stdout)
        except (OSError, TransportError) as exc:
            # A failed transfer must not replace an existing destination.
            if transfer_attempted:
                try:
                    self._remote_python(
                        f"import os\ntry:\n    os.unlink({temporary!r})\nexcept FileNotFoundError:\n    pass\n",
                        timeout=15,
                    )
                except TransportError:
                    pass
            raise TransportError(self._redact(str(exc))) from None

    def download(self, remote_path: str, local_path: str | Path) -> dict[str, Any]:
        """Download and verify before atomically replacing the local result."""
        source = self._remote_path(remote_path)
        destination = Path(local_path).expanduser().absolute()
        temporary: Path | None = None
        try:
            expected = self._remote_metadata(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
            os.close(descriptor)
            temporary = Path(name)
            copied = self._download_file(source, temporary)
            self._require_success(copied, "Download")
            actual = _local_metadata(temporary)
            if actual != expected:
                raise TransportError("Download verification failed (size or SHA-256 mismatch). Your existing result was kept.")
            os.replace(temporary, destination)
            return actual
        except (OSError, TransportError) as exc:
            raise TransportError(self._redact(str(exc))) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _bulk_names(names: Sequence[str], allowed: frozenset[str]) -> list[str]:
        names = list(names)
        if not isinstance(allowed, frozenset):
            raise TransportError("The explicit transfer allowlist must be an immutable frozenset")
        if any(not isinstance(name, str) or not name or name in {".", ".."} or name not in allowed
               or PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
               or any(part in {".", ".."} for part in PurePosixPath(name).parts)
               or any(c in name for c in "\\\r\n\x00:") for name in names) or len(names) != len(set(names)):
            raise TransportError("Batch filenames must be unique safe relative names from the explicit transfer allowlist")
        return names

    @staticmethod
    def _bulk_response(stdout: str) -> dict[str, Any]:
        try:
            line = next(line for line in reversed(stdout.splitlines()) if line.startswith(_BULK_META_PREFIX))
            value = json.loads(line[len(_BULK_META_PREFIX):])
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (StopIteration, TypeError, ValueError):
            raise TransportError("Remote batch response was missing or invalid") from None

    @staticmethod
    def _manifest(value: Any, names: Sequence[str]) -> dict[str, dict[str, Any]]:
        try:
            if not isinstance(value, dict) or set(value) != set(names):
                raise ValueError
            for entry in value.values():
                if (
                    not isinstance(entry, dict) or set(entry) != {"sha256", "size"}
                    or not isinstance(entry["size"], int) or entry["size"] < 0
                    or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in entry["sha256"])
                ):
                    raise ValueError
            return value
        except (TypeError, ValueError):
            raise TransportError("Remote batch manifest was missing or invalid") from None

    def _remove_remote_archive(self, path: str) -> Result:
        return self._remote_python(
            f"import os\ntry:\n    os.unlink({path!r})\nexcept FileNotFoundError:\n    pass\n", timeout=15
        )

    def upload_many(self, local_paths: Sequence[str | Path], remote_dir: str, *,
                    permitted_names: frozenset[str] | None = None) -> dict[str, dict[str, Any]]:
        """Transfer allowlisted inputs; verify all before per-file atomic replacement.

        Mark staged after all replacements. Excludes POTCAR, CHGCAR and WAVECAR.
        """
        destination = self._remote_path(remote_dir)
        sources = [Path(path).expanduser().absolute() for path in local_paths]
        names = self._bulk_names([path.name for path in sources], UPLOAD_ALLOWLIST if permitted_names is None else permitted_names)
        if not names:
            return {}
        remote_archive = destination.rstrip("/") + "/.vsa-upload-" + uuid.uuid4().hex + ".tar.gz"
        transfer_attempted = False
        try:
            if any(path.is_symlink() or not path.is_file() for path in sources):
                raise TransportError("Batch upload sources must be regular, non-symlink files")
            expected = {path.name: _local_metadata(path) for path in sources}
            with tempfile.TemporaryDirectory(prefix="bulk-", dir=self._temporary.name) as temporary:
                archive_path = Path(temporary) / "inputs.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    for path in sources:
                        info = tarfile.TarInfo(path.name)
                        info.size = expected[path.name]["size"]
                        info.mode = 0o600
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                archive_path.chmod(0o600)
                prepared = self._remote_python(f"import os\nos.makedirs({destination!r}, exist_ok=True)\n")
                self._require_success(prepared, "Batch upload directory preparation")
                transfer_attempted = True
                self._require_success(self._upload_file(archive_path, remote_archive), "Batch upload")
                code = (
                    "import json, os, shutil, tempfile\n"
                    + _CHECKED_ARCHIVE_CODE
                    + f"temporary = tempfile.mkdtemp(prefix='.vsa-unpack-', dir={destination!r})\n"
                    + "try:\n"
                    + f"    actual = _extract_checked_archive({remote_archive!r}, temporary, {expected!r})\n"
                    + "    for name in actual:\n"
                    + f"        os.replace(os.path.join(temporary, name), os.path.join({destination!r}, name))\n"
                    + f"    print({_BULK_META_PREFIX!r} + json.dumps({{'files': actual}}))\n"
                    + "finally:\n    shutil.rmtree(temporary)\n"
                    + f"    os.unlink({remote_archive!r})\n"
                )
                result = self._remote_python(code)
                self._require_success(result, "Batch upload verification")
                manifest = self._manifest(self._bulk_response(result.stdout).get("files"), names)
                if manifest != expected:
                    raise TransportError("Batch upload acknowledged unexpected file checksums")
                return manifest
        except (OSError, ValueError, EOFError, tarfile.TarError, TransportError) as exc:
            if transfer_attempted:
                try:
                    self._remove_remote_archive(remote_archive)
                except TransportError:
                    pass
            raise TransportError(self._redact(str(exc))) from None

    def download_many(
        self, remote_dir: str, local_dir: str | Path, filenames: Sequence[str], *,
        permitted_names: frozenset[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Download allowed output files in one archive and verify before publish."""
        source = self._remote_path(remote_dir)
        names = self._bulk_names(filenames, DOWNLOAD_ALLOWLIST if permitted_names is None else permitted_names)
        if not names:
            return {}
        destination = Path(local_dir).expanduser().absolute()
        remote_archive = source.rstrip("/") + "/.vsa-download-" + uuid.uuid4().hex + ".tar.gz"
        archive_attempted = False
        success = False
        try:
            code = (
                "import hashlib, json, os, tarfile\nfrom pathlib import Path\n"
                + _HASH_CODE
                + f"root = Path({source!r})\nfiles = {{}}\n"
                + f"for name in {names!r}:\n"
                + "    path = root / name\n"
                + "    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:\n"
                + "        raise ValueError('Batch download source is not a regular non-symlink file')\n"
                + "    files[name] = metadata(path)\n"
                + f"descriptor = os.open({remote_archive!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
                + "with os.fdopen(descriptor, 'wb') as target:\n"
                + "    with tarfile.open(fileobj=target, mode='w:gz') as archive:\n"
                + "        for name in files:\n"
                + "            info = tarfile.TarInfo(name)\n"
                + "            info.size = files[name]['size']\n            info.mode = 0o600\n"
                + "            with (root / name).open('rb') as item:\n                archive.addfile(info, item)\n"
                + f"print({_BULK_META_PREFIX!r} + json.dumps({{'files': files, 'archive': metadata({remote_archive!r})}}))\n"
            )
            archive_attempted = True
            prepared = self._remote_python(code, timeout=300)
            self._require_success(prepared, "Batch download preparation")
            response = self._bulk_response(prepared.stdout)
            expected = self._manifest(response.get("files"), names)
            archive_metadata = self._manifest({"archive": response.get("archive")}, ["archive"])["archive"]
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".vsa-download-", dir=destination) as temporary:
                temporary_path = Path(temporary)
                archive_path = temporary_path / "outputs.tar.gz"
                self._require_success(self._download_file(remote_archive, archive_path), "Batch download")
                if _local_metadata(archive_path) != archive_metadata:
                    raise TransportError("Download verification failed (archive size or SHA-256 mismatch). Your existing files were kept.")
                manifest = _extract_checked_archive(archive_path, temporary_path, expected)
                for name in names:
                    target = destination / name
                    if destination.resolve() not in target.resolve().parents or target.is_symlink():
                        raise TransportError("Batch download destination escapes its result folder")
                    target.parent.mkdir(parents=True, exist_ok=True)
                for name in names:
                    os.replace(temporary_path / name, destination / name)
            success = True
            return manifest
        except (OSError, ValueError, EOFError, tarfile.TarError, TransportError) as exc:
            raise TransportError(self._redact(str(exc))) from None
        finally:
            if archive_attempted:
                try:
                    removed = self._remove_remote_archive(remote_archive)
                    if success:
                        self._require_success(removed, "Remote temporary archive cleanup")
                except TransportError:
                    if success:
                        raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._native is not None:
                self._native.close()
            elif not self._external_control and Path(self._control_path).exists():
                self._execute([*self._ssh(), "-O", "exit", "--", self._host], timeout=5)
        except TransportError:
            pass
        finally:
            self._closed = True
            self._password = ""
            self._temporary.cleanup()

    def __enter__(self) -> SSHTransport:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
