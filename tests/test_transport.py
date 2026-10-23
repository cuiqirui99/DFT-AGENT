from __future__ import annotations

import ast
import hashlib
import io
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vasp_slurm_agent.transport import SSHTransport, TransportError
from vasp_slurm_agent import transport as transport_module


@pytest.fixture
def config():
    return SimpleNamespace(host="cluster.example.invalid", user="researcher@gateway", port=2222, connect_timeout=12)


@pytest.fixture
def local_wire(monkeypatch):
    """Execute the remote shell locally and model SCP without any network I/O."""
    original_run = subprocess.run
    state = SimpleNamespace(calls=[], corrupt=False, fail_scp=False)

    def remote_path(value):
        if value.startswith("cluster.example.invalid:"):
            # Legacy SCP passes the suffix as a shell operand on the remote host.
            parts = shlex.split(value.split(":", 1)[1])
            assert len(parts) == 1
            return parts[0]
        return value

    def fake_run(argv, **kwargs):
        state.calls.append((argv, kwargs))
        if argv[0] == "ssh":
            assert argv[argv.index("-l") + 1] == "researcher@gateway"
            assert argv[argv.index("--") + 1] == "cluster.example.invalid"
            assert "StrictHostKeyChecking=accept-new" in argv
            return original_run(shlex.split(argv[-1]), **kwargs)
        assert argv[0] == "scp"
        assert "-O" in argv
        assert "User=researcher@gateway" in argv
        assert "StrictHostKeyChecking=accept-new" in argv
        source, destination = map(remote_path, argv[-2:])
        if state.fail_scp:
            Path(destination).write_bytes(b"incomplete")
            return subprocess.CompletedProcess(argv, 1, "", "Connection closed")
        shutil.copyfile(source, destination)
        if state.corrupt:
            data = bytearray(Path(destination).read_bytes())
            data[len(data) // 2] ^= 0x01
            Path(destination).write_bytes(data)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(transport_module, "_run_process", fake_run)
    return state


def test_large_binary_roundtrip_exact_with_shell_characters(config, local_wire, tmp_path):
    data = bytes(range(256)) * 24577 + b"\x00\r\n\xffexact ending"
    source = tmp_path / "local input ' $.bin"
    source.write_bytes(data)
    injection_marker = tmp_path / "must-not-exist"
    remote = tmp_path / "remote" / f"quotes ' $(touch {injection_marker.name}) ; output.bin"
    target = tmp_path / "download" / "result.bin"
    expected = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    with SSHTransport(config, password="") as transport:
        assert transport.upload(source, str(remote)) == expected
        assert transport.download(str(remote), target) == expected
    assert source.read_bytes() == remote.read_bytes() == target.read_bytes()
    assert not injection_marker.exists()
    assert not Path(injection_marker.name).exists()
    assert not list(remote.parent.glob("*.part-*"))
    assert not list(target.parent.glob("*.part"))


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_checksum_mismatch_preserves_existing_file(config, local_wire, tmp_path, direction):
    source = tmp_path / "source"
    destination = tmp_path / "existing"
    source.write_bytes(b"new result" * 10000)
    destination.write_bytes(b"old verified result")
    local_wire.corrupt = True
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="mismatch"):
            if direction == "upload":
                transport.upload(source, str(destination))
            else:
                transport.download(str(source), destination)
    assert destination.read_bytes() == b"old verified result"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing", "source"]


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_failed_scp_preserves_existing_file(config, local_wire, tmp_path, direction):
    source = tmp_path / "source"
    destination = tmp_path / "existing"
    source.write_bytes(b"new result")
    destination.write_bytes(b"old verified result")
    local_wire.fail_scp = True
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="Connection closed"):
            if direction == "upload":
                transport.upload(source, str(destination))
            else:
                transport.download(str(source), destination)
    assert destination.read_bytes() == b"old verified result"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing", "source"]


def test_command_output_is_complete_and_status_preserved(config, local_wire):
    script = "import sys; sys.stdout.write('x' * 500000); sys.stderr.write('e' * 8000); sys.exit(7)"
    with SSHTransport(config, password="") as transport:
        result = transport.run("python3 -c " + shlex.quote(script))
    assert result.returncode == 7
    assert result.stdout == "x" * 500000
    assert result.stderr == "e" * 8000


@pytest.mark.parametrize("size, glyph", [(n, "a") for n in (100, 7600, 8000, 8040, 8600, 20000)] + [(2670, "算")])
def test_commands_near_8_kb_are_padded_past_the_macos_mux_failure(config, local_wire, size, glyph):
    # OpenSSH's mux client on macOS failed to pass its file descriptors for requests of 8084-8124
    # bytes on Arrhenius ("mm_send_fd: sendmsg(2): Message too long"), and only for those.
    script = "x = '" + glyph * size + "'; print(len(x))"
    with SSHTransport(config, password="") as transport:
        result = transport.run("python3 -c " + shlex.quote(script))
    assert result.returncode == 0 and result.stdout.strip() == str(size)
    sent = local_wire.calls[-1][0][-1]
    low, high = transport_module._MUX_WINDOW
    assert not low <= len(sent.encode("utf-8")) < high and sent.rstrip() == "bash -lc " + shlex.quote("python3 -c " + shlex.quote(script))


def test_timeout_preserves_complete_partial_output(config, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"x" * 25000, stderr=b"partial error")

    monkeypatch.setattr(transport_module, "_run_process", timeout)
    with SSHTransport(config, password="") as transport:
        result = transport.run("a command", timeout=0.1)
    assert result.returncode == 124
    assert result.stdout == "x" * 25000
    assert "partial error" in result.stderr
    assert "timed out" in result.stderr


def test_password_not_in_arguments_or_helper_and_diagnostics_redacted(config, monkeypatch, tmp_path):
    password = "temporary-test-secret-$()"
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", password)
    seen = []

    def fail(argv, **kwargs):
        seen.append((argv, kwargs))
        assert password not in " ".join(argv)
        environment = kwargs["env"]
        assert "DFT_AGENT_SSH_PASSWORD" not in environment
        assert environment["_DFT_AGENT_ASKPASS_PASSWORD"] == password
        assert environment["SSH_ASKPASS_REQUIRE"] == "force"
        helper = Path(environment["SSH_ASKPASS"])
        assert password not in helper.read_text()
        assert helper.stat().st_mode & 0o777 == 0o700
        return subprocess.CompletedProcess(argv, 255, "", "Permission denied " + password)

    monkeypatch.setattr(transport_module, "_run_process", fail)
    source = tmp_path / "source"
    source.write_text("input")
    with SSHTransport(config, password=password) as transport:
        result = transport.run("true")
        assert password not in repr(result)
        assert "[REDACTED]" in result.stderr
        with pytest.raises(TransportError) as error:
            transport.upload(source, "/remote/input")
        assert password not in str(error.value)
        helper = transport._askpass_path
    assert seen
    assert not helper.exists()


def test_hostkey_mismatch_is_not_bypassed(config, monkeypatch):
    calls = []

    def changed_key(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 255, "", "REMOTE HOST IDENTIFICATION HAS CHANGED!")

    monkeypatch.setattr(transport_module, "_run_process", changed_key)
    with SSHTransport(config, password="") as transport:
        result = transport.run("true")
    assert result.returncode == 255
    assert len(calls) == 1
    assert "StrictHostKeyChecking=accept-new" in calls[0]
    assert not any("UserKnownHostsFile" in arg for arg in calls[0])


def test_key_auth_uses_batch_mode_and_closed_transport_fails(config, local_wire):
    transport = SSHTransport(config, password="")
    assert not transport._external_control
    assert transport._control_path == str(Path(transport._temporary.name) / "socket")
    transport.run("true")
    assert "BatchMode=yes" in local_wire.calls[0][0]
    assert f"ControlPath={transport._control_path}" in local_wire.calls[0][0]
    assert "ControlPersist=60" in local_wire.calls[0][0]
    transport.close()
    transport.close()
    with pytest.raises(TransportError, match="closed"):
        transport.run("true")


def test_external_control_socket_is_shared_by_ssh_and_scp_and_never_closed(config, local_wire, tmp_path):
    socket = tmp_path / "user master.sock"
    socket.write_text("Owned by the user's SSH session")
    config.ssh_control_path = str(socket)
    source = tmp_path / "source"
    source.write_bytes(b"checked input")
    remote = tmp_path / "remote" / "input"
    transport = SSHTransport(config, password="test-password")
    helper = transport._askpass_path
    assert helper.exists()
    transport.run("true")
    transport.upload(source, str(remote))
    transport.close()
    transport.close()
    assert socket.read_text() == "Owned by the user's SSH session"
    assert not helper.exists()
    assert {argv[0] for argv, _ in local_wire.calls} == {"ssh", "scp"}
    for argv, _ in local_wire.calls:
        assert f'ControlPath="{socket}"' in argv
        assert "ControlMaster=no" in argv and "ControlPersist=no" in argv
        assert "exit" not in argv


@pytest.mark.skipif(shutil.which("ssh") is None, reason="OpenSSH is unavailable")
@pytest.mark.parametrize("name", ["user master.sock", 'user"master.sock', r"user\master.sock", "master.sock"])
def test_control_socket_survives_openssh_config_parsing(config, tmp_path, name):
    config.ssh_control_path = str(tmp_path / name)
    with SSHTransport(config, password="") as transport:
        command = transport._ssh()
        result = subprocess.run(
            [command[0], "-G", "-F", os.devnull, *command[1:], "--", config.host],
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 0, result.stderr
    parsed = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    assert parsed["controlpath"] == config.ssh_control_path


@pytest.mark.parametrize("path", ["relative/file", "/file\nnext", "/bad\x00name"])
def test_unsafe_remote_path_rejected_before_network(config, monkeypatch, tmp_path, path):
    def unexpected(*args, **kwargs):
        pytest.fail("No network command should have run")

    monkeypatch.setattr(transport_module, "_run_process", unexpected)
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="absolute"):
            transport.download(path, tmp_path / "result")


def test_missing_upload_source_fails_without_network(config, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("No network command should have run")

    monkeypatch.setattr(transport_module, "_run_process", unexpected)
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="No such file"):
            transport.upload(tmp_path / "missing", "/remote/result")


@pytest.mark.parametrize("child_keeps_pipes", [True, False])
def test_timeout_kills_descendants_even_when_they_ignore_term(config, tmp_path, child_keeps_pipes):
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('child ready', flush=True); time.sleep(120)"
    )
    pipes = "" if child_keeps_pipes else ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]{pipes}); "
        f"Path({str(child_pid)!r}).write_text(str(child.pid)); "
        "print('parent ready', flush=True); time.sleep(120)"
    )
    started = time.monotonic()
    with SSHTransport(config, password="") as transport:
        result = transport._execute([sys.executable, "-c", parent_code], timeout=0.4)
    elapsed = time.monotonic() - started
    assert result.returncode == 124
    assert "parent ready" in result.stdout
    assert elapsed < 6
    pid = int(child_pid.read_text())
    # A killed child may briefly remain as a reaped-by-init zombie on Linux.
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert not status.stdout.strip() or status.stdout.strip().startswith("Z")
    assert os.getpgrp() != pid


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_falls_back_to_own_child_when_group_signal_is_denied(config, monkeypatch, ignore_term):
    signals = []
    children = []
    real_popen = subprocess.Popen

    def own_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def deny_group(pid, sig):
        signals.append((pid, sig))
        raise PermissionError(1, "Operation not permitted")

    code = (
        "import os,signal,sys,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print('partial output', flush=True); "
        "print('partial error', file=sys.stderr, flush=True); time.sleep(10)"
    )
    # A separate session models a pre-existing, independently owned SSH master.
    master = real_popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    monkeypatch.setattr(transport_module.subprocess, "Popen", own_child)
    monkeypatch.setattr(transport_module.os, "killpg", deny_group)
    try:
        started = time.monotonic()
        with SSHTransport(config, password="") as transport:
            result = transport._execute([sys.executable, "-c", code], timeout=0.5)
        assert result.returncode == 124
        assert "timed out" in result.stderr
        assert "partial output" in result.stdout and "partial error" in result.stderr
        assert "Operation not permitted" not in result.stderr
        assert time.monotonic() - started < 6
        assert len(children) == 1 and children[0].poll() is not None
        assert {pid for pid, _ in signals} == {children[0].pid}
        if ignore_term:
            assert children[0].returncode == -signal.SIGKILL
        assert master.poll() is None
    finally:
        for child in [*children, master]:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)


def test_batch_roundtrip_uses_one_scp_and_verifies_every_file(config, local_wire, tmp_path):
    source = tmp_path / "local inputs"
    source.mkdir()
    data = {"INCAR": b"input settings\n", "POSCAR": bytes(range(256)) * 8193, "KPOINTS": b"mesh\n"}
    paths = []
    for name, content in data.items():
        path = source / name
        path.write_bytes(content)
        paths.append(path)
    remote = tmp_path / "remote quotes ' $ ; directory"
    output = tmp_path / "local output"
    expected = {name: {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
                for name, content in data.items()}
    with SSHTransport(config, password="") as transport:
        assert transport.upload_many(paths, str(remote)) == expected
        assert [call[0][0] for call in local_wire.calls] == ["ssh", "scp", "ssh"]
        for argv, _ in local_wire.calls:
            if argv[0] == "ssh":
                remote_command = shlex.split(argv[-1])[-1]
                ast.parse(shlex.split(remote_command)[-1], feature_version=(3, 6))
        local_wire.calls.clear()
        assert transport.download_many(str(remote), output, list(data)) == expected
        assert [call[0][0] for call in local_wire.calls] == ["ssh", "scp", "ssh"]
        for argv, _ in local_wire.calls:
            if argv[0] == "ssh":
                remote_command = shlex.split(argv[-1])[-1]
                ast.parse(shlex.split(remote_command)[-1], feature_version=(3, 6))
    for name, content in data.items():
        assert (remote / name).read_bytes() == (output / name).read_bytes() == content
    assert sorted(path.name for path in remote.iterdir()) == sorted(data)
    assert sorted(path.name for path in output.iterdir()) == sorted(data)


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_corrupt_batch_preserves_all_existing_destinations(config, local_wire, tmp_path, direction):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    names = ["INCAR", "POSCAR", "KPOINTS"]
    for name in names:
        (source / name).write_bytes(bytes(range(256)) * 10)
        (destination / name).write_bytes(b"previous verified file")
    local_wire.corrupt = True
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError):
            if direction == "upload":
                transport.upload_many([source / name for name in names], str(destination))
            else:
                transport.download_many(str(source), destination, names)
    for name in names:
        assert (destination / name).read_bytes() == b"previous verified file"
    assert sorted(path.name for path in source.iterdir()) == sorted(names)
    assert sorted(path.name for path in destination.iterdir()) == sorted(names)


@pytest.mark.parametrize("name", ["POTCAR", "../OUTCAR", "/INCAR", "arbitrary.txt"])
def test_batch_download_allowlist_rejected_without_network(config, local_wire, tmp_path, name):
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="allowlist"):
            transport.download_many("/remote", tmp_path, [name])
    assert not local_wire.calls


@pytest.mark.parametrize("name", ["CHGCAR", "WAVECAR", "LOCPOT", "PROCAR"])
def test_explicit_solver_output_download(config, local_wire, tmp_path, name):
    from vasp_slurm_agent.workflow import FILES
    remote = tmp_path / "remote"
    remote.mkdir()
    content = b"solver output\x00\xff"
    (remote / name).write_bytes(content)
    with SSHTransport(config, password="") as transport:
        result = transport.download_many(str(remote), tmp_path / "download", [name])
    assert (tmp_path / "download" / name).read_bytes() == content
    assert result[name]["size"] == len(content)
    if name != "PROCAR":
        assert name not in FILES


def test_batch_upload_allowlist_and_duplicates_rejected_without_network(config, local_wire, tmp_path):
    with SSHTransport(config, password="") as transport:
        for names in (["POTCAR"], ["OUTCAR"], ["INCAR", "INCAR"]):
            with pytest.raises(TransportError, match="allowlist"):
                transport.upload_many([tmp_path / name for name in names], "/remote")
    assert not local_wire.calls


def test_empty_batches_do_not_connect(config, local_wire, tmp_path):
    with SSHTransport(config, password="") as transport:
        assert transport.upload_many([], "/remote") == {}
        assert transport.download_many("/remote", tmp_path, []) == {}
    assert not local_wire.calls


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate"])
def test_archive_extraction_rejects_path_escape_links_and_duplicates(tmp_path, kind):
    archive_path = tmp_path / "unsafe.tar.gz"
    destination = tmp_path / "unpack"
    destination.mkdir()
    outside = tmp_path / "escaped"
    content = b"file"
    expected = {"POSCAR": {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}}
    with tarfile.open(archive_path, "w:gz") as archive:
        name = "../escaped" if kind == "traversal" else str(outside) if kind == "absolute" else "POSCAR"
        info = tarfile.TarInfo(name)
        info.size = len(content)
        if kind in {"symlink", "hardlink"}:
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "../escaped"
            archive.addfile(info)
        else:
            archive.addfile(info, io.BytesIO(content))
        if kind == "duplicate":
            archive.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="unsafe member"):
        transport_module._extract_checked_archive(archive_path, destination, expected)
    assert not outside.exists()


def test_batch_refuses_symlink_sources(config, local_wire, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = tmp_path / "real-input"
    original.write_bytes(b"input")
    (source / "POSCAR").symlink_to(original)
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="non-symlink"):
            transport.upload_many([source / "POSCAR"], str(tmp_path / "remote"))
        assert not local_wire.calls
        with pytest.raises(TransportError, match="non-symlink"):
            transport.download_many(str(source), tmp_path / "output", ["POSCAR"])
    assert not list(source.glob(".vsa-*.tar.gz"))


def test_qe_bulk_roundtrip_nested_outputs_and_bounded_command_count(config, local_wire, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = ["pw.in", "structure.cif", "qe-input.json", "runtime.json", "pseudo_manifest.json", "qe_runtime.py", "submit.sh", "dispatch.py"]
    for index, name in enumerate(names):
        (inputs / name).write_text(f"approved input {index}\n")
    remote = tmp_path / "remote"
    with SSHTransport(config, password="") as transport:
        uploaded = transport.upload_many([inputs / name for name in names], str(remote), permitted_names=frozenset(names))
        assert set(uploaded) == set(names)
        assert [call[0][0] for call in local_wire.calls] == ["ssh", "scp", "ssh"]
        for index in range(1, 11):
            path = remote / "elph_dir" / f"a2Fq2r.51.{index}"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"coupling {index}\n")
        names += [f"elph_dir/a2Fq2r.51.{index}" for index in range(1, 11)]
        local_wire.calls.clear()
        result = transport.download_many(str(remote), tmp_path / "result", names, permitted_names=frozenset(names))
        assert [call[0][0] for call in local_wire.calls] == ["ssh", "scp", "ssh"]
        for name in names:
            assert (tmp_path / "result" / name).read_bytes() == (remote / name).read_bytes()
            assert result[name] == transport_module._local_metadata(remote / name)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "elph_dir/../../outside", "elph_dir/./data", "elph_dir//data", "C:/file", "elph_dir\\data", "data\nfile", "."])
def test_custom_batch_allowlist_cannot_authorize_unsafe_paths(config, local_wire, tmp_path, name):
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="allowlist"):
            transport.download_many("/remote", tmp_path, [name], permitted_names=frozenset({name}))
    assert not local_wire.calls


def test_custom_batch_requires_immutable_allowlist_and_retains_default_restrictions(config, local_wire, tmp_path):
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="immutable"):
            transport.download_many("/remote", tmp_path, ["pw.out"], permitted_names={"pw.out"})
        with pytest.raises(TransportError, match="allowlist"):
            transport.download_many("/remote", tmp_path, ["pw.out"])
    assert not local_wire.calls


def test_nested_batch_refuses_symlink_parent_escape(config, local_wire, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_bytes(b"private")
    (remote / "elph_dir").symlink_to(outside)
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="non-symlink"):
            transport.download_many(str(remote), tmp_path / "output", ["elph_dir/data"], permitted_names=frozenset({"elph_dir/data"}))
    assert not (tmp_path / "output" / "elph_dir" / "data").exists()


def test_nested_batch_refuses_local_destination_escape(config, local_wire, tmp_path):
    remote = tmp_path / "remote" / "elph_dir"
    remote.mkdir(parents=True)
    (remote / "data").write_bytes(b"result")
    destination = tmp_path / "destination"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_bytes(b"keep")
    (destination / "elph_dir").symlink_to(outside)
    with SSHTransport(config, password="") as transport:
        with pytest.raises(TransportError, match="escapes"):
            transport.download_many(str(remote.parent), destination, ["elph_dir/data"], permitted_names=frozenset({"elph_dir/data"}))
    assert (outside / "data").read_bytes() == b"keep"


def test_ambient_password_is_not_used_and_contexts_are_thread_local(config, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from vasp_slurm_agent.transport import password_context, current_password
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "other-account")
    with SSHTransport(config) as remote:
        assert remote._password == ""
    barrier = Barrier(2)
    def worker(secret):
        with password_context(secret):
            barrier.wait(timeout=5)
            with SSHTransport(config) as remote:
                return remote._password
    with ThreadPoolExecutor(2) as pool:
        first, second = pool.submit(worker, "first"), pool.submit(worker, "second")
        assert first.result() == "first" and second.result() == "second"
    assert current_password() == ""
    with password_context("scoped"):
        with SSHTransport(config, password="explicit") as remote:
            assert remote._password == "explicit"
    assert current_password() == ""


def test_native_options_include_explicit_identity_and_jump_for_ssh_and_scp(config, monkeypatch, tmp_path):
    config.ssh_jump_host = "gateway-user@gateway:2222"
    config.ssh_identity_file = str(tmp_path / "folder with spaces" / "private_key")
    seen = []
    def capture(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(transport_module, "_run_process", capture)
    with SSHTransport(config) as remote:
        remote.run("true")
        remote._scp("source", "host:destination")
    assert len(seen) == 2
    for argv in seen:
        assert "ProxyJump=gateway-user@gateway:2222" in argv
        assert 'IdentityFile="' + config.ssh_identity_file + '"' in argv
        assert "IdentitiesOnly=yes" in argv


def test_connection_revocation_blocks_open_transport_and_file_transfer(tmp_path, monkeypatch):
    from vasp_slurm_agent.connections import ConnectionRegistry
    registry = ConnectionRegistry(tmp_path / "workspace")
    item = registry.create("hpc", "Loopback fixture", {"host": "localhost", "user": "scientist", "auth_mode": "agent"})
    config = registry._transport_config(item)
    calls = []
    monkeypatch.setattr(transport_module, "_run_process", lambda argv, **kw: (calls.append(argv) or subprocess.CompletedProcess(argv, 0, "ok", "")))
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "unrelated-password")
    with SSHTransport(config, password="explicit-cannot-override-profile") as remote:
        assert remote._password == ""
        assert remote.run("true").returncode == 0
        registry.revoke(item["id"])
        with pytest.raises(TransportError, match="revoked"):
            remote.run("true")
        with pytest.raises(TransportError, match="revoked"):
            remote._upload_file(tmp_path / "local", "/remote")
        with pytest.raises(TransportError, match="revoked"):
            remote._download_file("/remote", tmp_path / "local")
    assert len(calls) == 1
    with pytest.raises(TransportError, match="revoked"):
        SSHTransport(config, password="cannot-bypass")


def test_worker_credential_handoff_does_not_mutate_environment(config, monkeypatch):
    from vasp_slurm_agent.transport import password_context, worker_environment
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "unrelated-password")
    with password_context("this-task-only"):
        child = worker_environment(config)
        assert child["_DFT_AGENT_WORKER_PASSWORD"] == "this-task-only"
        assert "DFT_AGENT_SSH_PASSWORD" not in child
        config.connection_id = "registry-reference"
        assert "_DFT_AGENT_WORKER_PASSWORD" not in worker_environment(config)
    assert os.environ["DFT_AGENT_SSH_PASSWORD"] == "unrelated-password"
    assert "_DFT_AGENT_WORKER_PASSWORD" not in os.environ


def test_password_and_login_code_redacted_from_output_and_errors(config, monkeypatch):
    monkeypatch.setattr(transport_module, '_run_process', lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 255, 'secret-password', 'code 987654'))
    with SSHTransport(config, password='secret-password', code='987654') as transport:
        result = transport.run('true')
        assert result.stdout == '[REDACTED]'
        assert result.stderr == 'code [REDACTED]'
        def blocked(*args):
            raise transport_module.VerificationRequired('Verify 987654', 'Code 987654:')
        transport._native = SimpleNamespace(run=blocked, close=lambda: None)
        with pytest.raises(transport_module.VerificationRequired) as caught:
            transport.run('true')
        assert caught.value.prompt == 'Code [REDACTED]:'
        assert '987654' not in str(caught.value)


def test_authentication_password_is_not_a_login_code():
    assert transport_module.classify_prompt('Authentication password:') == 'password'
    assert transport_module.classify_prompt('Authentication code:') == 'code'


def test_detached_worker_does_not_inherit_login_code(config, monkeypatch):
    monkeypatch.setenv('_DFT_AGENT_ASKPASS_CODE', '987654')
    monkeypatch.setenv('_DFT_AGENT_ASKPASS_PROMPT_FILE', '/tmp/private-prompt')
    environment = transport_module.worker_environment(config)
    assert '_DFT_AGENT_ASKPASS_CODE' not in environment
    assert '_DFT_AGENT_ASKPASS_PROMPT_FILE' not in environment
