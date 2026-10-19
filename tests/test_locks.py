"""Attachment snapshots and explicit lock release on all exit paths."""

from importlib.resources import files
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import batch, workflow
from vasp_slurm_agent.config import ClusterConfig


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(host="unused.invalid", user="test", remote_root="/scratch/test",
                           vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu")
    return source, config


@pytest.mark.parametrize("failure", [False, True])
def test_windows_state_and_worker_locks_unlock_the_same_byte(tmp_path, monkeypatch, failure):
    held = set()
    calls = []

    def locking(fd, operation, size):
        assert size == 1 and os.lseek(fd, 0, os.SEEK_CUR) == 0
        calls.append(operation)
        if operation == 2:
            held.add(fd)
        else:
            assert operation == 0 and fd in held
            held.remove(fd)

    monkeypatch.setattr(workflow, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=2, LK_UNLCK=0))
    for name in (".state.lock", ".worker.lock"):
        (tmp_path / name).write_text("Existing lock files need not be empty.")
    try:
        with workflow._config_attachment_lock(tmp_path):
            assert len(held) == 2
            if failure:
                raise RuntimeError("operation failed")
    except RuntimeError:
        assert failure
    assert not held and calls == [2, 2, 0, 0]


@pytest.mark.parametrize("kind", ["run", "batch"])
def test_attachment_snapshot_never_copies_lock_files(prepared, tmp_path, monkeypatch, kind):
    source, config = prepared
    root = tmp_path / kind
    if kind == "run":
        workflow.prepare_plan(source, root, None, ["scf"])
        attach = workflow.attach_config
    else:
        batch.prepare_batch(source, root, None, ["scf"], [{"label": "A"}, {"label": "B"}])
        attach = batch.attach_batch_config
    commit = workflow._commit_config_attachment
    checked = []

    def check_snapshot(root, staged, paths, backup):
        assert not list(staged.rglob(".state.lock"))
        assert not list(staged.rglob(".worker.lock"))
        checked.append(True)
        return commit(root, staged, paths, backup)

    monkeypatch.setattr(workflow, "_commit_config_attachment", check_snapshot)
    attach(root, config)
    assert checked == [True]


@pytest.mark.parametrize("kind", ["run", "batch", "script"])
@pytest.mark.parametrize("failure_at", ["start", "close"])
def test_worker_unlocks_even_if_transport_start_or_close_fails(prepared, tmp_path, monkeypatch, kind, failure_at):
    source, config = prepared
    root = tmp_path / kind
    if kind == "run":
        workflow.prepare_plan(source, root, config, ["scf"])
        module, watch, advance, bundle = workflow, workflow.watch, "advance", "bundle_run"
    elif kind == "batch":
        batch.prepare_batch(source, root, config, ["scf"], [{"label": "A"}])
        module, watch, advance, bundle = batch, batch.watch_batch, "advance_batch", "bundle_batch"
    else:
        from vasp_slurm_agent import script_workflow
        source = tmp_path / "analysis.py"
        source.write_text("print('ok')\n")
        script_workflow.prepare_run(root, config, engine="python", files={source.name: source},
                                    entrypoint=source.name, command="python3")
        module, watch, advance, bundle = script_workflow, script_workflow.watch, "advance", "bundle_run"
    release = workflow._release
    released = []

    def record_release(stream):
        released.append(Path(stream.name).name)
        release(stream)

    def fail(*args, **kwargs):
        raise RuntimeError("transport failed")

    monkeypatch.setattr(workflow, "_release", record_release)
    if kind == "script":
        monkeypatch.setattr(module, "_release", record_release)
    monkeypatch.setattr(module, "SSHTransport", fail if failure_at == "start" else lambda *args, **kwargs: SimpleNamespace(close=fail))
    monkeypatch.setattr(module, advance, lambda *args: {"status": "cancelled"})
    monkeypatch.setattr(module, bundle, lambda *args: None)
    monkeypatch.setattr(workflow, "_announce", lambda *args: None)
    with pytest.raises(RuntimeError, match="transport failed"):
        watch(root)
    assert released == [".worker.lock"]
    with (root / ".worker.lock").open("a") as stream:
        workflow._acquire(stream, blocking=False)
        release(stream)


def test_app_worker_probe_explicitly_releases_its_lock(tmp_path, monkeypatch):
    from vasp_slurm_agent import app

    release = workflow._release
    released = []

    def record_release(stream):
        released.append(Path(stream.name).name)
        release(stream)

    monkeypatch.setattr(workflow, "_release", record_release)
    assert app._worker_alive(tmp_path) is False
    assert released == [".worker.lock"]
