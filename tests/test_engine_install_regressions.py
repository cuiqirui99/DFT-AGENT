"""Installer failure/reconnect checks using the actual remote Python scripts locally."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import pytest

from vasp_slurm_agent import engine_setup
from vasp_slurm_agent.config import ClusterConfig


pytestmark = pytest.mark.skipif(os.name != "posix", reason="The installation scripts run on Linux clusters.")


def _run(path):
    return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=30)


def _item(path, role):
    return {"name": path.name, "role": role, "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_reconnect_does_not_replace_a_live_download_worker(tmp_path):
    import fcntl

    packages = tmp_path / "pkgs"
    packages.mkdir()
    downloads = []
    for index in range(50):
        path = packages / f"cached-{index}.pkg"
        path.write_bytes(b"cached package")
        downloads.append({**_item(path, "package"), "url": "https://unused.invalid/" + path.name})
    (tmp_path / "plan.json").write_text(json.dumps({"downloads": downloads}))
    script = tmp_path / "stage.py"
    script.write_text(engine_setup._STAGE_PY)
    status = tmp_path / "stage-status.json"
    current = {"status": "staging", "done": 17, "pid": os.getpid()}
    status.write_text(json.dumps(current))
    with (tmp_path / "stage.lock").open("a") as active_worker:
        fcntl.flock(active_worker, fcntl.LOCK_EX)
        duplicate = _run(script)
        assert duplicate.returncode == 0, duplicate.stderr
        assert json.loads(status.read_text()) == current
    resumed = _run(script)
    assert resumed.returncode == 0, resumed.stderr
    completed = json.loads(status.read_text())
    assert completed["status"] == "staged" and completed["done"] == 50
    assert completed["failed"] == []
    assert not list(tmp_path.glob(".stage-status-*"))


def _installation(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    packages = root / "pkgs"
    packages.mkdir()
    prefix = tmp_path / "environment"
    # Stand-in micromamba exercises the installer process and prefix ownership,
    # without downloading packages or claiming a real scientific calculation.
    python = str(Path(sys.executable).resolve())
    program_script = "#!" + python + '\nprint("PROGRAM ENDED AT")\n'
    launcher_script = "#!" + python + "\nimport os,sys\nos.execv(sys.argv[3],sys.argv[3:])\n"
    mamba = f'''#!{python}
import os,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
prefix=pathlib.Path(sys.argv[sys.argv.index('-p')+1])
prefix.mkdir(parents=True,exist_ok=True)
(prefix/'conda-meta').mkdir(exist_ok=True)
if (root/'fail-once').exists():
    (root/'fail-once').unlink()
    (prefix/'unfinished.txt').write_text('keep the failed environment')
    sys.exit(12)
binary=prefix/'bin'
binary.mkdir(exist_ok=True)
program=binary/'cp2k.psmp'
program.write_text({program_script!r})
program.chmod(0o755)
launcher=binary/'mpiexec'
launcher.write_text({launcher_script!r})
launcher.chmod(0o755)
'''
    archive = packages / "micromamba.tar.bz2"
    with tarfile.open(archive, "w:bz2") as target:
        data = mamba.encode()
        member = tarfile.TarInfo("bin/micromamba")
        member.size = len(data)
        target.addfile(member, io.BytesIO(data))
    package = packages / "example.conda"
    package.write_bytes(b"package bytes verified by the installer")
    plan = {"id": "attempt-1", "prefix": str(prefix), "program": "cp2k.psmp", "engine": "cp2k",
            "version": "test", "build_cpus": 2, "mpi": "mpich", "library_root": str(tmp_path / "library"),
            "downloads": [_item(archive, "micromamba"), _item(package, "package")]}
    (root / "plan.json").write_text(json.dumps(plan))
    smoke = root / "smoke"
    smoke.mkdir()
    (smoke / "cp2k.inp").write_text("test input")
    (smoke / "metadata.json").write_text(json.dumps({"input_files": ["cp2k.inp"], "library_dir": "library",
        "library_files": [], "arguments": [], "stdout_file": "cp2k.out", "output_file": "cp2k.out"}))
    script = root / "install.py"
    script.write_text(engine_setup._INSTALL_PY)
    return root, prefix, script


def test_partial_owned_install_retries_and_keeps_the_failed_prefix(tmp_path):
    root, prefix, script = _installation(tmp_path)
    (root / "fail-once").touch()
    first = _run(script)
    assert first.returncode == 1
    assert (prefix / "unfinished.txt").is_file()
    assert not (prefix / ".dft-agent-environment.json").exists()
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["id"] = "attempt-2"
    plan_path.write_text(json.dumps(plan))
    retried = _run(script)
    assert retried.returncode == 0, retried.stderr
    result = json.loads((root / "install-status.json").read_text())
    assert result["status"] == "installed"
    assert (Path(result["retained_prefix"]) / "unfinished.txt").read_text() == "keep the failed environment"
    assert not (prefix / "unfinished.txt").exists()
    assert (prefix / ".dft-agent-environment.json").is_file()
    # A completed environment remains reusable after package archives were removed.
    again = _run(script)
    assert again.returncode == 0, again.stderr
    assert json.loads((root / "install-status.json").read_text())["created"] is False


def test_unknown_nonempty_installation_directory_is_preserved(tmp_path):
    root, prefix, script = _installation(tmp_path)
    prefix.mkdir()
    valuable = prefix / "existing-user-file"
    valuable.write_text("leave this alone")
    result = _run(script)
    assert result.returncode == 1
    assert "not created by this installation" in json.loads((root / "install-status.json").read_text())["error"]
    assert valuable.read_text() == "leave this alone"
    assert not list(tmp_path.glob("environment.incomplete-*"))


class _LocalTransport:
    def __init__(self, environment):
        self.environment = environment

    def run(self, command, timeout=60):
        return subprocess.run(["bash", "-c", command], env=self.environment,
                              capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("program_state", ["missing", "not_executable", "executable"])
def test_doctor_checks_absolute_program_path_and_executable_bit(tmp_path, program_state):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    program = tmp_path / "abinit"
    if program_state != "missing":
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755 if program_state == "executable" else 0o644)
    config = ClusterConfig(host="test.invalid", user="user", remote_root=str(tmp_path), partition="cpu", engine="abinit",
                           engine_command=str(program), engine_library_root=str(tmp_path))
    transport = _LocalTransport({**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]})
    result = engine_setup.doctor(config, transport=transport)
    assert result["checks"]["ABINIT"]["ok"] is (program_state == "executable")
    assert result["ok"] is (program_state == "executable")


def test_doctor_uses_the_configured_environment(tmp_path):
    tools = tmp_path / "module-bin"
    tools.mkdir()
    for name in ("abinit", "sbatch", "squeue", "sacct", "scancel"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    config = ClusterConfig(host="test.invalid", user="user", remote_root=str(tmp_path), partition="cpu", engine="abinit",
                           engine_command="abinit", engine_library_root=str(tmp_path),
                           setup_commands=["export PATH=" + shlex.quote(str(tools)) + ':"$PATH"'])
    result = engine_setup.doctor(config, transport=_LocalTransport(dict(os.environ)))
    assert result["ok"] and result["checks"]["ABINIT"]["detail"] == str(tools / "abinit")


def test_managed_mpi_rejects_multiple_nodes_but_keeps_site_builds(tmp_path):
    from vasp_slurm_agent.engine_workflow import prepare_plan

    common = {"host": "test.invalid", "user": "user", "remote_root": str(tmp_path), "partition": "cpu",
              "engine": "cp2k", "engine_command": "/env/bin/mpiexec -n {tasks} /env/bin/cp2k.psmp",
              "engine_library_root": str(tmp_path), "tasks": 8}
    managed = engine_setup.environment_setup("cp2k", {"mpi": "mpich"})
    engine_setup.validate_managed_resources(ClusterConfig(**common, nodes=1, setup_commands=managed))
    with pytest.raises(ValueError, match="one node"):
        engine_setup.validate_managed_resources(ClusterConfig(**common, nodes=2, setup_commands=managed))
    engine_setup.validate_managed_resources(ClusterConfig(**common, nodes=2, setup_commands=["module load cp2k"]))
    source = tmp_path / "Si.cif"
    source.write_text(engine_setup._SI)
    with pytest.raises(ValueError, match="one node"):
        prepare_plan(source, tmp_path / "run", ClusterConfig(**common, nodes=2, setup_commands=managed), ["scf"])
    assert not (tmp_path / "run").exists()
