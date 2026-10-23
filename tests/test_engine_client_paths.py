"""Exercise native client paths against a simulated POSIX remote file service.

Recorded scientific outputs test collection and parsing; the simulated receipts
only test the client protocol, not a live calculation or the Linux job runtime.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

import pytest

from test_engine_workflow import DATA, LIBRARY, PARAMETERS, PROGRAMS
from vasp_slurm_agent import engine_workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result


def _receipt(data):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class RemoteFiles:
    def __init__(self, engine, library):
        self.engine, self.library = engine, library
        self.files, self.uploads, self.downloads = {}, [], []

    def run(self, command, timeout=60):
        if command.startswith("command -v "):
            return Result(0)
        if command.startswith("squeue "):
            return Result(0, "")
        if command.startswith("sacct "):
            return Result(0, "9001|COMPLETED|0:0")
        args = shlex.split(command)
        if args[:2] == ["python3", "-c"]:
            if "validate_library" in args[2]:
                assert "/scratch/tester/library" in args[2]
                return Result(0, json.dumps({"files": self.library, "missing": []}))
            assert "root.iterdir()" in args[2] and self.remote in args[2]
            return Result(0, json.dumps(list(self.files)))
        assert args[:3] == ["python3", self.remote + "/dispatch.py", self.remote]
        for path in (DATA / self.engine / "02_scf").iterdir():
            self.files.setdefault(path.name, path.read_bytes())
        runtime = json.loads(self.files["runtime.json"])
        self.files["input_hashes.sha256"] = "".join(
            digest + "  " + name + "\n" for name, digest in runtime["input_sha256"].items()).encode()
        self.files["execution.json"] = json.dumps({"job_id": "9001", "exit_code": 0,
            "runtime_sha256": _receipt(self.files["runtime.json"])["sha256"],
            "finished_at": "2026-01-01T00:00:00Z"}).encode()
        self.files["checkpoint.json"] = json.dumps({
            name: _receipt(b"simulated checkpoint") for name in runtime["checkpoints"]}).encode()
        return Result(0, json.dumps({"job_id": "9001", "job_name": args[3]}))

    def upload_many(self, sources, remote, *, permitted_names):
        assert PurePosixPath(remote).is_absolute() and "\\" not in remote
        self.remote = remote
        self.uploads.append(remote)
        for source in map(Path, sources):
            assert source.is_file() and source.name in permitted_names
            self.files[source.name] = source.read_bytes()
        return {name: _receipt(data) for name, data in self.files.items()}

    def download_many(self, remote, folder, names, *, permitted_names):
        assert remote == self.remote and "\\" not in remote
        self.downloads.append(remote)
        for name in names:
            assert name in permitted_names
            (Path(folder) / name).write_bytes(self.files[name])
        return {name: _receipt(self.files[name]) for name in names}


@pytest.mark.parametrize("engine", list(PROGRAMS))
def test_native_client_submits_and_collects_from_posix_remote(tmp_path, engine):
    library = {name: _receipt(text.encode()) for name, text in LIBRARY[engine].items()}
    config = ClusterConfig(engine=engine, host="cluster.invalid", user="tester", tasks=1,
        partition="cpu", remote_root="/scratch/tester/runs", engine_library_root="/scratch/tester/library",
        engine_command="srun " + PROGRAMS[engine], engine_library=library)
    run = tmp_path / "Local user Ω" / "run"
    state = engine_workflow.prepare_plan(DATA / engine / "02_scf" / "structure.cif", run, config,
        ["scf"], PARAMETERS[engine])
    remote = RemoteFiles(engine, library)
    queued = engine_workflow.advance(run, remote)
    assert queued["status"] == "queued", queued["last_error"]
    assert remote.uploads == ["/scratch/tester/runs/" + state["run_id"] + "/01_scf"]
    assert b"\r\n" not in remote.files["submit.sh"]
    finished = engine_workflow.advance(run, remote)
    result = finished["stages"][0]["result"]
    assert finished["status"] == "succeeded", finished.get("last_error") or result
    assert result["input_identity_verified"] and result["execution_verified"] and result["library_verified"]
    assert remote.downloads == remote.uploads
    assert (run / "01_scf" / "outputs" / "CONTCAR").is_file()
