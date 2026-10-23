"""engine_workflow end to end with the real job script, stage runtime and collection.

A stand-in program named abinit, cp2k.psmp or elk replays outputs recorded from
real runs (tests/data/engines) in each stage folder, so every receipt, checkpoint
transfer, library check and parser runs exactly as on a cluster.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import zipfile

import pytest

from vasp_slurm_agent import engine_context, engine_workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result

DATA = Path(__file__).parent / "data" / "engines"
PROGRAMS = {"abinit": "abinit", "cp2k": "cp2k.psmp", "elk": "elk"}
PARAMETERS = {
    "abinit": {"mesh": [2, 2, 2], "dos_mesh": [3, 3, 3], "ecut_ha": 10, "line_density": 4, "max_ionic_steps": 20},
    "cp2k": {"mesh": [2, 2, 2], "dos_mesh": [3, 3, 3], "cutoff_ry": 200, "rel_cutoff_ry": 40, "line_density": 4, "max_ionic_steps": 20},
    "elk": {"mesh": [2, 2, 2], "dos_mesh": [3, 3, 3], "rgkmax": 5.0, "line_density": 4, "max_ionic_steps": 20},
}
LIBRARY = {
    "abinit": {"Si.pbe_sr.psp8": "Si stand-in psp8\n14.0000 4.0000 0 zatom,zion,pspd\n8 11 2 4 600 0 pspcod,pspxc\n"},
    "cp2k": {"BASIS_MOLOPT": "Si DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q4\n 1\n",
             "GTH_POTENTIALS": "Si GTH-PBE-q4 GTH-PBE\n 2 2\n"},
    "elk": {"Si.in": " 'Si'                                       : spsymb\n"},
}
STAND_IN = """#!{python}
import json, os, pathlib, shutil, sys
here = pathlib.Path.cwd()
engine, data = os.environ["REPLAY_ENGINE"], pathlib.Path(os.environ["REPLAY_DATA"])
metadata = json.loads((here / (engine + "-input.json")).read_text())
task = metadata["canonical_task"]
restart = {{"abinit": ["dft_agenti_DEN"], "cp2k": ["dft_agent-RESTART.kp"], "elk": ["STATE.OUT", "EFERMI.OUT"]}}[engine]
if task == "bands" and not all((here / name).is_file() for name in restart):
    sys.exit("the accepted checkpoint was not staged")
if os.environ.get("REPLAY_FAIL") == task:
    sys.exit(7)
for source in (data / engine / here.name).iterdir():
    if source.name in {{engine + "-input.json", "structure.cif"}}:
        continue
    if source.name == metadata["stdout_file"]:
        sys.stdout.write(source.read_text())
    else:
        shutil.copyfile(source, here / source.name)
if task == "scf":
    for name in {{"abinit": ["dft_agento_DEN"], "cp2k": ["dft_agent-RESTART.kp"], "elk": ["STATE.OUT", "EFERMI.OUT"]}}[engine]:
        if not (here / name).exists():
            (here / name).write_text("checkpoint")
if os.environ.get("REPLAY_CRASH_AT_EXIT") == task:
    sys.stderr.write("Lmod: module loaded\\ndouble free or corruption (out)\\n\\nProgram received signal SIGABRT\\n")
    sys.exit(134)
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReplayCluster:
    """Runs remote commands locally; jobs run synchronously at submission."""

    def __init__(self, root, engine):
        self.root, self.engine = root, engine
        self.bin = root / "bin"
        self.bin.mkdir(parents=True)
        (self.bin / "python3").symlink_to(sys.executable)
        program = self.bin / PROGRAMS[engine]
        program.write_text(STAND_IN.format(python=Path(sys.executable).resolve()))
        program.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ.get('PATH', '')}", "REPLAY_DATA": str(DATA),
                    "REPLAY_ENGINE": engine}
        self.jobs, self.exit, self.commands = {}, {}, []

    def run(self, command, timeout=60):
        self.commands.append(command)
        if command.startswith("command -v "):
            return Result(0)
        if "/dispatch.py" in command:
            words = shlex.split(command)
            remote, name = Path(words[2]), words[3]
            if "--reconcile-only" in words:
                return Result(0, json.dumps({"job_id": self.jobs[name], "job_name": name})) if name in self.jobs else Result(1, "", "none")
            job = self.jobs.setdefault(name, str(9001 + len(self.jobs)))
            done = subprocess.run(["bash", "submit.sh"], cwd=remote, env={**self.env, "SLURM_JOB_ID": job},
                                  capture_output=True, text=True, timeout=120)
            (remote / "slurm.out").write_text(done.stdout)
            (remote / "slurm.err").write_text(done.stderr)
            self.exit[job] = done.returncode
            return Result(0, json.dumps({"job_id": job, "job_name": name}))
        if command.startswith("squeue "):
            return Result(0, "")
        if command.startswith("sacct "):
            job = re.search(r"-j (\S+)", command)[1]
            code = self.exit[job]
            return Result(0, f"{job}|{'COMPLETED' if code == 0 else 'FAILED'}|{code}:0")
        done = subprocess.run(["bash", "-c", command], env=self.env, capture_output=True, text=True, timeout=timeout)
        return Result(done.returncode, done.stdout, done.stderr)

    def upload_many(self, sources, remote, *, permitted_names):
        target = Path(remote)
        target.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for source in map(Path, sources):
            assert source.name in permitted_names
            shutil.copyfile(source, target / source.name)
            manifest[source.name] = {"size": source.stat().st_size, "sha256": _sha(source)}
        return manifest

    def download_many(self, remote, folder, names, *, permitted_names):
        manifest = {}
        for name in names:
            assert name in permitted_names
            shutil.copyfile(Path(remote) / name, Path(folder) / name)
            manifest[name] = {"size": (Path(folder) / name).stat().st_size, "sha256": _sha(Path(folder) / name)}
        return manifest

    def close(self):
        pass


def _setup(tmp_path, engine, **changes):
    library = tmp_path / "library"
    library.mkdir()
    recorded = {}
    for name, text in LIBRARY[engine].items():
        (library / name).write_text(text + "#" * 1200 + "\n")
        recorded[name] = {"sha256": _sha(library / name)}
    values = dict(engine=engine, host="localhost", user="tester", remote_root=str(tmp_path / "remote"), partition="cpu",
                  engine_command=str(tmp_path / "cluster" / "bin" / PROGRAMS[engine]), engine_library_root=str(library),
                  engine_library=recorded, tasks=1)
    values.update(changes)
    config = ClusterConfig(**values)
    cluster = ReplayCluster(tmp_path / "cluster", engine)
    state = engine_workflow.prepare_plan(DATA / "Si-perturbed.cif", tmp_path / "run", config, ["relax", "bands", "dos"],
                                         PARAMETERS[engine])
    return tmp_path / "run", cluster, state, library


def _drive(run, cluster, limit=20):
    for _ in range(limit):
        state = engine_workflow.advance(run, cluster)
        if state["status"] in {"succeeded", "needs_attention", "failed", "cancelled"}:
            return state
    pytest.fail("The run did not finish.")


@pytest.mark.parametrize("engine", list(PROGRAMS))
def test_relax_then_spectra_run_end_to_end(tmp_path, engine):
    run, cluster, state, _ = _setup(tmp_path, engine)
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands", "dos"]
    state = _drive(run, cluster)
    assert state["status"] == "succeeded", state.get("last_error") or [s.get("result", {}).get("reason") for s in state["stages"]]
    for stage in state["stages"]:
        result = stage["result"]
        assert result["success"] and result["input_identity_verified"] and result["execution_verified"] and result["library_verified"]
    # The relaxed structure, not the upload, defines every later stage.
    relaxed = (run / "01_relax/outputs/final.cif").read_bytes()
    assert (run / "02_scf/inputs/structure.cif").read_bytes() != (run / "01_relax/inputs/structure.cif").read_bytes()
    assert state["stages"][1]["result"]["final_structure_sha256"] and relaxed
    context = engine_context.load_run_context(run)
    assert context["status"] == "succeeded" and context["outcome"] == "4 of 4 stages have accepted results."
    assert context["facts"]["stage_3.band_gap"]["value"] > 0.3
    assert engine_context.inspect_continuation(run)["structure_path"].endswith("02_scf/outputs/CONTCAR")
    with zipfile.ZipFile(engine_workflow.bundle_run(run)) as archive:
        names = archive.namelist()
    assert "03_bands/outputs/bands.png" in names and not any(name.endswith("submit.sh") for name in names)


def test_changed_library_file_blocks_submission(tmp_path):
    run, cluster, _, library = _setup(tmp_path, "elk")
    (library / "Si.in").write_text((library / "Si.in").read_text() + "changed\n")
    state = engine_workflow.advance(run, cluster)
    assert state["status"] == "needs_attention" and "differs from the pinned" in state["last_error"]
    assert not cluster.jobs


def test_missing_library_file_names_the_install_step(tmp_path):
    run, cluster, _, library = _setup(tmp_path, "abinit")
    (library / "Si.pbe_sr.psp8").unlink()
    state = engine_workflow.advance(run, cluster)
    assert state["status"] == "needs_attention" and "lacks Si.pbe_sr.psp8" in state["last_error"]
    assert not cluster.jobs


def test_changed_prepared_input_blocks_submission(tmp_path):
    run, cluster, _, _ = _setup(tmp_path, "cp2k")
    path = run / "01_relax/inputs/cp2k.inp"
    path.write_text(path.read_text().replace("CUTOFF 200", "CUTOFF 900"))
    state = engine_workflow.advance(run, cluster)
    assert state["status"] == "needs_attention" and "changed" in state["last_error"]
    assert not cluster.jobs


def test_program_failure_stops_before_dependent_stages(tmp_path, monkeypatch):
    run, cluster, _, _ = _setup(tmp_path, "abinit")
    cluster.env["REPLAY_FAIL"] = "scf"
    state = _drive(run, cluster)
    assert state["status"] == "needs_attention" and state["current_stage"] == 1
    assert "exited with code 7" in state["stages"][1]["result"]["reason"]
    assert state["stages"][2]["status"] == "planned" and len(cluster.jobs) == 2


def test_crash_after_complete_output_is_explained_and_not_accepted(tmp_path):
    run, cluster, _, _ = _setup(tmp_path, "abinit")
    cluster.env["REPLAY_CRASH_AT_EXIT"] = "relax"
    state = _drive(run, cluster)
    result = state["stages"][0]["result"]
    assert state["status"] == "needs_attention" and not result["success"] and not result["execution_verified"]
    assert "exited with code 134 (abort signal)" in result["reason"] and "failed while shutting down" in result["reason"]
    assert '"double free or corruption (out)"' in result["reason"] and len(cluster.jobs) == 1


def test_cancel_before_submission_stays_local(tmp_path):
    run, cluster, _, _ = _setup(tmp_path, "elk")
    state = engine_workflow.cancel(run, cluster)
    assert state["status"] == "cancelled" and not cluster.commands


def test_engines_route_through_the_shared_workflow_entry_points(tmp_path):
    from vasp_slurm_agent import workflow
    run, cluster, _, _ = _setup(tmp_path, "elk")
    state = workflow.advance(run, cluster)
    assert state["engine"] == "elk" and state["stages"][0]["job_id"]


def test_elk_jobs_get_the_stacks_and_threads_elk_recommends(tmp_path):
    run, cluster, state, _ = _setup(tmp_path, "elk")
    engine_workflow.advance(run, cluster)
    script = (run / "01_relax/inputs/submit.sh").read_text()
    assert "export OMP_STACKSIZE=512M" in script and "ulimit -Ss 524288" in script
    # Elk's OpenMP threads must not each start a threaded OpenBLAS.
    assert "export OPENBLAS_NUM_THREADS=1" in script
    assert script.index("OMP_STACKSIZE") < script.index("engine_runtime.py prepare")
