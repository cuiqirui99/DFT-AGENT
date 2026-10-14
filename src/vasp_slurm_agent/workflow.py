from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # Windows has no fcntl; msvcrt locks a byte of the lock file instead.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from .config import ClusterConfig
from .transport import SSHTransport, TransportError
from .vasp import prepare_inputs, analyze_outputs
from .restart import INPUT_FILES as RESTART_INPUTS, OUTPUT_FILES as RESTART_OUTPUTS, prepare_warm_inputs

TERMINAL = {"succeeded", "needs_attention", "failed", "cancelled"}
MAX_REMOTE_FAILURES = 3
SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
FILES = ("INCAR", "KPOINTS", "POSCAR", "OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR", "EIGENVAL", "DOSCAR", "PROCAR", "XDATCAR", "REPORT", "slurm.out", "slurm.err", "execution.json", "input_hashes.sha256", "potcar_hash.sha256", "potcar_titles.txt", *RESTART_INPUTS, *RESTART_OUTPUTS)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _acquire(stream, blocking=True):
    """Take an exclusive lock on an open file; BlockingIOError when busy and not blocking."""
    if fcntl is not None:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    import msvcrt  # pragma: no cover - Windows only
    stream.seek(0)
    while True:
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if not blocking:
                raise BlockingIOError("The lock is held by another process.")
            time.sleep(0.1)


def _release(stream):
    if fcntl is not None:
        fcntl.flock(stream, fcntl.LOCK_UN)
    else:
        import msvcrt  # pragma: no cover - Windows only
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _lock(root):
    with (Path(root) / ".state.lock").open("a") as stream:
        _acquire(stream)
        try:
            yield
        finally:
            _release(stream)


def read_state(run_dir):
    return json.loads((Path(run_dir) / "run.json").read_text())


def _save(root, state, event=None):
    state["updated_at"] = utc_now()
    if event:
        state["history"].append({"at": state["updated_at"], "event": event})
    _write(Path(root) / "run.json", state)
    return state


def prepare_run(structure_path, run_dir, config, task="relax", parameters=None):
    return prepare_plan(structure_path, run_dir, config, [task], parameters)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _symbols(config):
    """POTCAR folder names from the cluster settings; defaults before a cluster is attached."""
    return dict(config.potcar_symbols) if config is not None else {}


def _launch_command(config, metadata):
    if config is None:
        return ""
    command = config.vasp_ncl_command if metadata.get("requires_ncl") else config.vasp_command
    if not command.strip():
        raise ValueError("Set vasp_ncl_command before preparing SOC or noncollinear calculations.")
    return command


def _stage_recipes(tasks, parameters, stage_parameters):
    if stage_parameters is None:
        return [deepcopy(parameters) for _ in tasks]
    if not isinstance(stage_parameters, (list, tuple)) or len(stage_parameters) != len(tasks) or any(not isinstance(item, dict) for item in stage_parameters):
        raise ValueError("Supply one parameter override object per requested task.")
    return [{**deepcopy(parameters), **deepcopy(item)} for item in stage_parameters]


def _charge_recipe(parameters, input_policy=2):
    from .methods import METHOD_DEFAULTS
    from .vasp import _parameters
    if input_policy == 1:
        from .legacy_inputs import legacy_parameters
        parameters = legacy_parameters(parameters)
    settings = {**METHOD_DEFAULTS, **_parameters(parameters)}
    if settings.get("mesh") is not None:
        settings.pop("kspacing", None)
    return {key: value for key, value in settings.items()
            if key not in {"nsw", "ediffg", "cell_relax", "line_density", "nedos"}}


def prepare_plan(structure_path, run_dir, config, tasks, parameters=None, magnetic_states=None, initial_moment=3.0,
                 stage_parameters=None, *, _input_policy=2):
    """Freeze an offline plan; prepare post-relaxation inputs only after acceptance.

    With config=None the inputs are prepared locally for review; attach_config
    adds the cluster settings before submission.
    """
    if config is not None and config.engine == "qe":
        if magnetic_states:
            raise ValueError("Use explicit QE starting magnetizations for each magnetic calculation.")
        from .qe_workflow import prepare_plan as prepare_qe
        return prepare_qe(structure_path, run_dir, config, tasks, parameters,
                          stage_parameters=stage_parameters)
    if isinstance(magnetic_states, (list, tuple)) and not magnetic_states:
        magnetic_states = None
    if type(_input_policy) is not int or _input_policy not in {1, 2}:
        raise ValueError("Unknown frozen input policy.")
    if not isinstance(tasks, (list, tuple)) or not tasks or any(task not in {"relax", "scf", "bands", "dos"} for task in tasks):
        raise ValueError("Choose an ordered list of relax, scf, bands or dos tasks.")
    tasks = list(tasks)
    if len(set(tasks)) != len(tasks) or ("relax" in tasks and tasks[0] != "relax"):
        raise ValueError("Use each task once and put relaxation first.")
    if "scf" in tasks and any(task in {"bands", "dos"} for task in tasks[:tasks.index("scf")]):
        raise ValueError("Put SCF before bands and DOS.")
    if magnetic_states is not None and tasks != ["scf"]:
        raise ValueError("Magnetic comparisons currently support SCF only.")
    parameters = deepcopy(dict(parameters or {}))
    recipes = _stage_recipes(tasks, parameters, stage_parameters)
    root = Path(run_dir).expanduser().resolve()
    source = Path(structure_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This run already exists. Resume it or choose a new run directory.")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dft-plan-", dir=root.parent))
    try:
        source_dir = temporary / "source" / "input"
        source_dir.mkdir(parents=True)
        source_relative = "source/input/" + source.name
        shutil.copyfile(source, temporary / source_relative)
        frozen_sources = {source_relative: _digest(temporary / source_relative)}
        seed_info = None
        if magnetic_states is not None:
            from .magnetic import make_candidates
            from pymatgen.io.vasp import Poscar
            common, candidates, seed_info = make_candidates(temporary / source_relative, magnetic_states, recipes[0], initial_moment)
            source_relative = "source/comparison/POSCAR"
            (temporary / source_relative).parent.mkdir()
            Poscar(common).write_file(temporary / source_relative)
            frozen_sources[source_relative] = _digest(temporary / source_relative)
            sequence = [{"name": "scf", "label": candidate["label"], "parameters": candidate["parameters"]} for candidate in candidates]
        else:
            sequence = []
            scf_recipe = None
            for task, recipe in zip(tasks, recipes):
                hybrid = recipe.get("functional", "PBE") in {"HSE06", "PBE0"}
                identity = _charge_recipe(recipe, _input_policy)
                if task in {"bands", "dos"} and not hybrid and identity != scf_recipe:
                    sequence.append({"name": "scf", "parameters": deepcopy(recipe)})
                    scf_recipe = identity
                sequence.append({"name": task, "parameters": deepcopy(recipe)})
                if task == "scf":
                    scf_recipe = identity
                elif task == "relax":
                    scf_recipe = None
        run_id = "vsa-" + uuid.uuid4().hex[:16]
        stages = []
        relaxed = scf = None
        for index, spec in enumerate(sequence):
            name = spec["name"]
            suffix = name + ("_" + spec["label"].lower() if "label" in spec else "")
            charge_from = scf if name in {"bands", "dos"} and str(spec["parameters"].get("functional", "PBE")).upper() == "PBE" else None
            stage = {**spec, "folder": f"{index+1:02d}_{suffix}", "job_name": f"{run_id}-{index+1:02d}-{suffix}",
                     "status": "planned", "job_id": None, "metadata": {}, "staged": False,
                     "materialized": False, "result": None, "structure_from": relaxed,
                     "charge_from": charge_from, "source_path": source_relative}
            if _input_policy == 2 and str(spec["parameters"].get("functional", "PBE")).upper() in {"HSE06", "PBE0"}:
                stage["warm_start"] = True
            stages.append(stage)
            if name == "relax":
                relaxed, scf = index, None
            elif name == "scf" and magnetic_states is None:
                scf = index
        plan = {"schema_version": 1, "input_policy": _input_policy, "tasks": tasks, "parameters": parameters,
                "sources": frozen_sources, "magnetic_seeds": seed_info,
                "stages": [{key: deepcopy(stage[key]) for key in ("name", "folder", "parameters", "structure_from", "charge_from", "source_path")}
                           for stage in stages]}
        if stage_parameters is not None:
            plan["stage_parameters"] = deepcopy(list(stage_parameters))
        for spec, stage in zip(plan["stages"], stages):
            if "label" in stage:
                spec["label"] = stage["label"]
            if stage.get("warm_start"):
                spec["warm_start"] = True
        _write(temporary / "plan.json", plan)
        if config is not None:
            config.save(temporary / "config.json")
        state = {"schema_version": 2, "input_policy": _input_policy, "run_id": run_id,
                 "task": "magnetic" if magnetic_states is not None else tasks[0] if len(tasks) == 1 else "pipeline",
                 "tasks": tasks, "formula": "", "status": "planned", "parameters": parameters,
                 "current_stage": 0, "stages": stages, "created_at": utc_now(), "history": [],
                 "last_error": None, "cancel_requested": False,
                 "remote_root": config.remote_root.rstrip("/") + "/" + run_id if config is not None else None,
                 "plan_sha256": _digest(temporary / "plan.json"),
                 "config_sha256": _digest(temporary / "config.json") if config is not None else None}
        if magnetic_states is not None:
            state["comparison"] = {"scope": seed_info["scope"], "status": "pending", "ranking": [], "excluded": []}
        for stage in stages:
            if stage["structure_from"] is None:
                _materialize_stage(temporary, state, config, stage)
            elif stage_parameters is not None:
                # Validate later recipes before any job starts; their actual inputs
                # still wait for the accepted relaxed structure.
                preview = temporary / ".recipe-check"
                build = _input_builder(state, stage)
                metadata = build(temporary / source_relative, preview, stage["name"],
                                 stage["parameters"], _symbols(config))
                _launch_command(config, metadata)
                shutil.rmtree(preview)
        state["formula"] = stages[0]["metadata"].get("formula", "")
        _save(temporary, state, "Plan prepared. Ready to submit.")
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _input_builder(state, stage):
    policy = state.get("input_policy", 1)
    if policy == 1:
        from .legacy_inputs import prepare_legacy_inputs
        return prepare_legacy_inputs
    if policy != 2:
        raise ValueError("Unknown frozen input policy.")
    return prepare_warm_inputs if stage.get("warm_start") else prepare_inputs


def _materialize_stage(root, state, config, stage):
    if stage.get("materialized", True):
        return
    parameters = deepcopy(stage["parameters"])
    previous_index = stage.get("structure_from")
    source = root / stage["source_path"]
    if previous_index is not None:
        previous = state["stages"][previous_index]
        if previous["status"] != "succeeded" or not (previous.get("result") or {}).get("success"):
            raise ValueError("The preceding relaxation must succeed before preparing this stage.")
        output = root / previous["folder"] / "outputs"
        digest = previous["result"].get("final_structure_poscar_sha256")
        if digest:
            source = output / "final_structure.vasp"
        elif previous["metadata"].get("requires_ncl"):
            raise ValueError("SOC/noncollinear propagation requires the accepted final_structure.vasp to preserve its Cartesian basis.")
        else:
            source = output / "final_structure.cif"
            digest = previous["result"].get("final_structure_sha256")
        if not source.is_file() or _digest(source) != digest:
            raise ValueError("The accepted relaxed structure is missing or changed.")
        moments = parameters.get("magmom")
        if moments is not None:
            order = previous["metadata"].get("input_site_order")
            if order is not None:
                if len(order) != len(moments):
                    raise ValueError("The next-stage moments do not match the relaxed sites.")
                parameters["magmom"] = [deepcopy(moments[index]) for index in order]
    destination = root / stage["folder"] / "inputs"
    temporary = Path(tempfile.mkdtemp(prefix=".inputs-", dir=root))
    try:
        build = _input_builder(state, stage)
        metadata = build(source, temporary / "inputs", stage["name"], parameters, _symbols(config))
        _launch_command(config, metadata)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if any(not (destination / name).is_file() or _digest(destination / name) != digest
                   for name, digest in metadata["input_sha256"].items()):
                raise ValueError("Prepared inputs changed before stage materialization completed.")
        else:
            os.replace(temporary / "inputs", destination)
        stage["metadata"] = metadata
        stage["materialized"] = True
        stage["source_sha256"] = _digest(source)
    finally:
        shutil.rmtree(temporary)


def has_config(run_dir):
    """True when cluster settings are attached to this run."""
    return (Path(run_dir).expanduser() / "config.json").is_file()


def _prepare_config_attachment(root, config):
    """Rebuild a private copy; callers commit it only after every check passes."""
    state = read_state(root)
    if state["status"] != "planned" or any(stage.get("job_id") or stage.get("staged") for stage in state["stages"]):
        raise ValueError("Cluster settings can only be attached before the first submission.")
    _check_config(root, state)
    config.save(root / "config.json")
    state["config_sha256"] = _digest(root / "config.json")
    state["remote_root"] = config.remote_root.rstrip("/") + "/" + state["run_id"]
    for stage in state["stages"]:
        if stage.get("materialized"):
            shutil.rmtree(root / stage["folder"] / "inputs")
            stage["materialized"] = False
            stage["metadata"] = {}
    for stage in state["stages"]:
        if stage["structure_from"] is None:
            _materialize_stage(root, state, config, stage)
        else:
            # Later inputs still need the accepted relaxed structure, but all
            # recipes and launch commands must be usable before the first job.
            with tempfile.TemporaryDirectory(prefix=".recipe-check-", dir=root) as directory:
                metadata = _input_builder(state, stage)(
                    root / stage["source_path"], Path(directory) / "inputs", stage["name"],
                    stage["parameters"], _symbols(config),
                )
                _launch_command(config, metadata)
    state["formula"] = state["stages"][0]["metadata"].get("formula", state.get("formula", ""))
    return _save(root, state, "Cluster settings attached.")


def _config_attachment_files(state):
    return ["config.json", *(stage["folder"] + "/inputs" for stage in state["stages"]), "run.json"]


@contextmanager
def _config_attachment_lock(root):
    # Workers load the transport before taking the state lock. Keep them from
    # observing one cluster configuration and then submitting with another.
    with (root / ".worker.lock").open("a") as worker:
        try:
            _acquire(worker, blocking=False)
        except BlockingIOError as exc:
            raise ValueError("Cluster settings cannot be attached while monitoring is running.") from exc
        try:
            with _lock(root):
                yield
        finally:
            _release(worker)


@contextmanager
def _config_attachment_workspace(root):
    temporary = Path(tempfile.mkdtemp(prefix=".dft-attach-", dir=root.parent))
    try:
        yield temporary
    except BaseException:
        original = temporary / "original"
        # A filesystem error may also prevent rollback. Never delete the only
        # remaining copies of original files in that case.
        if not original.exists() or not any(path.is_file() for path in original.rglob("*")):
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    else:
        shutil.rmtree(temporary, ignore_errors=True)


def _commit_config_attachment(root, prepared, relative_paths, backup):
    """Replace checked files while retaining originals for rollback on failure.

    Keep the original run directories and lock files in place so another worker
    cannot acquire a different lock inode during the commit.
    """
    backup.mkdir()
    changed = []
    try:
        for relative in relative_paths:
            source, target = prepared / relative, root / relative
            saved = backup / relative
            if not source.exists() and not target.exists():
                continue
            if target.exists():
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, saved)
            changed.append((source, target, saved))
            if source.exists():
                os.replace(source, target)
    except BaseException:
        rollback_error = None
        for source, target, saved in reversed(changed):
            try:
                if target.exists():
                    os.replace(target, source)
                if saved.exists():
                    os.replace(saved, target)
            except OSError as exc:
                rollback_error = exc
        if rollback_error is not None:
            raise OSError(f"Could not fully restore the original run files. Originals are retained at {backup}.") from rollback_error
        raise


def attach_config(run_dir, config):
    """Add or replace cluster settings on a run that has not been submitted.

    Prepared inputs are rebuilt so POTCAR labels and launch commands follow the
    attached cluster. The frozen plan and source structure do not change.
    """
    if config is None:
        raise ValueError("Save Cluster setup before attaching it to a run.")
    if read_state(run_dir).get("engine") == "qe":
        from .qe_workflow import attach_config as attach_qe
        return attach_qe(run_dir, config)
    if config.engine != "vasp":
        raise ValueError("Attach a VASP cluster to this VASP run.")
    root = Path(run_dir).expanduser().resolve()
    with _config_attachment_lock(root):
        with _config_attachment_workspace(root) as temporary:
            prepared = temporary / "run"
            shutil.copytree(root, prepared, ignore=shutil.ignore_patterns(".state.lock", ".worker.lock"))
            state = _prepare_config_attachment(prepared, config)
            _commit_config_attachment(root, prepared, _config_attachment_files(state), temporary / "original")
            return state


def _command(transport, command, timeout=60):
    result = transport.run(command, timeout=timeout)
    if result.returncode:
        raise TransportError((result.stderr or result.stdout or f"Remote exit {result.returncode}")[-3000:])
    return result.stdout.strip()


def _check_config(root, state):
    if state.get("config_sha256") and hashlib.sha256((root / "config.json").read_bytes()).hexdigest() != state["config_sha256"]:
        raise ValueError("Run cluster configuration changed. Restore the approved snapshot to reconnect safely.")
    if state.get("plan_sha256"):
        if _digest(root / "plan.json") != state["plan_sha256"]:
            raise ValueError("The frozen plan changed. Prepare a new run.")
        plan = json.loads((root / "plan.json").read_text())
        if state.get("input_policy", 1) != plan.get("input_policy", 1):
            raise ValueError("The frozen input policy changed.")
        if len(plan["stages"]) != len(state["stages"]):
            raise ValueError("Run stages differ from the frozen plan.")
        for spec, stage in zip(plan["stages"], state["stages"]):
            if any(stage.get(key) != value for key, value in spec.items()):
                raise ValueError("Run stages differ from the frozen plan.")
        if any(_digest(root / name) != digest for name, digest in plan["sources"].items()):
            raise ValueError("The frozen source structure changed. Prepare a new run.")


def _script(config, stage, previous):
    command = _launch_command(config, stage["metadata"])
    labels = stage["metadata"]["potcar_labels"]
    paths = []
    for label in labels:
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
            raise ValueError("Unsafe POTCAR label")
        paths.append(shlex.quote(config.potcar_root.rstrip("/") + "/" + label + "/POTCAR"))
    account = f"#SBATCH --account={config.account}\n" if config.account else ""
    dependency = ""
    needs_charge = stage["metadata"].get("requires_chgcar", stage["name"] in {"bands", "dos"})
    needs_charge = needs_charge and stage["metadata"].get("spectral_charge_mode") != "self_consistent"
    if needs_charge and not previous:
        raise ValueError("This spectrum requires an accepted SCF charge density.")
    if needs_charge:
        dependency = f"test -s {shlex.quote(previous + '/CHGCAR')}\ncp {shlex.quote(previous + '/CHGCAR')} CHGCAR\n"
    setup = "\n".join(config.setup_commands).replace("\r\n", "\n")
    resources = "\n".join(config.sbatch_resource_lines())
    launch = command
    if stage["metadata"].get("warm_start"):
        launch = f'''(
set -e
python3 restart.py initialize
set +e
(cd .pbe_seed && {command}) > seed.stdout 2> seed.stderr
seed_code=$?
set -e
python3 restart.py seed "$seed_code"
set +e
{command} > hybrid.stdout 2> hybrid.stderr
hybrid_code=$?
set -e
cat hybrid.stdout
cat hybrid.stderr >&2
python3 restart.py finish "$hybrid_code"
)'''
    return f'''#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks={config.tasks}
#SBATCH --time={config.walltime}
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
{account}{resources + chr(10) if resources else ''}set -e
{setup}
{dependency}cat {' '.join(paths)} > POTCAR
test -s POTCAR
sha256sum POSCAR INCAR KPOINTS > input_hashes.sha256
sha256sum POTCAR > potcar_hash.sha256
grep TITEL POTCAR > potcar_titles.txt
python3 - <<'PY'
import json,os,datetime
json.dump({{"job_id":os.environ.get("SLURM_JOB_ID"),"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tasks":os.environ.get("SLURM_NTASKS")}},open("execution.json","w"))
PY
set +e
{launch}
code=$?
set -e
python3 - "$code" <<'PY'
import json,sys,datetime
p=json.load(open("execution.json"));p.update(exit_code=int(sys.argv[1]),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(p,open("execution.json","w"))
PY
exit "$code"
'''


def _submit(root, state, config, transport):
    index = state["current_stage"]
    stage = state["stages"][index]
    _materialize_stage(root, state, config, stage)
    _launch_command(config, stage["metadata"])
    charge_index = stage.get("charge_from", index - 1 if index and stage["name"] in {"bands", "dos"} else None)
    previous = None
    if charge_index is not None:
        charge_stage = state["stages"][charge_index]
        if charge_stage["name"] != "scf" or charge_stage["status"] != "succeeded" or not (charge_stage.get("result") or {}).get("success"):
            raise ValueError("The SCF dependency has not succeeded.")
        for key in ("method_fingerprint",):
            if key in stage["metadata"] and charge_stage["metadata"].get(key) != stage["metadata"][key]:
                raise ValueError("The SCF method does not match this spectrum.")
        if stage["metadata"]["input_sha256"]["POSCAR"] != charge_stage["metadata"]["input_sha256"]["POSCAR"]:
            raise ValueError("The SCF structure does not match this spectrum.")
        previous = state["remote_root"] + "/" + charge_stage["folder"]
    remote = state["remote_root"] + "/" + stage["folder"]
    if stage["status"] == "planned":
        stage["status"] = state["status"] = "submitting"
        _save(root, state, f"Submitting {stage['name']}")
    if not stage["staged"]:
        inputs = root / stage["folder"] / "inputs"
        for name in ("POSCAR", "INCAR", "KPOINTS"):
            if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != stage["metadata"]["input_sha256"][name]:
                raise ValueError(f"Prepared {name} changed. Prepare and review a new run before submitting.")
        labels = stage["metadata"]["potcar_labels"]
        checks = ["command -v sbatch >/dev/null", "command -v squeue >/dev/null", "command -v sacct >/dev/null", "command -v python3 >/dev/null"]
        checks += ["test -s " + shlex.quote(config.potcar_root.rstrip("/") + "/" + label + "/POTCAR") for label in labels]
        _command(transport, " && ".join(checks))
        _command(transport, "mkdir -p " + shlex.quote(remote))
        (inputs / "submit.sh").write_text(_script(config, stage, previous), encoding="utf-8", newline="\n")
        files = [inputs / name for name in ("POSCAR", "INCAR", "KPOINTS", "submit.sh")]
        if stage["metadata"].get("warm_start"):
            from .restart import sha256
            spec = stage["metadata"]["warm_start"]
            if sha256(inputs / "warm_start.spec.json") != spec["spec_sha256"]:
                raise ValueError("Prepared warm-start inputs changed. Prepare a new run.")
            files.extend(inputs / name for name in RESTART_INPUTS)
            files.append(Path(__file__).with_name("restart.py"))
        files.append(Path(__file__).with_name("dispatch.py"))
        if hasattr(transport, "upload_many"):
            transport.upload_many(files, remote)
        else:
            for file in files:
                transport.upload(file, remote + "/" + file.name)
        stage["staged"] = True
        _save(root, state, f"Uploaded {stage['name']} inputs")
    # Remote program owns the durable intent/receipt under a file lock. Replaying
    # this call after a lost SSH response returns/reconciles the same job.
    result = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])}", timeout=100)
    receipt = json.loads(result.splitlines()[-1])
    if not str(receipt["job_id"]).isdigit():
        raise ValueError("Invalid Slurm job identity")
    stage["job_id"] = str(receipt["job_id"])
    stage["status"] = state["status"] = "queued"
    state["last_error"] = None
    state["remote_failures"] = 0
    stage["submitted_at"] = utc_now()
    return _save(root, state, f"Slurm accepted job {stage['job_id']}")


def _poll(transport, stage):
    job = stage["job_id"]
    if not str(job).isdigit():
        raise ValueError("Invalid persisted job identity")
    try:
        live = _command(transport, f"squeue -h -j {job} -o '%T'")
    except TransportError as exc:
        # Slurm removes old completed jobs from squeue; sacct owns their result.
        if "slurm_load_jobs error: Invalid job id specified" not in str(exc):
            raise
        live = ""
    if live:
        return live.splitlines()[0].strip().split()[0]
    accounting = _command(transport, f"sacct -n -X -P -j {job} --format=JobIDRaw,State,ExitCode")
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job:
            state = fields[1].split()[0].rstrip("+")
            if state == "COMPLETED" and fields[2] != "0:0":
                return "FAILED"
            return state
    return "UNKNOWN"  # Scheduler/accounting lag is never treated as job death.


def _failure_diagnostic(output, scheduler):
    """Give a short, evidence-bound hint from scheduler state and log tails."""
    tails = []
    for name in ("slurm.out", "slurm.err", "OUTCAR"):
        path = Path(output) / name
        if path.is_file():
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65536))
                tails.append(stream.read().decode("utf-8", errors="replace"))
    text = "\n".join(tails)
    if scheduler == "OUT_OF_MEMORY" or re.search(r"out[ _-]of[ _-]memory|oom[ _-]kill", text, re.IGNORECASE):
        return {
            "failure_code": "OUT_OF_MEMORY",
            "message": "The job ran out of memory.",
            "recovery_hint": "Use more memory in a new run.",
        }
    if scheduler == "TIMEOUT" or re.search(r"due to time limit|time limit (?:exceeded|reached)|walltime.*(?:exceeded|limit)", text, re.IGNORECASE):
        return {
            "failure_code": "TIME_LIMIT",
            "message": "The job reached its time limit.",
            "recovery_hint": "Increase the time limit in a new run. Saved structures may be incomplete.",
        }
    if re.search(r"error\s+EDDDAV\b|\b(?:EDDDAV|ZHEGV)\b[^\n]{0,160}\b(?:failed|failure|error)\b", text, re.IGNORECASE):
        return {
            "failure_code": "VASP_EDDDAV_ZHEGV",
            "message": "VASP stopped during diagonalization (EDDDAV/ZHEGV).",
            "recovery_hint": "The log does not establish the cause. Check inputs; for small systems, consider fewer MPI tasks in a new run.",
        }
    return None


def _update_comparison(state):
    if "comparison" not in state:
        return
    method_keys = {"spin", "magmom", "soc", "saxis", "functional", "hubbard_u"}

    def compatibility(metadata):
        return {
            "method": metadata.get("method_comparison_fingerprint"),
            "structure": metadata.get("input_sha256", {}).get("POSCAR"),
            "atoms": metadata.get("number_of_atoms"),
            "formula": metadata.get("formula"),
            "settings": {key: value for key, value in metadata.get("parameters", {}).items() if key not in method_keys},
        }

    baseline = compatibility(state["stages"][0]["metadata"])
    ranked, excluded = [], []
    for stage in state["stages"]:
        result = stage.get("result")
        if result is None:
            continue
        reason = None
        metadata = stage["metadata"]
        if stage["status"] != "succeeded" or not result.get("success") or not result.get("converged_electronic"):
            reason = result.get("reason") or "Calculation did not converge."
        elif not baseline["method"] or compatibility(metadata) != baseline:
            reason = "Method, structure or numerical settings differ."
        elif result.get("method_comparison_fingerprint") != metadata.get("method_comparison_fingerprint"):
            reason = "Output method identity differs from the prepared inputs."
        energy, count = result.get("final_energy_ev"), metadata.get("number_of_atoms")
        if reason is None and (isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy)
                               or not isinstance(count, int) or count <= 0):
            reason = "Finite energy and atom count are required."
        if reason:
            excluded.append({"seed": stage["label"], "reason": reason})
        else:
            ranked.append({"seed": stage["label"], "stage": stage["folder"], "energy_ev_per_atom": energy / count,
                           "magnetization": result.get("magnetization")})
    ranked.sort(key=lambda row: row["energy_ev_per_atom"])
    for row in ranked:
        row["relative_energy_mev_per_atom"] = 1000 * (row["energy_ev_per_atom"] - ranked[0]["energy_ev_per_atom"])
    finished = all(stage.get("result") is not None for stage in state["stages"])
    state["comparison"].update(
        status="pending" if not finished else "partial" if excluded else "complete",
        ranking=ranked, excluded=excluded, comparison_available=len(ranked) >= 2,
        lowest_energy_seed=ranked[0]["seed"] if len(ranked) >= 2 else None,
    )


def _collect(root, state, transport, stage):
    remote = state["remote_root"] + "/" + stage["folder"]
    output = root / stage["folder"] / "outputs"
    output.mkdir(exist_ok=True)
    script = "from pathlib import Path;import json;root=Path(" + repr(remote) + ");print(json.dumps([n for n in " + repr(list(FILES)) + " if (root/n).is_file()]))"
    names = json.loads(_command(transport, "python3 -c " + shlex.quote(script)))
    if any(name not in FILES for name in names):
        raise ValueError("Unexpected remote artifact")
    if hasattr(transport, "download_many"):
        manifest = transport.download_many(remote, output, names)
    else:
        manifest = {name: transport.download(remote + "/" + name, output / name) for name in names}
    _write(output / "artifact_manifest.json", manifest)
    metadata_file = root / stage["folder"] / "inputs" / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        charge_index = stage.get("charge_from", state["current_stage"] - 1 if state["current_stage"] else None)
        if stage["name"] in {"bands", "dos"} and charge_index is not None:
            previous_result = state["stages"][charge_index].get("result") or {}
            metadata["scf_fermi_energy_ev"] = previous_result.get("fermi_energy_ev")
        _write(output / "metadata.json", metadata)
    expected = root / stage["folder"] / "inputs" / "POSCAR"
    result = analyze_outputs(output, stage["name"], expected)
    checks = {}
    hashes_file = output / "input_hashes.sha256"
    if hashes_file.exists():
        recorded = {line.split()[-1]: line.split()[0] for line in hashes_file.read_text().splitlines() if len(line.split()) >= 2}
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            digest = stage["metadata"]["input_sha256"][name]
            checks[name] = recorded.get(name) == digest
    result["input_identity_verified"] = len(checks) == 3 and all(checks.values())
    result["scheduler_state"] = stage.get("scheduler_state")
    parser_success = bool(result.get("success"))
    parser_reason = result.get("reason", "")
    reasons = [] if parser_success else [parser_reason] if parser_reason else []
    if not result["input_identity_verified"]:
        result["success"] = False
        reasons.append("Input checksums are missing or do not match the approved local inputs")
    if stage.get("scheduler_state") != "COMPLETED":
        result["success"] = False
        reasons.append(f"Slurm ended in {stage.get('scheduler_state')}; any structure is partial")
    if not result.get("success"):
        result["parser_reason"] = parser_reason
        diagnostic = _failure_diagnostic(output, stage.get("scheduler_state"))
        if diagnostic:
            reasons.insert(0, diagnostic.pop("message"))
            result.update(diagnostic)
        else:
            result["failure_code"] = "INPUT_IDENTITY_MISMATCH" if not result["input_identity_verified"] else "RESULT_REJECTED"
            result["recovery_hint"] = "Check the error message, input checksums and saved logs. If you need to change the calculation settings, prepare a new run."
        result["reason"] = " ".join(reasons)
    final_structure = output / "final_structure.cif"
    if result.get("success") and final_structure.is_file():
        result["final_structure_sha256"] = _digest(final_structure)
    _write(output / "result.json", result)
    stage["result"] = result
    stage["status"] = "succeeded" if result.get("success") else "needs_attention"
    bands = next((item for item in state["stages"] if item["name"] == "bands" and item["status"] == "succeeded"), None)
    dos = next((item for item in state["stages"] if item["name"] == "dos" and item["status"] == "succeeded"), None)
    if bands and dos and bands.get("charge_from") == dos.get("charge_from"):
        from .vasp import export_bands_dos
        try:
            artifacts = export_bands_dos(root / bands["folder"] / "outputs", root / dos["folder"] / "outputs")
            bands["result"]["artifacts"] = list(dict.fromkeys([*bands["result"].get("artifacts", []), *artifacts]))
            _write(root / bands["folder"] / "outputs" / "result.json", bands["result"])
        except (OSError, ValueError, KeyError) as exc:
            stage.setdefault("analysis_warnings", []).append("Combined plot unavailable: " + str(exc))
    _update_comparison(state)
    if (result.get("success") or "comparison" in state) and state["current_stage"] + 1 < len(state["stages"]):
        state["current_stage"] += 1
        state["status"] = "planned"
    else:
        state["status"] = stage["status"]
        if "comparison" in state and state["comparison"]["excluded"]:
            state["status"] = "needs_attention"
    return _save(root, state, f"Collected {stage['name']}: {result.get('reason', stage['status'])}")


def advance(run_dir, transport=None):
    if read_state(run_dir).get("workflow") == "pairing":
        from .qe_pairing_workflow import advance as advance_pairing
        return advance_pairing(run_dir, transport=transport)
    if read_state(run_dir).get("engine") == "qe":
        from .qe_workflow import advance as advance_qe
        return advance_qe(run_dir, transport=transport)
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        config = ClusterConfig.load(root / "config.json")
        owned = transport is None
        transport = transport or SSHTransport(config)
        stage = state["stages"][state["current_stage"]]
        try:
            _check_config(root, state)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                return _submit(root, state, config, transport)
            scheduler = _poll(transport, stage)
            stage["scheduler_state"] = scheduler
            state["last_error"] = None
            if state["cancel_requested"] and scheduler not in SLURM_TERMINAL:
                cancellation = transport.run("scancel " + stage["job_id"])
                if cancellation.returncode:
                    # Jobs can leave the live queue between poll and scancel.
                    # Keep the request and reconcile accounting next tick.
                    state["last_error"] = cancellation.stderr or cancellation.stdout
                    state["cancel_failures"] = state.get("cancel_failures", 0) + 1
                    if state["cancel_failures"] >= MAX_REMOTE_FAILURES:
                        state["status"] = "needs_attention"
                        return _save(root, state, "Slurm has not confirmed cancellation. The original job details have been saved.")
                else:
                    state["cancel_failures"] = 0
            if scheduler == "CANCELLED":
                stage["status"] = state["status"] = "cancelled"
                state["remote_failures"] = 0
                return _save(root, state, "Slurm confirmed cancellation")
            if scheduler in SLURM_TERMINAL:
                stage["status"] = state["status"] = "collecting"
                _save(root, state)
                result = _collect(root, state, transport, stage)
                result["remote_failures"] = 0
                return _save(root, result)
            if scheduler == "UNKNOWN":
                state["last_error"] = "Waiting for Slurm/accounting to report this job; no resubmission performed"
            else:
                stage["status"] = state["status"] = "queued" if scheduler == "PENDING" else "running"
            state["remote_failures"] = 0
            return _save(root, state)
        except TransportError as exc:
            state["last_error"] = str(exc)
            state["remote_failures"] = state.get("remote_failures", 0) + 1
            # A lost reply is not evidence that dispatch failed. Keep identity and
            # stage so the next tick can reconnect/reconcile the same request.
            if state["remote_failures"] >= MAX_REMOTE_FAILURES or "outcome uncertain" in str(exc) or "did not acknowledge" in str(exc):
                state["status"] = "needs_attention"
            return _save(root, state, "Remote operation failed. Run details have been saved.")
        except (ValueError, OSError, KeyError) as exc:
            state["status"] = "needs_attention"
            state["last_error"] = str(exc)
            return _save(root, state, "Paused. Review the error before continuing.")
        finally:
            if owned and hasattr(transport, "close"):
                transport.close()


def resume(run_dir, transport=None):
    """Resume without changing inputs or resubmitting.

    Unconverged output can be recollected; changed settings require a new run.
    """
    if read_state(run_dir).get("workflow") == "pairing":
        from .qe_pairing_workflow import resume as resume_pairing
        return resume_pairing(run_dir, transport=transport)
    if read_state(run_dir).get("engine") == "qe":
        from .qe_workflow import resume as resume_qe
        return resume_qe(run_dir, transport=transport)
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] not in {"needs_attention", "failed"}:
            return state
        _check_config(root, state)
        stage = state["stages"][state["current_stage"]]
        owned = transport is None
        try:
            if not stage.get("job_id") and stage.get("staged"):
                config = ClusterConfig.load(root / "config.json")
                transport = transport or SSHTransport(config)
                remote = state["remote_root"] + "/" + stage["folder"]
                script = remote + "/reconcile.py"
                transport.upload(Path(__file__).with_name("dispatch.py"), script)
                reply = _command(transport, f"python3 {shlex.quote(script)} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only", timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if not str(receipt["job_id"]).isdigit():
                    raise ValueError("Invalid reconciled job identity")
                stage["job_id"] = str(receipt["job_id"])
            if stage.get("job_id"):
                stage["status"] = state["status"] = "queued"
            elif state.get("cancel_requested"):
                stage["status"] = state["status"] = "cancelled"
            else:
                # staged=False means dispatch has never been called.
                stage["status"] = state["status"] = "planned"
            state["last_error"] = None
            state["remote_failures"] = 0
            state["cancel_failures"] = 0
            return _save(root, state, "Resumed this run with its original inputs.")
        except (TransportError, ValueError, OSError, KeyError) as exc:
            state["last_error"] = str(exc)
            return _save(root, state, "Could not reconnect to this run. No new job was submitted.")
        finally:
            if owned and transport is not None:
                transport.close()


def cancel(run_dir, transport=None):
    if read_state(run_dir).get("workflow") == "pairing":
        from .qe_pairing_workflow import cancel as cancel_pairing
        return cancel_pairing(run_dir, transport=transport)
    if read_state(run_dir).get("engine") == "qe":
        from .qe_workflow import cancel as cancel_qe
        return cancel_qe(run_dir, transport=transport)
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in {"succeeded", "cancelled"}:
            return state
        state["cancel_requested"] = True
        stage = state["stages"][state["current_stage"]]
        if not stage.get("job_id") and not stage.get("staged"):
            stage["status"] = state["status"] = "cancelled"
        elif not stage.get("job_id"):
            state["status"] = "needs_attention"
            state["last_error"] = "Submission acknowledgement missing. Reconcile remote job before cancellation; do not start a new run."
        elif state["status"] in {"needs_attention", "failed"}:
            stage["status"] = state["status"] = "queued"
            state["remote_failures"] = state["cancel_failures"] = 0
        _save(root, state, "Cancellation requested")
    if stage.get("job_id"):
        return advance(root, transport)
    return state


def watch(run_dir, interval=20):
    root = Path(run_dir).expanduser().resolve()
    if not has_config(root):
        raise ValueError("This run has no cluster settings yet. Save Cluster setup and attach it before submitting.")
    with (root / ".worker.lock").open("a") as lock:
        try:
            _acquire(lock, blocking=False)
        except BlockingIOError:
            return read_state(root)
        transport = None
        try:
            _write(root / "worker.json", {"pid": os.getpid(), "started_at": utc_now()})
            transport = SSHTransport(ClusterConfig.load(root / "config.json"))
            while True:
                state = advance(root, transport)
                if state["status"] in TERMINAL:
                    bundle_run(root)
                    _announce(root, state)
                    return state
                time.sleep(interval)
        finally:
            try:
                if transport is not None:
                    transport.close()
            finally:
                _release(lock)


def _announce(root, state):
    """Tell the user a run finished. Notification problems never affect the run."""
    try:
        from .notify import notify_run
        notify_run(state, str(root))
    except Exception:
        pass


def start_worker(run_dir):
    from .runtime import cli_command
    root = Path(run_dir).expanduser().resolve()
    # Worker lock, rather than an unverified PID file, owns singleton execution.
    with (root / "worker.log").open("ab") as stream:
        child = subprocess.Popen(cli_command("watch", root), stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
    return child.pid


def bundle_run(run_dir):
    if read_state(run_dir).get("workflow") == "pairing":
        from .qe_pairing_workflow import bundle_run as bundle_pairing
        return bundle_pairing(run_dir)
    if read_state(run_dir).get("engine") == "qe":
        from .qe_workflow import bundle_run as bundle_qe
        return bundle_qe(run_dir)
    root = Path(run_dir).expanduser().resolve()
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "run.json", "run.json")
        if (root / "proposal.json").is_file():
            archive.write(root / "proposal.json", "proposal.json")
        if (root / "parent.json").is_file():
            archive.write(root / "parent.json", "parent.json")
        if (root / "explanations.json").is_file():
            archive.write(root / "explanations.json", "explanations.json")
        if (root / "repair.json").is_file():
            archive.write(root / "repair.json", "repair.json")
        if (root / "structure_edit").is_dir():
            for file in sorted((root / "structure_edit").iterdir()):
                if file.is_file() and not file.is_symlink() and file.name in {"source.cif", "source.vasp", "edited.cif", "POSCAR", "structure.json"}:
                    archive.write(file, str(file.relative_to(root)))
        if (root / "plan.json").is_file():
            archive.write(root / "plan.json", "plan.json")
            for source in sorted((root / "source").rglob("*")):
                if (source.is_file() and not source.is_symlink()
                        and not any(parent.is_symlink() for parent in source.parents)):
                    archive.write(source, str(source.relative_to(root)))
        for stage in read_state(root)["stages"]:
            base = root / stage["folder"]
            for folder in (base / "inputs", base / "outputs"):
                if folder.exists():
                    for file in sorted(folder.rglob("*")):
                        if (file.is_file() and not file.is_symlink() and file.name != "POTCAR"
                                and not any(part.startswith(".") for part in file.relative_to(folder).parts)
                                and not any(parent.is_symlink() for parent in file.parents if parent != root)):
                            archive.write(file, file.relative_to(root).as_posix())
    os.replace(target.with_suffix(".tmp"), target)
    return target
