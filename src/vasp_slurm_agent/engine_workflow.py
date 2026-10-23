"""Durable ABINIT, CP2K and Elk runs with verified inputs, libraries and checkpoints.

The flow matches the Quantum ESPRESSO workflow: a plan is frozen locally,
each stage is materialized from accepted parent results, a small stdlib
runtime on the cluster checks every input, library file and inherited
checkpoint by SHA-256 before the program starts, and only hashed outputs
come back. Submission happens only through advance().
"""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tempfile
import uuid
import zipfile

from .config import ClusterConfig
from .scheduler import get_scheduler, shell_command
from .transport import SSHTransport, TransportError
from .workflow import (_lock, _save, _write, _digest, _command, _poll, _scheduler_flag,
                       _check_config, _config_attachment_lock,
                       _config_attachment_workspace, _commit_config_attachment,
                       read_state, utc_now, TERMINAL, SLURM_TERMINAL, MAX_REMOTE_FAILURES)

ENGINES = ("abinit", "cp2k", "elk")
LABELS = {"abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk"}
PROGRAMS = {"abinit": frozenset({"abinit"}), "cp2k": frozenset({"cp2k.psmp", "cp2k.popt", "cp2k.ssmp", "cp2k.sopt", "cp2k"}),
            "elk": frozenset({"elk"})}
RECEIPTS = frozenset({"execution.json", "input_hashes.sha256", "library_manifest.json", "checkpoint.json",
                      "runtime.json", "slurm.out", "slurm.err"})
_SHELL = {";", "&&", "||", "|", ">", "<", "&", "2>", ">>"}


def adapter(engine):
    if engine not in ENGINES:
        raise ValueError("Choose ABINIT, CP2K or Elk.")
    return import_module(f".{engine}", __package__)


# Every calculation offered by some program; each adapter's TASKS says which ones it runs.
TASKS = tuple(dict.fromkeys(task for engine in ENGINES for task in adapter(engine).TASKS))
# These stages compute their own electronic ground state; every other one continues an SCF.
STANDALONE = {"relax", "scf", "md"}


def _name(value):
    if not isinstance(value, str) or not value or PurePosixPath(value).name != value or value in {".", ".."} or any(c in value for c in "\r\n\x00"):
        raise ValueError("Stage files must be plain file names.")
    return value


def _require_cpu(config):
    from .engine_setup import validate_managed_resources

    if config is not None and config.compute_backend != "cpu":
        raise ValueError("ABINIT, CP2K and Elk run on CPUs in this version. Select CPU.")
    validate_managed_resources(config)


def _command_for(config, metadata):
    """The environment's launch command with this stage's arguments appended."""
    engine = metadata["engine"]
    command = config.engine_command.replace("{tasks}", str(config.tasks))
    tokens = shlex.split(command)
    matches = [index for index, token in enumerate(tokens) if PurePosixPath(token).name in PROGRAMS[engine]]
    if len(matches) != 1 or matches[0] != len(tokens) - 1 or any(token in _SHELL for token in tokens):
        raise ValueError(f"The {LABELS[engine]} command must end with the {LABELS[engine]} program, optionally after srun, mpirun or mpiexec.")
    if any(token in {"-i", "-o", "--input", "--output"} for token in tokens):
        raise ValueError("Leave input and output options out of the launch command; each stage sets them.")
    if "{" in command or "}" in command:
        raise ValueError("Only the {tasks} placeholder is supported in the launch command.")
    return shlex.join([*tokens, *metadata.get("arguments", [])])


def _sequence(tasks, parameters, stage_parameters, engine=None):
    offered = adapter(engine).TASKS if engine else TASKS
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("Choose at least one calculation.")
    if stage_parameters is not None and (len(stage_parameters) != len(tasks) or any(not isinstance(p, dict) for p in stage_parameters)):
        raise ValueError("Supply one parameter override per task.")
    # moved: the last relaxation or molecular dynamics; a later MD continues from its final frame.
    sequence, scf, moved = [], None, None

    def add(name, recipe, restart=None, structure=None):
        sequence.append({"name": name, "parameters": deepcopy(recipe), "structure_from": moved if structure is None else structure,
                         "restart_from": restart, "request_index": index})
        return len(sequence) - 1

    def charge(recipe):
        nonlocal scf
        # Iteration budgets and spectrum-only controls do not change a converged
        # SCF density. Keep physical settings and tolerances identical.
        spectral = {"max_scf_steps", "line_density", "kpoints", "dos_mesh", "dos_broadening_ev",
                    *getattr(adapter(engine), "STAGE_KEYS", ())} if engine else set()
        comparable = lambda values: {key: value for key, value in values.items() if key not in spectral}
        if scf is None or comparable(sequence[scf]["parameters"]) != comparable(recipe):
            scf = add("scf", recipe)
        return scf

    for index, task in enumerate(tasks):
        if task not in offered:
            raise ValueError(f"Unsupported task: {task}. Choose {', '.join(offered)}.")
        recipe = {**deepcopy(parameters), **deepcopy((stage_parameters or [{}] * len(tasks))[index])}
        if task == "relax":
            if sequence:
                raise ValueError("Put relaxation first in the plan.")
            moved = add(task, recipe)
            scf = None
        elif task == "md":
            moved = add(task, recipe, structure=moved)
            scf = None
        elif task in STANDALONE:
            current = add(task, recipe)
            scf = current if task == "scf" else scf
        else:
            add(task, recipe, charge(recipe))
    return sequence


def prepare_plan(structure_path, run_dir, config, tasks, parameters=None, stage_parameters=None, *, engine=None, **kwargs):
    """Freeze a plan locally. Only advance() can submit it."""
    engine = engine or (config.engine if config is not None else None)
    adapter(engine)
    if config is not None and config.engine != engine:
        raise ValueError(f"Select a {LABELS[engine]} environment for this plan.")
    _require_cpu(config)
    source = Path(structure_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(run_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This run already exists. Resume it or choose a new folder.")
    parameters = deepcopy(parameters or {})
    sequence = _sequence(tasks, parameters, stage_parameters, engine)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{engine}-plan-", dir=root.parent))
    try:
        (temporary / "source").mkdir()
        relative = "source/structure" + (source.suffix or ".vasp")
        shutil.copyfile(source, temporary / relative)
        sources = {relative: _digest(temporary / relative)}
        run_id = f"{engine}-" + uuid.uuid4().hex[:16]
        stages = [{**spec, "source_path": relative, "folder": f"{index + 1:02d}_{spec['name']}",
                   "job_name": f"{run_id}-{index + 1:02d}-{spec['name']}", "job_id": None, "status": "planned",
                   "metadata": {}, "staged": False, "materialized": False, "result": None}
                  for index, spec in enumerate(sequence)]
        plan = {"schema_version": 1, "engine": engine, "tasks": list(tasks), "parameters": deepcopy(parameters), "sources": sources,
                "stages": [{key: deepcopy(stage[key]) for key in
                            ("name", "parameters", "source_path", "folder", "structure_from", "restart_from", "request_index")}
                           for stage in stages]}
        if stage_parameters is not None:
            plan["stage_parameters"] = deepcopy(stage_parameters)
        _write(temporary / "plan.json", plan)
        if config:
            config.save(temporary / "config.json")
        state = {"schema_version": 2, "engine": engine, "run_id": run_id,
                 "task": tasks[0] if len(tasks) == 1 else "pipeline", "tasks": list(tasks),
                 "parameters": parameters, "formula": "", "status": "planned", "current_stage": 0,
                 "stages": stages, "history": [], "created_at": utc_now(), "last_error": None,
                 "cancel_requested": False, "plan_sha256": _digest(temporary / "plan.json"),
                 "config_sha256": _digest(temporary / "config.json") if config else None,
                 "remote_root": config.remote_root.rstrip("/") + "/" + run_id if config else None}
        # Check every recipe now; dependent inputs are rebuilt from accepted results.
        for stage in stages:
            _materialize(temporary, state, config, stage, preview=stage["structure_from"] is not None)
        state["formula"] = stages[0]["metadata"].get("formula", "")
        _save(temporary, state, f"{LABELS[engine]} plan prepared. Ready to submit.")
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _materialize(root, state, config, stage, preview=False):
    if stage["materialized"]:
        return
    science = adapter(state["engine"])
    source = root / stage["source_path"]
    if stage["structure_from"] is not None and not preview:
        parent = state["stages"][stage["structure_from"]]
        result = parent.get("result") or {}
        if parent["status"] != "succeeded" or not result.get("success"):
            raise ValueError("The relaxed structure has not been accepted.")
        source = root / parent["folder"] / "outputs" / "final.cif"
        if not source.is_file() or _digest(source) != result.get("final_structure_sha256"):
            raise ValueError("The accepted relaxed structure changed or is missing.")
    inputs = root / stage["folder"] / ("preview" if preview else "inputs")
    if inputs.exists():
        shutil.rmtree(inputs)
    metadata = science.prepare_inputs(source, inputs, stage["name"], deepcopy(stage["parameters"]),
                                      config.engine_library if config else None)
    for key in ("input_file", "output_file", "stdout_file"):
        _name(metadata[key])
    for name in metadata["library_files"]:
        _name(name)
    metadata["input_sha256"] = {_name(name): _digest(inputs / name) for name in metadata["input_files"]}
    if config:
        _command_for(config, metadata)
    if preview:
        shutil.rmtree(inputs)
        return
    stage["metadata"] = metadata
    stage["materialized"] = True
    _write(inputs / "metadata.json", metadata)


def attach_config(run_dir, config):
    root = Path(run_dir).expanduser().resolve()
    engine = read_state(root).get("engine")
    if config is None or config.engine != engine:
        raise ValueError(f"Save {LABELS.get(engine, 'matching')} cluster settings first.")
    _require_cpu(config)
    with _config_attachment_lock(root):
        original = read_state(root)
        _check_config(root, original)
        if original["status"] != "planned" or any(s.get("staged") or s.get("job_id") for s in original["stages"]):
            raise ValueError("Cluster settings can only be attached before submission.")
        with _config_attachment_workspace(root) as temporary:
            prepared = temporary / "run"
            shutil.copytree(root, prepared, ignore=shutil.ignore_patterns(".state.lock", ".worker.lock"))
            state = read_state(prepared)
            config.save(prepared / "config.json")
            state.update(config_sha256=_digest(prepared / "config.json"),
                         remote_root=config.remote_root.rstrip("/") + "/" + state["run_id"])
            for stage in state["stages"]:
                stage["materialized"] = False
                stage["metadata"] = {}
                _materialize(prepared, state, config, stage, preview=stage["structure_from"] is not None)
            _save(prepared, state, f"{LABELS[engine]} cluster settings attached.")
            _commit_config_attachment(root, prepared,
                ["config.json", *(s["folder"] + "/inputs" for s in state["stages"]), "run.json"],
                temporary / "original")
            return state


# This stdlib program runs on the cluster. Checkpoints stay there; only hashes,
# small scientific outputs and requested raw files cross SSH.
_RUNTIME = r'''import datetime, hashlib, json, os, shutil, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def safe(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise RuntimeError('Unsafe path: ' + name)
    if root.resolve() not in path.resolve().parents:
        raise RuntimeError('Path escaped its folder: ' + name)
    return path

root = Path.cwd()
spec = json.loads((root/'runtime.json').read_text())
if sys.argv[1] == 'prepare':
    for name, expected in spec['input_sha256'].items():
        if sha(safe(root, name)) != expected: raise RuntimeError('Prepared input changed: '+name)
    (root/'.execution_started').open('x').close()
    (root/'input_hashes.sha256').write_text(''.join(h+'  '+n+'\n' for n,h in spec['input_sha256'].items()))
    for dep in spec['dependencies']:
        source = Path(dep['directory'])
        receipt = source/'checkpoint.json'
        if sha(receipt) != dep['sha256']: raise RuntimeError('Accepted checkpoint receipt changed')
        records = json.loads(receipt.read_text())
        for name, destination in dep['mapping'].items():
            if name not in records: raise RuntimeError('Checkpoint member is missing: '+name)
            value = records[name]
            old = safe(source, name)
            if old.stat().st_size != value['size'] or sha(old) != value['sha256']:
                raise RuntimeError('Accepted checkpoint changed: '+name)
            new = safe(root, destination)
            if new.exists(): raise RuntimeError('Checkpoint would overwrite a prepared file: '+destination)
            shutil.copy2(old, new)
    library = safe(root, spec['library_dir'])
    library.mkdir(exist_ok=True)
    for name, value in spec['library'].items():
        old = safe(Path(spec['library_root']), name)
        if old.stat().st_size != value['size'] or sha(old) != value['sha256']:
            raise RuntimeError('Library file changed since it was checked: '+name)
        shutil.copyfile(old, library/name)
    (root/'execution.json').write_text(json.dumps(dict(job_id=os.environ.get('PBS_JOBID' if spec.get('scheduler') == 'pbs' else 'SLURM_JOB_ID'),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime_sha256=sha(root/'runtime.json'), engine=spec['engine'])))
else:
    code = int(sys.argv[2])
    receipt = json.loads((root/'execution.json').read_text())
    receipt.update(exit_code=code, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/'execution.json').write_text(json.dumps(receipt))
    manifest = {}
    for name in spec['checkpoints']:
        path = root/name
        if path.is_file() and not path.is_symlink():
            path = safe(root, name)
            manifest[name] = dict(size=path.stat().st_size, sha256=sha(path))
    (root/'checkpoint.json').write_text(json.dumps(manifest, sort_keys=True))
'''


def _library_snapshot(transport, config, stage):
    """Hash and header-check the library files on the cluster before upload."""
    metadata = stage["metadata"]
    science = adapter(metadata["engine"])
    entries = metadata["library_files"]
    if not entries:
        return {}
    root = config.engine_library_root
    code = "import hashlib,json,re\nfrom pathlib import Path\n" + science.LIBRARY_VALIDATOR + """
root=Path(%r)
result,missing={},[]
for name,entry in %r.items():
    path=root/name
    if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
        missing.append(name)
        continue
    if path.stat().st_size > 200*1024*1024:
        raise RuntimeError('Library file is unexpectedly large: '+name)
    data=path.read_bytes()
    validate_library(data, entry)
    result[name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}
print(json.dumps({'files':result,'missing':missing}))
""" % (root, entries)
    reply = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    if reply.get("missing"):
        raise ValueError(f"The {LABELS[metadata['engine']]} library folder lacks " + ", ".join(sorted(reply["missing"]))
                         + ". Install the library files from the step card or Connections, then submit again.")
    result = reply.get("files")
    if not isinstance(result, dict) or (set(result) != set(entries) or any(not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
                                           or type(value.get("size")) is not int or value["size"] <= 0 for value in result.values())):
        raise ValueError("Invalid library receipt.")
    recorded = config.engine_library or {}
    for name, entry in entries.items():
        allowed = {entry.get("sha256")}
        if isinstance(recorded.get(name), dict) and recorded[name].get("sha256"):
            allowed.add(recorded[name]["sha256"])
        if result[name]["sha256"] not in allowed - {None}:
            raise ValueError(f"The cluster's {name} differs from the pinned {LABELS[metadata['engine']]} library. Reinstall the library from Connections.")
    return result


def _companions(config, metadata):
    """Shell lines that run the adapter's post-processing programs, from the same installation and on one
    process, each only while the steps before it succeeded."""
    from .qe_workflow import _one_rank

    science = adapter(metadata["engine"])
    tokens = shlex.split(config.engine_command.replace("{tasks}", str(config.tasks)))
    index = len(tokens) - 1
    lines = []
    for step in metadata.get("post_commands", []):
        program, arguments, stdout, stdin = step.get("program"), step.get("arguments", []), step.get("stdout"), step.get("stdin")
        names = [*arguments, stdout, *([stdin] if stdin is not None else [])]
        if program not in getattr(science, "COMPANIONS", ()) or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
            raise ValueError("The stage asks for a post-processing step this program does not provide.")
        binary = PurePosixPath(tokens[index])
        command = _one_rank([*tokens[:index], str(binary.with_name(program)) if len(binary.parts) > 1 else program], index)
        redirect = (f" < {shlex.quote(stdin)}" if stdin is not None else "") + f" > {shlex.quote(stdout)} 2>&1"
        lines += ['if [ "$code" -eq 0 ]; then', f"  {shlex.join([*command, *arguments])}{redirect}", "  code=$?", "fi"]
    return lines


def _script(config, stage):
    metadata = stage["metadata"]
    launch = _command_for(config, metadata) + " > " + shlex.quote(metadata["stdout_file"])
    adapter_ = get_scheduler(config)
    lines = adapter_.header(config)
    science = adapter(metadata["engine"])
    lines.extend(["set -e", *adapter_.working_directory(), *config.setup_commands,
                  f"export OMP_NUM_THREADS={config.cpus_per_task}", *getattr(science, "JOB_ENVIRONMENT", ()),
                  "python3 engine_runtime.py prepare", "set +e",
                  launch, "code=$?", *_companions(config, metadata), "set -e", 'python3 engine_runtime.py finish "$code"', 'exit "$code"'])
    return "\n".join(lines) + "\n"


def _upload_names(metadata):
    return frozenset({*metadata["input_files"], "library_manifest.json", "runtime.json", "engine_runtime.py",
                      "submit.sh", "dispatch.py", "scheduler.py"})


def _validate_checkpoint(records, names):
    if not isinstance(records, dict):
        raise ValueError("The checkpoint receipt is invalid.")
    for name, value in records.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or not isinstance(value, dict)
                or type(value.get("size")) is not int or value["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
            raise ValueError("The checkpoint receipt is invalid.")
    missing = [name for name in names if not records.get(name, {}).get("size")]
    if missing:
        raise ValueError("The required checkpoint is missing: " + ", ".join(missing) + ".")


def _submit(root, state, config, transport):
    _require_cpu(config)
    stage = state["stages"][state["current_stage"]]
    _materialize(root, state, config, stage)
    science = adapter(state["engine"])
    metadata = stage["metadata"]
    inputs = root / stage["folder"] / "inputs"
    for name, digest in metadata["input_sha256"].items():
        if _digest(inputs / name) != digest:
            raise ValueError(f"Prepared {name} changed. Prepare a new run.")
    dependencies = []
    parent_index = stage.get("restart_from")
    if parent_index is not None:
        parent = state["stages"][parent_index]
        result = parent.get("result") or {}
        if parent["status"] != "succeeded" or not result.get("success") or not result.get("checkpoint_sha256"):
            raise ValueError("The required checkpoint has not been accepted.")
        checkpoint = root / parent["folder"] / "outputs" / "checkpoint.json"
        if not checkpoint.is_file() or _digest(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("The accepted checkpoint receipt changed.")
        mapping = science.restart_mapping(stage["name"], metadata)
        for name, destination in mapping.items():
            _name(name)
            _name(destination)
        _validate_checkpoint(json.loads(checkpoint.read_text(encoding="utf-8")), list(mapping))
        dependencies.append({"directory": state["remote_root"] + "/" + parent["folder"], "sha256": result["checkpoint_sha256"],
                             "mapping": mapping})
    remote = state["remote_root"] + "/" + stage["folder"]
    if stage["status"] == "planned":
        stage["status"] = state["status"] = "submitting"
        _save(root, state, f"Submitting {LABELS[state['engine']]} {stage['name']}")
    if not stage["staged"]:
        _command(transport, " && ".join("command -v " + name + " >/dev/null" for name in (*get_scheduler(config).required_commands, "python3")))
        library = _library_snapshot(transport, config, stage)
        _write(inputs / "library_manifest.json", library)
        runtime = {"engine": state["engine"], "input_sha256": metadata["input_sha256"], "dependencies": dependencies,
                   "library_root": config.engine_library_root, "library_dir": metadata["library_dir"], "library": library,
                   "checkpoints": list(science.checkpoint_names(stage["name"], metadata))}
        if config.scheduler != "slurm":
            runtime["scheduler"] = config.scheduler
        _write(inputs / "runtime.json", runtime)
        stage["runtime_sha256"] = _digest(inputs / "runtime.json")
        stage["library_manifest_sha256"] = _digest(inputs / "library_manifest.json")
        (inputs / "engine_runtime.py").write_text(_RUNTIME, encoding="utf-8")
        (inputs / "submit.sh").write_text(_script(config, stage), encoding="utf-8", newline="\n")
        names = [*metadata["input_sha256"], "library_manifest.json", "runtime.json", "engine_runtime.py", "submit.sh"]
        transport.upload_many([*(inputs / name for name in names), Path(__file__).with_name("dispatch.py"), Path(__file__).with_name("scheduler.py")],
                              remote, permitted_names=_upload_names(metadata))
        stage["staged"] = True
        _save(root, state, f"Uploaded {LABELS[state['engine']]} inputs.")
    reply = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])}{_scheduler_flag(config)}", timeout=100)
    receipt = json.loads(reply.splitlines()[-1])
    if receipt.get("job_name") != stage["job_name"] or not get_scheduler(config).valid_job_id(str(receipt.get("job_id", ""))):
        raise ValueError("Invalid scheduler submission receipt.")
    stage.update(job_id=str(receipt["job_id"]), status="queued", submitted_at=utc_now())
    state.update(status="queued", last_error=None, remote_failures=0)
    return _save(root, state, f"{config.scheduler.upper()} accepted job {stage['job_id']}")


def _remote_files(transport, remote):
    code = "from pathlib import Path;import json;root=Path(" + repr(remote) + ");print(json.dumps([p.name for p in root.iterdir() if p.is_file() and not p.is_symlink()]))"
    return [_name(name) for name in json.loads(_command(transport, "python3 -c " + shlex.quote(code)))]


def _collect(root, state, transport, stage):
    remote = state["remote_root"] + "/" + stage["folder"]
    output = root / stage["folder"] / "outputs"
    metadata = stage["metadata"]
    allowed = {*RECEIPTS, *metadata["input_sha256"], metadata["output_file"], metadata["stdout_file"], *metadata.get("output_files", [])}
    names = [name for name in _remote_files(transport, remote) if name in allowed]
    with tempfile.TemporaryDirectory(prefix=f".{state['engine']}-collect-", dir=output.parent) as directory:
        collected = Path(directory) / "outputs"
        collected.mkdir()
        manifest = transport.download_many(remote, collected, names, permitted_names=frozenset(names))
        if not isinstance(manifest, dict) or set(manifest) != set(names):
            raise ValueError("The collection receipt does not match the downloaded files.")
        for name, receipt in manifest.items():
            path = collected / name
            if (path.is_symlink() or not path.resolve().is_relative_to(collected.resolve()) or not path.is_file()
                    or not isinstance(receipt, dict) or type(receipt.get("size")) is not int
                    or path.stat().st_size != receipt["size"] or _digest(path) != receipt.get("sha256")):
                raise ValueError("A downloaded file does not match its collection receipt.")
        _write(collected / "artifact_manifest.json", manifest)
        shutil.copyfile(root / stage["folder"] / "inputs" / "metadata.json", collected / "metadata.json")
        result = _analyze_collection(root, state, stage, collected, manifest)
        _write(collected / "result.json", result)
        previous = Path(directory) / "previous"
        if output.exists():
            os.replace(output, previous)
        try:
            os.replace(collected, output)
        except OSError:
            if previous.exists():
                os.replace(previous, output)
            raise
    stage.update(result=result, status="succeeded" if result["success"] else "needs_attention")
    if result["success"] and not state["cancel_requested"] and state["current_stage"] + 1 < len(state["stages"]):
        state["current_stage"] += 1
        state["status"] = "planned"
    else:
        state["status"] = "cancelled" if state["cancel_requested"] else stage["status"]
    return _save(root, state, f"Collected {LABELS[state['engine']]} {stage['name']}: {stage['status']}")


_SIGNALS = {4: "illegal instruction", 6: "abort", 7: "bus error", 8: "floating-point exception", 9: "kill",
            11: "segmentation fault", 15: "termination"}
_CRASH = re.compile(r"double free|corrupted|free\(\)|munmap_chunk|segmentation fault|sigsegv|sigabrt|"
                    r"program received signal|stack smashing|bus error", re.I)


def _exit_reason(engine, code, complete, output):
    signal = _SIGNALS.get(code - 128) if 128 < code < 160 else None
    text = f"{LABELS[engine]} exited with code {code}" + (f" ({signal} signal)" if signal else "") + "."
    if not complete:
        return text
    error = output / "slurm.err"
    lines = error.read_text(errors="replace", encoding="utf-8").splitlines() if error.is_file() else []
    crash = next((line.strip() for line in lines if _CRASH.search(line)), "")
    detail = f' The error log reports "{crash[:160]}".' if crash else ""
    return (f"{text} Its output reports a finished calculation, so the program failed while shutting down.{detail} "
            f"Results from a crashing program are not accepted; this usually means the {LABELS[engine]} build or its "
            "MPI libraries are faulty. Use another build, such as the app's managed installation, or ask the cluster support.")


def _analyze_collection(root, state, stage, output, manifest):
    science = adapter(state["engine"])
    metadata = stage["metadata"]
    result = science.analyze_outputs(output, stage["name"], root / stage["folder"] / "inputs" / "structure.cif")
    reasons = [] if result.get("success") else [result.get("reason") or "The output was not accepted."]
    expected = metadata["input_sha256"]
    required = {*expected, "input_hashes.sha256", "execution.json", "runtime.json", "library_manifest.json", metadata["output_file"]}
    if not required <= set(manifest):
        reasons.append("Required collection files are missing: " + ", ".join(sorted(required - set(manifest))) + ".")
    recorded = {}
    if (output / "input_hashes.sha256").is_file():
        for line in (output / "input_hashes.sha256").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                recorded[fields[1]] = fields[0]
    identity = recorded == expected and set(expected) <= set(manifest) and all(
        (output / name).is_file() and _digest(output / name) == digest for name, digest in expected.items())
    result["input_identity_verified"] = identity
    if not identity:
        reasons.append("Input checksums do not match the approved inputs.")
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8")) if (output / "execution.json").is_file() else {}
    receipt_ok = (execution.get("job_id") == stage["job_id"] and type(execution.get("exit_code")) is int
                  and execution.get("runtime_sha256") == stage.get("runtime_sha256") and "runtime.json" in manifest
                  and _digest(output / "runtime.json") == stage.get("runtime_sha256") and bool(execution.get("finished_at")))
    executed = receipt_ok and execution["exit_code"] == 0
    result["execution_verified"] = executed
    if not executed:
        reasons.append(_exit_reason(state["engine"], execution["exit_code"], bool(result.get("success")), output)
                       if receipt_ok else "The job execution receipt is missing or inconsistent.")
    library_ok = (output / "library_manifest.json").is_file() and _digest(output / "library_manifest.json") == stage.get("library_manifest_sha256")
    result["library_verified"] = library_ok
    if not library_ok:
        reasons.append("The library receipt does not match this run.")
    result["scheduler_state"] = stage.get("scheduler_state")
    if stage.get("scheduler_state") != "COMPLETED":
        reasons.append(f"Scheduler ended in {stage.get('scheduler_state')}.")
    if result.get("success") and not reasons:
        checkpoint = output / "checkpoint.json"
        names = adapter(state["engine"]).checkpoint_names(stage["name"], metadata)
        try:
            records = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else None
            if names:
                _validate_checkpoint(records, list(names))
        except (ValueError, OSError) as exc:
            reasons.append(str(exc))
    result["success"] = not reasons
    if reasons:
        result["reason"] = " ".join(reasons)
    if result["success"]:
        if (output / "checkpoint.json").is_file():
            result["checkpoint_sha256"] = _digest(output / "checkpoint.json")
        if (output / "final.cif").is_file():
            result["final_structure_sha256"] = _digest(output / "final.cif")
        if (output / "CONTCAR").is_file():
            result["final_native_structure_sha256"] = _digest(output / "CONTCAR")
    return result


def advance(run_dir, transport=None, *, poll_only=False):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        owned = transport is None
        stage = state["stages"][state["current_stage"]]
        if poll_only and (not stage.get("job_id") or stage["status"] in {"planned", "submitting"}):
            return state
        try:
            _check_config(root, state)
            config = ClusterConfig.load(root / "config.json")
            transport = transport or SSHTransport(config)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                return _submit(root, state, config, transport)
            scheduler = _poll(transport, stage, config)
            stage["scheduler_state"] = scheduler
            if state["cancel_requested"] and scheduler not in SLURM_TERMINAL:
                _command(transport, shell_command(get_scheduler(config).cancel_command(stage["job_id"])))
            if scheduler == "CANCELLED":
                stage["status"] = state["status"] = "cancelled"
                return _save(root, state, "The scheduler confirmed cancellation.")
            if scheduler in SLURM_TERMINAL:
                stage["status"] = state["status"] = "collecting"
                _save(root, state)
                result = _collect(root, state, transport, stage)
                result["remote_failures"] = 0
                result["last_error"] = None
                return _save(root, result)
            if scheduler == "UNKNOWN":
                state["last_error"] = "Waiting for scheduler accounting; no resubmission performed."
            else:
                stage["status"] = state["status"] = "queued" if scheduler == "PENDING" else "running"
                state["last_error"] = None
            state["remote_failures"] = 0
            return _save(root, state)
        except TransportError as exc:
            state["last_error"] = str(exc)
            state["remote_failures"] = state.get("remote_failures", 0) + 1
            if state["remote_failures"] >= MAX_REMOTE_FAILURES or "outcome uncertain" in str(exc) or "did not acknowledge" in str(exc):
                state["status"] = "needs_attention"
            return _save(root, state, "Remote operation failed. Run details saved.")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            state.update(status="needs_attention", last_error=str(exc))
            return _save(root, state, "Paused. Review the error before continuing.")
        finally:
            if owned and transport is not None:
                transport.close()


def resume(run_dir, transport=None):
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
                reply = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only{_scheduler_flag(config)}", timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if receipt.get("job_name") != stage["job_name"] or not get_scheduler(config).valid_job_id(str(receipt.get("job_id", ""))):
                    raise ValueError("Invalid reconciled job identity.")
                stage["job_id"] = str(receipt["job_id"])
            status = "queued" if stage.get("job_id") else "cancelled" if state.get("cancel_requested") else "planned"
            if status == "planned" and not stage.get("staged"):
                stage["materialized"] = False
            stage["status"] = state["status"] = status
            state.update(last_error=None, remote_failures=0)
            return _save(root, state, f"Resumed the original {LABELS[state['engine']]} run.")
        except (TransportError, ValueError, OSError, KeyError) as exc:
            state["last_error"] = str(exc)
            return _save(root, state, "Reconnection failed. No new job submitted.")
        finally:
            if owned and transport is not None:
                transport.close()


def cancel(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in {"succeeded", "cancelled"}:
            return state
        stage = state["stages"][state["current_stage"]]
        state["cancel_requested"] = True
        if not stage.get("job_id") and not stage.get("staged"):
            stage["status"] = state["status"] = "cancelled"
        elif not stage.get("job_id"):
            state.update(status="needs_attention", last_error="Reconcile the original submission before cancellation.")
        else:
            stage["status"] = state["status"] = "queued"
        _save(root, state, "Cancellation requested.")
    return advance(root, transport) if stage.get("job_id") else state


def bundle_run(run_dir):
    """Bundle inputs, receipts and collected outputs, excluding credentials."""
    root = Path(run_dir).expanduser().resolve()
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("run.json", "plan.json", "proposal.json", "explanations.json", "conversation.json", "parent.json", "repair.json"):
            if (root / name).is_file() and not (root / name).is_symlink():
                archive.write(root / name, name)
        state = read_state(root)
        for path in (root / "source").rglob("*") if (root / "source").is_dir() else []:
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(root))
        for stage in state["stages"]:
            for folder in ("inputs", "outputs"):
                source = root / stage["folder"] / folder
                for path in source.rglob("*") if source.is_dir() else []:
                    if path.is_file() and not path.is_symlink() and path.name not in {"runtime.json", "submit.sh", "engine_runtime.py"}:
                        archive.write(path, path.relative_to(root))
    os.replace(target.with_suffix(".tmp"), target)
    return target


def checkpoint_files(run_dir, stage_index=None):
    """List remote files recorded by an accepted stage."""
    root = Path(run_dir).expanduser().resolve()
    state = read_state(root)
    _check_config(root, state)
    index = state["current_stage"] if stage_index is None else stage_index
    if type(index) is not int or not 0 <= index < len(state["stages"]):
        raise ValueError("Choose an existing stage.")
    stage = state["stages"][index]
    if stage["status"] != "succeeded":
        raise ValueError("Download checkpoints only from an accepted stage.")
    path = root / stage["folder"] / "outputs" / "checkpoint.json"
    if _digest(path) != stage["result"]["checkpoint_sha256"]:
        raise ValueError("The checkpoint receipt changed.")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_raw_files(run_dir, names, stage_index=None, transport=None):
    """Fetch selected checkpoint files only on explicit request."""
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        _check_config(root, state)
        index = state["current_stage"] if stage_index is None else stage_index
        stage = state["stages"][index]
        output = root / stage["folder"] / "outputs"
        checkpoint = checkpoint_files(root, index)
        if not names or any(name not in checkpoint for name in names):
            raise ValueError("Choose files listed in the accepted checkpoint receipt.")
        raw = output / "raw"
        receipt_path = raw / "artifact_manifest.json"
        for name in names:
            _name(name)
        previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        config = ClusterConfig.load(root / "config.json")
        owned = transport is None
        transport = transport or SSHTransport(config)
        try:
            with tempfile.TemporaryDirectory(prefix=".engine-download-", dir=output) as directory:
                staging = Path(directory)
                manifest = transport.download_many(state["remote_root"] + "/" + stage["folder"], staging,
                                                   names, permitted_names=frozenset(names))
                if manifest != {name: checkpoint[name] for name in names}:
                    raise ValueError("Remote checkpoint no longer matches the accepted result.")
                raw.mkdir(exist_ok=True)
                for name in names:
                    staged = staging / name
                    if (staged.is_symlink() or not staged.is_file() or staged.stat().st_size != manifest[name]["size"]
                            or _digest(staged) != manifest[name]["sha256"]):
                        raise ValueError("The downloaded checkpoint does not match its receipt.")
                    os.replace(staged, raw / name)
            _write(receipt_path, {**previous, **manifest})
            return manifest
        finally:
            if owned:
                transport.close()
