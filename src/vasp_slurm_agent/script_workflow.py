"""Reviewed Python and Wannier90 jobs with retained execution evidence.

Python output parsing establishes file integrity, not scientific validity.
Wannier validation is limited to localization and Hamiltonian consistency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile

import numpy as np

from .config import ClusterConfig
from .project_files import _safe_content
from .scheduler import get_scheduler, shell_command, TERMINAL_STATES
from .secret_guard import redact, reject_ordinary_text
from .transport import SSHTransport, TransportError, worker_environment
from .workflow import (_acquire, _release, _check_config, _command, _digest, _lock,
                       _poll, _save, _scheduler_flag, _write, read_state, utc_now)


TERMINAL = {"succeeded", "completed_unverified", "failed", "cancelled", "needs_attention",
            "unconverged", "missing_outputs", "suspect"}
_INTERNAL = frozenset({"execute.py", "runtime.json", "submit.sh", "dispatch.py", "scheduler.py"})
_RECEIPTS = frozenset({"execution.json", "job.json", "submit_intent.json", "stdout.log", "stderr.log",
                       "slurm.out", "slurm.err"})
_KINDS = {"file", "text", "json", "csv", "npy", "npz", "wannier_hr"}
_LIMIT = 512 * 1024 * 1024
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
WANNIER_SOURCE = "https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/"

_EXECUTE = '''import hashlib,json,os,platform,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): value.update(block)
    return value.hexdigest()
runtime=json.loads(Path("runtime.json").read_text())
actual={name:digest(name) for name in runtime["input_sha256"]}
if actual!=runtime["input_sha256"]: raise ValueError("Prepared inputs changed")
executables={}
for token in runtime["command_prefix"]:
    path=shutil.which(token)
    if path and Path(path).is_file():
        path=str(Path(path).resolve())
        executables[token]={"path":path,"sha256":digest(path)}
version=None
program=next(reversed(list(executables)),"")
if program in executables:
    try:
        probe=subprocess.run([program,"-V" if runtime["engine"]=="python" else "-v"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
        if probe.returncode==0: version=(probe.stdout or probe.stderr).strip()[:2048]
    except (OSError,subprocess.SubprocessError): pass
with open("stdout.log","w") as stdout,open("stderr.log","w") as stderr:
    code=subprocess.run(runtime["argv"],stdout=stdout,stderr=stderr).returncode
names=[entry["path"] for entry in runtime["outputs"]]+["stdout.log","stderr.log"]
outputs={name:{"sha256":digest(name),"size":Path(name).stat().st_size}
         for name in names if Path(name).is_file() and not Path(name).is_symlink()}
receipt={"job_id":os.environ.get("PBS_JOBID" if runtime["scheduler"]=="pbs" else "SLURM_JOB_ID"),
 "job_name":os.environ.get("PBS_JOBNAME" if runtime["scheduler"]=="pbs" else "SLURM_JOB_NAME"),
 "exit_code":code,"finished_at":datetime.now(timezone.utc).isoformat(),
 "input_sha256":actual,"final_input_sha256":{name:digest(name) for name in actual},
 "runtime_sha256":digest("runtime.json"),"outputs":outputs,
 "environment":{"wrapper_python":sys.version,"hostname":platform.node(),"executables":executables,"software_version":version}}
Path("execution.json").write_text(json.dumps(receipt,indent=2)+"\\n")
raise SystemExit(code)
'''


def _name(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,199}", value)
            or value.lower().startswith(("potcar", "id_rsa", "id_ed25519"))
            or value.lower() in {"vasp_license", "credentials.json", "auth.json"}):
        raise ValueError("Use a plain filename without directories or private credentials.")
    return value


def _command_prefix(command):
    if not isinstance(command, str) or not command.strip() or any(c in command for c in "\r\n\x00;&|><`$"):
        raise ValueError("Use an executable command, without shell operators or substitutions.")
    reject_ordinary_text(command)
    prefix = shlex.split(command)
    if not prefix or prefix[0].startswith("-"):
        raise ValueError("Choose a configured executable.")
    return prefix


def _validation(engine, value):
    if engine == "python":
        if value is not None:
            raise ValueError("Python results need a separately implemented scientific validator.")
        return None
    value = dict(value or {"kind": "wannier90"})
    if set(value) - {"kind", "spread_tolerance", "gradient_tolerance", "window"} or value.get("kind") != "wannier90":
        raise ValueError("Choose the Wannier90 localization validator.")
    result = {"kind": "wannier90", "spread_tolerance": value.get("spread_tolerance", 1e-6),
              "gradient_tolerance": value.get("gradient_tolerance", 1e-4), "window": value.get("window", 5)}
    for key in ("spread_tolerance", "gradient_tolerance"):
        if type(result[key]) not in {int, float} or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Wannier tolerances must be positive and finite.")
    if type(result["window"]) is not int or not 2 <= result["window"] <= 100:
        raise ValueError("Use a convergence window between 2 and 100 iterations.")
    return result


def _outputs(outputs, engine, entrypoint, inputs):
    values = []
    for entry in outputs:
        if not isinstance(entry, dict) or set(entry) - {"path", "kind", "required"}:
            raise ValueError("Declare each output path, kind and whether it is required.")
        name = _name(entry.get("path"))
        kind, required = entry.get("kind", "file"), entry.get("required", True)
        if name in inputs or name in _INTERNAL | _RECEIPTS or kind not in _KINDS or type(required) is not bool:
            raise ValueError("Outputs must have unique names distinct from inputs and execution receipts.")
        values.append({"path": name, "kind": kind, "required": required})
    if len({entry["path"] for entry in values}) != len(values) or len(values) > 200:
        raise ValueError("Declare at most 200 unique output files.")
    if engine == "wannier90":
        seed = Path(entrypoint).stem
        required = {seed + ".wout": "text", seed + "_hr.dat": "wannier_hr"}
        for name, kind in required.items():
            entry = next((entry for entry in values if entry["path"] == name), None)
            if entry and (entry["kind"] != kind or entry["required"] is not True):
                raise ValueError("Wannier validation requires its wout and hr.dat outputs.")
            if entry is None:
                values.append({"path": name, "kind": kind, "required": True})
    return values


def prepare_run(run_dir, config, *, engine, files, entrypoint, arguments=(), outputs=(), command, validation=None):
    """Freeze reviewed files locally; submission requires a separate advance call."""
    if engine not in {"python", "wannier90"}:
        raise ValueError("Choose Python or Wannier90.")
    if config is None:
        raise ValueError("Choose a software environment before preparing inputs.")
    config = ClusterConfig(**config.to_dict())
    prefix = _command_prefix(command)
    entrypoint = _name(entrypoint)
    if not isinstance(files, dict) or not files or len(files) > 200:
        raise ValueError("Choose up to 200 reviewed input files.")
    if entrypoint not in files or Path(entrypoint).suffix != (".py" if engine == "python" else ".win"):
        raise ValueError("Select the reviewed .py or .win entrypoint from the input files.")
    if not isinstance(arguments, (list, tuple)) or any(not isinstance(v, str) or any(c in v for c in "\0\n\r") for v in arguments):
        raise ValueError("Arguments must be a list of literal strings.")
    for argument in arguments:
        reject_ordinary_text(argument)
    validator = _validation(engine, validation)
    output_spec = _outputs(outputs, engine, entrypoint, files)
    root = Path(run_dir).expanduser().absolute()
    if root.is_symlink() or root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This run already exists. Resume it or choose a new folder.")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".script-plan-", dir=root.parent))
    try:
        folder = "01_" + engine
        inputs = temporary / folder / "inputs"
        inputs.mkdir(parents=True, mode=0o700)
        for name, source in files.items():
            _name(name)
            if name in _INTERNAL | _RECEIPTS:
                raise ValueError("An input uses a reserved execution filename.")
            source = Path(source).expanduser().absolute()
            if source.is_symlink() or not source.is_file() or source.stat().st_size > _LIMIT:
                raise ValueError("Choose regular input files smaller than 512 MiB.")
            data = source.read_bytes()
            _safe_content(data)
            (inputs / name).write_bytes(data)
        if engine == "python":
            compile((inputs / entrypoint).read_bytes(), entrypoint, "exec")
        else:
            _num_wann((inputs / entrypoint).read_text())
        for name in ("dispatch.py", "scheduler.py"):
            shutil.copyfile(Path(__file__).with_name(name), inputs / name)
        (inputs / "execute.py").write_text(_EXECUTE)
        adapter = get_scheduler(config)
        lines = adapter.header(config)
        lines.extend(["set -e", *adapter.working_directory(), *config.setup_commands,
                      f"export OMP_NUM_THREADS={config.cpus_per_task}", "python3 execute.py"])
        (inputs / "submit.sh").write_text("\n".join(lines) + "\n")
        hashes = {path.name: _digest(path) for path in inputs.iterdir()}
        argv = [*prefix, entrypoint if engine == "python" else Path(entrypoint).stem, *arguments]
        _write(inputs / "runtime.json", {"schema_version": 1, "engine": engine,
               "command_prefix": prefix, "argv": argv, "scheduler": config.scheduler,
               "input_sha256": hashes, "outputs": output_spec})
        hashes["runtime.json"] = _digest(inputs / "runtime.json")
        run_id = engine + "-" + uuid.uuid4().hex[:16]
        specification = {"name": engine, "folder": folder, "job_name": run_id,
                         "entrypoint": entrypoint, "command_prefix": prefix, "arguments": list(arguments),
                         "outputs": output_spec, "validation": validator, "metadata": {"input_sha256": hashes}}
        _write(temporary / "plan.json", {"schema_version": 1, "engine": engine, "workflow": "script",
               "stages": [specification], "sources": {folder + "/inputs/" + name: digest for name, digest in hashes.items()}})
        config.save(temporary / "config.json")
        for path in temporary.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        state = {"schema_version": 2, "engine": engine, "workflow": "script", "run_id": run_id,
                 "task": engine, "tasks": [engine], "formula": Path(entrypoint).stem, "status": "planned",
                 "current_stage": 0, "stages": [{**specification, "job_id": None, "status": "planned", "staged": False, "result": None}],
                 "history": [], "created_at": utc_now(), "last_error": None, "cancel_requested": False,
                 "plan_sha256": _digest(temporary / "plan.json"), "config_sha256": _digest(temporary / "config.json"),
                 "remote_root": config.remote_root.rstrip("/") + "/" + run_id}
        _save(temporary, state, "Reviewed inputs prepared. Ready to submit.")
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _num_wann(text):
    text = re.sub(r"[!#].*", "", text)
    values = re.findall(r"(?im)^\s*num_wann\s*[:=]\s*(\d+)\s*$", text)
    if len(values) != 1 or not 0 < int(values[0]) <= 10000:
        raise ValueError("The Wannier input must declare one positive num_wann value.")
    return int(values[0])


def _read_hr(path, expected):
    """Check the native multi-orbital HR format and H(R)=H(-R)† in eV."""
    with Path(path).open() as stream:
        stream.readline()
        try:
            count, points = int(stream.readline()), int(stream.readline())
            if count != expected or not 0 < points <= 100000 or points * count * count > 20_000_000:
                raise ValueError("The Wannier Hamiltonian dimensions differ from num_wann or exceed the parser limit.")
            weights = []
            while len(weights) < points:
                line = stream.readline()
                if not line:
                    raise ValueError("The Wannier Hamiltonian degeneracies are incomplete.")
                weights.extend(int(x) for x in line.split())
            if len(weights) != points or min(weights) < 1:
                raise ValueError("Invalid Wannier Hamiltonian degeneracies.")
            matrices = {}
            for weight in weights:
                matrix, seen, vector = np.zeros((count, count), complex), set(), None
                for _ in range(count * count):
                    parts = stream.readline().split()
                    if len(parts) != 7:
                        raise ValueError("Incomplete Wannier Hamiltonian matrix.")
                    current = tuple(map(int, parts[:3]))
                    i, j = int(parts[3]) - 1, int(parts[4]) - 1
                    if not 0 <= i < count or not 0 <= j < count or (i, j) in seen or vector is not None and vector != current:
                        raise ValueError("Invalid or duplicate Wannier matrix indices.")
                    vector = current
                    seen.add((i, j))
                    matrix[i, j] = complex(float(parts[5].replace("D", "E").replace("d", "e")),
                                           float(parts[6].replace("D", "E").replace("d", "e"))) / weight
                if vector in matrices or not np.isfinite(matrix).all():
                    raise ValueError("Duplicate or nonfinite Wannier Hamiltonian data.")
                matrices[vector] = matrix
            if stream.read().strip():
                raise ValueError("Unexpected trailing Wannier Hamiltonian data.")
        except (IndexError, OverflowError) as error:
            raise ValueError("Incomplete Wannier Hamiltonian data.") from error
    residual = 0.0
    for vector, matrix in matrices.items():
        opposite = tuple(-value for value in vector)
        if opposite not in matrices or not np.allclose(matrix, matrices[opposite].conj().T, atol=1e-8, rtol=1e-7):
            raise ValueError("The Wannier Hamiltonian is not Hermitian.")
        residual = max(residual, float(np.max(np.abs(matrix - matrices[opposite].conj().T))))
    return {"num_wann": count, "real_space_points": points, "hermiticity_residual_ev": residual}


def _wannier(output, stage):
    seed = Path(stage["entrypoint"]).stem
    text = (output / (seed + ".wout")).read_text()
    if "All done: wannier90 exiting" not in text or "Final State" not in text:
        raise ValueError("Wannier90 did not finish localization normally.")
    if re.search(r"Error:|Exiting.*error|convergence\s+(?:NOT|not)\s+achieved", text):
        raise ValueError("Wannier90 reports an error or missing convergence.")
    if "<-- DIS" in text and "Disentanglement convergence criteria satisfied" not in text:
        raise ValueError("Wannier disentanglement has not reported convergence.")
    rows = [[float(value.replace("D", "E").replace("d", "e")) for value in re.findall(_NUMBER, line.split("<-- CONV")[0])]
            for line in text.splitlines() if "<-- CONV" in line and re.match(r"\s*\d+\s", line)]
    validator = stage["validation"]
    recent = rows[-validator["window"]:]
    if (len(recent) != validator["window"] or any(len(row) != 5 or not np.isfinite(row).all()
            or abs(row[1]) > validator["spread_tolerance"] or not 0 <= row[2] <= validator["gradient_tolerance"]
            for row in recent) or any(b[0] <= a[0] for a, b in zip(recent, recent[1:]))):
        raise ValueError("Wannier localization has not met the reviewed spread and gradient tolerances.")
    count = _num_wann((output / stage["entrypoint"]).read_text())
    final = text.split("Final State")[-1]
    centres = re.findall(r"WF centre and spread\s+(\d+)\s*\(\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*,\s*(" + _NUMBER + r")\s*\)\s*(" + _NUMBER + r")", final)
    if len(centres) != count or [int(row[0]) for row in centres] != list(range(1, count + 1)):
        raise ValueError("The final Wannier centres differ from num_wann.")
    values = np.array([[float(value.replace("D", "E").replace("d", "e")) for value in row[1:]] for row in centres])
    if not np.isfinite(values).all() or np.any(values[:, 3] < 0):
        raise ValueError("The final Wannier centres or spreads are invalid.")
    result = _read_hr(output / (seed + "_hr.dat"), count)
    result.update(localization_checked=True, material_model_validated=False,
                  spread_sum=float(values[:, 3].sum()), spread_unit="native wout length unit squared",
                  source=WANNIER_SOURCE)
    return result


def _typed_output(path, kind):
    if path.stat().st_size > _LIMIT:
        raise ValueError("Output parsing is limited to 512 MiB per file.")
    if kind == "json":
        def number(text):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("Nonfinite JSON number.")
            return value
        json.loads(path.read_text(), parse_constant=number, parse_float=number)
    elif kind == "csv":
        with path.open(newline="") as stream:
            rows = csv.reader(stream, strict=True)
            first = next(rows, None)
            if first is None or not first:
                raise ValueError("The CSV output is empty.")
            for row in rows:
                if len(row) != len(first):
                    raise ValueError("The CSV output has inconsistent column counts.")
                for cell in row:
                    try:
                        value = float(cell)
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        raise ValueError("The CSV output contains a nonfinite number.")
    elif kind in {"npy", "npz"}:
        if kind == "npz":
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > _LIMIT:
                    raise ValueError("Expanded array data exceeds the parser limit.")
        arrays = np.load(path, allow_pickle=False)
        try:
            if kind == "npy" and not isinstance(arrays, np.ndarray) or kind == "npz" and not isinstance(arrays, np.lib.npyio.NpzFile):
                raise ValueError("The array format differs from the declared output kind.")
            values = [arrays] if kind == "npy" else [arrays[key] for key in arrays.files]
            if not values or any(value.dtype.kind not in "biufc" or not np.isfinite(value).all() for value in values):
                raise ValueError("The array output must contain finite numeric arrays.")
        finally:
            if isinstance(arrays, np.lib.npyio.NpzFile):
                arrays.close()
    elif kind == "text":
        path.read_text()
    elif kind == "wannier_hr":
        with path.open() as stream:
            stream.readline()
            expected = int(stream.readline())
        if not 0 < expected <= 10000:
            raise ValueError("Invalid Wannier orbital count.")
        _read_hr(path, expected)


def _evidence(root, state):
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    result = {"process_success": False, "accepted": False, "status": state["status"], "outputs": [],
              "validation": None, "error": None, "scientific_accuracy_validated": False}
    try:
        _check(root, state)
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Run evidence cannot contain symbolic links.")
        if not (output / "artifact_manifest.json").is_file():
            if state["status"] in {"succeeded", "completed_unverified"}:
                raise ValueError("The retained execution evidence is missing.")
            return result
        config = ClusterConfig.load(root / "config.json")
        adapter = get_scheduler(config)
        names = frozenset(stage["metadata"]["input_sha256"]) | _RECEIPTS | {entry["path"] for entry in stage["outputs"]}
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        if not isinstance(manifest, dict) or set(manifest) - names:
            raise ValueError("The collected file manifest differs from the reviewed output list.")
        for name, item in manifest.items():
            if (not (output / name).is_file() or _digest(output / name) != item["sha256"]
                    or (output / name).stat().st_size != item["size"]):
                raise ValueError("Collected evidence changed after download.")
        expected = stage["metadata"]["input_sha256"]
        for name, digest in expected.items():
            if name not in manifest or _digest(output / name) != digest:
                raise ValueError("The executed input files differ from the reviewed plan.")
        runtime = json.loads((output / "runtime.json").read_text())
        if "execution.json" not in manifest:
            raise ValueError("Execution receipt is missing. Check stdout.log, stderr.log and the scheduler logs.")
        if "job.json" not in manifest:
            raise ValueError("Submission receipt is missing. Reconnect to collect the job evidence.")
        receipt = json.loads((output / "execution.json").read_text())
        job = json.loads((output / "job.json").read_text())
        if (receipt.get("job_id") != stage.get("job_id") or receipt.get("job_name") != adapter.job_name(stage["job_name"])
                or type(receipt.get("exit_code")) is not int or not receipt.get("finished_at")
                or receipt.get("runtime_sha256") != expected["runtime.json"]
                or receipt.get("input_sha256") != runtime["input_sha256"]
                or receipt.get("final_input_sha256") != runtime["input_sha256"]
                or job.get("job_id") != stage.get("job_id") or job.get("job_name") != stage["job_name"]
                or job.get("scheduler", "slurm") != adapter.name):
            raise ValueError("The execution receipt differs from the reviewed job.")
        accounting = (output / "scheduler.txt").read_text()
        completed = (adapter.parse_status(accounting, stage["job_id"]) == "COMPLETED" if adapter.name == "pbs" else
                     any(line.split("|")[:3] == [stage["job_id"], "COMPLETED", "0:0"] for line in accounting.splitlines()))
        if not completed or receipt["exit_code"] != 0:
            result.update(status="cancelled" if str(stage.get("scheduler_state", "")).startswith("CANCELLED") else "failed",
                          error="The scheduler or program did not complete successfully.")
            return result
        for entry in [*stage["outputs"], {"path": "stdout.log", "kind": "text", "required": True},
                      {"path": "stderr.log", "kind": "text", "required": True}]:
            name = entry["path"]
            if name not in manifest:
                if entry["required"]:
                    result.update(status="missing_outputs", error="A required output is missing: " + name)
                    return result
                continue
            if receipt.get("outputs", {}).get(name) != manifest[name]:
                raise ValueError("An output differs from the native execution receipt.")
            _typed_output(output / name, entry["kind"])
            result["outputs"].append({**entry, **manifest[name]})
        result.update(process_success=True, status="completed_unverified")
        if state["engine"] == "wannier90":
            try:
                result["validation"] = _wannier(output, stage)
                result.update(accepted=True, status="succeeded")
            except (ValueError, OSError) as error:
                result.update(status="unconverged" if "tolerances" in str(error) or "convergence" in str(error) else "suspect", error=str(error))
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        result.update(status="needs_attention", error=redact(str(error)))
    return result


def doctor(config, stage, transport):
    commands = [*get_scheduler(config).required_commands, "python3", stage["command_prefix"][0]]
    # The last token convention covers executable and MPI-launcher prefixes.
    if not stage["command_prefix"][-1].startswith("-"):
        commands.append(stage["command_prefix"][-1])
    _command(transport, "\n".join(["set -e", *config.setup_commands,
             *["command -v " + shlex.quote(command) + " >/dev/null" for command in dict.fromkeys(commands)]]))
    return {"checked_at": utc_now(), "scheduler": config.scheduler, "command_available": True,
            "submission_permission_verified": False}


def _check(root, state):
    _check_config(root, state)
    config = ClusterConfig.load(root / "config.json")
    plan = json.loads((root / "plan.json").read_text())
    if (state.get("workflow") != "script" or state.get("engine") not in {"python", "wannier90"}
            or state["engine"] != plan.get("engine") or len(state["stages"]) != 1
            or state["run_id"] != state["stages"][0]["job_name"]
            or state["remote_root"] != config.remote_root.rstrip("/") + "/" + state["run_id"]):
        raise ValueError("The run identity or remote destination changed.")
    return config


def _receipt(reply, stage, config):
    value = json.loads(reply.splitlines()[-1])
    adapter = get_scheduler(config)
    if (value.get("job_name") != stage["job_name"] or value.get("scheduler", "slurm") != adapter.name
            or not adapter.valid_job_id(str(value.get("job_id", "")))):
        raise ValueError("Invalid scheduler submission receipt.")
    return str(value["job_id"])


def _collect(root, state, transport, config):
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    output.mkdir(exist_ok=True)
    allowed = frozenset(stage["metadata"]["input_sha256"]) | _RECEIPTS | {entry["path"] for entry in stage["outputs"]}
    code = "from pathlib import Path;import json;print(json.dumps([p.name for p in Path(" + repr(state["remote_root"]) + ").iterdir() if p.is_file() and not p.is_symlink()]))"
    names = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("The remote output listing is invalid.")
    manifest = transport.download_many(state["remote_root"], output, [name for name in names if name in allowed], permitted_names=allowed)
    _write(output / "artifact_manifest.json", manifest)
    adapter = get_scheduler(config)
    query = (shell_command(adapter.status_command(stage["job_id"])) if adapter.name == "pbs" else
             f"sacct -n -X -P -j {stage['job_id']} --format=JobIDRaw,State,ExitCode")
    (output / "scheduler.txt").write_text(_command(transport, query) + "\n")
    result = _evidence(root, state)
    if stage["scheduler_state"] != "COMPLETED":
        result.update(status="failed", accepted=False, process_success=False,
                      error="The scheduler ended with " + stage["scheduler_state"] + ". " + (result["error"] or ""))
    _write(output / "verification.json", result)
    stage.update(status=result["status"], result=result)
    state.update(status=result["status"], last_error=result["error"])
    return _save(root, state, "Outputs collected and checked.")


def advance(run_dir, transport=None, *, poll_only=False):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        stage = state["stages"][0]
        if poll_only and (not stage.get("job_id") or stage["status"] in {"planned", "submitting"}):
            return state
        owned = transport is None
        try:
            if state.get("workflow") != "script" or state.get("engine") not in {"python", "wannier90"}:
                raise ValueError("Choose a reviewed Python or Wannier90 run.")
            config = _check(root, state)
            transport = transport or SSHTransport(config)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                if not stage["staged"]:
                    _write(root / "doctor.json", doctor(config, stage, transport))
                    stage["status"] = state["status"] = "submitting"
                    _save(root, state)
                    inputs = root / stage["folder"] / "inputs"
                    names = frozenset(stage["metadata"]["input_sha256"])
                    transport.upload_many([inputs / name for name in sorted(names)], state["remote_root"], permitted_names=names)
                    stage["staged"] = True
                    _save(root, state)
                reply = _command(transport, "python3 " + shlex.quote(state["remote_root"] + "/dispatch.py") + " "
                                 + shlex.quote(state["remote_root"]) + " " + shlex.quote(stage["job_name"]) + _scheduler_flag(config), timeout=100)
                stage.update(job_id=_receipt(reply, stage, config), status="queued")
                state.update(status="queued", last_error=None, remote_failures=0)
                return _save(root, state, "The scheduler accepted the reviewed job.")
            scheduler = _poll(transport, stage, config)
            stage["scheduler_state"] = scheduler
            if state["cancel_requested"] and scheduler not in TERMINAL_STATES:
                _command(transport, shell_command(get_scheduler(config).cancel_command(stage["job_id"])))
            if scheduler == "CANCELLED":
                state["status"] = stage["status"] = "cancelled"
                state["last_error"] = None
                return _save(root, state, "The scheduler confirmed cancellation.")
            if scheduler in TERMINAL_STATES:
                state["status"] = stage["status"] = "collecting"
                _save(root, state)
                return _collect(root, state, transport, config)
            state["status"] = stage["status"] = "queued" if scheduler in {"PENDING", "UNKNOWN"} else "running"
            state.update(last_error="Waiting for scheduler accounting." if scheduler == "UNKNOWN" else None, remote_failures=0)
            return _save(root, state)
        except TransportError as error:
            state.update(status="needs_attention", last_error=redact(str(error)), remote_failures=state.get("remote_failures", 0) + 1)
            return _save(root, state, "Connection interrupted. Reconnect to the original job.")
        except (ValueError, OSError, KeyError, TypeError) as error:
            state.update(status="needs_attention", last_error=redact(str(error)))
            return _save(root, state, "Execution paused.")
        finally:
            if owned and transport is not None:
                transport.close()


def resume(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] != "needs_attention":
            return state
        stage = state["stages"][0]
        owned = transport is None
        try:
            config = _check(root, state)
            if not stage.get("job_id") and stage["staged"]:
                transport = transport or SSHTransport(config)
                remote = state["remote_root"]
                reply = _command(transport, f"python3 {shlex.quote(remote+'/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only{_scheduler_flag(config)}", timeout=100)
                stage["job_id"] = _receipt(reply, stage, config)
            status = "queued" if stage.get("job_id") else "cancelled" if state["cancel_requested"] else "planned"
            state["status"] = stage["status"] = status
            state.update(last_error=None, remote_failures=0)
            return _save(root, state, "Resuming the original job. No replacement submitted.")
        except (TransportError, ValueError, OSError, KeyError) as error:
            state["last_error"] = redact(str(error))
            return _save(root, state, "Reconnection failed. The original identity is retained.")
        finally:
            if owned and transport is not None:
                transport.close()


def cancel(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL - {"needs_attention"}:
            return state
        _check(root, state)
        stage = state["stages"][0]
        state["cancel_requested"] = True
        if not stage.get("job_id"):
            state["status"] = stage["status"] = "needs_attention" if stage["staged"] else "cancelled"
            state["last_error"] = "Reconcile the submission before cancelling." if stage["staged"] else None
        else:
            state["status"] = stage["status"] = "queued"
        _save(root, state, "Cancellation requested.")
    return advance(root, transport) if stage.get("job_id") else state


def load_run_context(run_dir):
    root = Path(run_dir).expanduser().resolve()
    state = read_state(root)
    if state.get("workflow") != "script":
        raise ValueError("Choose a reviewed script run.")
    result = _evidence(root, state)
    facts = {"run.status": {"label": "Result status", "value": result["status"], "source": "run.json", "unit": ""},
             "run.accepted": {"label": "Scoped validation accepted", "value": result["accepted"], "source": "native outputs", "unit": ""},
             "run.process_success": {"label": "Execution completed", "value": result["process_success"], "source": "execution and scheduler receipts", "unit": ""},
             "stage_1.accepted": {"label": "Scoped validation accepted", "value": result["accepted"], "source": "native outputs", "unit": ""}}
    limits = ["Process completion and readable files do not establish scientific correctness."]
    if state["engine"] == "wannier90":
        limits += ["Wannier validation checks localization and Hamiltonian consistency only.",
                   "Band interpolation accuracy, projection choice and material conclusions need separate validation."]
        for key, value in (result.get("validation") or {}).items():
            if key in {"num_wann", "real_space_points", "hermiticity_residual_ev", "material_model_validated"}:
                facts["stage_1." + key] = {"label": key.replace("_", " "), "value": value, "source": "native Wannier outputs", "unit": "eV" if key.endswith("_ev") else ""}
    if result["error"]:
        limits.append(result["error"])
    stage = state["stages"][0]
    artifacts = [{"id": "stage_1." + entry["path"], "label": entry["path"],
                  "path": stage["folder"] + "/outputs/" + entry["path"], "sha256": entry["sha256"],
                  "kind": entry["kind"], "size": entry["size"],
                  "status": "verified" if result["process_success"] else "present_unverified"} for entry in result["outputs"]]
    fingerprints = {path.relative_to(root).as_posix(): _digest(path) for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink() and path.name not in {"worker.log", "explanations.json", "conversation.json", "results.zip"} and not path.name.startswith(".")}
    context = {"engine": state["engine"], "workflow": "script", "goal": "Run reviewed " + state["engine"] + " inputs.",
               "dialogue": [], "status": result["status"], "outcome": "Scoped output checks passed." if result["accepted"] else "Execution completed; scientific results remain unverified." if result["process_success"] else "No verified completed result.",
               "final_plan": {"tasks": [state["engine"]], "parameters": {"validation": stage["validation"]}},
               "facts": facts, "artifacts": artifacts, "limits": limits}
    context["context_sha256"] = hashlib.sha256(json.dumps({"context": context, "files": fingerprints}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return context


def bundle_run(run_dir):
    root = Path(run_dir).expanduser().resolve()
    state = read_state(root)
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("plan.json", "proposal.json", "parent.json", "conversation.json", "explanations.json"):
            path = root / name
            if path.is_file() and not path.is_symlink():
                archive.write(path, name)
        public_state = {key: value for key, value in state.items() if key != "remote_root"}
        archive.writestr("run.json", json.dumps(public_state, indent=2))
        for path in (root / state["stages"][0]["folder"]).rglob("*"):
            if path.is_file() and not path.is_symlink() and path.name not in {"submit.sh", "execute.py", "dispatch.py", "scheduler.py"}:
                archive.write(path, path.relative_to(root))
    os.replace(target.with_suffix(".tmp"), target)
    return target


def watch(run_dir, interval=20):
    password = os.environ.pop("_DFT_AGENT_WORKER_PASSWORD", None)
    root = Path(run_dir).expanduser().resolve()
    with (root / ".worker.lock").open("a") as lock:
        try:
            _acquire(lock, blocking=False)
        except BlockingIOError:
            return read_state(root)
        transport = None
        try:
            _write(root / "worker.json", {"pid": os.getpid(), "started_at": utc_now()})
            transport = SSHTransport(ClusterConfig.load(root / "config.json"), password=password)
            while True:
                state = advance(root, transport)
                if state["status"] in TERMINAL:
                    bundle_run(root)
                    return state
                time.sleep(interval)
        finally:
            try:
                if transport is not None:
                    transport.close()
            finally:
                _release(lock)


def start_worker(run_dir):
    from .runtime import cli_command
    root = Path(run_dir).expanduser().resolve()
    with (root / "worker.log").open("ab") as stream:
        process = subprocess.Popen(cli_command("watch", root), stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                   start_new_session=True, env=worker_environment(ClusterConfig.load(root / "config.json")))
    return process.pid
