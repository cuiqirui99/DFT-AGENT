"""Frozen single-band pairing models submitted through the existing cluster transport."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import uuid
import zipfile

import numpy as np

from . import qe_pairing as pairing
from .config import ClusterConfig
from .scheduler import get_scheduler, shell_command
from .transport import SSHTransport, TransportError
from .workflow import (_check_config, _command, _digest, _lock, _poll, _scheduler_flag, _save, _write,
                       read_state, utc_now, TERMINAL, SLURM_TERMINAL, MAX_REMOTE_FAILURES)


INPUTS = frozenset({"hamiltonian_hr.dat", "request.json", "qe_pairing.py", "execute.py",
                    "runtime.json", "submit.sh", "dispatch.py", "scheduler.py"})
OUTPUTS = INPUTS | {"result.json", "pairing.npz", "pairing.out", "pairing.err", "execution.json",
                   "slurm.out", "slurm.err", "job.json", "submit_intent.json"}

_EXECUTE = '''import hashlib,json,os,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
runtime=json.loads(Path("runtime.json").read_text())
actual={name:digest(name) for name in runtime["input_sha256"]}
if actual!=runtime["input_sha256"]: raise ValueError("Prepared pairing inputs changed")
import numpy,scipy
environment={"python":sys.version,"numpy":numpy.__version__,"scipy":scipy.__version__,
             "python_sha256":digest(Path(sys.executable).resolve()),"hostname":platform.node()}
with open("pairing.out","w") as stdout,open("pairing.err","w") as stderr:
    code=subprocess.run([sys.executable,"qe_pairing.py","."],stdout=stdout,stderr=stderr).returncode
outputs={name:digest(name) for name in ("result.json","pairing.npz","pairing.out","pairing.err") if Path(name).is_file()}
receipt={"job_id":os.environ.get("PBS_JOBID" if runtime.get("scheduler") == "pbs" else "SLURM_JOB_ID"),"job_name":os.environ.get("PBS_JOBNAME" if runtime.get("scheduler") == "pbs" else "SLURM_JOB_NAME"),
         "exit_code":code,"finished_at":datetime.now(timezone.utc).isoformat(),
         "input_sha256":actual,"runtime_sha256":digest("runtime.json"),"output_sha256":outputs,
         "environment":environment}
Path("execution.json").write_text(json.dumps(receipt,indent=2)+"\\n")
raise SystemExit(code)
'''


def _python(value):
    if (not isinstance(value, str) or not value.strip() or any(c in value for c in "\n\r\x00")
            or not (value.startswith("/") or re.fullmatch(r"[A-Za-z0-9_.+-]+", value))):
        raise ValueError("Enter an existing Python executable name or absolute path, without arguments.")
    return value


def _resources(config):
    if config.engine != "qe" or config.compute_backend != "cpu" or config.nodes != 1 or config.tasks != 1:
        raise ValueError("Pairing uses one CPU process. Select QE, CPU, one node and one task.")


def doctor(config, python_executable, transport=None):
    """Check the user's Python runtime without installing or changing it."""
    _resources(config)
    python_executable = _python(python_executable)
    code = ("import hashlib,json,sys;from pathlib import Path;import numpy,scipy;"
            "assert sys.version_info>=(3,9),'Python 3.9 or newer is required';"
            "assert tuple(map(int,numpy.__version__.split('.')[:2]))>=(1,20),'NumPy 1.20 or newer is required';"
            "assert tuple(map(int,scipy.__version__.split('.')[:2]))>=(1,8),'SciPy 1.8 or newer is required';"
            "print('PAIRING_RUNTIME '+json.dumps({'python':sys.version,'numpy':numpy.__version__,"
            "'scipy':scipy.__version__,'python_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()}))")
    owned = transport is None
    transport = transport or SSHTransport(config)
    try:
        _command(transport, " && ".join("command -v " + name + " >/dev/null" for name in (*get_scheduler(config).required_commands, "python3")))
        output = _command(transport, "\n".join(["set -e", *config.setup_commands,
                          shlex.quote(python_executable) + " -c " + shlex.quote(code)]), timeout=120)
        records = [line.removeprefix("PAIRING_RUNTIME ") for line in output.splitlines() if line.startswith("PAIRING_RUNTIME ")]
        if len(records) != 1:
            raise ValueError("The Python runtime did not return a valid receipt.")
        result = json.loads(records[0])
        if not re.fullmatch(r"[0-9a-f]{64}", result.get("python_sha256", "")):
            raise ValueError("The Python runtime checksum is missing.")
        return {"success": True, "environment": result}
    finally:
        if owned:
            transport.close()


def prepare_run(hr_file, run_dir, config, parameters, *, python_executable, origin="wannier_unvalidated"):
    """Freeze a model and executable selection locally; this does not submit."""
    _resources(config)
    python_executable = _python(python_executable)
    root = Path(run_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This run already exists. Choose a new folder or resume it.")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pairing-plan-", dir=root.parent))
    try:
        folder = "01_pairing"
        (temporary / folder).mkdir()
        inputs = temporary / folder / "inputs"
        request = pairing.prepare_run(hr_file, inputs, parameters, origin=origin, backend="numpy")
        shutil.copyfile(Path(pairing.__file__), inputs / "qe_pairing.py")
        shutil.copyfile(Path(__file__).with_name("dispatch.py"), inputs / "dispatch.py")
        shutil.copyfile(Path(__file__).with_name("scheduler.py"), inputs / "scheduler.py")
        (inputs / "execute.py").write_text(_EXECUTE)
        adapter = get_scheduler(config)
        lines = adapter.header(config)
        lines.extend(["set -e", *adapter.working_directory(), *config.setup_commands,
                      f"export OMP_NUM_THREADS={config.cpus_per_task}", f"export OPENBLAS_NUM_THREADS={config.cpus_per_task}",
                      f"export MKL_NUM_THREADS={config.cpus_per_task}", shlex.quote(python_executable) + " execute.py"])
        (inputs / "submit.sh").write_text("\n".join(lines) + "\n")
        hashes = {path.name: _digest(path) for path in inputs.iterdir()}
        _write(inputs / "runtime.json", {"input_sha256": hashes, **({"scheduler": config.scheduler} if config.scheduler != "slurm" else {})})
        hashes["runtime.json"] = _digest(inputs / "runtime.json")
        run_id = "qe-pairing-" + uuid.uuid4().hex[:12]
        specification = {"name": "pairing", "folder": folder, "job_name": run_id,
                         "python_executable": python_executable, "parameters": request["parameters"],
                         "metadata": {"input_sha256": hashes}}
        _write(temporary / "plan.json", {"schema_version": 1, "engine": "qe", "workflow": "pairing",
               "stages": [specification], "sources": {folder + "/inputs/" + name: digest for name, digest in hashes.items()}})
        config.save(temporary / "config.json")
        state = {"schema_version": 2, "engine": "qe", "workflow": "pairing", "run_id": run_id,
                 "task": "pairing", "tasks": ["pairing"], "formula": "Pairing model", "status": "planned",
                 "current_stage": 0, "stages": [{**specification, "job_id": None, "status": "planned", "staged": False, "result": None}],
                 "history": [], "created_at": utc_now(), "last_error": None, "cancel_requested": False,
                 "plan_sha256": _digest(temporary / "plan.json"), "config_sha256": _digest(temporary / "config.json"),
                 "remote_root": config.remote_root.rstrip("/") + "/" + run_id}
        _save(temporary, state, "Pairing model prepared. Ready to submit.")
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_result(output, parameters):
    """Check stored eigenvalues against the native gap arrays, not flags alone."""
    result = json.loads((output / "result.json").read_text())
    request = json.loads((output / "request.json").read_text())
    if result.get("method") != "single_band_hubbard_rpa" or result.get("origin") != request["origin"]:
        raise ValueError("The result belongs to a different pairing model.")
    if (result.get("material_prediction_accepted") is not False or result.get("numerical_checks_passed") is not True
            or result.get("hamiltonian_sha256") != _digest(output / "hamiltonian_hr.dat")
            or result.get("temperature_k") != parameters["temperature_k"]):
        raise ValueError("The pairing result has inconsistent provenance or claims.")
    mesh, nw = tuple(parameters["kmesh"]), parameters["matsubara_points"]
    points = int(np.prod(mesh))
    beta = 1 / (pairing.KB_EV_K * parameters["temperature_k"])
    fractional = np.array(list(np.ndindex(*mesh))) / np.array(mesh)
    expected_energy = sum(value * np.cos(2 * np.pi * fractional @ np.array(vector))
                          for vector, value in pairing.read_hoppings(output / "hamiltonian_hr.dat").items())
    frequencies = (2 * np.arange(-nw, nw) + 1) * np.pi / beta
    with np.load(output / "pairing.npz", allow_pickle=False) as arrays:
        if (arrays["chi0"].shape != (4*nw-1, points) or arrays["energy_ev"].shape != (points,)
                or not np.allclose(arrays["energy_ev"], expected_energy, rtol=1e-12, atol=1e-10)
                or not np.allclose(arrays["fermionic_frequency_ev"], frequencies, rtol=1e-12, atol=1e-10)):
            raise ValueError("The native pairing grids or energies differ from the frozen model.")
        if any(not np.isfinite(arrays[key]).all() for key in arrays.files):
            raise ValueError("The native pairing arrays contain nonfinite values.")
        chi = arrays["chi0"]
        static = pairing._static_lindhard(expected_energy-parameters["mu_ev"], beta, mesh)
        if not np.allclose(chi[2*nw-1], static, rtol=1e-8, atol=1e-10) or np.max(static)*parameters["u_ev"] >= 1:
            raise ValueError("The native susceptibility fails the normal-state check.")
        filling = float(np.mean(1-np.tanh(beta*(expected_energy-parameters["mu_ev"])/2)))
        if (not np.isclose(result["stoner_factor"], np.max(static)*parameters["u_ev"], rtol=1e-8, atol=1e-10)
                or not np.isclose(result["filling_per_cell"], filling, rtol=1e-8, atol=1e-10)):
            raise ValueError("The reported filling or magnetic stability is inconsistent.")
        records = {entry["channel"]: entry for entry in result["channels"]}
        if len(records) != len(result["channels"]) or set(records) != set(parameters["channels"]):
            raise ValueError("The pairing channels differ from the request.")
        weight = 1 / (beta * points * (frequencies[:, None]**2 + (expected_energy-parameters["mu_ev"])[None, :]**2))
        for channel in parameters["channels"]:
            gap = arrays["gap_" + channel]
            sign = 1 if channel == "singlet" else -1
            inverse = pairing._symmetry_basis(nw, mesh, sign)[3]
            if gap.shape != (2*nw, points) or not np.isclose(np.linalg.norm(gap), 1, atol=1e-7):
                raise ValueError("The native gap array is invalid.")
            eigenvalue = records[channel]["eigenvalue"]
            convolution = pairing._convolution(pairing.pairing_vertex(chi, parameters["u_ev"], channel).real/2, mesh, 2*nw)
            residual = np.linalg.norm(-convolution(weight*gap)-eigenvalue*gap) / max(abs(eigenvalue), 1e-8)
            if (not np.isfinite(eigenvalue) or residual > 1e-6 or np.linalg.norm(gap[::-1]-gap)>1e-7
                    or np.linalg.norm(gap[:, inverse]-sign*gap)>1e-7):
                raise ValueError("The reported eigenvalue fails the native gap residual or exchange symmetry.")
            records[channel]["verified_relative_residual"] = float(residual)
    return result


def _accepted_model(root, state):
    """Recheck frozen inputs and native evidence whenever results are consumed."""
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    reasons, model = [], None
    try:
        _check_config(root, state)
        adapter = get_scheduler(ClusterConfig.load(root / "config.json"))
        if output.is_symlink() or output.parent.is_symlink():
            raise ValueError("Pairing evidence must stay inside its run folder.")
        for name in OUTPUTS | {"scheduler.txt", "artifact_manifest.json"}:
            if (output/name).is_symlink():
                raise ValueError("Pairing evidence cannot be a symbolic link.")
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        for name, entry in manifest.items():
            if name not in OUTPUTS or _digest(output/name) != entry["sha256"]:
                raise ValueError("Collected evidence changed after download.")
        accounting = (output / "scheduler.txt").read_text()
        expected = stage["metadata"]["input_sha256"]
        if any(not (output/name).is_file() or _digest(output/name) != digest for name, digest in expected.items()):
            raise ValueError("Collected inputs differ from the approved model.")
        receipt = json.loads((output / "execution.json").read_text())
        runtime = json.loads((output / "runtime.json").read_text())
        job = json.loads((output / "job.json").read_text())
        if (receipt.get("job_id") != stage["job_id"] or receipt.get("job_name") != adapter.job_name(stage["job_name"])
                or type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0 or not receipt.get("finished_at")
                or receipt.get("runtime_sha256") != expected["runtime.json"]
                or receipt.get("input_sha256") != runtime["input_sha256"]
                or job.get("job_id") != stage["job_id"] or job.get("job_name") != stage["job_name"]):
            raise ValueError("The native execution or submission receipt is inconsistent.")
        doctor_environment = json.loads((root/"doctor.json").read_text())["environment"]
        if any(receipt.get("environment", {}).get(key) != value for key, value in doctor_environment.items()):
            raise ValueError("The Python runtime changed after its check.")
        completed = (adapter.parse_status(accounting, stage["job_id"]) == "COMPLETED" if adapter.name == "pbs" else
                     any(line.split("|")[:3] == [stage["job_id"], "COMPLETED", "0:0"] for line in accounting.splitlines()))
        if not completed:
            raise ValueError("The scheduler has not confirmed successful completion.")
        for name in ("result.json", "pairing.npz", "pairing.out", "pairing.err"):
            if not (output/name).is_file() or _digest(output/name) != receipt.get("output_sha256", {}).get(name):
                raise ValueError("A native output differs from the execution receipt.")
        model = _validate_result(output, stage["parameters"])
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        reasons.append(str(exc))
    accepted = {"success": not reasons, "reason": " ".join(reasons), "model_result": model,
                "material_prediction_accepted": False, "scheduler_state": stage.get("scheduler_state")}
    return accepted


def _collect(root, state, transport):
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    output.mkdir(exist_ok=True)
    code = "from pathlib import Path;import json;print(json.dumps([p.name for p in Path(" + repr(state["remote_root"]) + ").iterdir() if p.is_file() and not p.is_symlink()]))"
    names = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    names = [name for name in names if name in OUTPUTS]
    manifest = transport.download_many(state["remote_root"], output, names, permitted_names=OUTPUTS)
    _write(output / "artifact_manifest.json", manifest)
    adapter = get_scheduler(ClusterConfig.load(root / "config.json"))
    query = (shell_command(adapter.status_command(stage["job_id"])) if adapter.name == "pbs" else
             f"sacct -n -X -P -j {stage['job_id']} --format=JobIDRaw,State,ExitCode")
    accounting = _command(transport, query)
    (output / "scheduler.txt").write_text(accounting + "\n")
    accepted = _accepted_model(root, state)
    _write(output / "acceptance.json", accepted)
    stage.update(result=accepted, status="succeeded" if accepted["success"] else "needs_attention")
    if str(stage.get("scheduler_state", "")).startswith("CANCELLED"):
        stage["status"] = "cancelled"
    state.update(status=stage["status"], last_error=None if accepted["success"] else accepted["reason"])
    return _save(root, state, "Pairing model outputs collected.")


def advance(run_dir, transport=None, *, poll_only=False):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        owned = transport is None
        stage = state["stages"][0]
        if poll_only and (not stage.get("job_id") or stage["status"] in {"planned", "submitting"}):
            return state
        try:
            if state.get("workflow") != "pairing":
                raise ValueError("Choose a pairing-model run.")
            _check_config(root, state)
            config = ClusterConfig.load(root / "config.json")
            _resources(config)
            transport = transport or SSHTransport(config)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                if not stage["staged"]:
                    _write(root / "doctor.json", doctor(config, stage["python_executable"], transport))
                    stage["status"] = state["status"] = "submitting"
                    _save(root, state)
                    inputs = root / stage["folder"] / "inputs"
                    transport.upload_many([inputs / name for name in stage["metadata"]["input_sha256"]],
                                          state["remote_root"], permitted_names=INPUTS)
                    stage["staged"] = True
                    _save(root, state)
                reply = _command(transport, "python3 " + shlex.quote(state["remote_root"] + "/dispatch.py") + " "
                                 + shlex.quote(state["remote_root"]) + " " + shlex.quote(stage["job_name"]) + _scheduler_flag(config), timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if receipt.get("job_name") != stage["job_name"] or not get_scheduler(config).valid_job_id(str(receipt.get("job_id", ""))):
                    raise ValueError("Invalid scheduler submission receipt.")
                stage.update(job_id=str(receipt["job_id"]), status="queued")
                state.update(status="queued", last_error=None, remote_failures=0)
                return _save(root, state, "The scheduler accepted the pairing model.")
            scheduler = _poll(transport, stage, config)
            stage["scheduler_state"] = scheduler
            if state["cancel_requested"] and scheduler not in SLURM_TERMINAL:
                _command(transport, shell_command(get_scheduler(config).cancel_command(stage["job_id"])))
            if scheduler in SLURM_TERMINAL:
                stage["status"] = state["status"] = "collecting"
                _save(root, state)
                return _collect(root, state, transport)
            stage["status"] = state["status"] = "queued" if scheduler in {"PENDING", "UNKNOWN"} else "running"
            state.update(last_error="Waiting for scheduler accounting." if scheduler == "UNKNOWN" else None, remote_failures=0)
            return _save(root, state)
        except TransportError as exc:
            state.update(last_error=str(exc), remote_failures=state.get("remote_failures", 0)+1)
            if state["remote_failures"] >= MAX_REMOTE_FAILURES or "outcome uncertain" in str(exc) or "did not acknowledge" in str(exc):
                state["status"] = "needs_attention"
            return _save(root, state, "Remote operation interrupted. The original job identity is retained.")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            state.update(status="needs_attention", last_error=str(exc))
            return _save(root, state, "Pairing model paused.")
        finally:
            if owned and transport is not None:
                transport.close()


def resume(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] not in {"failed", "needs_attention"}:
            return state
        _check_config(root, state)
        stage = state["stages"][0]
        owned = transport is None
        try:
            if not stage.get("job_id") and stage.get("staged"):
                config = ClusterConfig.load(root / "config.json")
                transport = transport or SSHTransport(config)
                remote = state["remote_root"]
                reply = _command(transport, f"python3 {shlex.quote(remote+'/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only{_scheduler_flag(config)}", timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if receipt.get("job_name") != stage["job_name"] or not get_scheduler(config).valid_job_id(str(receipt.get("job_id", ""))):
                    raise ValueError("Invalid reconciled job identity.")
                stage["job_id"] = str(receipt["job_id"])
            status = "queued" if stage.get("job_id") else "cancelled" if state["cancel_requested"] else "planned"
            stage["status"] = state["status"] = status
            state.update(last_error=None, remote_failures=0)
            return _save(root, state, "Resuming the original pairing model.")
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
        stage = state["stages"][0]
        state["cancel_requested"] = True
        if not stage["job_id"]:
            state["status"] = stage["status"] = "needs_attention" if stage["staged"] else "cancelled"
            state["last_error"] = "Reconcile the original submission before cancelling." if stage["staged"] else None
        else:
            stage["status"] = state["status"] = "queued"
        _save(root, state, "Cancellation requested.")
    return advance(root, transport) if stage["job_id"] else state


def bundle_run(run_dir):
    """Export the model, native outputs and receipts without cluster settings."""
    root = Path(run_dir).expanduser().resolve()
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("run.json", "plan.json", "proposal.json", "doctor.json", "explanations.json"):
            if (root/name).is_file() and not (root/name).is_symlink():
                archive.write(root/name, name)
        for path in (root / "01_pairing").rglob("*"):
            if path.is_file() and not path.is_symlink() and path.name not in {"submit.sh", "runtime.json"}:
                archive.write(path, path.relative_to(root))
    os.replace(target.with_suffix(".tmp"), target)
    return target


def load_run_context(run_dir):
    """Expose only freshly verified model facts to the result explanation layer."""
    root = Path(run_dir).expanduser().resolve()
    state = read_state(root)
    if state.get("workflow") != "pairing":
        raise ValueError("Choose a pairing-model run.")
    accepted = _accepted_model(root, state)
    stage = state["stages"][0]
    relative = stage["folder"] + "/outputs/"
    facts, artifacts, fingerprints = {}, [], {}
    def fact(name, label, value, source, unit=""):
        facts[name] = {"label": label, "value": value, "source": source, "unit": unit}
    status = state["status"]
    if status == "succeeded" and not accepted["success"]:
        status = "needs_attention"
    fact("run.status", "Current result status", status, "run.json")
    fact("run.accepted_stages", "Stages with accepted results", int(accepted["success"]), "run.json", "stages")
    fact("stage_1.accepted", "Pairing model accepted", accepted["success"], relative+"execution.json")
    limits = ["This is a single-orbital paramagnetic Hubbard RPA model without SOC or self-energy feedback.",
              "The uploaded Hamiltonian and chosen interaction need separate material validation.",
              "One temperature is not Tc. Momentum and Matsubara convergence are required.",
              "Pairing eigenvalues are model results, not evidence of material superconductivity."]
    if accepted["success"]:
        model = accepted["model_result"]
        for key, label, unit in (("temperature_k", "Temperature", "K"), ("filling_per_cell", "Model filling", "electrons/cell"),
                                 ("stoner_factor", "RPA Stoner factor", "")):
            fact("stage_1."+key, label, model[key], relative+"result.json", unit)
        for key in ("u_ev", "mu_ev", "kmesh", "matsubara_points"):
            fact("stage_1."+key, key, stage["parameters"][key], relative+"request.json", "eV" if key.endswith("_ev") else "")
        for entry in model["channels"]:
            prefix = "stage_1." + entry["channel"]
            fact(prefix+".eigenvalue", entry["channel"]+" pairing eigenvalue", entry["eigenvalue"], relative+"pairing.npz")
            fact(prefix+".relative_residual", "Verified eigenpair residual", entry["verified_relative_residual"], relative+"pairing.npz")
        fact("stage_1.material_prediction_accepted", "Material prediction accepted", False, relative+"request.json")
    else:
        limits.append("The stored evidence did not pass verification. Do not infer numbers from cached results.")
    for name in ("result.json", "pairing.npz", "execution.json", "pairing.out", "pairing.err", "scheduler.txt"):
        path = root / relative / name
        safe = path.is_file() and not path.is_symlink() and not path.parent.is_symlink() and not path.parent.parent.is_symlink()
        digest = _digest(path) if safe else None
        if digest:
            fingerprints[relative+name] = digest
        artifacts.append({"id": "stage_1."+name, "label": name, "path": relative+name, "sha256": digest,
                          "status": "verified" if accepted["success"] and digest else "present_unverified" if digest else "unavailable"})
    context = {"engine": "qe", "goal": "Analyze the stated single-band pairing model.", "dialogue": [], "status": status,
               "outcome": f"{int(accepted['success'])} of 1 stages have accepted model results.",
               "final_plan": {"tasks": ["pairing"], "parameters": {key: stage["parameters"][key] for key in
                    ("u_ev", "mu_ev", "temperature_k", "kmesh", "matsubara_points", "channels")} if accepted["success"] else {}},
               "facts": facts, "artifacts": artifacts, "limits": limits}
    context["context_sha256"] = hashlib.sha256(json.dumps({"context": context, "files": fingerprints}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return context
