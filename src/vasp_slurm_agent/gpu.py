"""GPU execution receipts; this file also runs alone on the compute node."""
from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


SLURM_FIELDS = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_NNODES",
                "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")
# VASP's own startup message, documented at https://vasp.at/wiki/Stdout.
OFFLOAD = re.compile(r"^\s*(?:Offloading|OpenACC runtime) initialized\s*\.\.\.\s*(\d+) GPUs? detected\s*$", re.MULTILINE)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(root, receipt):
    path = Path(root) / "execution.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def requested_resources(config):
    return {key: getattr(config, key) for key in ("nodes", "tasks", "gpus_per_node", "cpus_per_task")}


def validate_inputs(folder):
    for name in ("INCAR", "seed.INCAR"):
        path = Path(folder) / name
        if not path.is_file():
            continue
        text = "\n".join(re.split(r"[!#]", line, maxsplit=1)[0] for line in path.read_text().splitlines())
        for key, value in re.findall(r"(?:^|[;\n])\s*(NCORE|NPAR)\s*=\s*([^;\n]+)", text, flags=re.IGNORECASE):
            if key.upper() == "NPAR" or value.strip() != "1":
                raise ValueError(f"GPU runs require NCORE=1 (or unset) and NPAR unset. Check {name}.")


def _allocation_error(slurm, requested):
    if not slurm.get("SLURM_JOB_ID"):
        return "GPU diagnostics require a Slurm allocation; VASP was not launched."
    assigned = any(slurm.get(key, "").strip() not in {"", "N/A", "NoDevFiles", "-1"}
                   for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))
    count = slurm.get("SLURM_GPUS_ON_NODE", "")
    assigned = assigned or (count.isdigit() and int(count) > 0)
    if not assigned:
        return "Slurm did not report assigned GPUs; VASP was not launched."
    nodes = slurm.get("SLURM_JOB_NUM_NODES") or slurm.get("SLURM_NNODES")
    if nodes and (not nodes.isdigit() or int(nodes) != requested["nodes"]):
        return "The allocated node count differs from the requested GPU layout."
    tasks = slurm.get("SLURM_NTASKS")
    if tasks and (not tasks.isdigit() or int(tasks) != requested["tasks"]):
        return "The allocated MPI task count differs from the requested GPU layout."
    return None


def begin(root, requested):
    slurm = {key: os.environ[key] for key in SLURM_FIELDS if key in os.environ}
    receipt = {"compute_backend": "gpu", "job_id": slurm.get("SLURM_JOB_ID"),
               "started_at": _now(), "tasks": slurm.get("SLURM_NTASKS"),
               "requested_resources": requested, "slurm": slurm,
               "gpu_runtime": {"host": socket.gethostname(), "scope": "batch_node", "devices": []},
               "gpu_verified": False}
    try:
        error = _allocation_error(slurm, requested)
        if error:
            raise ValueError(error)
        probe = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
        if probe.returncode:
            raise ValueError("NVIDIA GPU discovery failed on the compute node. Check the GPU allocation and driver.")
        devices = [{"uuid": row[0].strip(), "name": row[1].strip(), "driver": row[2].strip()}
                   for row in csv.reader(probe.stdout.splitlines()) if len(row) == 3 and row[0].strip().startswith("GPU-")]
        receipt["gpu_runtime"]["devices"] = devices
        if not devices:
            raise ValueError("No NVIDIA GPU was detected on the compute node; VASP was not launched.")
        _write(root, receipt)
        return receipt
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        receipt.update(error=str(exc), exit_code=2, finished_at=_now())
        _write(root, receipt)
        raise ValueError(str(exc)) from exc


def backend_evidence(root):
    root = Path(root)
    names = ["vasp.stdout"]
    if (root / "seed.stdout").exists():
        names.append("seed.stdout")
    evidence = []
    for name in names:
        path = root / name
        # Startup banners occur before the calculation; avoid loading long logs.
        with path.open(errors="replace") as stream:
            text = stream.read(262144)
        matches = [match for match in OFFLOAD.finditer(text) if int(match[1]) > 0]
        if not matches:
            raise ValueError(f"VASP GPU offloading was not confirmed in {name}. Check the GPU-enabled VASP executable.")
        evidence.append({"file": name, "line": matches[0][0].strip(), "gpus_detected": int(matches[0][1])})
    return evidence


def finish(root, exit_code):
    root = Path(root)
    receipt = json.loads((root / "execution.json").read_text())
    receipt.update(exit_code=exit_code, finished_at=_now(), gpu_verified=False)
    try:
        receipt["vasp_gpu_evidence"] = backend_evidence(root)
        receipt["gpu_verified"] = True
    except (OSError, ValueError) as exc:
        receipt["error"] = str(exc)
        exit_code = exit_code or 2
        receipt["exit_code"] = exit_code
    _write(root, receipt)
    return exit_code


def verify_execution(root, config, job_id):
    try:
        receipt = json.loads((Path(root) / "execution.json").read_text())
        requested = requested_resources(config)
        if (receipt.get("compute_backend") != "gpu" or receipt.get("requested_resources") != requested
                or str(receipt.get("job_id")) != str(job_id)):
            raise ValueError("GPU execution evidence does not match this job and its requested resources.")
        error = _allocation_error(receipt.get("slurm", {}), requested)
        if error:
            raise ValueError(error)
        if not receipt.get("gpu_runtime", {}).get("devices"):
            raise ValueError(receipt.get("error") or "Compute-node GPU discovery is missing.")
        evidence = backend_evidence(root)
        if not receipt.get("gpu_verified") or receipt.get("exit_code") != 0:
            raise ValueError(receipt.get("error") or "The GPU launch did not finish successfully.")
        return {"verified": True, "requested_resources": requested,
                "slurm": receipt["slurm"], "runtime": receipt["gpu_runtime"], "vasp": evidence}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"verified": False, "reason": str(exc)}


if __name__ == "__main__":
    try:
        if sys.argv[1] == "begin":
            begin(Path.cwd(), json.loads(sys.argv[2]))
        elif sys.argv[1] == "finish":
            sys.exit(finish(Path.cwd(), int(sys.argv[2])))
        else:
            raise ValueError("Unknown GPU receipt operation.")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
