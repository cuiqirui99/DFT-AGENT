"""Submit once under a stage lock (remote, stdlib only).

Persist intent and receipt; reconcile uncertain submissions without resubmitting.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
try:
    from .scheduler import get_scheduler
except ImportError:  # Uploaded beside this script on the cluster.
    from scheduler import get_scheduler


def save(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def dispatch(root: Path, name: str, reconcile_only: bool = False, scheduler="slurm", script="submit.sh"):
    adapter = get_scheduler(scheduler)
    adapter.job_name(name)
    if script not in {"submit.sh", "build.sh"}:
        raise ValueError("Invalid submission script")
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = root / "job.json"
        intent = root / "submit_intent.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value.get("job_name") != name or value.get("scheduler", "slurm") != scheduler or not adapter.valid_job_id(str(value.get("job_id", ""))):
                raise RuntimeError("Stored submission receipt does not match this run; no job action performed")
            return value
        if intent.exists():
            original = json.loads(intent.read_text())
            if original.get("job_name") != name or original.get("scheduler", "slurm") != scheduler or original.get("script", "submit.sh") != script:
                raise RuntimeError("Stored submission intent does not match this run; no job action performed")
            jobs = set()
            commands = adapter.reconcile_commands(name, original["date"])
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError("Cannot reconcile submission: scheduler query failed")
                jobs.update(adapter.matching_jobs(result.stdout, name))
            if len(jobs) != 1:
                raise RuntimeError("Submission outcome uncertain; no automatic resubmission. Inspect queue/accounting using job name " + name)
            value = {"job_id": jobs.pop(), "job_name": name, "reconciled": True}
            if scheduler != "slurm":
                value["scheduler"] = scheduler
            save(receipt, value)
            return value
        if reconcile_only:
            raise RuntimeError("No recorded submission exists; reconciliation cannot submit a new job")
        intent_value = {"job_name": name, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "created_at": datetime.now(timezone.utc).isoformat()}
        if scheduler != "slurm":
            intent_value["scheduler"] = scheduler
        if script != "submit.sh":
            intent_value["script"] = script
        save(intent, intent_value)
        result = subprocess.run(adapter.submit_command(name, script), cwd=root, capture_output=True, text=True, timeout=60)
        job = adapter.submission_job_id(result.stdout)
        if result.returncode or not job:
            save(root / "submission_error.json", {"returncode": result.returncode, "stderr": result.stderr, "stdout": result.stdout})
            raise RuntimeError("Scheduler did not acknowledge a job; inspect submission_error.json before any new run")
        value = {"job_id": job, "job_name": name, "reconciled": False}
        if scheduler != "slurm":
            value["scheduler"] = scheduler
        save(receipt, value)
        return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit once, or reconcile an existing Slurm job")
    parser.add_argument("root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument("--scheduler", choices=("slurm", "pbs"), default="slurm")
    parser.add_argument("--script", choices=("submit.sh", "build.sh"), default="submit.sh")
    args = parser.parse_args()
    print(json.dumps(dispatch(args.root, args.name, reconcile_only=args.reconcile_only, scheduler=args.scheduler, script=args.script)))
