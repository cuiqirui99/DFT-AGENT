"""Explicit calculation variants using the ordinary run executor."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from . import workflow
from .config import ClusterConfig
from .explanation import load_run_context
from .transport import SSHTransport, worker_environment


def read_batch(batch_dir):
    return json.loads((Path(batch_dir) / "batch.json").read_text())


def _children(root, state):
    manifest = root / "batch-plan.json"
    if workflow._digest(manifest) != state.get("plan_sha256"):
        raise ValueError("The batch plan changed. Prepare a new batch.")
    plan = json.loads(manifest.read_text())
    config_file = root / "config.json"
    if (workflow._digest(config_file) if config_file.is_file() else None) != plan["config_sha256"]:
        raise ValueError("The batch cluster configuration changed. Restore the original settings.")
    if len(plan["runs"]) != len(state["runs"]):
        raise ValueError("Batch members differ from the frozen plan.")
    children = []
    for expected, member in zip(plan["runs"], state["runs"]):
        if any(member.get(key) != value for key, value in expected.items()):
            raise ValueError("Batch members differ from the frozen plan.")
        path = (root / member["folder"]).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Invalid batch member path.")
        child = workflow.read_state(path)
        if (child["run_id"] != member["run_id"] or child["plan_sha256"] != member["plan_sha256"]
                or child["config_sha256"] != plan["config_sha256"]):
            raise ValueError("The batch member changed. Restore its original run.")
        children.append((member, path, child))
    return children


def prepare_batch(structure_path, batch_dir, config, tasks, variants, parameters=None, stage_parameters=None):
    """Prepare explicit points offline. Strain is a fractional lattice change."""
    if not isinstance(variants, list) or not 1 <= len(variants) <= 256:
        raise ValueError("Supply 1–256 explicit calculation variants.")
    common = workflow._stage_recipes(tasks, dict(parameters or {}), stage_parameters)
    labels = set()
    for item in variants:
        if not isinstance(item, dict) or set(item) - {"label", "parameters", "stage_parameters", "strain", "operations"}:
            raise ValueError("Use label, parameters, stage_parameters, strain or operations for a variant.")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 120 or any(ord(c) < 32 for c in label):
            raise ValueError("Give each variant a short label.")
        if label in labels:
            raise ValueError("Variant labels must be unique.")
        labels.add(label)
        if not isinstance(item.get("parameters", {}), dict):
            raise ValueError("Variant parameters must be an object.")
        if "operations" in item and not isinstance(item["operations"], list):
            raise ValueError("Variant operations must be a list of structure edits.")
        workflow._stage_recipes(tasks, {}, item.get("stage_parameters"))
    root = Path(batch_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This batch already exists. Resume it or choose a new directory.")
    source = Path(structure_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dft-batch-", dir=root.parent))
    try:
        (temporary / "source").mkdir()
        frozen = temporary / "source" / source.name
        shutil.copyfile(source, frozen)
        if config is not None:
            config.save(temporary / "config.json")
        members = []
        for index, item in enumerate(variants):
            slug = re.sub(r"[^a-z0-9]+", "-", item["label"].lower()).strip("-")[:40] or "variant"
            folder = f"runs/{index + 1:03d}_{slug}"
            variant_source = frozen
            if item.get("operations"):
                from .structures import apply_structure_plan, preview_structure
                preview = preview_structure(frozen, item["operations"])
                edited = apply_structure_plan(frozen, temporary / "edits" / str(index + 1), preview)
                variant_source = Path(edited["files"]["poscar"])
            strain = item.get("strain")
            if strain is not None:
                from pymatgen.core import Structure
                from pymatgen.io.vasp import Poscar
                values = [strain] * 3 if type(strain) in (int, float) else strain
                if not isinstance(values, (list, tuple)) or len(values) != 3 or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value <= -1 for value in values
                ):
                    raise ValueError("Strain must be a fraction or three axis fractions, each greater than -1.")
                structure = Structure.from_file(variant_source)
                structure.apply_strain(values)
                variant_source = temporary / "strains" / str(index + 1) / "POSCAR"
                variant_source.parent.mkdir(parents=True)
                Poscar(structure, sort_structure=False).write_file(variant_source)
            overrides = workflow._stage_recipes(tasks, {}, item.get("stage_parameters"))
            recipes = [{**deepcopy(base), **deepcopy(item.get("parameters", {})), **override}
                       for base, override in zip(common, overrides)]
            if strain is not None:
                from .vasp import DEFAULTS
                if any(task == "relax" and recipe.get("cell_relax", DEFAULTS["cell_relax"])
                       for task, recipe in zip(tasks, recipes)):
                    raise ValueError("Strain comparisons require cell_relax=False during relaxation.")
            child = workflow.prepare_plan(variant_source, temporary / folder, config, tasks,
                                          parameters=parameters, stage_parameters=recipes)
            members.append({"label": item["label"], "folder": folder, "run_id": child["run_id"],
                            "plan_sha256": child["plan_sha256"]})
        plan = {"schema_version": 1, "tasks": list(tasks), "parameters": deepcopy(parameters or {}),
                "stage_parameters": deepcopy(stage_parameters), "variants": deepcopy(variants),
                "source": frozen.relative_to(temporary).as_posix(), "source_sha256": workflow._digest(frozen),
                "config_sha256": workflow._digest(temporary / "config.json") if config is not None else None,
                "runs": members}
        workflow._write(temporary / "batch-plan.json", plan)
        state = {"schema_version": 1, "batch_id": "dft-batch-" + uuid.uuid4().hex[:16],
                 "status": "planned", "cancel_requested": False, "created_at": workflow.utc_now(),
                 "plan_sha256": workflow._digest(temporary / "batch-plan.json"),
                 "runs": [{**member, "status": "planned", "last_error": None} for member in members]}
        workflow._write(temporary / "batch.json", state)
        _summarize_batch(temporary)
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _refresh(root, state):
    statuses = []
    for member, path, child in _children(root, state):
        member.update(status=child["status"], last_error=child.get("last_error"))
        statuses.append(child["status"])
    if all(status == "succeeded" for status in statuses):
        state["status"] = "succeeded"
    elif all(status in workflow.TERMINAL for status in statuses):
        state["status"] = "cancelled" if state["cancel_requested"] and all(
            status in {"succeeded", "cancelled"} for status in statuses) else "needs_attention"
    elif all(status == "planned" for status in statuses):
        state["status"] = "planned"
    else:
        state["status"] = "running"
    state["updated_at"] = workflow.utc_now()
    workflow._write(root / "batch.json", state)
    return state


def attach_batch_config(batch_dir, config):
    """Attach cluster settings to a prepared batch and all of its members."""
    if config is None:
        raise ValueError("Save Cluster setup before attaching it to a batch.")
    root = Path(batch_dir).expanduser().resolve()
    with workflow._config_attachment_lock(root), ExitStack() as locks:
        state = read_batch(root)
        if state["status"] != "planned":
            raise ValueError("Cluster settings can only be attached before the first submission.")
        for _, path, _ in _children(root, state):
            locks.enter_context(workflow._config_attachment_lock(path))
        # A child worker may have changed state while its lock was pending.
        _children(root, state)
        with workflow._config_attachment_workspace(root) as temporary:
            prepared = temporary / "batch"
            shutil.copytree(root, prepared, ignore=shutil.ignore_patterns(".state.lock", ".worker.lock"))
            plan = json.loads((prepared / "batch-plan.json").read_text())
            replacements = []
            for member in plan["runs"]:
                child = workflow._prepare_config_attachment(prepared / member["folder"], config)
                replacements.extend(member["folder"] + "/" + name for name in workflow._config_attachment_files(child))
            config.save(prepared / "config.json")
            plan["config_sha256"] = workflow._digest(prepared / "config.json")
            workflow._write(prepared / "batch-plan.json", plan)
            state["plan_sha256"] = workflow._digest(prepared / "batch-plan.json")
            state = _refresh(prepared, state)
            _summarize_batch(prepared)
            replacements.extend(("config.json", "batch-plan.json", "batch.json", "summary.json", "summary.csv"))
            workflow._commit_config_attachment(root, prepared, replacements, temporary / "original")
            return state


def advance_batch(batch_dir, transport=None):
    root = Path(batch_dir).expanduser().resolve()
    with workflow._lock(root):
        state = read_batch(root)
        for member, path, child in _children(root, state):
            if child["status"] not in workflow.TERMINAL:
                if state["cancel_requested"]:
                    workflow.cancel(path, transport)
                else:
                    workflow.advance(path, transport)
        state = _refresh(root, state)
        _summarize_batch(root)
        return state


def resume_batch(batch_dir, transport=None):
    root = Path(batch_dir).expanduser().resolve()
    with workflow._lock(root):
        state = read_batch(root)
        for member, path, child in _children(root, state):
            if child["status"] in {"needs_attention", "failed"}:
                workflow.resume(path, transport)
        return _refresh(root, state)


def cancel_batch(batch_dir, transport=None):
    root = Path(batch_dir).expanduser().resolve()
    with workflow._lock(root):
        state = read_batch(root)
        state["cancel_requested"] = True
        workflow._write(root / "batch.json", state)
        for member, path, child in _children(root, state):
            if child["status"] not in {"succeeded", "cancelled"}:
                workflow.cancel(path, transport)
        state = _refresh(root, state)
        _summarize_batch(root)
        return state


def summarize_batch(batch_dir):
    """Recheck retained outputs before exposing energies or ranking matched rows."""
    root = Path(batch_dir).expanduser().resolve()
    with workflow._lock(root):
        return _summarize_batch(root)


def _summarize_batch(root):
    state = read_batch(root)
    plan = json.loads((root / "batch-plan.json").read_text())
    rows, groups = [], {}
    for index, (member, path, child) in enumerate(_children(root, state)):
        context = load_run_context(path)
        facts = context["facts"]
        row = {"label": member["label"], "folder": member["folder"], "run_id": member["run_id"],
               "status": context["status"], "strain": plan["variants"][index].get("strain"),
               "energy_stage": None, "energy_ev": None, "energy_ev_per_atom": None,
               "functional": None, "hubbard_u": None, "soc": None, "total_moment": None,
               "band_gap_ev": None, "band_gap_scope": None, "band_gap_status": None,
               "comparison_group": None, "rank": None, "relative_energy_mev_per_atom": None,
               "context_sha256": context["context_sha256"], "reason": "No accepted SCF or relaxation energy."}
        for position in reversed(range(len(child["stages"]))):
            prefix = f"stage_{position + 1}."
            if facts.get(prefix + "accepted", {}).get("value") is not True:
                continue
            gap_status = facts.get(prefix + "band_gap_status", {}).get("value")
            if gap_status is not None:
                row.update(band_gap_ev=facts.get(prefix + "band_gap", {}).get("value"),
                           band_gap_scope=facts.get(prefix + "band_gap_scope", {}).get("value"),
                           band_gap_status=gap_status)
                break
        for position in reversed(range(len(child["stages"]))):
            stage = child["stages"][position]
            prefix = f"stage_{position + 1}."
            def value(name):
                return facts.get(prefix + name, {}).get("value")
            if stage["name"] not in {"scf", "relax"} or value("accepted") is not True:
                continue
            energy, per_atom = value("energy"), value("energy_per_atom")
            if any(type(number) not in (float, int) or not math.isfinite(number) for number in (energy, per_atom)):
                continue
            metadata = stage["metadata"]
            method = metadata.get("method", {})
            row.update(energy_stage=stage["folder"], energy_ev=energy, energy_ev_per_atom=per_atom,
                       functional=method.get("functional"), hubbard_u=method.get("hubbard_u"),
                       soc=method.get("soc"), total_moment=value("total_moment"), reason="")
            if context["status"] == "succeeded" and metadata.get("method_comparison_fingerprint"):
                compatible = {"task": stage["name"], "method": metadata["method_comparison_fingerprint"],
                              "structure": metadata["input_sha256"]["POSCAR"],
                              "settings": {key: val for key, val in metadata["parameters"].items()
                                           if key not in {"spin", "magmom"}}}
                group = hashlib.sha256(json.dumps(compatible, sort_keys=True, allow_nan=False).encode()).hexdigest()
                groups.setdefault(group, []).append(row)
            else:
                row["reason"] = "The complete run must pass current checks before comparison."
            break
        rows.append(row)
    comparisons = []
    for group, matched in groups.items():
        if len(matched) < 2:
            continue
        matched.sort(key=lambda row: row["energy_ev_per_atom"])
        for rank, row in enumerate(matched, 1):
            row.update(comparison_group=group, rank=rank,
                       relative_energy_mev_per_atom=1000 * (row["energy_ev_per_atom"] - matched[0]["energy_ev_per_atom"]))
        comparisons.append({"group": group, "labels": [row["label"] for row in matched],
                            "lowest_energy_variant": matched[0]["label"]})
    status = state["status"]
    if status == "succeeded" and any(row["status"] != "succeeded" for row in rows):
        status = "needs_attention"
    summary = {"schema_version": 1, "batch_id": state["batch_id"], "status": status,
               "rows": rows, "comparisons": comparisons,
               "scope": "Energy order among supplied, matched variants only; not a magnetic ground-state search."}
    workflow._write(root / "summary.json", summary)
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
    os.replace(temporary, root / "summary.csv")
    return summary


def watch_batch(batch_dir, interval=20, transport=None):
    worker_password = os.environ.pop("_DFT_AGENT_WORKER_PASSWORD", None)
    root = Path(batch_dir).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise ValueError("This batch has no cluster settings yet. Save Cluster setup and attach it before submitting.")
    with (root / ".worker.lock").open("a") as lock:
        try:
            workflow._acquire(lock, blocking=False)
        except BlockingIOError:
            return read_batch(root)
        owned = transport is None
        try:
            _children(root, read_batch(root))
            transport = transport or SSHTransport(ClusterConfig.load(root / "config.json"), password=worker_password)
            workflow._write(root / "worker.json", {"pid": os.getpid(), "started_at": workflow.utc_now()})
            while True:
                state = advance_batch(root, transport)
                if state["status"] in workflow.TERMINAL:
                    bundle_batch(root)
                    workflow._announce(root, {"status": state["status"], "formula": "batch", "task": state.get("batch_id", "")})
                    return state
                time.sleep(interval)
        finally:
            try:
                if owned and transport is not None:
                    transport.close()
            finally:
                workflow._release(lock)


def start_batch_worker(batch_dir):
    from .runtime import cli_command
    root = Path(batch_dir).expanduser().resolve()
    with (root / "worker.log").open("ab") as stream:
        child = subprocess.Popen(cli_command("watch-batch", root),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True,
                                 env=worker_environment(ClusterConfig.load(root / "config.json")))
    return child.pid


def bundle_batch(batch_dir):
    root = Path(batch_dir).expanduser().resolve()
    summarize_batch(root)
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("batch.json", "batch-plan.json", "summary.json", "summary.csv"):
            archive.write(root / name, name)
        for name in ("proposal.json", "parent.json", "conversation.json"):
            if (root / name).is_file() and not (root / name).is_symlink():
                archive.write(root / name, name)
        for name in ("source", "structure_edit"):
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, str(path.relative_to(root)))
        for member, path, child in _children(root, read_batch(root)):
            archive.write(workflow.bundle_run(path), member["folder"] + "/results.zip")
    os.replace(target.with_suffix(".tmp"), target)
    return target
