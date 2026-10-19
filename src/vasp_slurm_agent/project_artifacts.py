"""Verified geometry transfers between calculations in one project."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Poscar

from . import continuation, explanation, qe_continuation
from .project_context import _run_path
from .project_files import _atomic, _checked, _lock


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _source(workspace, project_id, run_id):
    record = workspace.get_run(project_id, run_id)
    root = _run_path(workspace, project_id, record, workspace.project_paths(project_id)["runs"])
    if record["state"] is None:
        raise ValueError("The registered run is missing or changed.")
    return record, root


def _inspect(record, root, stage):
    if record["kind"] != "run" or record["engine"] not in {"vasp", "qe"}:
        raise ValueError("Select a VASP or QE run with an accepted structure.")
    helper = qe_continuation if record["engine"] == "qe" else continuation
    inspected = helper.inspect_continuation(root, stage)
    source = Path(inspected["structure_path"])
    if not source.is_relative_to(root) or source.is_symlink():
        raise ValueError("The structure must belong to the selected run.")
    if source.name != "CONTCAR" and source.name != "POSCAR" and source.suffix.lower() != ".vasp":
        raise ValueError("Select an accepted SCF or relaxation with a native POSCAR structure.")
    digest = (inspected.get("structure_sha256") or
              inspected.get("provenance", {}).get("source_structure_sha256"))
    if not digest or _hash(source.read_bytes()) != digest:
        raise ValueError("The accepted structure changed. Inspect the source again.")
    return inspected, source, digest


def inspect_run(workspace, project_id, run_id):
    """Reparse retained evidence; scheduler completion alone is not acceptance."""
    record, root = _source(workspace, project_id, run_id)
    if record["engine"] in {"python", "wannier90"}:
        from . import script_workflow
        context = script_workflow.load_run_context(root)
    else:
        context = explanation.load_run_context(root)
    stages = []
    for index, stage in enumerate(record["state"].get("stages", [])):
        accepted = context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value") is True
        available = False
        if accepted and record["engine"] in {"vasp", "qe"}:
            try:
                _inspect(record, root, index)
                available = True
            except (ValueError, OSError):
                pass
        stages.append({"index": index, "folder": stage.get("folder"), "name": stage.get("name"),
                       "accepted": accepted, "structure_available": available})
    if record["engine"] in {"python", "wannier90"}:
        accepted = context["facts"].get("run.accepted", {}).get("value") is True
    else:
        accepted = context["status"] == "succeeded"
    if record["kind"] == "run" and record["engine"] in {"vasp", "qe"}:
        accepted = accepted and bool(stages) and all(item["accepted"] for item in stages)
    # Re-reading closes the interval used to inspect available structures.
    reader = script_workflow.load_run_context if record["engine"] in {"python", "wannier90"} else explanation.load_run_context
    if reader(root)["context_sha256"] != context["context_sha256"]:
        raise ValueError("The run changed during inspection. Inspect it again.")
    return {"run_id": run_id, "engine": record["engine"], "status": context["status"],
            "accepted": bool(accepted), "classification": "accepted_result" if accepted else context["status"],
            "process_success": bool(accepted) or context["facts"].get("run.process_success", {}).get("value") is True,
            "context_sha256": context["context_sha256"], "stages": stages,
            "structure_available": any(item["structure_available"] for item in stages),
            "limits": context.get("limits", [])}


def _destination(workspace, project_id, relative):
    path = Path(relative)
    if not path.parts or path.parts[0] not in {"files", "artifacts"}:
        raise ValueError("Save transferred structures in this project's files or artifacts folder.")
    if path.name != "POSCAR" and path.suffix.lower() != ".vasp":
        raise ValueError("Use POSCAR or a .vasp filename for the native structure.")
    return _checked(workspace, project_id, relative)


def _transfer_evidence(workspace, project_id, source_run_id, stage, target_engine, destination):
    if target_engine not in {"vasp", "qe"}:
        raise ValueError("Choose VASP or QE as the target structure engine.")
    record, root = _source(workspace, project_id, source_run_id)
    if record["kind"] != "run" or record["engine"] not in {"vasp", "qe"}:
        raise ValueError("Select a VASP or QE run, not a batch or script.")
    context = explanation.load_run_context(root)
    if stage is None:
        selected = None
        for index in reversed(range(len(record["state"].get("stages", [])))):
            if context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value") is not True:
                continue
            try:
                selected = _inspect(record, root, index)
                break
            except (ValueError, OSError):
                continue
        if selected is None:
            raise ValueError("This run has no accepted native structure to transfer.")
    else:
        selected = _inspect(record, root, stage)
    inspected, source, digest = selected
    if inspected["context_sha256"] != context["context_sha256"]:
        raise ValueError("The source results changed during inspection.")
    data = source.read_bytes()
    if _hash(data) != digest:
        raise ValueError("The source structure changed during transfer.")
    structure = Poscar.from_str(data.decode("utf-8")).structure
    if (not structure.is_ordered or not len(structure) or structure.volume <= 0
            or not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.frac_coords).all()):
        raise ValueError("The accepted structure has invalid geometry.")
    index = inspected["source_stage"]
    source_stage = record["state"]["stages"][index]
    receipt = {"schema_version": 1, "kind": "structure_transfer", "project_id": project_id,
               "source_run_id": source_run_id, "source_native_run_id": record["run_id"],
               "source_engine": record["engine"], "source_stage": index,
               "source_stage_folder": source_stage["folder"],
               "source_context_sha256": context["context_sha256"], "source_structure_sha256": digest,
               "target_engine": target_engine, "path": Path(destination).as_posix(), "sha256": digest,
               "format": "POSCAR", "units": {"lattice": "angstrom", "positions": "fractional"},
               "species": [str(site.specie) for site in structure],
               "lattice": structure.lattice.matrix.tolist(), "fractional_positions": structure.frac_coords.tolist(),
               "electronic_state_reused": False, "magnetic_state_transferred": False,
               "pseudopotentials_transferred": False, "requires_fresh_scf": True}
    if explanation.load_run_context(root)["context_sha256"] != context["context_sha256"]:
        raise ValueError("The source results changed during transfer.")
    return receipt, data


def transfer_structure(workspace, project_id, source_run_id, stage, target_engine, destination):
    """Copy an accepted native structure; never translate electronic states."""
    project_root = workspace.project_paths(project_id)["root"]
    receipt, data = _transfer_evidence(workspace, project_id, source_run_id, stage, target_engine, destination)
    with _lock(project_root):
        created = []
        try:
            with workspace.edit_project(project_id) as event:
                target = _destination(workspace, project_id, destination)
                proof = _checked(workspace, project_id, str(Path(destination).with_name(target.name + ".provenance.json")))
                if target.exists() or proof.exists():
                    verify_transfer(workspace, project_id, receipt)
                else:
                    _atomic(target, data)
                    created.append(target)
                    _atomic(proof, (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
                    created.append(proof)
                    verify_transfer(workspace, project_id, receipt)
                    event("structure.transferred", {"source_run_id": source_run_id, "path": receipt["path"],
                                                    "sha256": receipt["sha256"], "target_engine": target_engine})
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        # Indexing can be retried without replacing the frozen transfer files.
        workspace.register_artifact(project_id, target, kind="structure")
        workspace.register_artifact(project_id, proof, kind="structure_provenance")
    return receipt


def verify_transfer(workspace, project_id, receipt):
    """Require the same source evidence and geometry before using a transfer."""
    if not isinstance(receipt, dict) or receipt.get("project_id") != project_id or receipt.get("kind") != "structure_transfer":
        raise ValueError("The structure transfer does not belong to this project.")
    target = _destination(workspace, project_id, receipt.get("path", ""))
    proof = _checked(workspace, project_id, str(Path(receipt["path"]).with_name(target.name + ".provenance.json")), must_exist=True)
    try:
        saved = json.loads(proof.read_text())
        if saved != receipt or not target.is_file() or _hash(target.read_bytes()) != receipt.get("sha256"):
            raise ValueError("The transferred structure or receipt changed.")
        expected, _ = _transfer_evidence(workspace, project_id, receipt["source_run_id"],
                                         receipt["source_stage"], receipt["target_engine"], receipt["path"])
        if expected != receipt:
            raise ValueError("The source results changed after the structure transfer.")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The structure transfer receipt is invalid.") from error
    return receipt
