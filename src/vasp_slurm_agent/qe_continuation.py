"""Reviewed QE continuations and bounded repairs in new run folders."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from .config import ClusterConfig
from . import qe_context, qe_workflow

MAX_REPAIRS = 2
_CLUSTER_KEYS = ("host", "user", "port", "qe_command", "qe_bin_dir", "setup_commands")
_POST_KEYS = {"line_density", "kpoints", "qpoints", "qmesh", "q_mesh", "phonon_grid", "qpoint", "deltae", "tr2_ph", "asr",
              "nstep", "cell_relax", "etot_conv_thr", "forc_conv_thr", "images", "num_of_images", "nstep_path", "path_thr"}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid saved run record.")
    return value


def _index(state, choice):
    if isinstance(choice, str):
        matches = [i for i, stage in enumerate(state["stages"]) if stage["folder"] == choice]
        choice = matches[0] if len(matches) == 1 else None
    if type(choice) is not int or not 0 <= choice < len(state["stages"]):
        raise ValueError("Choose an existing QE stage.")
    return choice


def _settings(parameters, *, convergence=True):
    result = {key: deepcopy(value) for key, value in parameters.items() if key not in _POST_KEYS}
    native = {key: deepcopy(value) for key, value in result.pop("namelists", {}).items() if key in {"system", "electrons"}}
    electrons = native.get("electrons", {})
    for key in ("startingpot", "startingwfc", "electron_maxstep", "mixing_beta", "diagonalization",
                "diago_cg_maxiter", "diago_david_ndim", "diago_full_acc"):
        electrons.pop(key, None)
    result.pop("electron_maxstep", None)
    if not convergence:
        electrons.pop("conv_thr", None)
        result.pop("conv_thr", None)
    if electrons:
        native["electrons"] = electrons
    else:
        native.pop("electrons", None)
    if native:
        result["namelists"] = native
    return result


def inspect_continuation(source_run, source_stage):
    root = Path(source_run).expanduser().resolve()
    context = qe_context.load_run_context(root)
    state = qe_workflow.read_state(root)
    qe_workflow._check_config(root, state)
    index = _index(state, source_stage)
    stage = state["stages"][index]
    if not context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value"):
        raise ValueError("Continue only from an accepted QE stage.")
    artifact = next((item for item in context["artifacts"] if item["id"] == f"stage_{index + 1}.CONTCAR" and item["status"] == "verified"), None)
    if stage["name"] in {"eph_ph", "eph_lambda", "epw_interpolate"}:
        # Reuse the submitted Cartesian frame, which CIF serialization can rotate.
        relative = stage["source_path"]
        expected = _read(root / "plan.json")["sources"][relative]
        if stage["structure_from"] is not None:
            parent = state["stages"][stage["structure_from"]]
            expected = (parent.get("result") or {}).get("final_native_structure_sha256")
            if not expected:
                raise ValueError("The relaxed source has no verified native structure.")
            relative = parent["folder"] + "/outputs/CONTCAR"
        if qe_workflow._digest(root / relative) != expected:
            raise ValueError("The accepted spectrum's structure changed.")
        from . import qe
        from pymatgen.core import Structure
        xml = root / stage["folder"] / "outputs/data-file-schema.xml"
        native = qe._xml_structure(qe._required(qe.ET.parse(xml).getroot(), "output/atomic_structure"))
        from . import qe_epw
        if not qe_epw._matches_structure_file(native, Structure.from_file(root / relative), root / relative):
            raise ValueError("The accepted spectrum differs from its submitted structure.")
        artifact = {"path": relative, "sha256": expected}
    if artifact is None:
        raise ValueError("This stage has no verified final structure. Select its preceding SCF or relaxation.")
    config = ClusterConfig.load(root / "config.json")
    checkpoint_path = root / stage["folder"] / "outputs" / "checkpoint.json"
    checkpoint = _read(checkpoint_path) if checkpoint_path.is_file() else {}
    checkpoint_verified = (stage["name"] in {"scf", "eph_ph"} and bool(checkpoint)
                           and qe_workflow._digest(checkpoint_path) == (stage.get("result") or {}).get("checkpoint_sha256")
                           and "scratch/dft_agent.save/data-file-schema.xml" in checkpoint
                           and any("charge-density" in name for name in checkpoint))
    proof = None
    if checkpoint_verified:
        try:
            qe_workflow._validate_checkpoint(checkpoint, "eph_lambda" if stage["name"] == "eph_ph" else "scf")
        except ValueError:
            checkpoint_verified = False
    if stage["name"] == "epw_interpolate":
        from . import qe_epw
        if (not checkpoint or qe_workflow._digest(checkpoint_path)
                != (stage.get("result") or {}).get("checkpoint_sha256")):
            raise ValueError("The accepted Wannier checkpoint receipt changed or is missing.")
        qe_workflow._validate_checkpoint(checkpoint, "epw_aniso")
        evidence = root / stage["folder"] / "outputs"
        native = qe_epw.read_checkpoint(evidence,
            acoustic_cutoff_cm1=stage["parameters"].get("acoustic_cutoff_cm1", 5.0))
        required = {**native["sha256"], "dft_agent.a2f": qe_workflow._digest(evidence / "dft_agent.a2f")}
        for name, digest in required.items():
            path = evidence / name
            if checkpoint.get(name) != {"sha256": digest, "size": path.stat().st_size}:
                raise ValueError("The remote Wannier checkpoint differs from the accepted native evidence.")
        checkpoint_verified = True
    if checkpoint_verified:
        proof = {"source_run_id": state["run_id"], "source_stage": stage["folder"],
                 "source_context_sha256": context["context_sha256"],
                 "remote_directory": state["remote_root"] + "/" + stage["folder"],
                 "checkpoint_sha256": stage["result"]["checkpoint_sha256"],
                 "checkpoint": checkpoint, "cluster": {key: getattr(config, key) for key in _CLUSTER_KEYS},
                 "pseudos": _read(root / stage["folder"] / "outputs" / "pseudo_manifest.json")}
    return {"structure_path": str(root / artifact["path"]), "structure_sha256": artifact["sha256"],
            "parameters": deepcopy(stage["parameters"]), "source_stage": index,
            "context_sha256": context["context_sha256"], "checkpoint": proof,
            "config": config, "source_run_id": state["run_id"], "source_folder": stage["folder"],
            "source_task": stage["name"]}


def _destination(source, destination):
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a separate folder for the new run.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("The destination already contains a run.")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _attach_seed(root, state, inspected, *, skip_scf):
    proof_path = root / "source" / "checkpoint.json"
    qe_workflow._write(proof_path, inspected["checkpoint"])
    reference = {"receipt": "source/checkpoint.json", "receipt_sha256": qe_workflow._digest(proof_path)}
    plan = _read(root / "plan.json")
    plan["sources"][reference["receipt"]] = reference["receipt_sha256"]
    if skip_scf:
        obsolete = state["stages"].pop(0)
        plan["stages"].pop(0)
        shutil.rmtree(root / obsolete["folder"])
        for stage in state["stages"]:
            for key in ("restart_from", "structure_from"):
                value = stage.get(key)
                if value == 0:
                    stage[key] = None
                    stage["external_checkpoint"] = reference
                elif value is not None:
                    stage[key] -= 1
    else:
        stage = state["stages"][0]
        stage["external_checkpoint"] = reference
        # A new SCF still verifies convergence. Only its initial guess is reused.
        recipe = deepcopy(stage["parameters"])
        recipe.setdefault("namelists", {}).setdefault("electrons", {}).update(startingpot="file", startingwfc="file")
        stage["parameters"] = recipe
        stage["materialized"] = False
        qe_workflow._materialize(root, state, inspected["config"], stage)
    for spec, stage in zip(plan["stages"], state["stages"]):
        for key in ("parameters", "restart_from", "structure_from"):
            spec[key] = deepcopy(stage[key])
        if stage.get("external_checkpoint"):
            spec["external_checkpoint"] = deepcopy(stage["external_checkpoint"])
    qe_workflow._write(root / "plan.json", plan)
    state["plan_sha256"] = qe_workflow._digest(root / "plan.json")
    return qe_workflow._save(root, state, "Accepted QE checkpoint attached to the new plan.")


def prepare_continuation(source_run, source_stage, run_dir, config, tasks,
                         parameters=None, stage_parameters=None, expected_context_sha256=None,
                         reuse_checkpoint=True):
    source, target = Path(source_run).expanduser().resolve(), Path(run_dir).expanduser().resolve()
    _destination(source, target)
    inspected = inspect_continuation(source, source_stage)
    if expected_context_sha256 is not None and expected_context_sha256 != inspected["context_sha256"]:
        raise ValueError("The accepted source changed after planning. Review the plan again.")
    config = config or inspected["config"]
    if config.engine != "qe":
        raise ValueError("Select Quantum ESPRESSO for this continuation.")
    if inspected["source_task"] == "eph_lambda":
        return _prepare_eliashberg(source, target, inspected, config, tasks,
                                  parameters, stage_parameters)
    if inspected["source_task"] == "eph_ph":
        return _prepare_qsum(source, target, inspected, config, tasks,
                             parameters, stage_parameters)
    if inspected["source_task"] == "epw_interpolate":
        if "epw_aniso" in tasks and not reuse_checkpoint:
            raise ValueError("Anisotropic EPW continuation requires the accepted remote Wannier checkpoint.")
        return _prepare_wannier_epw(source, target, inspected, config, tasks,
                                   parameters, stage_parameters)
    recipe = {**inspected["parameters"], **deepcopy(parameters or {})}
    first_recipe = {**recipe, **deepcopy((stage_parameters or [{}])[0])}
    same_cluster = all(getattr(config, key) == getattr(inspected["config"], key) for key in _CLUSTER_KEYS)
    compatible = bool(reuse_checkpoint and inspected["checkpoint"] and same_cluster
                      and config.qe_pseudopotentials == inspected["config"].qe_pseudopotentials
                      and _settings(first_recipe, convergence=False) == _settings(inspected["parameters"], convergence=False))
    with tempfile.TemporaryDirectory(prefix=".qe-continue-", dir=target.parent) as directory:
        temporary = Path(directory)
        structure = temporary / "source.vasp"
        shutil.copyfile(inspected["structure_path"], structure)
        if qe_workflow._digest(structure) != inspected["structure_sha256"]:
            raise ValueError("The source structure changed during preparation.")
        prepared = temporary / "run"
        state = qe_workflow.prepare_plan(structure, prepared, config, tasks, recipe, stage_parameters=stage_parameters)
        skip = (compatible and tasks[0] not in {"scf", "relax", "vc-relax", "neb"}
                and state["stages"][0]["name"] == "scf"
                and _settings(first_recipe) == _settings(inspected["parameters"]))
        compatible = compatible and (skip or state["stages"][0]["name"] in {"scf", "relax", "vc-relax"})
        if compatible:
            state = _attach_seed(prepared, state, inspected, skip_scf=skip)
        qe_workflow._write(prepared / "parent.json", {
            "schema_version": 1, "engine": "qe", "source_run_id": inspected["source_run_id"],
            "source_stage": inspected["source_folder"], "source_context_sha256": inspected["context_sha256"],
            "source_structure_sha256": inspected["structure_sha256"], "electronic_state_reused": compatible,
            "scf_reused": bool(skip), "note": "Verified checkpoint reused." if compatible else "Structure reused; electronic state recalculated."})
        if qe_context.load_run_context(source)["context_sha256"] != inspected["context_sha256"]:
            raise ValueError("Source evidence changed during preparation.")
        os.replace(prepared, target)
    return state


def _freeze_eph_source(source, prepared, plan, inspected):
    evidence = source / inspected["source_folder"] / "outputs"
    records = _read(evidence / "artifact_manifest.json")
    retained = {"qe-input.json", "structure.cif", "data-file-schema.xml", "dft_agent.dyn0"}
    if inspected["source_task"] == "eph_lambda":
        retained.update({"lambda.out", "lambda.dat", "alpha2F.dat"})
    retained.update(name for name in records if re.fullmatch(r"elph_dir/(?:elph\.inp_lambda\.[0-9]+|a2Fq2r\.[0-9]+\.[0-9]+)", name))
    for name in sorted(retained):
        if name not in records or qe_workflow._digest(evidence / name) != records[name]["sha256"]:
            raise ValueError("The accepted electron-phonon evidence changed.")
        relative = "source/electron-phonon/" + name
        destination = prepared / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence / name, destination)
        plan["sources"][relative] = records[name]["sha256"]


def _prepare_qsum(source, target, inspected, config, tasks, parameters, stage_parameters):
    if tasks != ["electron_phonon"] and tasks != ("electron_phonon",):
        raise ValueError("Continue these DFPT data with electron_phonon and eph_integration=q_sum.")
    if stage_parameters is not None and len(stage_parameters) != 1:
        raise ValueError("Supply one electron-phonon recipe.")
    changes = {**deepcopy(parameters or {}), **deepcopy((stage_parameters or [{}])[0])}
    recipe = {**inspected["parameters"], **changes}
    if recipe.get("eph_integration") != "q_sum":
        raise ValueError("Choose eph_integration=q_sum explicitly to integrate the accepted DFPT modes.")
    allowed = {"eph_integration", "mu_star", "phonon_smearing_thz", "omega_max_thz"}
    if any(key not in allowed and value != inspected["parameters"].get(key) for key, value in changes.items()):
        raise ValueError("Changing the electronic or phonon grids requires a new DFPT calculation.")
    if not inspected["checkpoint"] or any(getattr(config, key) != getattr(inspected["config"], key) for key in _CLUSTER_KEYS):
        raise ValueError("Use the original cluster and QE installation for the accepted DFPT checkpoint.")
    with tempfile.TemporaryDirectory(prefix=".qe-qsum-", dir=target.parent) as directory:
        prepared = Path(directory) / "run"
        state = qe_workflow.prepare_plan(inspected["structure_path"], prepared, config, tasks, recipe)
        plan = _read(prepared / "plan.json")
        for stage in state["stages"][:-1]:
            shutil.rmtree(prepared / stage["folder"])
        state["stages"], plan["stages"] = state["stages"][-1:], plan["stages"][-1:]
        _freeze_eph_source(source, prepared, plan, inspected)
        proof_path = prepared / "source/checkpoint.json"
        qe_workflow._write(proof_path, inspected["checkpoint"])
        reference = {"receipt": "source/checkpoint.json", "receipt_sha256": qe_workflow._digest(proof_path)}
        plan["sources"][reference["receipt"]] = reference["receipt_sha256"]
        for item in (state["stages"][0], plan["stages"][0]):
            item.update(restart_from=None, source_evidence="source/electron-phonon", external_checkpoint=reference)
        stage = state["stages"][0]
        stage.update(materialized=False, metadata={})
        qe_workflow._write(prepared / "plan.json", plan)
        state["plan_sha256"] = qe_workflow._digest(prepared / "plan.json")
        qe_workflow._materialize(prepared, state, config, stage)
        qe_workflow._write(prepared / "parent.json", {
            "schema_version": 1, "engine": "qe", "source_run_id": inspected["source_run_id"],
            "source_stage": inspected["source_folder"], "source_context_sha256": inspected["context_sha256"],
            "integration": "q_sum", "note": "Separate q-point mode sum; any earlier interpolation failure remains unresolved."})
        if qe_context.load_run_context(source)["context_sha256"] != inspected["context_sha256"]:
            raise ValueError("The DFPT evidence changed during preparation.")
        qe_workflow._save(prepared, state, "Accepted DFPT data prepared for mode-sum integration.")
        os.replace(prepared, target)
    return state


def _prepare_eliashberg(source, target, inspected, config, tasks, parameters, stage_parameters):
    """Freeze an accepted spectrum; EPW needs no remote wavefunction restart."""
    from . import qe_epw

    if not tasks or any(task not in {"epw_iso", "epw_tc"} for task in tasks):
        raise ValueError("Continue this spectrum with epw_iso or epw_tc. Use its SCF stage for other calculations.")
    if stage_parameters is not None and len(stage_parameters) != len(tasks):
        raise ValueError("Supply one parameter override per EPW task.")
    base = {"mu_star": inspected["parameters"]["mu_star"], **deepcopy(parameters or {})}
    recipes = [{**base, **deepcopy(overrides)} for overrides in (stage_parameters or [{}] * len(tasks))]
    for task, recipe in zip(tasks, recipes):
        qe_epw.validate_parameters(recipe, task)
    with tempfile.TemporaryDirectory(prefix=".qe-eliashberg-", dir=target.parent) as directory:
        prepared = Path(directory) / "run"
        # Use the same preparation contract as a fresh coupled calculation,
        # then replace its prerequisites with the accepted, frozen evidence.
        state = qe_workflow.prepare_plan(inspected["structure_path"], prepared, config,
            ["electron_phonon", *tasks], stage_parameters=[inspected["parameters"], *recipes])
        plan = _read(prepared / "plan.json")
        removed = [stage for stage in state["stages"] if stage["request_index"] == 0]
        state["stages"] = [stage for stage in state["stages"] if stage["request_index"] != 0]
        plan["stages"] = [spec for spec in plan["stages"] if spec["request_index"] != 0]
        for stage in removed:
            shutil.rmtree(prepared / stage["folder"])
        _freeze_eph_source(source, prepared, plan, inspected)
        for spec, stage in zip(plan["stages"], state["stages"]):
            for item in (spec, stage):
                item.pop("evidence_from", None)
                item.update(source_evidence="source/electron-phonon", request_index=item["request_index"] - 1)
            stage.update(metadata={}, materialized=False)
        plan.update(tasks=list(tasks), parameters=base, stage_parameters=recipes)
        state.update(tasks=list(tasks), task=tasks[0] if len(tasks) == 1 else "pipeline", parameters=base)
        qe_workflow._write(prepared / "plan.json", plan)
        state["plan_sha256"] = qe_workflow._digest(prepared / "plan.json")
        for stage in state["stages"]:
            qe_workflow._materialize(prepared, state, config, stage)
        qe_workflow._write(prepared / "parent.json", {
            "schema_version": 1, "engine": "qe", "source_run_id": inspected["source_run_id"],
            "source_stage": inspected["source_folder"], "source_context_sha256": inspected["context_sha256"],
            "spectrum_reused": True, "electronic_state_reused": False,
            "note": "Accepted electron-phonon spectrum copied with its native evidence."})
        if qe_context.load_run_context(source)["context_sha256"] != inspected["context_sha256"]:
            raise ValueError("The source spectrum changed during preparation.")
        qe_workflow._save(prepared, state, "Accepted spectrum prepared for EPW.")
        os.replace(prepared, target)
    return state


def _freeze_wannier_source(source, prepared, plan, inspected, *, include_checkpoint):
    from . import qe_epw

    evidence = source / inspected["source_folder"] / "outputs"
    records = _read(evidence / "artifact_manifest.json")
    metadata = _read(evidence / "qe-input.json")
    retained = {"qe-input.json", "epw.in", "epw.out", "structure.cif", "data-file-schema.xml",
                "dft_agent.wout", "dft_agent.win", "dft_agent.a2f", "crystal.fmt", "selecq.fmt",
                *metadata["input_sha256"]}
    if include_checkpoint:
        native = qe_epw.read_checkpoint(evidence,
            acoustic_cutoff_cm1=inspected["parameters"].get("acoustic_cutoff_cm1", 5.0))
        retained.update(native["sha256"])
    for name in sorted(retained):
        path = evidence / name
        if (not qe_context._artifact_name(name) or path.is_symlink()
                or not path.resolve().is_relative_to(evidence.resolve())
                or name not in records or not path.is_file()
                or path.stat().st_size != records[name]["size"]
                or qe_workflow._digest(path) != records[name]["sha256"]):
            raise ValueError("The accepted Wannier evidence changed or is missing.")
        relative = "source/wannier/" + name
        destination = prepared / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        plan["sources"][relative] = records[name]["sha256"]


def _prepare_wannier_epw(source, target, inspected, config, tasks, parameters, stage_parameters):
    """Reuse an accepted Wannier spectrum or native matrix elements."""
    from . import qe_epw

    if not tasks or any(task not in {"epw_iso", "epw_tc", "epw_aniso"} for task in tasks):
        raise ValueError("Continue this Wannier spectrum with epw_iso, epw_tc or epw_aniso.")
    if stage_parameters is not None and len(stage_parameters) != len(tasks):
        raise ValueError("Supply one parameter override per EPW task.")
    needs_checkpoint = "epw_aniso" in tasks
    if needs_checkpoint and (not inspected["checkpoint"]
            or any(getattr(config, key) != getattr(inspected["config"], key) for key in _CLUSTER_KEYS)
            or config.qe_pseudopotentials != inspected["config"].qe_pseudopotentials):
        raise ValueError("Use the original cluster, QE installation and pseudopotentials for the accepted Wannier checkpoint.")
    base = deepcopy(parameters or {})
    recipes = [{**base, **deepcopy(overrides)} for overrides in (stage_parameters or [{}] * len(tasks))]
    for task, recipe in zip(tasks, recipes):
        qe_epw.validate_parameters(recipe, task)
    with tempfile.TemporaryDirectory(prefix=".qe-wannier-epw-", dir=target.parent) as directory:
        prepared = Path(directory) / "run"
        # The removed source stages use their original UPF metadata. Scalar
        # EPW itself needs no pseudopotentials or remote source checkpoint.
        preparation_config = replace(config, qe_pseudopotentials=inspected["config"].qe_pseudopotentials)
        state = qe_workflow.prepare_plan(inspected["structure_path"], prepared, preparation_config,
            ["wannier_electron_phonon", *tasks], stage_parameters=[inspected["parameters"], *recipes])
        plan = _read(prepared / "plan.json")
        removed = [stage for stage in state["stages"] if stage["request_index"] == 0]
        state["stages"] = [stage for stage in state["stages"] if stage["request_index"] != 0]
        plan["stages"] = [spec for spec in plan["stages"] if spec["request_index"] != 0]
        for stage in removed:
            shutil.rmtree(prepared / stage["folder"])
        _freeze_wannier_source(source, prepared, plan, inspected, include_checkpoint=needs_checkpoint)
        reference = None
        if needs_checkpoint:
            proof_path = prepared / "source/checkpoint.json"
            qe_workflow._write(proof_path, inspected["checkpoint"])
            reference = {"receipt": "source/checkpoint.json", "receipt_sha256": qe_workflow._digest(proof_path)}
            plan["sources"][reference["receipt"]] = reference["receipt_sha256"]
        for spec, stage in zip(plan["stages"], state["stages"]):
            for item in (spec, stage):
                item.pop("evidence_from", None)
                item.pop("auxiliary_from", None)
                item.update(restart_from=None, structure_from=None, source_evidence="source/wannier",
                            request_index=item["request_index"] - 1)
                if stage["name"] == "epw_aniso":
                    item["external_checkpoint"] = reference
            stage.update(metadata={}, materialized=False)
        plan.update(tasks=list(tasks), parameters=base, stage_parameters=recipes)
        state.update(tasks=list(tasks), task=tasks[0] if len(tasks) == 1 else "pipeline", parameters=base)
        qe_workflow._write(prepared / "plan.json", plan)
        state["plan_sha256"] = qe_workflow._digest(prepared / "plan.json")
        config.save(prepared / "config.json")
        state["config_sha256"] = qe_workflow._digest(prepared / "config.json")
        for stage in state["stages"]:
            qe_workflow._materialize(prepared, state, config, stage)
        qe_workflow._write(prepared / "parent.json", {
            "schema_version": 1, "engine": "qe", "source_run_id": inspected["source_run_id"],
            "source_stage": inspected["source_folder"], "source_context_sha256": inspected["context_sha256"],
            "source_structure_sha256": inspected["structure_sha256"], "wannier_checkpoint_reused": needs_checkpoint,
            "wannier_spectrum_reused": any(task in {"epw_iso", "epw_tc"} for task in tasks),
            "note": "Accepted Wannier evidence reused; source calculations are not repeated."})
        if qe_context.load_run_context(source)["context_sha256"] != inspected["context_sha256"]:
            raise ValueError("The source Wannier evidence changed during preparation.")
        qe_workflow._save(prepared, state, "Accepted Wannier evidence prepared for EPW.")
        os.replace(prepared, target)
    return state


def _failed_snapshot(run_dir):
    root = Path(run_dir).expanduser().resolve()
    state = qe_workflow.read_state(root)
    if state.get("engine") != "qe" or state.get("status") not in {"failed", "needs_attention"}:
        raise ValueError("Choose a collected failed QE run.")
    qe_workflow._check_config(root, state)
    if state.get("cancel_requested") or state.get("remote_failures") or state.get("last_error"):
        raise ValueError("Reconnect and collect the original job before preparing a repair.")
    index = _index(state, state["current_stage"])
    stage = state["stages"][index]
    if stage.get("scheduler_state") not in {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"} or not stage.get("job_id"):
        raise ValueError("Slurm has not confirmed a terminal job state.")
    if any(s.get("job_id") or s.get("staged") for s in state["stages"][index + 1:]):
        raise ValueError("Later stages already started. Reconcile the original run.")
    output = root / stage["folder"] / "outputs"
    result, manifest = _read(output / "result.json"), _read(output / "artifact_manifest.json")
    if result != stage.get("result") or result.get("success") is not False or not result.get("input_identity_verified"):
        raise ValueError("The failed output has no verified input identity.")
    for name, receipt in manifest.items():
        path = output / name
        if (not qe_context._artifact_name(name) or path.is_symlink()
                or not path.resolve().is_relative_to(output.resolve())
                or qe_workflow._digest(path) != receipt.get("sha256")):
            raise ValueError("Collected solver evidence changed. Recollect the original job.")
    execution = _read(output / "execution.json")
    if execution.get("job_id") != stage["job_id"] or execution.get("runtime_sha256") != stage.get("runtime_sha256"):
        raise ValueError("The failed execution receipt does not identify this job.")
    for name, expected in stage["metadata"]["input_sha256"].items():
        if qe_workflow._digest(root / stage["folder"] / "inputs" / name) != expected or qe_workflow._digest(output / name) != expected:
            raise ValueError("The failed inputs changed.")
    context = qe_context.load_run_context(root)
    for position in range(index):
        if not context["facts"].get(f"stage_{position + 1}.accepted", {}).get("value"):
            raise ValueError("A preceding stage no longer has accepted evidence.")
    previous_repair = _read(root / "repair.json") if (root / "repair.json").is_file() else {}
    attempt = int(previous_repair.get("attempt", 0)) + 1
    tails = []
    for name in (stage["metadata"]["output_file"], "slurm.out", "slurm.err"):
        if name in manifest:
            with (output / name).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65536))
                tails.append(stream.read().decode("utf-8", errors="replace"))
    # Band tables can push the actual failure outside the retained log tail.
    solver_log = stage["metadata"]["output_file"]
    if stage["name"] in {"bands", "nscf"} and solver_log in manifest:
        with (output / solver_log).open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if re.search(r"c_bands:.*eigenvalues not converged", line, re.I):
                    tails.append(line[:512])
                    break
    snapshot = {"context": context["context_sha256"], "config": qe_workflow._digest(root / "config.json"),
                "manifest": manifest, "result": result, "attempt": attempt}
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    return root, state, stage, index, "\n".join(tails), attempt, digest


def _repair_electrons(stage):
    """Read the options used by the failed calculation, including defaults."""
    parameters = stage["parameters"]
    return {"electron_maxstep": parameters.get("electron_maxstep", 120), "mixing_beta": 0.7,
            **qe_workflow.qe._extra(parameters, "electrons"),
            **stage.get("metadata", {}).get("namelists", {}).get("electrons", {})}


def _update_electrons(parameters, changes):
    electrons = qe_workflow.qe._extra(parameters, "electrons")
    electrons.update(changes)
    native = parameters.setdefault("namelists", {})
    section = next((key for key in native if key.lower() == "electrons"), "electrons")
    native[section] = electrons


def repair_options(run_dir):
    root, state, stage, index, logs, attempt, digest = _failed_snapshot(run_dir)
    options = []
    if attempt <= MAX_REPAIRS:
        if stage["scheduler_state"] == "TIMEOUT":
            try:
                _longer_walltime(ClusterConfig.load(root / "config.json").walltime)
            except ValueError:
                pass
            else:
                options.append({"action": "increase_walltime", "reason": "Slurm recorded a time limit."})
        if stage["name"] in {"scf", "relax", "vc-relax"} and re.search(r"convergence\s+NOT\s+achieved", logs, re.I):
            electrons = _repair_electrons(stage)
            if electrons["electron_maxstep"] < 1000:
                options.append({"action": "increase_electronic_steps", "reason": "The SCF iteration limit was reached."})
            if electrons["mixing_beta"] > 0.05:
                options.append({"action": "reduce_mixing", "reason": "SCF did not converge; a smaller mixing step may help."})
        unconverged = stage["name"] in {"bands", "nscf"} and re.search(r"c_bands:.*eigenvalues not converged", logs, re.I)
        # Davidson stops when its subspace overlap turns singular (cdiaghg/rdiaghg); CG avoids that subspace.
        singular = re.search(r"Error in routine [cr]diaghg|S matrix not positive definite|problems computing cholesky", logs, re.I)
        if unconverged or singular:
            electrons = _repair_electrons(stage)
            if (electrons.get("diagonalization") != "cg" or electrons.get("diago_cg_maxiter", 20) < 1000
                    or not electrons.get("diago_full_acc", False)):
                reason = ("Some band eigenvalues did not converge." if unconverged
                          else "The Davidson diagonalization stopped on a singular overlap matrix.")
                options.append({"action": "use_cg_diagonalization", "reason": reason + " Use CG with more iterations and full accuracy for empty states."})
    return {"engine": "qe", "snapshot_sha256": digest, "attempt": attempt, "max_attempts": MAX_REPAIRS,
            "stage": stage["name"], "scheduler_state": stage["scheduler_state"], "actions": options,
            "requires_review": True, "submits_jobs": False}


def _longer_walltime(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = map(int, clock.split(":"))
    total = (int(days) * 86400 + hours * 3600 + minutes * 60 + seconds) * 2
    if total > 7 * 86400:
        raise ValueError("Automatic repairs cannot request more than seven days.")
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _repair_frozen_epw(root, prepared, state, index, config):
    """Retry frozen EPW stages without rebuilding their source calculations."""
    original = _read(root / "plan.json")
    specs = deepcopy(original["stages"][index:])
    if any(spec["name"] not in {"epw_iso", "epw_tc", "epw_aniso"}
           or not spec.get("source_evidence")
           or any(spec.get(key) is not None for key in ("restart_from", "structure_from", "evidence_from", "auxiliary_from"))
           for spec in specs):
        raise ValueError("This repair requires independent EPW stages with frozen source evidence.")
    prepared.mkdir()
    for name, digest in original["sources"].items():
        source, destination = root / name, prepared / name
        if (source.is_symlink() or not source.resolve().is_relative_to(root)
                or not destination.resolve().is_relative_to(prepared) or not source.is_file()
                or qe_workflow._digest(source) != digest):
            raise ValueError("The frozen EPW source changed or is missing.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    run_id = "qe-" + uuid.uuid4().hex[:16]
    stages = []
    for position, spec in enumerate(specs):
        spec["request_index"] = position
        stages.append({**deepcopy(spec), "job_name": f"{run_id}-{position + 1:02d}-{spec['name']}",
            "job_id": None, "status": "planned", "metadata": {}, "staged": False,
            "materialized": False, "result": None})
    tasks = [spec["name"] for spec in specs]
    plan = {**deepcopy(original), "tasks": tasks, "stages": specs,
            "stage_parameters": [deepcopy(spec["parameters"]) for spec in specs]}
    qe_workflow._write(prepared / "plan.json", plan)
    config.save(prepared / "config.json")
    new_state = {"schema_version": 2, "engine": "qe", "run_id": run_id,
        "task": tasks[0] if len(tasks) == 1 else "pipeline", "tasks": tasks,
        "parameters": deepcopy(plan["parameters"]), "formula": state["formula"],
        "status": "planned", "current_stage": 0, "stages": stages, "history": [],
        "created_at": qe_workflow.utc_now(), "last_error": None, "cancel_requested": False,
        "plan_sha256": qe_workflow._digest(prepared / "plan.json"),
        "config_sha256": qe_workflow._digest(prepared / "config.json"),
        "remote_root": config.remote_root.rstrip("/") + "/" + run_id}
    for old_stage, stage in zip(state["stages"][index:], stages):
        qe_workflow._materialize(prepared, new_state, config, stage)
        if stage["metadata"]["input_sha256"] != old_stage["metadata"]["input_sha256"]:
            raise ValueError("Repaired EPW inputs differ from the frozen calculation.")
        external = stage.get("external_checkpoint")
        if stage["name"] == "epw_aniso" and not external:
            raise ValueError("The anisotropic EPW source checkpoint is missing.")
        if external:
            proof_path = prepared / external["receipt"]
            if (not proof_path.resolve().is_relative_to(prepared)
                    or qe_workflow._digest(proof_path) != external["receipt_sha256"]):
                raise ValueError("The frozen EPW checkpoint receipt changed.")
            proof = _read(proof_path)
            if proof["cluster"] != {key: getattr(config, key) for key in _CLUSTER_KEYS}:
                raise ValueError("The EPW checkpoint must use its original cluster and executable.")
            qe_workflow._validate_checkpoint(proof["checkpoint"], stage["name"])
            evidence = prepared / stage["source_evidence"]
            hashes = {**stage["metadata"]["checkpoint_sha256"],
                      "dft_agent.a2f": qe_workflow._digest(evidence / "dft_agent.a2f")}
            if any(proof["checkpoint"].get(name) != {"sha256": digest, "size": (evidence / name).stat().st_size}
                   for name, digest in hashes.items()):
                raise ValueError("The EPW checkpoint differs from its frozen native evidence.")
    qe_workflow._write(prepared / "parent.json", {
        "schema_version": 1, "engine": "qe", "source_run_id": state["run_id"],
        "source_stage": state["stages"][index]["folder"], "frozen_epw_source_reused": True})
    return qe_workflow._save(prepared, new_state, "Frozen EPW inputs prepared with a longer time limit.")


def prepare_repair(source_run, run_dir, action, expected_snapshot_sha256):
    """Apply one reviewed numerical repair; preserve the original job."""
    options = repair_options(source_run)
    if options["snapshot_sha256"] != expected_snapshot_sha256:
        raise ValueError("Failure evidence changed. Review a new repair proposal.")
    if action not in {entry["action"] for entry in options["actions"]}:
        raise ValueError("Choose an available repair action within the retry budget.")
    root, state, failed, index, logs, attempt, snapshot = _failed_snapshot(source_run)
    target = Path(run_dir).expanduser().resolve()
    _destination(root, target)
    config = ClusterConfig.load(root / "config.json")
    remaining = state["stages"][index:]
    request_index = failed.get("request_index")
    if type(request_index) is int:
        plan = _read(root / "plan.json")
        tasks = plan["tasks"][request_index:]
        overrides = plan.get("stage_parameters") or [{}] * len(plan["tasks"])
        recipes = [{**deepcopy(plan["parameters"]), **deepcopy(value)} for value in overrides[request_index:]]
    else:
        tasks = [stage["name"] for stage in remaining]
        recipes = [deepcopy(stage["parameters"]) for stage in remaining]
    changes = {}
    if action == "increase_walltime":
        changes["walltime"] = {"before": config.walltime, "after": _longer_walltime(config.walltime)}
        config = replace(config, walltime=changes["walltime"]["after"])
    elif action == "increase_electronic_steps":
        before = _repair_electrons(failed)["electron_maxstep"]
        after = min(1000, int(before) * 2)
        if after <= before:
            raise ValueError("The electronic iteration limit is already at the repair cap.")
        _update_electrons(recipes[0], {"electron_maxstep": after})
        changes["electron_maxstep"] = {"before": before, "after": after}
    elif action == "use_cg_diagonalization":
        electrons = _repair_electrons(failed)
        updated = {"diagonalization": "cg", "diago_cg_maxiter": min(1000, max(200, int(electrons.get("diago_cg_maxiter", 20)) * 2)),
                   "diago_full_acc": True}
        changes.update({key: {"before": electrons.get(key), "after": value} for key, value in updated.items()})
        _update_electrons(recipes[0], updated)
    else:
        before = _repair_electrons(failed)["mixing_beta"]
        after = max(0.05, min(0.3, float(before) / 2))
        if after >= before:
            raise ValueError("The mixing step is already at the repair floor.")
        _update_electrons(recipes[0], {"mixing_beta": after})
        changes["mixing_beta"] = {"before": before, "after": after}
    with tempfile.TemporaryDirectory(prefix=".qe-repair-", dir=target.parent) as directory:
        prepared = Path(directory) / "run"
        seed = next((position for position in range(index - 1, -1, -1)
                     if state["stages"][position]["name"] in {"scf", "relax", "vc-relax"}), None)
        if action == "increase_walltime" and failed.get("source_evidence") and failed["name"].startswith("epw_"):
            new_state = _repair_frozen_epw(root, prepared, state, index, config)
        elif seed is not None:
            new_state = prepare_continuation(root, seed, prepared, config, tasks, parameters=recipes[0], stage_parameters=recipes)
        else:
            new_state = qe_workflow.prepare_plan(root / failed["source_path"], prepared, config, tasks, recipes[0], stage_parameters=recipes)
        qe_workflow._write(prepared / "repair.json", {"schema_version": 1, "engine": "qe", "attempt": attempt,
            "max_attempts": MAX_REPAIRS, "action": action, "changes": changes,
            "source_run_id": state["run_id"], "source_stage": failed["folder"], "source_snapshot_sha256": snapshot})
        if _failed_snapshot(root)[-1] != snapshot:
            raise ValueError("Failure evidence changed during preparation.")
        os.replace(prepared, target)
    return new_state
