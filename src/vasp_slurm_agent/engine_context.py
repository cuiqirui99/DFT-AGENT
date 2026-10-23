"""Verified ABINIT, CP2K and Elk facts for result explanations and follow-up planning.

Every number comes from re-parsing the hash-checked outputs in a temporary
copy; a saved result that disagrees with its own evidence is not trusted.
"""
from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .engine_workflow import ENGINES, LABELS, TASKS, adapter
from .qe_context import saved_matches
from .scheduler import TERMINAL_STATES

_KEYS = ("final_energy_ev", "final_energy_ev_per_atom", "final_max_force_ev_angstrom", "fermi_energy_ev",
         "band_gap", "total_magnetization_mu_b", "total_magnetization_vector_mu_b", "absolute_magnetization_mu_b",
         "stress_gpa", "pressure_gpa", "site_magnetization", "site_magnetization_scope", "spin_mode",
         "spin_orbit_coupling", "optics", "time_evolution", "molecular_dynamics", "phonons")
# Results of the program-specific calculations: (result key, field, label, unit).
_SPECIAL = {"time_evolution": (("simulation_time_fs", "Simulated time", "fs"), ("absorbed_energy_ev", "Energy absorbed from the pulse", "eV"),
                               ("peak_field_v_per_angstrom", "Peak electric field", "V/Å"),
                               ("initial_moment_mu_b", "First recorded spin moment", "μB"),
                               ("final_moment_mu_b", "Spin moment at the end", "μB"),
                               ("moment_change_percent", "Spin moment change", "%")),
            "molecular_dynamics": (("simulated_time_fs", "Simulated time", "fs"), ("steps", "Molecular dynamics steps", ""),
                                   ("target_temperature_k", "Thermostat temperature", "K"),
                                   ("mean_temperature_k", "Mean temperature, second half", "K"),
                                   ("temperature_std_k", "Temperature fluctuation, second half", "K"),
                                   ("conserved_drift_mev_per_atom_ps", "Drift of the conserved energy", "meV/atom/ps"),
                                   ("stationary_second_half", "No settling detected in second half", ""),
                                   ("potential_energy_trend_mev_per_atom", "Potential energy trend, second half", "meV/atom")),
            "phonons": (("max_frequency_cm1", "Highest phonon frequency", "cm-1"), ("min_frequency_cm1", "Lowest phonon frequency", "cm-1"),
                        ("imaginary_mode_count", "Imaginary frequencies along the path", ""),
                        ("gamma_frequencies_cm1", "Frequencies at Γ", "cm-1"), ("qmesh", "Phonon q mesh", ""),
                        ("dielectric_constant", "Electronic dielectric constant", ""))}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(name):
    return isinstance(name, str) and Path(name).name == name and name not in {".", ".."}


def _log_excerpt(path):
    from .explanation import _safe_text

    with path.open("rb") as stream:
        head = stream.read(8192)
        stream.seek(max(0, path.stat().st_size - 8192))
        tail = stream.read()
    first = head.decode("utf-8", errors="replace").splitlines()
    last = tail.decode("utf-8", errors="replace").splitlines()
    errors = [line for line in first + last if re.search(
        r"error|abort|failed|not converg|segmentation|double free|negative|not found", line, re.I)]
    lines = list(dict.fromkeys(line[:240] for line in [*errors[:6], *first[:3], *last[-5:]] if line.strip()))
    return _safe_text("\n".join(lines))[:2000]


@lru_cache(maxsize=24)
def _parse(engine, folder, task, names, digests):
    """Cache on actual bytes, never on a saved success flag."""
    with tempfile.TemporaryDirectory(prefix=f"{engine}-explain-") as temporary:
        target = Path(temporary)
        for name in names:
            shutil.copyfile(Path(folder) / name, target / name)
        result = adapter(engine).analyze_outputs(target, task, target / "structure.cif")
        generated = {name: _sha(target / name) for name in result.get("artifacts", [])
                     if _plain(name) and (target / name).is_file()}
        return result, generated


def _numbers(value):
    if value is None or type(value) is bool:
        return True
    if type(value) in {int, float}:
        return math.isfinite(value)
    return isinstance(value, list) and all(_numbers(item) for item in value)


_PARAMETER_LABELS = {"functional": "Exchange-correlation functional", "mesh": "k-point mesh", "kshift": "k-point mesh shift", "kspacing": "k-point spacing",
                     "spin": "Spin treatment", "soc": "Spin-orbit coupling", "electronic_type": "Occupation treatment",
                     "smearing_ev": "Smearing width", "cell_relax": "Cell relaxation", "ecut_ha": "Plane-wave cutoff",
                     "cutoff_ry": "Grid cutoff", "rel_cutoff_ry": "Relative grid cutoff", "basis_family": "Basis family",
                     "rgkmax": "R·Kmax", "gmaxvr": "G max", "hubbard": "Hubbard terms", "dos_mesh": "DOS k-point mesh",
                     "initial_moments": "Starting moments", "site_moments": "Starting site moments"}
_UNITS = {"smearing_ev": "eV", "ecut_ha": "Ha", "cutoff_ry": "Ry", "rel_cutoff_ry": "Ry", "kspacing": "1/Å"}


def load_run_context(run_dir, *, _include_parent=True):
    """Reparse retained solver evidence without contacting or changing a run."""
    from .explanation import _safe_text
    root = Path(run_dir).expanduser().resolve()
    fingerprints = {}

    def path(relative):
        candidate = root / relative
        if Path(relative).is_absolute() or candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError("A run artifact points outside its run folder.")
        return candidate

    def digest(relative):
        try:
            candidate = path(relative)
            value = _sha(candidate) if candidate.is_file() else None
        except (OSError, ValueError):
            value = None
        fingerprints[relative] = value
        return value

    def read(relative):
        try:
            if not digest(relative):
                return {}
            value = json.loads(path(relative).read_text())
            return value if isinstance(value, dict) else {}
        except (ValueError, OSError):
            return {}

    state, plan, proposal = read("run.json"), read("plan.json"), read("proposal.json")
    engine = state.get("engine")
    if engine not in ENGINES:
        raise ValueError("This is not a saved ABINIT, CP2K or Elk run.")
    label = LABELS[engine]
    stages = state.get("stages") or []
    specifications = plan.get("stages") or []
    valid_plan = bool(plan.get("engine") == engine and state.get("plan_sha256") == fingerprints.get("plan.json")
                      and len(stages) == len(specifications) and stages)
    valid_plan = valid_plan and all(all(stage.get(key) == value for key, value in spec.items())
                                    for stage, spec in zip(stages, specifications))
    for relative, expected in (plan.get("sources") or {}).items():
        valid_plan = bool(digest(relative) == expected and expected and valid_plan)
    limits = ["Solver convergence does not establish convergence with respect to cutoff, basis, k-point mesh or other numerical settings.",
              "Only the calculated magnetic configurations were tested; a global magnetic ground state is not established."]
    if not valid_plan:
        limits.append("The frozen plan or structure has changed; accepted numerical results are unavailable.")
    linked = bool(valid_plan and (proposal.get("execution") or {}).get("plan_sha256") == fingerprints.get("plan.json"))
    goal = _safe_text(proposal.get("goal")) if linked else ""
    dialogue = [{"role": entry["role"], "content": _safe_text(entry.get("content"))}
                for entry in proposal.get("dialogue", []) if linked and isinstance(entry, dict)
                and entry.get("role") in {"user", "assistant"}]
    facts, artifacts, accepted_results = {}, [], []

    def fact(identifier, text, value, source, unit=""):
        facts[identifier] = {"label": text, "value": value, "source": source, "unit": unit}

    for index, stage in enumerate(stages):
        prefix = f"stage_{index + 1}"
        task, folder = stage.get("name"), stage.get("folder")
        accepted = bool(valid_plan and task in TASKS and isinstance(folder, str) and re.fullmatch(r"[0-9]+_[a-z]+", folder))
        issue = "Accepted solver evidence is unavailable, incomplete or changed."
        failure_reason, current, regenerated, metadata, manifest = None, {}, {}, {}, {}
        verified, execution = False, {}
        if accepted:
            output, inputs = folder + "/outputs", folder + "/inputs"
            result = read(output + "/result.json")
            metadata = read(inputs + "/metadata.json")
            manifest = read(output + "/artifact_manifest.json")
            accepted = bool(stage.get("status") in {"succeeded", "needs_attention", "failed"} and result == stage.get("result")
                            and metadata == stage.get("metadata") and type(result.get("success")) is bool
                            and stage.get("scheduler_state") == result.get("scheduler_state")
                            and stage.get("scheduler_state") in TERMINAL_STATES)
            expected_inputs = metadata.get("input_sha256") or {}
            accepted = accepted and bool(expected_inputs)
            for name, expected in expected_inputs.items():
                accepted = bool(_plain(name) and expected and digest(inputs + "/" + name) == expected
                                and digest(output + "/" + name) == expected and accepted)
            verified_names = []
            for name, receipt in manifest.items():
                if not _plain(name) or not isinstance(receipt, dict) or digest(output + "/" + name) != receipt.get("sha256"):
                    accepted = False
                else:
                    verified_names.append(name)
            required = {*expected_inputs, "execution.json", "input_hashes.sha256", "library_manifest.json", "runtime.json"}
            accepted = accepted and required.issubset(verified_names)
            try:
                input_hashes = dict((fields[1], fields[0]) for fields in
                                    (line.split() for line in path(output + "/input_hashes.sha256").read_text().splitlines()) if len(fields) == 2)
            except (OSError, ValueError):
                input_hashes = {}
            accepted = accepted and input_hashes == expected_inputs
            execution = read(output + "/execution.json")
            accepted = bool(accepted
                            and execution.get("job_id") == stage.get("job_id")
                            and execution.get("runtime_sha256") == stage.get("runtime_sha256")
                            and digest(output + "/runtime.json") == stage.get("runtime_sha256")
                            and digest(output + "/library_manifest.json") == stage.get("library_manifest_sha256"))
            for dependency in (stage.get("structure_from"), stage.get("restart_from")):
                if dependency is not None and (type(dependency) is not int or not 0 <= dependency < index or accepted_results[dependency] is None):
                    accepted = False
            verified = accepted
            if verified and metadata.get("output_file") in verified_names:
                try:
                    names = tuple(sorted(verified_names))
                    current, regenerated = _parse(engine, str(path(output)), task, names,
                                                  tuple(fingerprints.get(output + "/" + name) for name in names))
                    if not current.get("success"):
                        failure_reason = _safe_text(current.get("reason"))
                        issue = failure_reason or issue
                    accepted = bool(current.get("success") and result.get("success") and stage.get("status") == "succeeded"
                                    and stage.get("scheduler_state") == "COMPLETED"
                                    and type(execution.get("exit_code")) is int and execution["exit_code"] == 0
                                    and execution.get("finished_at"))
                    if accepted and any(key in result and not saved_matches(result[key], current.get(key)) for key in _KEYS):
                        accepted = False
                        issue = "Saved numerical results differ from the verified solver output."
                except (ValueError, OSError, KeyError, TypeError):
                    accepted = False
            else:
                accepted = False
            if accepted and current.get("final_structure"):
                accepted = bool(digest(output + "/final.cif") == result.get("final_structure_sha256")
                                and regenerated.get("final.cif") == result.get("final_structure_sha256"))
            if accepted and result.get("final_native_structure_sha256"):
                accepted = bool(digest(output + "/CONTCAR") == result["final_native_structure_sha256"]
                                and regenerated.get("CONTCAR") == result["final_native_structure_sha256"])
        accepted_results.append(current if accepted else None)
        fact(prefix + ".task", "Calculation", task, "plan.json")
        fact(prefix + ".program", "Program", label, "plan.json")
        fact(prefix + ".accepted", f"{task}: accepted result", accepted, "run.json")
        status = stage.get("status", "unavailable")
        if status == "succeeded" and not accepted:
            status = "needs_attention"
        fact(prefix + ".status", f"{task}: current status", status, "run.json")
        fact(prefix + ".evidence_verified", f"{task}: checked execution evidence", verified, "run.json")
        if not accepted:
            if verified:
                diagnostic = [f"Scheduler ended in {stage['scheduler_state']}."]
                fact(prefix + ".scheduler_state", f"{task}: scheduler state", stage["scheduler_state"], "run.json")
                code = execution.get("exit_code")
                if type(code) is int:
                    diagnostic.append(f"{label} exited with code {code}.")
                    fact(prefix + ".exit_code", f"{task}: program exit code", code, folder + "/outputs/execution.json")
                if failure_reason:
                    diagnostic.append(failure_reason)
                for field, suffix in (("converged_electronic", "electronic_converged"), ("converged_ionic", "ionic_converged")):
                    if type(current.get(field)) is bool:
                        fact(prefix + "." + suffix, f"{task}: {field}", current[field], folder + "/outputs/" + metadata["output_file"])
                for name in dict.fromkeys(("slurm.err", metadata.get("stdout_file"))):
                    if name and name in verified_names:
                        tail = _log_excerpt(path(folder + "/outputs/" + name))
                        if tail.strip():
                            fact(prefix + ".log." + name, f"{task}: checked log excerpt", tail, folder + "/outputs/" + name)
                            diagnostic.append(tail)
                failure_reason = _safe_text(" ".join(diagnostic))[:3500]
                issue = failure_reason
            fact(prefix + ".availability", f"{task}: availability", issue, "run.json")
            if failure_reason:
                fact(prefix + ".failure_reason", f"{task}: verified failure", failure_reason,
                     folder + "/outputs/" + str(metadata.get("output_file", "")))
            continue
        source = folder + "/outputs/" + stage["metadata"]["output_file"]
        parameters = stage["metadata"].get("parameters", {})
        for key, value in sorted(parameters.items()):
            if key in _PARAMETER_LABELS and (_numbers(value) or isinstance(value, (str, dict, list))):
                fact(prefix + ".parameter." + key, f"{task}: {_PARAMETER_LABELS[key]}", value,
                     folder + "/outputs/" + stage["metadata"]["input_file"], _UNITS.get(key, ""))
        for key, value in (("mesh", stage["metadata"].get("mesh")), ("dos_mesh", stage["metadata"].get("dos_mesh"))):
            if value is not None:
                fact(prefix + ".used." + key, f"{task}: {_PARAMETER_LABELS[key]} used", value, folder + "/outputs/" + stage["metadata"]["input_file"])
        for suffix, key, text, unit in (
            ("electronic_converged", "converged_electronic", "Electronic convergence", ""),
            ("ionic_converged", "converged_ionic", "Ionic convergence", ""),
            ("fermi_energy", "fermi_energy_ev", "Fermi energy", "eV"),
            ("force", "final_max_force_ev_angstrom", "Maximum force", "eV/Å"),
            ("total_magnetization", "total_magnetization_mu_b", "Total magnetization", "μB"),
            ("magnetization_vector", "total_magnetization_vector_mu_b", "Total magnetization vector", "μB"),
            ("absolute_magnetization", "absolute_magnetization_mu_b", "Absolute magnetization", "μB"),
            ("stress", "stress_gpa", "Stress tensor", "GPa"),
            ("pressure", "pressure_gpa", "Pressure", "GPa"),
        ):
            value = current.get(key)
            if value is not None and _numbers(value):
                fact(prefix + "." + suffix, f"{task}: {text}", value, source, unit)
        if task in {"scf", "relax"}:
            for suffix, key, text, unit in (("energy", "final_energy_ev", "Total energy", "eV"),
                                            ("energy_per_atom", "final_energy_ev_per_atom", "Energy per atom", "eV/atom")):
                if current.get(key) is not None:
                    fact(prefix + "." + suffix, f"{task}: {text}", current[key], source, unit)
        gap = current.get("band_gap") or {}
        for suffix, key, text, unit in (("band_gap", "gap_ev", "Sampled gap", "eV"),
                                        ("direct_gap", "direct_gap_ev", "Sampled direct gap", "eV"),
                                        ("band_gap_scope", "scope", "Gap sampling scope", ""),
                                        ("band_gap_status", "status", "Gap status", ""),
                                        ("vbm", "vbm_ev", "Sampled valence edge", "eV"),
                                        ("cbm", "cbm_ev", "Sampled conduction edge", "eV")):
            if gap.get(key) is not None:
                fact(prefix + "." + suffix, f"{task}: {text}", gap[key], source, unit)
        for index, (label, entry) in enumerate(((current.get("optics") or {}).get("spectra") or {}).items()):
            for key, text, unit in (("absorption_peak_ev", "strongest absorption", "eV"),
                                    ("dielectric_constant", "electronic dielectric constant, Re ε at zero frequency", "")):
                if entry.get(key) is not None and _numbers(entry[key]):
                    fact(f"{prefix}.optics_{index + 1}.{key}", f"{task}: {label}: {text}", entry[key], source, unit)
        for pair, distance in ((current.get("molecular_dynamics") or {}).get("rdf_first_maximum_angstrom") or {}).items():
            if _numbers(distance):
                fact(f"{prefix}.rdf.{len(facts)}", f"{task}: highest g(r) peak, {pair}", distance, source, "Å")
        for group, fields in _SPECIAL.items():
            values = current.get(group) or {}
            for key, text, unit in fields:
                if values.get(key) is not None and _numbers(values[key]):
                    fact(f"{prefix}.{group}.{key}", f"{task}: {text}", values[key], source, unit)
        for key, text in (("spin_mode", "Calculated spin mode"), ("spin_orbit_coupling", "Calculated spin-orbit coupling")):
            if current.get(key) is not None:
                fact(prefix + "." + key, text, current[key], source)
        for moment in current.get("site_magnetization", []):
            if type(moment.get("atom")) is int and _numbers(moment.get("moment_mu_b")):
                fact(f"{prefix}.site_moment.{moment['atom']}", f"Atom {moment['atom']} ({moment.get('element', '')}): local spin moment",
                     moment["moment_mu_b"], source, "μB")
        if current.get("site_magnetization_scope"):
            fact(prefix + ".site_magnetization_scope", "Local-moment definition", current["site_magnetization_scope"], source)
        for note in [*stage["metadata"].get("notes", []), *current.get("warnings", [])]:
            limits.append(_safe_text(note))
        if current.get("molecular_dynamics", {}).get("equilibration_note"):
            limits.append(current["molecular_dynamics"]["equilibration_note"])
        for name in current.get("artifacts", []):
            if not _plain(name):
                continue
            relative = folder + "/outputs/" + name
            current_hash = digest(relative)
            verified = bool(current_hash and (current_hash == regenerated.get(name) or current_hash == (manifest.get(name) or {}).get("sha256")))
            identifier = prefix + "." + name
            availability = "verified" if verified else "present_unverified" if current_hash else "unavailable"
            artifacts.append({"id": identifier, "label": f"{task}: {name}", "path": relative, "sha256": current_hash, "status": availability})
            fact(identifier, f"{task}: {name} availability", availability, relative)
    successful = sum(result is not None for result in accepted_results)
    status = state.get("status", "unavailable")
    if status == "succeeded" and successful != len(stages):
        status = "needs_attention"
    fact("run.status", "Current result status", status, "run.json")
    fact("run.accepted_stages", "Stages with accepted results", successful, "run.json", "stages")
    if valid_plan:
        fact("run.native_id", "Calculation ID", _safe_text(state.get("run_id")), "run.json")
    parent = read("parent.json") if _include_parent else {}
    if parent.get("kind") == "continuation":
        source_verified, unchanged = False, False
        try:
            matches = []
            for candidate in root.parent.iterdir():
                if candidate == root or candidate.is_symlink() or not candidate.is_dir():
                    continue
                source_state = candidate / "run.json"
                if source_state.is_symlink() or not source_state.is_file():
                    continue
                try:
                    saved = json.loads(source_state.read_text())
                except (ValueError, OSError):
                    continue
                if isinstance(saved, dict) and saved.get("run_id") == parent.get("source_run_id") and saved.get("engine") == engine:
                    matches.append(candidate)
            index = parent.get("source_stage")
            if len(matches) == 1 and type(index) is int and index >= 0 and valid_plan and accepted_results[0] is not None:
                source_root = matches[0]
                source_context = load_run_context(source_root, _include_parent=False)
                source_artifact = next((item for item in source_context["artifacts"]
                                        if item["id"] == f"stage_{index + 1}.CONTCAR"
                                        and item["status"] == "verified"), None)
                source_hash = parent.get("source_structure_sha256")
                source_verified = bool(source_artifact and source_artifact["sha256"] == source_hash
                    and source_hash in (plan.get("sources") or {}).values()
                    and _sha(source_root / "plan.json") == parent.get("source_plan_sha256")
                    and source_context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value") is True)
                if source_verified:
                    from .engine_common import load_structure, same_structure, structure_from_native
                    first = stages[0].get("metadata", {})
                    original = load_structure(source_root / source_artifact["path"])
                    unchanged = same_structure(structure_from_native(first["native_structure"]), original)
                    fact("continuation.source_task", "Accepted source calculation",
                         source_context["facts"][f"stage_{index + 1}.task"]["value"], "parent.json")
                    fact("continuation.source_stage", "Source stage (zero-based)", index, "parent.json")
                    fact("continuation.source_run_id", "Accepted source calculation ID",
                         _safe_text(parent["source_run_id"]), "parent.json")
                    fact("continuation.geometry_unchanged", "Starting geometry matches the accepted source", unchanged, "parent.json")
                    fact("continuation.electronic_state_reused", "Electronic state reused", False, "plan.json")
                fingerprints["continuation.source_context"] = source_context["context_sha256"]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            source_verified = False
        fact("continuation.source_verified", "Source geometry verified against an accepted parent", source_verified, "parent.json")
    if not any(key.endswith(".band_gap") for key in facts):
        limits.append("A band gap has not been quantified from these results.")
    if engine == "cp2k":
        limits.append("CP2K uses localized Gaussian basis sets; results depend on the basis family and are not basis-set converged by default.")
    if engine == "elk":
        limits.append("Elk is an all-electron code; total energies are not comparable with pseudopotential codes.")
    context = {"engine": engine, "goal": goal, "dialogue": dialogue, "status": status,
               "outcome": f"{successful} of {len(stages)} stages have accepted results.",
               "final_plan": {"tasks": plan.get("tasks", []), "parameters": plan.get("parameters", {})} if valid_plan else {},
               "facts": facts, "artifacts": artifacts, "limits": list(dict.fromkeys(limits))}
    context["context_sha256"] = hashlib.sha256(_canonical({"context": context, "files": fingerprints}).encode()).hexdigest()
    return context


def inspect_continuation(source_run, source_stage=None):
    """The verified native structure of an accepted relax or SCF stage, for a later calculation."""
    root = Path(source_run).expanduser().resolve()
    context = load_run_context(root)
    state = json.loads((root / "run.json").read_text())
    stages = state.get("stages") or []
    indices = [source_stage] if source_stage is not None else list(reversed(range(len(stages))))
    for index in indices:
        if type(index) is not int or not 0 <= index < len(stages):
            raise ValueError("Choose an existing stage.")
        if context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value") is not True:
            continue
        artifact = next((item for item in context["artifacts"]
                         if item["id"] == f"stage_{index + 1}.CONTCAR" and item["status"] == "verified"), None)
        if artifact is None:
            continue
        if load_run_context(root)["context_sha256"] != context["context_sha256"]:
            raise ValueError("The source results changed. Select the stage again.")
        return {"structure_path": str(root / artifact["path"]), "structure_sha256": artifact["sha256"],
                "parameters": json.loads(json.dumps(stages[index].get("parameters", {}))), "source_stage": index,
                "context_sha256": context["context_sha256"], "engine": state["engine"],
                "source_run_id": state["run_id"], "source_task": stages[index]["name"],
                "source_plan_sha256": state["plan_sha256"]}
    raise ValueError("This run has no accepted relaxation or SCF structure to continue from.")


def inspect_repair(source_run):
    """Use the failed stage's frozen geometry, never its unaccepted final positions."""
    from .workflow import _check_config
    root = Path(source_run).expanduser().resolve()
    state = json.loads((root / "run.json").read_text())
    _check_config(root, state)
    if state.get("status") not in {"failed", "needs_attention"} or state.get("last_error") or state.get("cancel_requested"):
        raise ValueError("Reconnect and collect the failed job before planning a repair.")
    index = state.get("current_stage")
    stages = state.get("stages") or []
    if type(index) is not int or not 0 <= index < len(stages):
        raise ValueError("The failed stage is unavailable.")
    context = load_run_context(root)
    prefix = f"stage_{index + 1}"
    stage = stages[index]
    if (stage.get("status") not in {"failed", "needs_attention"} or stage.get("scheduler_state") not in TERMINAL_STATES
            or context["facts"].get(prefix + ".evidence_verified", {}).get("value") is not True):
        raise ValueError("The failure has no verified terminal evidence. Reconnect first.")
    if any(item.get("job_id") or item.get("staged") for item in stages[index + 1:]):
        raise ValueError("A later stage has started. Reconcile this run first.")
    plan = json.loads((root / "plan.json").read_text())
    lineage = plan.get("repair")
    if lineage != state.get("repair"):
        raise ValueError("The repair record changed.")
    attempt = lineage.get("attempt") if lineage else 0
    if type(attempt) is not int or not 0 <= attempt < 2:
        raise ValueError("Two repair attempts have been used. Review a new calculation.")
    source = root / stage["folder"] / "inputs" / "structure.cif"
    expected = stage["metadata"]["input_sha256"].get("structure.cif")
    if source.is_symlink() or not source.resolve().is_relative_to(root) or not expected or _sha(source) != expected:
        raise ValueError("The failed stage's approved structure changed.")
    recipes = [{"task": item["name"], "parameters": deepcopy(item["parameters"])} for item in stages[index:]]
    allowed = {}
    if stage["scheduler_state"] == "COMPLETED":
        if context["facts"].get(prefix + ".electronic_converged", {}).get("value") is False:
            key, default, cap = "max_scf_steps", 200 if state["engine"] == "elk" else 100, 1000
        elif stage["name"] == "relax" and context["facts"].get(prefix + ".ionic_converged", {}).get("value") is False:
            key, default, cap = "max_ionic_steps", 100 if state["engine"] == "cp2k" else 50, 400
        else:
            key = None
        if key:
            before = recipes[0]["parameters"].get(key, default)
            if type(before) is int and before < cap:
                allowed[key] = {"before": before, "maximum": min(before * 2, cap)}
    if load_run_context(root)["context_sha256"] != context["context_sha256"]:
        raise ValueError("The failed run changed. Review it again.")
    return {"structure_path": str(source), "structure_sha256": expected, "source_stage": index,
            "context_sha256": context["context_sha256"], "engine": state["engine"], "source_run_id": state["run_id"],
            "source_task": stage["name"], "source_plan_sha256": state["plan_sha256"],
            "parameters": deepcopy(stage["parameters"]), "recipes": recipes, "allowed_changes": allowed,
            "attempt": attempt + 1,
            "diagnosis": context["facts"].get(prefix + ".failure_reason", {}).get("value", "Review the failed job.")}


def check_repair_plan(plan, inspected):
    """A repair cannot alter geometry, physics, convergence tolerances or later tasks."""
    from .engine_agent import _recipes
    if plan.get("operations"):
        raise ValueError("A repair cannot change the structure. Start a new calculation for structural edits.")
    recipes = _recipes(plan)
    previous = inspected["recipes"]
    if len(recipes) != len(previous) or [item["task"] for item in recipes] != [item["task"] for item in previous]:
        raise ValueError("Keep the failed stage and all remaining tasks in the repair.")
    allowed = inspected["allowed_changes"]
    changed = []
    for index, (old, new) in enumerate(zip(previous, recipes)):
        before, after = old["parameters"], new["parameters"]
        for key in before.keys() | after.keys():
            if before.get(key) == after.get(key):
                continue
            bounds = allowed.get(key) if index == 0 else None
            if (not bounds or type(after.get(key)) is not int
                    or not bounds["before"] < after[key] <= bounds["maximum"]):
                raise ValueError(f"A repair cannot change {key}. Keep the approved settings and review a separate calculation for method changes.")
            changed.append({"stage": index, "key": key, "before": bounds["before"], "after": after[key]})
    if not changed:
        raise ValueError("No supported parameter repair follows from this failure. Review the diagnostic and software environment before starting a new calculation.")
    return changed


def prepare_followup(source_run, source_stage, run_dir, config, plan, expected_context_sha256, *, repair=False):
    """Prepare a separate child from rechecked evidence; do not submit it."""
    from .engine_agent import prepare_task
    root, destination = Path(source_run).expanduser().resolve(), Path(run_dir).expanduser().resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Use a separate run folder for the continuation.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Choose a new run folder.")
    source = inspect_repair(root) if repair else inspect_continuation(root, source_stage)
    if source["context_sha256"] != expected_context_sha256 or plan.get("source_sha256") != source["structure_sha256"]:
        raise ValueError("The source results changed after planning. Review a new plan.")
    changes = check_repair_plan(plan, source) if repair else []
    provenance = {"schema_version": 1, "kind": "repair" if repair else "continuation", "source_engine": source["engine"],
                  "source_run_id": source["source_run_id"], "source_stage": source["source_stage"],
                  "source_task": source["source_task"], "source_plan_sha256": source["source_plan_sha256"],
                  "source_context_sha256": source["context_sha256"], "source_structure_sha256": source["structure_sha256"],
                  "electronic_state_reused": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".engine-followup-", dir=destination.parent) as directory:
        temporary = Path(directory)
        structure = temporary / ("structure.cif" if repair else "POSCAR")
        shutil.copyfile(source["structure_path"], structure)
        if _sha(structure) != source["structure_sha256"]:
            raise ValueError("The selected structure changed. Review a new plan.")
        child = temporary / "run"
        state = prepare_task(structure, child, config, plan)
        (child / "parent.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if repair:
            lineage = {"attempt": source["attempt"], "max_attempts": 2, "source_run_id": source["source_run_id"], "changes": changes}
            frozen = json.loads((child / "plan.json").read_text())
            frozen["repair"] = state["repair"] = lineage
            (child / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
            state["plan_sha256"] = _sha(child / "plan.json")
            (child / "run.json").write_text(json.dumps(state, indent=2) + "\n")
            proposal = json.loads((child / "proposal.json").read_text())
            proposal["execution"]["plan_sha256"] = state["plan_sha256"]
            (child / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
        if load_run_context(root)["context_sha256"] != source["context_sha256"]:
            raise ValueError("The source results changed during preparation. Review a new plan.")
        if destination.exists():
            destination.rmdir()
        os.replace(child, destination)
    return state
