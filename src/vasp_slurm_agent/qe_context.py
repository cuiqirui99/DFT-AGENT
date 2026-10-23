"""Verified QE facts for result explanations and continuation planning."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from . import qe


def saved_matches(saved, current):
    """Every saved value still matches a fresh analysis; fields added by a later version may be missing."""
    if isinstance(saved, dict):
        return isinstance(current, dict) and all(key in current and saved_matches(value, current[key]) for key, value in saved.items())
    if isinstance(saved, list):
        return isinstance(current, list) and len(saved) == len(current) and all(map(saved_matches, saved, current))
    return saved == current


def _science(task):
    if task.startswith("wannier_") or task == "epw_interpolate":
        from . import qe_wannier
        return qe_wannier
    if task.startswith("epw_"):
        from . import qe_epw
        return qe_epw
    if task.startswith("eph_"):
        from . import qe_eph
        return qe_eph
    return qe


def _artifact_name(name):
    from .qe_workflow import _EPW_RESULT_NAME
    return isinstance(name, str) and (Path(name).name == name and name not in {".", ".."}
                                    or bool(re.fullmatch(r"elph_dir/[A-Za-z0-9_.+-]+", name))
                                    or bool(_EPW_RESULT_NAME.fullmatch(name)))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=24)
def _parse(folder, task, names, digests):
    """Cache on actual bytes, never on a saved success flag."""
    with tempfile.TemporaryDirectory(prefix="qe-explain-") as temporary:
        target = Path(temporary)
        for name in names:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(folder) / name, target / name)
        result = _science(task).analyze_outputs(target, task, target / "structure.cif")
        generated = {name: _sha(target / name) for name in result.get("artifacts", [])
                     if _artifact_name(name) and (target / name).is_file()}
        return result, generated


def _species_label(value):
    return isinstance(value, str) and len(value) <= 3 and bool(re.fullmatch(r"[A-Z][a-z]?[A-Za-z0-9_]*", value))


def _finite_number(value):
    return type(value) in {int, float} and math.isfinite(value)


def _physics_namelists(value):
    if not isinstance(value, dict):
        return {}
    # Keep options that change the Hamiltonian, spin state or SCF accuracy.
    # Paths and arbitrary strings from native input never reach the model.
    numeric = {"system": {"nspin", "noncolin", "lspinorb", "tot_magnetization", "lambda",
                           "ecutwfc", "ecutrho", "nbnd", "degauss", "exx_fraction", "screening_parameter",
                           "nosym", "noinv", "no_t_rev", "lda_plus_u", "lda_plus_u_kind"},
               "electrons": {"conv_thr", "electron_maxstep", "mixing_beta", "diago_thr_init"}}
    indexed = re.compile(r"(?:starting_magnetization|angle1|angle2|fixed_magnetization|hubbard_[uj]|hubbard_j0|hubbard_alpha|hubbard_beta|starting_ns_eigenvalue)\(\s*[1-9][0-9]*(?:\s*,\s*[1-9][0-9]*){0,2}\s*\)")
    choices = {"input_dft": {"PBE", "LDA", "PZ", "PBE0", "HSE", "HSE06", "PBE_SOL", "PBESOL"},
               "occupations": {"fixed", "smearing", "tetrahedra", "tetrahedra_lin", "tetrahedra_opt"},
               "smearing": {"gaussian", "mp", "mv", "cold", "marzari-vanderbilt", "methfessel-paxton", "fd", "fermi-dirac"},
               "constrained_magnetization": {"none", "total", "atomic", "total direction", "atomic direction"}}
    result = {}
    for section, settings in value.items():
        if not isinstance(section, str) or section.lower() not in numeric or not isinstance(settings, dict):
            continue
        section = section.lower()
        clean = {}
        for key, item in settings.items():
            if not isinstance(key, str):
                continue
            key = key.lower()
            if (key in numeric[section] or section == "system" and indexed.fullmatch(key)) and (type(item) is bool or _finite_number(item)):
                clean[key] = item
            elif section == "system" and key in choices and isinstance(item, str):
                normalized = item.upper() if key == "input_dft" else item.lower()
                if normalized in choices[key]:
                    clean[key] = normalized
        if clean:
            result[section] = clean
    return result


def _parameters(value):
    """Keep numerical settings and known choices; omit paths and free text."""
    result = {}
    keys = set(qe.PARAMETER_KEYS)
    try:
        from . import qe_eph, qe_epw, qe_wannier
        keys.update(qe_eph.PARAMETER_KEYS)
        keys.update(qe_epw.PARAMETER_KEYS)
        keys.update(qe_wannier.PARAMETER_KEYS)
    except ImportError:
        pass
    for key, item in (value or {}).items():
        if key not in keys or key == "images":
            continue
        if key == "site_labels":
            if isinstance(item, list) and all(_species_label(label) for label in item):
                result[key] = list(item)
        elif key == "projections":
            if (isinstance(item, list) and item and all(isinstance(projection, str)
                    and re.fullmatch(r"[A-Za-z0-9_.,:=()+ -]+", projection) and projection.strip()
                    for projection in item)):
                result[key] = list(item)
        elif key == "starting_magnetization":
            if isinstance(item, dict):
                result[key] = {label: moment for label, moment in item.items()
                               if _species_label(label) and _finite_number(moment) and -1 <= moment <= 1}
        elif key == "hubbard":
            if isinstance(item, list):
                values = []
                for row in item:
                    if (not isinstance(row, dict) or not _species_label(row.get("label"))
                            or not isinstance(row.get("manifold"), str) or not re.fullmatch(r"[1-7][spdf]", row["manifold"])
                            or not _finite_number(row.get("U")) or row["U"] < 0):
                        continue
                    values.append({"label": row["label"], "manifold": row["manifold"], **{
                        name: row[name] for name in ("U", "J0") if _finite_number(row.get(name)) and row[name] >= 0}})
                result[key] = values
        elif key == "namelists":
            native = _physics_namelists(item)
            if native:
                result[key] = native
        elif isinstance(item, str):
            if key == "hubbard_projector" and item in {"atomic", "ortho-atomic", "pseudo"}:
                result[key] = item
            elif key == "functional" and item.upper() in {"PBE", "PBE0", "HSE", "HSE06", "LDA", "PZ", "PBE_SOL", "PBESOL"}:
                result[key] = item
            elif key == "spin" and item in {"none", "collinear", "noncollinear"}:
                result[key] = item
            elif key == "electronic_type" and item in {"metal", "insulator", "auto"}:
                result[key] = item
            elif key == "eph_integration" and item in {"interpolated", "q_sum"}:
                result[key] = item
            elif key == "smearing" and item in {"mv", "mp", "gaussian", "fd"}:
                result[key] = item
        elif _numbers(item):
            result[key] = item
    return result


def _numbers(value):
    if value is None or type(value) is bool:
        return True
    if type(value) in {int, float}:
        return math.isfinite(value)
    return isinstance(value, list) and all(_numbers(item) for item in value)


def _scf_comparison_signature(native_input, pseudo_sha256, result):
    """Match frozen SCF inputs, allowing only different initial spin seeds."""
    if (result.get("spin_mode") not in {"collinear", "noncollinear"}
            or type(result.get("number_of_atoms")) is not int or result["number_of_atoms"] <= 0
            or not _finite_number(result.get("final_energy_ev_per_atom"))):
        return None
    lines, seeds, section = [], [], ""
    seed = re.compile(r"\s*starting_magnetization\(\s*([1-9][0-9]*)\s*\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)\s*,?\s*", re.I)
    for line in native_input.decode("utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("&"):
            section = stripped[1:].lower()
        elif stripped == "/":
            section = ""
        match = seed.fullmatch(line) if section == "system" else None
        if match:
            seeds.append((int(match[1]), float(match[2].lower().replace("d", "e"))))
        else:
            lines.append(line)
    if len({index for index, _ in seeds}) != len(seeds) or any(not math.isfinite(value) for _, value in seeds):
        return None
    signature = hashlib.sha256(_canonical({
        "input": "".join(lines), "pseudopotentials": pseudo_sha256,
        "atoms": result["number_of_atoms"], "qe_version": result.get("qe_version"),
        "spin_mode": result["spin_mode"], "soc": result.get("spin_orbit_coupling"),
    }).encode()).hexdigest()
    return signature, tuple(sorted(seeds))


def _comparison_facts(candidates, fact):
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["signature"], []).append(candidate)
    comparable = [group for group in groups.values()
                  if len(group) > 1 and len({row["seed"] for row in group}) > 1]
    fact("comparison.status", "Magnetic comparison", "available" if comparable else "unavailable", "plan.json")
    if not comparable:
        fact("comparison.scope", "Comparison scope", "No accepted SCF pair differs only in initial magnetization. Energies are not ranked across different methods or settings.", "plan.json")
    for index, group in enumerate(comparable, 1):
        prefix = f"comparison.group_{index}"
        source = "; ".join(row["source"] for row in group)
        minimum = min(row["energy"] for row in group)
        lowest = [row["stage"] for row in group if row["energy"] == minimum]
        fact(prefix + ".stages", "Compared SCF stages", [row["stage"] for row in group], source)
        fact(prefix + ".scope", "Comparison scope", "Only initial magnetization differs in the verified inputs. Geometry, pseudopotentials and all other input settings match. This ranks tested seeds at these settings; it does not establish the magnetic ground state or converged phase ordering.", source)
        fact(prefix + ".lowest_tested_stages", "Lowest-energy tested SCF stages", lowest, source)
        if len(lowest) == 1:
            fact(prefix + ".lowest_tested_stage", "Lowest-energy tested SCF stage", lowest[0], source)
        for row in group:
            fact(prefix + "." + row["stage"] + ".relative_energy", "Energy above the lowest tested seed", 1000 * (row["energy"] - minimum), source, "meV/atom")


def load_run_context(run_dir):
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
    if state.get("engine") != "qe":
        raise ValueError("This is not a saved QE run.")
    stages = state.get("stages") or []
    specifications = plan.get("stages") or []
    valid_plan = bool(plan.get("engine") == "qe" and state.get("plan_sha256") == fingerprints.get("plan.json")
                      and len(stages) == len(specifications) and stages)
    valid_plan = valid_plan and all(all(stage.get(key) == value for key, value in spec.items())
                                    for stage, spec in zip(stages, specifications))
    for relative, expected in (plan.get("sources") or {}).items():
        valid_plan = bool(digest(relative) == expected and expected and valid_plan)
    limits = ["Solver convergence does not establish convergence with respect to cutoff, k-point mesh or other numerical settings.",
              "Only the calculated magnetic configurations were tested; a global magnetic ground state is not established."]
    if not valid_plan:
        limits.append("The frozen plan or structure has changed; accepted numerical results are unavailable.")
    linked = bool(valid_plan and (proposal.get("execution") or {}).get("plan_sha256") == fingerprints.get("plan.json"))
    goal = _safe_text(proposal.get("goal")) if linked else ""
    dialogue = [{"role": entry["role"], "content": _safe_text(entry.get("content"))}
                for entry in proposal.get("dialogue", []) if linked and isinstance(entry, dict)
                and entry.get("role") in {"user", "assistant"}]
    facts, artifacts, accepted_results, comparison_candidates = {}, [], [], []

    def fact(identifier, label, value, source, unit=""):
        facts[identifier] = {"label": label, "value": value, "source": source, "unit": unit}

    for index, stage in enumerate(stages):
        prefix = f"stage_{index + 1}"
        task, folder = stage.get("name"), stage.get("folder")
        accepted = bool(valid_plan and isinstance(task, str) and task in _science(task).TASKS and isinstance(folder, str)
                        and re.fullmatch(r"[0-9]+_[a-z][a-z0-9_-]*", folder))
        issue = "Accepted solver evidence is unavailable, incomplete or changed."
        failure_reason = None
        result, current, regenerated = {}, {}, {}
        if accepted:
            output, inputs = folder + "/outputs", folder + "/inputs"
            result = read(output + "/result.json")
            metadata = read(inputs + "/metadata.json")
            manifest = read(output + "/artifact_manifest.json")
            accepted = bool(stage.get("status") in {"succeeded", "needs_attention", "failed"} and result == stage.get("result")
                            and metadata == stage.get("metadata") and type(result.get("success")) is bool
                            and stage.get("scheduler_state") == result.get("scheduler_state") == "COMPLETED")
            expected_inputs = metadata.get("input_sha256") or {}
            accepted = accepted and bool(expected_inputs)
            for name, expected in expected_inputs.items():
                accepted = bool(Path(name).name == name and expected
                                and digest(inputs + "/" + name) == expected
                                and digest(output + "/" + name) == expected and accepted)
            # Every downloaded solver artifact must retain its transport hash.
            # Parsers only see this verified set, so newly inserted files cannot
            # fill gaps in an incomplete calculation.
            verified_names = []
            for name, receipt in manifest.items():
                if (not _artifact_name(name) or not isinstance(receipt, dict)
                        or digest(output + "/" + name) != receipt.get("sha256")):
                    accepted = False
                else:
                    verified_names.append(name)
            required = {*expected_inputs, "execution.json", "input_hashes.sha256", "pseudo_manifest.json",
                        "runtime.json", metadata.get("output_file", "")}
            accepted = accepted and required.issubset(verified_names)
            try:
                input_hashes = dict((fields[1], fields[0]) for fields in
                                    (line.split() for line in path(output + "/input_hashes.sha256").read_text().splitlines()) if len(fields) == 2)
            except (OSError, ValueError):
                input_hashes = {}
            accepted = accepted and input_hashes == expected_inputs
            execution = read(output + "/execution.json")
            accepted = bool(accepted and execution.get("exit_code") == 0 and execution.get("finished_at")
                            and execution.get("job_id") == stage.get("job_id")
                            and execution.get("runtime_sha256") == stage.get("runtime_sha256")
                            and digest(output + "/runtime.json") == stage.get("runtime_sha256")
                            and digest(output + "/pseudo_manifest.json") == stage.get("pseudo_manifest_sha256"))
            for dependency in (stage.get("structure_from"), stage.get("restart_from"), stage.get("evidence_from"), stage.get("auxiliary_from")):
                if dependency is not None and (type(dependency) is not int or not 0 <= dependency < index or accepted_results[dependency] is None):
                    accepted = False
            if accepted:
                try:
                    names = tuple(sorted(verified_names))
                    current, regenerated = _parse(str(path(output)), task, names,
                                                  tuple(fingerprints.get(output + "/" + name) for name in names))
                    if not current.get("success"):
                        failure_reason = _safe_text(current.get("reason"))
                        issue = failure_reason or issue
                    accepted = bool(current.get("success") and result.get("success")
                                    and stage.get("status") == "succeeded")
                    # UI code also reads saved results; flag numerical edits
                    # instead of showing one number there and another to a model.
                    keys = ("final_energy_ev", "final_energy_ev_per_atom", "final_max_force_ev_angstrom",
                            "fermi_energy_ev", "band_gap", "barrier_forward_ev", "barrier_reverse_ev",
                            "reaction_energy_ev", "min_frequency_cm1", "imaginary_mode_count", "acoustic_sum_rule_violation_cm1",
                            "electron_phonon", "interpolated_lambda", "mu_star", "tc_model", "integration_method",
                            "temperature_results", "tc_bracket_kelvin", "tc_bracketed", "pairing_model",
                            "lambda_integral", "omega_log_mev", "allen_dynes_estimate_kelvin",
                            "total_magnetization_mu_b", "total_magnetization_vector_mu_b", "absolute_magnetization_mu_b",
                            "stress_gpa", "pressure_gpa", "site_magnetization", "site_magnetization_scope",
                            "spin_mode", "spin_orbit_coupling", "converged_exx", "exx_exchange_error_ry")
                    # Results saved by an earlier version lack fields added since (phonon sum-rule
                    # checks in 0.4.5); every number they do hold must match.
                    if accepted and any(key in result and not saved_matches(result[key], current.get(key)) for key in keys):
                        accepted = False
                        issue = "Saved numerical results differ from the verified solver output."
                    neb_keys = ("path_converged", "endpoint_forces_ev_angstrom", "endpoint_relaxed",
                                "endpoint_force_threshold_ev_angstrom", "barrier_scope")
                    if any(key in result and current.get(key) != result[key] for key in neb_keys):
                        accepted = False
                        issue = "Saved NEB results differ from the verified solver output."
                except (ValueError, OSError, KeyError, TypeError):
                    accepted = False
            if accepted and current.get("final_structure"):
                final_name = current["final_structure"]
                accepted = bool(Path(final_name).name == final_name
                                and digest(output + "/" + final_name) == result.get("final_structure_sha256")
                                and regenerated.get(final_name) == result.get("final_structure_sha256"))
            if accepted and result.get("final_native_structure_sha256"):
                expected_native = result["final_native_structure_sha256"]
                accepted = bool(digest(output + "/CONTCAR") == expected_native
                                and regenerated.get("CONTCAR") == expected_native)
        accepted_results.append(current if accepted else None)
        fact(prefix + ".task", "Calculation", task, "plan.json")
        fact(prefix + ".accepted", f"{task}: accepted result", accepted, "run.json")
        status = stage.get("status", "unavailable")
        if status == "succeeded" and not accepted:
            status = "needs_attention"
        fact(prefix + ".status", f"{task}: current status", status, "run.json")
        if not accepted:
            fact(prefix + ".availability", f"{task}: availability", issue, "run.json")
            if failure_reason:
                fact(prefix + ".failure_reason", f"{task}: verified failure", failure_reason,
                     folder + "/outputs/" + metadata["output_file"])
            continue
        source = folder + "/outputs/" + stage["metadata"]["output_file"]
        if task == "scf":
            try:
                native_name = folder + "/outputs/pw.in"
                native_input = path(native_name).read_bytes()
                pseudo_hash = digest(folder + "/outputs/pseudo_manifest.json")
                if (hashlib.sha256(native_input).hexdigest() == fingerprints.get(native_name)
                        and pseudo_hash == stage.get("pseudo_manifest_sha256")):
                    signature = _scf_comparison_signature(native_input, pseudo_hash, current)
                    if signature:
                        comparison_candidates.append({"signature": signature[0], "seed": signature[1],
                            "stage": prefix, "energy": current["final_energy_ev_per_atom"],
                            "source": folder + "/outputs/data-file-schema.xml"})
            except (OSError, ValueError, UnicodeError):
                pass
        prepared_input = read(folder + "/outputs/qe-input.json")
        effective_parameters = {**stage.get("parameters", {}), **prepared_input.get("parameters", {})}
        for key in ("namelists", "site_labels", "mesh"):
            if key in prepared_input:
                effective_parameters[key] = prepared_input[key]
        for key, value in _parameters(effective_parameters).items():
            fact(prefix + ".parameter." + key, f"{task}: {key}", value, folder + "/outputs/qe-input.json")
        for suffix, key, label, unit in (
            ("electronic_converged", "converged_electronic", "Electronic convergence", ""),
            ("exx_converged", "converged_exx", "EXX outer-loop convergence", ""),
            ("exx_exchange_error", "exx_exchange_error_ry", "EXX exchange error", "Ry"),
            ("ionic_converged", "converged_ionic", "Ionic convergence", ""),
            ("path_converged", "path_converged", "NEB path convergence", ""),
            ("endpoint_relaxed", "endpoint_relaxed", "NEB endpoints meet the force threshold", ""),
            ("endpoint_forces", "endpoint_forces_ev_angstrom", "NEB endpoint force components", "eV/Å"),
            ("endpoint_force_threshold", "endpoint_force_threshold_ev_angstrom", "NEB endpoint force threshold", "eV/Å"),
            ("phonon_converged", "converged_phonon", "Phonon convergence", ""),
            ("fermi_energy", "fermi_energy_ev", "Fermi energy", "eV"),
            ("force", "final_max_force_ev_angstrom", "Maximum force", "eV/Å"),
            ("barrier_forward", "barrier_forward_ev", "Forward NEB barrier", "eV"),
            ("barrier_reverse", "barrier_reverse_ev", "Reverse NEB barrier", "eV"),
            ("reaction_energy", "reaction_energy_ev", "Reaction energy", "eV"),
            ("minimum_phonon_frequency", "min_frequency_cm1", "Minimum phonon frequency", "cm⁻¹"),
            ("imaginary_mode_count", "imaginary_mode_count", "Sampled imaginary modes", "modes"),
            ("acoustic_sum_rule_violation", "acoustic_sum_rule_violation_cm1", "Acoustic frequency at Γ before the sum rule", "cm⁻¹"),
            ("qpoint_count", "qpoint_count", "Phonon path points", "points"),
            ("total_magnetization", "total_magnetization_mu_b", "Total magnetization", "μB"),
            ("magnetization_vector", "total_magnetization_vector_mu_b", "Total magnetization vector", "μB"),
            ("absolute_magnetization", "absolute_magnetization_mu_b", "Absolute magnetization", "μB"),
            ("stress", "stress_gpa", "Stress tensor", "GPa"),
            ("pressure", "pressure_gpa", "Pressure", "GPa"),
        ):
            value = current.get(key)
            if value is not None and _numbers(value):
                fact(prefix + "." + suffix, f"{task}: {label}", value, source, unit)
        if task in {"scf", "relax", "vc-relax", "eph_dense_scf", "eph_scf"}:
            for suffix, key, label, unit in (("energy", "final_energy_ev", "Total energy", "eV"),
                                             ("energy_per_atom", "final_energy_ev_per_atom", "Energy per atom", "eV/atom")):
                if current.get(key) is not None:
                    fact(prefix + "." + suffix, f"{task}: {label}", current[key], source, unit)
        if task == "neb":
            fact(prefix + ".barrier_scope", "NEB barrier scope", current.get("barrier_scope"), source)
        # Every stage's own warnings, such as imaginary phonon branches, travel with its facts.
        limits.extend(current.get("warnings", []))
        gap = current.get("band_gap") or {}
        if "gap_ev" not in gap and "energy_ev" in gap:
            gap = {**gap, "gap_ev": gap["energy_ev"]}
        for suffix, key, label, unit in (("band_gap", "gap_ev", "Sampled gap", "eV"),
                                        ("direct_gap", "direct_gap_ev", "Sampled direct gap", "eV"),
                                        ("band_gap_scope", "scope", "Gap sampling scope", ""),
                                        ("band_gap_status", "status", "Gap status", ""),
                                        ("vbm", "vbm_ev", "Sampled valence edge", "eV"),
                                        ("cbm", "cbm_ev", "Sampled conduction edge", "eV")):
            if gap.get(key) is not None:
                fact(prefix + "." + suffix, f"{task}: {label}", gap[key], source, unit)
        magnetic_source = folder + "/outputs/data-file-schema.xml"
        for key, label in (("spin_mode", "Calculated spin mode"), ("spin_orbit_coupling", "Calculated spin-orbit coupling")):
            if current.get(key) is not None:
                fact(prefix + "." + key, label, current[key], magnetic_source)
        for moment in current.get("site_magnetization", []):
            if type(moment.get("atom")) is int and _numbers(moment.get("moment_mu_b")):
                fact(f"{prefix}.site_moment.{moment['atom']}",
                     f"Atom {moment['atom']} ({moment.get('element', '')}): projected spin moment",
                     moment["moment_mu_b"], magnetic_source, "μB")
        if current.get("site_magnetization_scope"):
            fact(prefix + ".site_magnetization_scope", "Site-moment projection scope",
                 current["site_magnetization_scope"], magnetic_source)
        for series_key in ("interpolated_lambda", "electron_phonon"):
            for row_index, row in enumerate(current.get(series_key, []), 1):
                for key, label, unit in (("degauss_ry", "Electronic broadening", "Ry"),
                                         ("lambda", "Electron-phonon coupling", ""),
                                         ("lambda_spectral_integral", "Coupling from spectral integral", ""),
                                         ("omega_log_kelvin", "Logarithmic phonon frequency", "K"),
                                         ("mu_star", "Assumed Coulomb pseudopotential", ""),
                                         ("tc_estimate_kelvin", "Estimated transition temperature", "K"),
                                         ("spectral_lambda_relative_difference", "Relative coupling-integral difference", "")):
                    value = row.get(key)
                    if value is not None and _numbers(value):
                        fact(f"{prefix}.{series_key}.{row_index}.{key}",
                             f"{task}: broadening {row_index}: {label}", value, source, unit)
        if current.get("tc_model"):
            fact(prefix + ".tc_model", "Transition-temperature model", current["tc_model"], source)
            if current.get("integration_method"):
                fact(prefix + ".integration_method", "Spectral integration", current["integration_method"], source)
            limits.append("The reported Tc values estimate conventional phonon-mediated pairing under the stated mu_star; they do not establish superconductivity or describe unconventional pairing.")
        if task.startswith("epw_"):
            for key, label, unit in (("pairing_model", "Pairing model", ""),
                                     ("mu_star", "Coulomb pseudopotential", ""),
                                     ("lambda_integral", "Integrated coupling", ""),
                                     ("omega_log_mev", "Logarithmic phonon energy", "meV"),
                                     ("allen_dynes_estimate_kelvin", "Allen-Dynes estimate", "K"),
                                     ("tc_bracket_kelvin", "Linearized Tc bracket", "K"),
                                     ("tc_bracketed", "Tc crossing bracketed", "")):
                if current.get(key) is not None:
                    fact(prefix + "." + key, label, current[key], source, unit)
            for row_index, row in enumerate(current.get("temperature_results", []), 1):
                for key, unit in (("temperature_kelvin", "K"), ("eigenvalue", ""),
                                  ("gap_iw0_mev", "meV"), ("gap_min_iw0_mev", "meV"),
                                  ("gap_max_iw0_mev", "meV"), ("renormalization_iw0", ""),
                                  ("residual", ""), ("matsubara_points", ""),
                                  ("cutoff_ev", "eV"), ("iterations", "")):
                    if key in row and _numbers(row[key]):
                        fact(f"{prefix}.temperature_{row_index}.{key}", key, row[key], source, unit)
            limits.append("EPW solves the stated phonon pairing model. Its gaps or temperature bracket require interpolation, grid, temperature and frequency-cutoff convergence; they do not establish unconventional pairing or measured superconductivity.")
            if current.get("warning"):
                limits.append(current["warning"])
        for name in current.get("artifacts", []):
            if not _artifact_name(name):
                continue
            relative = folder + "/outputs/" + name
            current_hash = digest(relative)
            verified = bool(current_hash and (current_hash == regenerated.get(name)
                            or current_hash == (manifest.get(name) or {}).get("sha256")))
            identifier = prefix + "." + name
            availability = "verified" if verified else "present_unverified" if current_hash else "unavailable"
            artifacts.append({"id": identifier, "label": f"{task}: {name}", "path": relative,
                              "sha256": current_hash, "status": availability})
            fact(identifier, f"{task}: {name} availability", availability, relative)
    if sum(stage.get("name") == "scf" for stage in stages) > 1:
        _comparison_facts(comparison_candidates, fact)
    successful = sum(result is not None for result in accepted_results)
    status = state.get("status", "unavailable")
    if status == "succeeded" and successful != len(stages):
        status = "needs_attention"
    fact("run.status", "Current result status", status, "run.json")
    fact("run.accepted_stages", "Stages with accepted results", successful, "run.json", "stages")
    if not any(key.endswith(".band_gap") for key in facts):
        limits.append("A band gap has not been quantified from these results.")
    context = {"engine": "qe", "goal": goal, "dialogue": dialogue, "status": status,
               "outcome": f"{successful} of {len(stages)} stages have accepted results.",
               "final_plan": {"tasks": plan.get("tasks", []), "parameters": _parameters(plan.get("parameters"))} if valid_plan else {},
               "facts": facts, "artifacts": artifacts, "limits": limits}
    context["context_sha256"] = hashlib.sha256(_canonical({"context": context, "files": fingerprints}).encode()).hexdigest()
    return context
