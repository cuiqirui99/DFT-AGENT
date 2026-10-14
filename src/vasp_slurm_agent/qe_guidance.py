"""Source-linked QE checks and bounded, verified local calculation examples."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import islice
import json
import os
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from . import qe_context


MAX_SCAN_ENTRIES = 128
MAX_RUNS = 64
MAX_CASES = 3
MAX_RECORD_BYTES = 1_000_000
MAX_CASE_BYTES = 7_000
_DOC = "https://www.quantum-espresso.org/Doc/"
_GUIDANCE = [
    {"id": "qe_inputs", "title": "Method and numerical checks", "applies_to": ["scf", "relax", "vc-relax", "bands", "dos", "pdos"],
     "checks": ["Use UPFs compatible with the functional; converge ecutwfc, ecutrho and the k mesh for the requested observable. Cutoffs are in Ry.",
                "Keep structure, UPFs, functional, U and occupations consistent in energy comparisons. SCF convergence alone does not establish numerical accuracy.",
                "For hybrid functionals, verify final EXX convergence; the inner SCF flag can be true after the outer loop exhausts its iterations.",
                "SOC requires noncolin and lspinorb with fully relativistic UPFs. Fully relativistic ultrasoft/PAW UPFs cannot be used for a no-SOC reference in QE 7.4.1. Specify magnetic seeds and Hubbard manifolds/U; do not infer them from composition."],
     "sources": [{"url": _DOC + "INPUT_PW.html"}]},
    {"id": "qe_spectra", "title": "Bands and DOS", "applies_to": ["bands", "nscf", "dos", "pdos"],
     "checks": ["Use a matching accepted SCF state. A band path and a DOS integration mesh serve different purposes.",
                "Unconverged eigenvalues invalidate the affected band calculation even when QE prints JOB DONE. Repair the diagonalization without loosening the requested accuracy.",
                "Check empty bands, occupations and sampling. The current runner does not implement hybrid NSCF/band interpolation; do not substitute PBE."],
     "sources": [{"url": _DOC + "INPUT_PW.html"}, {"url": _DOC + "INPUT_DOS.html"}, {"url": _DOC + "INPUT_PROJWFC.html"}]},
    {"id": "qe_phonons", "title": "Phonon evidence", "applies_to": ["phonon", "phonon_bands"],
     "checks": ["Check the ground-state geometry and electronic convergence before DFPT. A single q point is not a dispersion.",
                "For a dispersion, verify the q grid, q2r/matdyn chain and acoustic-sum-rule choice. Resolve imaginary modes with numerical and structural checks."],
     "sources": [{"url": _DOC + "INPUT_PH.html"}, {"url": _DOC + "INPUT_Q2R.html"}, {"url": _DOC + "INPUT_MATDYN.html"}]},
    {"id": "qe_neb", "title": "Barrier calculation", "applies_to": ["neb"],
     "checks": ["Supply endpoint/image structures with consistent atom order and cell; inspect the interpolated path for collisions.",
                "Verify endpoint forces separately. A converged path with fixed, unrelaxed endpoints does not establish a barrier between relaxed states.",
                "Check image forces, path convergence and image count before interpreting a barrier. Preserve the electronic method along the path."],
     "sources": [{"url": _DOC + "INPUT_NEB.html"}]},
    {"id": "qe_pairing", "title": "Electron-phonon checks", "applies_to": ["electron_phonon", "epw_iso", "epw_tc"],
     "checks": ["Specify and converge electronic meshes, q mesh and broadening; state the assumed mu_star for a conventional Tc estimate.",
                "Negative interpolated spectral weights fail acceptance. Explicit q_sum integrates the original DFPT modes by a separate method; preserve both results and check q-grid and spectral-width convergence.",
                "Isotropic EPW requires an accepted spectrum and explicit temperatures, Matsubara cutoff and broadening column. Check nonlinear residuals or a bracketed linearized eigenvalue crossing; never choose a spectrum to maximize Tc.",
                "If Tc is unknown, use epw_tc before selecting nonlinear gap temperatures. EPW 5.9 can fail as the gap vanishes above Tc. Preserve that failure and request a separate temperature range; do not silently change it or use Allen-Dynes alone to set the range.",
                "DFPT/lambda and isotropic Eliashberg results are model-dependent. Anisotropic interpolation and unconventional pairing require separate workflows and validation."],
     "sources": [{"url": _DOC + "INPUT_PH.html"}, {"url": "https://docs.epw-code.org/Inputs/Inputs.html"},
                 {"url": "https://github.com/QEF/q-e/blob/qe-7.4.1/EPW/src/supercond_iso.f90"}]},
    {"id": "qe_wannier_pairing", "title": "Wannier and anisotropic pairing", "applies_to": ["wannier_electron_phonon", "epw_aniso"],
     "checks": ["Use compatible scalar, nonmagnetic NC UPFs. Specify projections, outer/frozen energy windows, band counts and coarse/fine grids; do not infer them from composition.",
                "Preserve the full NSCF grid, DFPT derivative ordering and native coupling checkpoint. Check Wannier convergence and compare interpolated bands in the Fermi window before interpreting a material result.",
                "Specify temperatures, mu_star and frequency cutoff. Check gap residuals, momentum dependence and grid/cutoff convergence. A nonzero gap at one temperature is not a Tc determination.",
                "Anisotropic electron-phonon pairing is a conventional Migdal-Eliashberg model. It cannot substitute for an explicit unconventional interaction model."],
     "sources": [{"url": "https://github.com/QEF/q-e/tree/qe-7.4.1/test-suite/epw_super"},
                 {"url": "https://docs.epw-code.org/Theory.html#superconductivity"}]},
]
_LIMITS = [
    "Local cases are advisory examples, not settings to copy or proof of convergence, material identity, a ground state or superconductivity.",
    "Only succeeded QE runs with all stages reverified are included. Failed, planned, changed or incomplete records are excluded.",
    "Retrieval scans at most 128 directory entries and verifies at most 64 runs; at most three cases are returned. This is not an exhaustive history.",
    "Case facts are a bounded subset. Gaps describe sampled states; consult the retained result for its full scope.",
]
_FACT_UNITS = {
    "electronic_converged": "", "ionic_converged": "", "phonon_converged": "",
    "energy": "eV", "energy_per_atom": "eV/atom", "force": "eV/Å", "fermi_energy": "eV",
    "band_gap": "eV", "total_magnetization": "μB", "absolute_magnetization": "μB",
    "pressure": "GPa", "barrier_forward": "eV", "barrier_reverse": "eV",
    "minimum_phonon_frequency": "cm⁻¹", "imaginary_mode_count": "modes",
}


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Invalid local record.")
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Invalid local record.")
    return value, hashlib.sha256(data).hexdigest()


def _composition(value):
    composition = value.element_composition if isinstance(value, Composition) else Composition(value).element_composition
    if not composition or not composition.valid:
        raise ValueError("Choose a chemical composition.")
    fractions = composition.fractional_composition.get_el_amt_dict()
    return tuple(sorted((key, round(amount, 10)) for key, amount in fractions.items())), composition.reduced_formula


def _structure(path):
    if not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Invalid local structure.")
    structure = Structure.from_file(path)
    if not structure.is_ordered or not 0 < len(structure) <= 512:
        raise ValueError("Unsupported local structure.")
    return structure


def _candidates(root):
    if (root / "run.json").is_file():
        return [root]
    with os.scandir(root) as entries:
        directories = [Path(entry.path) for entry in islice(entries, MAX_SCAN_ENTRIES)
                       if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(".")]
    return sorted(directories, key=lambda path: path.stat().st_mtime_ns, reverse=True)[:MAX_RUNS]


def _case(root, signature, query_structure):
    state, _ = _read(root / "run.json")
    if state.get("engine") != "qe" or state.get("status") != "succeeded":
        return None
    plan, plan_hash = _read(root / "plan.json")
    stages = plan.get("stages", [])
    if state.get("plan_sha256") != plan_hash or not 0 < len(stages) <= 20:
        return None
    relative = stages[0]["source_path"]
    source = root / relative
    if (Path(relative).is_absolute() or source.is_symlink() or not source.resolve().is_relative_to(root)
            or not source.is_file() or source.stat().st_size > MAX_RECORD_BYTES):
        return None
    source_hash = (plan.get("sources") or {}).get(relative)
    if not source_hash or qe_context._sha(source) != source_hash:
        return None
    saved = _structure(source)
    saved_signature, formula = _composition(saved.composition)
    if signature != saved_signature:
        return None
    criterion = "same_composition_only; phase and geometry may differ"
    if query_structure is not None:
        matcher = StructureMatcher(ltol=0.1, stol=0.15, angle_tol=3, primitive_cell=True, scale=False)
        if not matcher.fit(query_structure, saved):
            return None
        criterion = "same_composition_and_source_structure; ltol=0.1, stol=0.15, angle_tol=3, scale=False, primitive_cell=True"
    context = qe_context.load_run_context(root)
    facts = context.get("facts", {})
    if context.get("status") != "succeeded" or any(
            facts.get(f"stage_{index + 1}.accepted", {}).get("value") is not True for index in range(len(stages))):
        return None
    if _read(root / "plan.json")[1] != plan_hash or qe_context._sha(source) != source_hash:
        return None
    rows = []
    for index, stage in enumerate(stages):
        prefix = f"stage_{index + 1}."
        parameters = {identifier[len(prefix + 'parameter.'):]: item["value"]
                      for identifier, item in facts.items() if identifier.startswith(prefix + "parameter.")}
        measured = [{"id": prefix + key, "value": facts[prefix + key]["value"], "unit": unit}
                    for key, unit in _FACT_UNITS.items() if prefix + key in facts
                    and qe_context._numbers(facts[prefix + key]["value"])]
        rows.append({"task": stage["name"], "method": qe_context._parameters(parameters), "facts": measured})
    result = {"id": "qe_case_" + context["context_sha256"][:16], "engine": "qe", "formula": formula,
              "match_criterion": criterion, "context_sha256": context["context_sha256"], "stages": rows}
    return result if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) <= MAX_CASE_BYTES else None


def build_context(structure_path=None, *, formula=None, runs_root=None, exclude_run=None, limit=3):
    """Return QE guidance and local evidence; never contact a model or cluster."""
    result = {"guidance": deepcopy(_GUIDANCE), "experience": [], "limits": list(_LIMITS)}
    if type(limit) is not int or not 0 <= limit <= MAX_CASES:
        raise ValueError("Choose between zero and three local examples.")
    if not runs_root or limit == 0:
        return result
    try:
        structure = _structure(Path(structure_path).expanduser()) if structure_path is not None else None
        signature, _ = _composition(structure.composition if structure is not None else formula)
        if structure is not None and formula is not None and _composition(formula)[0] != signature:
            raise ValueError("The formula differs from the structure.")
    except (OSError, ValueError, TypeError, KeyError):
        result["limits"].append("No usable target composition/structure was supplied; local cases were not retrieved.")
        return result
    result["limits"].append("Structure matching is approximate and does not establish the same phase or magnetic state."
                            if structure is not None else "Formula-only matches can have different phases or geometries.")
    try:
        root = Path(runs_root).expanduser().resolve()
        excluded = Path(exclude_run).expanduser().resolve() if exclude_run is not None else None
        candidates = _candidates(root)
    except (OSError, ValueError):
        return result
    seen = set()
    for candidate in candidates:
        if candidate.resolve() == excluded:
            continue
        try:
            case = _case(candidate.resolve(), signature, structure)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            continue
        if case and case["context_sha256"] not in seen:
            result["experience"].append(case)
            seen.add(case["context_sha256"])
            if len(result["experience"]) >= limit:
                break
    return result
