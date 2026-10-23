"""Shared structure, unit and evidence helpers for the ABINIT, CP2K and Elk adapters.

Every adapter writes one native stage and checks its own outputs. This module
keeps the conventions identical across codes: input structures are frozen with
exact coordinates, band paths come from seekpath, energies are reported in eV,
and a band gap is only a statement about the sampled k points.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Element, Structure

from .numerical_defaults import path_time_reversal, vacuum_axes
from .qe import (BOHR_ANGSTROM, HA_EV, RY_EV, _csv, _hash, _integer, _json, _number, _path,
                 _validate_structure, _vector)
from .spectra import band_reference, plot_bands, plot_dos

__all__ = ["BOHR_ANGSTROM", "HA_EV", "RY_EV", "KB_EV", "HA_PER_BOHR_EV_PER_ANGSTROM", "HA_BOHR3_GPA",
           "TASKS", "COMMON_KEYS", "load_structure", "native_structure", "same_structure",
           "check_parameters", "kpoint_mesh", "band_path", "spin_settings", "site_moments",
           "smearing", "hubbard_terms", "band_gap", "write_bands", "write_dos", "write_final_structure",
           "new_result", "fail", "species_order", "metadata_base", "band_reference", "plot_bands", "_csv", "_hash", "_integer", "_json",
           "_number", "_vector"]

KB_EV = 8.617333262e-5
HA_PER_BOHR_EV_PER_ANGSTROM = HA_EV / BOHR_ANGSTROM
HA_BOHR3_GPA = 29421.02648438959
TASKS = ("relax", "scf", "bands", "dos")
COMMON_KEYS = frozenset({
    "functional", "mesh", "kspacing", "spin", "soc", "initial_moments", "site_moments",
    "electronic_type", "smearing_ev", "cell_relax", "force_tolerance_ev_angstrom",
    "max_ionic_steps", "max_scf_steps", "energy_tolerance_ev", "line_density", "kpoints",
    "extra_bands", "dos_mesh", "hubbard",
})
# Elements that usually carry local moments. Only used for the default
# starting guess when spin polarization is requested without moments.
_MAGNETIC = frozenset({"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Pd", "Os", "Ir", "Pt",
                       "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                       "U", "Np", "Pu", "Mo", "W", "Re", "Tc"})


def load_structure(path):
    structure = Structure.from_file(path)
    _validate_structure(structure)
    return structure


def native_structure(structure):
    return {"lattice_angstrom": structure.lattice.matrix.tolist(),
            "elements": [site.specie.symbol for site in structure],
            "fractional_positions": structure.frac_coords.tolist()}


def structure_from_native(native):
    if not isinstance(native, dict) or set(native) != {"lattice_angstrom", "elements", "fractional_positions"}:
        raise ValueError("Invalid native structure metadata.")
    structure = Structure(native["lattice_angstrom"], native["elements"], native["fractional_positions"])
    _validate_structure(structure)
    return structure


def same_structure(actual, expected, positions=True, *, atol=1e-5):
    if [site.specie.symbol for site in actual] != [site.specie.symbol for site in expected]:
        return False
    if not positions:
        return True
    delta = actual.frac_coords - expected.frac_coords
    return bool(np.allclose(actual.lattice.matrix, expected.lattice.matrix, atol=atol, rtol=1e-6)
                and np.allclose(delta - np.rint(delta), 0, atol=atol))


def species_order(structure):
    return list(dict.fromkeys(site.specie.symbol for site in structure))


def check_parameters(parameters, allowed, engine):
    p = deepcopy(dict(parameters or {}))
    unknown = sorted(set(p) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {engine} parameters: {', '.join(unknown)}")
    if not isinstance(p.get("soc", False), bool) or not isinstance(p.get("cell_relax", False), bool):
        raise ValueError("soc and cell_relax must be true or false.")
    return p


def kpoint_mesh(structure, parameters, key="mesh"):
    mesh = parameters.get(key)
    if mesh is None:
        spacing = _number(parameters.get("kspacing", 0.25), "kspacing", True)
        mesh = [max(1, math.ceil(length / spacing)) for length in structure.lattice.reciprocal_lattice.abc]
        for axis in vacuum_axes(structure):
            mesh[axis["axis"]] = 1
    mesh = _vector(mesh, key, integers=True)
    if any(n > 64 for n in mesh):
        raise ValueError(f"{key} is unexpectedly dense; use at most 64 points per direction.")
    return mesh


def band_path(structure, parameters):
    values = {"line_density": parameters.get("line_density", 40), **{k: parameters[k] for k in ("kpoints", "soc") if k in parameters}}
    spin = parameters.get("spin", "none")
    moments = None
    if spin != "none":
        # Atoms with different starting moments are different for the path (an antiferromagnet's magnetic cell).
        moments = site_moments(structure, parameters, spin)
        values["site_labels"] = [repr(np.round(np.asarray(m, dtype=float), 2).tolist()) for m in moments]
    path = _path(structure, values, time_reversal=path_time_reversal(moments, bool(parameters.get("soc")) or spin == "noncollinear"))
    if len(path["reciprocal_fractional"]) > 2000:
        raise ValueError("The band path has too many points. Reduce line_density.")
    return path


def spin_settings(parameters, *, noncollinear=True, soc=True):
    spin = parameters.get("spin", "none")
    if spin not in {"none", "collinear", "noncollinear"}:
        raise ValueError("spin must be none, collinear or noncollinear.")
    if spin == "noncollinear" and not noncollinear:
        raise ValueError("This program does not run noncollinear magnetism here. Use collinear spin.")
    if parameters.get("soc") and not soc:
        raise ValueError("Spin-orbit coupling is not available for this program in DFT Agent.")
    if parameters.get("soc") and spin == "collinear":
        raise ValueError("Magnetic spin-orbit calculations need noncollinear spin.")
    return spin


def site_moments(structure, parameters, spin):
    """Starting local moments in μB, one per site (vectors for noncollinear spin)."""
    if spin == "none":
        if parameters.get("initial_moments") or parameters.get("site_moments"):
            raise ValueError("Starting moments need spin collinear or noncollinear.")
        return None
    sites = parameters.get("site_moments")
    if sites is not None:
        if not isinstance(sites, list) or len(sites) != len(structure):
            raise ValueError("site_moments needs one starting moment per atom.")
        moments = sites
    else:
        by_element = parameters.get("initial_moments")
        if by_element is not None:
            if not isinstance(by_element, dict) or any(not re.fullmatch(r"[A-Z][a-z]?", key) for key in by_element):
                raise ValueError("initial_moments maps element symbols to starting moments in μB.")
            moments = [by_element.get(site.specie.symbol, 0.0) for site in structure]
        else:
            moments = [3.0 if site.specie.symbol in _MAGNETIC else 0.5 for site in structure]
    checked = []
    for value in moments:
        if spin == "noncollinear":
            vector = [0.0, 0.0, _number(value, "moment")] if not isinstance(value, (list, tuple)) else _vector(value, "moment")
            if math.sqrt(sum(x * x for x in vector)) > 10:
                raise ValueError("Starting moments must be at most 10 μB.")
            checked.append(vector)
        else:
            if isinstance(value, (list, tuple)):
                raise ValueError("Collinear starting moments are single numbers.")
            moment = _number(value, "moment")
            if abs(moment) > 10:
                raise ValueError("Starting moments must be at most 10 μB.")
            checked.append(moment)
    return checked


def smearing(parameters, default_ev):
    kind = parameters.get("electronic_type", "auto")
    if kind not in {"auto", "metal", "insulator"}:
        raise ValueError("electronic_type must be auto, metal or insulator.")
    width = _number(parameters.get("smearing_ev", default_ev), "smearing_ev", True)
    if width > 1.0:
        raise ValueError("smearing_ev must be at most 1 eV.")
    return kind, width


def hubbard_terms(structure, parameters):
    """[{element, l, u_ev, j_ev}] for programs that support DFT+U on a local projector."""
    values = parameters.get("hubbard")
    if not values:
        return []
    if not isinstance(values, list):
        raise ValueError("hubbard is a list of {element, l, u_ev, j_ev} entries.")
    present = {site.specie.symbol for site in structure}
    terms, seen = [], set()
    for item in values:
        if not isinstance(item, dict) or set(item) - {"element", "l", "u_ev", "j_ev"} or not {"element", "l", "u_ev"} <= set(item):
            raise ValueError("Each Hubbard term needs element, l and u_ev (j_ev is optional).")
        element = item["element"]
        if element not in present or element in seen:
            raise ValueError("Give each Hubbard element once, using elements in the structure.")
        if type(item["l"]) is not int or item["l"] not in {1, 2, 3}:
            raise ValueError("Hubbard l must be 1 (p), 2 (d) or 3 (f).")
        u = _number(item["u_ev"], "u_ev")
        j = _number(item.get("j_ev", 0.0), "j_ev")
        if not 0 <= j <= u <= 15:
            raise ValueError("Use 0 ≤ J ≤ U ≤ 15 eV for Hubbard terms.")
        seen.add(element)
        terms.append({"element": element, "l": item["l"], "u_ev": u, "j_ev": j})
    return terms


def metadata_base(engine, task, structure, structure_path, parameters):
    return {"schema_version": 1, "engine": engine, "task": task, "canonical_task": task,
            "formula": structure.composition.reduced_formula, "number_of_atoms": len(structure),
            "parameters": parameters, "source_sha256": _hash(structure_path),
            "requires_restart": task in {"bands", "dos"}, "native_structure": native_structure(structure),
            "elements": species_order(structure), "scientific_accuracy_validated": False}


def new_result(engine, task):
    return {"schema_version": 1, "engine": engine, "task": task, "success": False,
            "converged_electronic": None, "converged_ionic": None, "energy_ev": None,
            "final_energy_ev": None, "fermi_energy_ev": None, "artifacts": [], "warnings": [],
            "scientific_accuracy_validated": False}


def fail(message):
    raise ValueError(message)


def band_gap(energies, *, electrons=None, fermi=None, spinor=False, occupations=None,
             full_occupation=None, occupations_are_weights=False, scope="sampled_k_points", tolerance=1e-4):
    """Gap from sampled eigenvalues in eV, shaped (spin, k, band).

    Counting electrons is exact for a fixed-occupation nonmagnetic insulator.
    A band crossing the Fermi level is metallic even when no sampled point is
    close to it. Thermal occupation tails alone do not close a spectral gap.
    """
    values = np.asarray(energies, dtype=float)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        return {"available": False, "status": "unavailable", "note": "No complete eigenvalue table.", "scope": scope}
    occupied = empty = None
    partial = False
    metallic = False
    count = None
    if fermi is not None:
        fermi = _number(fermi, "Fermi energy")
        lowest, highest = values.min(axis=1), values.max(axis=1)
        # A band rising from the Fermi level touches it, as at the Dirac point of graphene.
        metallic = bool(np.any((lowest < fermi - tolerance) & (highest > fermi + tolerance))
                        or np.any((np.abs(lowest - fermi) <= tolerance) & (highest > fermi + tolerance)))
    occupation_values = None
    if occupations is not None:
        occupation_values = np.asarray(occupations, dtype=float)
        full = full_occupation if full_occupation is not None else (1 if spinor or values.shape[0] > 1 else 2)
        full = _number(full, "full occupation", True)
        if (occupation_values.shape != values.shape or not np.isfinite(occupation_values).all()
                or (not occupations_are_weights and
                    (np.any(occupation_values < -1e-6) or np.any(occupation_values > full + 1e-6)))):
            return {"available": False, "status": "unavailable", "note": "No valid occupation table.", "scope": scope}
        fraction = occupation_values / full
        partial = bool(np.any((fraction > 1e-3) & (fraction < 1 - 1e-3)))
        metallic = metallic or bool(np.any((fraction.min(axis=1) < 0.5 - 1e-6)
                                           & (fraction.max(axis=1) > 0.5 + 1e-6)))
        # A half-filled state belongs to both edges, unlike small thermal tails.
        occupied = values[fraction >= 0.5 - 1e-6]
        empty = values[fraction <= 0.5 + 1e-6]
    if electrons is not None and values.shape[0] == 1:
        per_band = 1 if spinor else 2
        filling = _number(electrons, "electrons", True) / per_band
        metallic = metallic or abs(filling - round(filling)) >= 1e-6
        if abs(filling - round(filling)) < 1e-6 and 0 < round(filling) < values.shape[2]:
            count = int(round(filling))
            occupied, empty = values[0, :, :count], values[0, :, count:]
    if occupied is None and fermi is not None:
        below, above = values[values <= fermi + tolerance], values[values > fermi + tolerance]
        if below.size and above.size:
            occupied, empty = below, above
    if occupied is None or not occupied.size or not empty.size:
        return {"available": False, "status": "unavailable", "note": "Both occupied and empty states are needed.", "scope": scope}
    vbm, cbm = float(np.max(occupied)), float(np.min(empty))
    direct = None
    if count is not None:
        direct = float(np.min(empty[:, 0] - occupied[:, -1]))
    metallic = metallic or cbm <= vbm + tolerance
    gap = {"available": True, "energy_ev": 0.0 if metallic else cbm - vbm, "gap_ev": 0.0 if metallic else cbm - vbm,
           "vbm_ev": vbm, "cbm_ev": cbm, "metallic_or_smeared": metallic,
           "status": "metallic_or_partially_occupied" if metallic else "gap", "scope": scope,
           "note": "Sampled k points only; not a converged full-zone gap."}
    if occupation_values is not None:
        gap["partial_occupations"] = partial
        if partial and not metallic:
            gap["note"] += " Fractional occupations are present; the reported gap is the eigenvalue gap."
    if direct is not None and not metallic:
        gap["direct_gap_ev"] = max(0.0, direct)
    return gap


def write_bands(folder, result, energies, path, reference, *, labels=("up", "down")):
    """bands.csv and a band plot along the prepared path; energies (spin, k, band) in eV."""
    values = np.asarray(energies, dtype=float)
    x = path.get("distance_inv_angstrom", list(range(values.shape[1])))
    if len(x) != values.shape[1]:
        raise ValueError("Band output k-point count differs from the prepared path.")
    rows = []
    for spin in range(values.shape[0]):
        for k in range(values.shape[1]):
            for band in range(values.shape[2]):
                rows.append([labels[spin] if values.shape[0] > 1 else "", k + 1, *path["reciprocal_fractional"][k],
                             band + 1, float(values[spin, k, band])])
    _csv(Path(folder) / "bands.csv", ["spin", "k_index", "k1", "k2", "k3", "band", "energy_ev"], rows)
    result["artifacts"].append("bands.csv")
    result["plot_reference_ev"] = reference
    result["artifacts"] += plot_bands(folder, "bands", x, values - reference, path.get("labels"), path.get("segments"),
                                      result.get("energy_reference", "fermi_level"))


def write_dos(folder, result, energy_ev, dos, reference, *, columns=("dos",)):
    data = np.asarray(dos, dtype=float)
    data = data[:, None] if data.ndim == 1 else data
    _csv(Path(folder) / "dos.csv", ["energy_ev", *columns], ([float(e), *map(float, row)] for e, row in zip(energy_ev, data)))
    result["artifacts"].append("dos.csv")
    result["dos_reference_ev"] = reference
    result["artifacts"] += plot_dos(folder, "dos", np.asarray(energy_ev) - reference, data)


def write_final_structure(folder, result, structure):
    from pymatgen.io.vasp import Poscar
    folder = Path(folder)
    structure.to(filename=str(folder / "final.cif"))
    Poscar(structure).write_file(folder / "CONTCAR")
    result.update(valid_structure=True, final_structure="final.cif", number_of_atoms=len(structure))
    result["artifacts"] += ["final.cif", "CONTCAR"]


def finish_energies(result, energy_ev, atoms):
    result.update(energy_ev=energy_ev, final_energy_ev=energy_ev,
                  final_energy_ev_per_atom=energy_ev / atoms if energy_ev is not None and atoms else None)


def element_valence(symbol):
    return Element(symbol).Z
