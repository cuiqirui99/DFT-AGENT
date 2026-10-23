"""VASP inputs and output checks, adapted from AI_AGNET.

Defaults need material-specific convergence checks. POTCAR stays on the host.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Vasprun

from .methods import METHOD_DEFAULTS, HYBRID_BAND_NELMIN, method_fingerprint, method_incar, normalize_method, validate_method_output
from .numerical_defaults import high_symmetry_path, path_time_reversal, potcar_choices, resolve_numerics
from .spectra import band_window


TASKS = frozenset({"relax", "scf", "bands", "dos"})
PLOT_ENERGY_WINDOW_EV = (-8.0, 8.0)
HYBRID_KPOINT_TOLERANCE_EV = 0.05
DEFAULTS = {
    "encut": 520.0,
    "ediff": 1e-5,
    "ediffg": -0.03,
    "nsw": 100,
    "mesh": None,
    "kspacing": 0.25,
    "ismear": None,
    "sigma": None,
    "electronic_type": "auto",
    "cell_relax": False,
    "nelm": 120,
    "line_density": 60,
    "nedos": 2001,
    "kpar": None,
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_structure(structure: Structure) -> None:
    if not len(structure) or not structure.is_ordered:
        raise ValueError("A nonempty, fully ordered crystal structure is required.")
    if not np.isfinite(structure.lattice.matrix).all() or structure.volume <= 1e-6:
        raise ValueError("Structure has a nonfinite or singular unit cell.")
    if not np.isfinite(structure.frac_coords).all():
        raise ValueError("Structure contains nonfinite coordinates.")
    if not structure.is_valid(tol=0.5) or min(structure.lattice.abc) < 0.5:
        raise ValueError("Structure contains overlapping sites or an invalid short cell.")
    # Dummy species and partially occupied sites cannot select a licensed PAW file.
    if any(not getattr(site.specie, "Z", 0) for site in structure):
        raise ValueError("Every site must identify a chemical element.")


def _parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(parameters)
    if "kpoint_grid" in parameters:
        if parameters.get("mesh") is not None and parameters["mesh"] != parameters["kpoint_grid"]:
            raise ValueError("mesh and kpoint_grid disagree.")
        parameters["mesh"] = parameters.pop("kpoint_grid")
    unknown = set(parameters) - set(DEFAULTS) - set(METHOD_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported VASP parameters: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **parameters}
    for key in ("encut", "ediff", "kspacing", "sigma"):
        if key == "sigma" and result[key] is None:
            continue
        if isinstance(result[key], bool) or not math.isfinite(float(result[key])) or float(result[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
        result[key] = float(result[key])
    if isinstance(result["ediffg"], bool) or not math.isfinite(float(result["ediffg"])) or float(result["ediffg"]) >= 0:
        raise ValueError("ediffg must be negative: this release uses a force convergence criterion.")
    result["ediffg"] = float(result["ediffg"])
    for key in ("nsw", "nelm", "line_density", "nedos"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if result["nedos"] < 2:
        raise ValueError("nedos must be at least 2.")
    if type(result["cell_relax"]) is not bool:
        raise ValueError("cell_relax must be a boolean.")
    if result["kpar"] is not None and (type(result["kpar"]) is not int or not 1 <= result["kpar"] <= 4096):
        raise ValueError("kpar must be a positive integer.")
    if result["ismear"] is not None and (type(result["ismear"]) is not int or result["ismear"] not in {-5, -1, 0, 1, 2}):
        raise ValueError("ismear must be one of -5, -1, 0, 1, 2.")
    if result["electronic_type"] not in {"auto", "metal", "insulator"}:
        raise ValueError("electronic_type must be auto, metal or insulator.")
    mesh = result["mesh"]
    if mesh is not None and (not isinstance(mesh, (list, tuple)) or len(mesh) != 3 or any(type(n) is not int or n <= 0 for n in mesh)):
        raise ValueError("mesh must contain three positive integers.")
    result["mesh"] = list(mesh) if mesh is not None else None
    return result


def prepare_inputs(
    structure_path: str | Path,
    destination: str | Path,
    task: str,
    parameters: dict[str, Any],
    potcar_symbols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write inputs without changing the cell.

    PBE spectra require a compatible SCF CHGCAR. Hybrid spectra are self-consistent.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    source = Path(structure_path)
    settings = _parameters(parameters)
    structure = Structure.from_file(source)
    _validate_structure(structure)
    settings, numerical_choices = resolve_numerics(structure, task, settings)
    method = normalize_method(settings, [site.specie.symbol for site in structure])
    # Match species, counts and POTCAR order.
    site_order = sorted(range(len(structure)), key=lambda index: structure[index])
    structure = Structure.from_sites([structure[index] for index in site_order])
    if method["magmom"] is not None:
        method["magmom"] = [method["magmom"][index] for index in site_order]
    poscar = Poscar(structure)
    hybrid = method["functional"] != "PBE"
    fixed_charge = task in {"bands", "dos"} and not hybrid
    requires_ncl = method["soc"] or method["spin"] == "noncollinear"
    if hybrid and task == "bands" and settings["nelm"] <= HYBRID_BAND_NELMIN:
        raise ValueError(f"Hybrid bands require nelm > {HYBRID_BAND_NELMIN}.")
    labels, potential_choices = potcar_choices(poscar.site_symbols, potcar_symbols)
    if any(not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) for label in labels):
        raise ValueError("POTCAR labels must be simple symbols such as Si or Na_pv.")

    incar_data: dict[str, Any] = {
        "SYSTEM": f"dft-agent {structure.composition.reduced_formula} {task}",
        "GGA": "PE", "ISPIN": 1, "ENCUT": settings["encut"],
        "EDIFF": settings["ediff"], "NELM": settings["nelm"],
        "PREC": "Accurate", "ALGO": "Normal", "LREAL": False, "LASPH": True,
        "ISMEAR": settings["ismear"], "SIGMA": settings["sigma"],
        "IBRION": 2 if task == "relax" else -1,
        "NSW": settings["nsw"] if task == "relax" else 0,
        "ISIF": 3 if task == "relax" and settings["cell_relax"] else 2,
        "LWAVE": False, "LCHARG": task == "scf", "ISTART": 0,
        "ICHARG": 11 if task in {"bands", "dos"} else 2,
    }
    if task == "relax":
        incar_data["EDIFFG"] = settings["ediffg"]
    if task == "dos":
        incar_data.update(NEDOS=settings["nedos"], LORBIT=11)
    if task == "bands":
        incar_data.update(ISYM=0, LORBIT=11)
    if settings["kpar"] and settings["kpar"] > 1:
        incar_data["KPAR"] = settings["kpar"]
    incar_data.update(method_incar(method, poscar.site_symbols, task))
    expected_method = {key: incar_data[key] for key in method_incar(method, poscar.site_symbols, task)}
    expected_method.update({key: incar_data.get(key, False) for key in ("LSORBIT", "LNONCOLLINEAR", "LDAU", "LHFCALC")})
    expected_method["GGA"] = incar_data["GGA"]
    site_species = [site.specie.symbol for site in structure]

    metadata: dict[str, Any] = {
        "schema_version": 1, "task": task,
        "formula": structure.composition.reduced_formula,
        "number_of_atoms": len(structure), "parameters": settings,
        "numerical_choices": numerical_choices, "potential_choices": potential_choices,
        "source_sha256": _sha256(source), "potcar_labels": labels,
        "potcar_elements": poscar.site_symbols,
        "requires_chgcar": fixed_charge, "requires_ncl": requires_ncl,
        "spectral_charge_mode": "fixed_charge" if fixed_charge else "self_consistent",
        "method": method, "method_incar_expected": expected_method,
        "method_fingerprint": method_fingerprint(method, site_species, labels),
        "method_comparison_fingerprint": method_fingerprint(method, site_species, labels, comparison=True),
        "input_site_order": site_order,
        "scope": "Check convergence and the magnetic state for your material.",
    }
    if task == "bands":
        # orig_cell maps both the basis and rotation back to the input cell.
        moments = method["magmom"] if method["spin"] != "none" else None
        types = None if moments is None else [repr(np.round(np.asarray(m, dtype=float), 2).tolist()) for m in moments]
        path = high_symmetry_path(structure, settings["line_density"], site_types=types,
                                  time_reversal=path_time_reversal(moments, method["soc"] or method["spin"] == "noncollinear"))
        fractional = np.asarray(path["reciprocal_fractional"])
        cartesian = structure.lattice.reciprocal_lattice.get_cartesian_coords(fractional)
        point_labels = path["labels"]
        if not len(fractional) or not np.isfinite(fractional).all():
            raise ValueError("Could not construct a finite high-symmetry path.")
        mesh_points = [list(point) for point in product(*(np.arange(n) / n for n in settings["mesh"]))] if hybrid else []
        all_points = mesh_points + fractional.tolist()
        kpoints = Kpoints(
            comment="High-symmetry path in the actual POSCAR reciprocal basis",
            num_kpts=len(all_points), style=Kpoints.supported_modes.Reciprocal,
            kpts=all_points, kpts_weights=[1.0] * len(mesh_points) + [0.0 if hybrid else 1.0] * len(fractional),
            labels=[""] * len(mesh_points) + point_labels,
        )
        metadata["band_path_offset"] = len(mesh_points)
        metadata["band_path_count"] = len(fractional)
        if hybrid:
            metadata["hybrid_mesh"] = {"reciprocal_fractional": mesh_points, "weights": [1.0] * len(mesh_points)}
            metadata["band_convergence_note"] = "Zero-weight path convergence needs a separate check; SCF convergence alone does not prove it."
        metadata["band_path"] = {
            "method": path["method"], "symprec_angstrom": path["symprec_angstrom"],
            "min_segment_points": path["min_segment_points"],
            "symmetry_basis": "structural; magnetic space group not inferred",
            "labels": point_labels, "reciprocal_fractional": fractional.tolist(),
            "cartesian_inv_angstrom": np.asarray(cartesian).tolist(),
            "segments": path["segments"],
            "distance_inv_angstrom": path["distance_inv_angstrom"],
            "is_supercell": bool(path["is_supercell"]),
        }
        # Tetrahedron integration requires a regular mesh, not a symmetry line.
        if settings["ismear"] == -5:
            raise ValueError("ISMEAR=-5 is not supported on a band path; use ISMEAR=0.")
    else:
        kpoints = Kpoints.gamma_automatic(tuple(settings["mesh"]))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "POSCAR").write_text(poscar.get_str(), encoding="utf-8", newline="\n")
    (destination / "INCAR").write_text(str(Incar(incar_data)), encoding="utf-8", newline="\n")
    (destination / "KPOINTS").write_text(str(kpoints), encoding="utf-8", newline="\n")
    metadata["input_sha256"] = {name: _sha256(destination / name) for name in ("POSCAR", "INCAR", "KPOINTS")}
    _write_json(destination / "metadata.json", metadata)
    return metadata


def hybrid_kpar(structure_path: str | Path, parameters: dict[str, Any], task: str, mpi_tasks: int) -> int:
    """K-point groups for exact exchange on mpi_tasks ranks.

    Without KPAR, VASP spreads the bands of each k-point over all ranks and rounds NBANDS up to
    their number: silicon got 64 bands on 64 ranks, and its HSE06 SCF took 22 minutes instead of 34
    seconds with KPAR = 16. Groups never outnumber the k-points, and larger cells keep a few ranks
    per group for memory.
    """
    structure = Structure.from_file(structure_path)
    settings, _ = resolve_numerics(structure, task, _parameters(parameters))
    method = normalize_method(settings, [site.specie.symbol for site in structure])
    mesh = settings["mesh"]
    count = int(np.prod(mesh))
    # Band paths keep the whole weighted mesh; noncollinear runs use no symmetry.
    if task != "bands" and not (method["soc"] or method["spin"] == "noncollinear"):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        count = len(SpacegroupAnalyzer(structure, symprec=1e-3).get_ir_reciprocal_mesh(mesh))
    ranks_per_group = 1 + len(structure) // 16
    return max(d for d in range(1, mpi_tasks + 1) if mpi_tasks % d == 0 and d <= count and d * ranks_per_group <= max(mpi_tasks, ranks_per_group))


def _same_structure(actual: Structure, expected: Structure) -> bool:
    if [site.species for site in actual] != [site.species for site in expected]:
        return False
    if not np.allclose(actual.lattice.matrix, expected.lattice.matrix, rtol=1e-6, atol=1e-5):
        return False
    delta = actual.frac_coords - expected.frac_coords
    return bool(np.allclose(delta - np.rint(delta), 0, atol=1e-5))


def _write_csv(path: Path, header: list[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _magnetization(output: Path, count: int, metadata: dict[str, Any]) -> dict[str, Any]:
    """Read final OUTCAR spin moments without treating sphere sums as cell totals."""
    method = metadata.get("method", {})
    ncl = bool(metadata.get("requires_ncl", False))
    result: dict[str, Any] = {
        "available": False, "unit": "mu_B", "site_order": "POSCAR",
        "basis": "SAXIS spinor basis" if ncl else "collinear spin axis",
        "saxis": method.get("saxis", [0.0, 0.0, 1.0]),
        "site_moments": None, "total_moment": None,
        "note": "Site moments are PAW sphere projections; their sum is not the full cell moment.",
    }
    path = output / "OUTCAR"
    if not path.is_file():
        return result
    # OUTCAR tables are scalar for collinear runs and x/y/z in spinor space for NCL.
    tables: dict[str, list[float]] = {}
    axis = None
    values: list[float] = []
    number = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
    with path.open(errors="replace") as stream:
        for line in stream:
            match = re.fullmatch(r"\s*magnetization \(([xyz])\)\s*", line)
            if match:
                axis, values = match[1], []
                if axis == "x":
                    tables = {}
                else:
                    tables.pop(axis, None)
            elif axis is not None:
                row = re.fullmatch(r"\s*(\d+)\s+((?:" + number + r"\s+)*" + number + r")\s*", line)
                if row and int(row[1]) == len(values) + 1:
                    values.append(float(row[2].split()[-1]))
                    if len(values) == count:
                        tables[axis] = values
                        axis = None
            if "number of electron" in line and "magnetization" in line:
                raw = re.findall(number, line.split("magnetization", 1)[1])
                if len(raw) == (3 if ncl else 1):
                    moment = [float(value) for value in raw]
                    if np.isfinite(moment).all():
                        result["total_moment"] = moment if ncl else moment[0]
    if ncl and all(key in tables for key in ("x", "y", "z")):
        result["site_moments"] = np.asarray([tables[key] for key in ("x", "y", "z")]).T.tolist()
    elif not ncl and "x" in tables:
        result["site_moments"] = tables["x"]
    if result["site_moments"] is not None:
        result["available"] = True
    return result


def _hybrid_kpoint_consistency(run: Vasprun) -> dict[str, Any]:
    """Check orbital energies independently of the weighted total-energy stop."""
    points = np.asarray(run.actual_kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Hybrid k-points are unavailable or nonfinite.")
    if not run.eigenvalues:
        raise ValueError("Band eigenvalues are unavailable.")
    pairs = []
    for first in range(len(points)):
        difference = points[first + 1:] - points[first]
        equivalent = np.all(np.abs(difference - np.rint(difference)) <= 1e-6, axis=1)
        pairs.extend((first, first + 1 + int(index)) for index in np.flatnonzero(equivalent))
    check: dict[str, Any] = {
        "tolerance_ev": HYBRID_KPOINT_TOLERANCE_EV,
        "scope": "All returned bands, energy-sorted within each spin channel.",
        "equivalent_pair_count": len(pairs), "compared_eigenvalue_count": 0,
        "max_energy_deviation_ev": 0.0, "worst_pair": None,
        "passed": bool(pairs),
    }
    for spin, eigenvalues in run.eigenvalues.items():
        values = np.asarray(eigenvalues, dtype=float)
        if values.ndim != 3 or values.shape[0] != len(points) or values.shape[1] == 0 or values.shape[2] != 2 or not np.isfinite(values).all():
            raise ValueError("Band eigenvalues do not match the finite k-point path.")
        energies = np.sort(values[:, :, 0], axis=1)
        for first, second in pairs:
            deviations = np.abs(energies[first] - energies[second])
            check["compared_eigenvalue_count"] += len(deviations)
            band = int(np.argmax(deviations))
            maximum = float(deviations[band])
            if maximum > check["max_energy_deviation_ev"]:
                check["max_energy_deviation_ev"] = maximum
                check["worst_pair"] = {"kpoint_indices_zero_based": [first, second], "spin": int(spin), "band": band + 1}
    check["passed"] = bool(pairs) and check["max_energy_deviation_ev"] <= HYBRID_KPOINT_TOLERANCE_EV
    return check


def _save_figure(fig, output: Path, name: str) -> list[str]:
    from matplotlib import rc_context

    if len(fig.axes) == 1:
        fig.tight_layout(pad=1.2)
    with rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "dft-agent"}):
        for extension in ("png", "pdf", "svg"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None} if extension == "svg" else None
            fig.savefig(output / f"{name}.{extension}", dpi=240, facecolor="white", bbox_inches="tight", metadata=metadata)
    return [f"{name}.{extension}" for extension in ("png", "pdf", "svg")]


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=11, colors="#253247")
    for spine in ax.spines.values():
        spine.set_color("#8792a2")
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(axis="y", color="#e9edf2", linewidth=0.5, zorder=0)


def _projected_dos(run: Vasprun, output: Path, reference: float) -> list[str]:
    """Export PAW projections only when VASP actually returned them."""
    from matplotlib import pyplot as plt

    dos = run.complete_dos
    if not getattr(dos, "pdos", None):
        return []
    artifacts = []
    palette = ("#2366a8", "#c26a24", "#238878", "#8651a2", "#bb456b", "#788238")
    for name, groups in (("dos_elements", dos.get_element_dos()), ("dos_orbitals", dos.get_spd_dos())):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        try:
            _style_axes(ax)
            rows = []
            for index, (label, projected) in enumerate(groups.items()):
                energy = np.asarray(projected.energies, dtype=float) - reference
                if not np.isfinite(energy).all():
                    raise ValueError("Nonfinite projected DOS energies.")
                channels = sorted(projected.densities.items(), key=lambda item: -int(item[0]))
                for spin, density in channels:
                    density = np.asarray(density, dtype=float)
                    if density.shape != energy.shape or not np.isfinite(density).all():
                        raise ValueError("Nonfinite or inconsistent projected DOS.")
                    rows.extend([float(e), str(label), int(spin), float(value)] for e, value in zip(energy, density))
                    signed = -density if int(spin) == -1 else density
                    channel_label = str(label) if len(channels) == 1 else f"{label} {'↑' if int(spin) == 1 else '↓'}"
                    ax.plot(energy, signed, color=palette[index % len(palette)],
                            linestyle="--" if int(spin) == -1 else "-", linewidth=1.5, label=channel_label)
            _write_csv(output / f"{name}.csv", ["energy_minus_fermi_ev", "projection", "spin", "dos_states_per_ev"], rows)
            ax.set(xlabel=r"$E - E_\mathrm{F}$ (eV)", ylabel="Projected DOS (states/eV)", xlim=PLOT_ENERGY_WINDOW_EV)
            ax.axvline(0, color="#687588", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="#687588", linewidth=0.5)
            # Rescale from the visible interval, without clipping the CSV.
            visible = [value * (-1 if spin == -1 else 1) for e, _, spin, value in rows
                       if PLOT_ENERGY_WINDOW_EV[0] <= e <= PLOT_ENERGY_WINDOW_EV[1]]
            if visible:
                low, high = min(0.0, min(visible)), max(0.0, max(visible))
                ax.set_ylim(low * 1.08, high * 1.08 if high else 1.0)
            ax.legend(frameon=False, ncol=min(4, max(1, len(groups))), fontsize=10, loc="upper right")
            artifacts.extend([f"{name}.csv", *_save_figure(fig, output, name)])
        finally:
            plt.close(fig)
    rows = []
    for element in dos.structure.composition.elements:
        for orbital, projected in dos.get_element_spd_dos(element).items():
            for spin, density in projected.densities.items():
                energy = np.asarray(projected.energies) - reference
                density = np.asarray(density)
                if density.shape != energy.shape or not np.isfinite(density).all():
                    raise ValueError("Nonfinite or inconsistent element-orbital DOS.")
                rows.extend([float(e), str(element), str(orbital), int(spin), float(value)] for e, value in zip(energy, density))
    _write_csv(output / "dos_element_orbitals.csv", ["energy_minus_fermi_ev", "element", "orbital", "spin", "dos_states_per_ev"], rows)
    return [*artifacts, "dos_element_orbitals.csv"]


def _ionic_energy_ev(step: dict[str, Any]) -> float | None:
    """Read sigma-to-zero energy using Vasprun.final_energy's XML correction."""
    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    reported = finite(step.get("e_0_energy"))
    if reported is None:
        return None
    electronic = step.get("electronic_steps") or []
    if electronic:
        last = electronic[-1]
        values = [finite(last.get("e_0_energy")), finite(last.get("e_fr_energy")),
                  finite(step.get("e_fr_energy"))]
        if all(value is not None for value in values):
            # Some VASP XML files write a zero ionic E0. Keep the electronic
            # entropy correction and the ionic free energy, as pymatgen does.
            corrected = float(np.round(values[0] - values[1] + values[2], 8))
            if not math.isfinite(corrected):
                return None
            if abs(reported - corrected) > 1e-7:
                return corrected
    return reported


def _export_tables(run: Vasprun, output: Path) -> list[str]:
    """Keep numerical observations in VASP's units and site order."""
    artifacts = []
    structure = run.final_structure
    _write_json(output / "structure.json", {
        "formula": structure.composition.reduced_formula,
        "lattice_angstrom": np.asarray(structure.lattice.matrix).tolist(),
        "volume_angstrom3": float(structure.volume), "site_order": "POSCAR",
        "sites": [{"site": i + 1, "element": site.specie.symbol,
                   "fractional": site.frac_coords.tolist(), "cartesian_angstrom": site.coords.tolist()}
                  for i, site in enumerate(structure)],
    })
    artifacts.append("structure.json")
    ionic_rows, electronic_rows = [], []
    for index, step in enumerate(run.ionic_steps, 1):
        def finite(value):
            return float(value) if value is not None and math.isfinite(float(value)) else None
        forces = np.asarray(step.get("forces", []), dtype=float)
        valid_forces = forces.shape == (len(structure), 3) and np.isfinite(forces).all()
        maximum = float(np.linalg.norm(forces, axis=1).max()) if valid_forces else None
        step_structure = step.get("structure")
        volume = finite(step_structure.volume) if step_structure is not None else None
        ionic_rows.append([index, _ionic_energy_ev(step), finite(step.get("e_fr_energy")), maximum, volume])
        previous = None
        for iteration, electronic in enumerate(step.get("electronic_steps", []), 1):
            free = finite(electronic.get("e_fr_energy"))
            delta = free - previous if free is not None and previous is not None else None
            electronic_rows.append([index, iteration, free, finite(electronic.get("e_0_energy")), delta])
            previous = free
    if ionic_rows:
        _write_csv(output / "ionic_steps.csv", ["ionic_step", "energy_ev", "free_energy_ev", "max_force_ev_angstrom", "volume_angstrom3"], ionic_rows)
        artifacts.append("ionic_steps.csv")
    if electronic_rows:
        _write_csv(output / "electronic_steps.csv", ["ionic_step", "electronic_step", "free_energy_ev", "energy_ev", "delta_free_energy_ev"], electronic_rows)
        artifacts.append("electronic_steps.csv")
    if run.ionic_steps:
        final = run.ionic_steps[-1]
        forces = np.asarray(final.get("forces", []), dtype=float)
        if forces.shape == (len(structure), 3) and np.isfinite(forces).all():
            _write_csv(output / "forces.csv", ["site", "element", "fx_ev_angstrom", "fy_ev_angstrom", "fz_ev_angstrom", "norm_ev_angstrom"],
                       [[i + 1, site.specie.symbol, *force, float(np.linalg.norm(force))] for i, (site, force) in enumerate(zip(structure, forces))])
            artifacts.append("forces.csv")
        stress = np.asarray(final.get("stress", []), dtype=float)
        if stress.shape == (3, 3) and np.isfinite(stress).all():
            _write_csv(output / "stress.csv", ["component", "stress_kbar"],
                       [[first + second, float(stress[i, j])] for i, first in enumerate("xyz") for j, second in enumerate("xyz")])
            artifacts.append("stress.csv")
    return artifacts


def _export_plots(run: Vasprun, output: Path, task: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.6) if task == "bands" else (7.2, 4.8))
    _style_axes(ax)
    details: dict[str, Any] = {"schema_version": 1, "task": task, "csv_contains_full_data": True,
                               "figure_formats": ["png", "pdf", "svg"]}
    try:
        if task == "relax":
            energies = [_ionic_energy_ev(step) for step in run.ionic_steps]
            if not energies or any(energy is None for energy in energies):
                raise ValueError("No finite ionic energy history.")
            details["energy_definition"] = "sigma_to_zero"
            name = "relax_energy"
            _write_csv(output / f"{name}.csv", ["ionic_step", "energy_ev"], enumerate(energies, 1))
            ax.plot(range(1, len(energies) + 1), energies, color="#2366a8", marker="o", markersize=4, linewidth=1.6)
            ax.set(xlabel="Ionic step", ylabel="Energy (eV)")
            ax.ticklabel_format(axis="y", useOffset=False)
            from matplotlib.ticker import MaxNLocator
            # Whole steps only, also when the relaxation converged after one or two steps.
            ax.set_xlim(0.5, len(energies) + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        elif task == "dos":
            dos = run.complete_dos
            if dos is None or not np.isfinite(dos.energies).all() or not math.isfinite(float(run.efermi)):
                raise ValueError("No finite total DOS or Fermi energy.")
            name = "dos"
            energies = np.asarray(dos.energies) - run.efermi
            densities = list(dos.densities.items())
            if not densities or any(not np.isfinite(values).all() for _, values in densities):
                raise ValueError("Nonfinite DOS densities.")
            header = ["energy_minus_fermi_ev"] + [f"dos_spin_{int(spin)}" for spin, _ in densities]
            _write_csv(output / f"{name}.csv", header, zip(energies, *(values for _, values in densities)))
            for spin, values in densities:
                signed = -np.asarray(values) if int(spin) == -1 else np.asarray(values)
                color = "#2366a8" if int(spin) == 1 else "#c26a24"
                label = "Total" if len(densities) == 1 else ("Spin ↑" if int(spin) == 1 else "Spin ↓")
                ax.plot(energies, signed, color=color, linewidth=1.6, label=label)
                ax.fill_between(energies, signed, color=color, alpha=0.12, linewidth=0)
            ax.set(xlabel=r"$E - E_\mathrm{F}$ (eV)", ylabel="DOS (states/eV)")
            ax.legend(frameon=False, fontsize=11)
            ax.set_xlim(*PLOT_ENERGY_WINDOW_EV)
            visible = (energies >= PLOT_ENERGY_WINDOW_EV[0]) & (energies <= PLOT_ENERGY_WINDOW_EV[1])
            if visible.any():
                peak = max(float(np.max(values[visible])) for _, values in densities)
                ax.set_ylim(-peak * 1.08 if any(int(spin) == -1 for spin, _ in densities) else 0, peak * 1.08 if peak > 0 else 1.0)
            ax.axvline(0, color="#687588", linewidth=0.8, linestyle="--")
            details.update(energy_reference_ev=float(run.efermi), energy_reference_source="dos_run_fermi",
                           energy_window_ev=list(PLOT_ENERGY_WINDOW_EV), density_unit="states/eV per cell",
                           down_spin_plot_sign=-1, csv_densities_are_unsigned=True,
                           projection_note="PAW sphere projections need not sum to the total DOS.",
                           projected_dos_available=bool(getattr(dos, "pdos", None)))
        elif task == "bands":
            if not run.eigenvalues:
                raise ValueError("Band eigenvalues are unavailable.")
            all_kpts = np.asarray(run.actual_kpoints, dtype=float)
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            hybrid = metadata.get("method", {}).get("functional", "PBE") != "PBE"
            offset = int(metadata.get("band_path_offset", 0))
            kpts = all_kpts[offset:]
            if hybrid:
                mesh = np.asarray(metadata.get("hybrid_mesh", {}).get("reciprocal_fractional", []))
                if offset <= 0 or mesh.shape != all_kpts[:offset].shape or not np.allclose(mesh - all_kpts[:offset] - np.rint(mesh - all_kpts[:offset]), 0, atol=1e-6):
                    raise ValueError("Hybrid output does not match the frozen regular mesh.")
                weights = np.asarray(getattr(run, "actual_kpoints_weights", []))
                if weights.shape != (len(all_kpts),) or not np.isfinite(weights).all() or np.any(weights[:offset] <= 0) or not np.allclose(weights[offset:], 0):
                    raise ValueError("Hybrid bands require positive mesh weights and zero path weights.")
                if not np.allclose(weights[:offset] / sum(weights[:offset]), np.ones(offset) / offset, atol=1e-7):
                    raise ValueError("Hybrid regular-mesh weights differ from the prepared inputs.")
            fermi = run.efermi if hybrid else metadata.get("scf_fermi_energy_ev")
            if fermi is None or not math.isfinite(float(fermi)):
                raise ValueError("A finite hybrid-run Fermi energy is required." if hybrid else "A finite preceding SCF Fermi energy is required for band analysis.")
            fermi = float(fermi)
            path_metadata = metadata.get("band_path", {})
            if not path_metadata:
                raise ValueError("Frozen band path metadata.json is required for band analysis.")
            labels = path_metadata.get("labels", [])
            if len(labels) != len(kpts) or int(metadata.get("band_path_count", len(labels))) != len(kpts):
                raise ValueError("Band path metadata does not match output k-points.")
            planned = np.asarray(path_metadata.get("reciprocal_fractional", []))
            if planned.shape != kpts.shape or not np.allclose(planned, kpts, atol=1e-6, rtol=0):
                raise ValueError("Output k-points differ from the frozen band path.")
            distance = np.asarray(path_metadata.get("distance_inv_angstrom", []))
            segments = path_metadata.get("segments", [])
            if distance.shape != (len(kpts),) or not np.isfinite(distance).all() or not segments:
                raise ValueError("Frozen band path is missing its distance or segment definition.")
            covered = np.zeros(len(kpts), dtype=bool)
            for segment in segments:
                if len(segment) != 2 or any(type(index) is not int for index in segment):
                    raise ValueError("Invalid band path segment.")
                start, stop = segment
                if not 0 <= start < stop <= len(kpts) or stop - start < 2 or np.any(np.diff(distance[start:stop]) < 0):
                    raise ValueError("Invalid band path segment bounds or distances.")
                covered[start:stop] = True
            if not covered.all():
                raise ValueError("Band path segments do not cover all output k-points.")
            rows = []
            mesh_rows = []
            spin_labels = {}
            plotted = []
            for spin, eigenvalues in run.eigenvalues.items():
                all_values = np.asarray(eigenvalues)
                if all_values.ndim != 3 or all_values.shape[0] != len(all_kpts) or all_values.shape[2] != 2 or not np.isfinite(all_values).all():
                    raise ValueError("Band eigenvalues do not match the finite k-point path.")
                for ik, point in enumerate(all_kpts[:offset]):
                    for ib in range(all_values.shape[1]):
                        mesh_rows.append([ik, *point, int(spin), ib + 1, all_values[ik, ib, 0] - fermi, all_values[ik, ib, 1]])
                values = all_values[offset:]
                energies = values[:, :, 0] - fermi
                plotted.append(energies)
                for start, stop in segments:
                    lines = ax.plot(distance[start:stop], energies[start:stop], color="#2366a8" if int(spin) == 1 else "#c26a24",
                                    linestyle="-" if int(spin) == 1 else "--", linewidth=1.15, alpha=0.95)
                    if lines:
                        spin_labels[int(spin)] = lines[0]
                for ik, point in enumerate(kpts):
                    for ib in range(values.shape[1]):
                        rows.append([ik, distance[ik], *point, labels[ik], int(spin), ib + 1, energies[ik, ib], values[ik, ib, 1]])
            name = "bands"
            _write_csv(output / f"{name}.csv", ["kpoint", "distance_inv_angstrom", "kx", "ky", "kz", "label", "spin", "band", "energy_minus_fermi_ev", "occupation"], rows)
            if mesh_rows:
                _write_csv(output / "bands_mesh.csv", ["kpoint", "kx", "ky", "kz", "spin", "band", "energy_minus_fermi_ev", "occupation"], mesh_rows)
            ticks: dict[float, str] = {}
            for x, label in zip(distance, labels):
                if label:
                    previous = ticks.get(float(x))
                    ticks[float(x)] = label if not previous or previous == label else f"{previous}|{label}"
            def display_label(label):
                return "|".join("Γ" if part == "GAMMA" else re.sub(r"_([0-9]+)", r"$_{\1}$", part) for part in label.split("|"))
            ax.set_xticks(list(ticks), [display_label(label) for label in ticks.values()])
            for position in ticks:
                ax.axvline(position, color="0.85", linewidth=0.6, zorder=0)
            ax.set(xlabel="", ylabel=r"$E - E_\mathrm{F}$ (eV)")
            ax.grid(False)
            if len(spin_labels) > 1:
                ax.legend(list(spin_labels.values()), ["Spin ↑" if spin == 1 else "Spin ↓" for spin in spin_labels], frameon=False, fontsize=10)
            # Like the other codes' plots, stop just below the highest computed band: above it the
            # bands were not computed, and that band follows whichever uncomputed band crosses it.
            # With KPAR a silicon HSE06 path had 8 bands, whose highest showed false kinks at 7.6 eV.
            window = band_window(np.asarray(plotted))
            ax.set_ylim(*window)
            ax.set_xlim(float(distance[0]), float(distance[-1]))
            ax.axhline(0, color="#687588", linewidth=0.85, linestyle="--", zorder=0)
            details.update(energy_reference_ev=fermi, energy_reference_source="hybrid_scf_mesh" if hybrid else "preceding_scf",
                           energy_window_ev=list(window), segments=segments,
                           ticks=[{"distance_inv_angstrom": position, "label": label} for position, label in ticks.items()],
                           kpoint_count=len(kpts), interpolation="none", band_path_method=path_metadata.get("method"))
        else:
            return []
        artifacts = [f"{name}.csv", *_save_figure(fig, output, name)]
        if task == "bands" and mesh_rows:
            artifacts.append("bands_mesh.csv")
        if task == "dos":
            artifacts.extend(_projected_dos(run, output, float(run.efermi)))
        _write_json(output / "plot_metadata.json", details)
        return [*artifacts, "plot_metadata.json"]
    finally:
        plt.close(fig)


def export_bands_dos(bands_dir: str | Path, dos_dir: str | Path) -> list[str]:
    """Align both spectra to the band plot's SCF reference without changing CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    bands_dir, dos_dir = Path(bands_dir), Path(dos_dir)
    names = ["bands_dos.png", "bands_dos.pdf", "bands_dos.svg", "bands_dos_metadata.json"]
    for name in names:
        (bands_dir / name).unlink(missing_ok=True)
    def read_json(directory, name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    band_info, dos_info = read_json(bands_dir, "plot_metadata.json"), read_json(dos_dir, "plot_metadata.json")
    band_input, dos_input = read_json(bands_dir, "metadata.json"), read_json(dos_dir, "metadata.json")
    if band_info.get("task") != "bands" or dos_info.get("task") != "dos":
        raise ValueError("A band result and a DOS result are required.")
    if not band_input.get("method_fingerprint") or band_input["method_fingerprint"] != dos_input.get("method_fingerprint"):
        raise ValueError("Bands and DOS use different methods.")
    if not _same_structure(Structure.from_file(bands_dir / "final_structure.vasp"), Structure.from_file(dos_dir / "final_structure.vasp")):
        raise ValueError("Bands and DOS use different structures.")
    reference, dos_reference = float(band_info["energy_reference_ev"]), float(dos_info["energy_reference_ev"])
    if not math.isfinite(reference) or not math.isfinite(dos_reference):
        raise ValueError("Both spectra need a finite energy reference.")
    def read_csv(directory, name):
        with (directory / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    bands = read_csv(bands_dir, "bands.csv")
    total = read_csv(dos_dir, "dos.csv")
    if not bands or not total:
        raise ValueError("Both spectra need numerical data.")
    grouped = {}
    for row in bands:
        grouped.setdefault((int(row["spin"]), int(row["band"])), []).append(row)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 5.8), sharey=True, gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.08})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92)
    try:
        for ax in (left, right):
            _style_axes(ax)
            ax.grid(False)
            ax.axhline(0, color="#687588", linewidth=0.8, linestyle="--")
        for (spin, _), rows in grouped.items():
            rows.sort(key=lambda row: int(row["kpoint"]))
            x = np.asarray([float(row["distance_inv_angstrom"]) for row in rows])
            y = np.asarray([float(row["energy_minus_fermi_ev"]) for row in rows])
            if [int(row["kpoint"]) for row in rows] != list(range(band_info["kpoint_count"])) or not np.isfinite([x, y]).all():
                raise ValueError("Band CSV is incomplete or nonfinite.")
            for start, stop in band_info["segments"]:
                left.plot(x[start:stop], y[start:stop], color="#2366a8" if spin == 1 else "#c26a24", linewidth=1.1, linestyle="-" if spin == 1 else "--")
        ticks = band_info["ticks"]
        labels = [re.sub(r"_([0-9]+)", r"$_{\1}$", tick["label"].replace("GAMMA", "Γ")) for tick in ticks]
        left.set_xticks([tick["distance_inv_angstrom"] for tick in ticks], labels)
        for tick in ticks:
            left.axvline(tick["distance_inv_angstrom"], color="#e0e4ea", linewidth=0.6, zorder=0)
        left.set(xlim=(float(x[0]), float(x[-1])), ylim=tuple(band_info.get("energy_window_ev", PLOT_ENERGY_WINDOW_EV)),
                 ylabel=r"$E - E_\mathrm{F}$ (eV)", title="Bands")
        energy = np.asarray([float(row["energy_minus_fermi_ev"]) + dos_reference - reference for row in total])
        density = np.asarray([sum(float(value) for key, value in row.items() if key.startswith("dos_spin_")) for row in total])
        if not np.isfinite([energy, density]).all():
            raise ValueError("DOS CSV is nonfinite.")
        right.fill_betweenx(energy, density, color="#dfe5ed", alpha=0.8, linewidth=0)
        right.plot(density, energy, color="#687588", linewidth=1.0, label="Total")
        projected = dos_dir / "dos_elements.csv"
        if projected.is_file():
            groups = {}
            for row in read_csv(dos_dir, projected.name):
                key = (row["projection"], float(row["energy_minus_fermi_ev"]))
                groups[key] = groups.get(key, 0.0) + float(row["dos_states_per_ev"])
            palette = ("#2366a8", "#c26a24", "#238878", "#8651a2")
            for index, element in enumerate(dict.fromkeys(key[0] for key in groups)):
                values = sorted((e + dos_reference - reference, value) for (label, e), value in groups.items() if label == element)
                if not np.isfinite(values).all():
                    raise ValueError("Projected DOS CSV is nonfinite.")
                right.plot([item[1] for item in values], [item[0] for item in values], color=palette[index % len(palette)], linewidth=1.4, label=element)
        visible = (energy >= PLOT_ENERGY_WINDOW_EV[0]) & (energy <= PLOT_ENERGY_WINDOW_EV[1])
        right.set(xlim=(0, float(density[visible].max()) * 1.08 if visible.any() and density[visible].max() > 0 else 1.0), xlabel="DOS (states/eV)", title="DOS")
        right.tick_params(labelleft=False)
        right.legend(frameon=False, fontsize=10, loc="upper right")
        artifacts = _save_figure(fig, bands_dir, "bands_dos")
        sources = {"bands.csv": _sha256(bands_dir / "bands.csv"), "dos.csv": _sha256(dos_dir / "dos.csv"),
                   "bands/plot_metadata.json": _sha256(bands_dir / "plot_metadata.json"),
                   "dos/plot_metadata.json": _sha256(dos_dir / "plot_metadata.json")}
        if projected.is_file():
            sources["dos_elements.csv"] = _sha256(projected)
        _write_json(bands_dir / "bands_dos_metadata.json", {
            "schema_version": 1, "energy_reference_ev": reference,
            "energy_reference_source": band_info["energy_reference_source"],
            "dos_original_reference_ev": dos_reference, "dos_energy_shift_ev": dos_reference - reference,
            "energy_window_ev": list(PLOT_ENERGY_WINDOW_EV), "source_sha256": sources,
            "kpoint_count": band_info["kpoint_count"], "method_fingerprint": band_input["method_fingerprint"],
            "spin_channels": "summed DOS; separate bands", "interpolation": "none",
        })
        return [*artifacts, "bands_dos_metadata.json"]
    finally:
        plt.close(fig)


def _band_gap(run, task: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Occupation-based edges on the retained k-points, never a full-zone claim."""
    tolerance = 1e-3
    result = {"available": False, "status": "unavailable",
              "scope": "sampled_path" if task == "bands" else "sampled_kpoints",
              "gap_ev": None, "direct_gap_ev": None, "vbm_ev": None, "cbm_ev": None,
              "partial_occupations": False, "occupation_tolerance": tolerance,
              "full_bz_validated": False, "reason": "Eigenvalues and occupations are unavailable."}
    if task not in {"scf", "bands", "dos"}:
        return result
    eigenvalues = getattr(run, "eigenvalues", None)
    if not isinstance(eigenvalues, dict) or not eigenvalues:
        return result
    try:
        channels = [np.asarray(value, dtype=float) for value in eigenvalues.values()]
        count = len(getattr(run, "actual_kpoints", []))
        if not count or any(value.ndim != 3 or value.shape[0] != count or value.shape[2] != 2
                            or not np.isfinite(value).all() for value in channels):
            return result
        if task == "bands":
            offset, length = metadata.get("band_path_offset", 0), metadata.get("band_path_count")
            if type(offset) is not int or type(length) is not int or offset < 0 or length < 1 or offset + length != count:
                result["reason"] = "The eigenvalues are not linked to the prepared band path."
                return result
            channels = [value[offset:offset + length] for value in channels]
        # vasprun.xml reports occupations per state in [0, 1], including ISPIN=1.
        values = np.concatenate(channels, axis=1)
        energies, occupations = values[:, :, 0], values[:, :, 1]
        if np.any(occupations < -tolerance) or np.any(occupations > 1 + tolerance):
            result["reason"] = "Occupations are outside the supported zero-to-one convention."
            return result
        result["kpoint_count"] = len(values)
        partial = (occupations > tolerance) & (occupations < 1 - tolerance)
        if np.any(partial):
            result.update(status="metallic_or_partially_occupied", partial_occupations=True,
                          reason="Partial occupations prevent a resolved gap; smearing can also cause them.")
            return result
        occupied, empty = occupations >= 1 - tolerance, occupations <= tolerance
        if not np.all(occupied.any(axis=1)) or not np.all(empty.any(axis=1)):
            result["reason"] = "Both occupied and empty states are needed at every sampled k-point."
            return result
        valence = np.where(occupied, energies, -np.inf).max(axis=1)
        conduction = np.where(empty, energies, np.inf).min(axis=1)
        vbm, cbm = float(valence.max()), float(conduction.min())
        # A different number of occupied states at different k-points signals a
        # Fermi surface even if sparse sampling leaves a positive edge separation.
        crossing = any(np.ptp((channel[:, :, 1] >= 1 - tolerance).sum(axis=1)) > 0 for channel in channels)
        metallic = bool(crossing or cbm <= vbm)
        result.update(available=True, status="metallic" if metallic else "gapped",
                      gap_ev=0.0 if metallic else cbm - vbm,
                      direct_gap_ev=None if metallic else float((conduction - valence).min()),
                      vbm_ev=vbm, cbm_ev=cbm,
                      reason="Only the retained k-points were checked; this is not a converged full-zone or optical gap.")
    except (TypeError, ValueError):
        pass
    return result


def analyze_outputs(output_dir: str | Path, task: str, expected_structure_path: str | Path) -> dict[str, Any]:
    """Reject incomplete, unconverged or mismatched XML.

    Acceptance does not establish scientific accuracy.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Remove stale results before checking new output.
    generated = ["final_structure.cif", "final_structure.vasp", "relax_energy.csv", "relax_energy.png", "bands.csv", "bands.png", "bands_mesh.csv", "dos.csv", "dos.png", "magnetization.json"]
    generated += [f"{stem}.{extension}" for stem in ("relax_energy", "bands", "dos", "dos_elements", "dos_orbitals") for extension in ("png", "pdf", "svg", "csv")]
    generated += ["bands_dos.png", "bands_dos.pdf", "bands_dos.svg", "bands_dos_metadata.json"]
    generated += ["plot_metadata.json", "dos_element_orbitals.csv", "ionic_steps.csv", "electronic_steps.csv", "forces.csv", "stress.csv", "structure.json"]
    for filename in generated:
        (output / filename).unlink(missing_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1, "task": task, "success": False,
        "converged_electronic": False, "converged_ionic": False,
        "valid_structure": False, "final_energy_ev": None,
        "final_max_force_ev_angstrom": None, "vasp_version": None,
        "forces_available": False,
        "ionic_steps_count": None, "ionic_iteration_limit_reached": False,
        "ionic_force_limit_ev_angstrom": None,
        "fermi_energy_ev": None,
        "final_energy_ev_per_atom": None,
        "reason": "Results have not been checked yet.", "artifacts": [],
        "scientific_accuracy_validated": False,
    }
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        hybrid = metadata.get("method", {}).get("functional", "PBE") != "PBE"
        expected = Structure.from_file(expected_structure_path).get_sorted_structure()
        _validate_structure(expected)
        xml = output / "vasprun.xml"
        if not xml.is_file() or not xml.stat().st_size:
            logs = "".join((output / name).read_text(errors="replace") for name in ("slurm.out", "OUTCAR") if (output / name).is_file())
            if "No license file found" in logs:
                raise ValueError("VASP stopped before the calculation because it found no licence file. This VASP 6 build looks for "
                                 "your vasp_license key in ~/.vasp/, the run folder or VASP_LICENSE_FILE; install it as the cluster's "
                                 "VASP notes describe, or set the environment's VASP command to a build your licence already covers.")
            raise ValueError("vasprun.xml is missing or empty. The job may have ended before VASP finished.")
        run = Vasprun(xml, parse_potcar_file=False, parse_eigen=task in {"scf", "bands", "dos"}, parse_dos=task in {"scf", "dos"} or (hybrid and task == "bands"), exception_on_bad_xml=True)
        result["vasprun_sha256"] = _sha256(xml)
        result["vasp_version"] = str(run.vasp_version)
        fermi = getattr(run, "efermi", None)
        if fermi is not None and math.isfinite(float(fermi)):
            result["fermi_energy_ev"] = float(fermi)
        if task == "scf" and result["fermi_energy_ev"] is None:
            raise ValueError("SCF Fermi energy is unavailable from the complete XML.")
        result["converged_electronic"] = bool(run.converged_electronic)
        result["converged_ionic"] = bool(run.converged_ionic)
        if task == "relax":
            result["ionic_steps_count"] = len(run.ionic_steps)
            step_limit = int(run.incar.get("NSW", 0))
            result["ionic_iteration_limit_reached"] = step_limit > 0 and len(run.ionic_steps) >= step_limit
            ediffg = float(run.incar.get("EDIFFG", 0))
            if ediffg < 0:
                result["ionic_force_limit_ev_angstrom"] = abs(ediffg)
        validate_method_output(run.incar, metadata, vasp_version=str(run.vasp_version))
        if metadata.get("method"):
            result.update(method=metadata["method"], method_fingerprint=metadata["method_fingerprint"], method_comparison_fingerprint=metadata["method_comparison_fingerprint"])
        if task in {"bands", "dos"} and not hybrid and int(run.incar.get("ICHARG", 0)) != 11:
            raise ValueError("Bands/DOS output did not use the required SCF charge density (ICHARG=11).")
        if hybrid and int(run.incar.get("ICHARG", 0)) >= 10:
            raise ValueError("Hybrid output used a fixed charge density.")
        if hybrid and task == "bands" and len(run.ionic_steps[-1].get("electronic_steps", [])) < HYBRID_BAND_NELMIN:
            raise ValueError("Hybrid bands ended before the minimum electronic iterations.")
        if task == "scf" and int(run.incar.get("ICHARG", 0)) >= 10:
            raise ValueError("SCF output used a fixed charge density instead of self-consistency.")
        if task == "relax" and (int(run.incar.get("NSW", 0)) <= 0 or int(run.incar.get("IBRION", -1)) != 2):
            raise ValueError("Relaxation output does not match the requested ionic relaxation recipe.")
        if task != "relax" and int(run.incar.get("NSW", 0)) != 0:
            raise ValueError("Static task output has NSW != 0.")
        _validate_structure(run.final_structure)
        if not _same_structure(run.initial_structure, expected):
            raise ValueError("XML initial structure differs from the expected input structure.")
        if len(run.final_structure) != len(expected) or run.final_structure.composition != expected.composition:
            raise ValueError("Final structure atom count or composition differs from the input.")
        if task != "relax" and not _same_structure(run.final_structure, expected):
            raise ValueError("A static task unexpectedly changed the crystal structure.")
        result["valid_structure"] = True
        energy = float(run.final_energy)
        if not math.isfinite(energy):
            raise ValueError("Final energy is nonfinite or unavailable.")
        result["final_energy_ev"] = energy
        result["final_energy_ev_per_atom"] = energy / len(expected)
        forces = np.asarray(run.ionic_steps[-1].get("forces", []), dtype=float)
        if forces.size:
            if forces.shape != (len(expected), 3):
                raise ValueError("Final atomic forces are nonfinite or inconsistent with the structure.")
            if not np.isfinite(forces).all():
                if hybrid and task == "bands":
                    result["force_warning"] = "Forces are unavailable for this zero-weight hybrid band calculation; do not use it for force or stress analysis."
                else:
                    raise ValueError("Final atomic forces are nonfinite or inconsistent with the structure.")
            else:
                result["final_max_force_ev_angstrom"] = float(np.max(np.linalg.norm(forces, axis=1)))
                result["forces_available"] = True
        elif task == "relax":
            raise ValueError("Final atomic forces are missing for a relaxation.")
        if not result["converged_electronic"]:
            raise ValueError("Electronic convergence was not reached.")
        if task == "relax":
            if not result["converged_ionic"]:
                raise ValueError("Ionic convergence was not reached.")
            force_limit = float(run.incar.get("EDIFFG", 0))
            if force_limit >= 0 or result["final_max_force_ev_angstrom"] > abs(force_limit):
                result["converged_ionic"] = False
                raise ValueError("Final atomic force exceeds the requested negative EDIFFG criterion.")
        if hybrid and task == "bands":
            check = _hybrid_kpoint_consistency(run)
            result["hybrid_kpoint_consistency"] = check
            if not check["passed"]:
                if not check["equivalent_pair_count"]:
                    raise ValueError("Hybrid bands have no equivalent k-points for the orbital consistency check.")
                raise ValueError(
                    f"Hybrid band orbitals disagree at equivalent k-points: {check['max_energy_deviation_ev']:.4f} eV "
                    f"exceeds {HYBRID_KPOINT_TOLERANCE_EV:.2f} eV. Increase orbital iterations or change the electronic optimizer and rerun."
                )
        if metadata.get("warm_start"):
            from .restart import validate_warm_start
            result["warm_start"] = validate_warm_start(output, run, metadata)
        artifacts = _export_plots(run, output, task)
        artifacts.extend(_export_tables(run, output))
        if task in {"scf", "bands", "dos"}:
            result["band_gap"] = _band_gap(run, task, metadata)
        if task in {"bands", "dos"}:
            window = json.loads((output / "plot_metadata.json").read_text()).get("energy_window_ev") if task == "bands" else None
            result["plot_settings"] = {"energy_window_ev": window or list(PLOT_ENERGY_WINDOW_EV), "csv_contains_full_data": True}
        if task == "dos":
            result["energy_reference_ev"] = float(run.efermi)
            result["energy_reference_source"] = "dos_run_fermi"
        if task == "bands":
            result["energy_reference_ev"] = float(run.efermi if hybrid else metadata["scf_fermi_energy_ev"])
            result["energy_reference_source"] = "hybrid_scf_mesh" if hybrid else "preceding_scf"
            if hybrid:
                result["band_path_convergence_independently_validated"] = False
        result["magnetization"] = _magnetization(output, len(expected), metadata)
        result["magnetic_ground_state_validated"] = False
        if result["magnetization"]["available"] or result["magnetization"]["total_moment"] is not None:
            _write_json(output / "magnetization.json", result["magnetization"])
            artifacts.append("magnetization.json")
        run.final_structure.to(filename=str(output / "final_structure.cif"))
        # CIF stores cell lengths/angles, losing the Cartesian frame needed by SOC.
        Poscar(run.final_structure).write_file(output / "final_structure.vasp")
        result["final_structure_poscar_sha256"] = _sha256(output / "final_structure.vasp")
        result["artifacts"] = ["final_structure.cif", "final_structure.vasp", *artifacts]
        result["success"] = True
        result["reason"] = "The calculation finished and its output passed the checks for this task."
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(output / "result.json", result)
    return result
