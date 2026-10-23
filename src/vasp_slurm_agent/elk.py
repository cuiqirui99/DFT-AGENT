"""Elk all-electron (FP-LAPW) inputs and evidence-based result checks.

Elk works in Hartree atomic units; public results use eV and Å. Species files
come from the pinned Elk 10.2.4 release (elk_species.json). Reference:
https://elk.sourceforge.io/ and the Elk manual's input-block descriptions.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Structure

from .numerical_defaults import BAND_PATH_MIN_SEGMENT_POINTS
from .engine_common import (BOHR_ANGSTROM, HA_EV, HA_PER_BOHR_EV_PER_ANGSTROM, COMMON_KEYS, TASKS as SHARED_TASKS, _csv,
                            _hash, _integer, _json, _number, band_gap, band_path, check_parameters,
                            finish_energies, hubbard_terms, kpoint_mesh, load_structure, metadata_base,
                            new_result, same_structure, site_moments, smearing, species_order,
                            spin_settings, structure_from_native, band_reference, plot_bands, write_dos, write_final_structure)
from .spectra import plot_dielectric, plot_traces

ENGINE = "elk"
PROGRAM = "elk"
INPUT_FILE = "elk.in"
OUTPUT_FILE = "elk.log"
METADATA_FILE = "elk-input.json"
RESULT_FILE = "elk-result.json"
LIBRARY_DIR = "library"
# optics: linear response, the dielectric function without local fields (task 121) and, with a kernel,
# linear-response TDDFT with local fields (task 320). tddft: real-time evolution under a laser pulse (450, 460).
TASKS = (*SHARED_TASKS, "optics", "tddft")
OPTICS_KEYS = {"optics_mesh", "optics_max_ev", "optics_points", "optics_broadening_ev", "optics_components",
               "scissor_ev", "tddft_kernel", "response_cutoff"}
PULSE_KEYS = {"pulse_photon_ev", "pulse_field_v_per_angstrom", "pulse_polarization", "pulse_fwhm_fs", "pulse_peak_fs",
              "simulation_time_fs", "time_step_as"}
PARAMETER_KEYS = COMMON_KEYS | {"rgkmax", "gmaxvr", "nempty"} | OPTICS_KEYS | PULSE_KEYS
# Settings of a later stage alone; they do not call for a separate SCF.
STAGE_KEYS = OPTICS_KEYS | PULSE_KEYS
FUNCTIONALS = {"PBE": 20, "PBEsol": 22, "LDA": 3}
# Optics and time evolution first rebuild the ground-state orbitals from STATE.OUT on their own mesh (task 1).
ELK_TASKS = {"scf": ("0",), "relax": ("2",), "bands": ("20",), "dos": ("1", "10"), "optics": ("1", "120", "121"),
             "tddft": ("1", "450", "460")}
NATIVE_TASKS = frozenset({320, *(int(number) for numbers in ELK_TASKS.values() for number in numbers)})
# Elk's fxctype for each exchange-correlation kernel of the response (manual, block fxctype).
KERNELS = {"rpa": 0, "alda": 3, "bootstrap": 210}
CHECKPOINTS = {"scf": ("STATE.OUT", "EFERMI.OUT")}
RESTART_MAPPING = {"bands": {"STATE.OUT": "STATE.OUT", "EFERMI.OUT": "EFERMI.OUT"},
                   "dos": {"STATE.OUT": "STATE.OUT"}, "optics": {"STATE.OUT": "STATE.OUT"},
                   "tddft": {"STATE.OUT": "STATE.OUT"}}
AXES = "xyz"
FS_PER_AU = 0.02418884326585747
LIGHT_AU = 137.035999084
V_PER_ANGSTROM_PER_AU = 51.42206747632590
MOMENT_PANEL = r"$\Delta M / M_0$ (%)"
# Elk's own launcher (elk.sh) raises both stacks; large OpenMP work arrays need them.
# Elk parallelizes with OpenMP itself and keeps MKL single-threaded; a threaded
# OpenBLAS (conda-forge links the pthreads build) would start its own threads inside
# every Elk thread, which made a silicon SCF five times slower on 8 cores.
JOB_ENVIRONMENT = ("export OMP_PROC_BIND=false", "export OMP_STACKSIZE=512M", "export OPENBLAS_NUM_THREADS=1",
                   'ulimit -Ss 524288 2>/dev/null || ulimit -Ss "$(ulimit -Hs)" 2>/dev/null || true')
OUTPUT_FILES = ("elk.log", "INFO.OUT", "TOTENERGY.OUT", "EFERMI.OUT", "EIGVAL.OUT", "GEOMETRY.OUT",
                "GEOMETRY_OPT.OUT", "FORCES_OPT.OUT", "STRESS_OPT.OUT", "BAND.OUT", "BANDLINES.OUT", "TDOS.OUT",
                *(f"EPSILON_{i}{i}.OUT" for i in (1, 2, 3)), *(f"EPSILON_TDDFT_{i}{i}.OUT" for i in (1, 2, 3)),
                "EPSILON_TDDFT.OUT", "AFIELDT.OUT", "TD_INFO.OUT", "TOTENERGY_TD.OUT", "JTOT_TD.OUT",
                "MOMENT_TD.OUT", "MOMENTM_TD.OUT")
GENERATED = ("final.cif", "CONTCAR", "bands.csv", "dos.csv", "optics.csv", "tddft.csv",
             *(f"{stem}.{ext}" for stem in ("bands", "dos", "optics", "tddft") for ext in ("png", "pdf", "svg")))
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


@lru_cache(maxsize=1)
def catalog():
    return json.loads(files("vasp_slurm_agent").joinpath("elk_species.json").read_text(encoding="utf-8"))


def species_files(elements):
    table = catalog()["species"]
    missing = [element for element in elements if element not in table]
    if missing:
        raise ValueError(f"Elk {catalog()['version']} has no species file for {', '.join(missing)}.")
    return {element: {**table[element], "element": element} for element in elements}


def checkpoint_names(task, metadata):
    return CHECKPOINTS.get(task, ())


def restart_mapping(task, metadata):
    return dict(RESTART_MAPPING.get(task, {}))

def _f(value):
    return f"{value:.14g}"


def _vertices(path):
    """Seekpath segments as Elk plot1d vertices; bridges join separate segments."""
    points, labels = path["reciprocal_fractional"], path["labels"]
    vertices, names, bridges = [], [], []
    for start, stop in path["segments"]:
        first, last = points[start], points[stop - 1]
        if vertices and np.allclose(vertices[-1], first, atol=1e-8):
            pass
        else:
            if vertices:
                bridges.append(len(vertices) - 1)
            vertices.append(first)
            names.append(labels[start])
        vertices.append(last)
        names.append(labels[stop - 1])
    return vertices, names, bridges


def _plot_points(structure, path, vertices, bridges):
    """Elk spreads plot1d points over all vertex segments by length, bridges included, so ask for
    enough that the shortest real segment still gets the minimum."""
    cartesian = structure.lattice.reciprocal_lattice.get_cartesian_coords(np.asarray(vertices, dtype=float))
    lengths = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    real = [length for index, length in enumerate(lengths) if index not in bridges and length > 1e-9]
    needed = math.ceil(BAND_PATH_MIN_SEGMENT_POINTS * float(lengths.sum()) / min(real)) if real else 0
    return max(len(path["reciprocal_fractional"]), 2 * len(vertices), needed)


def _optics_blocks(structure, p, mesh, blocks, tasks, metadata):
    """Dense mesh, frequency grid, broadening and tensor components for tasks 120/121, and 320 with a kernel."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    dense = kpoint_mesh(structure, {"mesh": p.get("optics_mesh", [min(64, 2 * n) if n > 1 else 1 for n in mesh])}, "mesh")
    maximum = _number(p.get("optics_max_ev", 15.0), "optics_max_ev", True)
    points = _integer(p.get("optics_points", 500), "optics_points", 50)
    broadening = _number(p.get("optics_broadening_ev", 0.1), "optics_broadening_ev", True)
    scissor = _number(p.get("scissor_ev", 0.0), "scissor_ev")
    if maximum > 100 or broadening > 2 or not 0 <= scissor <= 20:
        raise ValueError("Use optics_max_ev up to 100, optics_broadening_ev up to 2 and scissor_ev from 0 to 20 eV.")
    cubic_nonmagnetic = (p.get("spin", "none") == "none"
                        and SpacegroupAnalyzer(structure, symprec=0.02).get_crystal_system() == "cubic")
    default = ["xx"] if cubic_nonmagnetic else ["xx", "yy", "zz"]
    names = p.get("optics_components", default)
    if not isinstance(names, list) or not names or any(name not in {a + a for a in AXES} for name in names):
        raise ValueError("optics_components lists diagonal components: xx, yy and zz.")
    components = [[AXES.index(name[0]) + 1] * 2 for name in dict.fromkeys(names)]
    blocks += [("ngridk", [" ".join(map(str, dense))]), ("wplot", [f"{points} 100 0", f"0.0 {_f(maximum / HA_EV)}"]),
               ("swidth", [_f(broadening / HA_EV)]), ("optcomp", [f"{i} {j}" for i, j in components])]
    if scissor:
        blocks.append(("scissor", [_f(scissor / HA_EV)]))
    kernel = p.get("tddft_kernel")
    if kernel is not None:
        if kernel not in KERNELS:
            raise ValueError("tddft_kernel is rpa, alda or bootstrap.")
        # Local fields up to this |G| in Bohr⁻¹; Elk's own default is 3.
        cutoff = _number(p.get("response_cutoff", 3.0), "response_cutoff")
        if not 0 <= cutoff <= 10:
            raise ValueError("response_cutoff is between 0 and 10 Bohr⁻¹.")
        tasks.append("320")
        blocks += [("fxctype", [str(KERNELS[kernel])]), ("gmaxrf", [_f(cutoff)])]
    metadata.update(mesh=mesh, optics_mesh=dense, optics_components=components, optics_max_ev=maximum,
                    optics_broadening_ev=broadening, scissor_ev=scissor, tddft_kernel=kernel)


def _pulse_blocks(p, spin, mesh, blocks, metadata):
    """A Gaussian laser pulse as Elk's vector potential, with the simulation time and step in atomic units."""
    photon = _number(p.get("pulse_photon_ev", 1.55), "pulse_photon_ev", True)
    # A zero field is the reference run Elk's authors ask for: the moment must stay constant without a pulse.
    field = _number(p.get("pulse_field_v_per_angstrom", 0.5), "pulse_field_v_per_angstrom")
    if field < 0:
        raise ValueError("pulse_field_v_per_angstrom is the peak field; give 0 or more and set the direction with pulse_polarization.")
    fwhm = _number(p.get("pulse_fwhm_fs", 3.0), "pulse_fwhm_fs", True)
    peak = _number(p.get("pulse_peak_fs", 2 * fwhm), "pulse_peak_fs", True)
    total = _number(p.get("simulation_time_fs", peak + 2 * fwhm), "simulation_time_fs", True)
    # Elk's Ni example keeps magnetic evolution stable with 0.1 atomic units (2.4 as). Elk compares the times
    # it reads back from AFIELDT.OUT (ten significant digits) to 1e-10, so the step is a short decimal in a.u.
    step_au = round(_number(p["time_step_as"], "time_step_as", True) / 1000 / FS_PER_AU, 4) if "time_step_as" in p else (
        0.1 if spin != "none" else 0.2)
    if step_au < 1e-4:
        raise ValueError("time_step_as must be at least 0.0024 as.")
    step = step_au * FS_PER_AU * 1000
    polarization = p.get("pulse_polarization", [0.0, 0.0, 1.0])
    if not isinstance(polarization, list) or len(polarization) != 3:
        raise ValueError("pulse_polarization is a nonzero three-component vector.")
    polarization = [_number(x, "pulse_polarization") for x in polarization]
    norm = math.sqrt(sum(x * x for x in polarization))
    if norm == 0:
        raise ValueError("pulse_polarization is a nonzero three-component vector.")
    steps = total * 1000 / step
    if not 0.05 <= photon <= 50 or field > 20 or steps > 200000 or total > 500:
        raise ValueError("Use a photon energy of 0.05-50 eV, a field up to 20 V/Å and at most 500 fs in 200000 steps.")
    omega = photon / HA_EV
    # A0 = c E0 / ω for E(t) = -(1/c) dA/dt in atomic units.
    amplitude = LIGHT_AU * field / V_PER_ANGSTROM_PER_AU / omega
    vector = [amplitude * x / norm for x in polarization]
    blocks += [("ngridk", [" ".join(map(str, mesh))]), ("tstime", [_f(total / FS_PER_AU)]),
               ("dtimes", [_f(step_au)]),
               ("pulse", ["1", " ".join(_f(x) for x in vector) + f" {_f(omega)} 0.0 0.0 {_f(peak / FS_PER_AU)} {_f(fwhm / FS_PER_AU)}"])]
    metadata.update(mesh=mesh, pulse={"photon_ev": photon, "field_v_per_angstrom": field, "fwhm_fs": fwhm, "peak_fs": peak,
                                      "polarization": [x / norm for x in polarization], "vector_potential_au": vector},
                    simulation_time_fs=total, time_step_as=step)


def prepare_inputs(structure_path, destination, task, parameters, library=None):
    """Write one Elk stage; the workflow supplies STATE.OUT for bands and DOS."""
    if task not in TASKS:
        raise ValueError(f"Unsupported Elk task: {task}")
    p = check_parameters(parameters, PARAMETER_KEYS, "Elk")
    structure = load_structure(structure_path)
    functional = p.get("functional", "PBE")
    if functional not in FUNCTIONALS:
        raise ValueError("Elk here supports PBE, PBEsol and LDA. Hybrid functionals are not available.")
    spin = spin_settings(p)
    moments = site_moments(structure, p, spin)
    elements = species_order(structure)
    species = species_files(elements)
    terms = hubbard_terms(structure, p)
    if terms and spin == "none":
        raise ValueError("Elk DFT+U here is used with spin polarization. Set spin to collinear or noncollinear.")
    kind, width = smearing(p, 0.027)
    rgkmax = _number(p.get("rgkmax", 7.0), "rgkmax", True)
    gmaxvr = _number(p.get("gmaxvr", 12.0), "gmaxvr", True)
    if not 4 <= rgkmax <= 12 or not 6 <= gmaxvr <= 30:
        raise ValueError("Use rgkmax between 4 and 12 and gmaxvr between 6 and 30.")
    extra = _integer(p["extra_bands"], "extra_bands", 0) if "extra_bands" in p else None
    # Elk counts empty first-variational states per atom and spin, not per cell.
    # https://elk.sourceforge.io/elk.pdf, input variable nempty.
    default_empty = extra / len(structure) if extra is not None else (8 if task in {"bands", "dos", "optics", "tddft"} or spin != "none" else 4)
    nempty = _number(p.get("nempty", default_empty), "nempty")
    if nempty < 0:
        raise ValueError("nempty must be non-negative.")
    if extra is not None and not math.isclose(nempty * len(structure), extra, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("extra_bands is per cell and spin; nempty is per atom and spin. Give consistent values or use one key.")
    mesh = kpoint_mesh(structure, p)
    metadata = metadata_base(ENGINE, task, structure, structure_path, p)
    tasks = list(ELK_TASKS[task])
    blocks = [("tasks", tasks), ("xctype", [str(FUNCTIONALS[functional])]), ("sppath", ["'./library/'"]),
              ("tshift", [".false."]), ("scale", ["1.0"]),
              ("avec", [" ".join(_f(x / BOHR_ANGSTROM) for x in row) for row in structure.lattice.matrix]),
              ("rgkmax", [_f(rgkmax)]), ("gmaxvr", [_f(gmaxvr)]), ("nempty", [str(nempty)]),
              ("maxscl", [str(_integer(p.get("max_scf_steps", 200), "max_scf_steps"))]),
              ("epsengy", [_f(_number(p.get("energy_tolerance_ev", 1e-5), "energy_tolerance_ev", True) / HA_EV)])]
    if "Li" in elements:
        # Elk's lithium species keeps 1s in the valence. On the default coarse radial mesh (lradstp 4)
        # its density loses about 0.2% of the charge and PBE self-consistency stalls; Elk's own LiF
        # examples use lradstp 2.
        blocks.append(("lradstp", ["2"]))
    if kind != "auto" or "smearing_ev" in p:
        blocks += [("stype", ["1" if kind == "metal" else "3"]), ("swidth", [_f(width / HA_EV)])]
    if spin != "none" or p.get("soc"):
        blocks += [("spinpol", [".true."]), ("reducebf", ["0.5"])]
    if p.get("soc"):
        blocks.append(("spinorb", [".true."]))
    atoms = [str(len(elements))]
    for element in elements:
        sites = [index for index, site in enumerate(structure) if site.specie.symbol == element]
        atoms += [f"'{species[element]['filename']}'", str(len(sites))]
        for index in sites:
            field = [0.0, 0.0, 0.0]
            if moments is not None:
                value = moments[index]
                vector = value if isinstance(value, list) else [0.0, 0.0, value]
                norm = math.sqrt(sum(x * x for x in vector))
                # A negative local field favours a positive moment in Elk.
                field = [-0.01 * x / norm for x in vector] if norm > 1e-8 else field
            atoms.append(" ".join(_f(x) for x in structure[index].frac_coords) + "   " + " ".join(_f(x) for x in field))
    blocks.append(("atoms", atoms))
    if terms:
        rows = ["1 1"]
        for term in terms:
            rows.append(f"{elements.index(term['element']) + 1} {term['l']} {_f(term['u_ev'] / HA_EV)} {_f(term['j_ev'] / HA_EV)}")
        blocks.append(("dft+u", rows))
    if task == "relax":
        force = _number(p.get("force_tolerance_ev_angstrom", 0.01), "force_tolerance_ev_angstrom", True)
        steps = _integer(p.get("max_ionic_steps", 50), "max_ionic_steps")
        blocks += [("epsforce", [_f(force / HA_PER_BOHR_EV_PER_ANGSTROM)]), ("maxatpstp", [str(steps)])]
        if p.get("cell_relax"):
            blocks += [("latvopt", ["1"]), ("maxlatvstp", [str(min(steps, 60))])]
    if task in {"scf", "relax"}:
        blocks.append(("ngridk", [" ".join(map(str, mesh))]))
        metadata["mesh"] = mesh
    elif task == "bands":
        path = band_path(structure, p)
        vertices, names, bridges = _vertices(path)
        blocks += [("ngridk", [" ".join(map(str, mesh))]),
                   ("plot1d", [f"{len(vertices)} {_plot_points(structure, path, vertices, bridges)}"]
                    + [" ".join(_f(x) for x in vertex) for vertex in vertices])]
        metadata.update(band_path=path, elk_vertices={"labels": names, "bridges": bridges}, mesh=mesh)
    elif task == "optics":
        # Elk broadens the dielectric function with swidth, which also smears the occupations of this restart.
        # Fermi-Dirac tails at that width half-filled the bands next to a small gap (an Re ε(0) spike in
        # silicon); Gaussian tails leave them empty.
        blocks = [block for block in blocks if block[0] not in {"stype", "swidth"}]
        blocks.append(("stype", ["1" if kind == "metal" else "0"]))
        _optics_blocks(structure, p, mesh, blocks, tasks, metadata)
    elif task == "tddft":
        _pulse_blocks(p, spin, mesh, blocks, metadata)
    else:
        dense = kpoint_mesh(structure, {"mesh": p.get("dos_mesh", [min(64, 2 * n) if n > 1 else 1 for n in mesh])}, "mesh")
        blocks.append(("ngridk", [" ".join(map(str, dense))]))
        metadata.update(dos_mesh=dense, mesh=mesh)
    text = "".join(f"{name}\n" + "".join(f"  {line}\n" for line in lines) + "\n" for name, lines in blocks)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / INPUT_FILE).write_text(text, encoding="utf-8")
    structure.to(filename=str(destination / "structure.cif"))
    metadata.update(input_file=INPUT_FILE, output_file=OUTPUT_FILE, stdout_file=OUTPUT_FILE, arguments=[],
                    input_files=[INPUT_FILE, "structure.cif", METADATA_FILE],
                    output_files=list(OUTPUT_FILES), library_dir=LIBRARY_DIR, species=species,
                    library_files={entry["filename"]: {"element": element, "sha256": entry["sha256"], "size": entry["size"],
                                                       "kind": "elk_species"} for element, entry in species.items()},
                    functional=functional, xctype=FUNCTIONALS[functional], spin=spin, soc=bool(p.get("soc")),
                    starting_moments=moments, hubbard=terms, rgkmax=rgkmax, gmaxvr=gmaxvr, nempty=nempty, notes=[])
    _json(destination / METADATA_FILE, metadata)
    return metadata


# --- outputs -------------------------------------------------------------------

def _blocks(text):
    """Elk geometry blocks keyed by name: avec rows and per-species atom rows."""
    lines = [line.split("!")[0].rstrip() for line in text.splitlines()]
    avec, species, index = None, [], 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "avec":
            avec = [[float(x) for x in _NUMBER.findall(lines[index + k])[:3]] for k in (1, 2, 3)]
            index += 4
            continue
        if line == "atoms":
            count = int(lines[index + 1].split()[0])
            cursor = index + 2
            for _ in range(count):
                name = re.search(r"'([A-Za-z]+)\.in'", lines[cursor])[1]
                atoms = int(lines[cursor + 1].split()[0])
                rows = [[float(x) for x in _NUMBER.findall(lines[cursor + 2 + k])[:3]] for k in range(atoms)]
                species.append((name, rows))
                cursor += 2 + atoms
            index = cursor
            continue
        index += 1
    if avec is None or not species:
        raise ValueError("Elk geometry is incomplete.")
    return avec, species


def _geometry(text, elements):
    avec, species = _blocks(text)
    order = [element for element, _ in species]
    if sorted(order) != sorted(set(elements)):
        raise ValueError("Elk geometry species differ from the prepared input.")
    lattice = np.asarray(avec) * BOHR_ANGSTROM
    return lattice, species


def _final_structure(text, expected):
    """Map Elk's per-species geometry back to the prepared site order."""
    lattice, species = _geometry(text, [site.specie.symbol for site in expected])
    rows = {element: list(positions) for element, positions in species}
    coords = []
    for site in expected:
        bucket = rows[site.specie.symbol]
        if not bucket:
            raise ValueError("Elk geometry lists fewer atoms than the prepared input.")
        coords.append(bucket.pop(0))
    if any(rows.values()):
        raise ValueError("Elk geometry lists more atoms than the prepared input.")
    return Structure(lattice, [site.specie.symbol for site in expected], coords)


def _last_geometry_block(text):
    marker = max(text.rfind("! Atomic position optimisation step"), text.rfind("! Lattice optimisation step"))
    if marker < 0:
        raise ValueError("Elk did not record an optimised geometry.")
    return text[marker:]


def _eigenvalues(path):
    """(1, k, state) eigenvalues in eV and occupations from EIGVAL.OUT."""
    text = Path(path).read_text(errors="replace", encoding="utf-8")
    count = int(re.search(r"(\d+)\s*:\s*nkpt", text)[1])
    states = int(re.search(r"(\d+)\s*:\s*nstsv", text)[1])
    energies, occupations = [], []
    for block in re.split(r"\n\s*\d+\s+" + r"\s+".join([_NUMBER.pattern] * 3) + r"\s*:\s*k-point, vkl", text)[1:]:
        rows = [line.split() for line in block.splitlines() if re.match(r"^\s*\d+\s+" + _NUMBER.pattern, line)]
        rows = rows[:states]
        if len(rows) != states:
            raise ValueError("Elk eigenvalue table is incomplete.")
        energies.append([float(row[1]) * HA_EV for row in rows])
        occupations.append([float(row[2]) for row in rows])
    if len(energies) != count:
        raise ValueError("Elk eigenvalue table is missing k points.")
    return np.asarray([energies]), np.asarray([occupations])


def _occupied_gap(energies, occupations, scope, *, full_occupation=2, fermi=None):
    # Elk's Methfessel-Paxton integration weights can be negative or exceed full occupancy.
    return band_gap(energies, occupations=occupations, full_occupation=full_occupation,
                    occupations_are_weights=True, fermi=fermi, scope=scope)


def _band_table(folder, metadata):
    """BAND.OUT energies (eV, relative to E_F) on the prepared path, bridges removed."""
    rows = [line.split() for line in (folder / "BAND.OUT").read_text(errors="replace", encoding="utf-8").splitlines()]
    bands, current = [], []
    for row in rows:
        if len(row) < 2:
            if current:
                bands.append(current)
                current = []
            continue
        current.append((float(row[0]), float(row[1])))
    if current:
        bands.append(current)
    if not bands or any(len(band) != len(bands[0]) for band in bands):
        raise ValueError("Elk band table is incomplete.")
    data = np.asarray(bands)
    x, energies = data[0, :, 0], data[:, :, 1].T * HA_EV
    lines = [line.split() for line in (folder / "BANDLINES.OUT").read_text(errors="replace", encoding="utf-8").splitlines() if line.split()]
    positions = sorted({float(row[0]) for row in lines})
    vertices = metadata["elk_vertices"]
    if len(positions) != len(vertices["labels"]):
        raise ValueError("Elk band vertices differ from the prepared path.")
    keep = np.ones(len(x), dtype=bool)
    shift = np.zeros(len(x))
    for bridge in vertices["bridges"]:
        start, stop = positions[bridge], positions[bridge + 1]
        keep &= ~((x > start + 1e-9) & (x < stop - 1e-9))
        shift[x >= stop - 1e-9] += stop - start
    distance = (x - shift) / BOHR_ANGSTROM
    kept = np.nonzero(keep)[0]
    starts = [0] + [int(np.searchsorted(kept, np.argmax(x >= positions[bridge + 1] - 1e-9))) for bridge in vertices["bridges"]]
    segments = [[start, stop] for start, stop in zip(starts, starts[1:] + [len(kept)])]
    ticks, labels = [], list(vertices["labels"])
    for bridge in vertices["bridges"]:
        labels[bridge] = f"{labels[bridge]}|{labels[bridge + 1]}"
    for index, label in enumerate(labels):
        if index - 1 in vertices["bridges"]:
            continue
        position = positions[index] - shift[np.argmin(abs(x - positions[index]))]
        ticks.append((position / BOHR_ANGSTROM, label))
    return distance[keep], energies[keep], ticks, segments


def _moments(text, result, structure):
    start = text.rfind("Moments :")
    if start < 0:
        return
    block = text[start:start + 6000]
    end = block.find("total moment")
    if end < 0:
        return
    block = block[:block.find("\n", end) if "\n" in block[end:] else len(block)]
    total = re.search(r"total moment\s*:\s*(" + _NUMBER.pattern + r")(?:\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + "))?", block)
    if total:
        values = [float(value) for value in total.groups() if value is not None]
        if len(values) == 3:
            result["total_magnetization_vector_mu_b"] = values
            result["total_magnetization_mu_b"] = float(np.linalg.norm(values))
        else:
            result["total_magnetization_mu_b"] = values[0]
    atoms = re.findall(r"atom\s+(\d+)\s*:\s*(" + _NUMBER.pattern + ")", block)
    elements = species_order(structure)
    ordered = [index for element in elements for index, site in enumerate(structure) if site.specie.symbol == element]
    if len(atoms) == len(structure):
        sites = []
        for position, (_, value) in enumerate(atoms):
            index = ordered[position]
            sites.append({"atom": index + 1, "element": structure[index].specie.symbol, "moment_mu_b": float(value)})
        result["site_magnetization"] = sorted(sites, key=lambda item: item["atom"])
        result["absolute_magnetization_mu_b"] = float(sum(abs(item["moment_mu_b"]) for item in sites))
        result["site_magnetization_scope"] = "Spin moment inside each Elk muffin-tin sphere."


def _tables(path):
    """Rows of numbers in an Elk table, split at blank lines."""
    blocks, rows = [], []
    for line in Path(path).read_text(errors="replace", encoding="utf-8").splitlines():
        numbers = [float(x.replace("D", "E").replace("d", "e")) for x in _NUMBER.findall(line)]
        if numbers:
            rows.append(numbers)
        elif rows:
            blocks.append(rows)
            rows = []
    if rows:
        blocks.append(rows)
    return blocks


def _epsilon(path):
    """Photon energy (eV), real and imaginary parts from an EPSILON file: a real block, then an imaginary block."""
    blocks = _tables(path)
    if len(blocks) != 2 or len(blocks[0]) != len(blocks[1]) or len(blocks[0]) < 10 or any(len(row) < 2 for block in blocks for row in block):
        raise ValueError(f"{Path(path).name} is incomplete.")
    real, imaginary = np.asarray(blocks[0])[:, :2], np.asarray(blocks[1])[:, :2]
    if not np.allclose(real[:, 0], imaginary[:, 0]) or not np.isfinite([real, imaginary]).all():
        raise ValueError(f"{Path(path).name} is inconsistent.")
    return real[:, 0] * HA_EV, real[:, 1], imaginary[:, 1]


def _optics(folder, metadata, result):
    curves, summary, energy = {}, {}, None
    kernel = metadata.get("tddft_kernel")
    for i, j in metadata["optics_components"]:
        name = AXES[i - 1] + AXES[j - 1]
        files = [(f"EPSILON_{i}{j}.OUT", name)]
        if kernel:
            tddft = f"EPSILON_TDDFT_{i}{j}.OUT"
            files.append((tddft if (folder / tddft).is_file() or i != 1 else "EPSILON_TDDFT.OUT", f"{name}, TDDFT {kernel}"))
        for filename, label in files:
            values = _epsilon(folder / filename)
            if energy is None:
                energy = values[0]
            # Elk's TDDFT spectra (task 320) leave out ω = 0; each spectrum is placed on the first grid.
            positions = np.searchsorted(energy, values[0] - 1e-6)
            if (positions >= len(energy)).any() or not np.allclose(energy[np.minimum(positions, len(energy) - 1)], values[0], atol=1e-6):
                raise ValueError("The dielectric spectra use different frequency grids.")
            real, imaginary = np.full(len(energy), np.nan), np.full(len(energy), np.nan)
            real[positions], imaginary[positions] = values[1], values[2]
            curves[label] = (real, imaginary)
            peak = int(np.nanargmax(imaginary))
            entry = {"absorption_peak_ev": float(energy[peak]), "peak_imaginary_epsilon": float(imaginary[peak])}
            # Re ε at zero frequency (or the first frequency, within 0.1 eV) is the electronic dielectric
            # constant only for a gapped crystal.
            first = int(positions[0])
            if result.get("band_gap", {}).get("status") == "gap" and energy[first] <= 0.1:
                entry["dielectric_constant"] = float(real[first])
            summary[label] = entry
            result["artifacts"].append(filename)
    columns, rows = ["energy_ev"], []
    for label in curves:
        key = label.replace(", TDDFT ", "_tddft_").replace(" ", "_")
        columns += [f"re_eps_{key}", f"im_eps_{key}"]
    for index, value in enumerate(energy):
        rows.append([float(value), *("" if np.isnan(part[index]) else float(part[index]) for pair in curves.values() for part in pair)])
    _csv(folder / "optics.csv", columns, rows)
    result["artifacts"].append("optics.csv")
    result["artifacts"] += plot_dielectric(folder, "optics", energy, curves)
    result["optics"] = {"spectra": summary, "kernel": kernel or "none (no local fields)",
                        "broadening_ev": metadata["optics_broadening_ev"], "scissor_ev": metadata["scissor_ev"],
                        "mesh": metadata["optics_mesh"]}


def _time_evolution(folder, stdout, metadata, result):
    def table(name, width):
        rows = [row[:width] for block in _tables(folder / name) for row in block if len(row) >= width]
        if len(rows) < 2:
            raise ValueError(f"{name} is incomplete.")
        result["artifacts"].append(name)
        return np.asarray(rows)

    afield = table("AFIELDT.OUT", 5)
    energy = table("TOTENERGY_TD.OUT", 2)
    current = table("JTOT_TD.OUT", 4)
    final = afield[-2, 1] if len(afield) > 1 else afield[-1, 1]
    steps = re.findall(r"time step\s+(\d+)\s+of\s+(\d+)", stdout)
    if not steps or steps[-1][0] != steps[-1][1] or energy[-1, 0] < final - 1e-6 * max(1.0, final):
        raise ValueError("The time evolution stopped before the end of the simulation.")
    if len(energy) != len(current) or not np.allclose(energy[:, 0], current[:, 0]):
        raise ValueError("The time-evolution tables disagree.")
    time = energy[:, 0]
    times = afield[:, 1]
    potential = np.stack([np.interp(time, times, afield[:, 2 + axis]) for axis in range(3)], axis=1)
    # E(t) = -(1/c) dA/dt in atomic units.
    efield = -np.gradient(potential, time, axis=0) / LIGHT_AU * V_PER_ANGSTROM_PER_AU
    time_fs = time * FS_PER_AU
    panels = {"E (V/Å)": [(f"E_{AXES[a]}", efield[:, a]) for a in range(3) if np.abs(efield[:, a]).max() > 1e-12] or [("E", efield[:, 2])],
              "j (a.u.)": [(f"j_{AXES[a]}", current[:, 1 + a]) for a in range(3)
                           if np.abs(current[:, 1 + a]).max() > 1e-12 * max(1.0, np.abs(current[:, 1:]).max())] or [("j", current[:, 3])],
              "ΔE (eV)": [("total energy", (energy[:, 1] - energy[0, 1]) * HA_EV)]}
    columns = ["time_fs", "e_x_v_per_angstrom", "e_y_v_per_angstrom", "e_z_v_per_angstrom", "j_x", "j_y", "j_z", "energy_change_ev"]
    data = np.column_stack([time_fs, efield, current[:, 1:4], (energy[:, 1] - energy[0, 1]) * HA_EV])
    summary = {"simulation_time_fs": float(time_fs[-1]), "time_steps": len(time), "absorbed_energy_ev": float((energy[-1, 1] - energy[0, 1]) * HA_EV),
               "peak_field_v_per_angstrom": float(np.linalg.norm(efield, axis=1).max()),
               "peak_current_au": float(np.linalg.norm(current[:, 1:4], axis=1).max())}
    if metadata.get("spin") != "none" or metadata.get("soc"):
        moment = table("MOMENTM_TD.OUT", 2)
        if len(moment) != len(time) or not np.allclose(moment[:, 0], time):
            raise ValueError("MOMENTM_TD.OUT disagrees with the other time-evolution tables.")
        if moment[0, 1]:
            panels[MOMENT_PANEL] = [("spin moment", 100 * (moment[:, 1] / moment[0, 1] - 1))]
        columns.append("moment_mu_b")
        data = np.column_stack([data, moment[:, 1]])
        summary.update(initial_moment_mu_b=float(moment[0, 1]), final_moment_mu_b=float(moment[-1, 1]),
                       moment_change_percent=float(100 * (moment[-1, 1] - moment[0, 1]) / moment[0, 1]) if moment[0, 1] else None)
    _csv(folder / "tddft.csv", columns, data.tolist())
    result["artifacts"] += ["TD_INFO.OUT", "tddft.csv"] if (folder / "TD_INFO.OUT").is_file() else ["tddft.csv"]
    result["artifacts"] += plot_traces(folder, "tddft", time_fs, panels, spans={MOMENT_PANEL: 1.0})
    result["time_evolution"] = {**summary, "pulse": metadata.get("pulse")}


def analyze_outputs(folder, task, expected_structure=None):
    """Require normal completion plus task-specific convergence and numerical evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in GENERATED:
        (folder / name).unlink(missing_ok=True)
    result = new_result(ENGINE, task)
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported Elk task: {task}")
        metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("canonical_task") != task or metadata.get("engine") != ENGINE:
            raise ValueError("Stage metadata does not match the requested analysis task.")
        log = folder / OUTPUT_FILE
        if not log.is_file():
            raise ValueError(f"{OUTPUT_FILE} is missing.")
        stdout = log.read_text(encoding="utf-8", errors="replace")
        result["stdout_sha256"] = _hash(log)
        result["artifacts"].append(OUTPUT_FILE)
        if "Elk code stopped" not in stdout:
            raise ValueError("Elk did not finish normally.")
        if re.search(r"^\s*Error\(", stdout, re.M):
            raise ValueError("Elk reported an error.")
        expected = structure_from_native(metadata["native_structure"])
        info_path = folder / "INFO.OUT"
        info = info_path.read_text(errors="replace", encoding="utf-8") if info_path.is_file() else ""
        # Before time zero, task 460 reruns one ground-state iteration under a large static A-field
        # along each axis, which rewrites INFO.OUT, EFERMI.OUT and EIGVAL.OUT; only the log still
        # tells whether the restarted ground state (task 1) converged.
        ground_state = task != "tddft"
        if task != "bands":
            if not info:
                raise ValueError("INFO.OUT is missing.")
            converged = "failed to reach self-consistency" not in stdout
            if ground_state:
                converged = (converged and info.rfind("Convergence targets achieved") > info.rfind("Loop number : 1 ")
                             and "Convergence targets achieved" in info)
            result["converged_electronic"] = converged
            if not converged:
                raise ValueError("The self-consistent loop did not reach its targets.")
            if ground_state:
                result["artifacts"].append("INFO.OUT")
        if task == "relax":
            forces = (folder / "FORCES_OPT.OUT").read_text(errors="replace", encoding="utf-8")
            ionic = "Force convergence target achieved" in forces[forces.rfind("optimisation step"):] and "Warning(geomopt)" not in stdout
            if metadata["parameters"].get("cell_relax"):
                stress = (folder / "STRESS_OPT.OUT").read_text(errors="replace", encoding="utf-8")
                ionic = ionic and "Stress convergence target achieved" in stress
            result["converged_ionic"] = ionic
            if not ionic:
                raise ValueError("The geometry optimisation did not reach its force or stress target.")
            structure = _final_structure(_last_geometry_block((folder / "GEOMETRY_OPT.OUT").read_text(errors="replace", encoding="utf-8")), expected)
            if not same_structure(structure, expected, positions=False):
                raise ValueError("Elk optimised a different set of atoms.")
            write_final_structure(folder, result, structure)
        else:
            geometry = folder / "GEOMETRY.OUT"
            if geometry.is_file():
                structure = _final_structure(geometry.read_text(errors="replace", encoding="utf-8"), expected)
                if not same_structure(structure, expected, atol=1e-6):
                    raise ValueError("Elk used a structure that differs from the prepared input.")
            structure = expected
            if task == "scf":
                write_final_structure(folder, result, expected)
        result["spin_mode"] = metadata.get("spin")
        result["spin_orbit_coupling"] = bool(metadata.get("soc"))
        if task in {"scf", "relax"}:
            energies = [float(x) for x in (folder / "TOTENERGY.OUT").read_text(encoding="utf-8").split()]
            finish_energies(result, energies[-1] * HA_EV, len(structure))
            result["artifacts"].append("TOTENERGY.OUT")
        fermi_path = folder / "EFERMI.OUT"
        if ground_state and fermi_path.is_file():
            result["fermi_energy_ev"] = float(fermi_path.read_text(encoding="utf-8").split()[0]) * HA_EV
        if ground_state and info:
            _moments(info, result, structure)
        if task == "bands":
            distance, energies, ticks, segments = _band_table(folder, metadata)
            gap = band_gap(energies[None, :, :], fermi=0.0, scope="sampled_path")
            if result.get("fermi_energy_ev") is not None and gap.get("available"):
                for key in ("vbm_ev", "cbm_ev"):
                    gap[key] += result["fermi_energy_ev"]
            reference, kind = band_reference(gap, result.get("fermi_energy_ev"))
            result.update(band_gap=gap, band_gap_ev=gap.get("gap_ev"), kpoint_count=len(distance), band_count=int(energies.shape[1]),
                          plot_reference_ev=reference, energy_reference=kind)
            _csv(folder / "bands.csv", ["k_index", "distance_inv_angstrom", "band", "energy_minus_fermi_ev"],
                 ([k + 1, float(distance[k]), b + 1, float(energies[k, b])] for k in range(len(distance)) for b in range(energies.shape[1])))
            result["artifacts"] += ["BAND.OUT", "BANDLINES.OUT", "bands.csv"]
            labels = [""] * len(distance)
            for position, label in ticks:
                labels[int(np.argmin(abs(distance - position)))] = label
            # BAND.OUT mixes both spins in second-variational states, so one colour; its energies are relative to E_F.
            result["artifacts"] += plot_bands(folder, "bands", distance, energies - (reference - (result.get("fermi_energy_ev") or 0.0)),
                                              labels, segments, kind)
        elif ground_state:
            energies, occupations = _eigenvalues(folder / "EIGVAL.OUT")
            gap = _occupied_gap(energies, occupations, "sampled_k_points",
                                full_occupation=1 if metadata.get("spin") != "none" or metadata.get("soc") else 2,
                                fermi=result.get("fermi_energy_ev"))
            result.update(band_gap=gap, band_gap_ev=gap.get("gap_ev"), kpoint_count=int(energies.shape[1]), band_count=int(energies.shape[2]))
            result["artifacts"].append("EIGVAL.OUT")
        if task == "optics":
            _optics(folder, metadata, result)
        elif task == "tddft":
            _time_evolution(folder, stdout, metadata, result)
        if task == "dos":
            blocks, rows = [], []
            for line in (folder / "TDOS.OUT").read_text(errors="replace", encoding="utf-8").splitlines():
                numbers = [float(x) for x in _NUMBER.findall(line)]
                if len(numbers) >= 2:
                    rows.append(numbers[:2])
                elif rows:
                    blocks.append(rows)
                    rows = []
            if rows:
                blocks.append(rows)
            if not blocks or any(len(block) != len(blocks[0]) for block in blocks):
                raise ValueError("Elk DOS table is incomplete.")
            data = np.asarray(blocks)
            energy = data[0, :, 0] * HA_EV
            dos = np.stack([abs(data[index, :, 1]) / HA_EV for index in range(len(blocks))], axis=1)
            if dos.shape[1] == 2 and metadata.get("spin") == "none":
                # Spin-orbit coupling needs spinpol in Elk; without a moment its two spin halves are one total.
                dos = dos.sum(axis=1, keepdims=True)
            result["artifacts"].append("TDOS.OUT")
            write_dos(folder, result, energy, dos, 0.0, columns=("dos_up", "dos_down") if dos.shape[1] == 2 else ("dos",))
            result["energy_reference"] = "fermi_level"
        result.update(success=True, reason="Required outputs and convergence checks passed.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result["reason"] = str(exc) or "Elk output is truncated."
    _json(folder / RESULT_FILE, result)
    return result


LIBRARY_VALIDATOR = r'''def validate_library(data, entry):
    if len(data) < 100 or len(data) > 5 * 1024 * 1024:
        raise ValueError("Elk species file is empty or unexpectedly large.")
    lines = data.decode("utf-8", errors="replace").splitlines()
    symbol = lines[0].split("'")[1] if lines[0].count("'") >= 2 else ""
    if symbol != entry["element"]:
        raise ValueError("Elk species file is for a different element.")
    return {"element": symbol}
'''
