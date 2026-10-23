"""CP2K (Quickstep, GPW) inputs and evidence-based result checks.

CP2K reports Hartree and Bohr; public results use eV and Å. Basis sets and
GTH pseudopotentials are named per element in cp2k_basis.json, pinned to the
CP2K 2026.2 data files by SHA-256. Cells are written as lengths and angles so
CP2K keeps its canonical orientation and full k-point symmetry; meshes are
Γ-centred like the other adapters, and inputs work with CP2K 2026.1 and 2026.2.
Reference: https://manual.cp2k.org/.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Lattice, Structure

from .engine_common import (BOHR_ANGSTROM, HA_EV, HA_PER_BOHR_EV_PER_ANGSTROM, KB_EV, COMMON_KEYS, TASKS as SHARED_TASKS,
                            _csv, _hash, _integer, _json, _number, band_gap, band_path,
                            check_parameters, finish_energies, hubbard_terms, kpoint_mesh, load_structure,
                            metadata_base, new_result, site_moments, smearing, species_order, spin_settings,
                            structure_from_native, band_reference, plot_bands, write_dos, write_final_structure)
from .spectra import plot_curves, plot_traces

ENGINE = "cp2k"
PROGRAM = "cp2k.psmp"
PROJECT = "dft_agent"
INPUT_FILE = "cp2k.inp"
OUTPUT_FILE = "cp2k.out"
STDOUT_FILE = "cp2k.log"
METADATA_FILE = "cp2k-input.json"
RESULT_FILE = "cp2k-result.json"
LIBRARY_DIR = "library"
# md: Born-Oppenheimer molecular dynamics, NVT with a CSVR thermostat or NVE.
TASKS = (*SHARED_TASKS, "md")
MD_KEYS = {"md_ensemble", "md_temperature_k", "md_steps", "md_timestep_fs", "md_thermostat_fs"}
PARAMETER_KEYS = COMMON_KEYS | {"cutoff_ry", "rel_cutoff_ry", "basis_family", "eps_scf", "added_mos", "dos_broadening_ev"} | MD_KEYS
STAGE_KEYS = MD_KEYS
FUNCTIONALS = {"PBE": "PBE", "LDA": "PADE"}
WAVEFUNCTION = {True: f"{PROJECT}-RESTART.kp", False: f"{PROJECT}-RESTART.wfn"}
OUTPUT_FILES = (OUTPUT_FILE, STDOUT_FILE, f"{PROJECT}.bs", f"{PROJECT}-1.restart", f"{PROJECT}-pos-1.xyz",
                f"{PROJECT}-1.cell", f"{PROJECT}-1.dos", f"{PROJECT}-ALPHA-1.dos", f"{PROJECT}-BETA-1.dos",
                f"{PROJECT}-1.ener")
GENERATED = ("final.cif", "CONTCAR", "bands.csv", "dos.csv", "md.csv", "rdf.csv",
             *(f"{stem}.{ext}" for stem in ("bands", "dos", "md", "rdf") for ext in ("png", "pdf", "svg")))
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")
GPA_PER_BAR = 1e-4


@lru_cache(maxsize=1)
def catalog():
    return json.loads(files("vasp_slurm_agent").joinpath("cp2k_basis.json").read_text())


def basis_choice(elements, family, functional):
    table = catalog()["families"].get(family)
    if table is None:
        raise ValueError("basis_family must be sr, uzh_dzvp or uzh_tzv2p.")
    missing = [element for element in elements if element not in table["elements"]
               or functional not in table["elements"][element]["potentials"]]
    if missing:
        raise ValueError(f"{table['label']} has no {functional} basis and potential for {', '.join(missing)}.")
    return table, {element: {"basis": table["elements"][element]["basis"],
                             "potential": table["elements"][element]["potentials"][functional],
                             "valence": table["elements"][element]["valence"]} for element in elements}


def checkpoint_names(task, metadata):
    if task != "scf":
        return ()
    return (WAVEFUNCTION[bool(metadata.get("kpoints"))],)


def restart_mapping(task, metadata):
    if task != "bands":
        return {}
    name = WAVEFUNCTION[bool(metadata.get("kpoints"))]
    return {name: name}


def _f(value):
    return f"{value:.14g}"


def _section(name, body, argument=""):
    head = f"&{name} {argument}".rstrip()
    return [head, *("  " + line for line in body), f"&END {name}"]


def _kpoints(mesh):
    """Keep the requested Γ-centred mesh and reduce by time reversal only.

    CP2K 2026.1's point-group path calls K290 with an invalid output unit.
    SYMMETRY OFF bypasses that path; FULL_GRID OFF retains k/-k pairing.
    """
    if all(n % 2 for n in mesh):
        return ["SCHEME MONKHORST-PACK " + " ".join(map(str, mesh)), "SYMMETRY OFF", "FULL_GRID OFF"]
    shifts = [0.5 / n if n % 2 == 0 else 0.0 for n in mesh]
    return ["SCHEME MACDONALD " + " ".join(map(str, mesh)) + " " + " ".join(_f(s) for s in shifts), "SYMMETRY OFF", "FULL_GRID OFF"]


def _band_sets(path):
    """One CP2K KPOINT_SET per seekpath segment, with that segment's point count."""
    points, labels = path["reciprocal_fractional"], path["labels"]
    return [((points[start], labels[start] or "K"), (points[stop - 1], labels[stop - 1] or "K"), max(1, stop - start - 1))
            for start, stop in path["segments"]]


def _md_motion(p, record):
    """MOTION/MD for Born-Oppenheimer dynamics; the trajectory is written at every step."""
    ensemble = str(p.get("md_ensemble", "nvt")).lower()
    if ensemble not in {"nvt", "nve"}:
        raise ValueError("md_ensemble is nvt or nve.")
    temperature = _number(p.get("md_temperature_k", 300.0), "md_temperature_k", True)
    steps = _integer(p.get("md_steps", 1000), "md_steps")
    timestep = _number(p.get("md_timestep_fs", 0.5), "md_timestep_fs", True)
    tau = _number(p.get("md_thermostat_fs", 100.0), "md_thermostat_fs", True)
    if steps > 200000 or not 0.05 <= timestep <= 5 or temperature > 5000:
        raise ValueError("Use at most 200000 steps, a time step of 0.05-5 fs and a temperature up to 5000 K.")
    body = [f"ENSEMBLE {ensemble.upper()}", f"STEPS {steps}", f"TIMESTEP [fs] {_f(timestep)}", f"TEMPERATURE [K] {_f(temperature)}"]
    if ensemble == "nvt":
        body += _section("THERMOSTAT", ["TYPE CSVR", *_section("CSVR", [f"TIMECON [fs] {_f(tau)}"])])
    record.update(ensemble=ensemble, temperature_k=temperature, steps=steps, timestep_fs=timestep,
                  thermostat_fs=tau if ensemble == "nvt" else None)
    printing = [*_section("TRAJECTORY", _section("EACH", ["MD 1"])), *_section("RESTART", _section("EACH", ["MD 100"])),
                *_section("RESTART_HISTORY", [], "OFF")]
    return _section("MOTION", [*_section("MD", body), *_section("PRINT", printing)])


def prepare_inputs(structure_path, destination, task, parameters, library=None):
    """Write one CP2K stage; bands restart from the accepted SCF wavefunction."""
    if task not in TASKS:
        raise ValueError(f"Unsupported CP2K task: {task}")
    p = check_parameters(parameters, PARAMETER_KEYS, "CP2K")
    if "energy_tolerance_ev" in p:
        # CP2K qs_scf_loop_utils: EPS_SCF bounds the density residual p_out-p_in.
        raise ValueError("CP2K uses eps_scf, an SCF residual tolerance, not an energy threshold in eV. "
                         "Replace energy_tolerance_ev with an explicit eps_scf value.")
    structure = load_structure(structure_path)
    functional = p.get("functional", "PBE")
    if functional not in FUNCTIONALS:
        raise ValueError("CP2K here supports PBE and LDA (PADE) with GTH pseudopotentials. Hybrid functionals need extra libraries and are not available.")
    spin = spin_settings(p, noncollinear=False, soc=False)
    moments = site_moments(structure, p, spin)
    elements = species_order(structure)
    family = p.get("basis_family", "sr")
    if functional == "LDA" and family != "sr":
        raise ValueError("LDA uses the DZVP-MOLOPT-SR-GTH family with GTH-PADE potentials.")
    table, kinds = basis_choice(elements, family, functional)
    terms = hubbard_terms(structure, p)
    mesh = kpoint_mesh(structure, p)
    if task == "dos":
        mesh = kpoint_mesh(structure, {"mesh": p.get("dos_mesh", [min(64, 2 * n) if n > 1 else 1 for n in mesh])}, "mesh")
    kpoints = mesh != [1, 1, 1]
    if terms and kpoints:
        raise ValueError("CP2K DFT+U is limited to Γ-point calculations here. Use a supercell with mesh [1, 1, 1], or Elk for DFT+U with k-points.")
    electrons = sum(kinds[site.specie.symbol]["valence"] for site in structure)
    kind, width = smearing(p, 0.026)
    occupied = math.ceil(electrons / 2)
    extra = _integer(p["extra_bands"], "extra_bands", 0) if "extra_bands" in p else None
    added = _integer(p.get("added_mos", extra if extra is not None else max(8, math.ceil(0.3 * occupied))), "added_mos", 0)
    if extra is not None and added != extra:
        raise ValueError("extra_bands and added_mos must specify the same number of extra orbitals per spin.")
    cutoff = _number(p.get("cutoff_ry", 500), "cutoff_ry", True)
    relative = _number(p.get("rel_cutoff_ry", 60), "rel_cutoff_ry", True)
    if not 100 <= cutoff <= 2000 or not 20 <= relative <= 200:
        raise ValueError("Use cutoff_ry between 100 and 2000 and rel_cutoff_ry between 20 and 200.")
    eps_scf = _number(p.get("eps_scf", 1e-7), "eps_scf", True)
    max_scf = _integer(p.get("max_scf_steps", 100), "max_scf_steps")
    # Orbital transformation needs no diagonalization or empty orbitals: CP2K's route for large insulators at Γ.
    ot = not kpoints and kind == "insulator" and task in {"scf", "relax", "md"}
    run_type = "ENERGY_FORCE"
    motion = []
    if task == "relax":
        force = _number(p.get("force_tolerance_ev_angstrom", 0.01), "force_tolerance_ev_angstrom", True)
        steps = _integer(p.get("max_ionic_steps", 100), "max_ionic_steps")
        if p.get("cell_relax"):
            run_type = "CELL_OPT"
            motion = _section("MOTION", _section("CELL_OPT", [
                "OPTIMIZER BFGS", f"MAX_ITER {steps}",
                f"MAX_FORCE {_f(force / HA_PER_BOHR_EV_PER_ANGSTROM)}", "PRESSURE_TOLERANCE [bar] 100"]))
        else:
            run_type = "GEO_OPT"
            motion = _section("MOTION", _section("GEO_OPT", [
                "OPTIMIZER BFGS", f"MAX_ITER {steps}", f"MAX_FORCE {_f(force / HA_PER_BOHR_EV_PER_ANGSTROM)}"]))
    md_settings = {}
    if task == "md":
        run_type = "MD"
        motion = _md_motion(p, md_settings)
    if ot:
        inner = min(50, max_scf)
        scf = ["SCF_GUESS ATOMIC", f"EPS_SCF {_f(eps_scf)}", f"MAX_SCF {inner}",
               *_section("OT", ["MINIMIZER DIIS", "PRECONDITIONER FULL_SINGLE_INVERSE"], "ON"),
               *_section("OUTER_SCF", [f"EPS_SCF {_f(eps_scf)}", f"MAX_SCF {max(1, math.ceil(max_scf / inner))}"], "ON")]
    else:
        scf = ["SCF_GUESS RESTART" if task == "bands" else "SCF_GUESS ATOMIC", f"EPS_SCF {_f(eps_scf)}",
               f"MAX_SCF {max_scf}", f"ADDED_MOS {added}",
               *_section("DIAGONALIZATION", ["ALGORITHM STANDARD"], "ON"),
               *_section("MIXING", ["METHOD BROYDEN_MIXING", "ALPHA 0.2", "NBROYDEN 8"])]
    if kind != "insulator" or kpoints:
        scf += _section("SMEAR", ["METHOD FERMI_DIRAC", f"ELECTRONIC_TEMPERATURE [K] {_f(width / KB_EV)}"], "ON")
    dft = [f"BASIS_SET_FILE_NAME ./{LIBRARY_DIR}/{table['basis_file']}",
           f"POTENTIAL_FILE_NAME ./{LIBRARY_DIR}/{table['potential_file']}",
           *_section("MGRID", [f"CUTOFF {_f(cutoff)}", f"REL_CUTOFF {_f(relative)}"]),
           *_section("QS", ["EPS_DEFAULT 1.0E-12", "METHOD GPW"]), *_section("SCF", scf),
           *_section("XC", _section("XC_FUNCTIONAL", [], FUNCTIONALS[functional]))]
    if task == "bands":
        dft.insert(2, f"WFN_RESTART_FILE_NAME {WAVEFUNCTION[kpoints]}")
    if spin == "collinear":
        # Start from the requested total moment; smearing then lets it relax.
        moment = round(sum(moments))
        if (electrons - moment) % 2:
            moment += 1 if moment + 1 <= electrons else -1
        dft.insert(2, f"MULTIPLICITY {abs(moment) + 1}")
        dft.insert(2, "UKS TRUE")
    if terms:
        dft.insert(2, "PLUS_U_METHOD MULLIKEN")
    if kpoints:
        dft += _section("KPOINTS", _kpoints(mesh))
    prints = []
    metadata = metadata_base(ENGINE, task, structure, structure_path, p)
    if task == "bands":
        path = band_path(structure, p)
        sets = []
        for first, last, count in _band_sets(path):
            body = ["UNITS B_VECTOR", *(f"SPECIAL_POINT {label} " + " ".join(_f(x) for x in point) for point, label in (first, last)),
                    f"NPOINTS {count}"]
            sets += _section("KPOINT_SET", body)
        prints += _section("BAND_STRUCTURE", [f"FILE_NAME {PROJECT}.bs", f"ADDED_MOS {added}", *sets])
        metadata["band_path"] = path
    if task == "dos":
        prints += _section("DOS", ["NDIGITS 8", "DELTA_E 0.0005"])
    if prints:
        dft += _section("PRINT", prints)
    # Kinds: one per element and starting moment so antiparallel sites differ.
    labels, kind_lines = [], []
    seen = {}
    for index, site in enumerate(structure):
        element = site.specie.symbol
        moment = float(moments[index]) if moments is not None else 0.0
        key = (element, round(moment, 6))
        if key not in seen:
            seen[key] = element if not any(k[0] == element for k in seen) else f"{element}_{sum(k[0] == element for k in seen) + 1}"
            body = [f"ELEMENT {element}", f"BASIS_SET {kinds[element]['basis']}", f"POTENTIAL {kinds[element]['potential']}"]
            if moments is not None and abs(moment) > 1e-8:
                body.append(f"MAGNETIZATION {_f(moment)}")
            term = next((t for t in terms if t["element"] == element), None)
            if term:
                body += _section("DFT_PLUS_U", [f"L {term['l']}", f"U_MINUS_J [eV] {_f(term['u_ev'] - term['j_ev'])}"])
            kind_lines += _section("KIND", body, seen[key])
        labels.append(seen[key])
    lattice = structure.lattice
    subsys = [*_section("CELL", [f"ABC {_f(lattice.a)} {_f(lattice.b)} {_f(lattice.c)}",
                                 f"ALPHA_BETA_GAMMA {_f(lattice.alpha)} {_f(lattice.beta)} {_f(lattice.gamma)}", "PERIODIC XYZ"]),
              *_section("COORD", ["SCALED", *(f"{label} " + " ".join(_f(x) for x in site.frac_coords)
                                              for label, site in zip(labels, structure))]),
              *kind_lines]
    if task == "md":
        # Stress and per-step force tables would cost time and fill the output at every step.
        force_eval = ["METHOD Quickstep", *_section("DFT", dft), *_section("SUBSYS", subsys)]
    else:
        force_eval = ["METHOD Quickstep", "STRESS_TENSOR ANALYTICAL", *_section("DFT", dft), *_section("SUBSYS", subsys),
                      *_section("PRINT", [*_section("FORCES", [], "ON"), *_section("STRESS_TENSOR", [], "ON")])]
    lines = [*_section("GLOBAL", [f"PROJECT {PROJECT}", f"RUN_TYPE {run_type}", "PRINT_LEVEL MEDIUM"]),
             *_section("FORCE_EVAL", force_eval), *motion]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / INPUT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    structure.to(filename=str(destination / "structure.cif"))
    files_needed = {table["basis_file"]: catalog()["files"][table["basis_file"]],
                    table["potential_file"]: catalog()["files"][table["potential_file"]]}
    metadata.update(input_file=INPUT_FILE, output_file=OUTPUT_FILE, stdout_file=STDOUT_FILE,
                    arguments=["-i", INPUT_FILE, "-o", OUTPUT_FILE], input_files=[INPUT_FILE, "structure.cif", METADATA_FILE],
                    output_files=list(OUTPUT_FILES), library_dir=LIBRARY_DIR, kinds=kinds, basis_family=family,
                    library_files={name: {"sha256": value["sha256"], "size": value["size"], "kind": "cp2k_data",
                                          "names": sorted({kinds[e]["basis" if name == table["basis_file"] else "potential"] for e in elements})}
                                   for name, value in files_needed.items()},
                    electrons=electrons, mesh=mesh, kpoints=kpoints, functional=functional, spin=spin, soc=False,
                    starting_moments=moments, hubbard=terms, cutoff_ry=cutoff, rel_cutoff_ry=relative,
                    added_mos=0 if ot else added, scf_method="orbital transformation" if ot else "diagonalization",
                    run_type=run_type, notes=[])
    if task == "md":
        metadata["md"] = md_settings
    if task == "dos":
        metadata["dos_mesh"] = mesh
    _json(destination / METADATA_FILE, metadata)
    return metadata


# --- outputs -------------------------------------------------------------------

def _last(pattern, text, label):
    values = re.findall(pattern, text)
    if not values:
        raise ValueError(f"CP2K output has no {label}.")
    return values[-1]


def _frames(path, count, species=None):
    """Cartesian frames (Å) and their step numbers from CP2K's xyz trajectory."""
    lines = Path(path).read_text(errors="replace").splitlines()
    frames, steps, index = [], [], 0
    while index + count + 2 <= len(lines):
        if lines[index].strip() != str(count):
            raise ValueError("The CP2K trajectory is malformed.")
        step = re.search(r"i\s*=\s*(\d+)", lines[index + 1])
        rows = [line.split() for line in lines[index + 2:index + 2 + count]]
        if not step or any(len(row) < 4 for row in rows):
            raise ValueError("The CP2K trajectory is malformed.")
        if species is not None and [re.sub(r"_\d+$", "", row[0]) for row in rows] != species:
            raise ValueError("The CP2K trajectory species differ from the prepared input.")
        steps.append(int(step[1]))
        frames.append([[float(x) for x in row[1:4]] for row in rows])
        index += count + 2
    if not frames:
        raise ValueError("The CP2K trajectory is empty.")
    frames = np.asarray(frames)
    if index != len(lines) or not np.isfinite(frames).all() or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("The CP2K trajectory is incomplete or contains invalid values.")
    return frames, steps


def _rdf(frames, lattice, species, width=0.05):
    """Pair radial distribution functions g(r) by element pair, with minimum images, up to half the shortest cell height."""
    matrix = np.asarray(lattice.matrix)
    heights = lattice.volume / np.linalg.norm(np.cross(matrix[[1, 2, 0]], matrix[[2, 0, 1]]), axis=1)
    edges = np.arange(0.0, min(10.0, 0.5 * float(heights.min())) + width / 2, width)
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    inverse = np.linalg.inv(matrix)
    names = np.asarray(species)
    curves = {}
    for a_index, a in enumerate(sorted(set(species))):
        for b in sorted(set(species))[a_index:]:
            first, second = np.where(names == a)[0], np.where(names == b)[0]
            counts = np.zeros(len(centres))
            for frame in frames:
                delta = frame[second][None, :, :] - frame[first][:, None, :]
                fractional = delta @ inverse
                fractional -= np.round(fractional)
                distance = np.linalg.norm(fractional @ matrix, axis=2)
                if a == b:
                    distance = distance[np.triu_indices(len(first), 1)]
                counts += np.histogram(distance.ravel(), bins=edges)[0]
            pairs = len(first) * (len(first) - 1) / 2 if a == b else len(first) * len(second)
            if pairs:
                curves[f"{a}–{b}"] = counts / (len(frames) * pairs * shell / lattice.volume)
    return centres, curves


def _molecular_dynamics(folder, text, metadata, expected, result):
    settings = metadata["md"]
    energy_file = folder / f"{PROJECT}-1.ener"
    rows = [[float(x) for x in line.split()[:7]] for line in energy_file.read_text(errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) < 2 or any(len(row) < 6 for row in rows) or int(rows[-1][0]) != settings["steps"]:
        raise ValueError("The molecular dynamics run stopped before its last step.")
    data = np.asarray([row[:6] for row in rows])
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) != 1) or np.any(np.diff(data[:, 1]) <= 0):
        raise ValueError("The molecular dynamics energy table contains invalid or missing steps.")
    time, temperature, potential, conserved = data[:, 1], data[:, 3], data[:, 4] * HA_EV, data[:, 5] * HA_EV
    atoms = len(expected)
    species = [site.specie.symbol for site in expected]
    frames, steps = _frames(folder / f"{PROJECT}-pos-1.xyz", atoms, species)
    if steps[-1] != settings["steps"]:
        raise ValueError("The trajectory stops before the last molecular dynamics step.")
    # ABC/ALPHA_BETA_GAMMA puts A along x and B in xy, regardless of the input CIF's frame.
    lattice = Lattice.from_parameters(*expected.lattice.abc, *expected.lattice.angles, vesta=True)
    final = Structure(lattice, species, frames[-1], coords_are_cartesian=True, to_unit_cell=True)
    write_final_structure(folder, result, final)
    half = slice(len(time) // 2, None)
    # Drift of the conserved quantity per atom; for NVE it is the total energy.
    slope = float(np.polyfit(time, conserved, 1)[0]) if len(time) > 2 else 0.0
    later = frames[len(frames) // 2:]
    later = later[:: max(1, len(later) // 200)]
    distance, curves = _rdf(later, lattice, species)
    peaks = {pair: float(distance[int(np.argmax(values))]) for pair, values in curves.items() if values.max() > 0}
    _csv(folder / "md.csv", ["time_fs", "temperature_k", "potential_energy_ev", "conserved_ev"],
         np.column_stack([time, temperature, potential, conserved]).tolist())
    _csv(folder / "rdf.csv", ["r_angstrom", *(f"g_{pair.replace('–', '_')}" for pair in curves)],
         np.column_stack([distance, *curves.values()]).tolist())
    result["artifacts"] += [energy_file.name, f"{PROJECT}-pos-1.xyz", "md.csv", "rdf.csv"]
    panels = {"T (K)": [("temperature", temperature)],
              "E_pot − E_pot(0) (eV)": [("potential energy", potential - potential[0])],
              "Conserved − start (eV)": [("conserved quantity", conserved - conserved[0])]}
    nvt = settings["ensemble"] == "nvt"
    result["artifacts"] += plot_traces(folder, "md", time, panels, guides={"T (K)": settings["temperature_k"]} if nvt else None)
    result["artifacts"] += plot_curves(folder, "rdf", distance, curves, "r (Å)", "g(r)")
    # This catches obvious settling; passing it does not establish thermodynamic equilibrium.
    fit = np.polyfit(time[half], potential[half], 1) if len(time[half]) > 2 else np.zeros(2)
    trend = float(fit[0] * (time[-1] - time[half][0]))
    noise = float(np.std(potential[half] - np.polyval(fit, time[half])))
    mean, spread = float(temperature[half].mean()), float(temperature[half].std())
    drifting = abs(trend) > 3 * noise and abs(trend) > 1e-3 * atoms
    off_target = nvt and abs(mean - settings["temperature_k"]) > max(0.1 * settings["temperature_k"], spread)
    if drifting or off_target:
        found = [f"the potential energy still {'falls' if trend < 0 else 'rises'} by {abs(trend) / atoms * 1e3:.0f} meV/atom"] if drifting else []
        if off_target:
            found.append(f"the mean temperature is {mean:.0f} K against the {settings['temperature_k']:g} K target")
        result["warnings"].append(f"The second half of the run is not in equilibrium yet: {' and '.join(found)}. Relax a crystal first, "
                                  "start a liquid with a short run under a strong thermostat (md_thermostat_fs 10), or run longer "
                                  "before using these averages.")
    result["molecular_dynamics"] = {
        "ensemble": settings["ensemble"], "steps": settings["steps"], "timestep_fs": settings["timestep_fs"],
        "simulated_time_fs": float(time[-1]), "target_temperature_k": settings["temperature_k"],
        "mean_temperature_k": mean, "temperature_std_k": spread,
        "conserved_drift_mev_per_atom_ps": slope * 1e3 * 1e3 / atoms,
        "stationary_second_half": not (drifting or off_target),
        "equilibration_note": "A stationarity check does not establish equilibrium; assess sampling length and independent trajectories.",
        "potential_energy_trend_mev_per_atom": trend / atoms * 1e3,
        "rdf_first_maximum_angstrom": peaks, "averaging": "second half of the run",
        "frames_used_for_rdf": len(later)}


def _restart_structure(path, expected):
    """Final cell (Å) and coordinates from CP2K's restart input."""
    text = Path(path).read_text(errors="replace")
    cell = re.search(r"&CELL\b(.*?)&END CELL", text, re.S | re.I)
    coord = re.search(r"&COORD\b(.*?)&END COORD", text, re.S | re.I)
    if not cell or not coord:
        raise ValueError("The CP2K restart file has no final geometry.")
    vectors = {}
    for name in ("A", "B", "C"):
        match = re.search(rf"^\s*{name}\s+(?:\[angstrom\]\s+)?(" + r")\s+(".join([_NUMBER.pattern] * 3) + r")\s*$", cell[1], re.M | re.I)
        if not match:
            raise ValueError("The CP2K restart cell is incomplete.")
        vectors[name] = [float(x) for x in match.groups()]
    lattice = Lattice([vectors["A"], vectors["B"], vectors["C"]])
    scaled = bool(re.search(r"^\s*SCALED\s+(?:T|TRUE|\.TRUE\.)?\s*$", coord[1], re.M | re.I))
    rows = [line.split() for line in coord[1].splitlines() if len(line.split()) >= 4 and not line.strip().upper().startswith(("SCALED", "UNIT"))]
    if len(rows) != len(expected):
        raise ValueError("The CP2K restart lists a different number of atoms.")
    coordinates = np.asarray([[float(x) for x in row[1:4]] for row in rows])
    species = [site.specie.symbol for site in expected]
    labels = [re.sub(r"_\d+$", "", row[0]) for row in rows]
    if labels != species:
        raise ValueError("CP2K restart species differ from the prepared input.")
    return Structure(lattice, species, coordinates, coords_are_cartesian=not scaled)


def _same_crystal(actual, expected, atol=1e-5):
    """Compare lengths, angles and fractional positions; CP2K rotates the frame."""
    if [site.specie.symbol for site in actual] != [site.specie.symbol for site in expected]:
        return False
    a, e = actual.lattice, expected.lattice
    delta = actual.frac_coords - expected.frac_coords
    return bool(np.allclose(a.abc, e.abc, atol=atol) and np.allclose(a.angles, e.angles, atol=1e-4)
                and np.allclose(delta - np.rint(delta), 0, atol=atol))


def _band_file(path):
    """Band sets: points, labels, energies in eV and occupations, shaped (spin, k, band)."""
    text = Path(path).read_text(errors="replace")
    sets = []
    for chunk in re.split(r"^#\s+Set\s+\d+:", text, flags=re.M)[1:]:
        specials = re.findall(r"^#\s+Special point\s+\d+\s+(" + r")\s+(".join([_NUMBER.pattern] * 3) + r")\s+(\S+)\s*$", chunk, re.M)
        points, spins, occupations = [], {}, {}
        for block in re.split(r"^#\s+Point\s+", chunk, flags=re.M)[1:]:
            header = block.splitlines()[0]
            spin = int(re.search(r"Spin\s+(\d+)", header)[1])
            coords = [float(x) for x in _NUMBER.findall(header.split(":", 1)[1])][:3]
            rows = [line.split() for line in block.splitlines()[1:] if re.match(r"^\s*\d+\s+" + _NUMBER.pattern, line)]
            if not rows or any(len(row) < 3 for row in rows):
                raise ValueError("CP2K band occupations are incomplete.")
            spins.setdefault(spin, []).append([float(row[1]) for row in rows])
            occupations.setdefault(spin, []).append([float(row[2]) for row in rows])
            if spin == 1:
                points.append(coords)
        if not points or any(len(values) != len(points) for values in spins.values()):
            raise ValueError("CP2K band-structure file is incomplete.")
        labels = {tuple(round(float(x), 6) for x in item[:3]): item[3] for item in specials}
        sets.append((np.asarray(points), [labels.get(tuple(round(x, 6) for x in point), "") for point in points],
                     np.asarray([spins[key] for key in sorted(spins)]),
                     np.asarray([occupations[key] for key in sorted(spins)])))
    if not sets:
        raise ValueError("CP2K wrote no band-structure sets.")
    return sets


def _dos_files(folder):
    names = [f"{PROJECT}-1.dos"] if (folder / f"{PROJECT}-1.dos").is_file() else [f"{PROJECT}-ALPHA-1.dos", f"{PROJECT}-BETA-1.dos"]
    if not all((folder / name).is_file() for name in names):
        raise ValueError("The CP2K DOS file is missing.")
    return names


def _histograms(text):
    """(energy, density, occupation) per spin from one CP2K DOS file: one spin, or both spins as
    energy, alpha density, occupation, beta density, occupation. Each density sums to one."""
    rows = np.asarray([[float(x) for x in line.split()] for line in text.splitlines()
                       if line.strip() and not line.lstrip().startswith("#")])
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 2:
        raise ValueError("The CP2K DOS histogram is empty.")
    return [rows[:, :3]] if rows.shape[1] < 5 else [rows[:, [0, 1, 2]], rows[:, [0, 3, 4]]]


def _broadened_dos(folder, names, states, width_ev):
    """Gaussian-broadened DOS in states/eV/cell from CP2K's normalized eigenvalue histograms."""
    columns, fermi = [], None
    grid = None
    histograms = []
    for name in names:
        text = (folder / name).read_text(errors="replace")
        match = re.search(r"E\(Fermi\)\s*=\s*(" + _NUMBER.pattern + r")\s*a\.u\.", text)
        if match:
            fermi = float(match[1]) * HA_EV
        histograms += _histograms(text)
    low = min(rows[:, 0].min() for rows in histograms) * HA_EV - 5 * width_ev
    high = max(rows[:, 0].max() for rows in histograms) * HA_EV + 5 * width_ev
    grid = np.linspace(low, high, 1200)
    for rows in histograms:
        weights = rows[:, 1] * states / len(histograms)
        centres = rows[:, 0] * HA_EV
        kernel = np.exp(-0.5 * ((grid[:, None] - centres[None, :]) / width_ev) ** 2) / (width_ev * math.sqrt(2 * math.pi))
        columns.append(kernel @ weights)
    return grid, np.stack(columns, axis=1), fermi


def _histogram_gap(folder, names, output_fermi_ev=None):
    """Band edges from CP2K's DOS histogram, split at its Fermi level.

    CP2K 2026.2 writes E(Fermi) in the histogram header; 2026.1 does not, so the
    Fermi energy that the same run printed in its main output is used instead."""
    fermi, filled, width, partial = None, [], None, False
    for name in names:
        text = (folder / name).read_text(errors="replace")
        match = re.search(r"E\(Fermi\)\s*=\s*(" + _NUMBER.pattern + r")\s*a\.u\.", text)
        level = float(match[1]) if match else None if output_fermi_ev is None else output_fermi_ev / HA_EV
        if level is None:
            return {"available": False, "status": "unavailable", "note": "The DOS histogram has no Fermi level.", "scope": "sampled_k_points"}
        fermi = level if fermi is None else max(fermi, level)
        for rows in _histograms(text):
            states = rows[:, 1] > 0
            filled += [float(energy) for energy in rows[states, 0]]
            if rows.shape[1] > 2 and states.any():
                # A populated bin that is neither empty nor full: the Fermi level cuts a band, even where
                # a sparse mesh leaves empty bins around it (Al and Co had shown 0.12 and 0.05 eV).
                share = rows[states, 2] / rows[states, 1]
                partial = partial or bool(np.any((share > 0.1 * share.max()) & (share < 0.9 * share.max())))
            step = float(np.min(np.diff(np.unique(rows[:, 0]))))
            width = step if width is None else min(width, step)
    below = [energy for energy in filled if energy <= fermi]
    above = [energy for energy in filled if energy > fermi]
    if not below or not above:
        return {"available": False, "status": "unavailable", "note": "Both occupied and empty states are needed.", "scope": "sampled_k_points"}
    vbm, cbm = max(below), min(above)
    metallic = bool(cbm - vbm <= 2 * width) or partial
    gap = 0.0 if metallic else float((cbm - vbm) * HA_EV)
    return {"available": True, "energy_ev": gap, "gap_ev": gap, "vbm_ev": vbm * HA_EV, "cbm_ev": cbm * HA_EV,
            "metallic_or_smeared": metallic, "status": "metallic_or_partially_occupied" if metallic else "gap",
            "scope": "sampled_k_points", "resolution_ev": float(width * HA_EV),
            "note": "From CP2K's eigenvalue histogram on the DOS mesh; resolution is one histogram bin."}


def _mulliken(text, result, structure):
    start = text.rfind("Mulliken Population Analysis")
    if start < 0:
        return
    block = text[start:start + 200 + 120 * len(structure)]
    if "Spin moment" not in block:
        return
    sites = []
    for line in block.splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[0].isdigit() and fields[1] in {site.specie.symbol for site in structure}:
            sites.append({"atom": int(fields[0]), "element": fields[1], "moment_mu_b": float(fields[-1])})
    total = re.search(r"# Total charge and spin\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + ")", block)
    if total:
        result["total_magnetization_mu_b"] = float(total[4])
    if len(sites) == len(structure):
        result["site_magnetization"] = sites
        result["absolute_magnetization_mu_b"] = float(sum(abs(item["moment_mu_b"]) for item in sites))
        result["site_magnetization_scope"] = "Mulliken spin populations; basis-dependent local moments."


def analyze_outputs(folder, task, expected_structure=None):
    """Require normal completion plus task-specific convergence and numerical evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in GENERATED:
        (folder / name).unlink(missing_ok=True)
    result = new_result(ENGINE, task)
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported CP2K task: {task}")
        metadata = json.loads((folder / METADATA_FILE).read_text())
        if metadata.get("canonical_task") != task or metadata.get("engine") != ENGINE:
            raise ValueError("Stage metadata does not match the requested analysis task.")
        output = folder / OUTPUT_FILE
        if not output.is_file():
            raise ValueError(f"{OUTPUT_FILE} is missing.")
        text = output.read_text(encoding="utf-8", errors="replace")
        result["stdout_sha256"] = _hash(output)
        result["artifacts"].append(OUTPUT_FILE)
        if "PROGRAM ENDED AT" not in text:
            raise ValueError("CP2K did not finish normally.")
        if re.search(r"\[ABORT\]|ABORT\b.*CP2K", text):
            raise ValueError("CP2K aborted.")
        # With orbital transformation the outer loop reports convergence; an MD run repeats the SCF at every step.
        converged = (("SCF run converged" in text or "outer SCF loop converged" in text)
                     and "SCF run NOT converged" not in text and "outer SCF loop FAILED" not in text)
        result["converged_electronic"] = converged
        if not converged:
            raise ValueError("The SCF did not converge.")
        expected = structure_from_native(metadata["native_structure"])
        if task == "relax":
            ionic = "GEOMETRY OPTIMIZATION COMPLETED" in text and "MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED" not in text
            result["converged_ionic"] = ionic
            if not ionic:
                raise ValueError("The geometry optimization did not converge within the step limit.")
            structure = _restart_structure(folder / f"{PROJECT}-1.restart", expected)
            write_final_structure(folder, result, structure)
        elif task == "md":
            structure = expected
            _molecular_dynamics(folder, text, metadata, expected, result)
        else:
            structure = expected
            if task == "scf":
                write_final_structure(folder, result, expected)
        energy = float(_last(r"ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[(?:hartree|a\.u\.)\]\s+(" + _NUMBER.pattern + ")", text, "total energy"))
        if task in {"scf", "relax"}:
            finish_energies(result, energy * HA_EV, len(structure))
        forces = re.findall(r"^\s*FORCES\|\s+\d+\s+(" + r")\s+(".join([_NUMBER.pattern] * 3) + r")\s+" + _NUMBER.pattern, text, re.M)
        if forces and task in {"scf", "relax"}:
            values = np.asarray(forces[-len(structure):], dtype=float) * HA_PER_BOHR_EV_PER_ANGSTROM
            result["final_max_force_ev_angstrom"] = float(np.max(np.linalg.norm(values, axis=1)))
        stress = text.rfind("STRESS| Analytical stress tensor [bar]")
        if stress >= 0 and task in {"scf", "relax"}:
            rows = re.findall(r"STRESS\|\s+[xyz]\s+(" + r")\s+(".join([_NUMBER.pattern] * 3) + ")", text[stress:stress + 600])[:3]
            if len(rows) == 3:
                tensor = np.asarray(rows, dtype=float) * GPA_PER_BAR
                result.update(stress_gpa=tensor.tolist(), pressure_gpa=float(np.trace(tensor) / 3))
        fermi = re.findall(r"Fermi energy:\s+(" + _NUMBER.pattern + ")", text)
        if fermi:
            result["fermi_energy_ev"] = float(fermi[-1]) * HA_EV
        result["spin_mode"] = metadata.get("spin")
        result["spin_orbit_coupling"] = False
        _mulliken(text, result, structure)
        if task == "bands":
            reciprocal = expected.lattice.reciprocal_lattice
            points, labels, blocks, occupation_blocks, distance, breaks, position = [], [], [], [], [], [0], 0.0
            for set_points, set_labels, set_energies, set_occupations in _band_file(folder / f"{PROJECT}.bs"):
                cartesian = reciprocal.get_cartesian_coords(set_points)
                local = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
                if points and np.allclose(points[-1], set_points[0], atol=1e-6):
                    # A continuous path repeats the shared special point; keep it once.
                    set_points, set_labels, set_energies, local = set_points[1:], set_labels[1:], set_energies[:, 1:], local[1:]
                    set_occupations = set_occupations[:, 1:]
                elif points:
                    breaks.append(len(points))
                    if labels and set_labels and set_labels[0]:
                        set_labels = list(set_labels)
                        set_labels[0] = f"{labels[-1]}|{set_labels[0]}" if labels[-1] else set_labels[0]
                        labels[-1] = ""
                points += [list(point) for point in set_points]
                labels += set_labels
                blocks.append(set_energies)
                occupation_blocks.append(set_occupations)
                distance += list(position + local)
                position = distance[-1]
            breaks.append(len(points))
            prepared = metadata["band_path"]["reciprocal_fractional"]
            if len(points) != len(prepared) or not np.allclose(points, prepared, atol=1e-5):
                raise ValueError("Band output k-points differ from the prepared path.")
            energies = np.concatenate(blocks, axis=1)
            occupations = np.concatenate(occupation_blocks, axis=1)
            # The .bs occupations split degenerate states of a magnet (CrI3: 39 and 31 filled bands for
            # 38 and 32 electrons per spin), so they decide only when the run printed no Fermi level.
            fermi = result.get("fermi_energy_ev")
            gap = band_gap(energies, electrons=metadata["electrons"] if energies.shape[0] == 1 else None,
                           fermi=fermi, occupations=occupations if fermi is None else None, scope="sampled_path")
            reference, kind = band_reference(gap, fermi)
            result.update(band_gap=gap, band_gap_ev=gap.get("gap_ev"), kpoint_count=len(points), band_count=int(energies.shape[2]),
                          plot_reference_ev=reference, energy_reference=kind)
            rows = [[spin + 1, k + 1, *points[k], b + 1, float(energies[spin, k, b])]
                    for spin in range(energies.shape[0]) for k in range(energies.shape[1]) for b in range(energies.shape[2])]
            _csv(folder / "bands.csv", ["spin", "k_index", "k1", "k2", "k3", "band", "energy_ev"], rows)
            result["artifacts"] += [f"{PROJECT}.bs", "bands.csv"]
            result["artifacts"] += plot_bands(folder, "bands", distance, energies - reference, labels,
                                              [[breaks[i], breaks[i + 1]] for i in range(len(breaks) - 1)], kind)
        elif task == "dos":
            names = _dos_files(folder)
            states = 2 * (math.ceil(metadata["electrons"] / 2) + metadata["added_mos"])
            width = _number(metadata["parameters"].get("dos_broadening_ev", 0.1), "dos_broadening_ev", True)
            energy, dos, fermi = _broadened_dos(folder, names, states, width)
            if fermi is not None:
                result["fermi_energy_ev"] = fermi
            result["artifacts"] += names
            write_dos(folder, result, energy, dos, result.get("fermi_energy_ev") or 0.0,
                      columns=("dos_up", "dos_down") if dos.shape[1] == 2 else ("dos",))
            result["dos_broadening_ev"] = width
            gap = _histogram_gap(folder, names, result.get("fermi_energy_ev"))
            result.update(band_gap=gap, band_gap_ev=gap.get("gap_ev"))
        result.update(success=True, reason="Required outputs and convergence checks passed.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result["reason"] = str(exc) or "CP2K output is truncated."
    _json(folder / RESULT_FILE, result)
    return result


LIBRARY_VALIDATOR = r'''def validate_library(data, entry):
    if len(data) < 1000 or len(data) > 200 * 1024 * 1024:
        raise ValueError("CP2K data file is empty or unexpectedly large.")
    text = data.decode("utf-8", errors="replace")
    for name in entry.get("names", []):
        if not re.search(r"\s" + re.escape(name) + r"(\s|$)", text):
            raise ValueError("CP2K data file lacks " + name + ".")
    return {"names": entry.get("names", [])}
'''
