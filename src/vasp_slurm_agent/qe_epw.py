"""EPW 5.9 isotropic Eliashberg calculations from a computed alpha2F table.

Native contracts: QE 7.4.1 EPW/src/{supercond_iso,supercond,io/io_supercond}.f90.
This solves a specified phonon pairing model; it does not establish a material's
superconductivity or replace Wannier interpolation and convergence studies.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import qe

TASKS = ("epw_iso", "epw_tc", "epw_aniso")
PROGRAMS = {task: "epw.x" for task in TASKS}
PARAMETER_KEYS = frozenset({"temperatures", "mu_star", "frequency_cutoff_ev", "iterations", "convergence", "mixing", "broadening_index"})
SOURCE = "https://github.com/QEF/q-e/tree/qe-7.4.1/EPW/src"
KB_EV = 8.617333262145179e-5
THZ_MEV = 4.135667696923859
AMU_RY = 1822.888486217313 / 2
RY_CM = 109737.31568539


def read_checkpoint(folder, prefix=qe.PREFIX, *, acoustic_cutoff_cm1=5.0):
    """Validate EPW 5.9 little-endian stream checkpoints before continuation."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Invalid EPW checkpoint prefix.")
    folder = Path(folder)
    directory = folder / "scratch" / (prefix + ".ephmat")
    if not directory.is_dir():
        directory = folder / (prefix + ".ephmat")
    files = [directory / name for name in ("freq", "egnv", "ikmap", "npool")]
    files += [folder / "crystal.fmt", folder / "selecq.fmt"]
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("The native EPW checkpoint is incomplete.")
    f = files[0]
    header = np.fromfile(f, dtype="<i4", count=5)
    if len(header) != 5 or (header <= 0).any():
        raise ValueError("Invalid phonon checkpoint dimensions or byte order.")
    nq, q1, q2, q3, modes = map(int, header)
    if nq != q1 * q2 * q3 or f.stat().st_size != 20 + 8 * nq * (3 + modes):
        raise ValueError("The phonon checkpoint size differs from its grid.")
    crystal_lines = (folder / "crystal.fmt").read_text().splitlines()
    if len(crystal_lines) < 3 or int(crystal_lines[0].strip()) * 3 != modes or int(crystal_lines[1].strip()) != modes:
        raise ValueError("The phonon modes disagree with the native crystal header.")
    qdata = np.fromfile(f, dtype="<f8", offset=20).reshape(nq, 3 + modes)
    if not np.isfinite(qdata).all():
        raise ValueError("Nonfinite phonon checkpoint data.")
    qcoords, frequencies = qdata[:, :3], qdata[:, 3:]
    acoustic = qe._number(acoustic_cutoff_cm1, "Acoustic cutoff", True)
    if (frequencies * RY_CM < -acoustic).any():
        raise ValueError("Imaginary phonons exceed the declared acoustic allowance.")
    qindices = np.rint(qcoords * [q1, q2, q3]).astype(int)
    if not np.allclose(qcoords * [q1, q2, q3], qindices, atol=1e-6, rtol=0) or len(set(map(tuple, qindices % [q1, q2, q3]))) != nq:
        raise ValueError("The native q grid is incomplete or shifted.")
    f = files[1]
    h = np.fromfile(f, dtype="<i4", count=6)
    if len(h) != 6 or (h <= 0).any():
        raise ValueError("Invalid electronic checkpoint dimensions or byte order.")
    nk, k1, k2, k3, nks, bands = map(int, h)
    if nks > nk or nk > k1 * k2 * k3 or f.stat().st_size != 64 + nk * (40 + 8 * bands):
        raise ValueError("The electronic checkpoint size differs from its grid.")
    ef, ef0, dos, smearing, window = np.fromfile(f, dtype="<f8", offset=24, count=5)
    if not np.isfinite([ef, ef0, dos, smearing, window]).all() or min(dos, smearing, window) <= 0:
        raise ValueError("Invalid Fermi level, DOS or electronic smearing.")
    dtype = np.dtype([("weight", "<f8"), ("k", "<f8", (3,)), ("index", "<i4"), ("shell", "<i4"), ("energies", "<f8", (bands,))])
    electrons = np.fromfile(f, dtype=dtype, offset=64)
    if not np.isfinite(electrons["weight"]).all() or (electrons["weight"] <= 0).any() or not np.isclose(electrons["weight"].sum(), 2, atol=1e-8, rtol=1e-7):
        raise ValueError("The scalar k-point weights must sum to two.")
    if not np.isfinite(electrons["k"]).all() or not np.isfinite(electrons["energies"]).all() or not np.array_equal(electrons["index"], np.arange(1, nk + 1)):
        raise ValueError("Invalid electronic checkpoint records.")
    shell = electrons["shell"]
    if not np.array_equal(shell[shell > 0], np.arange(1, nks + 1)) or (shell < 0).any():
        raise ValueError("The Fermi-shell map is incomplete.")
    in_window = np.abs(electrons["energies"] - ef0) < window
    if not np.array_equal(np.any(np.abs(electrons["energies"] - ef) < window, axis=1), shell > 0):
        raise ValueError("The shell map disagrees with its native energy window.")
    maps = np.fromfile(files[2], dtype="<i4")
    fullk = k1 * k2 * k3
    if len(maps) != 2 * fullk:
        raise ValueError("The k+q checkpoint mapping is truncated.")
    ixk, bzmap = maps[:fullk], maps[fullk:]
    if (ixk < 0).any() or (ixk > nks).any() or (bzmap < 1).any() or (bzmap > nk).any() or not np.array_equal(ixk, shell[bzmap - 1]):
        raise ValueError("The full-grid and Fermi-shell maps disagree.")
    selected = np.array([int(x) for x in (folder / "selecq.fmt").read_text().split()], dtype=int)
    if len(selected) < 3 or selected[0] != len(selected) - 2 or selected[1] != nq or len(set(selected[2:])) != len(selected[2:]) or min(selected[2:]) < 1 or max(selected[2:]) > nq:
        raise ValueError("Invalid selected-q checkpoint.")
    pools = np.fromfile(files[3], dtype="<i4")
    matrix_files = sorted(directory.glob("ephmat*"))
    if len(pools) != 1 or pools[0] <= 0 or len(matrix_files) != pools[0]:
        raise ValueError("The electron-phonon pool files are incomplete.")
    k_shell = electrons[shell > 0]
    counts = np.count_nonzero(in_window, axis=1)
    cursor, matrix_values = 0, 0
    for pool, file in enumerate(matrix_files, 1):
        ids = np.fromfile(file, dtype="<i4", count=2)
        if len(ids) != 2 or ids[0] != pool or ids[1] < 0 or cursor + ids[1] > nks:
            raise ValueError("Invalid electron-phonon pool header.")
        expected = 0
        for iq in selected[2:] - 1:
            ks = k_shell[cursor:cursor + ids[1]]
            grid = (ks["k"] + qcoords[iq]) * [k1, k2, k3]
            rounded = np.rint(grid)
            if not np.allclose(grid, rounded, atol=1e-5, rtol=0):
                raise ValueError("Fine k and q grids are not commensurate.")
            ijk = rounded.astype(int) % [k1, k2, k3]
            full_indices = (ijk[:, 0] * k2 + ijk[:, 1]) * k3 + ijk[:, 2]
            partners = bzmap[full_indices] - 1
            available = shell[partners] > 0
            expected += int(modes * np.sum(counts[ks["index"] - 1] * counts[partners] * available))
        if file.stat().st_size != 8 + expected * 8:
            raise ValueError("Electron-phonon matrix dimensions disagree with the checkpoint grids.")
        values = np.memmap(file, mode="r", dtype="<f8", offset=8, shape=(expected,)) if expected else np.array([])
        for start in range(0, expected, 1_000_000):
            part = values[start:start + 1_000_000]
            if not np.isfinite(part).all() or (part < 0).any():
                raise ValueError("Invalid electron-phonon matrix values.")
        matrix_values += expected
        cursor += int(ids[1])
    if cursor != nks or matrix_values == 0:
        raise ValueError("The electron-phonon matrix is incomplete or empty.")
    files += matrix_files
    states = []
    for row in k_shell:
        for band, energy in enumerate(row["energies"], 1):
            if abs(energy - ef0) < window:
                weight = .5 * row["weight"] * math.exp(-((energy - ef0) / smearing)**2) / (math.sqrt(math.pi) * smearing * dos)
                states.append({"kpoint": int(row["index"]), "band": band, "energy_minus_fermi_ev": float((energy - ef0) * qe.RY_EV), "weight": float(weight), "k_fractional": row["k"].tolist()})
    return {"fine_kmesh": [k1, k2, k3], "fine_qmesh": [q1, q2, q3], "modes": modes, "kpoints": nk, "shell_kpoints": nks, "states": states, "matrix_values": matrix_values, "acoustic_cutoff_cm1": acoustic, "minimum_frequency_cm1": float(frequencies.min() * RY_CM), "sha256": {p.relative_to(folder).as_posix(): qe._hash(p) for p in files}}


def validate_parameters(parameters, task="epw_iso"):
    if task not in TASKS:
        raise ValueError("Choose epw_iso, epw_tc or epw_aniso.")
    p = dict(parameters or {})
    if set(p) - PARAMETER_KEYS:
        raise ValueError("Unsupported EPW settings: " + ", ".join(sorted(set(p) - PARAMETER_KEYS)))
    for key in ("temperatures", "mu_star", "frequency_cutoff_ev"):
        if key not in p:
            raise ValueError(f"Specify {key} for the Eliashberg calculation.")
    if not isinstance(p["temperatures"], (list, tuple)) or not 1 <= len(p["temperatures"]) <= 100:
        raise ValueError("Supply between one and 100 temperatures in kelvin.")
    p["temperatures"] = [qe._number(t, "Temperature", True) for t in p["temperatures"]]
    if any(t >= 1000 or not np.isclose(t * 100, round(t * 100), atol=1e-8, rtol=0) for t in p["temperatures"]):
        raise ValueError("EPW output uses temperatures below 1000 K with at most two decimals.")
    if any(b <= a for a, b in zip(p["temperatures"], p["temperatures"][1:])):
        raise ValueError("Temperatures must be distinct and increasing.")
    if task == "epw_tc" and len(p["temperatures"]) < 2:
        raise ValueError("A Tc search needs at least two temperatures.")
    p["mu_star"] = qe._number(p["mu_star"], "mu_star", True)
    if p["mu_star"] >= 1:
        raise ValueError("mu_star must be between zero and one.")
    p["frequency_cutoff_ev"] = qe._number(p["frequency_cutoff_ev"], "frequency_cutoff_ev", True)
    p["iterations"] = qe._integer(p.get("iterations", 500), "iterations", 2)
    p["convergence"] = qe._number(p.get("convergence", 1e-6), "convergence", True)
    if p["convergence"] >= 0.01:
        raise ValueError("Use a convergence threshold below 0.01.")
    p["mixing"] = qe._number(p.get("mixing", 0.2), "mixing", True)
    if p["mixing"] > 1:
        raise ValueError("mixing must be at most one.")
    if "broadening_index" in p:
        p["broadening_index"] = qe._integer(p["broadening_index"], "broadening_index")
    if any(_nsiw(t, p["frequency_cutoff_ev"]) < 2 for t in p["temperatures"]):
        raise ValueError("The Matsubara cutoff must retain at least two frequencies at every temperature.")
    return p


def _nsiw(temperature, cutoff):
    return int(0.5 * (cutoff / math.pi / KB_EV / temperature - 1)) + 1


def _matches_structure_file(actual, expected, path):
    # pw.x runs the file's structure in the rotated frame of qe._standard_frame; runs prepared before
    # that rotation ran it in the file's own frame.
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter

    if any(qe._same_structure(actual, candidate) for candidate in (expected, qe._standard_frame(expected)[0], Structure.from_file(path))):
        return True
    if Path(path).suffix.lower() != ".cif":
        return False

    serialized = Structure.from_str(str(CifWriter(actual)), fmt="cif")
    return qe._same_structure(serialized, expected)


def read_spectrum(path, *, unit="meV", column=1, frequency_precision=None):
    """Retain native spectral weights; repair only documented decimal rounding."""
    if unit not in {"meV", "THz"}:
        raise ValueError("Specify spectrum frequencies in meV or THz.")
    column = qe._integer(column, "Spectrum column")
    path = Path(path)
    first = path.read_text().splitlines()[0]
    if re.fullmatch(r"\s*w\[meV\] a2f and integrated 2\*a2f/w for\s+\d+ smearing values\s*", first):
        data = read_native_spectrum(path)
        if unit != "meV" or column > (data.shape[1] - 1) // 2:
            raise ValueError("Choose an alpha2F column from the native meV spectrum.")
        frequency_precision = 1e-7 if frequency_precision is None else frequency_precision
    else:
        data = qe._table(path, column + 1)
    w, a2f = data[:, 0].copy(), data[:, column].copy()
    if len(w) < 10 or (w < 0).any() or (np.diff(w) <= 0).any() or (a2f < 0).any():
        raise ValueError("alpha2F needs increasing nonnegative frequencies and nonnegative weights.")
    if w[0] == 0:
        if a2f[0] != 0:
            raise ValueError("The zero-frequency spectral weight must vanish.")
        w, a2f = w[1:], a2f[1:]
    step = w[-1] / len(w)
    uniform = step * np.arange(1, len(w) + 1)
    tolerance = 1e-9 * max(1, w[-1])
    if frequency_precision is not None:
        precision = qe._number(frequency_precision, "Frequency precision", True)
        tolerance += precision
    if not np.allclose(w, uniform, atol=tolerance, rtol=0):
        raise ValueError("EPW needs positive frequencies i*delta on a uniform grid.")
    correction = float(np.max(np.abs(w - uniform)))
    w = uniform * (THZ_MEV if unit == "THz" else 1)
    lam = float(2 * w[-1] / len(w) * np.sum(a2f / w))
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("The integrated electron-phonon coupling must be positive.")
    omega_log = float(np.exp(2 * w[-1] / len(w) * np.sum(a2f * np.log(w) / w) / lam))
    return w, a2f, {"lambda": lam, "omega_log_mev": omega_log, "points": len(w), "source_unit": unit, "source_column": column, "frequency_rounding_correction": correction, "frequency_precision": frequency_precision}


def read_native_spectrum(path):
    lines = Path(path).read_text().splitlines()
    header = re.fullmatch(r"\s*w\[meV\] a2f and integrated 2\*a2f/w for\s+(\d+) smearing values\s*", lines[0])
    if not header:
        raise ValueError("Invalid native EPW alpha2F header.")
    smearings = int(header[1])
    records, cursor = [], 1
    while cursor < len(lines) and re.match(r"\s*[+-]?\d", lines[cursor]):
        row = qe._floats(lines[cursor])
        if len(row) != 1 + 2 * smearings:
            raise ValueError("The native alpha2F broadening columns are incomplete.")
        records.append(row)
        cursor += 1
    data = np.asarray(records)
    if len(data) < 10 or (data[:, 0] <= 0).any() or (np.diff(data[:, 0]) <= 0).any() or (data[:, 1:] < 0).any():
        raise ValueError("Invalid native alpha2F frequencies or weights.")
    trailer = "\n".join(lines[cursor:])
    for label in ("Electron smearing (eV)", "Fermi window (eV)", "Summed el-ph coupling"):
        values = re.findall(re.escape(label) + r"\s+(" + qe.FLOAT + r")", trailer)
        if len(values) != 1 or qe._floats(values[0])[0] <= 0:
            raise ValueError("The native alpha2F metadata are incomplete.")
    step = data[-1, 0] / len(data)
    frequencies = step * np.arange(1, len(data) + 1)
    integrated = np.cumsum(2 * step * data[:, 1:1 + smearings] / frequencies[:, None], axis=0)
    if not np.allclose(integrated, data[:, 1 + smearings:], atol=2e-6, rtol=2e-6):
        raise ValueError("The alpha2F integrals disagree with the native spectral weights.")
    return data


def _crystal_from_schema(schema, output, *, allow_nscf=False):
    """Create the scalar solver's geometry header from actual PW metadata.

    EPW requires crystal.fmt even for fila2f. Its scalar isotropic reader ignores
    Wannier centres and cutoffs; no Wannier Hamiltonian is manufactured here.
    """
    root = ET.parse(schema).getroot()
    node = qe._required(root, "output/atomic_structure")
    structure = qe._xml_structure(node)
    bands = qe._required(root, "output/band_structure")
    if qe._required(bands, "noncolin").text.strip().lower() != "false" or qe._required(bands, "lsda").text.strip().lower() != "false":
        raise ValueError("This scalar alpha2F route requires a nonmagnetic PW checkpoint without SOC.")
    is_nscf = root.findtext("input/control_variables/calculation", "").strip().lower() == "nscf"
    if qe._required(root, "output/convergence_info/scf_conv/convergence_achieved").text.strip().lower() != "true" and not (allow_nscf and is_nscf):
        raise ValueError("The PW source checkpoint is not converged.")
    species = qe._required(root, "input/atomic_species").findall("species")
    if not 1 <= len(species) <= 10:
        raise ValueError("The pinned QE build supports at most ten species in crystal.fmt.")
    labels = [s.attrib["name"] for s in species]
    masses = [qe._number(qe._required(s, "mass").text, "Atomic mass", True) * AMU_RY for s in species]
    masses += [0.0] * (10 - len(masses))
    types = [labels.index(atom.attrib["name"]) + 1 for atom in node.findall("atomic_positions/atom")]
    alat = qe._number(node.attrib["alat"], "alat", True)
    cell = structure.lattice.matrix / qe.BOHR_ANGSTROM
    at = cell.T / alat
    bg = np.linalg.inv(at).T
    coords = structure.cart_coords / qe.BOHR_ANGSTROM / alat
    values = [str(len(structure)), str(3 * len(structure)), f"{qe._number(qe._required(bands, 'nelec').text, 'nelec', True):.16g} 0"]
    fmt = lambda array: " ".join(f"{x:.16g}" for x in np.asarray(array).reshape(-1))
    values += [fmt(at.T), fmt(bg.T), f"{abs(np.linalg.det(cell)):.16g}", f"{alat:.16g}", fmt(coords), fmt(masses), " ".join(map(str, types)), "F", "F", "# Wannier centres are unused by the scalar fila2f solver", "# Wannier ranges are unused by the scalar fila2f solver"]
    Path(output).write_text("\n".join(values) + "\n")
    return {"number_of_atoms": len(structure), "formula": structure.composition.reduced_formula, "qe_schema_sha256": qe._hash(schema),
            "native_structure": {"lattice_angstrom": structure.lattice.matrix.tolist(), "elements": [site.specie.symbol for site in structure], "fractional_positions": structure.frac_coords.tolist()}}


def _prepare_native(destination, task, parameters, *, spectrum_path, qe_schema, unit="meV", column=1, frequency_precision=None, provenance=None, allow_nscf=False):
    """Prepare a standalone native EPW job; execution uses the existing runner."""
    p = validate_parameters(parameters, task)
    w, a2f, spectrum = read_spectrum(spectrum_path, unit=unit, column=column, frequency_precision=frequency_precision)
    if p["frequency_cutoff_ev"] <= w[-1] / 1000:
        raise ValueError("The Matsubara cutoff must exceed the spectrum's maximum energy.")
    if spectrum["lambda"] - p["mu_star"] * (1 + 0.62 * spectrum["lambda"]) <= 0:
        raise ValueError("This spectrum and mu_star do not give a valid initial EPW gap estimate.")
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ("epw.in", "epw.out", "qe-input.json")):
        raise ValueError("Use a new folder for an EPW calculation.")
    shutil.copyfile(spectrum_path, folder / "source.a2f")
    shutil.copyfile(qe_schema, folder / "source-schema.xml")
    crystal = _crystal_from_schema(folder / "source-schema.xml", folder / "crystal.fmt", allow_nscf=allow_nscf)
    np.savetxt(folder / "alpha2f.dat", np.column_stack([w, a2f]), fmt="%.16e", header="energy_meV alpha2F", comments="# ")
    native = {"prefix": qe.PREFIX, "outdir": "./scratch", "ep_coupling": False, "elph": False, "epwread": True, "epwwrite": False, "wannierize": False, "eliashberg": True, "liso": True, "laniso": False, "limag": True, "lpade": False, "lacon": False, "lreal": False, "fila2f": "alpha2f.dat", "nqstep": len(w), "muc": p["mu_star"], "wscut": p["frequency_cutoff_ev"], "nsiter": p["iterations"], "conv_thr_iaxis": p["convergence"], "broyden_beta": p["mixing"], "gridsamp": 0, "nstemp": len(p["temperatures"]), "tc_linear": task == "epw_tc", "tc_linear_solver": "lapack"}
    # Fortran may evaluate MOD before the fila2f guard; these grids are unused.
    native.update({f"{kind}{axis}": 1 for kind in ("nkf", "nqf") for axis in (1, 2, 3)})
    native.update({f"temps({i})": t for i, t in enumerate(p["temperatures"], 1)})
    (folder / "scratch").mkdir(exist_ok=True)
    (folder / "epw.in").write_text("DFT Agent\n" + qe.namelist("inputepw", native))
    files = ("epw.in", "alpha2f.dat", "source.a2f", "source-schema.xml", "crystal.fmt")
    outputs = ["epw.out"] + ([f"{qe.PREFIX}.imag_iso_{t:06.2f}" for t in p["temperatures"]] if task == "epw_iso" else [])
    metadata = {"schema_version": 1, "engine": "qe", "canonical_task": task, "task": task, "executable": "epw.x", "input_file": "epw.in", "output_file": "epw.out", "input_files": list(files), "output_files": outputs, "parameters": p, "input_sha256": {name: qe._hash(folder / name) for name in files}, "spectrum": spectrum, "source_provenance": provenance or {"kind": "user_supplied_spectrum", "scientific_accuracy_validated": False}, "sources": [SOURCE], "epw_version": "5.9", "no_mpi": False, "requires_restart": False, "pseudo_files": [], **crystal}
    qe._json(folder / "qe-input.json", metadata)
    return metadata


def prepare_from_lambda(evidence_dir, destination, task, parameters, *, qe_schema):
    """Reparse native lambda.x evidence before constructing an Eliashberg job."""
    from . import qe_eph

    folder = Path(evidence_dir)
    result = qe_eph.analyze_outputs(folder, "eph_lambda")
    if not result.get("success"):
        raise ValueError("The source lambda.x calculation failed verification: " + result.get("reason", "missing evidence"))
    p = validate_parameters(parameters, task)
    if "broadening_index" not in p:
        raise ValueError("Choose broadening_index explicitly; electronic smearings must not be selected to maximize Tc.")
    index = p["broadening_index"]
    records = result["electron_phonon"]
    if index > len(records):
        raise ValueError("broadening_index exceeds the available spectra.")
    return _prepare_native(destination, task, p, spectrum_path=folder / "alpha2F.dat", qe_schema=qe_schema, unit="THz", column=index, frequency_precision=0.0001, provenance={"kind": "verified_lambda", "broadening_index": index, "electronic_broadening_ry": records[index - 1]["degauss_ry"], "source_input_sha256": qe._hash(folder / "qe-input.json"), "scientific_accuracy_validated": False})


def prepare_from_wannier(evidence_dir, destination, task, parameters, *, qe_schema):
    """Use a specified phonon-broadening column from a verified EPW spectrum."""
    from . import qe_wannier

    folder = Path(evidence_dir)
    result = qe_wannier.analyze_spectrum(folder)
    if not result.get("success"):
        raise ValueError("The source EPW spectrum failed verification: " + result.get("reason", "missing evidence"))
    p = validate_parameters(parameters, task)
    if "broadening_index" not in p:
        raise ValueError("Choose broadening_index explicitly for the native EPW spectrum.")
    source = json.loads((folder / "qe-input.json").read_text())
    spectrum_path = folder / (qe.PREFIX + ".a2f")
    data = read_native_spectrum(spectrum_path)
    count = (data.shape[1] - 1) // 2
    index = p["broadening_index"]
    if index > count:
        raise ValueError("broadening_index exceeds the available EPW spectra.")
    text = spectrum_path.read_text()
    matches = re.findall(r"Phonon smearing \(meV\)\s*\n\s*#([^\n]+)", text)
    if len(matches) != 1:
        raise ValueError("The native phonon-broadening labels are missing.")
    broadenings = qe._floats(matches[0])
    electronic = qe._floats(re.search(r"Electron smearing \(eV\)\s+(" + qe.FLOAT + r")", text)[1])[0]
    window = qe._floats(re.search(r"Fermi window \(eV\)\s+(" + qe.FLOAT + r")", text)[1])[0]
    settings = source["parameters"]
    if len(broadenings) != count or min(broadenings) <= 0 or any(b <= a for a, b in zip(broadenings, broadenings[1:])):
        raise ValueError("The phonon-broadening labels are invalid.")
    if not np.isclose(broadenings[0], settings["ph_smearing_mev"], atol=5.1e-8, rtol=0) or not np.isclose(electronic, settings["el_smearing_ev"], atol=5.1e-8, rtol=0) or not np.isclose(window, settings["fsthick_ev"], atol=5.1e-8, rtol=0):
        raise ValueError("The native spectral broadenings differ from the interpolation inputs.")
    provenance = {"kind": "verified_wannier_spectrum", "broadening_index": index,
                  "phonon_broadening_mev": broadenings[index - 1], "available_phonon_broadenings_mev": broadenings,
                  "electronic_broadening_ev": electronic, "fermi_window_ev": window,
                  "source_input_sha256": qe._hash(folder / "qe-input.json"), "scientific_accuracy_validated": False}
    meta = _prepare_native(destination, task, p, spectrum_path=spectrum_path, qe_schema=qe_schema,
                           column=index, provenance=provenance, allow_nscf=True)
    destination = Path(destination)
    for name, original in (("source-epw.in", "epw.in"), ("source-epw.out", "epw.out"), ("source-wannier.wout", qe.PREFIX + ".wout")):
        shutil.copyfile(folder / original, destination / name)
        meta["input_files"].append(name)
        meta["input_sha256"][name] = qe._hash(destination / name)
    qe._json(destination / "qe-input.json", meta)
    return meta


def prepare_inputs(structure_path, destination, task, parameters, pseudopotentials=None, *, evidence_dir=None):
    """Prepare scalar EPW from verified lambda.x or Wannier spectra."""
    from pymatgen.core import Structure

    p = validate_parameters(parameters, task)
    if task == "epw_aniso":
        return _prepare_aniso(structure_path, destination, p, evidence_dir=evidence_dir)
    if "broadening_index" not in p:
        raise ValueError("Choose broadening_index explicitly for EPW.")
    structure = qe._standard_frame(Structure.from_file(structure_path))[0]
    qe._validate_structure(structure)
    folder = Path(destination)
    if evidence_dir is None:
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "qe-input.json").exists():
            raise ValueError("Use a new folder for an EPW calculation.")
        (folder / "epw.in").write_text("DFT Agent\n! Prepared after the source spectrum has passed verification.\n")
        meta = {"schema_version": 1, "engine": "qe", "canonical_task": task, "task": task, "executable": "epw.x", "input_file": "epw.in", "output_file": "epw.out", "input_files": ["epw.in"], "output_files": ["epw.out"] + ([f"{qe.PREFIX}.imag_iso_{t:06.2f}" for t in p["temperatures"]] if task == "epw_iso" else []), "parameters": p, "preview_only": True, "requires_restart": False, "pseudo_files": [], "formula": structure.composition.reduced_formula, "number_of_atoms": len(structure), "sources": [SOURCE]}
    else:
        evidence = Path(evidence_dir)
        schema = evidence / "data-file-schema.xml"
        if not schema.is_file():
            raise ValueError("The accepted spectrum needs its source PW data-file-schema.xml.")
        actual = qe._xml_structure(qe._required(ET.parse(schema).getroot(), "output/atomic_structure"))
        if not _matches_structure_file(actual, structure, structure_path):
            raise ValueError("The source spectrum checkpoint differs from the planned structure.")
        source = json.loads((evidence / "qe-input.json").read_text())
        if source.get("canonical_task") == "epw_interpolate":
            meta = prepare_from_wannier(evidence, folder, task, p, qe_schema=schema)
        else:
            meta = prepare_from_lambda(evidence, folder, task, p, qe_schema=schema)
    structure.to(filename=folder / "structure.cif")
    meta["input_files"].append("structure.cif")
    meta["input_sha256"] = {name: qe._hash(folder / name) for name in meta["input_files"]}
    qe._json(folder / "qe-input.json", meta)
    return meta


def _prepare_aniso(structure_path, destination, parameters, *, evidence_dir=None):
    from pymatgen.core import Structure
    from . import qe_wannier

    p = validate_parameters(parameters, "epw_aniso")
    if "broadening_index" in p:
        raise ValueError("Anisotropic EPW uses the frozen interpolation checkpoint, not a scalar broadening column.")
    structure = qe._standard_frame(Structure.from_file(structure_path))[0]
    qe._validate_structure(structure)
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "qe-input.json").exists() or (folder / "epw.out").exists():
        raise ValueError("Use a new folder for an anisotropic calculation.")
    meta = {"schema_version": 1, "engine": "qe", "canonical_task": "epw_aniso", "task": "epw_aniso", "executable": "epw.x", "input_file": "epw.in", "output_file": "epw.out", "input_files": ["epw.in", "structure.cif"], "output_files": ["epw.out"] + [f"{qe.PREFIX}.imag_aniso_{t:06.2f}" for t in p["temperatures"]], "parameters": p, "requires_restart": True, "pseudo_files": [], "formula": structure.composition.reduced_formula, "number_of_atoms": len(structure), "sources": [SOURCE]}
    if evidence_dir is None:
        meta["preview_only"] = True
        text = "DFT Agent\n! Prepared after the native Wannier checkpoint has passed verification.\n"
    else:
        evidence = Path(evidence_dir)
        accepted = qe_wannier.analyze_outputs(evidence, "epw_interpolate")
        if not accepted.get("success"):
            raise ValueError("The source Wannier interpolation failed verification: " + accepted.get("reason", "missing evidence"))
        source = json.loads((evidence / "qe-input.json").read_text())
        native = source["native_structure"]
        actual = Structure(native["lattice_angstrom"], native["elements"], native["fractional_positions"])
        if not _matches_structure_file(actual, structure, structure_path):
            raise ValueError("The Wannier checkpoint does not match the requested structure.")
        checkpoint = read_checkpoint(evidence, acoustic_cutoff_cm1=source["parameters"]["acoustic_cutoff_cm1"])
        for name, original in (("source-schema.xml", "data-file-schema.xml"), ("source-epw.in", "epw.in"), ("source-epw.out", "epw.out")):
            shutil.copyfile(evidence / original, folder / name)
            meta["input_files"].append(name)
        settings = source["parameters"]
        # The inherited UPFs belong to the verified electronic calculation.
        meta["namelists"] = {"system": {"input_dft": settings["functional"]}}
        native_options = {"prefix": qe.PREFIX, "outdir": "./scratch", "ep_coupling": False, "elph": False, "epwread": True, "wannierize": False, "eliashberg": True, "laniso": True, "liso": False, "limag": True, "lpade": False, "lacon": False, "lreal": False, "fbw": False, "gridsamp": 0, "mp_mesh_k": True, "positive_matsu": True, "nqstep": settings["spectral_points"], "degaussq": settings["ph_smearing_mev"], "eps_acoustic": settings["acoustic_cutoff_cm1"], "muc": p["mu_star"], "wscut": p["frequency_cutoff_ev"], "nsiter": p["iterations"], "conv_thr_iaxis": p["convergence"], "broyden_beta": p["mixing"], "nstemp": len(p["temperatures"])}
        native_options.update({f"temps({i})": t for i, t in enumerate(p["temperatures"], 1)})
        for kind, key in (("nkf", "fine_kmesh"), ("nqf", "fine_qmesh")):
            native_options.update({f"{kind}{i}": v for i, v in enumerate(checkpoint[key], 1)})
        text = "DFT Agent\n" + qe.namelist("inputepw", native_options)
        meta.update(native_structure=native, checkpoint_sha256=checkpoint["sha256"], checkpoint_settings={k: settings[k] for k in ("spectral_points", "ph_smearing_mev", "acoustic_cutoff_cm1")}, source_provenance={"kind": "verified_wannier_interpolation", "source_input_sha256": qe._hash(evidence / "qe-input.json"), "scientific_accuracy_validated": False})
    (folder / "epw.in").write_text(text)
    structure.to(filename=folder / "structure.cif")
    meta["input_sha256"] = {name: qe._hash(folder / name) for name in meta["input_files"]}
    qe._json(folder / "qe-input.json", meta)
    return meta


# The Tc summary of each EPW release (EPW/src/supercond.f90): 6.0 renamed the simplified Allen-Dynes estimate
# the "McMillan expression", kept its formula and prints four decimals instead of six.
EPW_SUMMARIES = {(5, 9): {"omega": "Estimated w_log in Allen-Dynes Tc", "tc": "Estimated Allen-Dynes Tc", "digits": 6},
                 (6, 0): {"omega": "Estimated w_log", "tc": "Estimated Tc using McMillan expression", "digits": 4},
                 (6, 1): {"omega": "Estimated w_log", "tc": "Estimated Tc using McMillan expression", "digits": 4}}


def _epw_version(text):
    match = re.search(r"Program\s+EPW\s+v\.(\d+)\.(\d+)\b", text)
    return (int(match[1]), int(match[2])) if match else None


def _native_scalar(text, label):
    matches = re.findall(re.escape(label) + r"\s*=\s*(" + qe.FLOAT + r")", text)
    if len(matches) != 1:
        raise ValueError(f"Missing or repeated EPW result: {label}.")
    return qe._floats(matches[0])[0]


def _analyze_aniso(folder, expected_structure=None):
    from pymatgen.core import Structure

    result = {"schema_version": 1, "engine": "qe", "task": "epw_aniso", "success": False, "artifacts": [], "warnings": [], "scientific_accuracy_validated": False, "superconductivity_established": False, "pairing_model": "anisotropic phonon Migdal-Eliashberg with constant Coulomb pseudopotential"}
    try:
        meta = json.loads((folder / "qe-input.json").read_text())
        if meta.get("preview_only") or meta.get("canonical_task") != "epw_aniso":
            raise ValueError("The anisotropic calculation lacks a verified native checkpoint.")
        p = validate_parameters(meta["parameters"], "epw_aniso")
        prefix = meta.get("prefix", qe.PREFIX)
        for name, digest in {**meta["input_sha256"], **meta["checkpoint_sha256"]}.items():
            if qe._hash(folder / name) != digest:
                raise ValueError("An anisotropic input or checkpoint changed: " + name)
        checkpoint = read_checkpoint(folder, prefix, acoustic_cutoff_cm1=meta["checkpoint_settings"]["acoustic_cutoff_cm1"])
        frozen = meta["native_structure"]
        actual = qe._xml_structure(qe._required(ET.parse(folder / "source-schema.xml").getroot(), "output/atomic_structure"))
        expected = Structure(frozen["lattice_angstrom"], frozen["elements"], frozen["fractional_positions"])
        if not qe._same_structure(actual, expected) or checkpoint["modes"] != 3 * len(actual):
            raise ValueError("The anisotropic checkpoint does not match its native structure.")
        if expected_structure is not None and not _matches_structure_file(actual, Structure.from_file(expected_structure), expected_structure):
            raise ValueError("The anisotropic checkpoint differs from the requested structure.")
        text = (folder / "epw.out").read_text()
        if "Total program execution" not in text or not re.search(r"\bEPW\s*:\s*.+?CPU\s*.+?WALL", text) or "functionality-dependent EPW.bib file." not in text:
            raise ValueError("EPW did not finish normally.")
        if _epw_version(text) != (5, 9):
            raise ValueError("Anisotropic EPW output is read for EPW 5.9 only; this run used another version.")
        if re.search(r"Error in routine|MPI_ABORT|convergence.*not reached", text, re.I):
            raise ValueError("EPW reports a failed anisotropic calculation.")
        if "Solve anisotropic Eliashberg equations" not in text or "full-bandwidth" in text:
            raise ValueError("The native pairing approximation differs from the plan.")
        parts = re.split(r"temp\(\s*(\d+)\)\s*=\s*(" + qe.FLOAT + r")\s+K", text)
        if (len(parts) - 1) // 3 != len(p["temperatures"]):
            raise ValueError("The anisotropic temperature records are incomplete.")
        states = checkpoint["states"]
        energies = np.array([s["energy_minus_fermi_ev"] for s in states])
        weights = np.array([s["weight"] for s in states])
        records, state_rows = [], []
        for i, temperature in enumerate(p["temperatures"]):
            index, temp, block = parts[1 + 3 * i:4 + 3 * i]
            if int(index) != i + 1 or abs(float(temp) - temperature) > 5.1e-6:
                raise ValueError("The anisotropic temperature differs from the plan.")
            converged = re.findall(r"Convergence was reached in nsiter\s*=\s*(\d+)", block)
            rows = re.findall(r"^\s*(\d+)\s+(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s*$", block, re.M)
            if len(converged) != 1 or not rows:
                raise ValueError("A temperature lacks native anisotropic convergence evidence.")
            iteration, residual, zmean, gapmean = qe._floats(" ".join(rows[-1]))
            if iteration != int(converged[0]) or iteration > p["iterations"] or not 0 <= residual < p["convergence"]:
                raise ValueError("The anisotropic residual exceeds the planned threshold.")
            name = f"{prefix}.imag_aniso_{temperature:06.2f}"
            table = np.loadtxt(folder / name, skiprows=1, ndmin=2)
            count = _nsiw(temperature, p["frequency_cutoff_ev"])
            omega = (2 * np.arange(count) + 1) * math.pi * KB_EV * temperature
            if table.shape != (len(states) * count, 4) or not np.isfinite(table).all() or (table[:, 2] <= 0).any():
                raise ValueError("The anisotropic state table is incomplete or invalid.")
            data = table.reshape(count, len(states), 4)
            if not np.allclose(data[:, :, 0], omega[:, None], atol=1e-10, rtol=1e-9) or not np.allclose(data[:, :, 1], energies[None, :], atol=1e-8, rtol=1e-9):
                raise ValueError("Anisotropic frequencies or band energies disagree with the native checkpoint.")
            computed_z = 1 + np.sum(weights * (data[0, :, 2] - 1))
            computed_gap = np.sum(weights * data[0, :, 2] * data[0, :, 3]) / computed_z * 1000
            if not np.isclose(computed_z, zmean, atol=1e-5, rtol=1e-5) or not np.isclose(computed_gap, gapmean, atol=1e-5, rtol=1e-5):
                raise ValueError("The resolved gap table disagrees with its final native iteration.")
            records.append({"temperature_kelvin": temperature, "gap_iw0_mev": float(computed_gap), "gap_min_iw0_mev": float(data[0, :, 3].min() * 1000), "gap_max_iw0_mev": float(data[0, :, 3].max() * 1000), "renormalization_iw0": float(computed_z), "residual": residual, "iterations": int(iteration), "states": len(states), "file": name})
            for state, values in zip(states, data[0]):
                state_rows.append([temperature, state["kpoint"], state["band"], *state["k_fractional"], state["energy_minus_fermi_ev"], values[2], values[3] * 1000, state["weight"]])
        qe._csv(folder / "eliashberg_anisotropic_states.csv", ["temperature_K", "kpoint", "band", "kx", "ky", "kz", "energy_minus_fermi_eV", "Z_iw0", "delta_iw0_meV", "fermi_surface_weight"], state_rows)
        qe._csv(folder / "eliashberg_gap.csv", ["temperature_K", "delta_iw0_meV", "minimum_meV", "maximum_meV", "residual", "iterations"], [[r[k] for k in ("temperature_kelvin", "gap_iw0_mev", "gap_min_iw0_mev", "gap_max_iw0_mev", "residual", "iterations")] for r in records])
        result.update(success=True, temperature_results=records, source_provenance=meta["source_provenance"], fine_kmesh=checkpoint["fine_kmesh"], fine_qmesh=checkpoint["fine_qmesh"], artifacts=["epw.out", "eliashberg_anisotropic_states.csv", "eliashberg_gap.csv"] + [r["file"] for r in records])
    except (ValueError, KeyError, OSError, TypeError, ET.ParseError, IndexError, OverflowError) as error:
        result["reason"] = str(error)
        for name in ("eliashberg_anisotropic_states.csv", "eliashberg_gap.csv"):
            (folder / name).unlink(missing_ok=True)
    return result


def analyze_outputs(folder, task=None, expected_structure=None):
    folder = Path(folder)
    if task == "epw_aniso" or (task is None and (folder / "qe-input.json").is_file() and json.loads((folder / "qe-input.json").read_text()).get("canonical_task") == "epw_aniso"):
        return _analyze_aniso(folder, expected_structure)
    result = {"schema_version": 1, "engine": "qe", "success": False, "artifacts": [], "warnings": [], "scientific_accuracy_validated": False, "superconductivity_established": False, "pairing_model": "isotropic Migdal-Eliashberg with constant Coulomb pseudopotential"}
    try:
        meta = json.loads((folder / "qe-input.json").read_text())
        task = task or meta["canonical_task"]
        p = validate_parameters(meta["parameters"], task)
        if meta.get("preview_only"):
            raise ValueError("The EPW inputs have not been prepared from an accepted spectrum.")
        if task != meta["canonical_task"]:
            raise ValueError("The requested EPW stage does not match its inputs.")
        for name in ("epw.in", "alpha2f.dat", "source.a2f", "source-schema.xml", "crystal.fmt"):
            if qe._hash(folder / name) != meta["input_sha256"].get(name):
                raise ValueError(f"EPW input {name} changed after preparation.")
        if "structure.cif" in meta["input_sha256"] and qe._hash(folder / "structure.cif") != meta["input_sha256"]["structure.cif"]:
            raise ValueError("The EPW structure changed after preparation.")
        for name in ("source-epw.in", "source-epw.out", "source-wannier.wout"):
            if name in meta["input_sha256"] and qe._hash(folder / name) != meta["input_sha256"][name]:
                raise ValueError("The source Wannier evidence changed after preparation: " + name)
        from pymatgen.core import Structure

        native = meta["native_structure"]
        frozen = Structure(native["lattice_angstrom"], native["elements"], native["fractional_positions"])
        actual = qe._xml_structure(qe._required(ET.parse(folder / "source-schema.xml").getroot(), "output/atomic_structure"))
        if not qe._same_structure(actual, frozen):
            raise ValueError("The EPW source checkpoint differs from its frozen native structure.")
        if expected_structure is not None:
            expected = Structure.from_file(expected_structure)
            if not _matches_structure_file(actual, expected, expected_structure):
                raise ValueError("The EPW source spectrum does not match the expected structure.")
        w, a2f, spectrum = read_spectrum(folder / "alpha2f.dat")
        text = (folder / "epw.out").read_text()
        finished = "Total program execution" in text and re.search(r"\bEPW\s*:\s*.+?CPU\s*.+?WALL", text) and "functionality-dependent EPW.bib file." in text
        if not finished:
            raise ValueError("EPW did not finish normally.")
        version = _epw_version(text)
        summary = EPW_SUMMARIES.get(version)
        if summary is None:
            raise ValueError("DFT Agent reads the output of EPW " + ", ".join(f"{a}.{b}" for a, b in EPW_SUMMARIES)
                             + "; this run used " + (f"EPW {version[0]}.{version[1]}." if version else "an unknown version."))
        if re.search(r"Error in routine|MPI_ABORT|convergence.*not reached|Maximum CPU time exceeded", text, re.I):
            raise ValueError("EPW reports a failed or unconverged calculation.")
        lam = _native_scalar(text, "Electron-phonon coupling strength")
        omega = _native_scalar(text, summary["omega"])
        tc_ad = _native_scalar(text, summary["tc"])
        printed = 0.51 * 10.0 ** -summary["digits"]
        if not np.isclose(lam, spectrum["lambda"], atol=5.1e-8, rtol=1e-7) or not np.isclose(omega, spectrum["omega_log_mev"], atol=printed, rtol=1e-7):
            raise ValueError("EPW lambda or omega_log differs from the saved spectral integral.")
        expected_ad = omega / (1000 * KB_EV * 1.2) * math.exp(-1.04 * (1 + lam) / (lam - p["mu_star"] * (1 + 0.62 * lam)))
        if not np.isclose(tc_ad, expected_ad, atol=max(1e-5, printed), rtol=2e-5):
            raise ValueError("The EPW Allen-Dynes estimate differs from its stated spectrum and mu_star.")
        result.update(task=task, mu_star=p["mu_star"], lambda_integral=lam, omega_log_mev=omega, allen_dynes_estimate_kelvin=tc_ad,
                      epw_version=f"{version[0]}.{version[1]}", source_provenance=meta["source_provenance"],
                      artifacts=["epw.out", "alpha2f.dat", "source.a2f", "source-schema.xml", "crystal.fmt", "epw.in"])
        if task == "epw_tc":
            records = _linear_results(text, p)
            qe._csv(folder / "eliashberg_eigenvalues.csv", ["temperature_K", "eigenvalue", "matsubara_points", "cutoff_eV", "iterations"], [[r[k] for k in ("temperature_kelvin", "eigenvalue", "matsubara_points", "cutoff_ev", "iterations")] for r in records])
            brackets = [[a["temperature_kelvin"], b["temperature_kelvin"]] for a, b in zip(records, records[1:]) if a["eigenvalue"] >= 1 and b["eigenvalue"] < 1]
            if len(brackets) > 1:
                raise ValueError("Multiple Tc crossings require temperature and cutoff refinement.")
            result.update(temperature_results=records, tc_bracket_kelvin=brackets[0] if brackets else None, tc_bracketed=bool(brackets))
            result["artifacts"].append("eliashberg_eigenvalues.csv")
            if not brackets:
                result["warnings"].append("The temperature range does not bracket an eigenvalue crossing of one; no Tc was determined.")
        else:
            result["temperature_results"] = _gap_results(folder, text, p)
            result["artifacts"] += [r["file"] for r in result["temperature_results"]]
            qe._csv(folder / "eliashberg_gap.csv", ["temperature_K", "delta_iw0_meV", "Z_iw0", "residual", "iterations"], [[r[k] for k in ("temperature_kelvin", "gap_iw0_mev", "renormalization_iw0", "residual", "iterations")] for r in result["temperature_results"]])
            result["artifacts"].append("eliashberg_gap.csv")
        result["success"] = True
    except (ValueError, KeyError, OSError, TypeError, ET.ParseError, IndexError) as error:
        result.update(success=False, reason=str(error))
        for name in ("eliashberg_eigenvalues.csv", "eliashberg_gap.csv"):
            (folder / name).unlink(missing_ok=True)
    return result


def _linear_results(text, p):
    start = "Start: Solving (isotropic) linearized Eliashberg equation with solver = lapack"
    end = "Finish: Solving (isotropic) linearized Eliashberg equation"
    normalized = re.sub(r"[ \t]+", " ", text)
    if normalized.count(start) != 1 or normalized.count(end) != 1:
        raise ValueError("The linearized Eliashberg calculation is incomplete.")
    block = normalized.split(start)[1].split(end)[0]
    pattern = r"^\s*(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s+(\d+)\s+(" + qe.FLOAT + r")\s+(\d+)\s*$"
    rows = re.findall(pattern, block, re.M)
    if len(rows) != len(p["temperatures"]):
        raise ValueError("The Eliashberg temperature table is incomplete.")
    records = []
    for row, temperature in zip(rows, p["temperatures"]):
        temp, eigen, count, cutoff, iterations = qe._floats(" ".join(row))
        expected_count = _nsiw(temperature, p["frequency_cutoff_ev"])
        expected_cutoff = (2 * expected_count + 1) * math.pi * KB_EV * temperature
        if abs(temp - temperature) > 0.00501 or count != expected_count or abs(cutoff - expected_cutoff) > 0.000051 or iterations != 1:
            raise ValueError("The native temperature, cutoff or solver differs from the plan.")
        records.append({"temperature_kelvin": temp, "eigenvalue": eigen, "matsubara_points": int(count), "cutoff_ev": cutoff, "iterations": int(iterations)})
    return records


def _gap_results(folder, text, p):
    parts = re.split(r"temp\(\s*(\d+)\)\s*=\s*(" + qe.FLOAT + r")\s+K", text)
    if (len(parts) - 1) // 3 != len(p["temperatures"]):
        raise ValueError("The nonlinear Eliashberg temperature records are incomplete.")
    records = []
    for i, temperature in enumerate(p["temperatures"]):
        index, temp, block = parts[1 + 3 * i:4 + 3 * i]
        if int(index) != i + 1 or abs(float(temp) - temperature) > 5.1e-6:
            raise ValueError("The nonlinear Eliashberg temperatures differ from the plan.")
        converged = re.findall(r"Convergence was reached in nsiter\s*=\s*(\d+)", block)
        rows = re.findall(r"^\s*(\d+)\s+(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s*$", block, re.M)
        if len(converged) != 1 or not rows:
            raise ValueError("A temperature lacks native Eliashberg convergence evidence.")
        iteration, error, z, gap = qe._floats(" ".join(rows[-1]))
        if iteration != int(converged[0]) or iteration > p["iterations"] or not 0 <= error < p["convergence"]:
            raise ValueError("The Eliashberg residual did not meet the planned threshold.")
        path = folder / f"{qe.PREFIX}.imag_iso_{temperature:06.2f}"
        data = np.loadtxt(path, skiprows=1, ndmin=2)
        count = _nsiw(temperature, p["frequency_cutoff_ev"])
        frequencies = (2 * np.arange(count) + 1) * math.pi * KB_EV * temperature
        if data.shape != (count, 3) or not np.isfinite(data).all() or not np.allclose(data[:, 0], frequencies, atol=1e-10, rtol=1e-9) or (data[:, 1] <= 0).any():
            raise ValueError("The Matsubara gap table has invalid dimensions, units or frequencies.")
        if not np.isclose(data[0, 1], z, atol=1e-6, rtol=1e-6) or not np.isclose(data[0, 2] * 1000, gap, atol=1e-6, rtol=1e-6):
            raise ValueError("The native gap table and final iteration disagree.")
        records.append({"temperature_kelvin": temperature, "gap_iw0_mev": float(data[0, 2] * 1000), "renormalization_iw0": float(data[0, 1]), "residual": error, "iterations": int(iteration), "file": path.name})
    return records


def check_linear_eigenvalues(folder, *, tolerance=2e-6):
    """Independently diagonalize the symmetric form of EPW's native kernel."""
    from scipy.linalg import eigh

    folder = Path(folder)
    result = analyze_outputs(folder, "epw_tc")
    if not result["success"]:
        raise ValueError(result["reason"])
    data = qe._table(folder / "alpha2f.dat", 2)
    w, a2f = data[:, 0] / 1000, data[:, 1]
    dw = w[-1] / len(w)
    checks = []
    for row in result["temperature_results"]:
        count = row["matsubara_points"]
        indices = np.arange(-count + 1, count)
        spacing = 2 * math.pi * KB_EV * row["temperature_kelvin"]
        differences = np.arange(2 * count - 1) * spacing
        couplings = 2 * dw * np.sum(w[:, None] * a2f[:, None] / (w[:, None]**2 + differences[None, :]**2), axis=0)
        kernel = couplings[np.abs(indices[:, None] - indices[None, :])]
        signs = np.sign(2 * indices + 1)
        diagonal = np.sum(kernel * signs[:, None] * signs[None, :], axis=1)
        denominator = np.sqrt(np.abs(2 * indices + 1))
        matrix = (kernel - result["mu_star"] - np.diag(diagonal)) / denominator[:, None] / denominator[None, :]
        values, vectors = eigh(matrix, subset_by_index=[len(indices) - 1, len(indices) - 1])
        eigen, vector = float(values[0]), vectors[:, 0]
        residual = float(np.linalg.norm(matrix @ vector - eigen * vector))
        difference = abs(eigen - row["eigenvalue"])
        checks.append({"temperature_kelvin": row["temperature_kelvin"], "native_eigenvalue": row["eigenvalue"], "independent_eigenvalue": eigen, "absolute_difference": difference, "eigenpair_residual": residual, "passed": difference <= tolerance and residual <= tolerance})
    return {"passed": all(row["passed"] for row in checks), "tolerance": tolerance, "checks": checks, "source": SOURCE + "/supercond_iso.f90"}


def check_nonlinear_equations(folder, *, tolerance=1e-5):
    """Check saved gaps against the finite-grid nonlinear equations."""
    folder = Path(folder)
    result = analyze_outputs(folder, "epw_iso")
    if not result["success"]:
        raise ValueError(result["reason"])
    data = qe._table(folder / "alpha2f.dat", 2)
    w, a2f = data[:, 0] / 1000, data[:, 1]
    dw = w[-1] / len(w)
    checks = []
    for row in result["temperature_results"]:
        table = np.loadtxt(folder / row["file"], skiprows=1, ndmin=2)
        omega, z, gap = table.T
        count = len(omega)
        pi_t = math.pi * KB_EV * row["temperature_kelvin"]
        differences = np.arange(2 * count) * 2 * pi_t
        coupling = 2 * dw * np.sum(w[:, None] * a2f[:, None] / (w[:, None]**2 + differences[None, :]**2), axis=0)
        indices = np.arange(count)
        minus = coupling[np.abs(indices[:, None] - indices[None, :])]
        plus = coupling[indices[:, None] + indices[None, :] + 1]
        denominator = np.sqrt(omega**2 + gap**2)
        predicted_z = 1 + pi_t / omega * ((minus - plus) @ (omega / denominator))
        predicted_gap = pi_t / predicted_z * ((minus + plus - 2 * result["mu_star"]) @ (gap / denominator))
        z_error = float(np.max(np.abs(predicted_z - z)))
        gap_error = float(np.sum(np.abs(predicted_gap - gap)) / max(np.sum(np.abs(gap)), 1e-12))
        checks.append({"temperature_kelvin": row["temperature_kelvin"], "max_z_residual": z_error,
                       "relative_gap_residual": gap_error, "passed": z_error <= tolerance and gap_error <= tolerance})
    return {"passed": all(row["passed"] for row in checks), "tolerance": tolerance, "checks": checks, "source": SOURCE + "/supercond_iso.f90"}
