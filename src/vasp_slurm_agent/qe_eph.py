"""Conventional electron-phonon workflow from QE PHonon/example03.

The Tc expression is the simplified Allen-Dynes/McMillan expression in
QE 7.4.1 PHonon/PH/lambda.f90, without strong-coupling correction factors.
It is an estimate for phonon-mediated pairing, not a superconductivity finding.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from . import qe

TASKS = ("eph_dense_scf", "eph_scf", "eph_ph", "eph_q2r", "eph_matdyn", "eph_lambda")
PROGRAMS = dict(zip(TASKS, ("pw.x", "pw.x", "ph.x", "q2r.x", "matdyn.x", "lambda.x")))
ALIASES = {"electron_phonon": "electron_phonon"}
EPh_KEYS = frozenset({"dense_mesh", "phonon_dos_mesh", "mu_star", "el_ph_sigma", "el_ph_nsigma", "phonon_smearing_thz", "phonon_dos_points", "omega_max_thz", "eph_integration"})
_PW_KEYS = frozenset({"ecutwfc", "ecutrho", "conv_thr", "electron_maxstep", "functional", "spin", "soc", "starting_magnetization", "site_labels", "electronic_type", "degauss", "nbnd", "hubbard", "namelists", "mesh", "qmesh", "asr", "tr2_ph"})
PARAMETER_KEYS = _PW_KEYS | EPh_KEYS
RY_THZ = 3289.828  # Matches lambda.f90, including its rounding.
THZ_KELVIN = 47.9924
SOURCE = "https://github.com/QEF/q-e/blob/qe-7.4.1/PHonon/PH/lambda.f90"


def validate_parameters(parameters, nat=None):
    p = dict(parameters or {})
    unknown = set(p) - PARAMETER_KEYS
    if unknown:
        raise ValueError("Unsupported electron-phonon parameters: " + ", ".join(sorted(unknown)))
    integration = p.get("eph_integration", "interpolated")
    if integration not in {"interpolated", "q_sum"}:
        raise ValueError("Use interpolated or q_sum for eph_integration.")
    meshes = ["dense_mesh", "mesh", "qmesh"]
    if integration == "interpolated" or "phonon_dos_mesh" in p:
        meshes.append("phonon_dos_mesh")
    for key in (*meshes, "mu_star"):
        if key not in p:
            raise ValueError(f"Specify {key} for the electron-phonon calculation.")
    for key in meshes:
        p[key] = qe._vector(p[key], key, integers=True)
    if "phonon_dos_mesh" in p and min(p["phonon_dos_mesh"]) < 2:
        raise ValueError("The phonon DOS mesh needs at least two points per direction.")
    if any(d % c or d % q or c % q for d, c, q in zip(p["dense_mesh"], p["mesh"], p["qmesh"])):
        raise ValueError("The dense mesh must contain the coarse mesh and q mesh; the coarse mesh must contain the q mesh.")
    if p.get("spin", "none") != "none" or p.get("soc") or p.get("starting_magnetization") or p.get("hubbard"):
        raise ValueError("This conventional electron-phonon route requires scalar-relativistic, nonmagnetic inputs without Hubbard U.")
    if p.get("functional", "PBE").upper() not in {"PBE", "LDA"}:
        raise ValueError("This electron-phonon route supports PBE or LDA, not hybrid functionals.")
    if p.get("electronic_type", "metal") != "metal":
        raise ValueError("The electron-phonon route requires a metallic ground state.")
    p["mu_star"] = qe._number(p["mu_star"], "mu_star")
    if not 0 <= p["mu_star"] < 1:
        raise ValueError("mu_star must be between 0 and 1.")
    for key, default in (("el_ph_sigma", 0.005), ("phonon_smearing_thz", 0.12)):
        p[key] = qe._number(p.get(key, default), key, True)
    p["el_ph_nsigma"] = qe._integer(p.get("el_ph_nsigma", 10), "el_ph_nsigma")
    if p["el_ph_nsigma"] > 10:
        raise ValueError("Use at most ten electronic broadenings for lambda.x output.")
    if p["el_ph_sigma"] < 0.001 or not np.isclose(p["el_ph_sigma"] * 1000, round(p["el_ph_sigma"] * 1000), atol=1e-8, rtol=0):
        raise ValueError("el_ph_sigma must be a multiple of 0.001 Ry; lambda.x reads broadenings printed to three decimals.")
    p["asr"] = p.get("asr", "simple")
    if p["asr"] not in {"simple", "crystal", "no"}:
        raise ValueError("Use simple, crystal or no for the bulk electron-phonon acoustic sum rule.")
    p["tr2_ph"] = qe._number(p.get("tr2_ph", 1e-12), "tr2_ph", True)
    p["phonon_dos_points"] = qe._integer(p.get("phonon_dos_points", 2001), "phonon_dos_points", 10)
    if "omega_max_thz" in p:
        p["omega_max_thz"] = qe._number(p["omega_max_thz"], "omega_max_thz", True)
    if nat is not None and 3 * nat > 100:
        raise ValueError("The supplied lambda.x program supports at most 100 modes (33 atoms).")
    native = p.get("namelists", {})
    if not isinstance(native, dict) or any(key.lower() not in {"control", "system", "electrons"} for key in native):
        raise ValueError("Electron-phonon native options are limited to control, system and electrons.")
    p["namelists"] = {section: qe._extra(p, section) for section in ("control", "system", "electrons")}
    system = p["namelists"]["system"]
    forbidden = {"nspin", "noncolin", "lspinorb", "input_dft", "occupations", "smearing", "degauss", "la2f", "nosym", "noinv", "no_t_rev", "tot_magnetization", "assume_isolated"}
    if any(key in forbidden or key.startswith(("starting_magnetization", "hubbard", "constrained_magnetization", "angle")) for key in system):
        raise ValueError("Use the electron-phonon settings for functional, spin, smearing and symmetry.")
    if "disk_io" in p["namelists"]["control"] and p["namelists"]["control"]["disk_io"] not in {"low", "medium", "high"}:
        raise ValueError("Electron-phonon calculations require saved wavefunctions and charge density.")
    return p


def _pw_parameters(p, dense=False):
    result = {key: value for key, value in p.items() if key in qe.PARAMETER_KEYS}
    result.update(mesh=p["dense_mesh"] if dense else p["mesh"], electronic_type="metal")
    native = {key: dict(value) for key, value in result.get("namelists", {}).items()}
    native.setdefault("system", {})["la2f"] = dense
    result["namelists"] = native
    return result


def _index(folder, qmesh):
    path = Path(folder) / (qe.PREFIX + ".dyn0")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2 or [int(x) for x in lines[0].split()] != qmesh:
        raise ValueError("The phonon grid index does not match qmesh.")
    count = int(lines[1])
    if not 1 <= count <= 200 or len(lines) != count + 2:
        raise ValueError("Invalid q-point count; lambda.x supports at most 200 irreducible q points.")
    points = [qe._floats(line) for line in lines[2:]]
    if any(len(point) != 3 for point in points):
        raise ValueError("Incomplete q-point coordinates.")
    if not np.allclose(points[0], [0, 0, 0], atol=1e-10, rtol=0) or len({tuple(point) for point in points}) != len(points):
        raise ValueError("The phonon q-point index must start at Gamma and contain distinct points.")
    return points


def parse_elph_file(path):
    """Read the formatted elphon.f90 contract, including mode-by-mode data."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 5:
        raise ValueError("Electron-phonon file is truncated.")
    header = qe._floats(lines[0])
    if len(header) != 5:
        raise ValueError("Invalid electron-phonon file header.")
    if any(x != int(x) for x in header[3:]):
        raise ValueError("Electron-phonon dimensions must be integers.")
    nsigma, nmodes = int(header[3]), int(header[4])
    if not 1 <= nsigma <= 10 or not 1 <= nmodes <= 100:
        raise ValueError("Electron-phonon file dimensions exceed lambda.x limits.")
    w2, cursor = [], 1
    while len(w2) < nmodes:
        if cursor >= len(lines):
            raise ValueError("Electron-phonon frequencies are truncated.")
        w2.extend(qe._floats(lines[cursor]))
        cursor += 1
    if len(w2) != nmodes:
        raise ValueError("Electron-phonon frequency count differs from its header.")
    spectra = []
    for _ in range(nsigma):
        if len(lines) < cursor + 2 + nmodes:
            raise ValueError("Electron-phonon mode records are truncated.")
        broadening = re.fullmatch(r"Gaussian Broadening:\s*(" + qe.FLOAT + r")\s+Ry,\s*ngauss=\s*(-?\d+)", lines[cursor])
        dos = re.fullmatch(r"DOS\s*=\s*(" + qe.FLOAT + r")\s+states/spin/Ry/Unit Cell at Ef=\s*(" + qe.FLOAT + r")\s+eV", lines[cursor + 1])
        if not broadening or not dos:
            raise ValueError("Invalid electron-phonon broadening or DOS record.")
        cursor += 2
        modes = []
        for index in range(nmodes):
            match = re.fullmatch(r"lambda\(\s*(\d+)\)\s*=\s*(" + qe.FLOAT + r")\s+gamma=\s*(" + qe.FLOAT + r")\s+GHz", lines[cursor])
            if not match or int(match[1]) != index + 1:
                raise ValueError("Electron-phonon mode indices are incomplete.")
            lam, gamma = qe._floats(match[2])[0], qe._floats(match[3])[0]
            if lam < 0 or gamma < 0:
                raise ValueError("Electron-phonon couplings and linewidths must be nonnegative.")
            modes.append({"lambda": lam, "gamma_ghz": gamma})
            cursor += 1
        spectra.append({"degauss_ry": qe._floats(broadening[1])[0], "ngauss": int(broadening[2]), "dos_ef_states_spin_ry": qe._floats(dos[1])[0], "fermi_ev": qe._floats(dos[2])[0], "modes": modes})
    if cursor != len(lines):
        raise ValueError("Unexpected trailing electron-phonon records.")
    return {"q": header[:3], "w2_ry2": w2, "nmodes": nmodes, "nsigma": nsigma, "spectra": spectra}


def parse_dense_save(path, parameters, nat, expected_bands=None):
    """Check the complete a2fmod.f90 record, not just its first line."""
    values = qe._floats(Path(path).read_text())
    if len(values) < 6:
        raise ValueError("Dense-grid a2Fsave is incomplete.")
    nbnd, nks = values[:2]
    if nbnd != int(nbnd) or nks != int(nks) or min(nbnd, nks) < 1:
        raise ValueError("Invalid dense-grid band or k-point count.")
    nbnd, nks = int(nbnd), int(nks)
    if expected_bands is not None and nbnd != expected_bands:
        raise ValueError("a2Fsave band count differs from the accepted electronic result.")
    cursor = 2 + nbnd * nks + 3 * nks
    weights = values[cursor:cursor + nks]
    cursor += nks
    if len(weights) != nks or min(weights) <= 0 or not np.isclose(sum(weights), 2, atol=1e-7):
        raise ValueError("The nonmagnetic dense-grid k-point weights must sum to two.")
    if values[cursor:cursor + 3] != parameters["dense_mesh"]:
        raise ValueError("a2Fsave does not contain the planned dense electronic mesh.")
    cursor += 3
    nsym = values[cursor] if cursor < len(values) else 0
    if nsym != int(nsym) or not 1 <= nsym <= 48 or len(values) != cursor + 1 + int(nsym) * (9 + nat):
        raise ValueError("Dense-grid symmetry data are incomplete.")
    return {"bands": nbnd, "kpoints": nks, "symmetries": int(nsym), "sha256": qe._hash(path)}


def _qstar(path, record, sigma):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("The q-star header is incomplete.")
    header = qe._floats(lines[0])
    star = int(lines[1])
    spec = record["spectra"][sigma]
    expected = [spec["degauss_ry"], spec["fermi_ev"] / qe.RY_EV, spec["dos_ef_states_spin_ry"]]
    if len(header) != 3 or star <= 0 or not np.allclose(header, expected, atol=1.1e-6, rtol=1e-7):
        raise ValueError("The a2Fq2r broadening, Fermi level or DOS differs from its coupling file.")
    nat = record["nmodes"] // 3
    if nat * 3 != record["nmodes"]:
        raise ValueError("Electron-phonon mode count must be three times the atom count.")
    if len(lines) != 2 + star * (2 + 4 * nat * nat):
        raise ValueError("The q-star matrix count differs from its weight.")
    cursor, points = 2, []
    for _ in range(star):
        if lines[cursor] != "Dynamical  Matrix in cartesian axes":
            raise ValueError("A q-star matrix is missing.")
        match = re.fullmatch(r"q\s*=\s*\(\s*(.*?)\s*\)", lines[cursor + 1])
        point = qe._floats(match[1]) if match else []
        if len(point) != 3:
            raise ValueError("Invalid q-star coordinates.")
        points.append(point)
        cursor += 2
        for a in range(1, nat + 1):
            for b in range(1, nat + 1):
                if qe._floats(lines[cursor]) != [a, b]:
                    raise ValueError("A q-star atom-pair block is incomplete.")
                for row in lines[cursor + 1:cursor + 4]:
                    if len(qe._floats(row)) != 6:
                        raise ValueError("A q-star coupling matrix is incomplete.")
                cursor += 4
    if cursor != len(lines) or not any(np.allclose(point, record["q"], atol=1.1e-6, rtol=0) for point in points):
        raise ValueError("The q-star does not contain its declared irreducible q point.")
    if len({tuple(np.round(point, 8)) for point in points}) != star:
        raise ValueError("The q-star contains duplicate coordinates.")
    return points


def coupling_evidence(folder, parameters, *, require_real=False):
    """Verify q coordinates, star weights and all broadenings before lambda.x."""
    folder = Path(folder)
    points = _index(folder, parameters["qmesh"])
    records, weights, hashes, full_grid = [], [], {}, []
    reference = None
    for index, qpoint in enumerate(points, 1):
        relative = f"elph_dir/elph.inp_lambda.{index}"
        path = folder / relative
        record = parse_elph_file(path)
        if not np.allclose(record["q"], qpoint, atol=1.1e-6, rtol=0):
            raise ValueError("Electron-phonon q points differ from the dynamical-matrix index.")
        if record["nsigma"] != parameters["el_ph_nsigma"]:
            raise ValueError("Electron-phonon broadening count differs from the plan.")
        if any(not np.isclose(spec["degauss_ry"], (i + 1) * parameters["el_ph_sigma"], atol=1e-7, rtol=0) for i, spec in enumerate(record["spectra"])):
            raise ValueError("The electronic broadening series differs from the plan.")
        if require_real and min(record["w2_ry2"]) < 0:
            raise ValueError("Imaginary phonon frequencies prevent a physical lambda.x/Tc estimate; converge and inspect the structure first.")
        if any(s["dos_ef_states_spin_ry"] <= 0 for s in record["spectra"]):
            raise ValueError("A finite positive electronic DOS at the Fermi level is required.")
        signature = [{key: value for key, value in spec.items() if key != "modes"} for spec in record["spectra"]]
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Broadening/DOS records disagree between q points.")
        hashes[relative] = qe._hash(path)
        stars = []
        for sigma in range(1, record["nsigma"] + 1):
            relative = f"elph_dir/a2Fq2r.{50 + sigma}.{index}"
            path = folder / relative
            stars.append(_qstar(path, record, sigma - 1))
            hashes[relative] = qe._hash(path)
        if any(star != stars[0] for star in stars[1:]):
            raise ValueError("q-star coordinates or weights disagree across electronic broadenings.")
        weights.append(len(stars[0]))
        full_grid.extend(stars[0])
        records.append(record)
    if sum(weights) != math.prod(parameters["qmesh"]):
        raise ValueError("q-star weights do not cover the complete q mesh.")
    if len({tuple(np.round(point, 8)) for point in full_grid}) != len(full_grid):
        raise ValueError("The irreducible q-star files duplicate points in the full mesh.")
    if len({record["nmodes"] for record in records}) != 1:
        raise ValueError("Electron-phonon mode counts differ between q points.")
    hashes[qe.PREFIX + ".dyn0"] = qe._hash(folder / (qe.PREFIX + ".dyn0"))
    return {"qpoints": points, "weights": weights, "records": records, "sha256": hashes}


def prepare_inputs(structure_path, destination, task, parameters, pseudopotentials=None, *, evidence_dir=None):
    if task not in TASKS:
        raise ValueError("Unknown electron-phonon stage.")
    structure = Structure.from_file(structure_path)
    p = validate_parameters(parameters, len(structure))
    destination = Path(destination)
    if task in {"eph_dense_scf", "eph_scf"}:
        metadata = qe.prepare_inputs(structure_path, destination, "scf", _pw_parameters(p, task == "eph_dense_scf"), pseudopotentials)
        text = (destination / "pw.in").read_text()
        stem = "pw"
    else:
        destination.mkdir(parents=True, exist_ok=True)
        structure.to(filename=destination / "structure.cif")
        metadata = {"schema_version": 1, "engine": "qe", "formula": structure.composition.reduced_formula, "number_of_atoms": len(structure), "source_sha256": qe._hash(structure_path), "pseudo_files": []}
        if task == "eph_ph":
            stem = "ph"
            options = {"prefix": qe.PREFIX, "outdir": "./scratch", "fildyn": qe.PREFIX + ".dyn", "fildvscf": "dft_agent.dvscf", "electron_phonon": "interpolated", "el_ph_sigma": p["el_ph_sigma"], "el_ph_nsigma": p["el_ph_nsigma"], "trans": True, "ldisp": True, "tr2_ph": p.get("tr2_ph", 1e-12), **{f"nq{i + 1}": n for i, n in enumerate(p["qmesh"])}}
            text = "DFT Agent electron-phonon coupling\n" + qe.namelist("inputph", options)
            metadata["phonon_grid"] = p["qmesh"]
        elif task == "eph_q2r":
            stem = "q2r"
            text = qe.namelist("input", {"fildyn": qe.PREFIX + ".dyn", "flfrc": "force_constants", "zasr": p.get("asr", "simple"), "la2f": True, "el_ph_nsigma": p["el_ph_nsigma"]})
        elif task == "eph_matdyn":
            stem = "matdyn"
            text = qe.namelist("input", {"flfrc": "force_constants", "flfrq": "phonon.freq", "fldos": "phonon.dos", "asr": p.get("asr", "simple"), "la2f": True, "dos": True, "el_ph_nsigma": p["el_ph_nsigma"], "ndos": p["phonon_dos_points"], **{f"nk{i + 1}": n for i, n in enumerate(p["phonon_dos_mesh"])}})
        else:
            stem = "lambda"
            metadata.update(no_mpi=True, requires_evidence=True)
            if evidence_dir is None:
                text = "! Prepared after the electron-phonon outputs have been verified.\n"
                metadata["preview_only"] = True
            else:
                evidence = coupling_evidence(evidence_dir, p, require_real=True)
                if evidence["records"][0]["nmodes"] != 3 * len(structure):
                    raise ValueError("Electron-phonon modes do not match the calculation structure.")
                highest = max(math.sqrt(max(record["w2_ry2"])) * RY_THZ for record in evidence["records"])
                emax = p.get("omega_max_thz", highest + 6 * p["phonon_smearing_thz"])
                if emax <= highest:
                    raise ValueError("omega_max_thz must exceed every computed phonon frequency.")
                text = f"{emax:.12g} {p['phonon_smearing_thz']:.12g} 0\n{len(evidence['qpoints'])}\n"
                text += "".join(" ".join(f"{x:.12g}" for x in [*point, weight]) + "\n" for point, weight in zip(evidence["qpoints"], evidence["weights"]))
                text += "".join(f"elph_dir/elph.inp_lambda.{i}\n" for i in range(1, len(evidence["qpoints"]) + 1))
                text += f"{p['mu_star']:.12g}\n"
                metadata.update(evidence_sha256=evidence["sha256"], qpoint_weights=evidence["weights"], omega_max_thz=emax, preview_only=False)
    metadata.update(task=task, canonical_task=task, parameters=p, executable=PROGRAMS[task], input_file=stem + ".in", output_file=stem + ".out", input_files=[stem + ".in", "structure.cif"], output_files=[stem + ".out"], requires_restart=task != "eph_dense_scf", scientific_accuracy_validated=False, sources=[SOURCE, "https://github.com/QEF/q-e/tree/qe-7.4.1/PHonon/examples/example03"])
    (destination / metadata["input_file"]).write_text(text)
    metadata["input_sha256"] = {name: qe._hash(destination / name) for name in metadata["input_files"]}
    qe._json(destination / "qe-input.json", metadata)
    return metadata


def tc_estimate(lam, omega_log_kelvin, mu_star):
    lam = qe._number(lam, "lambda", True)
    omega_log_kelvin = qe._number(omega_log_kelvin, "omega_log_kelvin", True)
    mu_star = qe._number(mu_star, "mu_star")
    if not 0 <= mu_star < 1:
        raise ValueError("Invalid mu_star.")
    denominator = lam - mu_star * (1 + 0.62 * lam)
    if denominator <= 0:
        raise ValueError("The Tc expression is outside its domain: lambda − mu* (1 + 0.62 lambda) must be positive.")
    return omega_log_kelvin / 1.2 * math.exp(-1.04 * (1 + lam) / denominator)


def parse_lambda_output(folder, metadata):
    folder = Path(folder)
    p = metadata["parameters"]
    evidence = coupling_evidence(folder, p, require_real=True)
    if evidence["records"][0]["nmodes"] != 3 * metadata["number_of_atoms"]:
        raise ValueError("Electron-phonon mode count does not match the structure.")
    if metadata.get("evidence_sha256") != evidence["sha256"]:
        raise ValueError("Electron-phonon evidence changed after the lambda input was prepared.")
    data = np.loadtxt(folder / "lambda.dat", comments="#", ndmin=2)
    spectrum = qe._table(folder / "alpha2F.dat", 2)
    if not np.isfinite(data).all() or data.shape != (p["el_ph_nsigma"], 5) or spectrum.shape[1] != 1 + len(data) or len(spectrum) != 2000:
        raise ValueError("lambda.x output dimensions differ from the requested broadenings.")
    if np.any(np.diff(spectrum[:, 0]) <= 0) or np.any(spectrum[:, 1:] < 0):
        raise ValueError("alpha2F requires increasing frequencies and nonnegative weights.")
    if abs(spectrum[0, 0]) > 1e-7 or abs(spectrum[-1, 0] - metadata["omega_max_thz"]) > 0.00011:
        raise ValueError("The alpha2F frequency grid differs from the requested range.")
    if not np.allclose(spectrum[:, 0], np.linspace(0, metadata["omega_max_thz"], 2000), atol=0.000051, rtol=0):
        raise ValueError("The alpha2F frequency grid is not the uniform lambda.x grid.")
    weights = np.asarray(evidence["weights"]) / sum(evidence["weights"])
    weighted = [sum(weight * sum(mode["lambda"] for mode in rec["spectra"][sigma]["modes"]) for rec, weight in zip(evidence["records"], weights)) for sigma in range(len(data))]
    x = spectrum[1:, 0]
    delta = metadata["omega_max_thz"] / 1999
    stdout = (folder / "lambda.out").read_text()
    table = stdout.split("T_c")[-1] if "T_c" in stdout else ""
    tc_rows = re.findall(r"^\s*(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s+(" + qe.FLOAT + r")\s*$", table, re.M)
    if len(tc_rows) != len(data):
        raise ValueError("lambda.x did not produce the complete Tc table.")
    rows = []
    for i, (sigma, lam, integral, omega, dos) in enumerate(data[:, :5]):
        expected_sigma = evidence["records"][0]["spectra"][i]["degauss_ry"]
        if abs(sigma - expected_sigma) > 1e-7 or abs(lam - weighted[i]) > 1.1e-6 or abs(dos - evidence["records"][0]["spectra"][i]["dos_ef_states_spin_ry"]) > 1.1e-6:
            raise ValueError("lambda.dat does not match the weighted mode couplings.")
        if min(lam, integral, omega, dos) <= 0:
            raise ValueError("lambda, omega_log and DOS must be finite and positive.")
        calculated = float(2 * delta * np.sum(spectrum[1:, i + 1] / x))
        calculated_omega = float(np.exp(2 * delta * np.sum(spectrum[1:, i + 1] * np.log(x) / x) / calculated) * THZ_KELVIN)
        if not np.isclose(calculated, integral, rtol=0.02, atol=0.002) or not np.isclose(calculated_omega, omega, rtol=0.02, atol=0.1):
            raise ValueError("alpha2F integrals disagree with lambda.dat beyond printed precision.")
        if not np.isclose(integral, lam, rtol=0.1, atol=0.002):
            raise ValueError("Spectral and mode-summed lambda differ by more than 10%; refine the spectral broadening/grid before reporting Tc.")
        tc = tc_estimate(lam, omega, p["mu_star"])
        printed = qe._floats(" ".join(tc_rows[i]))
        if not np.allclose(printed, [lam, omega, tc], rtol=0, atol=0.0011):
            raise ValueError("The printed Tc does not match the source-defined formula.")
        rows.append({"degauss_ry": float(sigma), "lambda": float(lam), "lambda_spectral_integral": float(integral), "omega_log_kelvin": float(omega), "mu_star": p["mu_star"], "tc_estimate_kelvin": tc, "spectral_lambda_relative_difference": abs(integral - lam) / lam})
    return rows, spectrum


def parse_interpolated_spectra(folder, parameters, nat):
    """Cross-check matdyn's Ry spectra against its independent mode sums."""
    folder = Path(folder)
    summary = (folder / "lambda").read_text()
    pattern = r"Broadening\s+(" + qe.FLOAT + r")\s+lambda\s+(" + qe.FLOAT + r")\s+dos\(Ef\)\s*(" + qe.FLOAT + r")(?:\s+omega_ln \[K\]\s*(" + qe.FLOAT + r"))?"
    matches = re.findall(pattern, summary)
    if len(matches) != parameters["el_ph_nsigma"]:
        raise ValueError("The interpolated lambda broadening series is incomplete.")
    rows, artifacts = [], []
    for i, values in enumerate(matches, 1):
        sigma, lam, dos = map(float, values[:3])
        if min(lam, dos) <= 0 or not np.isclose(sigma, i * parameters["el_ph_sigma"], atol=5.1e-5, rtol=0):
            raise ValueError("Invalid interpolated lambda, DOS or electronic broadening.")
        name = f"a2F.dos{i}"
        lines = (folder / name).read_text().splitlines()
        footer = [line for line in lines if line.lstrip().startswith("lambda")]
        footer_match = re.fullmatch(r"\s*lambda\s*=\s*(" + qe.FLOAT + r")\s+Delta\s*=\s*(" + qe.FLOAT + r")\s*", footer[0]) if len(footer) == 1 else None
        if footer_match is None:
            raise ValueError("The interpolated alpha2F integral record is missing.")
        reported, delta = qe._floats(" ".join(footer_match.groups()))
        numeric = [line for line in lines if line.strip() and not line.lstrip().startswith(("#", "lambda"))]
        data = np.asarray([qe._floats(line) for line in numeric])
        if data.shape != (parameters["phonon_dos_points"], 2 + 3 * nat) or not np.isfinite(data).all():
            raise ValueError("Interpolated alpha2F data are incomplete.")
        energy, spectrum = data[:, 0], data[:, 1]
        if min(energy) <= 0 or delta <= 0:
            raise ValueError("Interpolated alpha2F frequencies must form a positive regular grid in Ry.")
        # matdyn.f90 writes E16.6: adjacent printed differences lose precision
        # on a dense grid. Compare coordinates with their six-digit rounding.
        rounding = 0.5 * 10.0 ** (np.floor(np.log10(energy)) - 5)
        expected_grid = energy[0] + np.arange(len(energy)) * delta
        if np.any(np.abs(energy - expected_grid) > rounding + rounding[0] + 1e-14):
            raise ValueError("Interpolated alpha2F frequencies must form a positive regular grid in Ry.")
        if data[:, 1:].min() < -1e-10:
            raise ValueError("Negative interpolated electron-phonon weights prevent a Tc estimate; refine the electronic and phonon grids.")
        if not np.allclose(spectrum, data[:, 2:].sum(axis=1), rtol=2e-5, atol=1e-9):
            raise ValueError("Interpolated alpha2F mode weights disagree with the total spectrum.")
        integrated = float(2 * delta * np.sum(spectrum / energy))
        if not np.isclose(integrated, reported, rtol=2e-5, atol=1e-6):
            raise ValueError("The interpolated alpha2F integral differs from its reported value.")
        # This discrepancy is exposed rather than presented as a converged estimate.
        rows.append({"degauss_ry": sigma, "lambda": lam, "lambda_spectral_integral": integrated, "dos_ef_states_spin_ry": dos, "spectral_lambda_relative_difference": abs(integrated - lam) / lam, "omega_log_kelvin": float(values[3]) if values[3] else None})
        csv_name = f"alpha2F_interpolated_{i}.csv"
        qe._csv(folder / csv_name, ["frequency_ry", "alpha2F_total"] + [f"mode_{j + 1}" for j in range(3 * nat)], data)
        artifacts += [name, csv_name]
    return rows, artifacts


def analyze_outputs(folder, task, expected_structure=None):
    folder = Path(folder)
    result = {"schema_version": 1, "engine": "qe", "task": task, "success": False, "artifacts": [], "warnings": [], "scientific_accuracy_validated": False, "superconductivity_established": False}
    for pattern in ("electron_phonon.csv", "alpha2F.csv", "alpha2F.png", "alpha2F.pdf", "alpha2F.svg", "alpha2F_interpolated_*.csv"):
        for file in folder.glob(pattern):
            file.unlink()
    try:
        if task not in TASKS:
            raise ValueError("Unknown electron-phonon stage.")
        metadata = json.loads((folder / "qe-input.json").read_text())
        p = validate_parameters(metadata["parameters"], metadata.get("number_of_atoms"))
        if metadata.get("canonical_task") != task or metadata.get("preview_only"):
            raise ValueError("Electron-phonon stage metadata is incomplete or inconsistent.")
        stdout = folder / metadata["output_file"]
        text = stdout.read_text()
        if task != "eph_lambda" and "JOB DONE." not in text:
            raise ValueError("QE did not finish normally.")
        if re.search(r"Error in routine|MPI_ABORT|convergence NOT achieved|Maximum CPU time exceeded", text, re.I):
            raise ValueError("QE reports an incomplete or failed calculation.")
        result["artifacts"].append(stdout.name)
        if task in {"eph_dense_scf", "eph_scf"}:
            expected = qe._expected_structure(folder, metadata, expected_structure)
            base_metadata = {**metadata, "canonical_task": "scf"}
            qe._pw_result(folder, "scf", text, base_metadata, expected, result)
            # Coarse SCF must preserve the dense mesh and the same band count.
            file = folder / (qe.PREFIX + ".a2Fsave")
            if not file.is_file():
                file = folder / "scratch" / (qe.PREFIX + ".a2Fsave")
            result["dense_grid"] = parse_dense_save(file, p, len(expected), result.get("band_count"))
            result["artifacts"].append(str(file.relative_to(folder)))
        elif task == "eph_ph":
            qe._post_result(folder, "phonon", text, metadata, result)
            evidence = coupling_evidence(folder, p)
            if evidence["records"][0]["nmodes"] != 3 * metadata["number_of_atoms"]:
                raise ValueError("Electron-phonon mode count does not match the structure.")
            result.update(qpoint_count=len(evidence["qpoints"]), qpoint_weights=evidence["weights"], imaginary_mode_count=sum(sum(w < 0 for w in rec["w2_ry2"]) for rec in evidence["records"]))
            result["artifacts"] += list(evidence["sha256"])
        elif task == "eph_q2r":
            qe._post_result(folder, "q2r", text, metadata, result)
            for i in range(1, p["el_ph_nsigma"] + 1):
                name = f"elph_dir/a2Fmatdyn.{60 + i}"
                if not (folder / name).is_file() or (folder / name).stat().st_size < 50:
                    raise ValueError("Interpolated electron-phonon force constants are missing.")
                result["artifacts"].append(name)
        elif task == "eph_matdyn":
            dos = qe._table(folder / "phonon.dos", 2)
            if np.any(dos[:, 1] < -1e-10) or np.any(np.diff(dos[:, 0]) <= 0):
                raise ValueError("Phonon DOS contains negative density.")
            if np.any((dos[:, 0] < -1) & (dos[:, 1] > 1e-10)):
                raise ValueError("Interpolated imaginary phonon modes prevent a Tc estimate.")
            result["artifacts"] += ["phonon.dos", "lambda"]
            rows, artifacts = parse_interpolated_spectra(folder, p, metadata["number_of_atoms"])
            result["interpolated_lambda"] = rows
            result["artifacts"] += artifacts
            if any(row["spectral_lambda_relative_difference"] > 0.1 for row in rows):
                raise ValueError("Interpolated spectral and mode-summed lambda differ by more than 10%; refine the phonon DOS grid.")
        else:
            rows, spectrum = parse_lambda_output(folder, metadata)
            qe._csv(folder / "electron_phonon.csv", list(rows[0]), ([row[key] for key in rows[0]] for row in rows))
            qe._csv(folder / "alpha2F.csv", ["frequency_thz"] + [f"alpha2F_sigma_{row['degauss_ry']}_ry" for row in rows], spectrum)
            result["artifacts"] += ["lambda.dat", "alpha2F.dat", "electron_phonon.csv", "alpha2F.csv"] + qe._plot(folder, "alpha2F", spectrum[:, 0], spectrum[:, 1:], "Frequency (THz)", "α²F(ω)")
            result.update(electron_phonon=rows, integration_method="q-point mode sum with spectral broadening", tc_model="simplified Allen-Dynes/McMillan; f1=f2=1", tc_model_source=SOURCE, mu_star=p["mu_star"], scope="Conventional isotropic phonon-mediated pairing estimate. Mesh, broadening and pseudopotential convergence have not been established.")
            result["warnings"].append("Inspect the full broadening series; no single value has been selected as a prediction.")
        result["artifacts"] = list(dict.fromkeys(result["artifacts"]))
        result.update(success=True, reason="Required electron-phonon outputs passed consistency checks.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, StopIteration, OverflowError, qe.ET.ParseError) as exc:
        result["reason"] = str(exc) or "Electron-phonon output is truncated."
    qe._json(folder / "qe-result.json", result)
    return result
