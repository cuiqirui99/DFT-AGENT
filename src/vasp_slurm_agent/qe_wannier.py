"""Native QE/EPW Wannier interpolation with explicit grids and projections."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import numpy as np
from pymatgen.core import Structure

from . import qe

TASKS = ("wannier_scf", "wannier_ph", "wannier_nscf", "epw_interpolate")
PROGRAMS = dict(zip(TASKS, ("pw.x", "ph.x", "pw.x", "epw.x")))
PARAMETER_KEYS = frozenset({"mesh", "wannier_mesh", "qmesh", "fine_kmesh", "fine_qmesh", "nbnd", "num_wann", "projections", "dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max", "wannier_iterations", "fsthick_ev", "el_smearing_ev", "ph_smearing_mev", "spectral_points", "acoustic_cutoff_cm1", "ecutwfc", "ecutrho", "conv_thr", "tr2_ph", "degauss", "smearing", "functional"})
SOURCE = "https://github.com/QEF/q-e/tree/qe-7.4.1/test-suite/epw_super"


def validate_parameters(parameters):
    p = dict(parameters or {})
    unknown = set(p) - PARAMETER_KEYS
    if unknown:
        raise ValueError("Unsupported Wannier settings: " + ", ".join(sorted(unknown)))
    for key in ("mesh", "wannier_mesh", "qmesh", "fine_kmesh", "fine_qmesh", "nbnd", "num_wann", "projections", "dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max", "fsthick_ev", "el_smearing_ev", "ph_smearing_mev"):
        if key not in p:
            raise ValueError(f"Specify {key} for Wannier interpolation.")
    for key in ("mesh", "wannier_mesh", "qmesh", "fine_kmesh", "fine_qmesh"):
        p[key] = qe._vector(p[key], key, integers=True)
    if any(k % q for k, q in zip(p["wannier_mesh"], p["qmesh"])) or any(k % q for k, q in zip(p["mesh"], p["qmesh"])):
        raise ValueError("The SCF and Wannier meshes must contain the coarse q mesh.")
    if any(k % q for k, q in zip(p["fine_kmesh"], p["fine_qmesh"])):
        raise ValueError("The fine k mesh must contain the fine q mesh.")
    for key in ("nbnd", "num_wann"):
        p[key] = qe._integer(p[key], key)
    if p["num_wann"] > p["nbnd"]:
        raise ValueError("num_wann cannot exceed nbnd.")
    if not isinstance(p["projections"], list) or not p["projections"] or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.,:=()+ -]+", v) or not v.strip() for v in p["projections"]):
        raise ValueError("Supply Wannier projection strings as a nonempty list.")
    for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
        p[key] = qe._number(p[key], key)
    if not p["dis_win_min"] <= p["dis_froz_min"] < p["dis_froz_max"] <= p["dis_win_max"] or p["dis_win_min"] == p["dis_win_max"]:
        raise ValueError("The frozen energy window must lie within the outer window (eV).")
    for key in ("fsthick_ev", "el_smearing_ev", "ph_smearing_mev"):
        p[key] = qe._number(p[key], key, True)
    for key, default in (("wannier_iterations", 500), ("spectral_points", 500)):
        p[key] = qe._integer(p.get(key, default), key, 10)
    for key, default in (("acoustic_cutoff_cm1", 5), ("conv_thr", 1e-9), ("tr2_ph", 1e-12), ("degauss", .01)):
        p[key] = qe._number(p.get(key, default), key, True)
    if p["acoustic_cutoff_cm1"] > 10:
        raise ValueError("Use at most 10 cm-1 for the acoustic allowance; investigate larger imaginary modes.")
    for key in ("ecutwfc", "ecutrho"):
        if key in p:
            p[key] = qe._number(p[key], key, True)
    p["functional"] = str(p.get("functional", "PBE")).upper()
    if p["functional"] not in {"PBE", "LDA"}:
        raise ValueError("This Wannier route supports nonmagnetic scalar PBE or LDA.")
    p["smearing"] = p.get("smearing", "mv")
    if p["smearing"] not in {"mv", "mp", "gaussian", "fd"}:
        raise ValueError("Choose mv, mp, gaussian or fd smearing.")
    return p


def _require_nc(potentials):
    if not potentials or any(not isinstance(v, dict) or v.get("pseudo_type", "").upper() not in {"NC", "NCPP"} or not v.get("sha256") or v.get("fully_relativistic") or str(v.get("relativistic", "scalar")).lower() in {"full", "fully-relativistic"} for v in potentials.values()):
        raise ValueError("Wannier interpolation requires verified scalar-relativistic norm-conserving UPFs.")


def _epw_namelist(values):
    values = dict(values)
    projections = {key: values.pop(key) for key in list(values) if re.fullmatch(r"proj\(\d+\)", key)}
    for value in projections.values():
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.,:=()+ -]+", value):
            raise ValueError("Invalid Wannier projection string.")
    text = qe.namelist("inputepw", values)
    return text[:-2] + "".join(f"  {key} = '{value}'\n" for key, value in projections.items()) + "/\n"


def prepare_inputs(structure_path, destination, task, parameters, pseudopotentials=None, *, evidence_dir=None):
    if task not in TASKS:
        raise ValueError("Unknown Wannier stage.")
    p = validate_parameters(parameters)
    folder = Path(destination)
    structure = Structure.from_file(structure_path)
    qe._validate_structure(structure)
    if task in {"wannier_scf", "wannier_nscf"}:
        _require_nc(pseudopotentials)
        native_task = "scf" if task == "wannier_scf" else "nscf"
        native = {k: v for k, v in p.items() if k in {"ecutwfc", "ecutrho", "conv_thr", "degauss", "functional", "nbnd"}}
        native.update(mesh=p["mesh"] if native_task == "scf" else p["wannier_mesh"], electronic_type="metal", namelists={"control": {"disk_io": "low"}})
        native["namelists"]["system"] = {"smearing": p["smearing"]}
        meta = qe.prepare_inputs(structure_path, folder, native_task, native, pseudopotentials)
        if native_task == "nscf":
            text = (folder / "pw.in").read_text()
            grid = p["wannier_mesh"]
            points = [np.array(index) / grid for index in np.ndindex(*grid)]
            card = "K_POINTS crystal\n" + str(len(points)) + "\n" + "".join(" ".join(f"{x:.16g}" for x in [*point, 1 / len(points)]) + "\n" for point in points)
            text, replacements = re.subn(r"K_POINTS automatic\s*\n[^\n]*\n", card, text)
            if replacements != 1:
                raise ValueError("The NSCF input lacks its uniform k-point card.")
            (folder / "pw.in").write_text(text)
            meta["explicit_kpoints"] = [point.tolist() for point in points]
        stem = "pw"
    else:
        folder.mkdir(parents=True, exist_ok=True)
        structure.to(filename=folder / "structure.cif")
        meta = {"schema_version": 1, "engine": "qe", "number_of_atoms": len(structure), "formula": structure.composition.reduced_formula, "pseudo_files": [], "native_structure": {"lattice_angstrom": structure.lattice.matrix.tolist(), "elements": [site.specie.symbol for site in structure], "fractional_positions": structure.frac_coords.tolist()}}
        if task == "wannier_ph":
            stem = "ph"
            values = {"prefix": qe.PREFIX, "outdir": "./scratch", "fildyn": qe.PREFIX + ".dyn", "fildvscf": "dvscf", "ldisp": True, "trans": True, "tr2_ph": p["tr2_ph"], **{f"nq{i}": n for i, n in enumerate(p["qmesh"], 1)}}
            text = "DFT Agent\n" + qe.namelist("inputph", values)
            meta["phonon_grid"] = p["qmesh"]
        else:
            stem = "epw"
            values = {"prefix": qe.PREFIX, "outdir": "./scratch", "dvscf_dir": "save", "ep_coupling": True, "elph": True, "epbwrite": True, "epbread": False, "epwwrite": True, "epwread": False, "wannierize": True, "nbndsub": p["num_wann"], "num_iter": p["wannier_iterations"], "etf_mem": 1, "vme": "dipole", "ephwrite": True, "mp_mesh_k": True, "eliashberg": True, "liso": False, "laniso": False, "fsthick": p["fsthick_ev"], "degaussw": p["el_smearing_ev"], "degaussq": p["ph_smearing_mev"], "nqstep": p["spectral_points"], "eps_acoustic": p["acoustic_cutoff_cm1"]}
            values.update({k: p[k] for k in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max")})
            values.update({f"proj({i})": v for i, v in enumerate(p["projections"], 1)})
            for key, mesh in (("nk", "wannier_mesh"), ("nq", "qmesh"), ("nkf", "fine_kmesh"), ("nqf", "fine_qmesh")):
                values.update({f"{key}{i}": n for i, n in enumerate(p[mesh], 1)})
            for i, element in enumerate(dict.fromkeys(site.specie.symbol for site in structure), 1):
                from pymatgen.core import Element
                values[f"amass({i})"] = float(Element(element).atomic_mass)
            text = "DFT Agent\n" + _epw_namelist(values)
        (folder / (stem + ".in")).write_text(text)
    meta.update(canonical_task=task, task=task, parameters=p, executable=PROGRAMS[task], input_file=stem + ".in", output_file=stem + ".out", input_files=[stem + ".in", "structure.cif"], output_files=[stem + ".out"], requires_restart=task != "wannier_scf", scientific_accuracy_validated=False, sources=[SOURCE])
    if task == "epw_interpolate":
        meta["output_files"] += [qe.PREFIX + extension for extension in (".wout", ".win", ".a2f")] + ["crystal.fmt", "selecq.fmt"]
    meta["input_sha256"] = {name: qe._hash(folder / name) for name in meta["input_files"]}
    qe._json(folder / "qe-input.json", meta)
    return meta


def analyze_outputs(folder, task, expected_structure=None):
    from . import qe_epw

    folder = Path(folder)
    result = {"schema_version": 1, "engine": "qe", "task": task, "success": False, "artifacts": [], "warnings": [], "scientific_accuracy_validated": False, "superconductivity_established": False}
    try:
        if task not in TASKS:
            raise ValueError("Unknown Wannier stage.")
        meta = json.loads((folder / "qe-input.json").read_text())
        p = validate_parameters(meta["parameters"])
        if meta["canonical_task"] != task:
            raise ValueError("The Wannier stage differs from the plan.")
        for name, digest in meta["input_sha256"].items():
            if qe._hash(folder / name) != digest:
                raise ValueError("The native Wannier input changed: " + name)
        text = (folder / meta["output_file"]).read_text()
        if re.search(r"Error in routine|MPI_ABORT|convergence\s+NOT\s+achieved|Maximum CPU time exceeded", text, re.I):
            raise ValueError("QE reports an incomplete or failed Wannier stage.")
        result["artifacts"].append(meta["output_file"])
        if task in {"wannier_scf", "wannier_nscf"}:
            if "JOB DONE." not in text:
                raise ValueError("PW did not finish normally.")
            expected = qe._expected_structure(folder, meta, expected_structure)
            native_task = "scf" if task == "wannier_scf" else "nscf"
            qe._pw_result(folder, native_task, text, {**meta, "canonical_task": native_task}, expected, result)
            _require_nc(meta["pseudopotentials"])
            if native_task == "nscf":
                schema = ET.parse(folder / "data-file-schema.xml").getroot()
                bands = qe._required(schema, "output/band_structure")
                points = bands.findall("ks_energies/k_point")
                if len(points) != int(np.prod(p["wannier_mesh"])):
                    raise ValueError("NSCF did not preserve the complete Wannier grid.")
                node = qe._required(schema, "output/atomic_structure")
                cell = qe._xml_structure(node)
                alat = qe._number(node.attrib["alat"], "alat", True) * qe.BOHR_ANGSTROM
                crystal = np.array([qe._floats(point.text) for point in points]) @ cell.lattice.matrix.T / alat
                expected_k = np.array(meta["explicit_kpoints"])
                difference = crystal - expected_k
                if not np.allclose(difference - np.rint(difference), 0, atol=1e-7, rtol=0):
                    raise ValueError("NSCF k-point coordinates or order changed.")
        elif task == "wannier_ph":
            if "JOB DONE." not in text:
                raise ValueError("PH did not finish normally.")
            qe._post_result(folder, "phonon", text, meta, result)
            index = (folder / (qe.PREFIX + ".dyn0")).read_text().splitlines()
            count = int(index[1].strip())
            receipt = json.loads((folder / "checkpoint.json").read_text())
            mapping = phonon_checkpoint_mapping(receipt, count)
            result.update(qpoint_count=count, derivative_files=len(mapping))
        else:
            if not re.search(r"Program\s+EPW\s+v\.5\.9\b", text) or "Total program execution" not in text or not re.search(r"\bEPW\s*:\s*.+?CPU\s*.+?WALL", text) or "functionality-dependent EPW.bib file." not in text:
                raise ValueError("EPW 5.9 did not finish normally.")
            if "Finish writing .ephmat files" not in text or "Running Wannier90" not in text:
                raise ValueError("Native Wannier interpolation and matrix generation are incomplete.")
            actual = qe._xml_structure(qe._required(ET.parse(folder / "data-file-schema.xml").getroot(), "output/atomic_structure"))
            native = meta["native_structure"]
            frozen = Structure(native["lattice_angstrom"], native["elements"], native["fractional_positions"])
            if not qe._same_structure(actual, frozen):
                raise ValueError("The interpolated checkpoint differs from its native structure.")
            if expected_structure is not None and not qe_epw._matches_structure_file(actual, Structure.from_file(expected_structure), expected_structure):
                raise ValueError("The interpolated checkpoint differs from the planned structure.")
            wout = (folder / (qe.PREFIX + ".wout")).read_text()
            if "All done: wannier90 exiting" not in wout or "Final State" not in wout:
                raise ValueError("Wannier90 did not finish normally.")
            final = wout.rsplit("Final State", 1)[1]
            spreads = re.findall(r"WF centre and spread\s+(\d+)\s+\(\s*(.*?)\s*\)\s+(" + qe.FLOAT + r")", final)
            if len(spreads) != p["num_wann"] or [int(v[0]) for v in spreads] != list(range(1, p["num_wann"] + 1)):
                raise ValueError("The final Wannier orbital count differs from the plan.")
            values = [qe._floats(v[1].replace(",", " ")) + qe._floats(v[2]) for v in spreads]
            if any(len(v) != 4 or v[3] < 0 for v in values):
                raise ValueError("Invalid Wannier centres or spreads.")
            convergence = [qe._floats(line.split("<-- CONV")[0]) for line in wout.splitlines() if "<-- CONV" in line]
            if len(convergence) < 5 or any(len(v) != 5 or abs(v[1]) > 1e-6 or v[2] > 1e-4 for v in convergence[-5:]):
                raise ValueError("Wannier spreads or gradients have not stabilized; revise projections, windows or iterations.")
            checkpoint = qe_epw.read_checkpoint(folder, acoustic_cutoff_cm1=p["acoustic_cutoff_cm1"])
            if checkpoint["fine_kmesh"] != p["fine_kmesh"] or checkpoint["fine_qmesh"] != p["fine_qmesh"] or checkpoint["modes"] != 3 * len(actual):
                raise ValueError("Interpolated grids or mode count differ from the plan.")
            _, _, spectrum = qe_epw.read_spectrum(folder / (qe.PREFIX + ".a2f"))
            if spectrum["points"] != p["spectral_points"]:
                raise ValueError("The spectral grid differs from the plan.")
            lam = qe_epw._native_scalar(text, "Electron-phonon coupling strength")
            if not np.isclose(lam, spectrum["lambda"], atol=1e-6, rtol=1e-6):
                raise ValueError("The native coupling differs from its spectrum integral.")
            qe._csv(folder / "wannier_centres.csv", ["orbital", "x_A", "y_A", "z_A", "spread_A2"], [[i, *v] for i, v in enumerate(values, 1)])
            result.update(lambda_integral=lam, omega_log_mev=spectrum["omega_log_mev"], fine_kmesh=p["fine_kmesh"], fine_qmesh=p["fine_qmesh"], matrix_values=checkpoint["matrix_values"], wannier_spread_A2=sum(v[3] for v in values), checkpoint_sha256=checkpoint["sha256"])
            result["artifacts"] += [qe.PREFIX + ".a2f", qe.PREFIX + ".wout", "wannier_centres.csv", *checkpoint["sha256"]]
        result.update(success=True, reason="Native outputs and numerical checks passed.")
    except (ValueError, KeyError, OSError, TypeError, ET.ParseError, IndexError, OverflowError) as error:
        result["reason"] = str(error)
        (folder / "wannier_centres.csv").unlink(missing_ok=True)
    qe._json(folder / "qe-result.json", result)
    return result


def phonon_checkpoint_mapping(records, qpoint_count):
    """Copy only verified derivatives into EPW's separate save directory."""
    count = qe._integer(qpoint_count, "Phonon q-point count")
    if not isinstance(records, dict):
        raise ValueError("Phonon checkpoint records must be a mapping.")
    for name in records:
        path = PurePosixPath(name)
        if path.is_absolute() or path.as_posix() != name or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe phonon checkpoint member.")
    prefix = qe.PREFIX
    mapping = {}
    for index in range(1, count + 1):
        dyn = f"{prefix}.dyn{index}"
        if dyn not in records:
            raise ValueError(f"Missing phonon matrix {dyn}.")
        mapping[dyn] = f"save/{prefix}.dyn_q{index}"
        subfolder = "scratch/_ph0" if index == 1 else f"scratch/_ph0/{prefix}.q_{index}"
        candidates = [f"{subfolder}/{prefix}.dvscf{suffix}" for suffix in ("1", "")]
        present = [name for name in candidates if name in records]
        if len(present) != 1:
            raise ValueError(f"Expected one native dvscf file for q point {index}.")
        mapping[present[0]] = f"save/{prefix}.dvscf_q{index}"
    source = f"scratch/_ph0/{prefix}.phsave/"
    members = sorted(name for name in records if name.startswith(source))
    if source + "control_ph.xml" not in members or not any("patterns." in name for name in members):
        raise ValueError("The phonon checkpoint lacks its control or perturbation patterns.")
    for name in members:
        relative = name[len(source):]
        if "/" in relative:
            raise ValueError("Unexpected nested phonon metadata.")
        mapping[name] = f"save/{prefix}.phsave/{relative}"
    return mapping
