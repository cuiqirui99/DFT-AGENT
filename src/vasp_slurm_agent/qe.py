"""Quantum ESPRESSO inputs and evidence-based result checks.

Input energies use Ry; schema XML energies use Hartree. Public results use eV.
References: https://www.quantum-espresso.org/Doc/INPUT_PW.html and the
corresponding INPUT_PH, INPUT_Q2R, INPUT_MATDYN, INPUT_NEB descriptions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import Poscar

from .numerical_defaults import vacuum_axes

RY_EV = 13.605693122994
HA_EV = 2 * RY_EV
BOHR_ANGSTROM = 0.529177210903
HA_BOHR3_GPA = HA_EV / BOHR_ANGSTROM**3 * 160.2176634
PREFIX = "dft_agent"
ALIASES = {"vc-relax": "relax", "ph": "phonon", "projwfc": "pdos", "matdyn": "phonon_bands", "charge_density": "pp"}
TASKS = frozenset({"relax", "scf", "bands", "nscf", "dos", "pdos", "phonon", "q2r", "phonon_bands", "neb", "pp", *ALIASES})
EXECUTABLES = {"relax": "pw.x", "scf": "pw.x", "bands": "pw.x", "nscf": "pw.x", "dos": "dos.x", "pdos": "projwfc.x", "phonon": "ph.x", "q2r": "q2r.x", "phonon_bands": "matdyn.x", "neb": "neb.x", "pp": "pp.x"}
FILENAMES = {"relax": "pw", "scf": "pw", "bands": "pw", "nscf": "pw", "dos": "dos", "pdos": "projwfc", "phonon": "ph", "q2r": "q2r", "phonon_bands": "matdyn", "neb": "neb", "pp": "pp"}
PARAMETER_KEYS = frozenset({"ecutwfc", "ecutrho", "conv_thr", "electron_maxstep", "functional", "spin", "soc", "starting_magnetization", "site_labels", "electronic_type", "degauss", "nbnd", "cell_relax", "nstep", "etot_conv_thr", "forc_conv_thr", "mesh", "kpoint_grid", "kspacing", "kpoints", "line_density", "hubbard", "hubbard_projector", "namelists", "deltae", "tr2_ph", "qmesh", "q_mesh", "phonon_grid", "qpoint", "qpoints", "asr", "images", "num_of_images", "nstep_path", "path_thr"})
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value, label, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive." if positive else "."))
    return number


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _vector(value, label, integers=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} needs three numbers.")
    return [(_integer(x, label) if integers else _number(x, label)) for x in value]


def _validate_structure(structure):
    if not len(structure) or not structure.is_ordered or structure.volume <= 1e-6:
        raise ValueError("A nonempty ordered structure with a valid cell is required.")
    if not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.frac_coords).all():
        raise ValueError("Structure contains nonfinite coordinates.")
    if not structure.is_valid(tol=0.5):
        raise ValueError("Structure contains overlapping atoms.")
    if any(not getattr(site.specie, "Z", 0) for site in structure):
        raise ValueError("Every site must identify an element.")


def _literal(value):
    if type(value) is bool:
        return ".true." if value else ".false."
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, "Namelist value")
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.+() -]+", value):
        return "'" + value + "'"
    raise ValueError("Namelist values must be finite numbers, booleans or simple single-line strings.")


def namelist(name, values):
    """Serialize typed options; no raw Fortran or shell fragments are accepted."""
    lines = [f"&{name.upper()}"]
    for key, value in values.items():
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*(?:\(\d+(?:,\d+)*\))?", key):
            raise ValueError(f"Invalid namelist key: {key!r}")
        # These exact fixed paths are owned by the workflow.
        literal = f"'{value}'" if key in {"outdir", "pseudo_dir"} and value in {"./scratch", "./pseudo"} else _literal(value)
        lines.append(f"  {key} = {literal},")
    return "\n".join(lines + ["/", ""])


_PROTECTED = {"prefix", "outdir", "pseudo_dir", "wfcdir", "calculation", "ibrav", "nat", "ntyp", "celldm", "a", "b", "c", "cosab", "cosac", "cosbc", "fildyn", "flfrc", "flfrq", "flvec", "fldos", "filpdos", "filproj", "filband", "restart_mode", "num_of_images"}


def _extra(parameters, section):
    all_sections = parameters.get("namelists", {})
    if not isinstance(all_sections, dict):
        raise ValueError("namelists must map section names to typed options.")
    if len({key.lower() for key in all_sections}) != len(all_sections):
        raise ValueError("QE namelist sections must not be repeated with different capitalization.")
    allowed = {"control", "system", "electrons", "ions", "cell", "inputph", "dos", "projwfc", "q2r", "matdyn", "path"}
    if any(key.lower() not in allowed for key in all_sections):
        raise ValueError("Unknown QE namelist section.")
    values = next((value for key, value in all_sections.items() if key.lower() == section), {})
    if not isinstance(values, dict):
        raise ValueError(f"{section} options must be a dictionary.")
    values = {key.lower(): value for key, value in values.items()}
    for key, value in values.items():
        base = key.split("(")[0]
        if section == "system" and (base.startswith("lda_plus_u") or base.startswith("hubbard_")):
            raise ValueError("Use the Hubbard card settings (label, manifold, U and optional J0), not legacy native Hubbard options.")
        if base in _PROTECTED or base.startswith("fil") or base.endswith("dir"):
            raise ValueError(f"{key} is managed by the QE workflow.")
        _literal(value)
    return values


def validate_upf(path, element=None, require_soc=False):
    """Read only header metadata; accept UPF 2 XML and legacy header text."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    header = re.search(r"<PP_HEADER\b([^>]*)(?:/>|>(.*?)</PP_HEADER>)", text, re.S)
    if not header:
        raise ValueError("Not a UPF pseudopotential: PP_HEADER is missing.")
    attrs = dict(re.findall(r'([A-Za-z_]+)\s*=\s*[\"\']([^\"\']*)[\"\']', header[1]))
    found = attrs.get("element", "").strip()
    if not found:
        legacy = re.search(r"^\s*([A-Z][a-z]?)\s+Element\b", header[2] or "", re.M | re.I)
        found = legacy[1] if legacy else ""
    if not found or (element is not None and found != element):
        raise ValueError(f"UPF element mismatch: expected {element}, found {found or 'unknown'}.")
    full = attrs.get("relativistic", "").strip().lower() == "full" or attrs.get("has_so", "").strip().lower() in {"t", "true", ".true."}
    if require_soc and not full:
        raise ValueError(f"SOC requires a verified fully relativistic UPF for {found}.")
    legacy_lines = [line.strip() for line in (header[2] or "").splitlines() if line.strip()]
    pseudo_type = attrs.get("pseudo_type", "").strip().upper()
    if attrs.get("is_paw", "").lower() in {"t", "true", ".true."}:
        pseudo_type = "PAW"
    elif attrs.get("is_ultrasoft", "").lower() in {"t", "true", ".true."}:
        pseudo_type = "US"
    functional = attrs.get("functional", "").strip()
    if not attrs and len(legacy_lines) >= 5:
        pseudo_type, functional = legacy_lines[2].split()[0].upper(), legacy_lines[4]
    result = {"filename": Path(path).name, "element": found, "fully_relativistic": full, "sha256": _hash(path),
              "functional": functional, "pseudo_type": pseudo_type}
    for source, target in (("wfc_cutoff", "ecutwfc"), ("rho_cutoff", "ecutrho")):
        if attrs.get(source):
            value = float(attrs[source])
            if value > 0:
                result[target] = value
    return result


def _potentials(structure, mapping, soc):
    result = {}
    for element in dict.fromkeys(site.specie.symbol for site in structure):
        value = mapping.get(element)
        entry = dict(value) if isinstance(value, dict) else {"filename": value}
        filename = entry.get("filename", entry.get("file"))
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\.[uU][pP][fF]", filename):
            raise ValueError(f"Provide a UPF basename for {element}.")
        if entry.get("element", element) != element:
            raise ValueError(f"UPF element does not match {element}.")
        if soc and not (entry.get("fully_relativistic") is True or entry.get("relativistic") == "full"):
            raise ValueError(f"SOC requires verified fully relativistic UPF metadata for {element}.")
        result[element] = {**entry, "filename": filename, "element": element}
    return result


def _path(structure, parameters, key="kpoints"):
    supplied = parameters.get(key)
    if supplied is not None:
        if not isinstance(supplied, list) or len(supplied) < 2:
            raise ValueError(f"{key} must contain at least two fractional reciprocal coordinates.")
        points = np.asarray([_vector(point, key) for point in supplied])
        cart = structure.lattice.reciprocal_lattice.get_cartesian_coords(points)
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))]
        return {"reciprocal_fractional": points.tolist(), "distance_inv_angstrom": distance.tolist(), "labels": [""] * len(points), "segments": [[0, len(points)]], "method": "explicit"}
    import seekpath
    density = _integer(parameters.get("line_density", 60), "line_density", 2)
    path = seekpath.get_explicit_k_path_orig_cell((structure.lattice.matrix, structure.frac_coords, [site.specie.Z for site in structure]), reference_distance=1.0 / density, with_time_reversal=not bool(parameters.get("soc") or parameters.get("starting_magnetization")))
    return {"reciprocal_fractional": np.asarray(path["explicit_kpoints_rel"]).tolist(), "distance_inv_angstrom": np.asarray(path["explicit_kpoints_linearcoord"]).tolist(), "labels": path["explicit_kpoints_labels"], "segments": path["explicit_segments"], "method": "seekpath_hpkot_orig_cell"}


def _species(structure, parameters):
    labels = parameters.get("site_labels", [site.specie.symbol for site in structure])
    if not isinstance(labels, (list, tuple)) or len(labels) != len(structure):
        raise ValueError("site_labels must have one species label per atom.")
    species = {}
    for site, label in zip(structure, labels):
        if not isinstance(label, str) or not re.fullmatch(r"[A-Z][a-z]?[A-Za-z0-9_]*", label) or len(label) > 3:
            raise ValueError("QE species labels must be at most three letters or digits, e.g. Fe1.")
        element = site.specie.symbol
        if label in species and species[label] != element:
            raise ValueError("Each species label must identify one element.")
        species[label] = element
    return list(labels), species


def _positions(structure, labels):
    return "ATOMIC_POSITIONS crystal\n" + "\n".join(f"{label} " + " ".join(f"{x:.14g}" for x in site.frac_coords) for label, site in zip(labels, structure)) + "\n"


def validate_compatibility(task, parameters, potentials=None, *, verified=False):
    """Reject combinations excluded by the pinned QE 7.4.1 native routines."""
    cell_relax = task == "vc-relax" or (task == "relax" and parameters.get("cell_relax"))
    task = ALIASES.get(task, task)
    system = _extra(parameters, "system")
    control = _extra(parameters, "control")
    functional = str(system.get("input_dft", parameters.get("functional", "PBE"))).upper()
    if functional not in {"PBE", "LDA", "PZ", "PBE_SOL", "PBESOL", "HSE", "HSE06", "PBE0"}:
        raise ValueError("This QE preview supports PBE, LDA, PBEsol, HSE06 and PBE0. Other input_dft choices need separate compatibility checks.")
    hybrid = functional in {"HSE", "HSE06", "PBE0", "B3LYP"}
    soc = bool(system.get("lspinorb", parameters.get("soc", False)))
    noncollinear = bool(system.get("noncolin", parameters.get("soc") or parameters.get("spin") == "noncollinear"))
    # PW/src/exx.f90:3474 excludes noncollinear exact-exchange stress.
    if hybrid and noncollinear and (cell_relax or control.get("tstress", False)):
        raise ValueError("QE 7.4.1 does not support stress or cell relaxation for noncollinear hybrid calculations; use a fixed cell without stress.")
    hubbard = parameters.get("hubbard", [])
    projector = parameters.get("hubbard_projector", "ortho-atomic")
    phonon = task in {"phonon", "q2r", "phonon_bands", "matdyn"}
    # PHonon/PH/phq_readin.f90:754-791; changing a Hubbard projector would
    # change the meaning of U, so ask for a compatible recipe instead.
    if phonon and hybrid:
        raise ValueError("QE 7.4.1 does not support hybrid-functional phonons.")
    if phonon and hubbard:
        if projector != "atomic":
            raise ValueError("QE 7.4.1 DFT+U phonons require atomic projectors and U values chosen for them.")
        labels = [entry.get("label") for entry in hubbard if isinstance(entry, dict)]
        if len(labels) != len(set(labels)):
            raise ValueError("QE 7.4.1 phonons support only one Hubbard channel per species.")
    if phonon and str(system.get("constrained_magnetization", "none")).lower() != "none":
        raise ValueError("QE 7.4.1 phonons do not support constrained magnetization.")
    seeds = dict(parameters.get("starting_magnetization") or {})
    seeds.update({key: value for key, value in system.items() if key.startswith("starting_magnetization(")})
    magnetic_phonon = phonon and noncollinear and any(abs(_number(value, "starting_magnetization")) > 1e-6 for value in seeds.values())
    # Modules/read_cards.f90:3048-3052.
    if noncollinear and any(isinstance(entry, dict) and _number(entry.get("J0", 0), "J0") > 0 for entry in (hubbard or [])):
        raise ValueError("QE 7.4.1 does not support J0 with noncollinear spin or SOC.")
    # PW/src/force_hub.f90:78-80 and stres_hub.f90:72-74.
    if hubbard and projector == "pseudo" and task in {"scf", "relax", "nscf", "bands", "neb"}:
        if task in {"relax", "neb"} or control.get("tprnfor", True) or control.get("tstress", True):
            raise ValueError("QE 7.4.1 pseudo Hubbard projectors do not support forces or stress; use atomic/ortho-atomic projectors, or disable both outputs for a fixed-geometry calculation.")
    needs_nc = hybrid and (task in {"relax", "neb"} or noncollinear or control.get("tprnfor", False))
    for potential in (potentials or {}).values():
        if not isinstance(potential, dict):
            continue
        pseudo_type = str(potential.get("pseudo_type", "")).upper()
        full_relativistic = (potential.get("fully_relativistic") is True or potential.get("has_so") is True
                             or potential.get("relativistic") == "full")
        # PW/src/average_pp.f90:31-34: only NC potentials without Hubbard U
        # can be averaged from fully relativistic data when SOC is disabled.
        if full_relativistic and not soc and (hubbard or pseudo_type in {"US", "USPP", "PAW"}
                                             or (verified and pseudo_type not in {"NC", "NCPP"})):
            raise ValueError("QE 7.4.1 requires SOC for fully relativistic ultrasoft/PAW UPFs or DFT+U. Use scalar-relativistic UPFs for this no-SOC calculation.")
        # phq_readin.f90:793-800; PW/setup.f90:217-218 defines domag.
        if magnetic_phonon and pseudo_type == "PAW":
            raise ValueError("QE 7.4.1 noncollinear magnetic phonons do not support PAW UPFs.")
        if needs_nc and (pseudo_type in {"US", "USPP", "PAW"} or (verified and pseudo_type not in {"NC", "NCPP"})):
            raise ValueError("QE 7.4.1 hybrid forces, relaxation and noncollinear hybrid calculations require verified norm-conserving UPFs.")
        # INPUT_PW.def:1227-1233: input_dft overrides the UPF functional.
        # PBE-generated UPFs are allowed for PBE hybrids; LDA/PBE mixing is not.
        tokens = set(re.findall(r"[A-Z0-9]+", str(potential.get("functional", "")).upper()))
        family = "PBE" if tokens & {"PBE", "PBX", "PBC"} else "LDA" if tokens in ({"LDA"}, {"PZ"}) or {"NOGX", "NOGC"} <= tokens else None
        expected = "LDA" if functional in {"LDA", "PZ"} else "PBE" if functional in {"PBE", "PBE0", "HSE", "HSE06"} else None
        if family and expected and family != expected:
            raise ValueError(f"The {functional} calculation requires compatible {expected} UPFs; a {family} UPF was selected.")


def _pw(structure, task, p, potentials, metadata, include_positions=True):
    labels, species = _species(structure, p)
    ecut = _number(p.get("ecutwfc", max([60.0] + [float(v.get("ecutwfc", 0)) for v in potentials.values()])), "ecutwfc", True)
    rho = _number(p.get("ecutrho", max([8 * ecut] + [float(v.get("ecutrho", 0)) for v in potentials.values()])), "ecutrho", True)
    if rho < ecut:
        raise ValueError("ecutrho must be at least ecutwfc (both in Ry).")
    control = {"calculation": "vc-relax" if task == "relax" and p.get("cell_relax") else task, "prefix": PREFIX, "outdir": "./scratch", "pseudo_dir": "./pseudo", "tprnfor": True, "tstress": True, "verbosity": "high"}
    system = {"ibrav": 0, "nat": len(structure), "ntyp": len(species), "ecutwfc": ecut, "ecutrho": rho}
    electrons = {"conv_thr": _number(p.get("conv_thr", 1e-8), "conv_thr", True), "electron_maxstep": _integer(p.get("electron_maxstep", 120), "electron_maxstep")}
    functional = p.get("functional", "PBE").upper()
    if functional not in {"PBE", "LDA", "PBE0", "HSE", "HSE06"}:
        raise ValueError("Use a supported functional or SYSTEM input_dft in native namelists.")
    system["input_dft"] = "HSE" if functional == "HSE06" else functional
    if functional == "HSE06":
        system.update(exx_fraction=0.25, screening_parameter=0.106)
    if task in {"bands", "nscf"} and functional in {"HSE", "HSE06", "PBE0"}:
        raise ValueError("Hybrid band paths/NSCF require a separately validated EXX interpolation workflow; use SCF or relax.")
    spin = p.get("spin", "none")
    soc = p.get("soc", False)
    if type(soc) is not bool or spin not in {"none", "collinear", "noncollinear"}:
        raise ValueError("spin must be none, collinear or noncollinear; soc must be boolean.")
    if soc or spin == "noncollinear":
        system["noncolin"] = True
        system["lspinorb"] = soc
    elif spin == "collinear":
        system["nspin"] = 2
    magnetization = p.get("starting_magnetization", {})
    if not isinstance(magnetization, dict) or set(magnetization) - set(species):
        raise ValueError("starting_magnetization must map species labels to fractions in [-1, 1].")
    if spin == "collinear" and not magnetization and not any(k.startswith("starting_magnetization(") for k in _extra(p, "system")):
        raise ValueError("Specify starting_magnetization for a collinear calculation.")
    for label, value in magnetization.items():
        value = _number(value, "starting_magnetization")
        if abs(value) > 1:
            raise ValueError("starting_magnetization is a fraction of valence electrons, in [-1, 1].")
        system[f"starting_magnetization({list(species).index(label) + 1})"] = value
    electronic_type = p.get("electronic_type", "auto")
    if electronic_type not in {"auto", "metal", "insulator"}:
        raise ValueError("electronic_type must be auto, metal or insulator.")
    if electronic_type == "insulator":
        system["occupations"] = "fixed"
    else:
        system.update(occupations="smearing", smearing="mv" if electronic_type == "metal" else "gaussian", degauss=_number(p.get("degauss", 0.01), "degauss", True))
    if "nbnd" in p:
        system["nbnd"] = _integer(p["nbnd"], "nbnd")
    if task in {"bands", "nscf"}:
        electrons["startingpot"] = "file"
    if task == "bands":
        system.update(nosym=True, noinv=True)
    if task == "relax":
        control.update(nstep=_integer(p.get("nstep", 100), "nstep"), etot_conv_thr=_number(p.get("etot_conv_thr", 1e-5), "etot_conv_thr", True), forc_conv_thr=_number(p.get("forc_conv_thr", 1e-3), "forc_conv_thr", True))
    for name, section in (("control", control), ("system", system), ("electrons", electrons)):
        section.update(_extra(p, name))
    if str(system.get("input_dft", "")).upper() in {"HSE", "HSE06", "PBE0", "B3LYP"}:
        known_nc = bool(potentials) and all(isinstance(value, dict) and str(value.get("pseudo_type", "")).upper() in {"NC", "NCPP"} for value in potentials.values())
        if task == "scf" and not known_nc:
            for flag in ("tprnfor", "tstress"):
                if flag not in _extra(p, "control"):
                    control[flag] = False
    if (system.get("noncolin") and str(system.get("input_dft", "")).upper() in {"HSE", "HSE06", "PBE0", "B3LYP"}
            and "tstress" not in _extra(p, "control")):
        control["tstress"] = False
    if system.get("lspinorb"):
        if system.get("noncolin") is not True or system.get("nspin", 1) != 1:
            raise ValueError("SOC requires noncolin=.true. without nspin=2.")
        _potentials(structure, potentials, True)
    if system.get("noncolin") and system.get("nspin", 1) != 1:
        raise ValueError("Do not combine nspin=2 with noncolin.")
    if any(key.startswith("starting_magnetization(") for key in system) and system.get("nspin", 1) != 2 and not system.get("noncolin"):
        raise ValueError("Starting magnetization requires collinear or noncollinear spin.")
    if task in {"bands", "nscf"} and str(system.get("input_dft", "")).upper() in {"HSE", "HSE06", "PBE0", "B3LYP"}:
        raise ValueError("Hybrid NSCF/bands are not supported by this workflow.")
    _number(system["ecutwfc"], "ecutwfc", True)
    _number(system["ecutrho"], "ecutrho", True)
    if system["ecutrho"] < system["ecutwfc"]:
        raise ValueError("ecutrho must be at least ecutwfc (both in Ry).")
    _number(electrons["conv_thr"], "conv_thr", True)
    text = namelist("control", control) + namelist("system", system) + namelist("electrons", electrons)
    if task == "relax":
        text += namelist("ions", {"ion_dynamics": "bfgs", **_extra(p, "ions")})
        if p.get("cell_relax"):
            text += namelist("cell", {"cell_dynamics": "bfgs", **_extra(p, "cell")})
    text += "ATOMIC_SPECIES\n" + "\n".join(f"{label} {float(Element(element).atomic_mass):.10g} {potentials[element]['filename']}" for label, element in species.items()) + "\n"
    text += "CELL_PARAMETERS angstrom\n" + "\n".join(" ".join(f"{x:.14g}" for x in vector) for vector in structure.lattice.matrix) + "\n"
    if include_positions:
        text += _positions(structure, labels)
    if task == "bands":
        if str(system.get("occupations", "")).startswith("tetrahedra"):
            raise ValueError("Tetrahedra occupations cannot be used on a band path.")
        path = _path(structure, p)
        metadata["band_path"] = path
        points = path["reciprocal_fractional"]
        text += f"K_POINTS crystal\n{len(points)}\n" + "\n".join(" ".join(f"{x:.14g}" for x in point) + " 1" for point in points) + "\n"
    else:
        mesh = p.get("mesh", p.get("kpoint_grid"))
        if mesh is None:
            spacing = _number(p.get("kspacing", 0.25), "kspacing", True)
            mesh = [max(1, math.ceil(x / spacing)) for x in structure.lattice.reciprocal_lattice.abc]
            for axis in vacuum_axes(structure):
                mesh[axis["axis"]] = 1
        mesh = _vector(mesh, "mesh", integers=True)
        metadata["mesh"] = mesh
        text += "K_POINTS automatic\n" + " ".join(map(str, mesh)) + " 0 0 0\n"
        if system.get("occupations", "").startswith("tetrahedra") and min(mesh) < 2:
            raise ValueError("Tetrahedra occupations require a three-dimensional mesh.")
    hubbard = p.get("hubbard", [])
    if hubbard:
        if not isinstance(hubbard, list):
            raise ValueError("hubbard must be a list of label, manifold, U and optional J0 dictionaries (eV).")
        projector = p.get("hubbard_projector", "ortho-atomic")
        if projector not in {"atomic", "ortho-atomic", "pseudo"}:
            raise ValueError("Unsupported Hubbard projector.")
        text += f"HUBBARD ({projector})\n"
        for item in hubbard:
            if not isinstance(item, dict) or item.get("label") not in species or not re.fullmatch(r"[1-7][spdf]", str(item.get("manifold", ""))) or "U" not in item:
                raise ValueError("Each Hubbard entry requires a species label, manifold (e.g. 3d), and explicit U in eV.")
            for key in ("U", "J0"):
                if key in item:
                    value = _number(item[key], key)
                    if value < 0:
                        raise ValueError("Hubbard U and J0 must be nonnegative.")
                    text += f"{key} {item['label']}-{item['manifold']} {value:.12g}\n"
    metadata.update(namelists={"control": control, "system": system, "electrons": electrons}, site_labels=labels, species=species, energy_input_unit="Ry", force_input_unit="Ry/bohr", hubbard_unit="eV")
    return text


def prepare_inputs(structure_path, destination, task, parameters, pseudopotentials=None):
    """Write one native QE stage; the workflow supplies its accepted checkpoint."""
    if task not in TASKS:
        raise ValueError(f"Unsupported QE task: {task}")
    original_task = task
    task = ALIASES.get(task, task)
    p = dict(parameters or {})
    unknown = set(p) - PARAMETER_KEYS
    if unknown:
        raise ValueError(f"Unsupported QE parameters: {', '.join(sorted(unknown))}")
    if not isinstance(p.get("namelists", {}), dict):
        raise ValueError("namelists must map section names to typed options.")
    for old, new in (("q_mesh", "qmesh"), ("phonon_grid", "qmesh")):
        if old in p:
            if new in p and p[new] != p[old]:
                raise ValueError(f"{old} and {new} disagree.")
            p[new] = p.pop(old)
    if original_task == "vc-relax":
        p["cell_relax"] = True
    structure = Structure.from_file(structure_path)
    _validate_structure(structure)
    # Validate every native section even when it is not used in this stage.
    for section in p.get("namelists", {}):
        _extra(p, section.lower())
    metadata = {"schema_version": 1, "engine": "qe", "task": original_task, "canonical_task": task, "formula": structure.composition.reduced_formula, "number_of_atoms": len(structure), "nat": len(structure), "parameters": p, "source_sha256": _hash(structure_path), "requires_restart": task in {"bands", "nscf", "dos", "pdos", "phonon", "q2r", "phonon_bands", "pp"}, "prefix": PREFIX, "outdir": "scratch", "scientific_accuracy_validated": False}
    metadata["native_structure"] = {"lattice_angstrom": structure.lattice.matrix.tolist(),
                                    "elements": [site.specie.symbol for site in structure],
                                    "fractional_positions": structure.frac_coords.tolist()}
    potentials = _potentials(structure, pseudopotentials or {}, bool(p.get("soc"))) if task in {"relax", "scf", "bands", "nscf", "neb"} else {}
    validate_compatibility(task, p, potentials)
    metadata["pseudopotentials"] = potentials
    metadata["pseudo_files"] = list(dict.fromkeys(v["filename"] for v in potentials.values()))
    if task in {"relax", "scf", "bands", "nscf"}:
        text = _pw(structure, task, p, potentials, metadata)
    elif task == "dos":
        text = namelist("dos", {"prefix": PREFIX, "outdir": "./scratch", "fildos": "dos.dat", "deltae": _number(p.get("deltae", 0.01), "deltae", True), **_extra(p, "dos")})
    elif task == "pdos":
        text = namelist("projwfc", {"prefix": PREFIX, "outdir": "./scratch", "filpdos": "pdos", "deltae": _number(p.get("deltae", 0.01), "deltae", True), **_extra(p, "projwfc")})
    elif task == "pp":
        text = namelist("inputpp", {"prefix": PREFIX, "outdir": "./scratch", "filplot": "charge_density.dat", "plot_num": 0})
        text += namelist("plot", {"nfile": 1, "filepp(1)": "charge_density.dat", "weight(1)": 1.0, "iflag": 3, "output_format": 6, "fileout": "charge_density.cube"})
    elif task == "phonon":
        values = {"prefix": PREFIX, "outdir": "./scratch", "fildyn": PREFIX + ".dyn", "tr2_ph": _number(p.get("tr2_ph", 1e-12), "tr2_ph", True)}
        qmesh = p.get("qmesh")
        if qmesh is not None:
            qmesh = _vector(qmesh, "qmesh", integers=True)
            values.update(ldisp=True, **{f"nq{i + 1}": n for i, n in enumerate(qmesh)})
        else:
            values["ldisp"] = False
        values.update(_extra(p, "inputph"))
        if values.get("only_init") or values.get("trans") is False or any(k in values for k in ("start_irr", "last_irr", "start_q", "last_q", "nat_todo", "modenum")):
            raise ValueError("Partial phonon runs cannot satisfy the full-stage acceptance check.")
        if values.get("ldisp") and qmesh is None:
            raise ValueError("Provide qmesh for a phonon dispersion.")
        metadata["phonon_grid"] = qmesh
        text = "DFT Agent phonons\n" + namelist("inputph", values)
        if not values["ldisp"]:
            text += " ".join(map(str, _vector(p.get("qpoint", [0, 0, 0]), "qpoint"))) + "\n"
    elif task == "q2r":
        text = namelist("input", {"fildyn": PREFIX + ".dyn", "flfrc": "force_constants", "zasr": p.get("asr", "simple"), **_extra(p, "q2r")})
    elif task == "phonon_bands":
        path = _path(structure, p, "qpoints")
        metadata["phonon_path"] = path
        points = path["reciprocal_fractional"]
        text = namelist("input", {"flfrc": "force_constants", "flfrq": "phonon.freq", "asr": p.get("asr", "simple"), "q_in_cryst_coord": True, **_extra(p, "matdyn")})
        text += f"{len(points)}\n" + "\n".join(" ".join(f"{x:.14g}" for x in point) for point in points) + "\n"
    else:
        images = p.get("images")
        if not isinstance(images, list) or len(images) < 2:
            raise ValueError("NEB requires at least two image structure paths in parameters.images.")
        image_structures = [Structure.from_file(path) for path in images]
        for image in image_structures:
            _validate_structure(image)
            if [s.specie.symbol for s in image] != [s.specie.symbol for s in structure] or not np.allclose(image.lattice.matrix, structure.lattice.matrix, atol=1e-7, rtol=1e-7):
                raise ValueError("NEB images must share the cell, atom count, species and site order.")
        count = _integer(p.get("num_of_images", max(5, len(images))), "num_of_images", 3)
        if count < len(images):
            raise ValueError("num_of_images cannot be smaller than the supplied image count.")
        path_options = {"restart_mode": "from_scratch", "string_method": "neb", "nstep_path": _integer(p.get("nstep_path", 100), "nstep_path"), "num_of_images": count, "opt_scheme": "broyden", "path_thr": _number(p.get("path_thr", 0.05), "path_thr", True), **_extra(p, "path")}
        metadata.update(neb_num_images=count, neb_path_threshold_ev_angstrom=path_options["path_thr"], image_sha256=[_hash(x) for x in images])
        text = "BEGIN\nBEGIN_PATH_INPUT\n" + namelist("path", path_options) + "END_PATH_INPUT\nBEGIN_ENGINE_INPUT\n"
        text += _pw(structure, "scf", p, potentials, metadata, include_positions=False)
        text += "BEGIN_POSITIONS\n"
        for index, image in enumerate(image_structures):
            text += ("FIRST_IMAGE" if index == 0 else "LAST_IMAGE" if index == len(images) - 1 else "INTERMEDIATE_IMAGE") + "\n"
            text += _positions(image, metadata["site_labels"])
        text += "END_POSITIONS\nEND_ENGINE_INPUT\nEND\n"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stem = FILENAMES[task]
    metadata.update(executable=EXECUTABLES[task], input_file=stem + ".in", output_file=stem + ".out", input_files=[stem + ".in", "structure.cif"], output_files=[stem + ".out"])
    (destination / metadata["input_file"]).write_text(text, encoding="utf-8")
    structure.to(filename=destination / "structure.cif")
    metadata["input_sha256"] = {name: _hash(destination / name) for name in metadata["input_files"]}
    _json(destination / "qe-input.json", metadata)
    return metadata


def _floats(text):
    values = [float(x.replace("D", "E").replace("d", "e")) for x in (text or "").split()]
    if not np.isfinite(values).all():
        raise ValueError("QE output contains nonfinite numbers.")
    return values


def _required(parent, path):
    node = parent.find(path)
    if node is None:
        raise ValueError(f"QE XML is missing {path}.")
    return node


def _xml_structure(node):
    cell = np.array([_floats(_required(node, f"cell/a{i}").text) for i in (1, 2, 3)]) * BOHR_ANGSTROM
    atoms = _required(node, "atomic_positions").findall("atom")
    labels = [atom.attrib["name"] for atom in atoms]
    species = []
    for label in labels:
        match = re.match(r"([A-Z][a-z]?)", label)
        if not match:
            raise ValueError("Invalid element label in QE XML.")
        species.append(match[1])
    coords = np.array([_floats(atom.text) for atom in atoms]) * BOHR_ANGSTROM
    if cell.shape != (3, 3) or coords.shape != (len(atoms), 3):
        raise ValueError("Incomplete cell or positions in QE XML.")
    structure = Structure(cell, species, coords, coords_are_cartesian=True)
    _validate_structure(structure)
    if int(node.attrib.get("nat", len(atoms))) != len(atoms):
        raise ValueError("QE XML atom count is inconsistent.")
    return structure


def _same_structure(actual, expected, positions=True):
    if [site.specie.symbol for site in actual] != [site.specie.symbol for site in expected]:
        return False
    if not positions:
        return True
    delta = actual.frac_coords - expected.frac_coords
    return bool(np.allclose(actual.lattice.matrix, expected.lattice.matrix, atol=1e-5, rtol=1e-6) and np.allclose(delta - np.rint(delta), 0, atol=1e-5))


def _expected_structure(folder, metadata, expected_structure=None):
    """Recover the submitted orientation and check it against the native input."""
    native = metadata.get("native_structure")
    if native is None:
        if expected_structure:
            return Structure.from_file(expected_structure)
        path = Path(folder) / "structure.cif"
        return Structure.from_file(path) if path.is_file() else None
    if not isinstance(native, dict) or set(native) != {"lattice_angstrom", "elements", "fractional_positions"}:
        raise ValueError("Invalid native structure metadata.")
    expected = Structure(native["lattice_angstrom"], native["elements"], native["fractional_positions"])
    _validate_structure(expected)
    source = Path(folder) / "pw.in"
    if not source.is_file() or _hash(source) != metadata.get("input_sha256", {}).get("pw.in"):
        raise ValueError("The native PW input changed or is missing.")
    lines = [line.strip() for line in source.read_text().splitlines()]
    if lines.count("CELL_PARAMETERS angstrom") != 1 or lines.count("ATOMIC_POSITIONS crystal") != 1:
        raise ValueError("Native PW structure cards are missing or ambiguous.")
    cell_start = lines.index("CELL_PARAMETERS angstrom") + 1
    positions_start = lines.index("ATOMIC_POSITIONS crystal") + 1
    cell = [_floats(line) for line in lines[cell_start:cell_start + 3]]
    positions = [line.split() for line in lines[positions_start:positions_start + len(expected)]]
    if (np.asarray(cell).shape != (3, 3) or len(positions) != len(expected)
            or any(len(row) != 4 for row in positions)):
        raise ValueError("Native PW structure cards are incomplete.")
    labels = [row[0] for row in positions]
    species = metadata.get("species", {})
    if labels != metadata.get("site_labels") or [species.get(label) for label in labels] != native["elements"]:
        raise ValueError("Native PW species differ from the recorded structure.")
    coords = np.asarray([_floats(" ".join(row[1:])) for row in positions])
    if (not np.allclose(cell, expected.lattice.matrix, atol=1e-10, rtol=1e-12)
            or not np.allclose(coords, expected.frac_coords, atol=1e-10, rtol=1e-12)):
        raise ValueError("Native PW cell or positions differ from the recorded structure.")
    return expected


def _csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _plot(folder, stem, x, y, xlabel, ylabel, labels=None, segments=None):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(6.6, 4.4), constrained_layout=True)
    ax = figure.subplots()
    data = np.asarray(y)
    data = data[:, None] if data.ndim == 1 else data
    for column in range(data.shape[1]):
        for start, stop in segments or [[0, len(x)]]:
            ax.plot(np.asarray(x)[start:stop], data[start:stop, column], color="#245a81", linewidth=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if labels and len(labels) == len(x):
        ticks = [(float(x[i]), label.replace("GAMMA", "Γ")) for i, label in enumerate(labels) if label]
        if ticks:
            ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.grid(axis="y", alpha=0.18)
    names = []
    for ext in ("png", "pdf", "svg"):
        name = f"{stem}.{ext}"
        figure.savefig(Path(folder) / name, dpi=180)
        names.append(name)
    return names


def _bands(output, result, folder, metadata):
    band = output.find("band_structure")
    if band is None:
        return
    fermi = band.findtext("fermi_energy")
    if fermi is not None:
        result["fermi_energy_ev"] = _floats(fermi)[0] * HA_EV
    states = band.findall("ks_energies")
    if not states:
        return
    energies, occupations, points = [], [], []
    for state in states:
        energies.append(np.asarray(_floats(_required(state, "eigenvalues").text)) * HA_EV)
        occupations.append(np.asarray(_floats(_required(state, "occupations").text)))
        points.append(_floats(_required(state, "k_point").text))
    ev = np.asarray(energies)
    occ = np.asarray(occupations)
    if ev.shape != occ.shape or ev.ndim != 2 or not ev.size:
        raise ValueError("Eigenvalues and occupations have incompatible dimensions.")
    if band.findtext("nks") and int(band.findtext("nks")) != len(ev):
        raise ValueError("QE XML is missing k-point eigenvalues.")
    declared = band.findtext("nbnd")
    if declared is None and band.findtext("nbnd_up") and band.findtext("nbnd_dw"):
        declared = int(band.findtext("nbnd_up")) + int(band.findtext("nbnd_dw"))
    if declared is not None and int(declared) != ev.shape[1]:
        raise ValueError("QE XML is missing band eigenvalues.")
    result.update(kpoint_count=len(ev), band_count=ev.shape[1])
    # QE schema occupations are normalized to one state, including spin-degenerate runs.
    full = 1.0
    if np.max(occ) > 1.001:
        full = 2.0
    partial = np.any((occ > 1e-4) & (occ < full - 1e-4))
    occupied, empty = occ > full / 2, occ <= full / 2
    if partial:
        result["band_gap"] = {"available": True, "energy_ev": 0.0, "metallic_or_smeared": True, "scope": "Sampled states with partial occupations; reduce smearing before concluding metallicity."}
    elif occupied.any() and empty.any():
        vbm, cbm = float(ev[occupied].max()), float(ev[empty].min())
        result["band_gap"] = {"available": True, "energy_ev": max(0.0, cbm - vbm), "vbm_ev": vbm, "cbm_ev": cbm, "metallic_or_smeared": cbm <= vbm, "scope": "Sampled k points only; not a converged full-zone gap."}
    else:
        result["band_gap"] = {"available": False, "reason": "Both occupied and empty bands are needed."}
    gap = result["band_gap"]
    gap["gap_ev"] = gap.get("energy_ev")
    gap["status"] = "metallic_or_partially_occupied" if gap.get("metallic_or_smeared") else "gap" if gap.get("available") else "unavailable"
    gap["note"] = gap.pop("scope", gap.get("reason", ""))
    gap["scope"] = "sampled_path" if metadata.get("canonical_task") == "bands" else "sampled_k_points"
    result["band_gap_ev"] = result["band_gap"].get("energy_ev")
    _csv(folder / "bands.csv", ["k_index", "kx_2pi_over_alat", "ky_2pi_over_alat", "kz_2pi_over_alat", "band", "energy_ev", "occupation"], ([k + 1, *points[k], b + 1, float(ev[k, b]), float(occ[k, b])] for k in range(ev.shape[0]) for b in range(ev.shape[1])))
    result["artifacts"].append("bands.csv")
    if metadata.get("canonical_task") == "bands":
        path = metadata.get("band_path", {})
        x = path.get("distance_inv_angstrom", list(range(len(ev))))
        if len(x) != len(ev):
            raise ValueError("Band output k-point count differs from the prepared path.")
        reference = result.get("fermi_energy_ev")
        if reference is None:
            reference = result["band_gap"].get("vbm_ev", 0.0)
        result["plot_reference_ev"] = reference
        result["artifacts"] += _plot(folder, "bands", x, ev - reference, "Wave vector", "Energy − reference (eV)", path.get("labels"), path.get("segments"))


def _pw_result(folder, task, text, metadata, expected, result):
    xmls = [folder / "data-file-schema.xml", folder / "scratch" / (PREFIX + ".save") / "data-file-schema.xml"]
    xml = next((path for path in xmls if path.is_file()), None)
    if xml is None:
        raise ValueError("data-file-schema.xml is missing.")
    if xml.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("QE XML exceeds the result parser size limit.")
    raw = xml.read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("XML entities are not supported.")
    root = ET.fromstring(raw)
    for element in root.iter():
        element.tag = element.tag.split("}")[-1]
    if root.findtext("exit_status") not in (None, "0"):
        raise ValueError("QE XML reports a nonzero exit status.")
    output = _required(root, "output")
    result["xml_sha256"] = _hash(xml)
    result["qe_version"] = (root.find("general_info/creator").attrib.get("VERSION") if root.find("general_info/creator") is not None else None)
    final = _xml_structure(_required(output, "atomic_structure"))
    initial_node = root.find("input/atomic_structure")
    if expected is not None:
        if initial_node is not None and not _same_structure(_xml_structure(initial_node), expected):
            raise ValueError("QE XML input structure does not match the submitted structure.")
        if not _same_structure(final, expected, positions=task != "relax"):
            raise ValueError("QE output structure does not match the submitted structure.")
    calculation = root.findtext("input/control_variables/calculation")
    desired = "vc-relax" if task == "relax" and metadata.get("parameters", {}).get("cell_relax") else task
    if calculation and calculation.strip() != desired:
        raise ValueError(f"QE XML calculation is {calculation}, expected {desired}.")
    scf_flag = output.findtext("convergence_info/scf_conv/convergence_achieved")
    result["converged_electronic"] = scf_flag is not None and scf_flag.strip().lower() in {"true", "t", ".true."}
    if task in {"bands", "nscf"}:
        # PW/src/pw_restart_new.f90 intentionally sets scf_conv=false for these.
        if not re.search(r"End of band structure calculation", text, re.I):
            raise ValueError("Band diagonalization did not finish.")
        if re.search(r"(?:c_bands|eigenvalues).*not converged", text, re.I):
            raise ValueError("Some eigenvalues did not converge.")
        result.update(converged_electronic=True, electronic_check="diagonalization; SCF convergence belongs to the parent checkpoint")
    elif not result["converged_electronic"]:
        raise ValueError("Electronic convergence is not confirmed in QE XML.")
    native_system = metadata.get("namelists", {}).get("system", _extra(metadata.get("parameters", {}), "system"))
    functional = str(native_system.get("input_dft", metadata.get("parameters", {}).get("functional", "PBE"))).upper()
    hybrid = functional in {"HSE", "HSE06", "PBE0", "B3LYP"} or output.find("dft/hybrid") is not None
    if hybrid:
        # QE 7.4.1 electrons.f90:342-346 sets conv_elec=true at the outer
        # iteration limit to write restart files. That XML flag is insufficient.
        events = list(re.finditer(r"EXX self-consistency reached|EXX convergence NOT achieved|Calculation\s*\(EXX\)\s*stopped|EXX:\s*now go back to refine exchange calculation", text, re.I))
        result["converged_exx"] = bool(events and "self-consistency reached" in events[-1][0].lower())
        if not result["converged_exx"]:
            result["converged_electronic"] = False
            raise ValueError("Hybrid EXX outer-loop convergence is not confirmed; the XML SCF flag alone is insufficient.")
        result["electronic_check"] = "SCF and EXX outer-loop convergence"
        exchange_errors = re.findall(r"est\. exchange err \(dexx\)\s*=\s*(" + FLOAT + r")\s+Ry", text, re.I)
        if exchange_errors:
            result["exx_exchange_error_ry"] = _floats(exchange_errors[-1])[0]
    if task == "relax":
        opt_flag = output.findtext("convergence_info/opt_conv/convergence_achieved")
        result["converged_ionic"] = opt_flag is not None and opt_flag.strip().lower() in {"true", "t", ".true."}
        if not result["converged_ionic"]:
            raise ValueError("Ionic/cell convergence is not confirmed in QE XML.")
    ground_state = task in {"scf", "relax"}
    # Non-SCF XML may contain zero placeholders for ground-state quantities.
    energy = output.findtext("total_energy/etot") if ground_state else None
    if energy is not None:
        result["energy_ev"] = result["final_energy_ev"] = _floats(energy)[0] * HA_EV
        result["final_energy_ev_per_atom"] = result["energy_ev"] / len(final)
    elif task in {"scf", "relax"}:
        raise ValueError("QE XML has no finite total energy.")
    force_supported = True
    if hybrid:
        header_records = metadata.get("pseudopotentials", {})
        receipt_path = folder / "pseudo_manifest.json"
        if receipt_path.is_file():
            header_records = json.loads(receipt_path.read_text())
        force_supported = bool(header_records) and all(isinstance(value, dict) and str(value.get("pseudo_type", "")).upper() in {"NC", "NCPP"} for value in header_records.values())
        if not force_supported and output.find("forces") is not None:
            result["warnings"].append("Hybrid forces require verified norm-conserving UPFs. Forces were omitted; energy and spectra are checked separately.")
    native_control = metadata.get("namelists", {}).get("control", _extra(metadata.get("parameters", {}), "control"))
    force_node = output.find("forces") if ground_state and force_supported and native_control.get("tprnfor", True) else None
    if force_node is not None:
        forces = np.asarray(_floats(force_node.text)).reshape((-1, 3)) * HA_EV / BOHR_ANGSTROM
        if len(forces) != len(final):
            raise ValueError("QE force count differs from the atom count.")
        result.update(forces_available=True, final_max_force_ev_angstrom=float(np.linalg.norm(forces, axis=1).max()))
        _csv(folder / "forces.csv", ["atom", "element", "fx_ev_angstrom", "fy_ev_angstrom", "fz_ev_angstrom"], ([i + 1, final[i].specie.symbol, *row] for i, row in enumerate(forces)))
        result["artifacts"].append("forces.csv")
    stress = output.findtext("stress") if ground_state and native_control.get("tstress", True) else None
    if stress is not None:
        tensor = np.asarray(_floats(stress))
        if tensor.size != 9:
            raise ValueError("QE stress tensor is incomplete.")
        tensor = tensor.reshape((3, 3), order="F") * HA_BOHR3_GPA
        if not np.allclose(tensor, tensor.T, atol=1e-7, rtol=1e-6):
            raise ValueError("QE stress tensor is not symmetric.")
        result.update(stress_gpa=tensor.tolist(), pressure_gpa=float(np.trace(tensor) / 3))
        _csv(folder / "stress.csv", ["axis", "x_gpa", "y_gpa", "z_gpa"], ([axis, *row] for axis, row in zip("xyz", tensor)))
        result["artifacts"].append("stress.csv")
    magnetic = output.find("magnetization")
    params = metadata.get("parameters", {})
    native_spin = _extra(params, "system")
    expected_soc = bool(native_spin.get("lspinorb", params.get("soc", False)))
    expected_noncollinear = bool(native_spin.get("noncolin", expected_soc or params.get("spin") == "noncollinear"))
    expected_collinear = native_spin.get("nspin", 2 if params.get("spin") == "collinear" else 1) == 2
    if (expected_soc or expected_noncollinear or expected_collinear) and magnetic is None:
        raise ValueError("The magnetic calculation has no magnetization result in QE XML.")
    if magnetic is not None:
        flags = {key: (magnetic.findtext(key) or "").strip().lower() in {"true", "t", ".true."} for key in ("lsda", "noncolin", "spinorbit")}
        if (expected_collinear and not flags["lsda"]) or (expected_noncollinear and not flags["noncolin"]) or (expected_soc and not flags["spinorbit"]):
            raise ValueError("QE output spin treatment differs from the submitted magnetic calculation.")
        result["spin_mode"] = "noncollinear" if flags["noncolin"] else "collinear" if flags["lsda"] else "none"
        result["spin_orbit_coupling"] = flags["spinorbit"]
        if ground_state:
            for tag, key in (("total", "total_magnetization_mu_b"), ("absolute", "absolute_magnetization_mu_b")):
                value = magnetic.findtext(tag)
                if value is not None:
                    numbers = _floats(value)
                    if len(numbers) != 1 or (tag == "absolute" and numbers[0] < 0):
                        raise ValueError("Invalid QE magnetization.")
                    result[key] = numbers[0]
            vector = magnetic.findtext("total_vec")
            if vector is not None:
                numbers = _floats(vector)
                if len(numbers) != 3:
                    raise ValueError("The noncollinear total magnetization needs three components.")
                result.update(total_magnetization_vector_mu_b=numbers, total_magnetization_mu_b=float(np.linalg.norm(numbers)))
            if (expected_collinear or expected_noncollinear) and "total_magnetization_mu_b" not in result:
                raise ValueError("QE magnetic output is missing the total moment.")
            moments = magnetic.findall("Scalar_Site_Magnetic_Moments/SiteMagnetization") or magnetic.findall("Site_Magnetizations/SiteMagnetization")
            if moments:
                rows = []
                for i, moment in enumerate(moments, 1):
                    values = _floats(moment.text)
                    if int(moment.get("atom", i)) != i or len(values) not in {1, 3}:
                        raise ValueError("QE site magnetic moments are incomplete.")
                    rows.append({"atom": i, "element": final[i - 1].specie.symbol, "moment_mu_b": values[0] if len(values) == 1 else values})
                if len(rows) != len(final) or len({len(_floats(moment.text)) for moment in moments}) != 1:
                    raise ValueError("QE site moment count differs from the structure.")
                result["site_magnetization"] = rows
                result["site_magnetization_scope"] = "QE atom-centered projections; their sum need not equal the cell magnetization."
                _csv(folder / "magnetization.csv", ["atom", "element", "moment_mu_b"] if len(_floats(moments[0].text)) == 1 else ["atom", "element", "mx_mu_b", "my_mu_b", "mz_mu_b"], ([row["atom"], row["element"], *(row["moment_mu_b"] if isinstance(row["moment_mu_b"], list) else [row["moment_mu_b"]])] for row in rows))
                result["artifacts"].append("magnetization.csv")
    # electrons.f90:307-310 marks only converged outer-loop energies with !!.
    # Inner SCF totals marked ! must not become fictitious ionic steps.
    marker = r"!!" if hybrid else r"!"
    energies = re.findall(r"^[ \t]*" + marker + r"[ \t]+total energy\s*=\s*(" + FLOAT + r")\s+Ry", text, re.M | re.I)
    if energies:
        values = [_floats(x)[0] * RY_EV for x in energies]
        _csv(folder / "relax_energy.csv", ["ionic_step", "energy_ev"], enumerate(values, 1))
        result["ionic_steps_count"] = len(values)
        result["artifacts"].append("relax_energy.csv")
        if task == "relax":
            result["artifacts"] += _plot(folder, "relax_energy", list(range(1, len(values) + 1)), values, "Ionic step", "Energy (eV)")
    _bands(output, result, folder, metadata)
    if task in {"bands", "nscf"} and not result.get("kpoint_count"):
        raise ValueError("Band eigenvalues are missing.")
    final.to(filename=folder / "final.cif")
    Poscar(final).write_file(folder / "CONTCAR")
    result.update(valid_structure=True, final_structure="final.cif", number_of_atoms=len(final))
    result["artifacts"] += ["final.cif", "CONTCAR", xml.relative_to(folder).as_posix()]


def _table(path, min_columns=2):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Required output {path.name} is missing or empty.")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if len(data) < 2 or data.shape[1] < min_columns or not np.isfinite(data).all():
        raise ValueError(f"Invalid numerical data in {path.name}.")
    return data


def _post_result(folder, task, text, metadata, result):
    if task == "pp":
        path = folder / "charge_density.cube"
        if not path.is_file():
            raise ValueError("charge_density.cube is missing.")
        with path.open() as stream:
            next(stream)
            next(stream)
            origin = _floats(next(stream))
            nat = int(origin[0])
            axes = [_floats(next(stream)) for _ in range(3)]
            counts = [int(row[0]) for row in axes]
            if len(origin) < 4 or nat <= 0 or nat != metadata.get("number_of_atoms", nat) or any(len(row) != 4 for row in axes) or min(counts) <= 0:
                raise ValueError("Invalid charge-density cube header.")
            for _ in range(nat):
                if len(_floats(next(stream))) != 5:
                    raise ValueError("Invalid cube atom record.")
            total, count = 0.0, 0
            for line in stream:
                values = _floats(line)
                count += len(values)
                total += sum(values)
            if count != math.prod(counts):
                raise ValueError("Charge-density cube is incomplete.")
            volume = abs(float(np.linalg.det(np.asarray(axes)[:, 1:])))
            if volume <= 0:
                raise ValueError("Charge-density cube has a singular grid.")
        result.update(cube_grid=counts, integrated_pseudocharge_electrons=total * volume, density_unit="electrons/bohr^3")
        result["artifacts"].append(path.name)
        if (folder / "charge_density.dat").is_file():
            result["artifacts"].append("charge_density.dat")
    elif task in {"dos", "pdos"}:
        name = "dos.dat" if task == "dos" else "pdos.pdos_tot"
        data = _table(folder / name)
        if np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError("DOS energies must be strictly increasing.")
        result["artifacts"].append(name)
        if task == "pdos":
            files = sorted(folder.glob("pdos.pdos_atm*"))
            if not files:
                raise ValueError("Projected DOS files are missing.")
            for path in files:
                _table(path)
                result["artifacts"].append(path.name)
        _csv(folder / f"{task}.csv", ["energy_ev"] + [f"column_{i}" for i in range(1, data.shape[1])], data)
        result["artifacts"].append(f"{task}.csv")
        # Last DOS column is integrated DOS; do not plot it as a density.
        density = data[:, 1:-1] if task == "dos" and data.shape[1] > 2 else data[:, 1:2]
        result["artifacts"] += _plot(folder, task, data[:, 0], density, "Energy (eV)", "DOS (states/eV)")
        result["energy_grid_points"] = len(data)
    elif task == "phonon":
        if re.search(r"convergence\s+NOT\s+achieved", text, re.I):
            raise ValueError("A phonon response did not converge.")
        files = [p for p in sorted(folder.glob(PREFIX + ".dyn*")) if p.is_file() and p.stat().st_size]
        if not files:
            raise ValueError("Phonon dynamical matrices are missing.")
        expected_q = 1
        if metadata.get("phonon_grid"):
            index = folder / (PREFIX + ".dyn0")
            if not index.is_file():
                raise ValueError("Phonon grid index is missing.")
            lines = [line.strip() for line in index.read_text().splitlines() if line.strip()]
            if len(lines) < 2 or len(lines[0].split()) != 3:
                raise ValueError("Invalid phonon grid index.")
            grid = [int(x) for x in lines[0].split()]
            count = int(lines[1])
            expected_q = count
            if grid != metadata["phonon_grid"] or count <= 0:
                raise ValueError("Phonon grid index differs from the requested grid.")
            for i in range(1, count + 1):
                file = folder / f"{PREFIX}.dyn{i}"
                if not file.is_file() or not file.stat().st_size:
                    raise ValueError(f"Phonon dynamical matrix {i} is missing.")
        # ph.x writes frequencies after completing a dynamical matrix.
        frequencies = re.findall(r"freq\s*\(\s*\d+\s*\)\s*=\s*" + FLOAT + r"\s*\[THz\]\s*=\s*(" + FLOAT + r")\s*\[cm-1\]", text, re.I)
        if not frequencies or not re.search(r"Convergence has been achieved", text, re.I):
            raise ValueError("Complete, converged phonon modes were not found.")
        expected_modes = 3 * metadata.get("number_of_atoms", 0)
        if expected_modes and (len(frequencies) % expected_modes or len(frequencies) < expected_q * expected_modes):
            raise ValueError("The phonon output contains an incomplete mode set.")
        values = [_floats(x)[0] for x in frequencies]
        result.update(min_frequency_cm1=min(values), imaginary_mode_count=sum(x < -1 for x in values), phonon_mode_count=len(values), converged_phonon=True)
        result["artifacts"] += [p.name for p in files]
        _csv(folder / "phonons.csv", ["mode_index", "frequency_cm1"], enumerate(values, 1))
        result["artifacts"].append("phonons.csv")
    elif task == "q2r":
        path = folder / "force_constants"
        if not path.is_file() or path.stat().st_size < 40 or not re.search(r"\d", path.read_text(errors="replace")):
            raise ValueError("Force constants are missing or invalid.")
        result["artifacts"].append(path.name)
    elif task == "phonon_bands":
        path = folder / "phonon.freq"
        if not path.is_file():
            raise ValueError("phonon.freq is missing.")
        raw = path.read_text()
        header = re.search(r"nbnd\s*=\s*(\d+)\s*,\s*nks\s*=\s*(\d+)\s*/", raw)
        if not header:
            raise ValueError("Invalid matdyn frequency header.")
        nbnd, nks = map(int, header.groups())
        if nbnd <= 0 or nks <= 0 or nbnd != 3 * metadata.get("number_of_atoms", nbnd // 3):
            raise ValueError("matdyn mode count differs from the atom count.")
        lines = [line for line in raw[header.end():].splitlines() if line.strip()]
        rows_per_point = 1 + (nbnd + 5) // 6
        if len(lines) != nks * rows_per_point:
            raise ValueError("matdyn output is incomplete.")
        data = []
        for point in range(nks):
            start = point * rows_per_point
            line = lines[start]
            # QE 7.4.1 matdyn writes (10x,4f10.6): Cartesian q and an
            # unused path weight, which can overflow even when q is valid.
            if line[:10] == " " * 10 and len(line) in {40, 50}:
                coordinate = [line[10 + i * 10:20 + i * 10] for i in range(3)]
            else:
                coordinate = line.split()
                if len(coordinate) not in {3, 4}:
                    raise ValueError("Invalid matdyn q-point coordinates.")
                coordinate = coordinate[:3]
            row = []
            for field in coordinate:
                values = _floats(field)
                if len(values) != 1:
                    raise ValueError("Invalid matdyn q-point coordinates.")
                row.append(values[0])
            for offset in range(1, rows_per_point):
                count = min(6, nbnd - (offset - 1) * 6)
                line = lines[start + offset]
                fields = [line[i * 10:(i + 1) * 10] for i in range(count)] if len(line) == count * 10 else line.split()
                if len(fields) != count:
                    raise ValueError("matdyn frequencies are incomplete.")
                for field in fields:
                    values = _floats(field)
                    if len(values) != 1:
                        raise ValueError("Invalid matdyn frequency.")
                    row.append(values[0])
            data.append(row)
        data = np.asarray(data)
        values = data[:, 3:]
        path_meta = metadata.get("phonon_path", {})
        x = path_meta.get("distance_inv_angstrom", list(range(nks)))
        if len(x) != nks:
            raise ValueError("Phonon output does not match the requested path.")
        requested = path_meta.get("reciprocal_fractional")
        if requested is not None:
            structure = Structure.from_file(folder / "structure.cif")
            # Generated PW cells use ibrav=0 and alat=|a1|. matdyn converts
            # crystal q to Cartesian coordinates in units of 2*pi/alat.
            expected = np.asarray(requested) @ structure.lattice.reciprocal_lattice.matrix
            expected *= np.linalg.norm(structure.lattice.matrix[0]) / (2 * np.pi)
            if expected.shape != (nks, 3) or not np.allclose(data[:, :3], expected, atol=1e-6, rtol=0):
                raise ValueError("matdyn q-point coordinates differ from the requested path.")
        _csv(folder / "phonon_bands.csv", ["q_index", "qx_2pi_alat", "qy_2pi_alat", "qz_2pi_alat"] + [f"mode_{i + 1}_cm1" for i in range(nbnd)], ([i + 1, *row] for i, row in enumerate(data)))
        result["artifacts"] += ["phonon.freq", "phonon_bands.csv"] + _plot(folder, "phonon_bands", x, values, "Wave vector", "Frequency (cm⁻¹)", path_meta.get("labels"), path_meta.get("segments"))
        result.update(min_frequency_cm1=float(values.min()), imaginary_mode_count=int((values < -1).sum()), qpoint_count=nks)
    elif task == "neb":
        if not re.search(r"neb:\s+convergence achieved", text, re.I) or re.search(r"scf convergence\s+NOT\s+achieved", text, re.I):
            raise ValueError("NEB path and image convergence are not confirmed.")
        tables = re.split(r"image\s+energy\s*\(eV\)\s+error\s*\(eV/A\)\s+frozen", text)
        rows = re.findall(r"^\s*(\d+)\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+([TF])\s*$", tables[-1], re.M) if len(tables) > 1 else []
        count = metadata.get("neb_num_images")
        if len(rows) < 3 or (count and len(rows) != count) or [int(x[0]) for x in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Final NEB image table is incomplete.")
        energies = [_floats(row[1])[0] for row in rows]
        errors = [_floats(row[2])[0] for row in rows]
        if any(error < 0 for error in errors):
            raise ValueError("NEB force errors must be nonnegative.")
        threshold = metadata.get("neb_path_threshold_ev_angstrom")
        active = [error for error, row in zip(errors, rows) if row[3] == "F"]
        if not active or (threshold is not None and max(active) > float(threshold) * 1.0001):
            raise ValueError("NEB forces exceed the requested path threshold.")
        endpoint_forces = [errors[0], errors[-1]]
        endpoint_relaxed = max(endpoint_forces) <= float(threshold) * 1.0001 if threshold is not None else None
        result.update(path_converged=True, converged_ionic=endpoint_relaxed, converged_electronic=True,
                      endpoint_forces_ev_angstrom=endpoint_forces, endpoint_relaxed=endpoint_relaxed,
                      endpoint_force_threshold_ev_angstrom=threshold,
                      barrier_scope="relaxed_endpoints" if endpoint_relaxed else "unrelaxed_endpoints" if endpoint_relaxed is False else "unverified_endpoints",
                      barrier_forward_ev=max(energies) - energies[0], barrier_reverse_ev=max(energies) - energies[-1],
                      reaction_energy_ev=energies[-1] - energies[0], image_count=len(rows), path_max_force_ev_angstrom=max(active))
        if endpoint_relaxed is False:
            result["warnings"].append("The path converged, but its endpoints exceed the force threshold. Barriers are relative to unrelaxed endpoints.")
        elif endpoint_relaxed is None:
            result["warnings"].append("Endpoint relaxation is unverified. Barriers are relative to the supplied endpoints.")
        _csv(folder / "neb.csv", ["image", "energy_ev", "relative_energy_ev", "path_force_ev_angstrom", "frozen"], ([i + 1, energy, energy - energies[0], errors[i], rows[i][3]] for i, energy in enumerate(energies)))
        result["artifacts"] += ["neb.csv"] + _plot(folder, "neb", list(range(1, len(rows) + 1)), np.asarray(energies) - energies[0], "Image", "Relative energy (eV)")


def analyze_outputs(folder, task, expected_structure=None):
    """Require completion plus task-specific numerical and convergence evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    generated = ["final.cif", "CONTCAR", "forces.csv", "stress.csv", "magnetization.csv", "bands.csv", "phonons.csv", "phonon_bands.csv", "neb.csv", "relax_energy.csv", "dos.csv", "pdos.csv"]
    generated += [f"{stem}.{ext}" for stem in ("bands", "phonon_bands", "neb", "relax_energy", "dos", "pdos") for ext in ("png", "pdf", "svg")]
    for name in generated:
        (folder / name).unlink(missing_ok=True)
    result = {"schema_version": 1, "engine": "qe", "task": task, "success": False, "converged_electronic": None, "converged_ionic": None, "energy_ev": None, "final_energy_ev": None, "fermi_energy_ev": None, "artifacts": [], "warnings": [], "scientific_accuracy_validated": False}
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported QE task: {task}")
        original_task = task
        task = ALIASES.get(task, task)
        meta_file = folder / "qe-input.json"
        metadata = json.loads(meta_file.read_text()) if meta_file.is_file() else {"canonical_task": task}
        if metadata.get("canonical_task", task) != task:
            raise ValueError("Stage metadata does not match the requested analysis task.")
        output = folder / (FILENAMES[task] + ".out")
        if not output.is_file():
            raise ValueError(f"{output.name} is missing.")
        text = output.read_text(encoding="utf-8", errors="replace")
        if "JOB DONE." not in text:
            raise ValueError("QE did not finish normally.")
        if re.search(r"Error in routine|MPI_ABORT|convergence NOT achieved|Maximum CPU time exceeded", text, re.I):
            raise ValueError("QE output reports a calculation error or incomplete convergence.")
        result["stdout_sha256"] = _hash(output)
        result["artifacts"].append(output.name)
        if task in {"relax", "scf", "bands", "nscf"}:
            expected = _expected_structure(folder, metadata, expected_structure)
            _pw_result(folder, task, text, metadata, expected, result)
        else:
            _post_result(folder, task, text, metadata, result)
        result.update(success=True, reason="Required outputs and convergence checks passed.", task=original_task)
        if task == "neb" and result.get("endpoint_relaxed") is not True:
            result["reason"] = "NEB path converged; endpoint relaxation is incomplete or unverified."
    except (ValueError, OSError, KeyError, TypeError, ET.ParseError, OverflowError, StopIteration, IndexError) as exc:
        result["reason"] = str(exc) or "QE output is truncated."
    _json(folder / "qe-result.json", result)
    return result


build_stage_inputs = prepare_inputs
parse_stage_result = analyze_outputs
