"""ABINIT inputs and evidence-based result checks.

Inputs use Hartree and Bohr internally; public results use eV and Å.
Pseudopotentials are PseudoDojo 0.4 norm-conserving psp8 files pinned by
commit and SHA-256 in abinit_pseudos.json. References:
https://docs.abinit.org/variables/ and https://www.pseudo-dojo.org/.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Element, Structure

from .engine_common import (BOHR_ANGSTROM, HA_BOHR3_GPA, HA_EV, HA_PER_BOHR_EV_PER_ANGSTROM, COMMON_KEYS,
                            TASKS as SHARED_TASKS, _csv, _hash, _integer, _json, _number, band_gap, band_path,
                            check_parameters, finish_energies, hubbard_terms, kpoint_mesh, load_structure,
                            metadata_base, new_result, same_structure, site_moments, smearing, species_order,
                            spin_settings, structure_from_native, band_reference, write_bands, write_dos,
                            write_final_structure)
from .numerical_defaults import vacuum_axes
from .spectra import plot_phonons

ENGINE = "abinit"
PROGRAM = "abinit"
INPUT_FILE = "run.abi"
OUTPUT_FILE = "run.abo"
LOG_FILE = "run.log"
METADATA_FILE = "abinit-input.json"
RESULT_FILE = "abinit-result.json"
PREFIX = "dft_agent"
LIBRARY_DIR = "library"
# phonons: density-functional perturbation theory on a q mesh, merged by mrgddb and
# interpolated by anaddb along the band path and onto a dense mesh for the DOS.
TASKS = (*SHARED_TASKS, "phonons")
PHONON_KEYS = {"qmesh", "phonon_dos_mesh"}
STAGE_KEYS = PHONON_KEYS
PARAMETER_KEYS = COMMON_KEYS | {"ecut_ha", "kshift"} | PHONON_KEYS
# Programs of the same installation that run after abinit in the phonon job, on one process.
COMPANIONS = ("mrgddb", "anaddb")
ANADDB_INPUT = "anaddb.abi"
MRGDDB_INPUT = "mrgddb.in"
ANADDB_OUTPUT = "anaddb.abo"
MERGED_DDB = f"{PREFIX}_DDB"
CM1_PER_HA = 219474.6313705
FUNCTIONALS = ("PBE", "PBEsol")
# Files a later stage may inherit and the names it reads them under.
CHECKPOINTS = {"scf": ("dft_agento_DEN",)}
RESTART_MAPPING = {"bands": {"dft_agento_DEN": "dft_agenti_DEN"}, "dos": {"dft_agento_DEN": "dft_agenti_DEN"}}
# A rank other than the first writes its error only to the abort file.
OUTPUT_FILES = ("run.abo", "run.log", "dft_agento_EIG", "dft_agento_DOS", "__ABI_MPIABORTFILE__",
                MERGED_DDB, ANADDB_OUTPUT, "anaddb_PHFRQ", "anaddb_PHDOS", "mrgddb.log", "anaddb.log")
GENERATED = ("final.cif", "CONTCAR", "bands.csv", "dos.csv", "forces.csv", "magnetization.csv", "phonon_bands.csv", "phonon_dos.csv",
             *(f"{stem}.{ext}" for stem in ("bands", "dos", "phonons") for ext in ("png", "pdf", "svg")))


@lru_cache(maxsize=1)
def catalog():
    return json.loads(files("vasp_slurm_agent").joinpath("abinit_pseudos.json").read_text(encoding="utf-8"))


def library_key(parameters):
    functional = parameters.get("functional", "PBE")
    if functional not in FUNCTIONALS:
        raise ValueError("ABINIT here uses PseudoDojo norm-conserving PBE or PBEsol pseudopotentials. "
                         "LDA and hybrid functionals are not available.")
    if functional == "PBEsol":
        if parameters.get("soc"):
            raise ValueError("No pinned fully relativistic PBEsol pseudopotentials are available. Use PBE for spin-orbit coupling.")
        return "pbesol_sr"
    return "pbe_fr" if parameters.get("soc") else "pbe_sr"


def pseudopotentials(elements, key):
    library = catalog()["libraries"][key]
    missing = [element for element in elements if element not in library["elements"]]
    if missing:
        raise ValueError(f"{library['label']} has no pseudopotential for {', '.join(missing)}.")
    return {element: {**library["elements"][element], "library": key, "element": element} for element in elements}


def checkpoint_names(task, metadata):
    return CHECKPOINTS.get(task, ())


def restart_mapping(task, metadata):
    return dict(RESTART_MAPPING.get(task, {}))

def _format(value):
    return f"{value:.14g}"


def _matrix(rows):
    return "\n".join("  " + " ".join(_format(float(x)) for x in row) for row in rows)


def _shiftk(p):
    """Meshes stay Γ-centred. ABINIT stops on a mesh that a symmetry maps outside itself (chksymbreak), as
    QE's half-step shift along each axis of GaAs's fcc cell was; a symmetric set needs several shifts."""
    if any(p.get("kshift") or []):
        raise ValueError("ABINIT here uses Γ-centred meshes: a half-step shift along each axis breaks the symmetry of "
                         "centred cells, which ABINIT refuses. Use a denser mesh, or QE with kshift.")
    return "0 0 0"


def _irreducible_qpoints(structure, qmesh, moments):
    """One q point per star of the Γ-centred mesh, Γ first, in the input cell's reduced coordinates.
    Atoms with different starting moments count as different, as for ABINIT's symmetry finder."""
    import warnings

    import spglib

    kinds = {}
    values = moments if moments is not None else [0.0] * len(structure)
    numbers = [kinds.setdefault((site.specie.Z, round(float(value), 3)), len(kinds) + 1) for site, value in zip(structure, values)]
    with warnings.catch_warnings():
        # spglib 2.5+ warns about its legacy error handling on every call.
        warnings.simplefilter("ignore", DeprecationWarning)
        mapping, grid = spglib.get_ir_reciprocal_mesh(list(qmesh), (structure.lattice.matrix, structure.frac_coords, numbers),
                                                      is_shift=[0, 0, 0], is_time_reversal=True, symprec=1e-5)
    stars = sorted(set(int(index) for index in mapping), key=lambda index: (bool(np.any(grid[index])), index))
    return [(np.asarray(grid[index], dtype=float) / np.asarray(qmesh, dtype=float)).tolist() for index in stars]


def _phonon_datasets(structure, p, mesh, moments, polar, scf_steps, time_reversal, metadata):
    """Ground state, then the electric-field response of a polar insulator, then phonons at each q."""
    qmesh = p.get("qmesh", [max(1, math.ceil(n / 2)) for n in mesh])
    if not isinstance(qmesh, list) or len(qmesh) != 3 or any(type(n) is not int or not 1 <= n <= 16 for n in qmesh):
        raise ValueError("qmesh needs three integers from 1 to 16.")
    qpoints = _irreducible_qpoints(structure, qmesh, moments)
    if len(qpoints) > 60:
        raise ValueError(f"The {'x'.join(map(str, qmesh))} q mesh has {len(qpoints)} irreducible points. Use a coarser qmesh.")
    # The ground state, the k derivative of a polar insulator, then one dataset per irreducible q.
    lines = [f"ndtset {1 + polar + len(qpoints)}", "ngkpt " + " ".join(map(str, mesh)), "nshiftk 1", "shiftk " + _shiftk(p), f"nstep {scf_steps}",
             "nqpt 1", f"rfatpol 1 {len(structure)}", "rfdir 1 1 1", "getwfk 1", "prtwf 0", "prtden 0",
             # Dataset 1: the ground state the perturbations start from, converged far below the SCF stage.
             f"kptopt1 {time_reversal}", "nqpt1 0", "getwfk1 0", "tolvrs1 1.0e-18", "prtwf1 1", "prtden1 1"]
    phonon_sets, index = [], 2
    if polar:
        # Dataset 2: derivatives of the wavefunctions with respect to k, needed by the field response.
        lines += ["rfelfd2 2", "iscf2 -3", "kptopt2 2", "tolwfr2 1.0e-22", "qpt2 0 0 0", "prtwf2 1"]
        index = 3
    for number, q in enumerate(qpoints):
        gamma = not any(q)
        lines += [f"rfphon{index} 1", f"kptopt{index} {2 if gamma else 3}", f"tolvrs{index} 1.0e-8",
                  f"qpt{index} " + " ".join(_format(x) for x in q)]
        if gamma and polar:
            lines += [f"rfelfd{index} 3", f"getddk{index} 2"]
        phonon_sets.append(index)
        index += 1
    path = band_path(structure, p)
    metadata.update(mesh=mesh, qmesh=qmesh, qpoints=qpoints, phonon_datasets=phonon_sets, polar=polar, phonon_path=path)
    ddbs = [f"{PREFIX}o_DS{n}_DDB" for n in phonon_sets]
    # mrgddb needs two inputs; a Γ-only mesh is read directly. mrgddb 10.0.3 reads its own output name as an
    # input when given on the command line, so it reads the list from standard input.
    metadata["ddb_file"] = MERGED_DDB if len(ddbs) > 1 else ddbs[0]
    metadata["mrgddb_input"] = f"{MERGED_DDB}\nDFT Agent phonons\n{len(ddbs)}\n" + "".join(name + "\n" for name in ddbs) if len(ddbs) > 1 else None
    metadata["post_commands"] = ([{"program": "mrgddb", "arguments": [], "stdin": MRGDDB_INPUT, "stdout": "mrgddb.log"}] if len(ddbs) > 1 else []) + [
        {"program": "anaddb", "arguments": [ANADDB_INPUT], "stdout": "anaddb.log"}]
    return lines


def _anaddb_input(structure, p, metadata, polar):
    """Interpolated frequencies at every band-path point and a phonon DOS on a dense mesh."""
    path = metadata["phonon_path"]
    points = np.asarray(path["reciprocal_fractional"], dtype=float)
    queried = points.copy()
    if polar:
        # At Γ the dipole term depends on the direction of approach; a tiny q along the path picks it,
        # so the LO-TO splitting shows at Γ instead of only beside it.
        for segment in path.get("segments") or [[0, len(points)]]:
            for i in range(segment[0], segment[1]):
                if not np.any(np.abs(points[i]) > 1e-9):
                    neighbour = i - 1 if i - 1 >= segment[0] else i + 1
                    if neighbour < segment[1] and np.any(points[neighbour]):
                        queried[i] = 1e-4 * points[neighbour] / np.linalg.norm(points[neighbour])
    qmesh = metadata["qmesh"]
    dense = p.get("phonon_dos_mesh", [min(48, 8 * n) if n > 1 else 1 for n in qmesh])
    if not isinstance(dense, list) or len(dense) != 3 or any(type(n) is not int or not 1 <= n <= 96 for n in dense):
        raise ValueError("phonon_dos_mesh needs three integers from 1 to 96.")
    tetrahedra = min(dense) > 1
    lines = [f'ddb_filepath "{metadata["ddb_file"]}"', f'output_file "{ANADDB_OUTPUT}"', 'outdata_prefix "anaddb"',
             "ifcflag 1", "brav 1", "ngqpt " + " ".join(map(str, qmesh)), "nqshft 1", "q1shft 3*0.0", "asr 1",
             f"chneut {1 if polar else 0}", f"dipdip {1 if polar else 0}", *(["dieflag 1"] if polar else []),
             f"prtdos {2 if tetrahedra else 1}", f"dosdeltae {_format(1 / CM1_PER_HA)}", "ng2qpt " + " ".join(map(str, dense)),
             *([] if tetrahedra else [f"dossmear {_format(5 / CM1_PER_HA)}"]),
             f"nph1l {len(queried)}", "qph1l", *("  " + " ".join(_format(x) for x in q) + " 1.0" for q in queried)]
    metadata.update(phonon_dos_mesh=dense, phonon_dos_method="tetrahedron" if tetrahedra else "gaussian_smearing",
                    anaddb_qpoints=queried.tolist())
    return "\n".join(lines) + "\n"


def prepare_inputs(structure_path, destination, task, parameters, library=None):
    """Write one ABINIT stage; the workflow supplies an accepted density for bands and DOS."""
    if task not in TASKS:
        raise ValueError(f"Unsupported ABINIT task: {task}")
    p = check_parameters(parameters, PARAMETER_KEYS, "ABINIT")
    structure = load_structure(structure_path)
    if hubbard_terms(structure, p):
        raise ValueError("DFT+U in ABINIT needs PAW datasets, which DFT Agent does not install. Use CP2K or Elk for DFT+U.")
    spin = spin_settings(p)
    key = library_key(p)
    elements = species_order(structure)
    potentials = pseudopotentials(elements, key)
    electrons = sum(potentials[site.specie.symbol]["valence"] for site in structure)
    ecut = _number(p.get("ecut_ha", max(entry["hints_ha"]["normal"] for entry in potentials.values())), "ecut_ha", True)
    if not 5 <= ecut <= 200:
        raise ValueError("ecut_ha must be between 5 and 200 Hartree.")
    kind, width = smearing(p, 0.136)
    moments = site_moments(structure, p, spin)
    spinor = bool(p.get("soc")) or spin == "noncollinear"
    if task == "phonons" and spinor:
        raise ValueError("ABINIT phonons here are for spin-unpolarized or collinear calculations; perturbation theory with "
                         "spin-orbit coupling or noncollinear spin is not set up.")
    # A nonmagnetic insulator without vacuum also gets the electric-field response: Born charges, the
    # electronic dielectric constant and the LO-TO splitting of a polar crystal.
    polar = task == "phonons" and kind == "insulator" and spin == "none" and not vacuum_axes(structure)
    occupied = math.ceil(electrons / (1 if spinor else 2))
    if spin == "collinear":
        occupied = math.ceil((electrons + sum(abs(m) for m in moments)) / 2)
    extra = p.get("extra_bands")
    if extra is None and task == "phonons":
        # Perturbation theory needs only the occupied bands of an insulator.
        extra = 0 if kind == "insulator" and spin == "none" else max(4, math.ceil(0.2 * occupied))
    elif extra is None:
        extra = max(8, math.ceil(0.5 * occupied)) if task in {"bands", "dos"} else max(4, math.ceil(0.2 * occupied))
    extra = _integer(extra, "extra_bands", 0)
    if spinor and (occupied + extra) % 2:
        # m_inwffil.F90 stops a spinor run whose band count is odd.
        extra += 1
    notes = []
    if kind == "insulator" and spin != "none":
        notes.append("Spin-polarized fixed occupations need a fixed moment; a 0.03 eV Gaussian smearing is used instead.")
        occopt, tsmear = 7, 0.001
    elif kind == "insulator":
        occopt, tsmear = 1, None
    else:
        occopt, tsmear = (4 if kind == "metal" else 7), width / HA_EV
    mesh = kpoint_mesh(structure, p)
    lines = [f"# DFT Agent ABINIT {task}", f'output_file "{OUTPUT_FILE}"', f'indata_prefix "{PREFIX}i"',
             f'outdata_prefix "{PREFIX}o"', f'tmpdata_prefix "{PREFIX}t"', f'pp_dirpath "./{LIBRARY_DIR}"',
             'pseudos "' + ", ".join(potentials[element]["filename"] for element in elements) + '"',
             "chksymtnons 0", f"natom {len(structure)}", f"ntypat {len(elements)}",
             "znucl " + " ".join(str(Element(element).Z) for element in elements),
             "typat " + " ".join(str(elements.index(site.specie.symbol) + 1) for site in structure),
             "acell 3*1.0 Angstrom", "rprim", _matrix(structure.lattice.matrix), "xred", _matrix(structure.frac_coords),
             f"ecut {_format(ecut)}", f"nband {occupied + extra}", f"occopt {occopt}"]
    if tsmear is not None:
        lines.append(f"tsmear {_format(tsmear)}")
    if vacuum_axes(structure):
        # The default metallic preconditioner (diemac 1e6) all but freezes the long-wavelength
        # potential along a vacuum gap: graphene, MoS2 and CrI3 layers kept a residual (res2) of
        # 1-20 while the energy test passed, and the noisy forces stopped an MoS2 relaxation.
        lines.append("diemac 2")
    if spin == "collinear":
        lines += ["nsppol 2", "spinat", _matrix([[0.0, 0.0, m] for m in moments])]
    elif spinor:
        lines += ["nspinor 2", f"nspden {4 if spin == 'noncollinear' else 1}"]
        if spin == "noncollinear":
            lines += ["spinat", _matrix(moments)]
    time_reversal = 4 if spin == "noncollinear" else 1
    scf_steps = _integer(p.get("max_scf_steps", 100), "max_scf_steps")
    metadata = metadata_base(ENGINE, task, structure, structure_path, p)
    if task in {"relax", "scf"}:
        lines += [f"kptopt {time_reversal}", "ngkpt " + " ".join(map(str, mesh)), "nshiftk 1", "shiftk " + _shiftk(p),
                  f"nstep {scf_steps}", "prtden 1", "prtwf 0"]
        if task == "relax":
            force = _number(p.get("force_tolerance_ev_angstrom", 0.01), "force_tolerance_ev_angstrom", True)
            tolmxf = force / HA_PER_BOHR_EV_PER_ANGSTROM
            lines += ["ionmov 22", f"ntime {_integer(p.get('max_ionic_steps', 50), 'max_ionic_steps')}",
                      f"tolmxf {_format(tolmxf)}", f"toldff {_format(tolmxf / 20)}"]
            if p.get("cell_relax"):
                lines += ["optcell 2", "dilatmx 1.1", "ecutsm 0.5"]
        else:
            tolerance = _number(p.get("energy_tolerance_ev", 1e-6), "energy_tolerance_ev", True)
            lines.append(f"toldfe {_format(tolerance / HA_EV)}")
        metadata["mesh"] = mesh
    elif task == "phonons":
        lines += _phonon_datasets(structure, p, mesh, moments, polar, scf_steps, time_reversal, metadata)
        anaddb = _anaddb_input(structure, p, metadata, polar)
    else:
        buffer = max(2, extra // 4)
        lines += ["iscf -2", f'getden_filepath "{PREFIX}i_DEN"', "tolwfr 1.0e-10", f"nstep {scf_steps}", f"nbdbuf {buffer}"]
        metadata["buffer_bands"] = buffer
        if task == "bands":
            path = band_path(structure, p)
            points = path["reciprocal_fractional"]
            lines += ["kptopt 0", f"nkpt {len(points)}", "kpt", _matrix(points), f"wtk {len(points)}*1.0"]
            metadata["band_path"] = path
        else:
            dense = kpoint_mesh(structure, {"mesh": p.get("dos_mesh", [min(64, 2 * n) if n > 1 else 1 for n in mesh])}, "mesh")
            tetrahedra = min(dense) >= 2
            lines += [f"kptopt {time_reversal}", "ngkpt " + " ".join(map(str, dense)), "nshiftk 1", "shiftk 0 0 0",
                      f"prtdos {2 if tetrahedra else 1}"]
            if not tetrahedra and tsmear is None:
                # Tetrahedra need a three-dimensional mesh; a monolayer's DOS is broadened with Gaussians instead.
                occopt, tsmear = 7, width / HA_EV
                lines[lines.index("occopt 1")] = "occopt 7"
                lines.append(f"tsmear {_format(tsmear)}")
                notes.append(f"Tetrahedra need a three-dimensional mesh, so this DOS uses {width:.3g} eV Gaussian broadening.")
            metadata.update(dos_mesh=dense, dos_method="tetrahedron" if tetrahedra else "gaussian_smearing")
    text = "\n".join(lines) + "\n"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / INPUT_FILE).write_text(text, encoding="utf-8")
    structure.to(filename=str(destination / "structure.cif"))
    inputs = [INPUT_FILE, "structure.cif", METADATA_FILE]
    if task == "phonons":
        (destination / ANADDB_INPUT).write_text(anaddb, encoding="utf-8")
        inputs.insert(1, ANADDB_INPUT)
        if metadata["mrgddb_input"]:
            (destination / MRGDDB_INPUT).write_text(metadata["mrgddb_input"], encoding="utf-8")
            inputs.insert(1, MRGDDB_INPUT)
    metadata.update(input_file=INPUT_FILE, output_file=OUTPUT_FILE, stdout_file=LOG_FILE, arguments=[INPUT_FILE],
                    input_files=inputs, output_files=list(dict.fromkeys([*OUTPUT_FILES, *([metadata["ddb_file"]] if task == "phonons" else [])])),
                    pseudopotentials=potentials, library_key=key, library_dir=LIBRARY_DIR,
                    library_files={entry["filename"]: {"element": element, "sha256": entry["sha256"], "size": entry["size"],
                                                       "kind": "psp8", "valence": entry["valence"]}
                                   for element, entry in potentials.items()},
                    electrons=electrons, ecut_ha=ecut, occopt=occopt, tsmear_ha=tsmear, spin=spin, spinor=spinor,
                    scf_fixed_occupations=kind == "insulator" and spin == "none",
                    soc=bool(p.get("soc")), nsppol=2 if spin == "collinear" else 1, bands=occupied + extra,
                    starting_moments=moments, notes=notes)
    _json(destination / METADATA_FILE, metadata)
    return metadata


# --- outputs -------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


def _outvars(text):
    """Variables echoed after computation, as name -> list of numbers."""
    marker = text.rfind("-outvars: echo values of variables after computation")
    if marker < 0:
        raise ValueError("ABINIT did not echo its final variables.")
    values, name = {}, None
    for line in text[marker:].splitlines()[1:]:
        if line.startswith("=") or line.startswith(" ===") or not line.strip():
            if values:
                break
            continue
        match = re.match(r"^[ P\-*]{0,3}\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$", line)
        if match and not _NUMBER.fullmatch(match[1]):
            name = match[1]
            values[name] = []
            rest = match[2]
        elif name and re.match(r"^\s+[-+0-9.]", line):
            rest = line
        else:
            continue
        values[name] += [float(item.replace("D", "E").replace("d", "e")) for item in _NUMBER.findall(rest)]
    return values


def _final_structure(variables, natom):
    # ABINIT echoes only values that differ from their defaults.
    try:
        acell = np.asarray(variables.get("acell", [1.0, 1.0, 1.0])[:3]) * BOHR_ANGSTROM
        rprim = np.asarray(variables.get("rprim", [1, 0, 0, 0, 1, 0, 0, 0, 1])[:9]).reshape(3, 3)
        xred = np.asarray(variables.get("xred", [0.0] * (3 * natom))[:3 * natom]).reshape(natom, 3)
        typat = [int(round(x)) for x in variables.get("typat", [1] * natom)[:natom]]
        znucl = [int(round(x)) for x in variables["znucl"]]
    except (KeyError, ValueError) as exc:
        raise ValueError("ABINIT final structure variables are incomplete.") from exc
    lattice = rprim * acell[:, None]
    species = [Element.from_Z(znucl[index - 1]).symbol for index in typat]
    return Structure(lattice, species, xred)


def _eigenvalues(path):
    """(spin, k, band) eigenvalues in Hartree and the Fermi level, from the EIG text file."""
    text = Path(path).read_text(errors="replace", encoding="utf-8")
    fermi = re.search(r"Fermi \(or HOMO\) energy \(hartree\) =\s*(" + _NUMBER.pattern + ")", text)
    blocks = re.split(r"Eigenvalues \(hartree\) for nkpt=\s*\d+\s+k points(?:, SPIN (?:UP|DOWN))?:", text)[1:]
    spins = []
    for block in blocks:
        kpoints = re.split(r"\s*kpt#\s*\d+,", block)[1:]
        table = []
        for entry in kpoints:
            header, _, values = entry.partition("(reduced coord)")
            count = int(re.search(r"nband=\s*(\d+)", header)[1])
            numbers = [float(x) for x in _NUMBER.findall(values)][:count]
            if len(numbers) != count:
                raise ValueError("ABINIT eigenvalue table is incomplete.")
            table.append(numbers)
        spins.append(table)
    if not spins or any(len(table) != len(spins[0]) or not table for table in spins):
        raise ValueError("ABINIT eigenvalue table is missing or inconsistent.")
    return np.asarray(spins), (float(fermi[1]) if fermi else None)


def _dos(path, spins):
    """Total DOS in states/eV/cell from the DOS file; spin blocks are summed."""
    rows, blocks = [], []
    for line in Path(path).read_text(errors="replace", encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            if rows:
                blocks.append(rows)
                rows = []
            continue
        numbers = [float(x) for x in _NUMBER.findall(line)]
        if len(numbers) >= 3:
            rows.append(numbers[:3])
    if rows:
        blocks.append(rows)
    if not blocks or len(blocks) != spins or any(len(block) != len(blocks[0]) for block in blocks):
        raise ValueError("ABINIT DOS table is missing or inconsistent.")
    data = np.asarray(blocks)
    energy = data[0, :, 0] * HA_EV
    columns = [data[index, :, 1] / HA_EV for index in range(spins)]
    return energy, np.stack(columns, axis=1)


def _magnetization(text, result, structure):
    total = re.findall(r"Magnetization \(Bohr magneton\)=\s*(" + _NUMBER.pattern + ")", text)
    if total:
        result["total_magnetization_mu_b"] = float(total[-1].replace("D", "E"))
    start = text.rfind("Integrated electronic and magnetization densities in atomic spheres:")
    if start < 0:
        return
    sites = []
    for line in text[start:].splitlines()[4:4 + len(structure)]:
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit():
            atom = int(fields[0])
            sites.append({"atom": atom, "element": structure[atom - 1].specie.symbol, "moment_mu_b": float(fields[5])})
    if len(sites) == len(structure):
        result["site_magnetization"] = sites
        result["absolute_magnetization_mu_b"] = float(sum(abs(item["moment_mu_b"]) for item in sites))
        result["site_magnetization_scope"] = "Integrated in ABINIT atomic spheres (ratsph); approximate local moments."


def _phonon_table(path, rows_expected, modes):
    """Frequencies (q, mode) in Hartree from anaddb's PHFRQ file."""
    raw = Path(path).read_text(errors="replace", encoding="utf-8")
    header = re.search(r"number_of_qpoints\s+(\d+).*?number_of_phonon_modes\s+(\d+)", raw, re.S)
    rows = [line.split() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not header or int(header[1]) != rows_expected or int(header[2]) != modes or len(rows) != rows_expected:
        raise ValueError("anaddb's phonon frequencies do not match the band path.")
    if any(len(row) != modes + 1 or int(row[0]) != index + 1 for index, row in enumerate(rows)):
        raise ValueError("anaddb's phonon frequency table is incomplete.")
    values = np.asarray([[float(x) for x in row[1:]] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("anaddb's phonon frequencies contain invalid values.")
    return values


def _born_and_dielectric(text, structure):
    """Electronic (ε∞) and relaxed-ion (ε0) dielectric constants and each atom's mean Born charge, from anaddb."""
    found = {}
    row = r"\s*(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s*\n"
    for key, title in (("dielectric", "Electronic dielectric tensor"), ("static_dielectric", "Relaxed ion dielectric tensor")):
        block = re.search(re.escape(title) + r"\s*\n" + row * 3, text)
        if block:
            tensor = np.asarray([float(x) for x in block.groups()]).reshape(3, 3)
            found.update({f"{key}_constant": float(np.trace(tensor) / 3), f"{key}_tensor": tensor.tolist()})
    # How far the computed charges miss neutrality before anaddb imposes it: a measure of k-mesh convergence.
    violation = text.split("by the effective charges is as follows", 1)[-1].split("Effective charge tensors after", 1)[0] \
        if "by the effective charges is as follows" in text else ""
    values = re.findall(r"^\s+[123]\s+[123]\s+(" + _NUMBER.pattern + r")\s+" + _NUMBER.pattern + r"\s*$", violation, re.M)
    if values:
        found["charge_neutrality_violation"] = float(max(abs(float(value)) for value in values))
    # Rows of "atom, displacement direction, then the three field directions", after charge neutrality.
    start = text.find("Effective charge tensors after")
    if start >= 0:
        block = text[start:].split("Now, the imaginary part", 1)[0]
        rows = re.findall(r"^\s+(\d+)\s+([123])\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s+(" + _NUMBER.pattern + r")\s*$",
                          block, re.M)
        tensors = {}
        for atom, direction, *values in rows:
            tensors.setdefault(int(atom), {})[int(direction)] = [float(value) for value in values]
        if sorted(tensors) == list(range(1, len(structure) + 1)) and all(len(item) == 3 for item in tensors.values()):
            found["born_charges"] = [{"atom": atom, "element": structure[atom - 1].specie.symbol,
                                      "born_charge": float(np.trace([tensors[atom][d] for d in (1, 2, 3)]) / 3)} for atom in sorted(tensors)]
    return found


def _phonons(folder, metadata, result, structure):
    """Dispersion along the band path and the DOS from anaddb, in cm⁻¹; a polar insulator adds ε∞ and Born charges."""
    output = folder / ANADDB_OUTPUT
    text = output.read_text(errors="replace", encoding="utf-8") if output.is_file() else ""
    if "the run completed succes" not in text:
        raise ValueError("anaddb did not finish." + _reported_error(folder, text))
    result["artifacts"].append(ANADDB_OUTPUT)
    path = metadata["phonon_path"]
    modes = 3 * len(structure)
    values = _phonon_table(folder / "anaddb_PHFRQ", len(path["reciprocal_fractional"]), modes) * CM1_PER_HA
    rows = [line.split() for line in (folder / "anaddb_PHDOS").read_text(errors="replace", encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    dos = np.asarray([[float(x) for x in row[:3]] for row in rows if len(row) >= 3])
    if dos.ndim != 2 or len(dos) < 10:
        raise ValueError("anaddb's phonon DOS is missing.")
    # The DOS integrates to three modes per atom.
    if abs(dos[-1, 2] - modes) > 0.02 * modes:
        raise ValueError(f"The phonon DOS integrates to {dos[-1, 2]:.3g} modes instead of {modes}.")
    frequency, states = dos[:, 0] * CM1_PER_HA, dos[:, 1] / CM1_PER_HA
    distance = path["distance_inv_angstrom"]
    _csv(folder / "phonon_bands.csv", ["q_index", "distance_inv_angstrom", "q1", "q2", "q3", *(f"mode_{m + 1}_cm1" for m in range(modes))],
         ([index + 1, float(distance[index]), *path["reciprocal_fractional"][index], *map(float, row)] for index, row in enumerate(values)))
    _csv(folder / "phonon_dos.csv", ["frequency_cm1", "dos_states_per_cm1"], zip(frequency.tolist(), states.tolist()))
    result["artifacts"] += ["anaddb_PHFRQ", "anaddb_PHDOS", "phonon_bands.csv", "phonon_dos.csv"]
    if (folder / metadata["ddb_file"]).is_file():
        result["artifacts"].append(metadata["ddb_file"])
    result["artifacts"] += plot_phonons(folder, "phonons", distance, values, path.get("labels"), path.get("segments"), dos=(frequency, states))
    gamma = [index for index, q in enumerate(path["reciprocal_fractional"]) if not np.any(np.abs(q) > 1e-9)]
    # Acoustic modes next to Γ can dip a few cm⁻¹ below zero from the interpolation; deeper modes are instabilities.
    unstable = values < -10.0
    summary = {"qmesh": metadata["qmesh"], "irreducible_qpoints": len(metadata["qpoints"]), "frequency_unit": "cm-1",
               "max_frequency_cm1": float(values.max()), "min_frequency_cm1": float(values.min()),
               "imaginary_mode_count": int(unstable.sum()), "gamma_frequencies_cm1": values[gamma[0]].tolist() if gamma else None,
               "dos_method": metadata["phonon_dos_method"], "polar_response": bool(metadata.get("polar"))}
    if metadata.get("polar"):
        summary.update(_born_and_dielectric(text, structure))
        miss = summary.get("charge_neutrality_violation")
        if miss is not None and miss > 0.1:
            # GaAs on a Γ-centred 8x8x8 mesh missed by 0.27 e and gave ε∞ = 29.7, about twice the converged value.
            result["warnings"].append(f"The Born charges missed charge neutrality by {miss:.2f} e before it was imposed, so the "
                                      "dielectric constant and the LO-TO splitting are not converged: use a denser k mesh.")
    if unstable.any():
        result["warnings"].append(f"{int(unstable.sum())} phonon frequencies along the path are imaginary, down to {values.min():.0f} cm⁻¹: "
                                  "the structure is not at a stable minimum, or the k and q meshes are too coarse.")
    result["phonons"] = summary


def _reported_error(folder, text):
    """The message of the first error document ABINIT wrote, as a sentence to append, or nothing."""
    sources = [text] + [(folder / name).read_text(errors="replace", encoding="utf-8") for name in ("run.log", "__ABI_MPIABORTFILE__")
                        if (folder / name).is_file()]
    for source in sources:
        for match in re.finditer(r"^--- !(\w*(?:Error|ERROR|BUG))[ \t]*\n(.*?)^\.\.\.", source, re.M | re.S):
            block = re.search(r"^message:[ \t]*\|?[ \t]*\n((?:[ \t]+\S.*(?:\n|$))+)", match[2], re.M)
            if block:
                message = " ".join(line.strip() for line in block[1].splitlines() if line.strip())
                return f' It stopped with {match[1]}: "{message[:300]}".'
    return ""


def analyze_outputs(folder, task, expected_structure=None):
    """Require normal completion plus task-specific convergence and numerical evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in GENERATED:
        (folder / name).unlink(missing_ok=True)
    result = new_result(ENGINE, task)
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported ABINIT task: {task}")
        metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("canonical_task") != task or metadata.get("engine") != ENGINE:
            raise ValueError("Stage metadata does not match the requested analysis task.")
        output = folder / OUTPUT_FILE
        if not output.is_file():
            raise ValueError(f"{OUTPUT_FILE} is missing.")
        text = output.read_text(encoding="utf-8", errors="replace")
        result["stdout_sha256"] = _hash(output)
        result["artifacts"].append(OUTPUT_FILE)
        if "Calculation completed." not in text:
            raise ValueError("ABINIT did not finish normally." + _reported_error(folder, text))
        if re.search(r"--- !(?:ERROR|BUG)|^\s*\w+\s*:\s*(?:ERROR|BUG)\b", text, re.M):
            raise ValueError("ABINIT reported an error." + _reported_error(folder, text))
        converged = "was not enough SCF cycles to converge" not in text
        result["converged_electronic"] = converged
        if not converged:
            raise ValueError("The electronic iterations did not reach the tolerance within nstep.")
        if task == "relax":
            ionic = "gradients are converged" in text and "was not enough Broyd/MD steps" not in text
            result["converged_ionic"] = ionic
            if not ionic:
                raise ValueError("The geometry optimization did not converge within the step limit.")
        variables = _outvars(text)
        expected = structure_from_native(metadata["native_structure"])
        structure = _final_structure(variables, len(expected))
        # ABINIT symmetrizes nearly symmetric inputs within tolsym (1e-5), so allow 1e-4.
        if not same_structure(structure, expected, positions=task != "relax", atol=1e-4):
            raise ValueError("ABINIT used a structure that differs from the prepared input.")
        if task in {"relax", "scf"}:
            finish_energies(result, variables["etotal"][0] * HA_EV, len(structure))
            forces = np.asarray(variables.get("fcart", []))
            if forces.size == 3 * len(structure):
                forces = forces.reshape(-1, 3) * HA_PER_BOHR_EV_PER_ANGSTROM
                result["final_max_force_ev_angstrom"] = float(np.max(np.linalg.norm(forces, axis=1)))
            stress = variables.get("strten", [])
            if len(stress) == 6:
                s = -np.asarray(stress) * HA_BOHR3_GPA
                tensor = [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]]
                result.update(stress_gpa=tensor, pressure_gpa=float((s[0] + s[1] + s[2]) / 3))
        if task in {"relax", "scf"}:
            write_final_structure(folder, result, structure if task == "relax" else expected)
        result["spin_mode"] = metadata.get("spin")
        result["spin_orbit_coupling"] = bool(metadata.get("soc"))
        _magnetization(text, result, structure)
        if task == "phonons":
            _phonons(folder, metadata, result, expected)
            result.update(success=True, reason="Required outputs and convergence checks passed.")
            _json(folder / RESULT_FILE, result)
            return result
        eig = folder / f"{PREFIX}o_EIG"
        energies, fermi = _eigenvalues(eig)
        energies = energies * HA_EV
        if metadata.get("buffer_bands"):
            # Buffer bands are excluded from ABINIT's convergence test.
            energies = energies[:, :, :energies.shape[2] - int(metadata["buffer_bands"])]
        if fermi is None and task in {"bands", "dos"}:
            # A non-self-consistent run keeps the Fermi level of the SCF density it reads.
            values = re.findall(r"^fermie\s*:\s*(" + _NUMBER.pattern + ")", text, re.M)
            fermi = float(values[-1]) if values else None
        elif fermi is None:
            values = re.findall(r"Fermi \(or HOMO\) energy \(hartree\) =\s*(" + _NUMBER.pattern + ")", text)
            fermi = float(values[-1]) if values else None
        if fermi is not None:
            result["fermi_energy_ev"] = fermi * HA_EV
        result["artifacts"].append(eig.name)
        counted = metadata.get("occopt") == 1 or metadata.get("nsppol") == 1
        # After a fixed-occupation SCF that level is the HOMO of its mesh, which a path or a denser mesh can pass.
        homo = task in {"bands", "dos"} and metadata.get("scf_fixed_occupations", metadata.get("occopt") == 1)
        gap = band_gap(energies, electrons=metadata.get("electrons") if counted else None,
                       fermi=None if homo else result.get("fermi_energy_ev"), spinor=bool(metadata.get("spinor")),
                       scope="sampled_path" if task == "bands" else "sampled_k_points")
        result["band_gap"] = gap
        result["band_gap_ev"] = gap.get("gap_ev")
        result.update(kpoint_count=int(energies.shape[1]), band_count=int(energies.shape[2]))
        if task == "bands":
            reference, result["energy_reference"] = band_reference(gap, None if homo else result.get("fermi_energy_ev"))
            write_bands(folder, result, energies, metadata["band_path"], reference)
        elif task == "dos":
            dos_file = folder / f"{PREFIX}o_DOS"
            energy, dos = _dos(dos_file, int(metadata.get("nsppol", 1)))
            header = re.search(r"#\s*Fermi energy\s*:\s*(" + _NUMBER.pattern + ")", dos_file.read_text(errors="replace", encoding="utf-8"))
            if header and result.get("fermi_energy_ev") is None:
                result["fermi_energy_ev"] = float(header[1]) * HA_EV
            result["artifacts"].append(dos_file.name)
            reference = result["fermi_energy_ev"] if result.get("fermi_energy_ev") is not None else 0.0
            write_dos(folder, result, energy, dos, reference, columns=("dos_up", "dos_down") if dos.shape[1] == 2 else ("dos",))
        result.update(success=True, reason="Required outputs and convergence checks passed.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result["reason"] = str(exc) or "ABINIT output is truncated."
    _json(folder / RESULT_FILE, result)
    return result


# Runs on the cluster with only the standard library: a header check, not a parser.
LIBRARY_VALIDATOR = r'''def validate_library(data, entry):
    if len(data) < 200 or len(data) > 50 * 1024 * 1024:
        raise ValueError("psp8 file is empty or unexpectedly large.")
    lines = data.decode("utf-8", errors="replace").splitlines()
    zatom, zion = (float(x.replace("D", "E")) for x in lines[1].split()[:2])
    code = int(lines[2].split()[0])
    if code != 8:
        raise ValueError("Not a psp8 pseudopotential.")
    if round(zion, 6) != round(float(entry["valence"]), 6):
        raise ValueError("The psp8 valence charge differs from the prepared input.")
    return {"zatom": zatom, "valence": zion}
'''
