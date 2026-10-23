"""Verified PBE wavefunction seeds; remote entry points use only the stdlib."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET


INPUT_FILES = ("seed.INCAR", "seed.metadata.json", "warm_start.spec.json")
OUTPUT_FILES = ("seed.vasprun.xml", "seed.OUTCAR", "seed.IBZKPT", "seed.stdout", "seed.stderr",
                "hybrid.stdout", "hybrid.stderr", "warm_start.json", "IBZKPT")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PARALLEL_TAGS = frozenset({"KPAR", "NCORE", "NPAR", "NSIM"})


def _text(path, content):
    # This module runs on the cluster's own python3, which may be 3.6; Path.write_text has newline only from 3.10.
    with open(str(path), "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _write(path, data):
    _text(path, json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare_warm_inputs(source, destination, task, parameters, potcar_symbols=None):
    """Prepare a hybrid target and a static PBE seed on its exact input basis."""
    from pymatgen.io.vasp import Incar, Poscar
    from .vasp import prepare_inputs

    destination = Path(destination)
    metadata = prepare_inputs(source, destination, task, parameters, potcar_symbols)
    if metadata["method"]["functional"] == "PBE":
        raise ValueError("A PBE wavefunction seed is only needed for a hybrid target.")
    target = Incar.from_file(destination / "INCAR")
    target.update(ISTART=1, ICHARG=0, LWAVE=False)
    _text(destination / "INCAR", str(target))
    metadata["method_incar_expected"].update(ISTART=1, ICHARG=0, LWAVE=False)
    metadata["input_sha256"]["INCAR"] = sha256(destination / "INCAR")
    recipe = {**metadata["parameters"], "functional": "PBE"}
    with tempfile.TemporaryDirectory(prefix="dft-pbe-seed-") as directory:
        seed_dir = Path(directory)
        seed_metadata = prepare_inputs(source, seed_dir, "scf", recipe, potcar_symbols)
        seed = Incar.from_file(seed_dir / "INCAR")
        # Keep the target's symmetry and parallel basis. The seed is static even
        # when the target relaxes; it never borrows orbitals from a changed cell.
        # VASP 5.4.4 stops a non-hybrid run with ISYM = 3 ("KPOINTS_FULL not properly
        # initialised"); ISYM = 2 reduces the k-points in the same way, which the
        # hybrid check below confirms point by point.
        seed.update(ISYM=2 if target["ISYM"] == 3 else target["ISYM"], LWAVE=True, LCHARG=False, ISTART=0, ICHARG=2)
        if task == "bands":
            seed["NELMIN"] = target.get("NELMIN", 10)
        _text(destination / "seed.INCAR", str(seed))
        seed_metadata["method_incar_expected"].update(ISYM=seed["ISYM"], LWAVE=True, ISTART=0, ICHARG=2)
        seed_metadata["input_sha256"] = {"INCAR": sha256(destination / "seed.INCAR"),
                                         "POSCAR": metadata["input_sha256"]["POSCAR"],
                                         "KPOINTS": metadata["input_sha256"]["KPOINTS"]}
        _write(destination / "seed.metadata.json", seed_metadata)
    structure = Poscar.from_file(destination / "POSCAR").structure
    spec = {"schema_version": 1, "mode": "pbe_wavecar", "target_task": task,
            "target_inputs": metadata["input_sha256"], "target_tags": dict(target), "seed_tags": dict(seed),
            "seed_incar_sha256": sha256(destination / "seed.INCAR"),
            "seed_metadata_sha256": sha256(destination / "seed.metadata.json"),
            "lattice": structure.lattice.matrix.tolist(), "positions": structure.frac_coords.tolist(),
            "nbands_rule": "Use the VASP default on the same electronic, geometric and parallel basis; require identical actual NBANDS.",
            "kpoint_rule": "Require identical actual seed and hybrid k-points and weights."}
    _write(destination / "warm_start.spec.json", spec)
    metadata["warm_start"] = {"mode": "pbe_wavecar", "spec_sha256": sha256(destination / "warm_start.spec.json"),
                              "seed_metadata_sha256": spec["seed_metadata_sha256"],
                              "nbands_rule": spec["nbands_rule"], "kpoint_rule": spec["kpoint_rule"]}
    _write(destination / "metadata.json", metadata)
    return metadata


def _same(left, right, tolerance=1e-7):
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b, tolerance) for a, b in zip(left, right))
    if type(left) is bool or type(right) is bool:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    return str(left).lower() == str(right).lower()


def _xml_value(node):
    values = (node.text or "").split()
    kind = node.get("type", "float")
    def parse(value):
        if kind == "logical":
            return value.upper() in {"T", ".TRUE."}
        if kind == "int":
            return int(value)
        if kind == "string":
            return value
        return float(value)
    if kind == "string":
        return (node.text or "").strip()
    parsed = [parse(value) for value in values]
    return parsed if node.tag == "v" else parsed[0]


def xml_summary(path):
    """Read complete static-seed/restart identities without loading eigenvalues."""
    tree = ET.parse(path).getroot()
    def values(parent):
        return {node.get("name"): _xml_value(node) for node in parent.iter() if node.tag in {"i", "v"} and node.get("name")}
    incar, actual = values(tree.find("incar")), values(tree.find("parameters"))
    def array(node, name):
        return [[float(value) for value in row.text.split()] for row in node.findall(".//varray[@name='" + name + "']/v")]
    initial = tree.find("structure[@name='initialpos']")
    final = tree.find("structure[@name='finalpos']")
    calculations = tree.findall("calculation")
    energy = calculations[-1].find("energy/i[@name='e_0_energy']") if calculations else None
    return {"incar": incar, "actual": actual, "kpoints": array(tree.find("kpoints"), "kpointlist"),
            "weights": array(tree.find("kpoints"), "weights"), "lattice": array(initial, "basis"),
            "positions": array(initial, "positions"), "final_lattice": array(final, "basis"),
            "final_positions": array(final, "positions"), "ionic_steps": len(calculations),
            "final_energy": float(energy.text) if energy is not None else None,
            "electronic_steps": len(calculations[-1].findall("scstep")) if calculations else 0}


def _check_tags(actual, expected):
    for key, value in expected.items():
        # VASP 5.4.4 does not echo KPAR in vasprun.xml; parallel layout tags do not change results,
        # and the identical NBANDS check below covers what they would. VASP 5 also has no adaptively
        # compressed exchange and ignores LFOCKACE, which changes the algorithm, not the result.
        if key in PARALLEL_TAGS or key == "SYSTEM" or (key == "LFOCKACE" and key not in actual):
            continue
        found = actual.get(key, False if type(value) is bool else None)
        if key == "LDAUTYPE" and isinstance(found, list) and len(found) == 1:
            found = found[0]
        if key == "MAGMOM":
            def flatten(values):
                return [number for item in values for number in (flatten(item) if isinstance(item, (list, tuple)) else [item])]
            found, value = flatten(found or []), flatten(value)
        if not _same(found, value):
            raise ValueError("Output differs from the approved " + key + " setting.")


def _effective(summary, key):
    # VASP calls the actual plane-wave cutoff ENMAX in <parameters>, while
    # preserving the requested ENCUT spelling in <incar>.
    names = ("ENMAX", "ENCUT") if key == "ENCUT" else (key,)
    for name in names:
        if name in summary["actual"]:
            return summary["actual"][name]
    raise ValueError("The actual " + key + " setting is unavailable.")


def _check_seed(summary, spec):
    _check_tags(summary["incar"], spec["seed_tags"])
    for key in ("ENCUT", "ISPIN", "ISYM", "LSORBIT", "LNONCOLLINEAR"):
        expected = spec["seed_tags"].get(key, False if key in {"LSORBIT", "LNONCOLLINEAR"} else None)
        if not _same(_effective(summary, key), expected):
            raise ValueError("The PBE seed differs from the approved actual " + key + ".")
    if (summary["ionic_steps"] != 1 or not 0 < summary["electronic_steps"] < int(summary["actual"]["NELM"])
            or int(summary["actual"]["NSW"]) != 0 or summary["final_energy"] is None
            or not math.isfinite(summary["final_energy"])):
        raise ValueError("The static PBE seed did not reach electronic convergence.")
    for key in ("lattice", "positions"):
        if not _same(summary[key], spec[key]) or not _same(summary["final_" + key], spec[key]):
            raise ValueError("The PBE seed changed the structure or Cartesian basis.")
    if not summary["kpoints"] or len(summary["kpoints"]) != len(summary["weights"]):
        raise ValueError("The PBE seed k-point list is unavailable.")


def wavecar_header(path):
    """Read standard VASP direct-access headers, without wavefunction coefficients."""
    with Path(path).open("rb") as stream:
        length, spin, tag = struct.unpack("<3d", stream.read(24))
        if any(not math.isfinite(value) or int(value) != value for value in (length, spin, tag)):
            raise ValueError("Invalid WAVECAR header.")
        length, spin, tag = int(length), int(spin), int(tag)
        if length < 104 or length > 2 ** 30 or spin not in {1, 2} or tag not in {45200, 45210, 53300, 53310}:
            raise ValueError("Unsupported WAVECAR format.")
        stream.seek(length)
        header = struct.unpack("<12d", stream.read(96))
        nk, nb = header[:2]
        if any(not math.isfinite(value) or int(value) != value or value <= 0 for value in (nk, nb)):
            raise ValueError("Invalid WAVECAR dimensions.")
        nk, nb = int(nk), int(nb)
        if Path(path).stat().st_size < length * (2 + spin * nk * (nb + 1)):
            raise ValueError("WAVECAR is truncated.")
        points = []
        for channel in range(spin):
            rows = []
            for index in range(nk):
                stream.seek(length * (2 + (channel * nk + index) * (nb + 1)))
                row = struct.unpack("<4d", stream.read(32))
                if not row[0] > 0 or not all(math.isfinite(value) for value in row):
                    raise ValueError("Invalid WAVECAR k-point record.")
                rows.append(list(row[1:]))
            if points and not _same(points, rows):
                raise ValueError("WAVECAR spin channels have different k-points.")
            points = rows
    return {"nbands": nb, "nkpoints": nk, "spin_channels": spin, "encut": header[2],
            "lattice": [list(header[3 + i * 3:6 + i * 3]) for i in range(3)], "kpoints": points}


def _check_wave(header, seed):
    expected_spin = 2 if seed["actual"]["ISPIN"] == 2 else 1
    checks = ((header["nbands"], seed["actual"]["NBANDS"]), (header["nkpoints"], len(seed["kpoints"])),
              (header["spin_channels"], expected_spin), (header["encut"], _effective(seed, "ENCUT")),
              (header["lattice"], seed["lattice"]), (header["kpoints"], seed["kpoints"]))
    if any(not _same(a, b) for a, b in checks):
        raise ValueError("WAVECAR does not match the converged PBE seed.")


def initialize(root):
    root = Path(root)
    spec = json.loads((root / "warm_start.spec.json").read_text())
    for name, digest in {**spec["target_inputs"], "seed.INCAR": spec["seed_incar_sha256"],
                         "seed.metadata.json": spec["seed_metadata_sha256"]}.items():
        if sha256(root / name) != digest:
            raise ValueError("Warm-start inputs changed before the job began.")
    directory = root / ".pbe_seed"
    directory.mkdir()
    for name in ("POSCAR", "KPOINTS", "POTCAR"):
        shutil.copyfile(root / name, directory / name)
    shutil.copyfile(root / "seed.INCAR", directory / "INCAR")
    _write(root / "warm_start.json", {"schema_version": 1, "spec_sha256": sha256(root / "warm_start.spec.json"),
                                    "potcar_sha256": sha256(root / "POTCAR"), "seed_accepted": False,
                                    "restart_verified": False})


def accept_seed(root, exit_code):
    root = Path(root)
    spec = json.loads((root / "warm_start.spec.json").read_text())
    receipt = json.loads((root / "warm_start.json").read_text())
    receipt["seed_exit_code"] = exit_code
    try:
        seed_dir = root / ".pbe_seed"
        for name in ("vasprun.xml", "OUTCAR", "IBZKPT"):
            if (seed_dir / name).is_file():
                shutil.copyfile(seed_dir / name, root / ("seed." + name))
        if exit_code:
            raise ValueError("The PBE seed exited with an error.")
        seed = xml_summary(root / "seed.vasprun.xml")
        _check_seed(seed, spec)
        for name in ("POSCAR", "KPOINTS", "INCAR"):
            expected_seed = spec["seed_incar_sha256"] if name == "INCAR" else spec["target_inputs"][name]
            if sha256(seed_dir / name) != expected_seed or sha256(root / name) != spec["target_inputs"][name]:
                raise ValueError("The seed and hybrid input bases differ.")
        if any(sha256(directory / "POTCAR") != receipt["potcar_sha256"] for directory in (root, seed_dir)):
            raise ValueError("The seed and hybrid POTCAR differ.")
        wave = seed_dir / "WAVECAR"
        header = wavecar_header(wave)
        _check_wave(header, seed)
        if "NBANDS" in spec["target_tags"] and spec["target_tags"]["NBANDS"] != header["nbands"]:
            raise ValueError("The hybrid NBANDS differs from the seed.")
        digest = sha256(wave)
        temporary = root / "WAVECAR.part"
        shutil.copyfile(wave, temporary)
        if sha256(temporary) != digest or sha256(wave) != digest:
            temporary.unlink(missing_ok=True)
            raise ValueError("WAVECAR changed during the copy.")
        os.replace(temporary, root / "WAVECAR")
        receipt.update(seed_accepted=True, wavecar={"sha256": digest, "size": wave.stat().st_size, **header},
                       seed_xml_sha256=sha256(root / "seed.vasprun.xml"),
                       seed_outcar_sha256=sha256(root / "seed.OUTCAR"))
    except Exception as exc:
        receipt["reason"] = str(exc)
        _write(root / "warm_start.json", receipt)
        raise
    _write(root / "warm_start.json", receipt)


def _check_target(seed, target, spec, stdout):
    _check_tags(target["incar"], spec["target_tags"])
    if target["actual"].get("ISTART") != 1 or target["actual"].get("ICHARG") != 0:
        raise ValueError("VASP did not retain the requested wavefunction restart.")
    if not re.search(r"WAVECAR\s+(?:successfully\s+read|(?:file\s+)?(?:was\s+)?read\s+successfully)", stdout, re.I) or re.search(r"WAVECAR\s+not\s+read", stdout, re.I):
        raise ValueError("VASP did not confirm a successful WAVECAR read.")
    for key in ("NBANDS", "NELECT", "ISPIN", "LNONCOLLINEAR", "LSORBIT", "ENCUT", "ISYM"):
        values = _effective(seed, key), _effective(target, key)
        if key == "ISYM" and [int(value) for value in values] == [2, 3]:
            continue
        if not _same(*values):
            raise ValueError("Seed and hybrid differ in actual " + key + ".")
    for key in ("kpoints", "weights", "lattice", "positions"):
        if not _same(seed[key], target[key]):
            raise ValueError("Seed and hybrid differ in actual " + key + ".")


def finish(root, exit_code):
    root = Path(root)
    spec = json.loads((root / "warm_start.spec.json").read_text())
    receipt = json.loads((root / "warm_start.json").read_text())
    receipt["hybrid_exit_code"] = exit_code
    try:
        if exit_code or not receipt.get("seed_accepted"):
            raise ValueError("The hybrid calculation exited with an error.")
        if sha256(root / "WAVECAR") != receipt["wavecar"]["sha256"] or sha256(root / "POTCAR") != receipt["potcar_sha256"]:
            raise ValueError("Restart wavefunctions or POTCAR changed after preparation.")
        _check_target(xml_summary(root / "seed.vasprun.xml"), xml_summary(root / "vasprun.xml"), spec,
                      (root / "hybrid.stdout").read_text(errors="replace"))
        receipt.update(restart_verified=True, hybrid_xml_sha256=sha256(root / "vasprun.xml"),
                       hybrid_stdout_sha256=sha256(root / "hybrid.stdout"))
    except Exception as exc:
        receipt["reason"] = str(exc)
        _write(root / "warm_start.json", receipt)
        raise
    _write(root / "warm_start.json", receipt)


def validate_warm_start(output, run, metadata):
    """Recheck retained raw evidence locally, including scientific seed acceptance."""
    from pymatgen.io.vasp import Vasprun
    from .methods import validate_method_output

    root = Path(output)
    expected = metadata["warm_start"]
    if sha256(root / "warm_start.spec.json") != expected["spec_sha256"] or sha256(root / "seed.metadata.json") != expected["seed_metadata_sha256"]:
        raise ValueError("The approved warm-start specification changed.")
    spec = json.loads((root / "warm_start.spec.json").read_text())
    receipt = json.loads((root / "warm_start.json").read_text())
    if sha256(root / "seed.INCAR") != spec["seed_incar_sha256"]:
        raise ValueError("The approved PBE seed input changed.")
    if receipt.get("spec_sha256") != expected["spec_sha256"] or not receipt.get("seed_accepted") or not receipt.get("restart_verified"):
        raise ValueError("The PBE wavefunction restart was not verified.")
    if (root / "potcar_hash.sha256").read_text().split()[0] != receipt.get("potcar_sha256"):
        raise ValueError("The restart POTCAR receipt differs from the executed potential.")
    for filename, key in (("seed.vasprun.xml", "seed_xml_sha256"), ("seed.OUTCAR", "seed_outcar_sha256"),
                          ("vasprun.xml", "hybrid_xml_sha256"), ("hybrid.stdout", "hybrid_stdout_sha256")):
        if sha256(root / filename) != receipt.get(key):
            raise ValueError("Warm-start output evidence changed.")
    seed_summary = xml_summary(root / "seed.vasprun.xml")
    _check_seed(seed_summary, spec)
    _check_wave(receipt["wavecar"], seed_summary)
    _check_target(seed_summary, xml_summary(root / "vasprun.xml"), spec, (root / "hybrid.stdout").read_text(errors="replace"))
    seed_metadata = json.loads((root / "seed.metadata.json").read_text())
    seed = Vasprun(root / "seed.vasprun.xml", parse_potcar_file=False, parse_eigen=False, parse_dos=False)
    validate_method_output(seed.incar, seed_metadata, vasp_version=str(seed.vasp_version))
    if not seed.converged_electronic or not math.isfinite(float(seed.final_energy)) or receipt.get("seed_exit_code") != 0 or receipt.get("hybrid_exit_code") != 0:
        raise ValueError("The seed or hybrid calculation did not finish successfully.")
    if not _same(run.parameters.get("NBANDS"), seed.parameters.get("NBANDS")):
        raise ValueError("The hybrid changed the wavefunction band count.")
    return {"mode": "pbe_wavecar", "verified": True, "seed_converged_electronic": True,
            "wavecar_sha256": receipt["wavecar"]["sha256"], "nbands": receipt["wavecar"]["nbands"],
            "nkpoints": receipt["wavecar"]["nkpoints"], "seed_xml_sha256": receipt["seed_xml_sha256"]}


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "initialize":
        initialize(Path.cwd())
    elif action == "seed":
        accept_seed(Path.cwd(), int(sys.argv[2]))
    elif action == "finish":
        finish(Path.cwd(), int(sys.argv[2]))
    else:
        raise SystemExit("Unknown restart action.")
