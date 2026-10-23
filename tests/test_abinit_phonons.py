"""ABINIT phonons: perturbation datasets, the mrgddb and anaddb steps of the job, and the analysis.

Inputs and the job script are checked directly; anaddb outputs here are synthetic in the format
ABINIT 10.0.3 wrote for aluminium on Arrhenius, and test parsing, units and refusals only.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import abinit, engine_workflow
from vasp_slurm_agent.config import ClusterConfig

DATA = Path(__file__).parent / "data" / "engines" / "abinit"


def _crystal(tmp_path, name, species, a):
    path = tmp_path / f"{name}.cif"
    coords = [[0, 0, 0], [0.25, 0.25, 0.25]] if len(species) == 2 else [[0, 0, 0]]
    group = "F-43m" if len(set(species)) == 2 else "Fd-3m" if len(species) == 2 else "Fm-3m"
    Structure.from_spacegroup(group, Lattice.cubic(a), species, coords).get_primitive_structure().to(filename=str(path))
    return path


def test_polar_insulator_gets_field_response_and_one_dataset_per_q(tmp_path):
    gaas = _crystal(tmp_path, "GaAs", ["Ga", "As"], 5.75)
    metadata = abinit.prepare_inputs(gaas, tmp_path / "gaas", "phonons", {"mesh": [8, 8, 8], "electronic_type": "insulator", "qmesh": [4, 4, 4]})
    text = (tmp_path / "gaas" / "run.abi").read_text()
    # The eight stars of the 4x4x4 mesh, as in ABINIT's own AlAs tutorial.
    assert len(metadata["qpoints"]) == 8 and metadata["qpoints"][0] == [0.0, 0.0, 0.0] and metadata["polar"]
    assert "ndtset 10" in text and "tolvrs1 1.0e-18" in text and "prtwf1 1" in text
    assert "rfelfd2 2" in text and "iscf2 -3" in text and "tolwfr2 1.0e-22" in text
    assert "rfelfd3 3" in text and "getddk3 2" in text and "qpt3 0 0 0" in text and "kptopt3 2" in text
    assert "kptopt10 3" in text and "nband 9" not in text and "occopt 1" in text
    # Every dataset sets exactly one tolerance.
    for index in range(1, 11):
        assert sum(f"{name}{index} " in text for name in ("tolvrs", "tolwfr")) == 1
    assert metadata["phonon_datasets"] == list(range(3, 11))
    assert (tmp_path / "gaas" / "mrgddb.in").read_text().splitlines()[:3] == ["dft_agent_DDB", "DFT Agent phonons", "8"]
    anaddb = (tmp_path / "gaas" / "anaddb.abi").read_text()
    assert "dipdip 1" in anaddb and "chneut 1" in anaddb and "dieflag 1" in anaddb and "prtdos 2" in anaddb
    assert f"nph1l {len(metadata['phonon_path']['reciprocal_fractional'])}" in anaddb
    # Γ on the path is approached along the path, so the dipole term gives the LO-TO splitting there.
    queried = np.asarray(metadata["anaddb_qpoints"])
    assert not (np.abs(queried) < 1e-9).all(axis=1).any()
    assert {"mrgddb.in", "anaddb.abi"} <= set(metadata["input_files"])


def test_metals_and_gamma_only_meshes(tmp_path):
    aluminum = _crystal(tmp_path, "Al", ["Al"], 4.04)
    metadata = abinit.prepare_inputs(aluminum, tmp_path / "al", "phonons", {"mesh": [12, 12, 12], "electronic_type": "metal", "qmesh": [4, 4, 4]})
    text = (tmp_path / "al" / "run.abi").read_text()
    assert "rfelfd" not in text and "ndtset 9" in text and "dipdip 0" in (tmp_path / "al" / "anaddb.abi").read_text()
    assert not metadata["polar"] and "occopt 4" in text
    gamma = abinit.prepare_inputs(aluminum, tmp_path / "gamma", "phonons", {"mesh": [4, 4, 4], "electronic_type": "metal", "qmesh": [1, 1, 1]})
    # A single DDB needs no merge.
    assert gamma["ddb_file"] == "dft_agento_DS2_DDB" and [step["program"] for step in gamma["post_commands"]] == ["anaddb"]
    assert "mrgddb.in" not in gamma["input_files"]
    with pytest.raises(ValueError, match="spin-orbit"):
        abinit.prepare_inputs(aluminum, tmp_path / "soc", "phonons", {"mesh": [4, 4, 4], "soc": True})
    # ABINIT stopped GaAs on the half-step shifted mesh (chksymbreak): shifts are refused with the reason.
    with pytest.raises(ValueError, match="breaks the symmetry"):
        abinit.prepare_inputs(aluminum, tmp_path / "shift", "phonons", {"mesh": [4, 4, 4], "kshift": [1, 1, 1]})
    with pytest.raises(ValueError, match="qmesh"):
        abinit.prepare_inputs(aluminum, tmp_path / "bad", "phonons", {"mesh": [4, 4, 4], "qmesh": [4, 4]})


def _anaddb_outputs(folder, metadata, *, lowest=0.0, neutrality=0.02):
    """anaddb files in ABINIT 10.0.3's formats; frequencies in Hartree."""
    points = len(metadata["phonon_path"]["reciprocal_fractional"])
    ha = 1 / abinit.CM1_PER_HA
    rows = [[lowest * ha if index == 3 else 0.0, 0.0, 0.0, 500 * ha, 500 * ha, 520 * ha] for index in range(points)]
    (folder / "anaddb_PHFRQ").write_text("# ABINIT generated phonon band structure file. All in Ha atomic units\n# \n"
                                         f"# number_of_qpoints {points}\n# number_of_phonon_modes 6\n# \n"
                                         + "".join(f"{index + 1:5d}" + "".join(f"{x:20.10E}" for x in row) + "\n" for index, row in enumerate(rows)))
    grid = np.linspace(0, 530 * ha, 200)
    (folder / "anaddb_PHDOS").write_text("# \n# Phonon density of states\n# omega     PHDOS    INT_PHDOS\n#\n"
                                         + "".join(f"{w:16.8E}{1.0:16.8E}{6 * i / 199:16.8E}\n" for i, w in enumerate(grid)))
    tensor = "\n".join("   " + "  ".join(f"{13.1 if i == j else 0.0:14.8f}" for j in range(3)) for i in range(3))
    born = "".join(f"         {atom}         {d}    " + "    ".join(f"{(2.1 if atom == 1 else -2.1) * (d == e):.6E}" for e in (1, 2, 3)) + "\n"
                   for atom in (1, 2) for d in (1, 2, 3))
    (folder / abinit.ANADDB_OUTPUT).write_text(
        " The violation of the charge neutrality conditions\n by the effective charges is as follows :\n"
        "    atom        electric field\n displacement     direction\n"
        + "".join(f"       {d}               {e}       {neutrality if d == e else 0.0:.6f}        0.000000\n" for d in (1, 2, 3) for e in (1, 2, 3))
        + "\n Effective charge tensors after \n imposition of the charge neutrality (if requested by user),\n"
        " and eventual restriction to some part :\n   atom    displacement\n" + born + " Now, the imaginary part of the dynamical matrix is zeroed\n\n"
        f" Electronic dielectric tensor\n{tensor}\n\n\n Relaxed ion dielectric tensor\n{tensor}\n\n anaddb : the run completed succesfully.\n")


def test_phonon_results_are_read_in_wavenumbers_with_born_charges(tmp_path):
    source = DATA / "02_scf"
    folder = tmp_path / "outputs"
    metadata = abinit.prepare_inputs(source / "structure.cif", folder, "phonons", {"mesh": [4, 4, 4], "electronic_type": "insulator", "qmesh": [2, 2, 2]})
    shutil.copy(source / "run.abo", folder / "run.abo")
    _anaddb_outputs(folder, metadata)
    result = abinit.analyze_outputs(folder, "phonons")
    assert result["success"], result["reason"]
    phonons = result["phonons"]
    assert phonons["max_frequency_cm1"] == pytest.approx(520) and phonons["min_frequency_cm1"] == pytest.approx(0)
    assert phonons["dielectric_constant"] == pytest.approx(13.1) and phonons["static_dielectric_constant"] == pytest.approx(13.1)
    assert [item["born_charge"] for item in phonons["born_charges"]] == pytest.approx([2.1, -2.1])
    assert phonons["charge_neutrality_violation"] == pytest.approx(0.02) and not result["warnings"]
    assert {"phonons.png", "phonon_bands.csv", "phonon_dos.csv", "anaddb_PHFRQ"} <= set(result["artifacts"])
    header = (folder / "phonon_bands.csv").read_text().splitlines()[0]
    assert header.startswith("q_index,distance_inv_angstrom,q1,q2,q3,mode_1_cm1")
    # An imaginary branch and charges far from neutrality are reported, not hidden.
    _anaddb_outputs(folder, metadata, lowest=-40.0, neutrality=0.27)
    result = abinit.analyze_outputs(folder, "phonons")
    assert result["success"] and result["phonons"]["imaginary_mode_count"] == 1
    assert any("imaginary" in item for item in result["warnings"]) and any("denser k mesh" in item for item in result["warnings"])
    # A DOS that does not count three modes per atom, or an unfinished anaddb, is refused.
    (folder / "anaddb_PHDOS").write_text((folder / "anaddb_PHDOS").read_text().replace("6.00000000E+00", "4.00000000E+00"))
    assert "integrates to 4 modes" in abinit.analyze_outputs(folder, "phonons")["reason"]
    (folder / abinit.ANADDB_OUTPUT).write_text("anaddb started\n")
    assert "anaddb did not finish" in abinit.analyze_outputs(folder, "phonons")["reason"]


def test_phonons_continue_an_scf_and_keep_their_keys_out_of_it():
    sequence = engine_workflow._sequence(["phonons"], {"mesh": [8, 8, 8], "qmesh": [4, 4, 4]}, None, "abinit")
    assert [(stage["name"], stage["restart_from"]) for stage in sequence] == [("scf", None), ("phonons", 0)]


@pytest.mark.parametrize("command,launcher", [
    ("srun /opt/abinit/bin/abinit", "srun -n 1 /opt/abinit/bin/"),
    ("/opt/abinit/bin/mpiexec -n {tasks} /opt/abinit/bin/abinit", "/opt/abinit/bin/mpiexec -n 1 /opt/abinit/bin/"),
    ("abinit", ""),
])
def test_job_merges_and_interpolates_on_one_process_after_abinit(tmp_path, command, launcher):
    gaas = _crystal(tmp_path, "GaAs", ["Ga", "As"], 5.75)
    metadata = abinit.prepare_inputs(gaas, tmp_path / "gaas", "phonons", {"mesh": [4, 4, 4], "electronic_type": "insulator", "qmesh": [2, 2, 2]})
    config = ClusterConfig(host="cluster", user="me", remote_root="/scratch/me", partition="cpu", engine="abinit", engine_command=command,
                           engine_library_root="/scratch/me/libraries/abinit", tasks=32)
    script = engine_workflow._script(config, {"metadata": metadata})
    lines = script.splitlines()
    abinit_line = next(index for index, line in enumerate(lines) if line.endswith("run.abi > run.log"))
    assert lines[abinit_line + 2:abinit_line + 10] == [
        'if [ "$code" -eq 0 ]; then', f"  {launcher}mrgddb < mrgddb.in > mrgddb.log 2>&1", "  code=$?", "fi",
        'if [ "$code" -eq 0 ]; then', f"  {launcher}anaddb anaddb.abi > anaddb.log 2>&1", "  code=$?", "fi"]
    metadata["post_commands"][0]["program"] = "cut3d"
    with pytest.raises(ValueError, match="post-processing"):
        engine_workflow._script(config, {"metadata": metadata})


def test_phonon_table_requires_every_mode(tmp_path):
    table = tmp_path / 'anaddb_PHFRQ'
    table.write_text('# number_of_qpoints 2\n# number_of_phonon_modes 6\n1 0 0 0\n2 0 0 0\n')
    with pytest.raises(ValueError, match='incomplete'):
        abinit._phonon_table(table, 2, 6)


def test_gamma_only_phonons_collect_the_unmerged_database(tmp_path):
    aluminum = _crystal(tmp_path, 'Al', ['Al'], 4.04)
    metadata = abinit.prepare_inputs(aluminum, tmp_path / 'gamma', 'phonons', {'mesh': [4, 4, 4], 'qmesh': [1, 1, 1]})
    assert metadata['ddb_file'] in metadata['output_files']
