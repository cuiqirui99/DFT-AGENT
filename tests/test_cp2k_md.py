"""CP2K for large cells: orbital transformation at Γ, and Born-Oppenheimer molecular dynamics.

Outputs here are synthetic and test inputs, units, statistics and refusals only.
"""
import json

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import cp2k, engine_workflow


@pytest.fixture
def argon(tmp_path):
    """An fcc argon supercell: 32 atoms with a 3.72 Å nearest-neighbour distance."""
    path = tmp_path / "Ar.cif"
    structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.26), ["Ar"], [[0, 0, 0]])
    structure.make_supercell(2)
    structure.to(filename=str(path))
    return path


def test_gamma_point_insulators_use_orbital_transformation(argon, tmp_path):
    metadata = cp2k.prepare_inputs(argon, tmp_path / "ot", "scf", {"mesh": [1, 1, 1], "electronic_type": "insulator"})
    text = (tmp_path / "ot" / "cp2k.inp").read_text()
    assert "&OT ON" in text and "&OUTER_SCF ON" in text and "PRECONDITIONER FULL_SINGLE_INVERSE" in text
    assert "ADDED_MOS" not in text and "&SMEAR" not in text and "&DIAGONALIZATION" not in text
    assert metadata["scf_method"] == "orbital transformation" and metadata["added_mos"] == 0
    # Metals, k-point meshes and DOS stages keep diagonalization with empty orbitals.
    for name, parameters, task in (("metal", {"mesh": [1, 1, 1], "electronic_type": "metal"}, "scf"),
                                   ("mesh", {"mesh": [2, 2, 2], "electronic_type": "insulator"}, "scf"),
                                   ("dos", {"mesh": [1, 1, 1], "electronic_type": "insulator"}, "dos")):
        cp2k.prepare_inputs(argon, tmp_path / name, task, parameters)
        text = (tmp_path / name / "cp2k.inp").read_text()
        assert "&DIAGONALIZATION ON" in text and "&OT" not in text


def test_md_inputs(argon, tmp_path):
    metadata = cp2k.prepare_inputs(argon, tmp_path / "nvt", "md", {"mesh": [1, 1, 1], "electronic_type": "insulator",
                                                                   "md_temperature_k": 85, "md_steps": 200, "md_timestep_fs": 2.0})
    text = (tmp_path / "nvt" / "cp2k.inp").read_text()
    assert "RUN_TYPE MD" in text and "ENSEMBLE NVT" in text and "STEPS 200" in text and "TIMESTEP [fs] 2" in text
    assert "TEMPERATURE [K] 85" in text and "TYPE CSVR" in text and "TIMECON [fs] 100" in text and "&OT ON" in text
    # Stress and per-step force tables would cost time and fill the output at every step.
    assert "STRESS_TENSOR" not in text and "&FORCES ON" not in text
    assert metadata["md"] == {"ensemble": "nvt", "temperature_k": 85.0, "steps": 200, "timestep_fs": 2.0, "thermostat_fs": 100.0}
    cp2k.prepare_inputs(argon, tmp_path / "nve", "md", {"mesh": [1, 1, 1], "md_ensemble": "nve"})
    assert "THERMOSTAT" not in (tmp_path / "nve" / "cp2k.inp").read_text()
    for parameters, message in (({"md_ensemble": "npt"}, "nvt or nve"), ({"md_timestep_fs": 10}, "time step")):
        with pytest.raises(ValueError, match=message):
            cp2k.prepare_inputs(argon, tmp_path / "bad", "md", parameters)
    # Molecular dynamics computes its own SCF at every step, and may follow a relaxation.
    sequence = engine_workflow._sequence(["relax", "md"], {}, None, "cp2k")
    assert [(stage["name"], stage["restart_from"], stage["structure_from"]) for stage in sequence] == [
        ("relax", None, None), ("md", None, 0)]
    # A production run continues from the final frame of an equilibration run: a packed water box
    # did not finish 200 relaxation steps, while a short, strongly coupled run removes its excess heat.
    sequence = engine_workflow._sequence(["md", "md", "scf"], {}, [{"md_thermostat_fs": 10, "md_steps": 400}, {}, {}], "cp2k")
    assert [(stage["name"], stage["structure_from"]) for stage in sequence] == [("md", None), ("md", 0), ("scf", 1)]
    assert sequence[0]["parameters"]["md_thermostat_fs"] == 10 and "md_thermostat_fs" not in sequence[1]["parameters"]
    sequence = engine_workflow._sequence(["relax", "md", "md"], {}, None, "cp2k")
    assert [stage["structure_from"] for stage in sequence] == [None, 0, 1]


def _md_outputs(folder, structure, steps=40, finished=True, failed_scf=False, settling=0.0):
    """settling: potential energy released per fs (Ha), as by an unrelaxed start; it heats the cell."""
    rng = np.random.default_rng(3)
    atoms = len(structure)
    times = np.arange(steps + 1) * 2.0
    last = steps if finished else steps // 2
    rows = ["#     Step Nr.          Time[fs]        Kin.[a.u.]          Temp[K]            Pot.[a.u.]        Cons Qty[a.u.]        UsedTime[s]"]
    for step in range(last + 1):
        rows.append(f"{step:10d}{times[step]:18.6f}{0.004:18.9f}{85 + 4000 * settling * 40 + rng.normal(0, 5):18.6f}"
                    f"{-21.1 - settling * times[step] + 1e-4 * rng.normal():18.9f}{-21.0 + 2e-6 * times[step]:18.9f}{1.0:18.6f}")
    (folder / "dft_agent-1.ener").write_text("\n".join(rows) + "\n")
    frames = []
    for step in range(last + 1):
        coords = structure.cart_coords + rng.normal(0, 0.05, (atoms, 3))
        frames.append(f"{atoms:8d}\n i = {step:8d}, time = {times[step]:12.3f}, E = -21.1\n"
                      + "".join(f"  Ar {x:16.8f}{y:16.8f}{z:16.8f}\n" for x, y, z in coords))
    (folder / "dft_agent-pos-1.xyz").write_text("".join(frames))
    scf = " *** SCF run NOT converged ***\n" if failed_scf else " *** SCF run converged in 7 steps ***\n"
    (folder / "cp2k.out").write_text(scf * (last + 1) + " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]   -21.100000\n"
                                     "  **** **** ******  **  PROGRAM ENDED AT                 2026-10-22 17:00:00.000\n")


def _staged(tmp_path, argon, parameters):
    folder = tmp_path / "outputs"
    metadata = cp2k.prepare_inputs(argon, folder, "md", parameters)
    (folder / "cp2k-input.json").write_text(json.dumps(metadata))
    return folder, Structure.from_file(argon)


def test_md_results_report_temperature_drift_frame_and_rdf(tmp_path, argon):
    folder, structure = _staged(tmp_path, argon, {"mesh": [1, 1, 1], "electronic_type": "insulator", "md_temperature_k": 85,
                                                  "md_steps": 40, "md_timestep_fs": 2.0})
    _md_outputs(folder, structure)
    result = cp2k.analyze_outputs(folder, "md")
    assert result["success"], result["reason"]
    dynamics = result["molecular_dynamics"]
    assert dynamics["mean_temperature_k"] == pytest.approx(85, abs=4) and dynamics["simulated_time_fs"] == 80.0
    # 2e-6 Ha per fs over 32 atoms.
    assert dynamics["conserved_drift_mev_per_atom_ps"] == pytest.approx(2e-6 * cp2k.HA_EV * 1e6 / 32, rel=1e-6)
    assert dynamics["rdf_first_maximum_angstrom"]["Ar–Ar"] == pytest.approx(5.26 / 2 ** 0.5, abs=0.15)
    assert {"final.cif", "md.csv", "rdf.csv", "md.png", "rdf.png", "dft_agent-1.ener"} <= set(result["artifacts"])
    assert len(Structure.from_file(folder / "final.cif")) == 32
    assert dynamics["stationary_second_half"] and not result["warnings"]
    assert "does not establish equilibrium" in dynamics["equilibration_note"]


def test_md_still_settling_is_flagged(tmp_path, argon):
    folder, structure = _staged(tmp_path, argon, {"mesh": [1, 1, 1], "electronic_type": "insulator", "md_temperature_k": 85,
                                                  "md_steps": 40, "md_timestep_fs": 2.0})
    _md_outputs(folder, structure, settling=2e-4)
    result = cp2k.analyze_outputs(folder, "md")
    # The run is valid; its averages are not equilibrium values, and the result says so.
    assert result["success"] and not result["molecular_dynamics"]["stationary_second_half"]
    assert result["molecular_dynamics"]["potential_energy_trend_mev_per_atom"] < 0
    assert "not in equilibrium yet" in result["warnings"][0] and "md_thermostat_fs 10" in result["warnings"][0]
    assert "K against the 85 K target" in result["warnings"][0]


def test_md_that_stops_early_or_loses_scf_is_refused(tmp_path, argon):
    folder, structure = _staged(tmp_path, argon, {"mesh": [1, 1, 1], "md_steps": 40})
    _md_outputs(folder, structure, finished=False)
    assert "stopped before its last step" in cp2k.analyze_outputs(folder, "md")["reason"]
    _md_outputs(folder, structure, failed_scf=True)
    assert "SCF did not converge" in cp2k.analyze_outputs(folder, "md")["reason"]


def test_scf_after_md_uses_the_final_frame_and_drops_the_old_density():
    sequence = engine_workflow._sequence(["scf", "md", "bands", "dos"], {"mesh": [2, 2, 2]}, None, "cp2k")
    assert [(s["name"], s["structure_from"], s["restart_from"]) for s in sequence] == [
        ("scf", None, None), ("md", None, None), ("scf", 1, None), ("bands", 1, 2), ("dos", 1, 2)]


def test_md_cartesian_trajectory_uses_cp2k_cell_frame(tmp_path):
    original = Structure(Lattice.from_parameters(8, 9, 10, 75, 80, 65), ["Ar", "Ar"], [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
    source = tmp_path / "rotated.cif"
    original.to(filename=str(source))
    folder, expected = _staged(tmp_path, source, {"mesh": [1, 1, 1], "md_steps": 4, "md_temperature_k": 85, "md_timestep_fs": 2})
    canonical = Lattice.from_parameters(*expected.lattice.abc, *expected.lattice.angles, vesta=True)
    assert not np.allclose(expected.lattice.matrix, canonical.matrix)
    native = Structure(canonical, ["Ar", "Ar"], expected.frac_coords)
    _md_outputs(folder, native, steps=4)
    frames, _ = cp2k._frames(folder / "dft_agent-pos-1.xyz", 2)
    result = cp2k.analyze_outputs(folder, "md")
    assert result["success"], result["reason"]
    final = Structure.from_file(folder / "final.cif")
    assert np.allclose(final.frac_coords, canonical.get_fractional_coords(frames[-1]) % 1, atol=1e-7)
    assert final.distance_matrix[0, 1] == pytest.approx(canonical.get_all_distances(
        canonical.get_fractional_coords(frames[-1]), canonical.get_fractional_coords(frames[-1]))[0, 1], abs=1e-6)


def test_md_trajectory_refuses_wrong_atom_order(tmp_path):
    trajectory = tmp_path / "traj.xyz"
    trajectory.write_text("2\n i = 0\n O 0 0 0\n H 1 0 0\n")
    with pytest.raises(ValueError, match="species differ"):
        cp2k._frames(trajectory, 2, ["H", "O"])
