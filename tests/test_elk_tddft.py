"""Elk optics (linear response, RPA and TDDFT kernels) and real-time TDDFT under a laser pulse.

The recorded silicon DOS run supplies a finished ground-state restart (task 1); the
spectra and time series are synthetic and test parsing, units and refusals only.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from vasp_slurm_agent import elk, engine_workflow

DATA = Path(__file__).parent / "data" / "engines"
SILICON = DATA / "Si-perturbed.cif"


def _staged(tmp_path, task, parameters):
    """The recorded restart outputs with the metadata a prepared optics or tddft stage writes."""
    inputs, folder = tmp_path / "inputs", tmp_path / "outputs"
    metadata = elk.prepare_inputs(DATA / "elk" / "04_dos" / "structure.cif", inputs, task, parameters)
    shutil.copytree(DATA / "elk" / "04_dos", folder)
    (folder / "elk-input.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "TDOS.OUT").unlink()
    return metadata, folder


def _epsilon(path, energies_ha, real, imaginary):
    rows = [f"{w:18.10E}{v:18.10E}" for w, v in zip(energies_ha, real)]
    rows += [""] + [f"{w:18.10E}{v:18.10E}" for w, v in zip(energies_ha, imaginary)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_optics_inputs_restart_the_ground_state_and_add_a_kernel(tmp_path):
    from pymatgen.core import Lattice, Structure

    ideal = tmp_path / "Si.cif"
    Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]).get_primitive_structure().to(filename=str(ideal))
    metadata = elk.prepare_inputs(ideal, tmp_path / "rpa", "optics", {"mesh": [4, 4, 4]})
    text = (tmp_path / "rpa" / "elk.in").read_text(encoding="utf-8")
    assert "tasks\n  1\n  120\n  121\n\n" in text and "320" not in text
    assert "ngridk\n  8 8 8\n" in text and "optcomp\n  1 1\n" in text and metadata["optics_components"] == [[1, 1]]
    assert "wplot\n  500 100 0\n" in text and text.count("swidth") == 1 and text.count("stype") == 1
    # Gaussian occupations: Fermi-Dirac tails at the 0.1 eV broadening half-filled bands near a small gap.
    assert "stype\n  0\n" in text
    kernel = elk.prepare_inputs(ideal, tmp_path / "boot", "optics",
                                {"tddft_kernel": "bootstrap", "scissor_ev": 0.6, "optics_components": ["xx", "zz"]})
    text = (tmp_path / "boot" / "elk.in").read_text(encoding="utf-8")
    assert "  121\n  320\n" in text and "fxctype\n  210\n" in text and "gmaxrf\n  3\n" in text
    assert "optcomp\n  1 1\n  3 3\n" in text and kernel["tddft_kernel"] == "bootstrap"
    assert math.isclose(float(text.split("scissor\n")[1].split()[0]) * elk.HA_EV, 0.6, rel_tol=1e-9)
    with pytest.raises(ValueError, match="rpa, alda or bootstrap"):
        elk.prepare_inputs(SILICON, tmp_path / "bad", "optics", {"tddft_kernel": "lrc"})
    with pytest.raises(ValueError, match="diagonal components"):
        elk.prepare_inputs(SILICON, tmp_path / "bad2", "optics", {"optics_components": ["xy"]})


def test_optics_and_pulse_stages_continue_one_scf():
    sequence = engine_workflow._sequence(["optics", "tddft"], {"tddft_kernel": "alda", "pulse_photon_ev": 3.0}, None, "elk")
    assert [(stage["name"], stage["restart_from"]) for stage in sequence] == [("scf", None), ("optics", 0), ("tddft", 0)]
    assert elk.RESTART_MAPPING["optics"] == elk.RESTART_MAPPING["tddft"] == {"STATE.OUT": "STATE.OUT"}
    with pytest.raises(ValueError, match="Unsupported task: optics"):
        engine_workflow._sequence(["optics"], {}, None, "cp2k")


def test_optics_spectra_are_parsed_plotted_and_summarised(tmp_path):
    _, folder = _staged(tmp_path, "optics", {"tddft_kernel": "bootstrap"})
    w = np.linspace(0, 15 / elk.HA_EV, 500, endpoint=False)
    peak = 4.0 / elk.HA_EV
    lorentz = 30 * 0.01 / ((w - peak) ** 2 + 0.01 ** 2) * 0.01
    _epsilon(folder / "EPSILON_11.OUT", w, 12.0 - 50 * w ** 2, lorentz)
    # Elk's TDDFT spectrum (task 320) starts at the first nonzero frequency (LiF on Arrhenius).
    _epsilon(folder / "EPSILON_TDDFT_11.OUT", w[1:], (11.0 - 45 * w ** 2)[1:], np.roll(lorentz, -20)[1:])
    result = elk.analyze_outputs(folder, "optics")
    assert result["success"], result["reason"]
    spectra = result["optics"]["spectra"]
    assert spectra["xx"]["absorption_peak_ev"] == pytest.approx(4.0, abs=0.04)
    assert spectra["xx, TDDFT bootstrap"]["absorption_peak_ev"] < spectra["xx"]["absorption_peak_ev"]
    # Re ε at zero frequency is reported as a dielectric constant only for a gapped crystal.
    assert result["band_gap"]["status"] == "gap" and spectra["xx"]["dielectric_constant"] == pytest.approx(12.0)
    assert spectra["xx, TDDFT bootstrap"]["dielectric_constant"] == pytest.approx(11.0 - 45 * w[1] ** 2)
    assert {"optics.csv", "optics.png", "EPSILON_11.OUT", "EPSILON_TDDFT_11.OUT"} <= set(result["artifacts"])
    lines = (folder / "optics.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "energy_ev,re_eps_xx,im_eps_xx,re_eps_xx_tddft_bootstrap,im_eps_xx_tddft_bootstrap"
    assert lines[1].endswith(",,") and lines[2].count(",") == 4 and not lines[2].endswith(",")
    (folder / "EPSILON_TDDFT_11.OUT").write_text("\n".join(f"{x:18.10E}{1.0:18.10E}" for x in w) + "\n", encoding="utf-8")
    assert "incomplete" in elk.analyze_outputs(folder, "optics")["reason"]


def test_pulse_inputs_use_atomic_units(tmp_path):
    metadata = elk.prepare_inputs(SILICON, tmp_path / "td", "tddft",
                                  {"pulse_photon_ev": 3.1, "pulse_field_v_per_angstrom": 1.0, "pulse_polarization": [1, 0, 0],
                                   "pulse_fwhm_fs": 2.0, "simulation_time_fs": 10.0, "time_step_as": 4.0})
    text = (tmp_path / "td" / "elk.in").read_text(encoding="utf-8")
    assert "tasks\n  1\n  450\n  460\n" in text
    line = text.split("pulse\n  1\n  ")[1].splitlines()[0].split()
    omega = 3.1 / elk.HA_EV
    # A0 = c E0 / ω, with E0 = 1 V/Å in atomic units.
    assert float(line[0]) == pytest.approx(elk.LIGHT_AU / elk.V_PER_ANGSTROM_PER_AU / omega, rel=1e-9)
    assert float(line[3]) == pytest.approx(omega) and float(line[6]) == pytest.approx(4.0 / elk.FS_PER_AU)
    assert float(text.split("tstime\n")[1].split()[0]) == pytest.approx(10.0 / elk.FS_PER_AU)
    # Elk checks read-back times to 1e-10, so the step is rounded to a short decimal in atomic units.
    assert text.split("dtimes\n")[1].split()[0] == "0.1654" and metadata["time_step_as"] == pytest.approx(4.0, abs=0.003)
    assert metadata["pulse"]["polarization"] == [1.0, 0.0, 0.0]
    magnet = elk.prepare_inputs(SILICON, tmp_path / "mag", "tddft", {"spin": "collinear", "initial_moments": {"Si": 1.0}})
    assert "dtimes\n  0.1\n" in (tmp_path / "mag" / "elk.in").read_text(encoding="utf-8") and magnet["time_step_as"] == pytest.approx(2.4189, abs=1e-4)
    with pytest.raises(ValueError, match="nonzero"):
        elk.prepare_inputs(SILICON, tmp_path / "bad", "tddft", {"pulse_polarization": [0, 0, 0]})
    # The zero-field reference Elk's authors ask for: on Arrhenius Ni's moment then stayed within 1e-9 over 15 fs.
    elk.prepare_inputs(SILICON, tmp_path / "zero", "tddft", {"pulse_field_v_per_angstrom": 0})
    assert [float(x) for x in (tmp_path / "zero" / "elk.in").read_text(encoding="utf-8").split("pulse\n  1\n  ")[1].split()[:3]] == [0, 0, 0]
    with pytest.raises(ValueError, match="peak field"):
        elk.prepare_inputs(SILICON, tmp_path / "negative", "tddft", {"pulse_field_v_per_angstrom": -1})


def _time_series(folder, steps=200, dt=0.2, finished=True, moment=None):
    times = np.arange(steps + 1) * dt
    field = 0.05 * np.exp(-((times - 20) / 8) ** 2) * np.cos(0.1 * times)
    (folder / "AFIELDT.OUT").write_text(f"{steps + 1:8d} : number of time steps\n"
                                        + "".join(f"{i + 1:8d}{t:18.10E}{0:18.10E}{0:18.10E}{a:18.10E}\n" for i, (t, a) in enumerate(zip(times, field))), encoding="utf-8")
    done = steps - 1 if finished else steps // 2
    written = times[1:done + 1]
    (folder / "TOTENERGY_TD.OUT").write_text("".join(f"{t:18.10E}{-578.0 + 0.001 * t / times[-1]:18.10E}\n" for t in written), encoding="utf-8")
    (folder / "JTOT_TD.OUT").write_text("".join(f"{t:18.10E}{0:18.10E}{0:18.10E}{1e-3 * math.sin(t):18.10E}\n" for t in written), encoding="utf-8")
    if moment is not None:
        (folder / "MOMENTM_TD.OUT").write_text("".join(f"{t:18.10E}{moment * (1 - 0.1 * t / times[-1]):18.10E}\n" for t in written), encoding="utf-8")
    log = (folder / "elk.log").read_text(encoding="utf-8")
    (folder / "elk.log").write_text(log.replace("Elk code stopped", "".join(
        f" Info(tddft): time step {i:8d} of {steps - 1:8d},   t = {t}\n" for i, t in enumerate(written, 1)) + " Elk code stopped"), encoding="utf-8")


def test_time_evolution_reports_field_current_energy_and_moment(tmp_path):
    _, folder = _staged(tmp_path, "tddft", {"spin": "collinear", "initial_moments": {"Si": 1.0}})
    _time_series(folder, moment=0.6)
    result = elk.analyze_outputs(folder, "tddft")
    assert result["success"], result["reason"]
    evolution = result["time_evolution"]
    # The first written step is t = 0.2 and the last t = 39.8 of a 40 a.u. run.
    assert evolution["absorbed_energy_ev"] == pytest.approx(0.001 * elk.HA_EV * 39.6 / 40, rel=1e-6)
    assert evolution["moment_change_percent"] == pytest.approx(100 * (0.9005 - 0.9995) / 0.9995, rel=1e-6)
    assert evolution["peak_field_v_per_angstrom"] > 0 and evolution["simulation_time_fs"] == pytest.approx(39.8 * elk.FS_PER_AU)
    assert {"tddft.csv", "tddft.png", "JTOT_TD.OUT", "MOMENTM_TD.OUT"} <= set(result["artifacts"])
    assert (folder / "tddft.csv").read_text(encoding="utf-8").splitlines()[0].endswith("energy_change_ev,moment_mu_b")


def test_time_evolution_ignores_the_static_field_iteration(tmp_path):
    """Task 460 rewrites INFO.OUT and EFERMI.OUT with one iteration under a large static A-field."""
    _, folder = _staged(tmp_path, "tddft", {})
    _time_series(folder)
    (folder / "INFO.OUT").write_text("│ Loop number : 1 │\n\nReached self-consistent loops maximum\n", encoding="utf-8")
    (folder / "EFERMI.OUT").write_text("   0.3463259169\n", encoding="utf-8")
    result = elk.analyze_outputs(folder, "tddft")
    assert result["success"], result["reason"]
    assert result["fermi_energy_ev"] is None and "band_gap" not in result and "INFO.OUT" not in result["artifacts"]
    # The restarted ground state (task 1) still has to converge.
    log = (folder / "elk.log").read_text(encoding="utf-8")
    (folder / "elk.log").write_text(log.replace("Elk code stopped", "Warning(gndstate): failed to reach self-consistency after 200 loops\n Elk code stopped", 1), encoding="utf-8")
    assert "did not reach" in elk.analyze_outputs(folder, "tddft")["reason"]


def test_time_evolution_that_stops_early_is_refused(tmp_path):
    _, folder = _staged(tmp_path, "tddft", {})
    _time_series(folder, finished=False)
    result = elk.analyze_outputs(folder, "tddft")
    assert not result["success"] and "stopped before the end" in result["reason"]


def test_magnetic_optics_does_not_assume_cubic_response(tmp_path):
    from pymatgen.core import Lattice, Structure
    crystal = tmp_path / 'Ni.cif'
    Structure.from_spacegroup('Fm-3m', Lattice.cubic(3.52), ['Ni'], [[0, 0, 0]]).get_primitive_structure().to(filename=str(crystal))
    metadata = elk.prepare_inputs(crystal, tmp_path / 'magnetic', 'optics', {'spin': 'noncollinear', 'soc': True, 'initial_moments': {'Ni': [0.0, 0.0, 1.0]}})
    assert metadata['optics_components'] == [[1, 1], [2, 2], [3, 3]]


def test_tddft_refuses_misaligned_spin_timestamps(tmp_path):
    _, folder = _staged(tmp_path, 'tddft', {'spin': 'collinear', 'initial_moments': {'Si': 1.0}})
    _time_series(folder, moment=0.6)
    values = np.loadtxt(folder / 'MOMENTM_TD.OUT')
    values[:, 0] += 1.0
    np.savetxt(folder / 'MOMENTM_TD.OUT', values)
    result = elk.analyze_outputs(folder, 'tddft')
    assert not result['success'] and 'disagrees' in result['reason']
