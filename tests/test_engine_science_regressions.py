"""Regressions for metallic gaps and parameters reaching native engine inputs."""
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from vasp_slurm_agent import cp2k, elk
from vasp_slurm_agent.engine_common import HA_EV, band_gap


DATA = Path(__file__).parent / "data" / "engines"
STRUCTURE = DATA / "Si-perturbed.cif"


@pytest.mark.parametrize("electrons", [None, 1, 2])
def test_a_band_crossing_fermi_between_samples_has_no_gap(electrons):
    # Neither k point is within 20 meV of EF, but the first band crosses it.
    result = band_gap([[[-1.0, 2.0], [1.0, 3.0]]], electrons=electrons, fermi=0.0)
    assert result["gap_ev"] == 0.0
    assert result["status"] == "metallic_or_partially_occupied"


def test_one_metallic_spin_channel_closes_the_gap():
    energies = [[[-0.8, 1.2], [0.4, 1.8]], [[-1.5, 0.6], [-1.0, 0.9]]]
    result = band_gap(energies, fermi=0.0)
    assert result["gap_ev"] == 0.0 and result["metallic_or_smeared"]


def test_overlapping_valence_and_conduction_bands_are_semimetallic():
    result = band_gap([[[-1.0, -0.1], [0.2, 0.8]]], electrons=2)
    assert result["vbm_ev"] > result["cbm_ev"]
    assert result["gap_ev"] == 0.0


@pytest.mark.parametrize("electrons", [None, 2])
def test_a_small_positive_gap_is_not_closed_by_proximity_to_fermi(electrons):
    result = band_gap([[[-0.005, 0.007], [-0.008, 0.005]]], electrons=electrons, fermi=0.0)
    assert result["status"] == "gap"
    assert result["gap_ev"] == pytest.approx(0.01)


def test_bands_touching_at_the_fermi_level_have_no_gap():
    # Elk put graphene's Fermi level 2e-6 eV from its Dirac point; both K states then counted as
    # occupied and the next path point gave a 0.13 eV "gap".
    path = [[[-3.0, 3.0], [-0.4, 0.3], [-1e-6, 2e-6], [-0.5, 0.134]]]
    result = band_gap(path, fermi=0.0)
    assert result["gap_ev"] == 0.0 and result["status"] == "metallic_or_partially_occupied"


def test_fermi_at_the_homo_does_not_make_an_insulator_metallic():
    result = band_gap([[[-1.0, 1.2], [0.0, 0.8]]], fermi=0.0)
    assert result["status"] == "gap" and result["gap_ev"] == pytest.approx(0.8)


def test_thermal_occupation_tails_do_not_close_a_spectral_gap():
    energies = np.array([[[-0.15, 0.20], [-0.12, 0.18]]])
    occupations = 2 / (1 + np.exp(energies / 0.05))
    result = band_gap(energies, occupations=occupations, fermi=0.0)
    assert result["status"] == "gap"
    assert result["gap_ev"] == pytest.approx(0.30)
    assert result["partial_occupations"]


def test_occupation_crossing_is_used_when_fermi_is_missing():
    result = band_gap([[[-0.8, 1.2], [0.4, 1.8]]], occupations=[[[1.99, 0.0], [0.01, 0.0]]])
    assert result["status"] == "metallic_or_partially_occupied"
    assert result["gap_ev"] == 0.0


def test_half_filled_spinor_level_has_no_gap():
    result = band_gap([[[-1.0, 0.0, 0.8]]], occupations=[[[1.0, 0.5, 0.0]]], full_occupation=1)
    assert result["gap_ev"] == 0.0


def test_invalid_occupations_cannot_be_used_as_gap_evidence():
    result = band_gap([[[-1.0, 0.8]]], occupations=[[[1.0, float("nan")]]])
    assert not result["available"]


def test_elk_spinor_gap_uses_one_electron_per_state():
    result = elk._occupied_gap(np.array([[[-0.2, 0.3]]]), np.array([[[0.98, 0.02]]]),
                               "sampled_k_points", full_occupation=1, fermi=0.0)
    assert result["status"] == "gap" and result["gap_ev"] == pytest.approx(0.5)
    assert result["partial_occupations"]


def test_elk_methfessel_paxton_weights_may_leave_the_occupation_interval():
    # Native Al with Elk stype=1 had weights down to -0.04635 and above 2.
    energies = np.array([[[-0.8, 1.2], [0.4, 1.8]]])
    weights = np.array([[[2.000000125, 0.0], [-0.046350578, 0.0]]])
    result = elk._occupied_gap(energies, weights, "sampled_k_points", fermi=0.0)
    assert result["status"] == "metallic_or_partially_occupied"
    assert result["gap_ev"] == 0.0
    weights[0, 0, 0] = float("nan")
    assert not elk._occupied_gap(energies, weights, "sampled_k_points", fermi=0.0)["available"]


def test_cp2k_band_parser_retains_the_recorded_occupations():
    sets = cp2k._band_file(DATA / "cp2k" / "03_bands" / "dft_agent.bs")
    for _, _, energies, occupations in sets:
        assert occupations.shape == energies.shape
        assert np.allclose(occupations[:, :, :4], 2.0)
        assert np.allclose(occupations[:, :, 4:], 0.0)


def test_cp2k_recorded_bands_still_have_the_silicon_gap(tmp_path):
    folder = tmp_path / "bands"
    shutil.copytree(DATA / "cp2k" / "03_bands", folder)
    result = cp2k.analyze_outputs(folder, "bands")
    assert result["success"], result["reason"]
    assert result["band_gap"]["status"] == "gap"
    assert 0.3 < result["band_gap_ev"] < 1.2


@pytest.mark.parametrize("extra", [0, 2, 60])
def test_cp2k_common_extra_bands_reaches_scf_and_band_print(tmp_path, extra):
    meta = cp2k.prepare_inputs(STRUCTURE, tmp_path, "bands", {"extra_bands": extra})
    deck = (tmp_path / cp2k.INPUT_FILE).read_text()
    assert re.findall(r"\bADDED_MOS (\d+)", deck) == [str(extra), str(extra)]
    assert meta["added_mos"] == extra


@pytest.mark.parametrize("task", ["relax", "scf", "bands", "dos"])
def test_cp2k_preserves_odd_mesh_without_the_broken_k290_symmetry_path(tmp_path, task):
    meta = cp2k.prepare_inputs(STRUCTURE, tmp_path, task, {"mesh": [5, 5, 5], "dos_mesh": [5, 5, 5]})
    deck = (tmp_path / cp2k.INPUT_FILE).read_text()
    assert meta["mesh"] == [5, 5, 5]
    assert "SCHEME MONKHORST-PACK 5 5 5" in deck
    assert "SYMMETRY OFF" in deck and "FULL_GRID OFF" in deck
    assert "SYMMETRY ON" not in deck


def test_cp2k_conflicting_empty_state_aliases_are_refused(tmp_path):
    with pytest.raises(ValueError, match="extra_bands and added_mos"):
        cp2k.prepare_inputs(STRUCTURE, tmp_path, "scf", {"extra_bands": 4, "added_mos": 8})
    meta = cp2k.prepare_inputs(STRUCTURE, tmp_path, "scf", {"extra_bands": 4, "added_mos": 4})
    assert meta["added_mos"] == 4


@pytest.mark.parametrize("extra", [0, 3, 60])
def test_elk_common_extra_bands_is_per_cell_not_per_atom(tmp_path, extra):
    meta = elk.prepare_inputs(STRUCTURE, tmp_path, "scf", {"extra_bands": extra})
    deck = (tmp_path / elk.INPUT_FILE).read_text()
    native = float(re.search(r"^nempty\n\s+(\S+)", deck, re.M)[1])
    assert native * meta["number_of_atoms"] == pytest.approx(extra)
    assert meta["nempty"] == pytest.approx(extra / 2)


def test_elk_accepts_consistent_fractional_nempty_but_refuses_conflicts(tmp_path):
    meta = elk.prepare_inputs(STRUCTURE, tmp_path, "scf", {"extra_bands": 3, "nempty": 1.5})
    assert meta["nempty"] == 1.5
    with pytest.raises(ValueError, match="per atom and spin"):
        elk.prepare_inputs(STRUCTURE, tmp_path / "bad", "scf", {"extra_bands": 3, "nempty": 3})


@pytest.mark.parametrize("eps_scf", [None, 1e-8])
def test_cp2k_energy_threshold_is_not_silently_converted_to_a_residual(tmp_path, eps_scf):
    parameters = {"energy_tolerance_ev": 1e-8}
    if eps_scf is not None:
        parameters["eps_scf"] = eps_scf
    with pytest.raises(ValueError, match="residual tolerance, not an energy threshold"):
        cp2k.prepare_inputs(STRUCTURE, tmp_path, "scf", parameters)
    assert not (tmp_path / cp2k.INPUT_FILE).exists()


@pytest.mark.parametrize("threshold", [1e-6, 1e-10])
def test_native_thresholds_reach_the_input_with_correct_units(tmp_path, threshold):
    cp2k.prepare_inputs(STRUCTURE, tmp_path / "cp2k", "scf", {"eps_scf": threshold})
    cp2k_deck = (tmp_path / "cp2k" / cp2k.INPUT_FILE).read_text()
    assert float(re.search(r"EPS_SCF (\S+)", cp2k_deck)[1]) == pytest.approx(threshold, rel=1e-12, abs=0)
    elk.prepare_inputs(STRUCTURE, tmp_path / "elk", "scf", {"energy_tolerance_ev": threshold})
    elk_deck = (tmp_path / "elk" / elk.INPUT_FILE).read_text()
    assert float(re.search(r"^epsengy\n\s+(\S+)", elk_deck, re.M)[1]) == pytest.approx(threshold / HA_EV, rel=1e-12, abs=0)
