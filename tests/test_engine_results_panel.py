"""The run panel's ABINIT, CP2K and Elk result cards render each program's own results and warnings."""
from streamlit.testing.v1 import AppTest

SCRIPT = """
import streamlit as st
from vasp_slurm_agent.engine_app import stage_results
stage_results(st.session_state["run_dir"], st.session_state["stage"])
"""


def _render(tmp_path, engine, name, result):
    (tmp_path / "01" / "outputs").mkdir(parents=True)
    app = AppTest.from_string(SCRIPT)
    app.session_state["run_dir"] = str(tmp_path)
    app.session_state["stage"] = {"folder": "01", "name": name, "metadata": {"engine": engine},
                                  "result": {"success": True, "warnings": [], **result}}
    return app.run()


def test_phonon_card_shows_frequencies_dielectric_constant_and_born_charges(tmp_path):
    app = _render(tmp_path, "abinit", "phonons", {"phonons": {
        "max_frequency_cm1": 271.3, "min_frequency_cm1": 0.0, "dielectric_constant": 13.1,
        "born_charges": [{"atom": 1, "element": "Ga", "born_charge": 2.1}, {"atom": 2, "element": "As", "born_charge": -2.1}]}})
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Highest frequency", "Lowest frequency", "Dielectric constant"]
    assert app.metric[0].value == "271.3 cm⁻¹" and app.metric[2].value == "13.1"
    assert list(app.dataframe[0].value["Born charge"]) == [2.1, -2.1]


def test_molecular_dynamics_card_shows_its_equilibrium_warning(tmp_path):
    warning = "The second half of the run is not in equilibrium yet: the potential energy still falls by 19 meV/atom."
    app = _render(tmp_path, "cp2k", "md", {"warnings": [warning], "molecular_dynamics": {
        "mean_temperature_k": 397.0, "simulated_time_fs": 500.0, "conserved_drift_mev_per_atom_ps": -0.33}})
    assert not app.exception and app.warning[0].value == warning
    assert [metric.label for metric in app.metric] == ["Mean temperature", "Simulated time", "Energy drift"]


def test_optics_and_laser_cards(tmp_path):
    app = _render(tmp_path, "elk", "optics", {"optics": {"spectra": {"xx": {"absorption_peak_ev": 3.72, "dielectric_constant": 13.5}}}})
    assert not app.exception and app.dataframe[0].value["Dielectric constant"][0] == 13.5
    app = _render(tmp_path / "td", "elk", "tddft", {"time_evolution": {"absorbed_energy_ev": 1.99, "peak_field_v_per_angstrom": 1.0,
                                                                        "moment_change_percent": 0.005}})
    assert not app.exception and [metric.label for metric in app.metric] == ["Absorbed energy", "Peak field", "Moment change"]
