"""Unit/contract checks only: these tests do not run or scientifically validate VASP."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from vasp_slurm_agent import vasp
from vasp_slurm_agent.numerical_defaults import POTCAR_RECOMMENDATIONS


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", ["Si", "C", "Ge", "Al", "Cu", "MgO", "NaCl", "SiC"])
def test_public_examples_prepare_without_licensed_data(tmp_path, name):
    result = vasp.prepare_inputs(EXAMPLES / f"{name}.cif", tmp_path, "scf", {})
    poscar = Poscar.from_file(tmp_path / "POSCAR", check_for_potcar=False)
    incar = Incar.from_file(tmp_path / "INCAR")
    assert result["potcar_labels"] == [POTCAR_RECOMMENDATIONS.get(symbol, symbol) for symbol in poscar.site_symbols]
    assert incar["LCHARG"] is True
    assert incar["ISPIN"] == 1 and incar["GGA"] == "Pe"
    assert incar["ENCUT"] == 520 and incar["EDIFF"] == 1e-5
    assert incar["NSW"] == 0 and incar["ICHARG"] == 2
    assert Kpoints.from_file(tmp_path / "KPOINTS").kpts == [tuple(result["parameters"]["mesh"])]
    assert result["numerical_choices"]["mesh_mode"] == "reciprocal_spacing"
    assert not (tmp_path / "POTCAR").exists()
    assert len(result["input_sha256"]) == 3
    json.dumps(result, allow_nan=False)


def test_relax_and_dos_recipe(tmp_path):
    source = EXAMPLES / "NaCl.cif"
    result = vasp.prepare_inputs(source, tmp_path / "relax", "relax", {"cell_relax": True, "kpoint_grid": [6, 6, 6]}, {"Na": "Na_pv", "Fe": "Fe_pv"})
    assert result["potcar_labels"] == ["Na_pv", "Cl"]
    relax = Incar.from_file(tmp_path / "relax/INCAR")
    assert relax["NSW"] == 100 and relax["IBRION"] == 2
    assert relax["EDIFFG"] == -0.03 and relax["ISIF"] == 3
    result = vasp.prepare_inputs(source, tmp_path / "dos", "dos", {"mesh": [8, 8, 8]})
    dos = Incar.from_file(tmp_path / "dos/INCAR")
    assert dos["NSW"] == 0 and dos["ICHARG"] == 11 and dos["NEDOS"] == 2001
    assert dos["NELM"] == 120 and result["requires_chgcar"]


def test_bands_coordinates_use_actual_nonprimitive_poscar_cell(tmp_path):
    # Conventional cubic Si differs from the standardized primitive path basis.
    structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    source = tmp_path / "conventional.cif"
    structure.to(filename=str(source))
    result = vasp.prepare_inputs(source, tmp_path / "bands", "bands", {"line_density": 6})
    actual = Poscar.from_file(tmp_path / "bands/POSCAR", check_for_potcar=False).structure
    kpoints = Kpoints.from_file(tmp_path / "bands/KPOINTS")
    reconstructed_cartesian = actual.lattice.reciprocal_lattice.get_cartesian_coords(kpoints.kpts)
    assert len(actual) == 8  # no unrequested primitive-cell change
    np.testing.assert_allclose(reconstructed_cartesian, result["band_path"]["cartesian_inv_angstrom"], atol=1e-8)
    incar = Incar.from_file(tmp_path / "bands/INCAR")
    assert incar["ICHARG"] == 11 and incar["NSW"] == 0
    assert incar["NELM"] > 1


def test_rotating_crystal_rotates_the_cartesian_band_path(tmp_path):
    # A POSCAR preserves Cartesian orientation, whereas a CIF may canonicalize it.
    a = 5.431
    base = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    paths = []
    for name, matrix in [("base", base), ("rotated", base @ rotation.T)]:
        structure = Structure(matrix, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        source = tmp_path / f"POSCAR.{name}"
        Poscar(structure).write_file(source)
        paths.append(vasp.prepare_inputs(source, tmp_path / name, "bands", {"line_density": 6})["band_path"])
    np.testing.assert_allclose(paths[0]["reciprocal_fractional"], paths[1]["reciprocal_fractional"], atol=1e-7)
    np.testing.assert_allclose(np.asarray(paths[0]["cartesian_inv_angstrom"]) @ rotation.T, paths[1]["cartesian_inv_angstrom"], atol=1e-7)


def test_relaxed_cell_with_residual_distortion_keeps_its_band_path(tmp_path):
    # Relaxations from a displaced start leave ~0.003 Å of distortion; silicon must stay fcc.
    from vasp_slurm_agent import qe

    a = 5.431
    lattice = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2 + 0.003, a/2, 0]])
    structure = Structure(lattice, ["Si", "Si"], [[0.0006, 0.0003, 0.9996], [0.25, 0.25, 0.25]])
    source = tmp_path / "POSCAR"
    Poscar(structure).write_file(source)
    path = vasp.prepare_inputs(source, tmp_path / "bands", "bands", {"line_density": 6})["band_path"]
    assert [label for label in path["labels"] if label] == ["GAMMA", "X", "U", "K", "GAMMA", "L", "W", "X"]
    assert path["symprec_angstrom"] == 0.02
    assert [label for label in qe._path(structure, {"line_density": 6})["labels"] if label] == \
        ["GAMMA", "X", "U", "K", "GAMMA", "L", "W", "X"]


@pytest.mark.parametrize("parameters", [
    {"mesh": [0, 4, 4]}, {"mesh": [4, 4]}, {"ediff": float("nan")},
    {"encut": -1}, {"nsw": True}, {"cell_relax": "false"},
    {"ediffg": 0.03}, {"magmom": [1]}, {"mesh": [4, 4, 4], "kpoint_grid": [6, 6, 6]},
])
def test_invalid_parameters_refused_before_files(tmp_path, parameters):
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError):
        vasp.prepare_inputs(EXAMPLES / "Si.cif", destination, "scf", parameters)
    assert not destination.exists()


def test_overlap_and_unsafe_potential_label_refused(tmp_path):
    source = tmp_path / "bad.cif"
    Structure(Lattice.cubic(4), ["Si", "C"], [[0, 0, 0], [0.001, 0, 0]]).to(filename=str(source))
    with pytest.raises(ValueError, match="overlapping"):
        vasp.prepare_inputs(source, tmp_path / "bad", "scf", {})
    with pytest.raises(ValueError, match="POTCAR labels"):
        vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path / "bad", "scf", {}, {"Si": "../../POTCAR"})


@pytest.mark.parametrize("xml", [None, "", "128.0000 </r>\n</modeling>", "<modeling><calculation>"])
def test_missing_or_truncated_xml_is_not_success(tmp_path, xml):
    if xml is not None:
        (tmp_path / "vasprun.xml").write_text(xml)
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and not result["valid_structure"]
    assert not (tmp_path / "final_structure.cif").exists()
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_missing_vasp_licence_is_named(tmp_path):
    # A central VASP 6 module on Arrhenius wrote an empty vasprun.xml and this banner without a licence key file.
    (tmp_path / "vasprun.xml").write_text("")
    (tmp_path / "slurm.out").write_text(" |     No license file found. Please provide a valid license file. VASP        |\n")
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and "no licence file" in result["reason"] and "vasp_license" in result["reason"]


def _fake_run(**changes):
    """A parser stub for analyzer logic, deliberately not scientific evidence."""
    structure = Structure.from_file(EXAMPLES / "Si.cif").get_sorted_structure()
    run = SimpleNamespace(
        vasp_version="test-stub", initial_structure=structure,
        final_structure=structure.copy(), final_energy=-10.0,
        efermi=0.0,
        converged_electronic=True, converged_ionic=True,
        ionic_steps=[{"forces": [[0, 0, 0], [0, 0, 0]], "e_0_energy": -10.0}],
        incar={"EDIFFG": -0.03, "NSW": 0, "ICHARG": 2},
    )
    for key, value in changes.items():
        setattr(run, key, value)
    return run


def _install_parser_stub(monkeypatch, tmp_path, run):
    (tmp_path / "vasprun.xml").write_text("parser stub only")
    def parser(path, **kwargs):
        assert kwargs["exception_on_bad_xml"] is True
        assert kwargs["parse_potcar_file"] is False
        return run
    monkeypatch.setattr(vasp, "Vasprun", parser)


@pytest.mark.parametrize("change,reason", [
    ({"converged_electronic": False}, "Electronic convergence"),
    ({"final_energy": float("nan")}, "Final energy"),
    ({"final_structure": Structure(Lattice.cubic(5), ["C"], [[0, 0, 0]])}, "atom count or composition"),
    ({"initial_structure": Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])}, "initial structure differs"),
    ({"incar": {"NSW": 100}}, "NSW != 0"),
])
def test_analyzer_refuses_numerical_or_identity_failure(monkeypatch, tmp_path, change, reason):
    _install_parser_stub(monkeypatch, tmp_path, _fake_run(**change))
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and reason in result["reason"]
    json.dumps(result, allow_nan=False)


def test_relax_force_gate_is_independent_of_parser_convergence(monkeypatch, tmp_path):
    run = _fake_run(ionic_steps=[{"forces": [[0.04, 0, 0], [0, 0, 0]], "e_0_energy": -10}], incar={"NSW": 100, "IBRION": 2, "EDIFFG": -0.03})
    _install_parser_stub(monkeypatch, tmp_path, run)
    result = vasp.analyze_outputs(tmp_path, "relax", EXAMPLES / "Si.cif")
    assert result["converged_ionic"] is False
    assert result["ionic_steps_count"] == 1
    assert result["ionic_iteration_limit_reached"] is False
    assert result["ionic_force_limit_ev_angstrom"] == .03
    assert not result["success"] and "force exceeds" in result["reason"]


def test_numerically_accepted_stub_exports_no_science_claim(monkeypatch, tmp_path):
    _install_parser_stub(monkeypatch, tmp_path, _fake_run())
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert result["success"] and result["valid_structure"]
    assert result["scientific_accuracy_validated"] is False
    assert result["final_energy_ev"] == -10.0
    assert (tmp_path / "final_structure.cif").is_file()


def test_relax_curve_is_exported_for_accepted_stub(monkeypatch, tmp_path):
    _install_parser_stub(monkeypatch, tmp_path, _fake_run(incar={"NSW": 100, "IBRION": 2, "EDIFFG": -0.03}))
    result = vasp.analyze_outputs(tmp_path, "relax", EXAMPLES / "Si.cif")
    assert result["success"]
    assert (tmp_path / "relax_energy.csv").read_text().startswith("ionic_step,energy_ev")
    assert (tmp_path / "relax_energy.png").stat().st_size > 100


def test_dos_export_contract_with_parser_stub(monkeypatch, tmp_path):
    from pymatgen.electronic_structure.core import Spin
    dos = SimpleNamespace(energies=np.array([-30.0, -1.0, 0, 1.0, 70.0]), densities={Spin.up: np.array([40.0, 0.0, 1.0, 0.0, 60.0])})
    run = _fake_run(incar={"NSW": 0, "ICHARG": 11}, efermi=0.0, complete_dos=dos)
    _install_parser_stub(monkeypatch, tmp_path, run)
    result = vasp.analyze_outputs(tmp_path, "dos", EXAMPLES / "Si.cif")
    assert result["success"]
    assert len((tmp_path / "dos.csv").read_text().splitlines()) == 6
    assert "70.0,60.0" in (tmp_path / "dos.csv").read_text()
    assert result["plot_settings"] == {"energy_window_ev": [-8.0, 8.0], "csv_contains_full_data": True}
    assert (tmp_path / "dos.png").stat().st_size > 100


def test_band_window_stops_below_the_highest_computed_band(monkeypatch, tmp_path):
    # With KPAR, a silicon HSE06 path had 8 bands; the highest follows whichever uncomputed band crosses it.
    from pymatgen.electronic_structure.core import Spin
    metadata = vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path, "bands", {"line_density": 3})
    metadata["scf_fermi_energy_ev"] = 0.0
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    points = metadata["band_path"]["reciprocal_fractional"]
    values = np.zeros((len(points), 2, 2))
    values[:, 0, 0] = -1.0
    values[:, 1, 0] = np.linspace(3.0, 7.5, len(points))
    run = _fake_run(incar=dict(Incar.from_file(tmp_path / "INCAR")), efermi=0.0, actual_kpoints=points, eigenvalues={Spin.up: values})
    _install_parser_stub(monkeypatch, tmp_path, run)
    result = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert result["success"] and result["plot_settings"]["energy_window_ev"] == [-8.0, 2.9]
    assert json.loads((tmp_path / "plot_metadata.json").read_text())["energy_window_ev"] == [-8.0, 2.9]


def test_bands_export_checks_frozen_path_with_parser_stub(monkeypatch, tmp_path):
    from pymatgen.electronic_structure.core import Spin
    metadata = vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path, "bands", {"line_density": 3})
    metadata["scf_fermi_energy_ev"] = 0.0
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    points = metadata["band_path"]["reciprocal_fractional"]
    values = np.zeros((len(points), 2, 2))
    values[:, 0, 0] = -1.0
    values[:, 0, 1] = 2.0
    values[:, 1, 0] = 70.0  # Full CSV retains even bands outside the display window.
    run = _fake_run(incar=dict(Incar.from_file(tmp_path / "INCAR")), efermi=123.0, actual_kpoints=points, eigenvalues={Spin.up: values})
    _install_parser_stub(monkeypatch, tmp_path, run)
    result = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert result["success"]
    assert result["energy_reference_ev"] == 0.0
    assert result["plot_settings"] == {"energy_window_ev": [-8.0, 8.0], "csv_contains_full_data": True}
    assert ",70.0," in (tmp_path / "bands.csv").read_text()
    assert len((tmp_path / "bands.csv").read_text().splitlines()) == 1 + 2 * len(points)
    assert (tmp_path / "bands.png").stat().st_size > 100
    changed = np.asarray(points).copy()
    changed[0, 0] += 0.1
    run.actual_kpoints = changed
    refused = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert not refused["success"] and "frozen band path" in refused["reason"]
    assert not (tmp_path / "bands.png").exists()
    assert not (tmp_path / "final_structure.cif").exists()
    run.actual_kpoints = points
    metadata.pop("scf_fermi_energy_ev")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    refused = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert not refused["success"] and "SCF Fermi energy is required" in refused["reason"]
