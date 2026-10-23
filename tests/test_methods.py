"""Method contracts and parser stubs; these tests do not execute VASP."""

from __future__ import annotations

import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Vasprun

from vasp_slurm_agent import vasp
from vasp_slurm_agent.methods import HYBRID_BAND_NELMIN, validate_method_output

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _source(tmp_path):
    path = tmp_path / "POSCAR.upload"
    structure = Structure(Lattice.cubic(6), ["O", "Fe", "O", "Fe"], [[0, 0, 0], [.5, 0, 0], [0, .5, 0], [.5, .5, .5]])
    Poscar(structure).write_file(path)
    return path


def test_collinear_site_order_and_method_identity(tmp_path):
    source = _source(tmp_path)
    parameters = {"spin": "collinear", "magmom": [.1, 4, -.1, -4]}
    meta = vasp.prepare_inputs(source, tmp_path / "afm", "scf", parameters)
    incar = Incar.from_file(tmp_path / "afm/INCAR")
    assert meta["input_site_order"] == [1, 3, 0, 2]
    assert incar["MAGMOM"] == [4, -4, .1, -.1]
    assert incar["ISPIN"] == 2 and incar["ISYM"] == 2 and incar["LORBIT"] == 11
    assert not meta["requires_ncl"]
    fm = vasp.prepare_inputs(source, tmp_path / "fm", "scf", {**parameters, "magmom": [.1, 4, .1, 4]})
    assert fm["method_fingerprint"] != meta["method_fingerprint"]
    assert fm["method_comparison_fingerprint"] == meta["method_comparison_fingerprint"]
    assert meta["method"]["magmom"] == incar["MAGMOM"]
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("soc", [False, True])
def test_noncollinear_vectors_follow_sites_and_spinor_axis(tmp_path, soc):
    moments = [[0, 0, .1], [4, 1, 0], [0, 0, -.1], [-4, -1, 0]]
    meta = vasp.prepare_inputs(_source(tmp_path), tmp_path / "ncl", "scf", {"spin": "noncollinear", "magmom": moments, "soc": soc, "saxis": [2, 0, 0]})
    incar = Incar.from_file(tmp_path / "ncl/INCAR")
    assert incar["ISPIN"] == 1 and incar["LNONCOLLINEAR"] and incar["LSORBIT"] == soc
    assert not incar["GGA_COMPAT"] and incar["ISYM"] == -1
    assert incar["SAXIS"] == [1, 0, 0] and meta["requires_ncl"]
    np.testing.assert_allclose(np.asarray(incar["MAGMOM"]).reshape(-1, 3), np.asarray(moments)[[1, 3, 0, 2]])


def test_soc_nonmagnetic_has_explicit_zero_moments(tmp_path):
    meta = vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path, "scf", {"soc": True})
    incar = Incar.from_file(tmp_path / "INCAR")
    assert np.asarray(incar["MAGMOM"]).reshape(-1).tolist() == [0] * 6
    assert incar["ISPIN"] == 1 and incar["LSORBIT"] and meta["requires_ncl"]


def test_u_block_uses_actual_species_order_and_lmaxmix(tmp_path):
    source = _source(tmp_path)
    parameters = {"hubbard_u": {"Fe": {"l": 2, "u": 5, "j": 1}}}
    for task in ("scf", "dos"):
        meta = vasp.prepare_inputs(source, tmp_path / task, task, parameters)
        incar = Incar.from_file(tmp_path / task / "INCAR")
        assert meta["potcar_elements"] == ["Fe", "O"]
        assert incar["LDAU"] and incar["LDAUTYPE"] == 2
        assert incar["LDAUL"] == [2, -1] and incar["LDAUU"] == [5, 0] and incar["LDAUJ"] == [1, 0]
        assert incar["LMAXMIX"] == 4
    assert meta["requires_chgcar"]
    other = tmp_path / "POSCAR.Ce"
    Poscar(Structure(Lattice.cubic(5), ["Ce"], [[0, 0, 0]])).write_file(other)
    vasp.prepare_inputs(other, tmp_path / "f", "scf", {"hubbard_u": {"Ce": {"l": 3, "u": 4, "j": 0}}})
    assert Incar.from_file(tmp_path / "f/INCAR")["LMAXMIX"] == 6


def test_vasp5_ldautype_xml_scalar_vector_matches_prepared_method():
    # Shape observed in VASP 5.4.4 INCAR XML; species arrays remain arrays.
    elem = ET.fromstring('''<incar>
      <i type="logical" name="LDAU">T</i>
      <v type="int" name="LDAUTYPE">2</v>
      <v type="int" name="LDAUL">2 -1</v>
      <v name="LDAUU">5.00000000 0.00000000</v>
      <v name="LDAUJ">0.00000000 0.00000000</v>
    </incar>''')
    parser = object.__new__(Vasprun)
    parser.filename = "vasprun.xml"
    actual = parser._parse_params(elem)
    assert actual["LDAUTYPE"] == [2]
    expected = {"LDAU": True, "LDAUTYPE": 2, "LDAUL": [2, -1], "LDAUU": [5., 0.], "LDAUJ": [0., 0.]}
    validate_method_output(actual, {"method_incar_expected": expected}, vasp_version="5.4.4.18Apr17-6-g9f103f2a35")
    assert actual["LDAUTYPE"] == [2]


@pytest.mark.parametrize("actual,version", [([1], "5.4.4"), ([2, 2], "5.4.4"),
                                            ([[2]], "5.4.4"), ([True], "5.4.4"),
                                            ([2], "6.2.1"), ([2], "")])
def test_ldautype_normalization_does_not_hide_mismatch(actual, version):
    with pytest.raises(ValueError, match="Method mismatch.*LDAUTYPE"):
        validate_method_output({"LDAUTYPE": actual}, {"method_incar_expected": {"LDAUTYPE": 2}}, vasp_version=version)


@pytest.mark.parametrize("functional", ["HSE06", "PBE0"])
@pytest.mark.parametrize("task", ["relax", "scf", "bands", "dos"])
def test_hybrid_recipes_are_self_consistent(tmp_path, functional, task):
    meta = vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path, task, {"functional": functional, "mesh": [2, 2, 2], "line_density": 3})
    incar = Incar.from_file(tmp_path / "INCAR")
    assert incar["LHFCALC"] and incar["AEXX"] == .25
    assert incar["HFSCREEN"] == (.2 if functional == "HSE06" else 0)
    # Only unscreened exchange truncates the Coulomb kernel; with HSE06 VASP 5.4.4 then lost the screening.
    assert incar.get("HFRCUT") == (None if functional == "HSE06" else -1)
    assert incar["ALGO"].lower() == ("normal" if task == "bands" else "damped")
    assert incar["ICHARG"] == 2 and incar["ISTART"] == 0
    assert not meta["requires_chgcar"] and meta["spectral_charge_mode"] == "self_consistent"
    if task == "bands":
        assert incar["IMIX"] == 1 and incar["AMIX"] == .2 and incar["LFOCKACE"]
        assert "TIME" not in incar
        kpoints = Kpoints.from_file(tmp_path / "KPOINTS")
        assert meta["band_path_offset"] == 8
        assert len(kpoints.kpts) == 8 + meta["band_path_count"]
        assert kpoints.kpts_weights[:8] == [1] * 8
        assert kpoints.kpts_weights[8:] == [0] * meta["band_path_count"]
        assert len(set(tuple(point) for point in kpoints.kpts[:8])) == 8
        np.testing.assert_allclose(kpoints.kpts[8:], meta["band_path"]["reciprocal_fractional"])
        assert incar["NELMIN"] == HYBRID_BAND_NELMIN and incar["ISYM"] == -1
    else:
        assert incar["TIME"] == .4


@pytest.mark.parametrize("parameters", [
    {"spin": "collinear"}, {"spin": "noncollinear"}, {"spin": "other"},
    {"spin": "collinear", "magmom": [1]}, {"spin": "collinear", "magmom": [True, 1]},
    {"spin": "collinear", "magmom": [1, 1], "soc": True},
    {"spin": "noncollinear", "magmom": [1, -1]},
    {"spin": "noncollinear", "magmom": [[1, 0, 0], [float("nan"), 0, 0]]},
    {"magmom": [1, 1]}, {"soc": "true"}, {"saxis": [0, 0, 0]}, {"saxis": [1, 0, 0]},
    {"functional": "B3LYP"}, {"hubbard_u": {"Fe": {"l": 2, "u": 4, "j": 0}}},
    {"hubbard_u": {"Si": {"u": 4}}}, {"hubbard_u": {"Si": {"l": 2, "u": 1, "j": 2}}},
    {"hubbard_u": {"Si": {"l": 4, "u": 4, "j": 0}}},
])
def test_incomplete_or_inconsistent_method_is_refused(tmp_path, parameters):
    with pytest.raises(ValueError):
        vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path / "inputs", "scf", parameters)
    assert not (tmp_path / "inputs").exists()


def _parser_stub(monkeypatch, output, task, parameters):
    meta = vasp.prepare_inputs(EXAMPLES / "Si.cif", output, task, parameters)
    structure = Poscar.from_file(output / "POSCAR", check_for_potcar=False).structure
    run = SimpleNamespace(
        vasp_version="method-test-stub", initial_structure=structure, final_structure=structure.copy(),
        final_energy=-10.0, efermi=.5, converged_electronic=True, converged_ionic=True,
        incar=dict(Incar.from_file(output / "INCAR")),
        ionic_steps=[{"forces": [[0, 0, 0]] * len(structure), "electronic_steps": [{}] * HYBRID_BAND_NELMIN}],
    )
    (output / "vasprun.xml").write_text("parser stub; not solver evidence")
    monkeypatch.setattr(vasp, "Vasprun", lambda *args, **kwargs: run)
    return meta, run


def test_analyzer_accepts_vasp5_singleton_ldautype(monkeypatch, tmp_path):
    _, run = _parser_stub(monkeypatch, tmp_path, "scf", {"hubbard_u": {"Si": {"l": 1, "u": 2, "j": 0}}})
    run.vasp_version = "5.4.4.18Apr17-6-g9f103f2a35"
    run.incar["LDAUTYPE"] = [2]
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert result["success"]
    run.incar["LDAUU"] = [0]
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and "LDAUU" in result["reason"]


@pytest.mark.parametrize("version", ["5.4.4.18Apr17-6-g9f103f2a35", "6.2.1"])
def test_hybrid_bands_need_ace_only_where_vasp_has_it(monkeypatch, tmp_path, version):
    # VASP 5 has no adaptively compressed exchange and ignores LFOCKACE; silicon's HSE06 path on 5.4.4
    # converged in 12 Davidson steps. VASP 6 records the tag and must use it.
    meta, run = _parser_stub(monkeypatch, tmp_path, "bands", {"functional": "PBE0", "soc": True})
    del run.incar["LFOCKACE"]
    if version.startswith("5."):
        validate_method_output(run.incar, meta, vasp_version=version)
    else:
        with pytest.raises(ValueError, match="LFOCKACE is missing"):
            validate_method_output(run.incar, meta, vasp_version=version)
    run.vasp_version = version
    reason = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")["reason"] or ""
    assert ("LFOCKACE" in reason) == (not version.startswith("5."))


@pytest.mark.parametrize("parameters,tag,value", [
    ({"spin": "collinear", "magmom": [2, -2]}, "ISPIN", 1),
    ({"soc": True}, "LSORBIT", False),
    ({"soc": True}, "SAXIS", [1, 0, 0]),
    ({"functional": "HSE06"}, "HFSCREEN", 0),
    ({"functional": "PBE0"}, "LHFCALC", False),
    ({"hubbard_u": {"Si": {"l": 1, "u": 2, "j": 0}}}, "LDAUU", [0]),
    ({}, "ISPIN", 2),
])
def test_analyzer_rejects_method_drift(monkeypatch, tmp_path, parameters, tag, value):
    _, run = _parser_stub(monkeypatch, tmp_path, "scf", parameters)
    run.incar[tag] = value
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and "Method mismatch" in result["reason"]


def test_hybrid_band_export_excludes_mesh_uses_own_fermi_and_preserves_all_data(monkeypatch, tmp_path):
    meta, run = _parser_stub(monkeypatch, tmp_path, "bands", {"functional": "HSE06", "mesh": [2, 2, 2], "line_density": 3})
    kpoints = Kpoints.from_file(tmp_path / "KPOINTS")
    run.actual_kpoints = kpoints.kpts
    run.actual_kpoints_weights = np.asarray(kpoints.kpts_weights) / 8
    values = np.zeros((len(kpoints.kpts), 2, 2))
    values[:, :, 0] = [1.5, 70.5]
    run.eigenvalues = {Spin.up: values}
    result = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert result["success"], result["reason"]
    assert result["energy_reference_ev"] == .5 and result["energy_reference_source"] == "hybrid_scf_mesh"
    assert result["final_energy_ev_per_atom"] == -5
    assert result["band_path_convergence_independently_validated"] is False
    assert result["hybrid_kpoint_consistency"]["passed"]
    assert result["hybrid_kpoint_consistency"]["equivalent_pair_count"] > 0
    assert result["hybrid_kpoint_consistency"]["max_energy_deviation_ev"] == 0
    assert len((tmp_path / "bands.csv").read_text().splitlines()) == 1 + 2 * meta["band_path_count"]
    assert len((tmp_path / "bands_mesh.csv").read_text().splitlines()) == 17
    assert ",70.0," in (tmp_path / "bands.csv").read_text()
    run.ionic_steps[-1]["forces"] = [[float("nan")] * 3] * 2
    spectral = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert spectral["success"], spectral["reason"]
    assert spectral["final_max_force_ev_angstrom"] is None and not spectral["forces_available"]
    assert "zero-weight hybrid band" in spectral["force_warning"]
    run.actual_kpoints_weights[-1] = 1
    bad = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert not bad["success"] and "zero path weights" in bad["reason"]
    assert not (tmp_path / "bands.png").exists()


def test_hybrid_rejects_inconsistent_repeated_gamma_despite_total_energy_convergence(monkeypatch, tmp_path):
    _, run = _parser_stub(monkeypatch, tmp_path, "bands", {"functional": "HSE06", "mesh": [2, 2, 2], "line_density": 1})
    kpoints = Kpoints.from_file(tmp_path / "KPOINTS")
    run.actual_kpoints = np.asarray(kpoints.kpts)
    run.actual_kpoints_weights = np.asarray(kpoints.kpts_weights) / 8
    values = np.zeros((len(kpoints.kpts), 8, 2))
    # The failed Si run had one spurious unoccupied state at its second path Γ.
    values[:, :, 0] = [-8.3632, 4.7185, 4.7185, 4.7185, 9.0656, 9.0656, 9.0656, 10.2886]
    values[:, :4, 1] = 1
    gamma = np.flatnonzero(np.all(np.abs(run.actual_kpoints - np.rint(run.actual_kpoints)) < 1e-6, axis=1))
    assert len(gamma) == 3
    values[gamma[-1], 3] = [5.9906, 0]
    run.eigenvalues = {Spin.up: values}
    (tmp_path / "bands.png").write_bytes(b"stale plot")
    result = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert result["converged_electronic"] and not result["success"]
    check = result["hybrid_kpoint_consistency"]
    assert not check["passed"] and check["tolerance_ev"] == .05
    assert check["max_energy_deviation_ev"] == pytest.approx(1.2721)
    assert check["worst_pair"] == {"kpoint_indices_zero_based": [0, int(gamma[-1])], "spin": 1, "band": 4}
    assert "electronic optimizer" in result["reason"]
    assert not (tmp_path / "bands.png").exists() and result["artifacts"] == []
    assert json.loads((tmp_path / "result.json").read_text())["hybrid_kpoint_consistency"] == check


def test_hybrid_kpoint_check_includes_empty_bands_both_spins_and_reciprocal_translations():
    values = np.array([[[1., 1.], [70., 0.]], [[1., 1.], [70., 0.]]])
    run = SimpleNamespace(actual_kpoints=[[0, .5, .5], [1, -.5, .5]], eigenvalues={Spin.up: values.copy(), Spin.down: values.copy()})
    run.eigenvalues[Spin.down][1, 1, 0] += .04
    check = vasp._hybrid_kpoint_consistency(run)
    assert check["passed"] and check["equivalent_pair_count"] == 1
    assert check["compared_eigenvalue_count"] == 4
    assert check["max_energy_deviation_ev"] == pytest.approx(.04)
    run.eigenvalues[Spin.down][1, 1, 0] += .02
    check = vasp._hybrid_kpoint_consistency(run)
    assert not check["passed"]
    assert check["worst_pair"]["spin"] == -1 and check["worst_pair"]["band"] == 2


@pytest.mark.parametrize("task", ["relax", "scf", "dos"])
def test_nonfinite_forces_still_refuse_other_hybrid_tasks(monkeypatch, tmp_path, task):
    _, run = _parser_stub(monkeypatch, tmp_path, task, {"functional": "HSE06"})
    run.ionic_steps[-1]["forces"] = [[float("nan")] * 3] * 2
    result = vasp.analyze_outputs(tmp_path, task, EXAMPLES / "Si.cif")
    assert not result["success"] and "forces are nonfinite" in result["reason"]


def test_hybrid_refuses_too_few_iterations(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="nelm"):
        vasp.prepare_inputs(EXAMPLES / "Si.cif", tmp_path / "invalid", "bands", {"functional": "HSE06", "nelm": 10})
    _, run = _parser_stub(monkeypatch, tmp_path, "bands", {"functional": "HSE06", "mesh": [2, 2, 2]})
    run.ionic_steps[-1]["electronic_steps"] = [{}]
    result = vasp.analyze_outputs(tmp_path, "bands", EXAMPLES / "Si.cif")
    assert not result["success"] and "minimum electronic iterations" in result["reason"]


def test_outcar_moments_are_exported_with_basis_without_ground_state_claim(monkeypatch, tmp_path):
    _, run = _parser_stub(monkeypatch, tmp_path, "scf", {"spin": "noncollinear", "magmom": [[1, 0, 0], [-1, 0, 0]], "saxis": [1, 0, 0]})
    lines = ["number of electron 8.000 magnetization 0.10 0.20 0.30\n"]
    for axis, values in zip("xyz", [[1, -1], [2, -2], [3, -3]]):
        lines += [f"magnetization ({axis})\n", "# of ion s p d tot\n", "----------\n"]
        lines += [f"{i} 0.0 0.0 {value:.4f} {value:.4f}\n" for i, value in enumerate(values, 1)]
        lines += ["tot 0.0 0.0 0.0 0.0\n"]
    (tmp_path / "OUTCAR").write_text("".join(lines))
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert result["success"], result["reason"]
    moments = result["magnetization"]
    assert moments["site_moments"] == [[1, 2, 3], [-1, -2, -3]]
    assert moments["total_moment"] == [.1, .2, .3]
    assert moments["basis"] == "SAXIS spinor basis" and moments["saxis"] == [1, 0, 0]
    assert not result["magnetic_ground_state_validated"]
    assert "magnetization.json" in result["artifacts"]
    json.dumps(result, allow_nan=False)
    with (tmp_path / "OUTCAR").open("a") as stream:
        stream.write("magnetization (x)\n1 0.0 0.0 9.0 9.0\n")
    incomplete = vasp._magnetization(tmp_path, 2, json.loads((tmp_path / "metadata.json").read_text()))
    assert incomplete["site_moments"] is None and not incomplete["available"]


def test_analyzer_requires_frozen_checks_for_method_metadata(monkeypatch, tmp_path):
    meta, _ = _parser_stub(monkeypatch, tmp_path, "scf", {"soc": True})
    meta.pop("method_incar_expected")
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    result = vasp.analyze_outputs(tmp_path, "scf", EXAMPLES / "Si.cif")
    assert not result["success"] and "method checks are missing" in result["reason"]


def test_accepted_poscar_preserves_cartesian_frame_and_is_cleared_on_failure(monkeypatch, tmp_path):
    _, run = _parser_stub(monkeypatch, tmp_path, "relax", {"soc": True})
    run.ionic_steps[-1]["e_0_energy"] = -10.0
    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    run.final_structure = Structure(run.final_structure.lattice.matrix @ rotation.T, run.final_structure.species, run.final_structure.frac_coords)
    result = vasp.analyze_outputs(tmp_path, "relax", EXAMPLES / "Si.cif")
    assert result["success"], result["reason"]
    final = tmp_path / "final_structure.vasp"
    saved = Poscar.from_file(final, check_for_potcar=False).structure
    np.testing.assert_allclose(saved.lattice.matrix, run.final_structure.lattice.matrix, atol=1e-10)
    np.testing.assert_allclose(saved.frac_coords, run.final_structure.frac_coords, atol=1e-10)
    assert result["final_structure_poscar_sha256"] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert "final_structure.cif" in result["artifacts"] and "final_structure.vasp" in result["artifacts"]
    run.converged_electronic = False
    assert not vasp.analyze_outputs(tmp_path, "relax", EXAMPLES / "Si.cif")["success"]
    assert not final.exists()
