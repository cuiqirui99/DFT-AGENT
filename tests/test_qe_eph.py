"""Numerical parser tests; successful fixtures are not cluster acceptance."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import qe, qe_eph as eph


def parameters(**updates):
    return eph.validate_parameters({"dense_mesh": [16, 16, 16], "mesh": [8, 8, 8], "qmesh": [4, 4, 4], "phonon_dos_mesh": [10, 10, 10], "mu_star": 0.1, **updates})


@pytest.fixture
def aluminum(tmp_path):
    path = tmp_path / "Al.cif"
    Structure(Lattice([[0, 2, 2], [2, 0, 2], [2, 2, 0]]), ["Al"], [[0, 0, 0]]).to(filename=path)
    return path


@pytest.fixture
def reference():
    return json.loads((Path(__file__).parent / "fixtures/qe/al_eph_reference.json").read_text())


def write_coupling_fixture(folder, records):
    """Real reference numbers, synthetic file formatting and q-star matrices."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "elph_dir").mkdir(exist_ok=True)
    (folder / "dft_agent.dyn0").write_text("4 4 4\n" + str(len(records)) + "\n" + "".join(" ".join(map(str, r["q"])) + "\n" for r in records))
    for i, record in enumerate(records, 1):
        text = " ".join(map(str, record["q"])) + f" {len(record['spectra'])} 3\n"
        text += " ".join(f"{(w / eph.RY_THZ)**2:.6e}" for w in record["frequency_thz"]) + "\n"
        for spec in record["spectra"]:
            text += f"     Gaussian Broadening: {spec['degauss_ry']:7.3f} Ry, ngauss=   0\n"
            text += f"     DOS ={spec['dos_ef_states_spin_ry']:10.6f} states/spin/Ry/Unit Cell at Ef={spec['fermi_ev']:10.6f} eV\n"
            for j, mode in enumerate(spec["modes"], 1):
                text += f"     lambda({j:5})={mode['lambda']:8.4f}   gamma={mode['gamma_ghz']:8.2f} GHz\n"
        (folder / f"elph_dir/elph.inp_lambda.{i}").write_text(text)
        for j, spec in enumerate(record["spectra"], 1):
            text = f"{spec['degauss_ry']} {spec['fermi_ev'] / qe.RY_EV:.15g} {spec['dos_ef_states_spin_ry']}\n{record['weight']}\n"
            # Matrix completeness is tested; these are explicitly not DFPT matrices.
            for point in record["q_star"]:
                text += "Dynamical  Matrix in cartesian axes\nq = ( " + " ".join(map(str, point)) + " )\n1 1\n"
                text += "0.01 0 0 0 0 0\n0 0 0.01 0 0 0\n0 0 0 0 0.01 0\n"
            (folder / f"elph_dir/a2Fq2r.{50 + j}.{i}").write_text(text)


def lambda_fixture(folder, aluminum, reference, *, sigma=0.12, mu_star=0.1):
    write_coupling_fixture(folder, reference["records"])
    p = parameters(phonon_smearing_thz=sigma, mu_star=mu_star)
    meta = eph.prepare_inputs(aluminum, folder, "eph_lambda", p, {}, evidence_dir=folder)
    evidence = eph.coupling_evidence(folder, p, require_real=True)
    x = np.linspace(0, meta["omega_max_thz"], 2000)
    spectra, data, tc = [], [], []
    for j in range(10):
        spectrum = np.zeros_like(x)
        lam = 0
        for rec, weight in zip(evidence["records"], evidence["weights"]):
            for w2, mode in zip(rec["w2_ry2"], rec["spectra"][j]["modes"]):
                freq, part = math.sqrt(w2) * eph.RY_THZ, weight / 64 * mode["lambda"]
                lam += part
                spectrum += part * freq / (2 * sigma * math.sqrt(math.pi)) * np.exp(-((x - freq) / sigma)**2)
        integral = 2 * (x[1] - x[0]) * np.sum(spectrum[1:] / x[1:])
        omega = np.exp(2 * (x[1] - x[0]) * np.sum(spectrum[1:] * np.log(x[1:]) / x[1:]) / integral) * eph.THZ_KELVIN
        spectra.append(spectrum)
        data.append([(j + 1) * 0.005, lam, integral, omega, evidence["records"][0]["spectra"][j]["dos_ef_states_spin_ry"]])
        tc.append((lam, omega, eph.tc_estimate(lam, omega, mu_star) if lam - mu_star * (1 + 0.62 * lam) > 0 else None))
    np.savetxt(folder / "alpha2F.dat", np.column_stack([x, *spectra]), fmt=["%8.4f"] + ["%10.5f"] * 10)
    np.savetxt(folder / "lambda.dat", data, fmt=["%7.3f", "%12.6f", "%12.6f", "%10.3f", "%12.6f"])
    # lambda.x writes asterisks where the McMillan expression has no solution.
    (folder / "lambda.out").write_text("lambda        omega_log          T_c\n" + "".join(
        f"{a:10.5f}     {b:9.3f}          {'*********' if c is None else f'{c:9.3f}'}\n" for a, b, c in tc))
    return meta


def test_pinned_reference_mode_sum(tmp_path, reference):
    write_coupling_fixture(tmp_path, reference["records"])
    ev = eph.coupling_evidence(tmp_path, parameters(), require_real=True)
    assert ev["weights"] == [1, 8, 4, 6, 24, 12, 3, 6]
    weighted = sum(w * sum(mode["lambda"] for mode in r["spectra"][0]["modes"]) for w, r in zip(ev["weights"], ev["records"])) / 64
    assert weighted == pytest.approx(0.29219375)
    assert ev["records"][0]["spectra"][0]["dos_ef_states_spin_ry"] == pytest.approx(1.339210)


def test_native_units_and_stages(tmp_path, aluminum):
    p = parameters(namelists={"SYSTEM": {"ecutwfc": 35}})
    for task in eph.TASKS:
        meta = eph.prepare_inputs(aluminum, tmp_path / task, task, p, {"Al": "Al.UPF"})
        text = (tmp_path / task / meta["input_file"]).read_text()
        assert meta["executable"] == eph.PROGRAMS[task]
        if task in {"eph_dense_scf", "eph_scf"}:
            assert f"la2f = {'.true.' if task == 'eph_dense_scf' else '.false.'}" in text
            assert "ecutwfc = 35" in text
        if task == "eph_ph":
            assert "electron_phonon = 'interpolated'" in text
        if task == "eph_lambda":
            assert meta["preview_only"] and meta["no_mpi"]


def test_electron_phonon_needs_norm_conserving_upfs(tmp_path, aluminum):
    # QE 7.6 ph.x crashed in elphel -> adddvscf with PAW aluminium; the PseudoDojo NC run went through.
    p = parameters()
    for entry in ({"filename": "Al.pbe-n-kjpaw_psl.1.0.0.UPF", "pseudo_type": "PAW"}, "al_pbe_v1.uspp.F.UPF"):
        with pytest.raises(ValueError, match="norm-conserving UPFs.*PseudoDojo \\(NC\\) UPFs for Al"):
            eph.prepare_inputs(aluminum, tmp_path / "paw", "eph_dense_scf", p, {"Al": entry})
    eph.prepare_inputs(aluminum, tmp_path / "nc", "eph_dense_scf", p, {"Al": {"filename": "Al.pbe-nc-sr-04-pd.upf", "pseudo_type": "NC"}})


@pytest.mark.parametrize("change,reason", [({"mu_star": None}, "number"), ({"mu_star": 1}, "mu_star"), ({"dense_mesh": [15, 16, 16]}, "contain"), ({"mesh": [6, 8, 8]}, "contain"), ({"soc": True}, "nonmagnetic"), ({"functional": "HSE06"}, "hybrid"), ({"electronic_type": "insulator"}, "metallic"), ({"el_ph_sigma": 0.0001}, "three decimals"), ({"el_ph_nsigma": 11}, "ten"), ({"qpoint": [0, 0, 0]}, "Unsupported"), ({"namelists": {"inputph": {"only_init": True}}}, "limited"), ({"namelists": {"system": {"nosym": True}}}, "symmetry")])
def test_invalid_requests_rejected(change, reason):
    with pytest.raises((ValueError, TypeError), match=reason):
        parameters(**change)


@pytest.mark.parametrize("mutation,reason", [("coordinate", "q points differ"), ("truncated", "truncated"), ("weight", "q-star"), ("sigma", "broadening"), ("mode", "indices"), ("imaginary", "Imaginary"), ("matrix", "matrix")])
def test_bad_coupling_evidence(tmp_path, reference, mutation, reason):
    write_coupling_fixture(tmp_path, reference["records"])
    path = tmp_path / "elph_dir/elph.inp_lambda.1"
    text = path.read_text()
    if mutation == "coordinate":
        text = text.replace("0.0 0.0 0.0", "0.2 0.0 0.0", 1)
    elif mutation == "truncated":
        text = "\n".join(text.splitlines()[:5])
    elif mutation == "sigma":
        text = text.replace("0.005 Ry", "0.006 Ry", 1)
    elif mutation == "mode":
        text = text.replace("lambda(    1)", "lambda(    4)", 1)
    elif mutation == "imaginary":
        text = text.replace(text.splitlines()[1], "-1e-7 1e-7 1e-7", 1)
    else:
        star = tmp_path / "elph_dir/a2Fq2r.51.1"
        raw = star.read_text()
        star.write_text(raw.replace("\n1\n", "\n2\n", 1) if mutation == "weight" else raw.replace("0.01 0 0 0 0 0", "0.01 0", 1))
    path.write_text(text)
    with pytest.raises((ValueError, IndexError), match=reason):
        eph.coupling_evidence(tmp_path, parameters(), require_real=True)


def test_lambda_integrals_units_and_evidence(tmp_path, aluminum, reference):
    meta = lambda_fixture(tmp_path, aluminum, reference)
    result = eph.analyze_outputs(tmp_path, "eph_lambda")
    assert result["success"], result
    assert result["electron_phonon"][0]["lambda"] == pytest.approx(0.292194)
    assert 100 < result["electron_phonon"][0]["omega_log_kelvin"] < 500
    assert result["mu_star"] == 0.1
    assert result["scientific_accuracy_validated"] is False
    assert result["superconductivity_established"] is False
    assert (tmp_path / "electron_phonon.csv").is_file()
    path = tmp_path / "elph_dir/elph.inp_lambda.1"
    path.write_text(path.read_text() + "\n")
    failed = eph.analyze_outputs(tmp_path, "eph_lambda")
    assert not failed["success"] and "changed" in failed["reason"]
    assert not (tmp_path / "electron_phonon.csv").exists()


@pytest.mark.parametrize("file,alter,reason", [("lambda.dat", "lambda", "weighted"), ("alpha2F.dat", "spectra", "integrals"), ("lambda.out", "tc", "Tc"), ("alpha2F.dat", "truncate", "dimensions")])
def test_corrupt_lambda_outputs(tmp_path, aluminum, reference, file, alter, reason):
    lambda_fixture(tmp_path, aluminum, reference)
    path = tmp_path / file
    if alter in {"lambda", "spectra"}:
        data = np.loadtxt(path)
        data[:, 1] *= 1.5
        np.savetxt(path, data)
    elif alter == "tc":
        lines = path.read_text().splitlines()
        lines[-1] = "0.38375 250 99"
        path.write_text("\n".join(lines))
    else:
        path.write_text("\n".join(path.read_text().splitlines()[:500]))
    result = eph.analyze_outputs(tmp_path, "eph_lambda")
    assert not result["success"] and reason in result["reason"], result


def test_invalid_tc_domain():
    with pytest.raises(ValueError, match="domain"):
        eph.tc_estimate(0.1, 200, 0.2)
    assert eph.tc_estimate(0.5, 300, 0.1) == pytest.approx(3.6466853854695427)


def test_complete_dense_save(tmp_path):
    path = tmp_path / "dft_agent.a2Fsave"
    # 2 bands, 1 irreducible point, unit symmetry, one atom.
    path.write_text("2 1\n-0.1 0.1\n0 0 0\n2\n16 16 16\n1\n1 0 0 0 1 0 0 0 1\n1\n")
    assert eph.parse_dense_save(path, parameters(), 1, 2)["bands"] == 2
    with pytest.raises(ValueError, match="band count"):
        eph.parse_dense_save(path, parameters(), 1, 3)
    path.write_text(path.read_text().rsplit("\n1\n", 1)[0])
    with pytest.raises(ValueError, match="symmetry"):
        eph.parse_dense_save(path, parameters(), 1, 2)


def test_matdyn_spectrum_ry_integral(tmp_path):
    p = parameters(el_ph_nsigma=1, phonon_dos_points=100)
    energy = (np.arange(100) + 0.5) * 0.0001
    modes = np.column_stack([energy, 2 * energy, 3 * energy])
    spectrum = modes.sum(axis=1)
    lam = 2 * 0.0001 * np.sum(spectrum / energy)
    (tmp_path / "lambda").write_text(f"Broadening 0.0050 lambda {lam:.4f} dos(Ef) 1.3392 omega_ln [K] 200.0\n")
    path = tmp_path / "a2F.dos1"
    np.savetxt(path, np.column_stack([energy, spectrum, modes]))
    with path.open("a") as stream:
        stream.write(f" lambda = {lam} Delta = 0.0001\n")
    rows, artifacts = eph.parse_interpolated_spectra(tmp_path, p, 1)
    assert rows[0]["lambda_spectral_integral"] == pytest.approx(0.12)
    assert "alpha2F_interpolated_1.csv" in artifacts
    path.write_text(path.read_text().replace("Delta = 0.0001", "Delta = 0.001"))
    with pytest.raises(ValueError, match="grid"):
        eph.parse_interpolated_spectra(tmp_path, p, 1)


@pytest.mark.parametrize("alter,reason", [(None, None), ("negative", None), ("negative-wide", "Negative interpolated"),
                                         ("grid", "regular grid")])
def test_native_matdyn_print_precision_preserves_physical_checks(tmp_path, alter, reason):
    p = parameters(el_ph_nsigma=1, phonon_dos_points=2001)
    delta = 1.475233599040527e-6
    energy = (np.arange(2001) + .5) * delta
    modes = np.column_stack([energy, 2 * energy, 3 * energy])
    data = np.column_stack([energy, modes.sum(axis=1), modes])
    if alter == "negative":
        # One low-frequency point, 0.1 % of lambda: kept and reported, as matdyn's interpolation produces.
        data[100, 1:] *= -1
    elif alter == "negative-wide":
        data[:100, 1:] *= -1
    elif alter == "grid":
        data[100, 0] += delta / 10
    # matdyn's own integral includes whatever weights it wrote.
    lam = 2 * delta * np.sum(data[:, 1] / energy)
    (tmp_path / "lambda").write_text(f"Broadening 0.0050 lambda {lam:.4f} dos(Ef) 1.3392 omega_ln [K] 200.0\n")
    path = tmp_path / "a2F.dos1"
    # QE's 0.ddddddE+nn has six significant digits.
    np.savetxt(path, data, fmt="%.5E")
    with path.open("a") as stream:
        stream.write(f" lambda = {lam} Delta = {delta}\n")
    if reason:
        with pytest.raises(ValueError, match=reason):
            eph.parse_interpolated_spectra(tmp_path, p, 1)
    else:
        rows, _ = eph.parse_interpolated_spectra(tmp_path, p, 1)
        if alter == "negative":
            assert rows[0]["negative_lambda_fraction"] == pytest.approx(1 / 1999, rel=0.01) and not rows[0].get("excluded")
            assert rows[0]["negative_below_mev"] == pytest.approx(100.5 * delta * eph.RY_MEV, rel=1e-4)
        else:
            assert rows[0]["lambda_spectral_integral"] == pytest.approx(lam, rel=2e-5) and "negative_lambda_fraction" not in rows[0]


def test_weightless_gamma_acoustic_modes_reach_lambda_x_at_zero_frequency(tmp_path, aluminum, reference):
    # ph.x left aluminium's three acoustic modes at Γ at -2.5 cm⁻¹ (ω² ≈ -5e-10 Ry²) with λ = 0; lambda.x
    # takes sqrt(ω²) and would write NaN spectra. A copy with zero frequencies goes to lambda.x instead.
    write_coupling_fixture(tmp_path, reference["records"])
    gamma = tmp_path / "elph_dir/elph.inp_lambda.1"
    original = gamma.read_text()
    assert all(mode["lambda"] == 0 for spec in eph.parse_elph_file(gamma)["spectra"] for mode in spec["modes"])
    gamma.write_text(original.replace(original.splitlines()[1], "-0.548462E-09 -0.543706E-09 -0.532475E-09", 1))
    p = parameters()
    evidence = eph.coupling_evidence(tmp_path, p, require_real=True)
    assert evidence["zeroed_modes"] == {1: [0, 1, 2]}
    meta = eph.prepare_inputs(aluminum, tmp_path / "lambda", "eph_lambda", p, {}, evidence_dir=tmp_path)
    text = (tmp_path / "lambda/lambda.in").read_text()
    assert "elph_zeroed.1\n" in text and "elph_dir/elph.inp_lambda.1\n" not in text and "elph_dir/elph.inp_lambda.2\n" in text
    assert eph.parse_elph_file(tmp_path / "lambda/elph_zeroed.1")["w2_ry2"] == [0.0, 0.0, 0.0]
    assert "elph_zeroed.1" in meta["input_files"] and meta["zeroed_gamma_modes"]["modes"] == [1, 2, 3]
    from vasp_slurm_agent.qe_workflow import _QE_UPLOAD_NAMES
    assert set(meta["input_files"]) <= _QE_UPLOAD_NAMES
    assert meta["zeroed_gamma_modes"]["frequencies_cm1"][0] == pytest.approx(-2.57, abs=0.01)
    # The evidence keeps the original file; anything beyond weightless noise at Γ is still refused.
    assert gamma.read_text().splitlines()[1].startswith("-0.548462E-09")
    for line, field in (("-1e-7 -1e-7 -1e-7", 1), ("-5e-10 -5e-10 -5e-10", 2)):
        write_coupling_fixture(tmp_path, reference["records"])
        path = tmp_path / f"elph_dir/elph.inp_lambda.{field}"
        raw = path.read_text()
        path.write_text(raw.replace(raw.splitlines()[1], line, 1))
        with pytest.raises(ValueError, match="Imaginary"):
            eph.coupling_evidence(tmp_path, p, require_real=True)
    write_coupling_fixture(tmp_path, reference["records"])
    raw = gamma.read_text()
    gamma.write_text(raw.replace(raw.splitlines()[1], "-5e-10 -5e-10 -5e-10", 1).replace("lambda(    1)=  0.0000", "lambda(    1)=  0.0100", 1))
    with pytest.raises(ValueError, match="Imaginary"):
        eph.coupling_evidence(tmp_path, p, require_real=True)


def test_weak_coupling_has_no_mcmillan_tc_but_keeps_its_spectrum(tmp_path, aluminum, reference):
    # A coarse aluminium check gave lambda = 0.049 in its narrowest broadening; lambda.x printed asterisks there.
    lambda_fixture(tmp_path, aluminum, reference, mu_star=0.5)
    result = eph.analyze_outputs(tmp_path, "eph_lambda")
    assert result["success"], result["reason"]
    assert all(row["tc_estimate_kelvin"] is None for row in result["electron_phonon"])
    assert sum("gives no Tc there" in warning for warning in result["warnings"]) == 10
    table = (tmp_path / "lambda.out").read_text().replace("*********", "    1.000", 1)
    (tmp_path / "lambda.out").write_text(table)
    assert eph.analyze_outputs(tmp_path, "eph_lambda")["success"]
