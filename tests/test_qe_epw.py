"""EPW native-contract tests. Synthetic records are not cluster validation."""

import json
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import qe, qe_epw as epw

FINISH = "Total program execution\nEPW : 1.00s CPU 1.00s WALL\nfunctionality-dependent EPW.bib file.\n"


def parameters(**changes):
    return {"temperatures": [5.0, 10.0, 15.0], "mu_star": 0.1, "frequency_cutoff_ev": 0.05, "broadening_index": 1, **changes}


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    w = np.linspace(0, 20, 201)
    spectrum = np.exp(-((w - 8) / 2)**2) * (w / 8)**2
    np.savetxt(folder / "spectrum.dat", np.column_stack([w, spectrum]), fmt="%.16e")
    schema = """<espresso><input><atomic_species ntyp="1"><species name="Al"><mass>26.9815386</mass></species></atomic_species></input>
    <output><convergence_info><scf_conv><convergence_achieved>true</convergence_achieved></scf_conv></convergence_info>
    <atomic_structure nat="1" alat="8"><atomic_positions><atom name="Al">0 0 0</atom></atomic_positions><cell><a1>8 0 0</a1><a2>0 8 0</a2><a3>0 0 8</a3></cell></atomic_structure>
    <band_structure><lsda>false</lsda><noncolin>false</noncolin><nelec>3</nelec></band_structure></output></espresso>"""
    (folder / "schema.xml").write_text(schema)
    structure = Structure(Lattice.cubic(8 * qe.BOHR_ANGSTROM), ["Al"], [[0, 0, 0]])
    structure.to(filename=folder / "Al.cif")
    return folder


def prepared(tmp_path, source, task="epw_iso", **changes):
    folder = tmp_path / task
    meta = epw._prepare_native(folder, task, parameters(**changes), spectrum_path=source / "spectrum.dat", qe_schema=source / "schema.xml")
    return folder, meta


def header(meta):
    lam, omega = meta["spectrum"]["lambda"], meta["spectrum"]["omega_log_mev"]
    mu = meta["parameters"]["mu_star"]
    tc = omega / (1000 * epw.KB_EV * 1.2) * math.exp(-1.04 * (1 + lam) / (lam - mu * (1 + 0.62 * lam)))
    return f"""Program EPW v.5.9 starts on synthetic parser fixture
Electron-phonon coupling strength = {lam:.7f}
Estimated w_log in Allen-Dynes Tc = {omega:.6f} meV
Estimated Allen-Dynes Tc = {tc:.6f} K for muc = {mu:.5f}
"""


def write_gap(folder, meta):
    text = header(meta)
    for i, t in enumerate(meta["parameters"]["temperatures"], 1):
        count = epw._nsiw(t, meta["parameters"]["frequency_cutoff_ev"])
        w = (2 * np.arange(count) + 1) * math.pi * epw.KB_EV * t
        z = 1.3 + 0 * w
        gap = 0.0004 / (1 + (w / 0.01)**2)
        np.savetxt(folder / f"dft_agent.imag_iso_{t:06.2f}", np.column_stack([w, z, gap]), header="w [eV] znorm(w) delta(w) [eV]", comments="", fmt="%.10e")
        text += f"""temp( {i}) = {t:.5f} K
Solve isotropic Eliashberg equations on imaginary-axis
Total number of frequency points nsiw( {i}) = {count}
iter      ethr          znormi      deltai [meV]
1 1.0e-2 1.2 0.5
8 1.0e-8 {z[0]:.6e} {gap[0] * 1000:.6e}
Convergence was reached in nsiter = 8
Temp (itemp = {i}) = {t} K  Free energy = -0.01 meV
"""
    (folder / "epw.out").write_text(text + FINISH)


def write_linear(folder, meta, eigenvalues=(1.3, 1.1, 0.9)):
    text = header(meta) + "Start: Solving (isotropic) linearized Eliashberg equation with solver = lapack\n"
    text += "Temp. Max. nsiw wscut Nr. of iters\n(K) eigenvalue (itemp) (eV) to Converge\n"
    for t, eigen in zip(meta["parameters"]["temperatures"], eigenvalues):
        count = epw._nsiw(t, meta["parameters"]["frequency_cutoff_ev"])
        cutoff = (2 * count + 1) * math.pi * epw.KB_EV * t
        text += f"{t:12.2f}{eigen:15.7f}{count:9d}{cutoff:12.4f}{1:9d}\n"
    text += "Finish: Solving (isotropic) linearized Eliashberg equation\n" + FINISH
    (folder / "epw.out").write_text(text)


@pytest.mark.parametrize("changes,match", [({"temperatures": []}, "temperatures"), ({"temperatures": [0]}, "positive"), ({"temperatures": [10, 5]}, "increasing"), ({"temperatures": [5.001]}, "two decimals"), ({"mu_star": 0}, "positive"), ({"mu_star": 1}, "between"), ({"frequency_cutoff_ev": 1e-5}, "two frequencies"), ({"convergence": .1}, "threshold"), ({"mixing": 2}, "mixing"), ({"broadening_index": True}, "integer"), ({"laniso": True}, "Unsupported")])
def test_invalid_parameters(changes, match):
    with pytest.raises(ValueError, match=match):
        epw.validate_parameters(parameters(**changes))


def test_prepare_native_from_actual_schema(tmp_path, source):
    folder, meta = prepared(tmp_path, source)
    text = (folder / "epw.in").read_text()
    assert "ep_coupling = .false." in text
    assert "tc_linear = .false." in text
    assert "temps(3) = 15.0" in text
    assert "nqstep = 200" in text
    for kind in ("nkf", "nqf"):
        for axis in (1, 2, 3):
            assert f"{kind}{axis} = 1" in text
    assert meta["number_of_atoms"] == 1 and meta["formula"] == "Al"
    assert not meta["requires_restart"]
    assert meta["output_files"] == ["epw.out", "dft_agent.imag_iso_005.00", "dft_agent.imag_iso_010.00", "dft_agent.imag_iso_015.00"]
    crystal = (folder / "crystal.fmt").read_text().splitlines()
    assert crystal[:3] == ["1", "3", "3 0"]
    assert len(crystal[8].split()) == 10
    assert float(crystal[5]) == pytest.approx(512)
    for name in meta["input_files"]:
        assert qe._hash(folder / name) == meta["input_sha256"][name]
    with pytest.raises(ValueError, match="new folder"):
        epw._prepare_native(folder, "epw_iso", parameters(), spectrum_path=source / "spectrum.dat", qe_schema=source / "schema.xml")


def test_preview_never_runs_without_verified_spectrum(tmp_path, source):
    folder = tmp_path / "preview"
    meta = epw.prepare_inputs(source / "Al.cif", folder, "epw_iso", parameters())
    assert meta["preview_only"]
    assert "structure.cif" in meta["input_files"]
    result = epw.analyze_outputs(folder, "epw_iso")
    assert not result["success"] and "accepted spectrum" in result["reason"]


@pytest.mark.parametrize("mutation,reason", [("negative", "nonnegative"), ("offset", "uniform"), ("zero", "zero-frequency"), ("nan", "Invalid"), ("duplicate", "increasing")])
def test_invalid_spectrum_not_silently_fixed(source, mutation, reason):
    file = source / "spectrum.dat"
    data = np.loadtxt(file)
    if mutation == "negative":
        data[5, 1] = -1e-12
    elif mutation == "offset":
        data[5, 0] += 0.001
    elif mutation == "zero":
        data[0, 1] = 0.1
    elif mutation == "nan":
        data[5, 1] = np.nan
    else:
        data[5, 0] = data[4, 0]
    np.savetxt(file, data)
    with pytest.raises(ValueError, match=reason):
        epw.read_spectrum(file)


def test_native_thz_decimal_quantization(source):
    file = source / "rounded.dat"
    data = np.loadtxt(source / "spectrum.dat")
    data[:, 0] /= epw.THZ_MEV
    np.savetxt(file, data, fmt=["%.4f", "%.16e"])
    with pytest.raises(ValueError, match="uniform"):
        epw.read_spectrum(file, unit="THz")
    w, spectrum, meta = epw.read_spectrum(file, unit="THz", frequency_precision=.0001)
    assert meta["frequency_rounding_correction"] < .0001
    assert w[-1] == pytest.approx(20, abs=.0003)
    assert np.array_equal(spectrum, data[1:, 1])


@pytest.mark.parametrize("change", ["convergence", "magnetism"])
def test_rejects_bad_scf_source(tmp_path, source, change):
    file = source / "schema.xml"
    raw = file.read_text()
    file.write_text(raw.replace("<convergence_achieved>true", "<convergence_achieved>false") if change == "convergence" else raw.replace("<lsda>false", "<lsda>true"))
    with pytest.raises(ValueError, match="converged|nonmagnetic"):
        prepared(tmp_path, source)


def test_nonlinear_gap_validation_and_units(tmp_path, source):
    folder, meta = prepared(tmp_path, source)
    write_gap(folder, meta)
    result = epw.analyze_outputs(folder, "epw_iso")
    assert result["success"], result
    assert result["temperature_results"][0]["temperature_kelvin"] == 5
    assert 0 < result["temperature_results"][0]["gap_iw0_mev"] < .4
    assert not result["scientific_accuracy_validated"]
    assert not result["superconductivity_established"]
    assert (folder / "eliashberg_gap.csv").is_file()


@pytest.mark.parametrize("mutation,reason", [("unfinished", "finish normally"), ("unconverged", "unconverged"), ("gap", "disagree"), ("frequency", "frequencies"), ("temperature", "temperatures"), ("spectrum", "changed"), ("scalar", "spectral integral"), ("old_version", "5.9"), ("missing_temperature", "incomplete")])
def test_rejects_unaccepted_native_gap(tmp_path, source, mutation, reason):
    folder, meta = prepared(tmp_path, source)
    write_gap(folder, meta)
    output = folder / "epw.out"
    text = output.read_text()
    if mutation == "unfinished":
        output.write_text(text.replace("functionality-dependent EPW.bib file.", ""))
    elif mutation == "unconverged":
        output.write_text(text.replace("Convergence was reached", "Convergence was not reached", 1))
    elif mutation in {"gap", "frequency"}:
        file = folder / "dft_agent.imag_iso_005.00"
        data = np.loadtxt(file, skiprows=1)
        data[0, 2 if mutation == "gap" else 0] += .001
        np.savetxt(file, data, header="w Z gap")
    elif mutation == "temperature":
        output.write_text(text.replace("= 5.00000 K", "= 6.00000 K"))
    elif mutation == "spectrum":
        with (folder / "alpha2f.dat").open("a") as stream:
            stream.write("\n")
    elif mutation == "scalar":
        output.write_text(text.replace(f"strength = {meta['spectrum']['lambda']:.7f}", "strength = 9.0000000"))
    elif mutation == "old_version":
        output.write_text(text.replace("v.5.9", "v.5.8"))
    else:
        output.write_text(text[:text.index("temp( 3)")] + FINISH)
    result = epw.analyze_outputs(folder)
    assert not result["success"] and reason in result["reason"], result
    assert not (folder / "eliashberg_gap.csv").exists()


def test_tc_requires_crossing_not_an_allen_dynes_estimate(tmp_path, source):
    folder, meta = prepared(tmp_path, source, "epw_tc")
    write_linear(folder, meta)
    result = epw.analyze_outputs(folder)
    assert result["success"], result
    assert result["tc_bracket_kelvin"] == [10, 15]
    assert "tc_kelvin" not in result
    write_linear(folder, meta, [.9, .8, .7])
    result = epw.analyze_outputs(folder)
    assert result["success"] and not result["tc_bracketed"]
    assert result["tc_bracket_kelvin"] is None


@pytest.mark.parametrize("mutation,reason", [("missing", "incomplete"), ("wrong_grid", "cutoff"), ("wrong_solver", "incomplete")])
def test_native_linear_table_checks(tmp_path, source, mutation, reason):
    folder, meta = prepared(tmp_path, source, "epw_tc")
    write_linear(folder, meta)
    output = folder / "epw.out"
    text = output.read_text()
    if mutation == "missing":
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("15.00"))
    elif mutation == "wrong_solver":
        text = text.replace("solver = lapack", "solver = power")
    else:
        count = epw._nsiw(5, .05)
        text = text.replace(f"{count:9d}", f"{count + 1:9d}", 1)
    output.write_text(text)
    result = epw.analyze_outputs(folder)
    assert not result["success"] and reason in result["reason"], result


def test_independent_linear_check_detects_wrong_native_eigenvalue(tmp_path, source):
    folder, meta = prepared(tmp_path, source, "epw_tc", temperatures=[10, 15])
    write_linear(folder, meta, [1.3, 0.9])
    first = epw.check_linear_eigenvalues(folder)
    assert not first["passed"]
    assert all(row["eigenpair_residual"] < 1e-10 for row in first["checks"])
    write_linear(folder, meta, [row["independent_eigenvalue"] for row in first["checks"]])
    checked = epw.check_linear_eigenvalues(folder)
    assert checked["passed"], checked


def test_frozen_native_structure_rejects_changed_matrix(tmp_path, source):
    folder, meta = prepared(tmp_path, source)
    write_gap(folder, meta)
    meta['native_structure']['lattice_angstrom'][0][0] *= 1.01
    (folder / 'qe-input.json').write_text(json.dumps(meta))
    result = epw.analyze_outputs(folder)
    assert not result['success'] and 'frozen native structure' in result['reason']


def test_cif_serialization_preserves_native_orientation_check(tmp_path, source):
    schema = source / 'schema.xml'
    schema.write_text(schema.read_text().replace('<a1>8 0 0</a1><a2>0 8 0</a2>', '<a1>0 8 0</a1><a2>-8 0 0</a2>'))
    folder, meta = prepared(tmp_path, source)
    write_gap(folder, meta)
    result = epw.analyze_outputs(folder, expected_structure=source / 'Al.cif')
    assert result['success'], result.get('reason')
    assert meta['native_structure']['lattice_angstrom'][0][0] == 0
    different = Structure(Lattice.cubic(8.01 * qe.BOHR_ANGSTROM), ['Al'], [[0, 0, 0]])
    different.to(filename=source / 'different.cif')
    result = epw.analyze_outputs(folder, expected_structure=source / 'different.cif')
    assert not result['success'] and 'expected structure' in result['reason']


def test_nonlinear_equations_reject_plausible_but_wrong_gap_table(tmp_path, source):
    folder, meta = prepared(tmp_path, source)
    write_gap(folder, meta)
    assert epw.analyze_outputs(folder)['success']
    check = epw.check_nonlinear_equations(folder)
    assert not check['passed']
    assert any(row['relative_gap_residual'] > check['tolerance'] for row in check['checks'])


def checkpoint_fixture(folder):
    """One-state synthetic stream; dimensions follow EPW's real file contract."""
    directory = folder / 'scratch/dft_agent.ephmat'
    directory.mkdir(parents=True)
    (folder / 'crystal.fmt').write_text('1\n3\n2 0\n')
    (folder / 'selecq.fmt').write_text('1\n1\n1\n')
    with (directory / 'freq').open('wb') as f:
        np.array([1,1,1,1,3],dtype='<i4').tofile(f)
        np.array([0,0,0,.001,.002,.003],dtype='<f8').tofile(f)
    with (directory / 'egnv').open('wb') as f:
        np.array([1,1,1,1,1,1],dtype='<i4').tofile(f)
        np.array([0,0,1/(math.sqrt(math.pi)*.2),.2,.5],dtype='<f8').tofile(f)
        np.array([2,0,0,0],dtype='<f8').tofile(f)
        np.array([1,1],dtype='<i4').tofile(f)
        np.array([0],dtype='<f8').tofile(f)
    np.array([1,1],dtype='<i4').tofile(directory/'ikmap')
    np.array([1],dtype='<i4').tofile(directory/'npool')
    with (directory/'ephmat1').open('wb') as f:
        np.array([1,1],dtype='<i4').tofile(f)
        np.array([.01,.02,.03],dtype='<f8').tofile(f)
    return directory


def test_binary_checkpoint_dimensions_and_states(tmp_path):
    checkpoint_fixture(tmp_path)
    result=epw.read_checkpoint(tmp_path)
    assert result['matrix_values']==3 and len(result['states'])==1
    assert result['states'][0]['weight']==pytest.approx(1)
    assert result['fine_kmesh']==[1,1,1]


@pytest.mark.parametrize('mutation', ['truncated','negative','imaginary','map','pool','selection'])
def test_incomplete_or_inconsistent_binary_checkpoint_rejected(tmp_path, mutation):
    directory=checkpoint_fixture(tmp_path)
    if mutation=='truncated':
        file=directory/'ephmat1';file.write_bytes(file.read_bytes()[:-8])
    elif mutation=='negative':
        with (directory/'ephmat1').open('r+b') as f:
            f.seek(8);np.array([-.1],dtype='<f8').tofile(f)
    elif mutation=='imaginary':
        with (directory/'freq').open('r+b') as f:
            f.seek(44);np.array([-.001],dtype='<f8').tofile(f)
    elif mutation=='map': np.array([0,1],dtype='<i4').tofile(directory/'ikmap')
    elif mutation=='pool': np.array([2],dtype='<i4').tofile(directory/'npool')
    elif mutation=='selection': (tmp_path/'selecq.fmt').write_text('1\n1\n2\n')
    with pytest.raises(ValueError): epw.read_checkpoint(tmp_path)


def aniso_fixture(tmp_path, source):
    from shutil import copyfile
    folder=tmp_path/'aniso';folder.mkdir()
    checkpoint_fixture(folder)
    copyfile(source/'schema.xml',folder/'source-schema.xml')
    (folder/'epw.in').write_text('synthetic parser fixture\n')
    p=parameters(temperatures=[15],frequency_cutoff_ev=.05);p.pop('broadening_index')
    structure=Structure.from_file(source/'Al.cif')
    native={'lattice_angstrom':structure.lattice.matrix.tolist(),'elements':['Al'],'fractional_positions':[[0,0,0]]}
    checkpoint=epw.read_checkpoint(folder)
    meta={'canonical_task':'epw_aniso','parameters':p,'input_sha256':{n:qe._hash(folder/n) for n in ['epw.in','source-schema.xml']},'checkpoint_sha256':checkpoint['sha256'],'checkpoint_settings':{'acoustic_cutoff_cm1':5},'native_structure':native,'source_provenance':{'kind':'synthetic_parser_fixture'}}
    (folder/'qe-input.json').write_text(json.dumps(meta))
    count=epw._nsiw(15,.05);omega=(2*np.arange(count)+1)*math.pi*epw.KB_EV*15
    np.savetxt(folder/'dft_agent.imag_aniso_015.00',np.column_stack([omega,np.zeros(count),np.ones(count)*1.5,np.ones(count)*.002]),header='omega E-Ef Z Delta',fmt='%.10e')
    text='Program EPW v.5.9\nSolve anisotropic Eliashberg equations\ntemp( 1) = 15.00000 K\n8 1e-8 1.5 2.0\nConvergence was reached in nsiter = 8\n'+FINISH
    (folder/'epw.out').write_text(text)
    return folder


def test_anisotropic_state_resolved_results(tmp_path,source):
    folder=aniso_fixture(tmp_path,source)
    result=epw.analyze_outputs(folder,'epw_aniso')
    assert result['success'],result
    assert result['temperature_results'][0]['gap_iw0_mev']==pytest.approx(2)
    assert (folder/'eliashberg_anisotropic_states.csv').is_file()
    assert not result['superconductivity_established']


@pytest.mark.parametrize('mutation',['energies','gap','residual','checkpoint'])
def test_anisotropic_false_success_rejected(tmp_path,source,mutation):
    folder=aniso_fixture(tmp_path,source)
    if mutation in {'energies','gap'}:
        file=folder/'dft_agent.imag_aniso_015.00';data=np.loadtxt(file);data[0,1 if mutation=='energies' else 3]+=.1
        np.savetxt(file,data,header='omega E-Ef Z Delta',fmt='%.10e')
    elif mutation=='residual':
        file=folder/'epw.out';file.write_text(file.read_text().replace('8 1e-8','8 1e-2'))
    else:
        file=folder/'scratch/dft_agent.ephmat/ephmat1';file.write_bytes(file.read_bytes()+b'changed')
    result=epw.analyze_outputs(folder,'epw_aniso')
    assert not result['success'] and not (folder/'eliashberg_gap.csv').exists()


def test_native_multismearing_spectrum_integrals(tmp_path):
    path=tmp_path/'native.a2f';w=np.arange(1,11,dtype=float);a=np.full(10,.1);integrated=np.cumsum(2*a/w)
    path.write_text(' w[meV] a2f and integrated 2*a2f/w for 1 smearing values\n'+''.join(f'{x:.7f} {y:.7f} {z:.7f}\n' for x,y,z in zip(w,a,integrated))+'# smearing .5\nElectron smearing (eV) .1\nFermi window (eV) 10\nSummed el-ph coupling .5\n')
    assert epw.read_spectrum(path)[2]['points']==10
    text=path.read_text().replace(f'{integrated[-1]:.7f}','9.0000000')
    path.write_text(text)
    with pytest.raises(ValueError,match='integrals'):epw.read_spectrum(path)


def wannier_spectrum_fixture(tmp_path, source):
    """Flat synthetic spectrum evidence, deliberately without a g-matrix checkpoint."""
    from vasp_slurm_agent import qe_wannier
    from test_qe_wannier import settings
    folder = tmp_path / 'wannier-spectrum'
    qe_wannier.prepare_inputs(source / 'Al.cif', folder, 'epw_interpolate', settings(spectral_points=10))
    schema = (source / 'schema.xml').read_text().replace('<input>', '<input><control_variables><calculation>nscf</calculation></control_variables>')
    schema = schema.replace('<convergence_achieved>true', '<convergence_achieved>false')
    (folder / 'data-file-schema.xml').write_text(schema)
    w = np.arange(1, 11, dtype=float)
    a = np.full((10, 2), [.1, .2])
    integrals = np.cumsum(2 * a / w[:, None], axis=0)
    rows = np.column_stack([w, a, integrals])
    spectrum = ' w[meV] a2f and integrated 2*a2f/w for 2 smearing values\n'
    spectrum += ''.join(' '.join(f'{x:.7f}' for x in row) + '\n' for row in rows)
    spectrum += 'Phonon smearing (meV)\n # .5 .55\nElectron smearing (eV) .1\nFermi window (eV) 10\nSummed el-ph coupling .5\n'
    (folder / 'dft_agent.a2f').write_text(spectrum)
    (folder / 'epw.out').write_text(f'Program EPW v.5.9\nRunning Wannier90\nFinish writing .ephmat files\nElectron-phonon coupling strength = {integrals[-1, 0]:.7f}\n' + FINISH)
    wout = ' +------------------------------------+<-- CONV\n | Iter Delta Spread RMS Gradient Spread Time |<-- CONV\n'
    wout += ''.join(f'{i} 1e-9 1e-8 4 1 <-- CONV\n' for i in range(1, 6)) + 'Final State\n'
    wout += ''.join(f'WF centre and spread {i} ( 0.0, 0.0, 0.0 ) 1.0\n' for i in range(1, 5))
    (folder / 'dft_agent.wout').write_text(wout + 'All done: wannier90 exiting\n')
    return folder


@pytest.mark.parametrize('task', ['epw_iso', 'epw_tc'])
def test_scalar_wannier_handoff_preserves_all_columns_and_labels(tmp_path, source, task):
    from vasp_slurm_agent import qe_wannier
    evidence = wannier_spectrum_fixture(tmp_path, source)
    assert not qe_wannier.analyze_outputs(evidence, 'epw_interpolate')['success']
    result_before = (evidence / 'qe-result.json').read_bytes()
    spectral = qe_wannier.analyze_spectrum(evidence)
    assert spectral['success'] and spectral['scope'] == 'native_spectrum', spectral
    assert 'checkpoint_sha256' not in spectral
    assert (evidence / 'qe-result.json').read_bytes() == result_before
    folder = tmp_path / task
    meta = epw.prepare_inputs(source / 'Al.cif', folder, task, parameters(broadening_index=2), evidence_dir=evidence)
    assert meta['source_provenance']['phonon_broadening_mev'] == .55
    assert meta['source_provenance']['electronic_broadening_ev'] == .1
    assert meta['source_provenance']['available_phonon_broadenings_mev'] == [.5, .55]
    assert not meta['requires_restart']
    assert (folder / 'source.a2f').read_bytes() == (evidence / 'dft_agent.a2f').read_bytes()
    assert np.loadtxt(folder / 'alpha2f.dat')[:, 1] == pytest.approx(.2)
    assert {'source-epw.in', 'source-epw.out', 'source-wannier.wout'} <= set(meta['input_files'])
    assert all(qe._hash(folder / name) == digest for name, digest in meta['input_sha256'].items())


@pytest.mark.parametrize('mutation,match', [('column', 'exceeds'), ('labels', 'labels'), ('smearing', 'broadenings differ'), ('unfinished', 'failed verification'), ('non_nscf', 'not converged')])
def test_scalar_wannier_rejects_unverified_source(tmp_path, source, mutation, match):
    evidence = wannier_spectrum_fixture(tmp_path, source)
    p = parameters()
    if mutation == 'column':
        p['broadening_index'] = 3
    elif mutation in {'labels', 'smearing'}:
        file = evidence / 'dft_agent.a2f'
        file.write_text(file.read_text().replace('Phonon smearing', 'Missing smearing') if mutation == 'labels' else file.read_text().replace('Electron smearing (eV) .1', 'Electron smearing (eV) .2'))
    elif mutation == 'unfinished':
        (evidence / 'epw.out').write_text('Program EPW v.5.9\n')
    else:
        file = evidence / 'data-file-schema.xml'
        file.write_text(file.read_text().replace('<calculation>nscf', '<calculation>scf'))
    with pytest.raises(ValueError, match=match):
        epw.prepare_inputs(source / 'Al.cif', tmp_path / 'rejected', 'epw_iso', p, evidence_dir=evidence)
