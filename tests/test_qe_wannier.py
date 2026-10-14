import pytest

from vasp_slurm_agent import qe_wannier


def records():
    return {name: {"size": 10, "sha256": "a" * 64} for name in (
        "dft_agent.dyn1", "dft_agent.dyn2", "scratch/_ph0/dft_agent.dvscf1",
        "scratch/_ph0/dft_agent.q_2/dft_agent.dvscf1",
        "scratch/_ph0/dft_agent.phsave/control_ph.xml",
        "scratch/_ph0/dft_agent.phsave/patterns.1.xml",
        "scratch/dft_agent.save/data-file-schema.xml", "scratch/dft_agent.wfc1",
    )}


def test_derivative_mapping_does_not_overwrite_nscf():
    mapping = qe_wannier.phonon_checkpoint_mapping(records(), 2)
    assert len(mapping) == 6
    assert mapping["scratch/_ph0/dft_agent.q_2/dft_agent.dvscf1"] == "save/dft_agent.dvscf_q2"
    assert all(name.startswith("save/") for name in mapping.values())
    assert not any("wfc" in name or "data-file-schema" in name for name in mapping)


@pytest.mark.parametrize("change", ["missing_matrix", "missing_derivative", "ambiguous_derivative", "unsafe", "missing_patterns"])
def test_bad_derivative_checkpoint_rejected(change):
    data = records()
    if change == "missing_matrix": del data["dft_agent.dyn2"]
    if change == "missing_derivative": del data["scratch/_ph0/dft_agent.dvscf1"]
    if change == "ambiguous_derivative": data["scratch/_ph0/dft_agent.dvscf"] = {}
    if change == "unsafe": data["scratch/../escape"] = {}
    if change == "missing_patterns": del data["scratch/_ph0/dft_agent.phsave/patterns.1.xml"]
    with pytest.raises(ValueError): qe_wannier.phonon_checkpoint_mapping(data, 2)


def settings(**changes):
    return dict(mesh=[6,6,6],wannier_mesh=[3,3,3],qmesh=[3,3,3],fine_kmesh=[6,6,6],fine_qmesh=[6,6,6],nbnd=8,num_wann=4,projections=['Al:s','Al:p'],dis_win_min=-10,dis_win_max=35,dis_froz_min=-10,dis_froz_max=8,fsthick_ev=10,el_smearing_ev=.1,ph_smearing_mev=.5,**changes)


def test_generated_full_nscf_and_phonon_contract(tmp_path):
    from pymatgen.core import Structure,Lattice
    structure=tmp_path/'Al.cif'
    Structure(Lattice.cubic(4),['Al'],[[0,0,0]]).to(filename=structure)
    nc={'Al':{'filename':'Al.upf','pseudo_type':'NC','sha256':'a'*64,'fully_relativistic':False}}
    ns=qe_wannier.prepare_inputs(structure,tmp_path/'nscf','wannier_nscf',settings(),nc)
    text=(tmp_path/'nscf/pw.in').read_text()
    assert 'K_POINTS crystal\n27\n' in text and 'nosym' not in text
    assert len(ns['explicit_kpoints'])==27
    ph=qe_wannier.prepare_inputs(structure,tmp_path/'ph','wannier_ph',settings(),nc)
    assert "fildvscf = 'dvscf'" in (tmp_path/'ph/ph.in').read_text()
    interp=qe_wannier.prepare_inputs(structure,tmp_path/'epw','epw_interpolate',settings(),nc)
    assert "dvscf_dir = 'save'" in (tmp_path/'epw/epw.in').read_text()
    assert interp['requires_restart'] and ph['phonon_grid']==[3,3,3]


@pytest.mark.parametrize('mutation', [{'num_wann':9},{'fine_kmesh':[5,6,6]},{'dis_froz_max':40},{'projections':['Al:s\nmalformed']},{'functional':'HSE06'},{'acoustic_cutoff_cm1':20}])
def test_bad_wannier_settings_rejected(mutation):
    p=settings();p.update(mutation)
    with pytest.raises(ValueError):qe_wannier.validate_parameters(p)


def test_paw_is_not_silently_used_for_epw():
    with pytest.raises(ValueError,match='norm-conserving'):
        qe_wannier._require_nc({'Al':{'filename':'Al.upf','pseudo_type':'PAW','sha256':'a'*64}})
