"""Dependency and input checks for the coupled superconductivity workflow."""
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import qe_workflow as workflow
from vasp_slurm_agent.config import ClusterConfig


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / 'Al.cif'
    Structure(Lattice([[0, 2, 2], [2, 0, 2], [2, 2, 0]]), ['Al'], [[0, 0, 0]]).to(filename=source)
    config = ClusterConfig(engine='qe', host='cluster.invalid', user='test', remote_root='/scratch/test',
                           partition='cpu', qe_pseudo_root='/pseudo', qe_pseudopotentials={'Al': 'Al.UPF'})
    return source, config


def eph(**updates):
    return dict(mesh=[8, 8, 8], dense_mesh=[16, 16, 16], qmesh=[4, 4, 4],
                phonon_dos_mesh=[20, 20, 20], mu_star=.1, **updates)


def epw(**updates):
    return dict(temperatures=[1, 2, 4], mu_star=.1, frequency_cutoff_ev=.2,
                broadening_index=2, **updates)


def test_epw_is_a_dependent_spectrum_stage_without_a_wavefunction_copy(tmp_path, setup):
    source, config = setup
    root = tmp_path / 'run'
    state = workflow.prepare_plan(source, root, config, ['electron_phonon', 'epw_tc'],
                                  stage_parameters=[eph(), epw()])
    stage = state['stages'][-1]
    assert stage['name'] == 'epw_tc' and stage['evidence_from'] == 5
    assert stage['restart_from'] is None and stage['materialized'] is False
    assert not (root / stage['folder'] / 'inputs').exists()
    assert not any(s['job_id'] for s in state['stages'])
    state['current_stage'] = 6
    workflow._save(root, state)
    class NoNetwork:
        def run(self, *args, **kwargs):
            pytest.fail('Unaccepted source must not reach the cluster')
    blocked = workflow.advance(root, NoNetwork())
    assert blocked['status'] == 'needs_attention'
    assert 'has not been accepted' in blocked['last_error']


def test_q_sum_is_explicit_and_keeps_the_interpolated_default(tmp_path, setup):
    source, config = setup
    direct = eph(eph_integration='q_sum')
    direct.pop('phonon_dos_mesh')
    a = workflow.prepare_plan(source, tmp_path / 'direct', config,
                              ['electron_phonon', 'epw_iso'], stage_parameters=[direct, epw()])
    b = workflow.prepare_plan(source, tmp_path / 'fourier', config,
                              ['electron_phonon'], eph())
    assert [s['name'] for s in a['stages']] == ['eph_dense_scf', 'eph_scf', 'eph_ph', 'eph_lambda', 'epw_iso']
    assert [s['name'] for s in b['stages']][3:5] == ['eph_q2r', 'eph_matdyn']
    assert a['stages'][-1]['evidence_from'] == 3


def test_epw_cannot_invent_a_spectrum_or_silently_choose_a_broadening(tmp_path, setup):
    source, config = setup
    with pytest.raises(ValueError, match='accepted electron-phonon spectrum'):
        workflow.prepare_plan(source, tmp_path / 'no-source', config, ['epw_iso'], epw())
    parameters = epw()
    parameters.pop('broadening_index')
    with pytest.raises(ValueError, match='broadening_index'):
        workflow.prepare_plan(source, tmp_path / 'no-choice', config, ['electron_phonon', 'epw_iso'],
                              stage_parameters=[eph(), parameters])


def test_fresh_wannier_chain_freezes_both_checkpoint_dependencies(tmp_path, setup):
    import json
    from dataclasses import replace
    source, config = setup
    config = replace(config, qe_pseudopotentials={'Al': {'filename': 'Al.UPF', 'pseudo_type': 'NC',
                     'sha256': 'a' * 64, 'ecutwfc': 60, 'ecutrho': 240}})
    recipe = dict(mesh=[6, 6, 6], wannier_mesh=[3, 3, 3], qmesh=[3, 3, 3], fine_kmesh=[6, 6, 6],
                  fine_qmesh=[6, 6, 6], nbnd=8, num_wann=4, projections=['Al:s', 'Al:p'],
                  dis_win_min=-10, dis_win_max=35, dis_froz_min=-10, dis_froz_max=8,
                  fsthick_ev=2, el_smearing_ev=.1, ph_smearing_mev=.5)
    gap = dict(temperatures=[1, 2, 4], mu_star=.1, frequency_cutoff_ev=.2)
    root = tmp_path / 'wannier'
    state = workflow.prepare_plan(source, root, config, ['wannier_electron_phonon', 'epw_aniso'],
                                  stage_parameters=[recipe, gap])
    assert [stage['name'] for stage in state['stages']] == ['wannier_scf', 'wannier_ph', 'wannier_nscf', 'epw_interpolate', 'epw_aniso']
    assert [stage['restart_from'] for stage in state['stages']] == [None, 0, 0, 2, 3]
    assert state['stages'][3]['auxiliary_from'] == 1
    assert state['stages'][4]['evidence_from'] == 3
    assert not state['stages'][4]['materialized']
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['stages'][3]['auxiliary_from'] == 1 and plan['stages'][4]['evidence_from'] == 3
    assert all(stage['job_id'] is None for stage in state['stages'])
    nscf = (root / '03_wannier_nscf/inputs/pw.in').read_text()
    assert 'K_POINTS crystal\n27\n' in nscf


@pytest.mark.parametrize('submitted', [False, True])
def test_resume_refreshes_only_unsubmitted_unstaged_inputs(tmp_path, setup, submitted):
    from copy import deepcopy
    source, config = setup
    root = tmp_path / 'resume'
    state = workflow.prepare_plan(source, root, config, ['scf'])
    stage = state['stages'][0]
    before = deepcopy(stage['metadata'])
    frozen_plan = (root / 'plan.json').read_bytes()
    stage.update(status='submitting', staged=submitted, job_id='8001' if submitted else None)
    state.update(status='needs_attention', last_error='Preparation failed')
    workflow._save(root, state)
    resumed = workflow.resume(root)
    assert resumed['status'] == ('queued' if submitted else 'planned')
    assert resumed['stages'][0]['materialized'] is submitted
    assert resumed['stages'][0]['metadata'] == before
    assert resumed['stages'][0]['job_id'] == ('8001' if submitted else None)
    assert (root / 'plan.json').read_bytes() == frozen_plan
