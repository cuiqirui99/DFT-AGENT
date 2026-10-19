"""Merged result evidence uses retained parser fixtures; model responses are mocked."""

import json
from pathlib import Path
import shutil

import pytest

from vasp_slurm_agent import agent, explanation, project_agent, project_analysis
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.workspace import Workspace
from test_explanation import accepted_run, model_reply
from test_qe_context import accepted_qe as qe_fixture
from test_project_input_artifacts import chain


SETTINGS = agent.ModelSettings('codex')


@pytest.fixture
def selected(tmp_path, accepted_run):
    store = Workspace(tmp_path / 'analysis-workspace')
    pid = store.create_project('Compare results')['id']
    sid = store.create_session(pid)['id']
    first = store.project_paths(pid)['runs'] / 'vasp'
    shutil.copytree(accepted_run[0], first)
    vasp = store.register_run(pid, first)
    qe_parent = tmp_path / 'qe-fixture'
    qe_parent.mkdir()
    qe_root = qe_fixture.__wrapped__(qe_parent)
    second = store.project_paths(pid)['runs'] / 'qe'
    shutil.copytree(qe_root, second)
    qe = store.register_run(pid, second)
    return store, pid, sid, [vasp['id'], qe['id']], [first, second]


def response():
    return model_reply('VASP reports {{run_a.stage_1.energy}}; QE reports {{run_b.stage_1.energy}}.',
        ['run_a.stage_1.energy', 'run_b.stage_1.energy'],
        limits=['The total energies are not directly comparable.'])


def provider(monkeypatch, replies):
    pending, captured = list(replies), []
    def request(payload, schema, settings, **kwargs):
        captured.append((kwargs['name'], json.loads(payload)))
        value = pending.pop(0)
        return json.dumps(value() if callable(value) else value)
    monkeypatch.setattr(agent, '_request_structured', request)
    return captured


def test_merged_namespaces_preserve_values_and_individual_hashes(selected):
    store, pid, _, ids, roots = selected
    merged = project_analysis.load_results(store, pid, ids)
    assert merged['facts']['run_a.stage_1.energy']['value'] == -10
    assert merged['facts']['run_b.stage_1.energy']['value'] == pytest.approx(-272.11386245988)
    for source, root in zip(merged['runs'], roots):
        assert source['context_sha256'] == explanation.load_run_context(root)['context_sha256']
    assert merged['runs'][0]['engine'] == 'vasp' and merged['runs'][1]['engine'] == 'qe'
    assert all(item['id'].startswith(('run_a.', 'run_b.')) for item in merged['artifacts'])
    assert not any(key.startswith('comparison.energy') for key in merged['facts'])
    payload = json.dumps(merged)
    assert 'private-host' not in payload and str(roots[0]) not in payload and '/private/licensed' not in payload
    assert project_analysis.load_results(Workspace(store.home), pid, ids) == merged


def test_grounded_values_use_shared_explainer_and_preserve_scope(selected, monkeypatch):
    store, pid, _, ids, _ = selected
    calls = provider(monkeypatch, [response()])
    answer = project_analysis.explain_results(store, pid, ids, SETTINGS, 'Show the recorded energies.')
    assert '-10.0 eV' in answer['answer'] and '-272.11386245988 eV' in answer['answer']
    assert len(answer['sources']) == 2
    assert answer['sources'][0]['id'] == ids[0] and answer['sources'][1]['id'] == ids[1]
    assert any('comparability has not been established' in limit for limit in answer['limits'])
    assert [name for name, _ in calls] == ['dft_explanation']
    assert 'run_a.stage_1.energy' in calls[0][1]['context']['facts']


@pytest.mark.parametrize('bad', ['number', 'identifier', 'missing_run', 'ranking', 'difference'])
def test_invalid_cross_run_answers_are_rejected(selected, monkeypatch, bad):
    store, pid, _, ids, _ = selected
    reply = response()
    if bad == 'number':
        reply['answer'] = 'The reported energy is -999 eV.'
    elif bad == 'identifier':
        reply['evidence'][0] = 'stage_1.energy'
    elif bad == 'missing_run':
        reply = model_reply('The VASP energy is {{run_a.stage_1.energy}}.', ['run_a.stage_1.energy'])
    elif bad == 'ranking':
        reply['answer'] = 'QE has lower energy than VASP and is more stable.'
    else:
        # Zero exists in the source moment facts, but is not a computed energy difference.
        reply['evidence'].append('run_a.stage_1.total_moment')
        reply['answer'] = 'The energy difference is 0 eV.'
    provider(monkeypatch, [reply])
    with pytest.raises(agent.AgentError, match='could not be verified'):
        project_analysis.explain_results(store, pid, ids, SETTINGS, 'Compare them.')


def test_source_change_during_model_answer_invalidates_entire_set(selected, monkeypatch):
    store, pid, _, ids, roots = selected
    def changing():
        (roots[1] / '01_scf/outputs/pw.out').write_text('changed evidence')
        return response()
    provider(monkeypatch, [changing])
    with pytest.raises(agent.AgentError, match='results changed'):
        project_analysis.explain_results(store, pid, ids, SETTINGS, 'Show both results.')


def test_foreign_selection_rejected_before_model(selected, monkeypatch):
    store, pid, _, ids, roots = selected
    foreign = store.create_project('Other')['id']
    target = store.project_paths(foreign)['runs'] / 'private'
    shutil.copytree(roots[1], target)
    outsider = store.register_run(foreign, target)
    monkeypatch.setattr(agent, '_request_structured', lambda *a, **k: pytest.fail('Foreign results reached model'))
    with pytest.raises(ValueError, match='not found'):
        project_analysis.explain_results(store, pid, [ids[0], outsider['id']], SETTINGS, 'Compare.')


@pytest.mark.parametrize('selection', ['one', 'four', 'duplicate'])
def test_selected_set_size_is_explicit(selected, selection):
    store, pid, _, ids, _ = selected
    chosen = ids[:1] if selection == 'one' else ids * 2 if selection == 'four' else [ids[0], ids[0]]
    with pytest.raises(ValueError, match='two or three'):
        project_analysis.load_results(store, pid, chosen)


def test_python_process_success_is_never_scientific_acceptance(selected, chain, monkeypatch):
    store, pid, _, ids, _ = selected
    root = store.project_paths(pid)['runs'] / 'python'
    shutil.copytree(chain[-1], root)
    record = store.register_run(pid, root)
    selected_ids = [*ids, record['id']]
    merged = project_analysis.load_results(store, pid, selected_ids)
    assert merged['facts']['run_c.run.process_success']['value'] is True
    assert merged['facts']['run_c.run.accepted']['value'] is False
    valid = model_reply('VASP and QE passed their scoped checks. The Python program completed.',
        ['run_a.stage_1.accepted', 'run_b.stage_1.accepted', 'run_c.run.process_success'])
    invalid = {**valid, 'answer': 'VASP and QE passed their checks. Python results are validated.'}
    provider(monkeypatch, [valid, invalid])
    answer = project_analysis.explain_results(store, pid, selected_ids, SETTINGS, 'What finished?')
    assert any('Python execution' in limit for limit in answer['limits'])
    with pytest.raises(agent.AgentError, match='could not be verified'):
        project_analysis.explain_results(store, pid, selected_ids, SETTINGS, 'Are these scientific results accepted?')


def test_project_chat_explains_all_explicit_attachments_and_retains_provenance(selected, monkeypatch):
    store, pid, sid, ids, _ = selected
    registry = ConnectionRegistry(store)
    model = registry.create('model', 'Fixture model', {'provider': 'codex'})
    registry.bind(pid, model['id'])
    routing = {'action': 'explain', 'message': 'Review the selected results.', 'run_id': None,
        'workflow_id': None, 'step_id': None, 'workflow': None, 'file': None, 'memory': None}
    calls = provider(monkeypatch, [routing, response()])
    answer = project_agent.send(store, pid, sid, 'Show both recorded energies, without ranking.', model['id'], run_ids=ids)
    assert answer['action'] == 'explain' and len(answer['result']['sources']) == 2
    assert [item['id'] for item in answer['result']['sources']] == ids
    assert [name for name, _ in calls] == ['dft_project_action', 'dft_explanation']
    assert 'run_a.stage_1.energy' in calls[-1][1]['context']['facts']
    saved = Workspace(store.home).get_session(pid, sid)['messages']
    assert len(saved) == 2
    assert saved[-1]['metadata']['analysis']['sources'] == answer['result']['sources']
    assert saved[-1]['metadata']['analysis']['limits'] == answer['result']['limits']
