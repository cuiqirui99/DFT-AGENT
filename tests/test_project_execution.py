"""Real project/plan/input integration; model and process launch explicitly mocked."""

import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import pytest

from vasp_slurm_agent import agent, conversation, conversation_store, workflow
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_execution import ProjectExecution
from vasp_slurm_agent.project_files import write_file
from vasp_slurm_agent.workspace import Workspace
from test_conversation import plan, provider, route, SETTINGS
from test_continuation import parent
from test_explanation import accepted_run


@pytest.fixture
def project(tmp_path):
    store = Workspace(tmp_path / 'workspace')
    pid = store.create_project('Silicon')['id']
    sid = store.create_session(pid)['id']
    reg = ConnectionRegistry(store)
    connection = reg.create('hpc', 'Test fixture cluster', {'host': 'test.invalid', 'user': 'researcher',
        'scheduler': 'slurm', 'remote_base': '/scratch/test'})
    reg.bind(pid, connection['id'])
    env = reg.create_environment(connection['id'], 'VASP fixture', 'vasp',
        {'vasp_command': 'srun vasp_std', 'potcar_root': '/licensed', 'partition': 'cpu'})
    write_file(store, pid, 'files/Si.cif', files('vasp_slurm_agent').joinpath('examples/Si.cif').read_bytes())
    spec = {'goal': 'SCF only', 'source': {'kind': 'file', 'path': 'files/Si.cif'}, 'environment_id': env['id']}
    graph = store.create_workflow(pid, sid, 'Si SCF', 'SCF only',
        [{'id': 'scf', 'name': 'SCF', 'engine': 'vasp', 'spec': spec, 'depends_on': []}])
    return store, reg, ProjectExecution(store, reg), pid, graph['id'], connection['id']


def draft(project, monkeypatch, size=2):
    store, reg, execution, pid, wid, conn = project
    provider(monkeypatch, [('dft_conversation', route()), ('dft_task', plan('vasp', size))])
    return execution.draft_step(pid, wid, 'scf', SETTINGS)


def prepared(project, monkeypatch):
    draft(project, monkeypatch)
    _, _, execution, pid, wid, _ = project
    return execution.prepare_step(pid, wid, 'scf')


def test_revise_prepare_restart_and_submit_once(project, monkeypatch):
    store, reg, execution, pid, wid, _ = project
    first = draft(project, monkeypatch, 2)
    second = draft(project, monkeypatch, 3)
    assert first['id'] == second['id'] and second['state'] == 'ready'
    result = execution.prepare_step(pid, wid, 'scf')
    assert result['state'] == 'prepared'
    root = Path(store.get_run(pid, result['run_id'])['run_dir'])
    from pymatgen.core import Structure
    assert len(Structure.from_file(root / '01_scf/inputs/POSCAR')) == 6
    assert all(not s['job_id'] for s in workflow.read_state(root)['stages'])
    execution = ProjectExecution(Workspace(store.home))
    assert execution.prepare_step(pid, wid, 'scf')['run_id'] == result['run_id']
    with pytest.raises(ValueError, match='Authorize'):
        execution.launch_step(pid, wid, 'scf')
    review = execution.review(pid, wid, 'scf')
    execution.authorize_step(pid, wid, 'scf', review['digest'])
    launched = []
    monkeypatch.setattr(workflow, 'start_worker', lambda path: launched.append(path) or 123)
    execution.launch_step(pid, wid, 'scf')
    execution.launch_step(pid, wid, 'scf')
    assert launched == [str(root)]
    with pytest.raises(ValueError, match='active'):
        store.delete_project(pid, confirmation='Silicon')


def test_modified_inputs_invalidate_authorization(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    attempt = prepared(project, monkeypatch)
    review = execution.review(pid, wid, 'scf')
    execution.authorize_step(pid, wid, 'scf', review['digest'])
    incar = Path(review['run_dir']) / '01_scf/inputs/INCAR'
    incar.write_text(incar.read_text() + '\nENCUT=100\n')
    monkeypatch.setattr(workflow, 'start_worker', lambda *a: pytest.fail('Changed inputs launched'))
    with pytest.raises(ValueError, match='changed'):
        execution.launch_step(pid, wid, 'scf')
    assert execution.latest(pid, wid, 'scf')['approved_digest'] is None


def test_revoked_connection_blocks_previously_authorized_launch(project, monkeypatch):
    _, reg, execution, pid, wid, conn = project
    prepared(project, monkeypatch)
    execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])
    reg.revoke(conn)
    monkeypatch.setattr(workflow, 'start_worker', lambda *a: pytest.fail('Revoked connection launched'))
    with pytest.raises(ValueError, match='active HPC|revoked'):
        execution.launch_step(pid, wid, 'scf')


def test_archive_blocks_planning_and_launch(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    prepared(project, monkeypatch)
    execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])
    store.archive_project(pid)
    monkeypatch.setattr(workflow, 'start_worker', lambda *a: pytest.fail('Archived project launched'))
    with pytest.raises(ValueError, match='active project|Restore'):
        execution.launch_step(pid, wid, 'scf')


def test_missing_environment_allows_draft_but_not_preparation(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    with store._transaction() as db:
        spec = json.loads(db.execute('SELECT spec FROM steps WHERE workflow_id=?', (wid,)).fetchone()[0])
        spec['environment_id'] = None
        db.execute('UPDATE steps SET spec=? WHERE workflow_id=?', (json.dumps(spec), wid))
    assert draft(project, monkeypatch)['state'] == 'ready'
    with pytest.raises(ValueError, match='environment'):
        execution.prepare_step(pid, wid, 'scf')


def test_project_and_source_boundaries(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    foreign = store.create_project('Other')['id']
    with pytest.raises(ValueError):
        execution.latest(foreign, wid, 'scf')
    with store._transaction() as db:
        spec = json.loads(db.execute('SELECT spec FROM steps WHERE workflow_id=?', (wid,)).fetchone()[0])
        spec['source']['path'] = '../Other/Si.cif'
        db.execute('UPDATE steps SET spec=? WHERE workflow_id=?', (json.dumps(spec), wid))
    monkeypatch.setattr(agent, '_request_structured', lambda *a, **k: pytest.fail('Invalid source reached model'))
    with pytest.raises(ValueError, match='input structure'):
        execution.draft_step(pid, wid, 'scf', SETTINGS)


def test_source_edit_requires_new_plan(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    draft(project, monkeypatch)
    source = store.resolve_path(pid, 'files/Si.cif')
    source.write_text(source.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='structure changed'):
        execution.prepare_step(pid, wid, 'scf')
    assert store.list_runs(pid) == []


def test_changed_source_can_be_redrafted_without_reusing_old_geometry(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    original = draft(project, monkeypatch, 2)
    source = store.resolve_path(pid, 'files/Si.cif')
    from pymatgen.core import Structure
    changed = Structure.from_file(source)
    changed.make_supercell([1, 2, 1])
    changed.to(filename=str(source))
    fresh = draft(project, monkeypatch, 3)
    assert fresh['id'] != original['id']
    attempts = execution.attempts(pid, wid, 'scf')
    assert [item['state'] for item in attempts] == ['superseded', 'ready']
    old = conversation_store.load(store.resolve_path(pid, original['conversation_path']))
    new = conversation_store.load(store.resolve_path(pid, fresh['conversation_path']))
    assert [m['content'] for m in old['messages'] if m['role'] == 'user'][0] in [m['content'] for m in new['messages'] if m['role'] == 'user']
    prepared = execution.prepare_step(pid, wid, 'scf')
    root = Path(store.get_run(pid, prepared['run_id'])['run_dir'])
    assert len(Structure.from_file(root / '01_scf/inputs/POSCAR')) == 12


def test_missing_conversation_does_not_block_prepared_run_recovery(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    initial = draft(project, monkeypatch)
    register = store.register_run
    monkeypatch.setattr(store, 'register_run', lambda *a, **k: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError):
        execution.prepare_step(pid, wid, 'scf')
    shutil.rmtree(store.resolve_path(pid, initial['conversation_path']))
    monkeypatch.setattr(store, 'register_run', register)
    recovered = execution.recover_step(pid, wid, 'scf')
    assert recovered['id'] == initial['id'] and recovered['state'] == 'prepared'
    assert recovered['conversation_error'] and recovered['approved_digest'] is None
    assert len(store.list_runs(pid)) == 1


def test_recovery_does_not_offer_new_authorization_for_already_submitted_run(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    initial = draft(project, monkeypatch)
    register = store.register_run
    monkeypatch.setattr(store, 'register_run', lambda *a, **k: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError):
        execution.prepare_step(pid, wid, 'scf')
    root = store.project_paths(pid)['runs'] / initial['id']
    state = workflow.read_state(root)
    state['status'] = 'running'
    state['stages'][0].update(job_id='123', staged=True, status='running', scheduler_state='RUNNING')
    (root / 'run.json').write_text(json.dumps(state))
    monkeypatch.setattr(store, 'register_run', register)
    recovered = execution.recover_step(pid, wid, 'scf')
    assert recovered['state'] == 'running' and recovered['approved_digest'] is None
    with pytest.raises(ValueError, match='Review'):
        execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])


def test_repair_questions_reuse_one_attempt_and_preserve_original_run(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    original = prepared(project, monkeypatch)
    root = Path(store.get_run(pid, original['run_id'])['run_dir'])
    state = workflow.read_state(root)
    state['status'] = 'failed'
    state['stages'][0].update(status='failed', scheduler_state='FAILED', job_id='123', staged=True)
    (root / 'run.json').write_text(json.dumps(state))
    before = (root / 'run.json').read_bytes()
    def question(folder, message, settings, **kwargs):
        record = conversation_store.load(folder)
        record['messages'].append({'role': 'user', 'content': message})
        conversation_store.save(folder, record, expected_revision=record['revision'])
        return record
    monkeypatch.setattr(conversation, 'send', question)
    first = execution.retry_step(pid, wid, 'scf', SETTINGS, 'Plan a repair.')
    second = execution.retry_step(pid, wid, 'scf', SETTINGS, 'Keep the accuracy settings.')
    assert first['id'] == second['id'] != original['id']
    assert second['state'] == 'needs_input' and second['run_id'] is None
    assert len(execution.attempts(pid, wid, 'scf')) == 2
    assert (root / 'run.json').read_bytes() == before


def test_prepared_source_tamper_is_rejected_before_authorization(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    attempt = prepared(project, monkeypatch)
    review = execution.review(pid, wid, 'scf')
    root = Path(review['run_dir'])
    plan = json.loads((root / 'plan.json').read_text())
    source = root / next(iter(plan['sources']))
    source.write_text(source.read_text() + '\nchanged\n')
    with pytest.raises(ValueError, match='source structure changed'):
        execution.authorize_step(pid, wid, 'scf', review['digest'])
    assert execution.latest(pid, wid, 'scf')['state'] == 'prepared'


def test_failed_register_recovers_same_prepared_run_without_repeating(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    draft(project, monkeypatch)
    original = store.register_run
    monkeypatch.setattr(store, 'register_run', lambda *a, **k: (_ for _ in ()).throw(OSError('Simulated DB failure')))
    with pytest.raises(OSError):
        execution.prepare_step(pid, wid, 'scf')
    attempt = execution.latest(pid, wid, 'scf')
    monkeypatch.setattr(store, 'register_run', original)
    recovered = execution.recover_step(pid, wid, 'scf')
    assert recovered['id'] == attempt['id'] and recovered['state'] == 'prepared'
    assert len(store.list_runs(pid)) == 1
    assert execution.prepare_step(pid, wid, 'scf')['run_id'] == recovered['run_id']


def test_replace_workflow_preserves_history_and_rejects_prepared(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    old = store.get_workflow(pid, wid)
    steps = [{k: s[k] for k in ('id','name','engine','spec','depends_on')} for s in old['steps']]
    prepared(project, monkeypatch)
    with pytest.raises(ValueError, match='follow-up'):
        execution.replace_workflow(pid, wid, 'Changed', 'Changed', steps)
    assert store.get_workflow(pid, wid)['title'] == 'Si SCF'
    new = store.create_workflow(pid, old['session_id'], 'Draft', 'Draft', steps)
    replacement = execution.replace_workflow(pid, new['id'], 'Revised', 'Revised', steps)
    assert replacement['id'] != new['id']
    assert store.get_workflow(pid, new['id'])['status'] == 'superseded'


def batch_attempt(project):
    from vasp_slurm_agent import batch
    store, registry, execution, pid, wid, _ = project
    _, step = execution._step(pid, wid, 'scf')
    attempt = execution._new(pid, wid, step)
    root = store.project_paths(pid)['runs'] / attempt['id']
    batch.prepare_batch(store.resolve_path(pid, 'files/Si.cif'), root,
        execution._config(pid, step, attempt, required=True), ['scf'], [{'label': 'A'}, {'label': 'B'}])
    record = store.register_run(pid, root, wid, 'scf')
    return execution._save(attempt, run_id=record['id'], state='prepared'), root


def test_batch_review_binds_every_member_and_reconnect_cannot_start_jobs(project, monkeypatch):
    from vasp_slurm_agent import batch
    store, _, execution, pid, wid, _ = project
    attempt, root = batch_attempt(project)
    review = execution.review(pid, wid, 'scf')
    assert review['kind'] == 'batch'
    assert len([key for key in review['files'] if key.endswith('/inputs/INCAR')]) == 2
    children = batch._children(root, batch.read_batch(root))
    first = children[0][1] / '01_scf/inputs/INCAR'
    first.write_text(first.read_text() + '\n# review changed\n')
    with pytest.raises(ValueError, match='inputs changed'):
        execution.authorize_step(pid, wid, 'scf', review['digest'])
    for name in ('advance', 'start_worker'):
        monkeypatch.setattr(workflow, name, lambda *a, **k: pytest.fail('Reconnect must not start planned work'))
    monkeypatch.setattr(batch, 'start_batch_worker', lambda *a, **k: pytest.fail('No batch launch'))
    result = execution.refresh(pid, wid, reconnect=True)
    assert result['steps'][0]['attempt']['state'] == 'prepared'
    assert all(not stage.get('job_id') for _, _, child in batch._children(root, batch.read_batch(root)) for stage in child['stages'])


def test_batch_repair_requires_member_selection_before_creating_attempt(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    batch_attempt(project)
    monkeypatch.setattr(conversation, 'open_run', lambda *a, **k: pytest.fail('Select a member before opening repair'))
    with pytest.raises(ValueError, match='comparison member'):
        execution.retry_step(pid, wid, 'scf', SETTINGS, 'Repair the calculation.')
    assert len(execution.attempts(pid, wid, 'scf')) == 1


def test_reconnect_uses_locked_poll_only_for_known_submitted_stage(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    attempt = prepared(project, monkeypatch)
    root = Path(store.get_run(pid, attempt['run_id'])['run_dir'])
    state = workflow.read_state(root)
    state['status'] = 'running'
    state['stages'][0].update(status='running', scheduler_state='RUNNING', job_id='123', staged=True)
    (root / 'run.json').write_text(json.dumps(state))
    calls = []
    monkeypatch.setattr(workflow, 'advance', lambda path, *, poll_only: calls.append(poll_only))
    execution.refresh(pid, wid, reconnect=True)
    assert calls == [True]


def test_resume_monitoring_restarts_existing_job_without_resubmitting(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    attempt = prepared(project, monkeypatch)
    review = execution.review(pid, wid, 'scf')
    execution.authorize_step(pid, wid, 'scf', review['digest'])
    root = Path(review['run_dir'])
    state = workflow.read_state(root)
    state['status'] = 'needs_attention'
    state['stages'][0].update(status='running', scheduler_state='RUNNING', job_id='123', staged=True)
    (root / 'run.json').write_text(json.dumps(state))
    execution._save(attempt, state='needs_attention')
    launched = []
    monkeypatch.setattr(workflow, 'start_worker', lambda path: launched.append(Path(path)))
    monkeypatch.setattr(workflow, 'advance', lambda *a, **k: pytest.fail('Resuming must restart the existing worker path'))
    resumed = execution.resume_monitoring(pid, wid, 'scf', confirmed=True)
    assert resumed['state'] == 'active' and resumed['run_id'] == attempt['run_id']
    assert launched == [root]
    state = workflow.read_state(root)
    assert state['status'] == 'queued' and state['stages'][0]['job_id'] == '123'
    assert state['stages'][0]['staged'] and len(execution.attempts(pid, wid, 'scf')) == 1


def test_resume_monitoring_can_continue_previously_authorized_remaining_stages(project, monkeypatch):
    store, _, execution, pid, wid, _ = project
    attempt = draft(project, monkeypatch)
    _, step = execution._step(pid, wid, 'scf')
    root = store.project_paths(pid)['runs'] / attempt['id']
    workflow.prepare_plan(store.resolve_path(pid, 'files/Si.cif'), root,
        execution._config(pid, step, attempt, required=True), ['scf', 'bands'])
    run = store.register_run(pid, root, wid, 'scf')
    attempt = execution._save(attempt, run_id=run['id'], state='prepared')
    execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])
    execution._save(attempt, state='active')
    state = workflow.read_state(root)
    state.update(status='planned', current_stage=1)
    state['stages'][0].update(status='succeeded', scheduler_state='COMPLETED', job_id='123', staged=True)
    (root / 'run.json').write_text(json.dumps(state))
    launched = []
    monkeypatch.setattr(workflow, 'start_worker', lambda path: launched.append(Path(path)))
    result = execution.resume_monitoring(pid, wid, 'scf', confirmed=True)
    assert launched == [root] and result['run_id'] == run['id']
    current = workflow.read_state(root)
    assert current['stages'][0]['job_id'] == '123'
    assert current['stages'][1]['job_id'] is None and current['status'] == 'planned'


@pytest.mark.parametrize('problem', ['unsubmitted', 'unconfirmed', 'cancelled', 'cancelling', 'inputs', 'config', 'revoked', 'archived'])
def test_resume_monitoring_keeps_submission_and_integrity_gates(project, monkeypatch, problem):
    store, registry, execution, pid, wid, connection = project
    attempt = prepared(project, monkeypatch)
    root = Path(store.get_run(pid, attempt['run_id'])['run_dir'])
    if problem != 'unsubmitted':
        execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])
        execution._save(attempt, state='active')
    if problem in {'cancelled', 'cancelling'}:
        state = workflow.read_state(root)
        state.update(status='cancelled' if problem == 'cancelled' else 'running', cancel_requested=True)
        (root / 'run.json').write_text(json.dumps(state))
    elif problem == 'inputs':
        path = root / '01_scf/inputs/INCAR'
        path.write_text(path.read_text() + '\nENCUT=100\n')
    elif problem == 'config':
        path = root / 'config.json'
        path.write_text(path.read_text() + '\n')
    elif problem == 'revoked':
        registry.revoke(connection)
    elif problem == 'archived':
        store.archive_project(pid)
    monkeypatch.setattr(workflow, 'start_worker', lambda *a, **k: pytest.fail('Invalid resume launched'))
    with pytest.raises(ValueError):
        execution.resume_monitoring(pid, wid, 'scf', confirmed=problem != 'unconfirmed')


def test_resume_batch_monitoring_preserves_members_and_authorization(project, monkeypatch):
    from vasp_slurm_agent import batch
    _, _, execution, pid, wid, _ = project
    attempt, root = batch_attempt(project)
    execution.authorize_step(pid, wid, 'scf', execution.review(pid, wid, 'scf')['digest'])
    execution._save(attempt, state='active')
    before = [(member['run_id'], path) for member, path, _ in batch._children(root, batch.read_batch(root))]
    launched = []
    monkeypatch.setattr(batch, 'start_batch_worker', lambda path: launched.append(Path(path)))
    monkeypatch.setattr(workflow, 'start_worker', lambda *a, **k: pytest.fail('Use the batch worker'))
    result = execution.resume_monitoring(pid, wid, 'scf', confirmed=True)
    assert launched == [root] and result['run_id'] == attempt['run_id']
    assert before == [(member['run_id'], path) for member, path, _ in batch._children(root, batch.read_batch(root))]


def script_graph(project, engine='python', environment=True):
    import shlex
    import sys
    store, registry, execution, pid, old_wid, connection = project
    old = store.get_workflow(pid, old_wid)
    env = registry.create_environment(connection, 'Program fixture', engine,
        {'command': shlex.quote(sys.executable) if engine == 'python' else 'srun wannier90.x', 'partition': 'cpu', 'tasks': 1})
    relative = 'files/analyze.py' if engine == 'python' else 'files/model.win'
    write_file(store, pid, relative, "from pathlib import Path\nPath('result.json').write_text('{\"value\": 45}')\n" if engine == 'python' else 'num_wann = 2\nwrite_hr = true\n')
    spec = {'goal': 'Run the reviewed inputs', 'source': {'kind': 'file', 'path': relative},
            'environment_id': env['id'] if environment else None, 'inputs': [], 'arguments': [],
            'outputs': [{'path': 'result.json', 'kind': 'json', 'required': True}] if engine == 'python' else []}
    graph = store.create_workflow(pid, old['session_id'], 'Program', spec['goal'],
        [{'id': 'program', 'name': 'Program', 'engine': engine, 'spec': spec, 'depends_on': []}])
    return graph['id'], env['id']


@pytest.mark.parametrize('engine', ['python', 'wannier90'])
def test_generic_draft_prepare_restart_approval_and_single_launch(project, monkeypatch, engine):
    from vasp_slurm_agent import script_workflow
    store, _, execution, pid, _, _ = project
    wid, _ = script_graph(project, engine)
    monkeypatch.setattr(agent, '_request_structured', lambda *a, **k: pytest.fail('File review cannot fabricate another plan'))
    monkeypatch.setattr(script_workflow, 'SSHTransport', lambda *a, **k: pytest.fail('Drafting cannot contact a cluster'))
    draft = execution.draft_step(pid, wid, 'program', SETTINGS)
    assert draft['state'] == 'ready' and draft['proposal']['kind'] == 'script'
    assert draft['proposal']['limits'] and not draft['run_id']
    result = execution.prepare_step(pid, wid, 'program')
    root = Path(store.get_run(pid, result['run_id'])['run_dir'])
    assert workflow.read_state(root)['engine'] == engine
    assert not workflow.read_state(root)['stages'][0]['job_id']
    if engine == 'wannier90':
        assert {entry['path'] for entry in workflow.read_state(root)['stages'][0]['outputs']} == {'model.wout', 'model_hr.dat'}
    execution = ProjectExecution(Workspace(store.home))
    assert execution.prepare_step(pid, wid, 'program')['run_id'] == result['run_id']
    with pytest.raises(ValueError, match='Authorize'):
        execution.launch_step(pid, wid, 'program')
    execution.authorize_step(pid, wid, 'program', execution.review(pid, wid, 'program')['digest'])
    launched = []
    monkeypatch.setattr(script_workflow, 'start_worker', lambda root: launched.append(root))
    execution.launch_step(pid, wid, 'program')
    execution.launch_step(pid, wid, 'program')
    assert launched == [str(root)]


@pytest.mark.parametrize('status', ['needs_attention', 'completed_unverified', 'failed', 'unconverged', 'missing_outputs', 'suspect'])
def test_resume_script_monitoring_respects_execution_outcome(project, monkeypatch, status):
    from vasp_slurm_agent import script_workflow
    store, _, execution, pid, _, _ = project
    wid, _ = script_graph(project)
    execution.draft_step(pid, wid, 'program', SETTINGS)
    attempt = execution.prepare_step(pid, wid, 'program')
    review = execution.review(pid, wid, 'program')
    execution.authorize_step(pid, wid, 'program', review['digest'])
    execution._save(attempt, state='active')
    root = Path(review['run_dir'])
    state = workflow.read_state(root)
    state['status'] = status
    state['stages'][0].update(status=status, job_id='123', staged=True)
    (root / 'run.json').write_text(json.dumps(state))
    launched = []
    monkeypatch.setattr(script_workflow, 'start_worker', lambda path: launched.append(Path(path)))
    monkeypatch.setattr(workflow, 'start_worker', lambda *a: pytest.fail('Use the program worker'))
    if status == 'needs_attention':
        result = execution.resume_monitoring(pid, wid, 'program', confirmed=True)
        assert result['state'] == 'active' and launched == [root]
        assert workflow.read_state(root)['status'] == 'queued'
    else:
        with pytest.raises(ValueError):
            execution.resume_monitoring(pid, wid, 'program', confirmed=True)
        assert not launched and workflow.read_state(root)['status'] == status
    assert workflow.read_state(root)['stages'][0]['job_id'] == '123'


def test_generic_invalid_revision_clears_ready_and_new_source_is_reviewable(project):
    store, _, execution, pid, _, _ = project
    wid, _ = script_graph(project, environment=False)
    first = execution.draft_step(pid, wid, 'program', SETTINGS)
    with pytest.raises(ValueError, match='environment'):
        execution.prepare_step(pid, wid, 'program')
    path = store.resolve_path(pid, 'files/analyze.py')
    path.write_text('not python ?')
    with pytest.raises(SyntaxError):
        execution.draft_step(pid, wid, 'program', SETTINGS)
    assert execution.latest(pid, wid, 'program')['state'] == 'needs_attention'
    path.write_text("print('updated')\n")
    second = execution.draft_step(pid, wid, 'program', SETTINGS)
    assert second['id'] != first['id'] and second['state'] == 'ready'


def test_generic_frozen_inputs_are_bound_to_review(project):
    store, _, execution, pid, _, _ = project
    wid, _ = script_graph(project)
    execution.draft_step(pid, wid, 'program', SETTINGS)
    execution.prepare_step(pid, wid, 'program')
    review = execution.review(pid, wid, 'program')
    (Path(review['run_dir']) / '01_python/inputs/analyze.py').write_text("print('changed')")
    with pytest.raises(ValueError, match='changed'):
        execution.authorize_step(pid, wid, 'program', review['digest'])


def test_verified_python_dependency_preserves_unverified_scope_and_detects_tampering(project, tmp_path, monkeypatch):
    from vasp_slurm_agent import script_workflow, project_artifacts
    from test_script_workflow import Transport
    store, _, execution, pid, _, _ = project
    wid, environment = script_graph(project)
    graph = store.get_workflow(pid, wid)
    first = {key: graph['steps'][0][key] for key in ('id', 'name', 'engine', 'spec', 'depends_on')}
    second = {**first, 'id': 'after', 'depends_on': ['program']}
    graph = store.create_workflow(pid, graph['session_id'], 'Two programs', 'Verify process dependency', [first, second])
    wid = graph['id']
    execution.draft_step(pid, wid, 'program', SETTINGS)
    prepared = execution.prepare_step(pid, wid, 'program')
    root = Path(store.get_run(pid, prepared['run_id'])['run_dir'])
    transport = Transport(tmp_path / 'remote')
    script_workflow.advance(root, transport)
    assert transport.execute().returncode == 0
    assert script_workflow.advance(root, transport)['status'] == 'completed_unverified'
    report = project_artifacts.inspect_run(store, pid, prepared['run_id'])
    assert report['process_success'] and not report['accepted']
    following = execution.draft_step(pid, wid, 'after', SETTINGS)
    evidence = following['source_context']['dependencies']['program']
    assert evidence['process_success'] and not evidence['accepted'] and evidence['limits']
    (root / '01_python/outputs/result.json').write_text('{"value": 46}')
    with pytest.raises(ValueError, match='preceding step'):
        execution.prepare_step(pid, wid, 'after')
    assert len(store.list_runs(pid)) == 1


def test_generic_file_is_not_automatically_a_native_structure(project, tmp_path):
    from vasp_slurm_agent import script_workflow
    from test_script_workflow import Transport
    store, _, execution, pid, native_wid, _ = project
    wid, _ = script_graph(project)
    graph = store.get_workflow(pid, wid)
    first = {key: graph['steps'][0][key] for key in ('id', 'name', 'engine', 'spec', 'depends_on')}
    native = store.get_workflow(pid, native_wid)['steps'][0]
    second = {key: native[key] for key in ('id', 'name', 'engine', 'spec', 'depends_on')}
    second['depends_on'] = ['program']
    second['spec']['source'] = {'kind': 'step', 'step_id': 'program', 'stage': None}
    graph = store.create_workflow(pid, graph['session_id'], 'Invalid geometry transfer', 'No inference of geometry', [first, second])
    wid = graph['id']
    execution.draft_step(pid, wid, 'program', SETTINGS)
    prepared = execution.prepare_step(pid, wid, 'program')
    root = Path(store.get_run(pid, prepared['run_id'])['run_dir'])
    transport = Transport(tmp_path / 'remote')
    script_workflow.advance(root, transport)
    assert transport.execute().returncode == 0
    script_workflow.advance(root, transport)
    with pytest.raises(ValueError, match='VASP or QE|structure'):
        execution.draft_step(pid, wid, 'scf', SETTINGS)
    assert len(store.list_runs(pid)) == 1


@pytest.mark.parametrize('bad_input', ['files/private.UPF', '../outside.py', 'config/auth.json'])
def test_generic_inputs_reject_private_and_external_files(project, bad_input):
    store, _, execution, pid, _, _ = project
    wid, _ = script_graph(project)
    with store._transaction() as db:
        spec = json.loads(db.execute('SELECT spec FROM steps WHERE workflow_id=?', (wid,)).fetchone()[0])
        spec['inputs'] = [bad_input]
        db.execute('UPDATE steps SET spec=? WHERE workflow_id=?', (json.dumps(spec), wid))
    with pytest.raises(ValueError):
        execution.draft_step(pid, wid, 'program', SETTINGS)
    assert store.list_runs(pid) == []


def test_three_projects_remain_isolated_after_restart_and_native_transfer(tmp_path, monkeypatch, parent):
    """Mocked planning and accepted VASP parser fixture; real stores and input writers."""
    from vasp_slurm_agent.project_context import build_project_context
    store = Workspace(tmp_path / 'three-projects')
    registry = ConnectionRegistry(store)
    hpc = registry.create('hpc', 'Fixture HPC', {'host': 'test.invalid', 'user': 'researcher',
        'scheduler': 'slurm', 'remote_base': '/scratch/test'})
    vasp_env = registry.create_environment(hpc['id'], 'VASP', 'vasp',
        {'vasp_command': 'srun vasp_std', 'potcar_root': '/licensed', 'partition': 'cpu'})
    qe_env = registry.create_environment(hpc['id'], 'QE', 'qe',
        {'qe_pseudo_root': '/pseudo', 'qe_pseudopotentials': {'Si': 'Si.UPF'}, 'partition': 'cpu'})
    projects = []
    for name, engines in [('VASP only', ['vasp']), ('QE only', ['qe']), ('Mixed', ['vasp', 'qe'])]:
        pid = store.create_project(name)['id']
        session = store.create_session(pid, name + ' conversation')
        store.append_message(pid, session['id'], 'user', name + ' private goal')
        registry.bind(pid, hpc['id'])
        write_file(store, pid, 'files/Si.cif', files('vasp_slurm_agent').joinpath('examples/Si.cif').read_bytes())
        graph_steps = []
        for index, engine in enumerate(engines):
            source = {'kind': 'file', 'path': 'files/Si.cif'} if index == 0 else {'kind': 'step', 'step_id': 'first', 'stage': None}
            graph_steps.append({'id': 'first' if index == 0 else 'second', 'name': engine.upper(), 'engine': engine,
                'spec': {'goal': name + ' SCF', 'source': source, 'environment_id': (vasp_env if engine == 'vasp' else qe_env)['id']},
                'depends_on': [] if index == 0 else ['first']})
        graph = store.create_workflow(pid, session['id'], name, name + ' SCF', graph_steps)
        execution = ProjectExecution(store, registry)
        if name == 'Mixed':
            root = store.project_paths(pid)['runs'] / 'accepted-source'
            shutil.copytree(parent[0], root)
            registered = store.register_run(pid, root, graph['id'], 'first')
            attempt = execution._new(pid, graph['id'], graph['steps'][0])
            execution._save(attempt, run_id=registered['id'], state='succeeded')
            index = 1
        else:
            index = 0
        step = graph['steps'][index]
        provider(monkeypatch, [('dft_conversation', route()), ('qe_task' if step['engine'] == 'qe' else 'dft_task', plan(step['engine'], 1))])
        drafted = execution.draft_step(pid, graph['id'], step['id'], SETTINGS)
        assert drafted['state'] == 'ready', drafted
        if name == 'Mixed':
            assert drafted['source_context']['kind'] == 'transfer'
            assert drafted['source_context']['receipt']['source_engine'] == 'vasp'
            assert drafted['source_context']['receipt']['target_engine'] == 'qe'
            assert drafted['source_context']['receipt']['electronic_state_reused'] is False
            history = conversation_store.load(store.resolve_path(pid, drafted['conversation_path']))['messages']
            assert any(item.get('action') == 'dependency_evidence' for item in history)
            transfer_text = next(item['content'] for item in history if item.get('action') == 'structure_transfer')
            assert 'already been copied and supplied' in transfer_text and 'requires_fresh_scf' in transfer_text
            assert 'fractional_positions' not in transfer_text and '/scratch/' not in transfer_text
            folder = store.resolve_path(pid, drafted['conversation_path'])
            saved = conversation_store.load(folder)
            saved['messages'] = [item for item in saved['messages'] if item.get('action') != 'structure_transfer']
            conversation_store.save(folder, saved, expected_revision=saved['revision'])
            provider(monkeypatch, [('dft_conversation', route()), ('qe_task', plan('qe', 1))])
            execution.draft_step(pid, graph['id'], step['id'], SETTINGS, message='Use the accepted structure already supplied.')
            assert any(item.get('action') == 'structure_transfer' for item in conversation_store.load(folder)['messages'])
        ready = execution.prepare_step(pid, graph['id'], step['id'])
        projects.append((pid, session['id'], graph['id'], ready['run_id'], step['engine']))
    restored = Workspace(store.home)
    for pid, sid, wid, run_id, engine in projects:
        own = restored.get_project(pid)['name']
        assert restored.get_session(pid, sid)['messages'][0]['content'] == own + ' private goal'
        run = restored.get_run(pid, run_id)
        assert run['engine'] == engine and run['state']['status'] == 'planned'
        context = build_project_context(restored, pid, session_id=sid)
        assert len(context['workflows']) == 1 and context['workflows'][0]['id'] == wid
        assert all(not stage['job_id'] for stage in run['state']['stages'])
        for foreign, foreign_session, foreign_workflow, foreign_run, _ in projects:
            if foreign == pid:
                continue
            assert restored.get_project(foreign)['name'] + ' private goal' not in json.dumps(context)
            for method, identifier in [(restored.get_session, foreign_session), (restored.get_workflow, foreign_workflow), (restored.get_run, foreign_run)]:
                with pytest.raises(ValueError):
                    method(pid, identifier)
