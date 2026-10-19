"""Real local program/output transfers with mocked scheduler transport, no SSH."""

from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path
import shlex
import shutil
import sys

import pytest

from vasp_slurm_agent import script_workflow, workflow, project_input_artifacts
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_execution import ProjectExecution
from vasp_slurm_agent.project_files import write_file
from vasp_slurm_agent.workspace import Workspace
from test_script_workflow import Transport
from test_continuation import parent
from test_explanation import accepted_run


@pytest.fixture
def chain(tmp_path):
    store = Workspace(tmp_path / 'workspace')
    pid = store.create_project('Program chain')['id']
    sid = store.create_session(pid)['id']
    registry = ConnectionRegistry(store)
    connection = registry.create('hpc', 'Fixture', {'host': 'test.invalid', 'user': 'researcher',
        'scheduler': 'slurm', 'remote_base': '/scratch/project'})
    registry.bind(pid, connection['id'])
    env = registry.create_environment(connection['id'], 'Python fixture', 'python',
        {'command': shlex.quote(sys.executable), 'partition': 'cpu', 'tasks': 1})
    write_file(store, pid, 'files/source.py', "from pathlib import Path\n"
        "Path('data.json').write_text('{\"value\":45}')\n"
        "Path('matrix.txt').write_text('reviewed matrix fixture')\n")
    write_file(store, pid, 'files/consume.py', "from pathlib import Path\nimport json\n"
        "data=json.loads(Path('source.json').read_text())\n"
        "Path('doubled.json').write_text(json.dumps({'value':2*data['value']}))\n")
    refs = [{'step_id': 'source', 'path': 'data.json', 'name': 'source.json'}]
    steps = [
        {'id': 'source', 'name': 'Source', 'engine': 'python', 'depends_on': [], 'spec': {
            'goal': 'Generate the reviewed data', 'source': {'kind': 'file', 'path': 'files/source.py'},
            'environment_id': env['id'], 'outputs': [{'path': 'data.json', 'kind': 'json', 'required': True},
                {'path': 'matrix.txt', 'kind': 'text', 'required': True}]}},
        {'id': 'consume', 'name': 'Consume', 'engine': 'python', 'depends_on': ['source'], 'spec': {
            'goal': 'Read the source data', 'source': {'kind': 'file', 'path': 'files/consume.py'},
            'environment_id': env['id'], 'input_artifacts': refs,
            'outputs': [{'path': 'doubled.json', 'kind': 'json', 'required': True}]}},
    ]
    graph = store.create_workflow(pid, sid, 'Transfer', 'Explicit reviewed data flow', steps)
    execution = ProjectExecution(store, registry)
    execution.draft_step(pid, graph['id'], 'source', None)
    attempt = execution.prepare_step(pid, graph['id'], 'source')
    root = Path(store.get_run(pid, attempt['run_id'])['run_dir'])
    transport = Transport(tmp_path / 'source-remote')
    script_workflow.advance(root, transport)
    assert transport.execute().returncode == 0
    assert script_workflow.advance(root, transport)['status'] == 'completed_unverified'
    return store, registry, execution, pid, graph['id'], attempt, root


def test_python_to_python_copies_verified_bytes_and_provenance_after_restart(chain, tmp_path):
    store, registry, execution, pid, wid, source, source_root = chain
    before = {path.relative_to(source_root).as_posix(): path.read_bytes() for path in source_root.rglob('*') if path.is_file()}
    drafted = execution.draft_step(pid, wid, 'consume', None)
    proof = drafted['proposal']['input_artifacts'][0]
    assert proof['path'] == 'data.json' and proof['resolved_path'] == '01_python/outputs/data.json'
    assert proof['kind'] == 'json' and proof['source_process_success'] and not proof['source_accepted']
    assert proof['limits'] and proof['source_run_id'] == source['run_id']
    store = Workspace(store.home)
    execution = ProjectExecution(store)
    prepared = execution.prepare_step(pid, wid, 'consume')
    root = Path(store.get_run(pid, prepared['run_id'])['run_dir'])
    assert (root / '01_python/inputs/source.json').read_bytes() == (source_root / '01_python/outputs/data.json').read_bytes()
    provenance = json.loads((root / '01_python/inputs/input_provenance.json').read_text())
    assert provenance['inputs'] == [proof]
    assert 'test.invalid' not in json.dumps(provenance) and str(source_root) not in json.dumps(provenance)
    review = execution.review(pid, wid, 'consume')
    assert '01_python/inputs/input_provenance.json' in review['files']
    execution.authorize_step(pid, wid, 'consume', review['digest'])
    # Execute the actual consumer locally behind the explicit fake scheduler.
    transport = Transport(tmp_path / 'consumer-remote')
    script_workflow.advance(root, transport)
    assert transport.execute().returncode == 0
    assert script_workflow.advance(root, transport)['status'] == 'completed_unverified'
    assert json.loads((root / '01_python/outputs/doubled.json').read_text()) == {'value': 90}
    assert not script_workflow.load_run_context(root)['facts']['run.accepted']['value']
    assert before == {path.relative_to(source_root).as_posix(): path.read_bytes() for path in source_root.rglob('*') if path.is_file()}


def test_python_output_can_be_reviewed_wannier_input_without_scientific_promotion(chain):
    store, registry, execution, pid, wid, _, source_root = chain
    source = store.get_workflow(pid, wid)['steps'][0]
    connection = registry.environment(source['spec']['environment_id'])['connection_id']
    env = registry.create_environment(connection, 'Wannier fixture', 'wannier90',
        {'command': 'srun wannier90.x', 'partition': 'cpu', 'tasks': 1})
    write_file(store, pid, 'files/model.win', 'num_wann=2\nwrite_hr=true\n')
    with store._transaction() as db:
        spec = {'goal': 'Use the reviewed matrix', 'source': {'kind': 'file', 'path': 'files/model.win'},
            'environment_id': env['id'], 'input_artifacts': [
                {'step_id': 'source', 'path': '01_python/outputs/matrix.txt', 'name': 'model.mmn'}]}
        db.execute("UPDATE steps SET engine='wannier90',spec=? WHERE workflow_id=? AND id='consume'", (json.dumps(spec), wid))
    drafted = execution.draft_step(pid, wid, 'consume', None)
    assert not drafted['proposal']['input_artifacts'][0]['source_accepted']
    prepared = execution.prepare_step(pid, wid, 'consume')
    root = Path(store.get_run(pid, prepared['run_id'])['run_dir'])
    assert (root / '01_wannier90/inputs/model.mmn').read_bytes() == (source_root / '01_python/outputs/matrix.txt').read_bytes()
    assert workflow.read_state(root)['status'] == 'planned'
    assert not workflow.read_state(root)['stages'][0]['job_id']


@pytest.mark.parametrize('gate', ['prepare', 'authorize', 'launch'])
def test_changed_source_blocks_each_execution_gate(chain, monkeypatch, gate):
    store, _, execution, pid, wid, _, source_root = chain
    execution.draft_step(pid, wid, 'consume', None)
    review = None
    if gate != 'prepare':
        execution.prepare_step(pid, wid, 'consume')
        review = execution.review(pid, wid, 'consume')
    if gate == 'launch':
        execution.authorize_step(pid, wid, 'consume', review['digest'])
    (source_root / '01_python/outputs/data.json').write_text('{"value":46}')
    monkeypatch.setattr(script_workflow, 'start_worker', lambda *a: pytest.fail('Changed source launched'))
    with pytest.raises(ValueError, match='preceding step|evidence changed'):
        if gate == 'prepare':
            execution.prepare_step(pid, wid, 'consume')
        elif gate == 'authorize':
            execution.authorize_step(pid, wid, 'consume', review['digest'])
        else:
            execution.launch_step(pid, wid, 'consume')


@pytest.mark.parametrize('name', ['source.json', 'input_provenance.json'])
def test_frozen_input_or_provenance_change_blocks_approval(chain, name):
    store, _, execution, pid, wid, _, _ = chain
    execution.draft_step(pid, wid, 'consume', None)
    execution.prepare_step(pid, wid, 'consume')
    review = execution.review(pid, wid, 'consume')
    target = Path(review['run_dir']) / '01_python/inputs' / name
    target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        execution.authorize_step(pid, wid, 'consume', review['digest'])


def test_cross_project_reference_rejected(chain):
    store, _, _, pid, _, source, _ = chain
    foreign = store.create_project('Foreign')['id']
    reference = {'step_id': 'source', 'path': '01_python/outputs/data.json', 'name': 'data.json'}
    with pytest.raises(ValueError, match='not found'):
        project_input_artifacts.resolve_input(store, foreign, source['run_id'], reference)
    assert store.list_runs(foreign) == []


@pytest.mark.parametrize('change', ['undeclared', 'receipt', 'private', 'traversal', 'duplicate_name', 'undeclared_dependency'])
def test_invalid_output_mapping_never_prepares_a_consumer(chain, change):
    store, _, execution, pid, wid, _, _ = chain
    with store._transaction() as db:
        spec = json.loads(db.execute("SELECT spec FROM steps WHERE workflow_id=? AND id='consume'", (wid,)).fetchone()[0])
        reference = spec['input_artifacts'][0]
        if change == 'undeclared':
            reference['path'] = '01_python/outputs/not-declared.json'
        elif change == 'receipt':
            reference['path'] = '01_python/outputs/stdout.log'
        elif change == 'private':
            reference['name'] = 'POTCAR'
        elif change == 'traversal':
            reference['path'] = '../outputs/data.json'
        elif change == 'duplicate_name':
            reference['name'] = 'consume.py'
        else:
            reference['step_id'] = 'not-a-dependency'
        db.execute("UPDATE steps SET spec=? WHERE workflow_id=? AND id='consume'", (json.dumps(spec), wid))
    with pytest.raises(ValueError):
        execution.draft_step(pid, wid, 'consume', None)
    assert len(store.list_runs(pid)) == 1


def test_native_artifact_requires_fresh_acceptance_and_stays_native(chain, parent):
    store, _, _, pid, _, _, _ = chain
    native = store.project_paths(pid)['runs'] / 'native-source'
    shutil.copytree(parent[0], native)
    record = store.register_run(pid, native)
    ref = {'step_id': 'native', 'path': '01_scf/outputs/final_structure.vasp', 'name': 'geometry.vasp'}
    path, proof = project_input_artifacts.resolve_input(store, pid, record['id'], ref)
    assert path.read_bytes() == (parent[1] / 'final_structure.vasp').read_bytes()
    assert proof['source_accepted'] and proof['source_engine'] == 'vasp'
    (native / '01_scf/outputs/OUTCAR').write_text('changed')
    with pytest.raises(ValueError, match='verified completed output'):
        project_input_artifacts.resolve_input(store, pid, record['id'], ref)


def test_context_lists_only_fresh_declared_output_metadata(chain):
    from vasp_slurm_agent import project_agent
    store, registry, _, pid, wid, source, root = chain
    graph = store.get_workflow(pid, wid)
    context = project_agent._context(store, registry, pid, graph['session_id'], 'Connect the output', wid, 'consume', ())
    listed = context['selected_steps'][0]['available_artifacts']
    assert {item['path'] for item in listed['artifacts']} == {'data.json', 'matrix.txt'}
    assert listed['process_success'] and not listed['accepted'] and listed['limits']
    assert '"value":45' not in json.dumps(context)
    assert 'test.invalid' not in json.dumps(context)
    assert str(root) not in json.dumps(context)
    (root / '01_python/outputs/data.json').write_text('{"value":46}')
    refreshed = project_agent._context(store, registry, pid, graph['session_id'], 'Connect the output', wid, 'consume', ())
    assert refreshed['selected_steps'][0]['available_artifacts'] == {'artifacts': [], 'status': 'unavailable'}


def test_symlink_output_is_not_a_transfer_source(chain, tmp_path):
    store, _, _, pid, _, source, root = chain
    output = root / '01_python/outputs/data.json'
    outside = tmp_path / 'outside.json'
    outside.write_bytes(output.read_bytes())
    output.unlink()
    output.symlink_to(outside)
    with pytest.raises(ValueError, match='symbolic'):
        project_input_artifacts.resolve_input(store, pid, source['run_id'],
            {'step_id': 'source', 'path': '01_python/outputs/data.json', 'name': 'input.json'})
