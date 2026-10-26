"""Project instructions reach generated inputs through a local API simulator.

The OpenAI client, HTTP transport, project store and engine planners run normally;
the localhost server is a deterministic fixture, not a live language model.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import re
from threading import Thread

import pytest

from vasp_slurm_agent import conversation_store, project_agent, qe_workflow, workflow
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_execution import ProjectExecution
from vasp_slurm_agent.project_files import write_file
from vasp_slurm_agent.workspace import Workspace
from test_conversation import route
from test_project_agent import reply
from test_qe_agent import response as qe_response
from test_task_agent import response as vasp_response, stage


@contextmanager
def local_provider(respond):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            name = request['response_format']['json_schema']['name']
            payload = json.loads(request['messages'][-1]['content'])
            calls.append((name, payload))
            answer = respond(name, payload)
            result = {'id': 'audit', 'object': 'chat.completion', 'created': 0,
                'model': 'audit-simulator', 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': json.dumps(answer)},
                    'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            content = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def setup_project(tmp_path, monkeypatch, engine, base_url):
    def blocked(*args, **kwargs):
        pytest.fail('Guided planning and preparation must not submit jobs')

    monkeypatch.setattr(workflow, 'SSHTransport', blocked)
    monkeypatch.setattr(qe_workflow, 'SSHTransport', blocked)
    store = Workspace(tmp_path / 'workspace')
    registry = ConnectionRegistry(store)
    pid = store.create_project('Intent integration')['id']
    sid = store.create_session(pid)['id']
    model = registry.create('model', 'Local API fixture',
        {'provider': 'local', 'model': 'audit-simulator', 'base_url': base_url})['id']
    registry.bind(pid, model)
    cluster = registry.create('hpc', 'Fixture', {'host': 'test.invalid', 'user': 'researcher',
        'scheduler': 'slurm', 'remote_base': '/scratch/test'})['id']
    registry.bind(pid, cluster)
    config = ({'vasp_command': 'srun vasp_std', 'potcar_root': '/licensed', 'partition': 'cpu'}
        if engine == 'vasp' else {'qe_pseudo_root': '/pseudo', 'qe_pseudopotentials': {'Si': 'Si.UPF'},
                                'partition': 'cpu'})
    environment = registry.create_environment(cluster, 'Fixture', engine, config)['id']
    write_file(store, pid, 'files/Si.cif', files('vasp_slurm_agent').joinpath('examples/Si.cif').read_bytes())
    return store, registry, pid, sid, model, environment


def graph(engine, environment, goal):
    return {'title': 'Si SCF', 'goal': goal, 'replace_selected': False, 'steps': [
        {'id': 'scf', 'name': 'SCF', 'engine': engine, 'goal': goal,
         'source': {'kind': 'file', 'path': 'files/Si.cif', 'step_id': None, 'stage': None},
         'environment_id': environment, 'depends_on': [], 'inputs': [], 'input_artifacts': [],
         'arguments': [], 'outputs': []}]}


def cutoff_reply(name, payload):
    # The simulator can retain only requirements actually present in its input.
    user_text = '\n'.join(item['content'] for item in payload.get('history', [])
                          if item['role'] == 'user')
    values = re.findall(r'cutoff (\d+)', user_text + '\n' + payload.get('goal', ''))
    cutoff = int(values[-1]) if values else 400
    if name == 'qe_task':
        return qe_response(('scf',), {'ecutwfc': cutoff})
    return vasp_response([stage('scf', encut=cutoff)])


def planned_cutoff(record, engine):
    item = record['pending']['plan']['stages'][0]
    return item['parameters']['encut'] if engine == 'vasp' else json.loads(item['parameters_json'])['ecutwfc']


@pytest.mark.parametrize('engine', ['vasp', 'qe'])
def test_saved_step_goal_reaches_inputs_and_later_revision_wins(tmp_path, monkeypatch, engine):
    environment = None

    def respond(name, payload):
        if name == 'dft_project_action':
            message = payload['context']['question']
            if message.startswith('Create'):
                return reply('workflow', workflow=graph(engine, environment, 'Run SCF with cutoff 600.'))
            return reply('draft_step')
        if name == 'dft_conversation':
            return route()
        return cutoff_reply(name, payload)

    with local_provider(respond) as (url, calls):
        store, registry, pid, sid, model, environment = setup_project(tmp_path, monkeypatch, engine, url)
        created = project_agent.send(store, pid, sid, 'Create an SCF workflow with cutoff 600.', model)
        assert created['action'] == 'workflow', created
        wid = created['workflow_id']
        # Reopen storage: requirements must survive without relying on UI state.
        store = Workspace(store.home)
        draft = project_agent.send(store, pid, sid, 'Prepare this step.', model, workflow_id=wid, step_id='scf')
        assert draft['result']['state'] == 'ready', draft
        folder = store.resolve_path(pid, draft['result']['conversation_path'])
        record = conversation_store.load(folder)
        assert planned_cutoff(record, engine) == 600
        revised = project_agent.send(store, pid, sid, 'Use cutoff 650 instead.', model, workflow_id=wid, step_id='scf')
        assert revised['result']['state'] == 'ready', revised
        record = conversation_store.load(folder)
        assert planned_cutoff(record, engine) == 650
        execution = ProjectExecution(store, registry)
        result = execution.prepare_step(pid, wid, 'scf')
        root = store.resolve_path(pid, store.get_run(pid, result['run_id'])['relative_path'])
        inputs = root / '01_scf/inputs'
        if engine == 'vasp':
            from pymatgen.io.vasp import Incar
            assert Incar.from_file(inputs / 'INCAR')['ENCUT'] == 650
        else:
            assert re.search(r'ecutwfc\s*=\s*650(?:\.0)?', (inputs / 'pw.in').read_text())
        state = json.loads((root / 'run.json').read_text())
        assert not any(item['job_id'] for item in state['stages'])
        assert [name for name, _ in calls].count('dft_project_action') == 3


def test_early_requirement_survives_more_than_forty_messages(tmp_path, monkeypatch):
    environment = None

    def respond(name, payload):
        if name == 'dft_project_action':
            if payload['context']['question'].startswith('Create'):
                history = payload['context']['session']['history']
                goal = next((m['content'] for m in history if 'cutoff 700' in m['content']),
                            'Run SCF with cutoff 400.')
                return reply('workflow', workflow=graph('vasp', environment, goal))
            return reply('draft_step')
        return route() if name == 'dft_conversation' else cutoff_reply(name, payload)

    with local_provider(respond) as (url, calls):
        store, registry, pid, sid, model, environment = setup_project(tmp_path, monkeypatch, 'vasp', url)
        requirement = 'Run SCF with cutoff 700. Keep SOC off.'
        store.append_message(pid, sid, 'user', requirement)
        for index in range(21):
            store.append_message(pid, sid, 'assistant', f'Planning discussion {index}.')
            store.append_message(pid, sid, 'user', 'Keep the original settings; continue the discussion.')
        planned = project_agent.send(store, pid, sid, 'Create the workflow from my requirements.', model)
        assert planned['action'] == 'workflow', planned
        assert planned['result']['steps'][0]['spec']['goal'] == requirement
        context = calls[0][1]['context']
        assert len(context['session']['history']) > 40 and context['omitted']['messages'] == 0
        assert context['session']['history'][0]['content'] == requirement
        wid = planned['workflow_id']
        drafted = project_agent.send(store, pid, sid, 'Prepare this step.', model, workflow_id=wid, step_id='scf')
        assert drafted['result']['state'] == 'ready', drafted
        prepared = ProjectExecution(store, registry).prepare_step(pid, wid, 'scf')
        root = store.resolve_path(pid, store.get_run(pid, prepared['run_id'])['relative_path'])
        from pymatgen.io.vasp import Incar
        assert Incar.from_file(root / '01_scf/inputs/INCAR')['ENCUT'] == 700


@pytest.mark.parametrize('already_over_budget', [False, True])
def test_context_budget_blocks_call_without_changing_history_or_inputs(tmp_path, monkeypatch, already_over_budget):
    environment = None

    def respond(name, payload):
        if name == 'dft_project_action':
            if payload['context']['question'].startswith('Create'):
                return reply('workflow', workflow=graph('vasp', environment, 'Run SCF with cutoff 600.'))
            return reply('draft_step')
        return route() if name == 'dft_conversation' else cutoff_reply(name, payload)

    with local_provider(respond) as (url, calls):
        store, registry, pid, sid, model, environment = setup_project(tmp_path, monkeypatch, 'vasp', url)
        created = project_agent.send(store, pid, sid, 'Create the workflow.', model)
        wid = created['workflow_id']
        draft = project_agent.send(store, pid, sid, 'Prepare this step.', model, workflow_id=wid, step_id='scf')
        prepared = ProjectExecution(store, registry).prepare_step(pid, wid, 'scf')
        root = store.resolve_path(pid, store.get_run(pid, prepared['run_id'])['relative_path'])
        before_files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        message = 'Use cutoff 650 instead. ' + 'Keep the previous constraints. ' * 40
        padding = 'Audit context padding: '
        store.append_message(pid, sid, 'assistant', padding)
        context = project_agent._context(store, registry, pid, sid, message, wid, 'scf', ())
        size = len(json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True))
        extra = 80000 - size + (100 if already_over_budget else -500)
        with store._transaction() as db:
            db.execute('UPDATE messages SET content=? WHERE project_id=? AND session_id=? AND content=?',
                       (padding + 'x' * extra, pid, sid, padding))
        before_session = store.get_session(pid, sid)
        count = len(calls)
        if not already_over_budget:
            # The current context fits; accepting the pending user turn would not.
            project_agent._context(store, registry, pid, sid, message, wid, 'scf', ())
        with pytest.raises(ValueError, match='too large'):
            project_agent.send(store, pid, sid, message, model, workflow_id=wid, step_id='scf')
        assert len(calls) == count
        assert store.get_session(pid, sid) == before_session
        assert {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before_files
        assert len(store.list_runs(pid)) == 1
