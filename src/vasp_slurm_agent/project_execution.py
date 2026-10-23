"""Project-bound workflow tools over the existing scientific executors.

Planning and preparation are local. A reviewed input digest authorizes exactly
one run; an interrupted launch keeps the same identity and submission receipts.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid
import tempfile

from . import conversation, conversation_store, workflow
from .connections import ConnectionRegistry
from .secret_guard import redact, reject_ordinary_text
from .workspace import _id, _now, _object


_BUSY = {'planning', 'preparing', 'starting', 'reconnecting', 'cancelling'}
_FAILED = {'failed', 'needs_attention', 'unconverged', 'missing_outputs', 'suspect', 'unavailable'}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProjectExecution:
    def __init__(self, workspace, registry=None):
        self.workspace = workspace
        self.registry = registry or ConnectionRegistry(workspace)

    @contextmanager
    def _lock(self, project_id, workflow_id):
        root = self.workspace.project_paths(project_id)['config']
        path = root / ('workflow-' + _id(workflow_id) + '.lock')
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        with os.fdopen(descriptor, 'r+b') as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'\0'); stream.flush()
            stream.seek(0)
            try:
                workflow._acquire(stream, blocking=False)
            except BlockingIOError:
                raise ValueError('This workflow is busy. Wait for its current operation.') from None
            try:
                yield
            finally:
                workflow._release(stream)

    def _step(self, project_id, workflow_id, step_id):
        item = self.workspace.get_workflow(project_id, workflow_id)
        if item['status'] == 'superseded':
            raise ValueError('Open the revised workflow.')
        step = next((step for step in item['steps'] if step['id'] == step_id), None)
        if step is None:
            raise ValueError('Step not found in this project workflow.')
        return item, step

    def attempts(self, project_id, workflow_id, step_id=None):
        self.workspace.get_workflow(project_id, workflow_id)
        with self.workspace._transaction(write=False) as db:
            rows = db.execute('SELECT * FROM step_attempts WHERE project_id=? AND workflow_id=? '
                              'AND (? IS NULL OR step_id=?) ORDER BY created_at,attempt',
                              (project_id, workflow_id, step_id, step_id)).fetchall()
            return [self._record(row) for row in rows]

    def _record(self, row):
        item = dict(row)
        item['source_context'] = json.loads(item['source_context'])
        if item['source_context'].get('kind') == 'script':
            item['source_context'].setdefault('input_artifacts', [])
        item['proposal'], item['reply'] = None, ''
        if item['conversation_path']:
            try:
                path = self.workspace.resolve_path(item['project_id'], item['conversation_path'], must_exist=True)
                record = conversation_store.load(path)
                item['proposal'] = record.get('pending')
                item['reply'] = next((m['content'] for m in reversed(record['messages']) if m['role'] == 'assistant'), '')
            except (ValueError, OSError):
                item['conversation_error'] = 'The saved step conversation is unavailable. Review or recover this step.'
        elif item['source_context'].get('kind') == 'script':
            item['proposal'] = {'status': 'ready', **item['source_context']}
            item['reply'] = 'Review the input files and expected outputs before preparing this step.'
        item['tasks'] = list(((item['proposal'] or {}).get('plan') or {}).get('tasks') or [])
        if item['run_id']:
            item['tasks'] = []
            try:
                state = self.workspace.get_run(item['project_id'], item['run_id'])['state'] or {}
                item['tasks'] = [stage['name'] for stage in state.get('stages', []) if stage.get('name')]
            except (ValueError, OSError):
                pass
        return item

    def latest(self, project_id, workflow_id, step_id):
        self._step(project_id, workflow_id, step_id)
        values = self.attempts(project_id, workflow_id, step_id)
        return values[-1] if values else None

    def _save(self, attempt, **changes):
        allowed = {'state', 'conversation_path', 'run_id', 'environment_id', 'source_context', 'approved_digest', 'last_error'}
        if set(changes) - allowed:
            raise ValueError('Invalid attempt update.')
        if 'source_context' in changes:
            changes['source_context'] = _object(changes['source_context'])
        if changes.get('last_error'):
            changes['last_error'] = redact(changes['last_error'])[:4000]
        changes['updated_at'] = _now()
        with self.workspace._transaction() as db:
            self.workspace._project_row(db, attempt['project_id'],
                                         writable=changes.get('state') in _BUSY | {'authorized'})
            db.execute('UPDATE step_attempts SET ' + ','.join(key + '=?' for key in changes) + ' WHERE id=? AND project_id=?',
                       (*changes.values(), attempt['id'], attempt['project_id']))
            if 'state' in changes:
                db.execute('UPDATE steps SET status=? WHERE project_id=? AND workflow_id=? AND id=?',
                           (changes['state'], attempt['project_id'], attempt['workflow_id'], attempt['step_id']))
            self.workspace._event(db, attempt['project_id'], 'step.updated',
                                  {'workflow_id': attempt['workflow_id'], 'step_id': attempt['step_id'],
                                   'attempt_id': attempt['id'], 'state': changes.get('state', attempt['state'])})
            row = db.execute('SELECT * FROM step_attempts WHERE id=?', (attempt['id'],)).fetchone()
        return self._record(row)

    def _new(self, project_id, workflow_id, step):
        now, identifier = _now(), str(uuid.uuid4())
        with self.workspace._transaction() as db:
            self.workspace._project_row(db, project_id, writable=True)
            number = db.execute('SELECT COALESCE(MAX(attempt),0)+1 FROM step_attempts '
                                'WHERE project_id=? AND workflow_id=? AND step_id=?',
                                (project_id, workflow_id, step['id'])).fetchone()[0]
            db.execute('INSERT INTO step_attempts(id,project_id,workflow_id,step_id,attempt,state,environment_id,created_at,updated_at) '
                       "VALUES (?,?,?,?,?,'planning',?,?,?)", (identifier, project_id, workflow_id, step['id'], number,
                                                              step['spec'].get('environment_id'), now, now))
            self.workspace._event(db, project_id, 'step.created', {'workflow_id': workflow_id, 'step_id': step['id'], 'attempt_id': identifier})
            return self._record(db.execute('SELECT * FROM step_attempts WHERE id=?', (identifier,)).fetchone())

    def _config(self, project_id, step, attempt=None, *, required=False):
        identifier = (attempt or {}).get('environment_id') or step['spec'].get('environment_id')
        if not identifier:
            if required:
                raise ValueError('Choose a software environment in Connections before preparing this step.')
            return None
        environment = self.registry.environment(identifier)
        if environment['engine'] != step['engine']:
            raise ValueError('The software environment does not match this step.')
        return self.registry.resolve_environment(project_id, identifier)['config']

    def configure_step(self, project_id, workflow_id, step_id, environment_id):
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            attempt = self.latest(project_id, workflow_id, step_id)
            if attempt and (attempt['run_id'] or attempt['state'] in _BUSY):
                raise ValueError('Choose the environment before preparing the run.')
            candidate = {'environment_id': environment_id}
            self._config(project_id, step, candidate, required=True)
            attempt = attempt or self._new(project_id, workflow_id, step)
            return self._save(attempt, environment_id=environment_id, state='draft', approved_digest=None)

    def _dependencies(self, project_id, workflow_id, step):
        records = {}
        for dependency in step['depends_on']:
            from .project_artifacts import inspect_run
            attempt = self.latest(project_id, workflow_id, dependency)
            if not attempt or not attempt['run_id']:
                raise ValueError('Complete the preceding workflow steps first.')
            report = inspect_run(self.workspace, project_id, attempt['run_id'])
            generic = report.get('engine') in {'python', 'wannier90'}
            if not (report.get('process_success') if generic else report.get('accepted')):
                raise ValueError('A preceding step has no accepted results or verified execution. Review its evidence first.')
            records[dependency] = {**attempt, 'evidence': {
                'run_id': attempt['run_id'], 'engine': report['engine'],
                'context_sha256': report['context_sha256'], 'accepted': report.get('accepted', False),
                'process_success': report.get('process_success', False), 'limits': report.get('limits', [])}}
        return records

    def _script_source(self, project_id, workflow_id, step):
        from . import project_files, script_workflow
        from .project_context import _private_file
        dependencies = self._dependencies(project_id, workflow_id, step)
        spec = step['spec']
        source = spec.get('source', {})
        if source.get('kind') != 'file':
            raise ValueError('Choose reviewed project files for this program. Structure transfer is only for VASP and QE.')
        relative = source.get('path')
        inputs = spec.get('inputs', [])
        if not isinstance(inputs, list) or any(not isinstance(value, str) for value in inputs):
            raise ValueError('List the project input files for this step.')
        paths, receipts = {}, []
        for name in dict.fromkeys([relative, *inputs]):
            if not isinstance(name, str) or _private_file(name):
                raise ValueError('Choose an input file without credentials or private potentials.')
            path = project_files._checked(self.workspace, project_id, name, draft=True, must_exist=True)
            if not path.is_file() or path.stat().st_size > project_files.MAX_UPLOAD_BYTES:
                raise ValueError('Choose regular input files smaller than 512 MiB.')
            script_workflow._name(path.name)
            if path.name in paths or path.name in script_workflow._INTERNAL | script_workflow._RECEIPTS:
                raise ValueError('Input filenames must be unique and not reserved for execution.')
            content = path.read_bytes()
            project_files._safe_content(content)
            try:
                reject_ordinary_text(content.decode('utf-8'))
            except UnicodeDecodeError:
                pass
            paths[path.name] = path
            receipts.append({'path': name, 'name': path.name, 'sha256': hashlib.sha256(content).hexdigest()})
        entrypoint = Path(relative).name
        if Path(entrypoint).suffix != ('.py' if step['engine'] == 'python' else '.win'):
            raise ValueError('Choose a .py or .win entrypoint for this program.')
        if step['engine'] == 'python':
            compile(paths[entrypoint].read_bytes(), entrypoint, 'exec')
        else:
            script_workflow._num_wann(paths[entrypoint].read_text())
        arguments = spec.get('arguments', [])
        if not isinstance(arguments, list) or any(not isinstance(value, str) or any(c in value for c in '\0\n\r') for value in arguments):
            raise ValueError('Arguments must be a list of literal strings.')
        reject_ordinary_text(json.dumps(arguments))
        from .project_input_artifacts import PROVENANCE_NAME, resolve_input, validate_reference
        references = spec.get('input_artifacts', [])
        if not isinstance(references, list) or len(references) > 100:
            raise ValueError('Choose up to 100 predecessor output files.')
        artifact_receipts = []
        for reference in references:
            validate_reference(reference, step['depends_on'])
            if reference['name'] in paths:
                raise ValueError('Each staged input needs a unique filename.')
            path, receipt = resolve_input(self.workspace, project_id, dependencies[reference['step_id']]['run_id'], reference)
            paths[reference['name']] = path
            artifact_receipts.append(receipt)
        if references and PROVENANCE_NAME in paths:
            raise ValueError('The input provenance filename is reserved.')
        names = set(paths) | ({PROVENANCE_NAME} if references else set())
        outputs = script_workflow._outputs(spec.get('outputs', []), step['engine'], entrypoint, names)
        validation = script_workflow._validation(step['engine'], spec.get('validation'))
        return paths, {'kind': 'script', 'entrypoint': entrypoint, 'files': receipts,
            'arguments': arguments, 'outputs': outputs, 'validation': validation, 'input_artifacts': artifact_receipts,
            'dependencies': {key: value['evidence'] for key, value in dependencies.items()},
            'limits': ['Execution and readable outputs do not establish scientific correctness.']}

    def _draft_script(self, project_id, workflow_id, step, settings, message):
        reject_ordinary_text(message or step['spec'].get('goal', ''))
        with self.workspace.edit_project(project_id):
            pass
        previous = self.latest(project_id, workflow_id, step['id'])
        if previous and previous['run_id']:
            raise ValueError('Prepared runs are immutable. Create a follow-up workflow with reviewed input files.')
        try:
            self._config(project_id, step, previous)
            _, source = self._script_source(project_id, workflow_id, step)
        except Exception as error:
            if previous:
                self._save(previous, state='needs_attention', approved_digest=None, last_error=str(error))
            raise
        if previous and previous['source_context'] and previous['source_context'] != source:
            self._save(previous, state='superseded', approved_digest=None)
            attempt = self._new(project_id, workflow_id, step)
            attempt = self._save(attempt, environment_id=previous['environment_id'])
        else:
            attempt = previous or self._new(project_id, workflow_id, step)
        return self._save(attempt, source_context=source, state='ready', approved_digest=None, last_error=None)

    def _source(self, project_id, workflow_id, step, attempt):
        dependencies = self._dependencies(project_id, workflow_id, step)
        source = step['spec'].get('source', {})
        if source.get('kind') == 'file':
            relative = source.get('path', '')
            if not isinstance(relative, str) or not relative.startswith('files/'):
                raise ValueError('Choose an input structure from this project.')
            from .project_files import read_file
            reject_ordinary_text(read_file(self.workspace, project_id, relative)['content'])
            path = self.workspace.resolve_path(project_id, relative, must_exist=True)
            return path, {'kind': 'file', 'path': relative, 'sha256': _digest(path),
                          'dependencies': {key: value['evidence'] for key, value in dependencies.items()}}
        if source.get('kind') == 'step' and source.get('step_id') in dependencies:
            from .project_artifacts import transfer_structure
            parent = dependencies[source['step_id']]
            relative = 'artifacts/transfers/' + attempt['id'] + '/POSCAR'
            receipt = transfer_structure(self.workspace, project_id, parent['run_id'], source.get('stage'), step['engine'], relative)
            return self.workspace.resolve_path(project_id, relative, must_exist=True), {
                'kind': 'transfer', 'receipt': receipt,
                'dependencies': {key: value['evidence'] for key, value in dependencies.items()}}
        raise ValueError('Select a project input or a declared dependency as the structure source.')

    def _check_source(self, project_id, workflow_id, step, attempt):
        dependencies = self._dependencies(project_id, workflow_id, step)
        source = attempt['source_context']
        if 'dependencies' in source and source['dependencies'] != {key: value['evidence'] for key, value in dependencies.items()}:
            raise ValueError('The preceding evidence changed. Review the draft again.')
        if source.get('kind') == 'script':
            _, current = self._script_source(project_id, workflow_id, step)
            if current != source:
                raise ValueError('The program inputs or preceding evidence changed. Review the draft again.')
        elif source.get('kind') == 'file':
            path = self.workspace.resolve_path(project_id, source['path'], must_exist=True)
            if _digest(path) != source['sha256']:
                raise ValueError('The input structure changed. Create a new plan before preparing.')
        elif source.get('kind') in {'retry', 'continuation'}:
            from .explanation import load_run_context
            record = self.workspace.get_run(project_id, source['run_id'])
            if load_run_context(record['run_dir'])['context_sha256'] != source['context_sha256']:
                raise ValueError('The source run changed. Review the follow-up again.')
        elif source:
            from .project_artifacts import verify_transfer
            verify_transfer(self.workspace, project_id, source['receipt'] if source.get('kind') == 'transfer' else source)

    def _source_history(self, project_id, workflow_id, step, attempt, folder):
        self._check_source(project_id, workflow_id, step, attempt)
        source = attempt['source_context']
        updates = []
        if source.get('dependencies'):
            updates.append(('dependency_evidence',
                'Verified dependency records (execution success alone is not scientific validation): '
                + json.dumps(source['dependencies'], sort_keys=True)))
        receipt = source.get('receipt') if source.get('kind') == 'transfer' else source
        if receipt.get('kind') == 'structure_transfer':
            compact = {key: receipt[key] for key in ('source_engine', 'source_stage', 'source_stage_folder',
                'source_structure_sha256', 'target_engine', 'format', 'units', 'electronic_state_reused',
                'magnetic_state_transferred', 'pseudopotentials_transferred', 'requires_fresh_scf')}
            compact['source_stage_accepted'] = True
            updates.append(('structure_transfer', 'An accepted native structure has already been copied and supplied '
                'as this conversation input. Plan from this structure; no active run is required. '
                + json.dumps(compact, sort_keys=True)))
        if not updates:
            return
        record = conversation_store.load(folder)
        changed = False
        for action, content in updates:
            prior = next((item for item in reversed(record['messages']) if item.get('action') == action), None)
            if prior is None or prior['content'] != content:
                record['messages'].append({'role': 'assistant', 'content': content, 'action': action})
                changed = True
        if changed:
            conversation_store.save(folder, record, expected_revision=record['revision'])

    def draft_step(self, project_id, workflow_id, step_id, settings, message=None):
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            if step['engine'] in {'python', 'wannier90'}:
                return self._draft_script(project_id, workflow_id, step, settings, message)
            if step['engine'] not in {'vasp', 'qe', 'abinit', 'cp2k', 'elk'}:
                raise ValueError('Choose a supported program for this step.')
            text = message or step['spec'].get('goal', '')
            reject_ordinary_text(text)
            with self.workspace.edit_project(project_id):
                pass
            attempt = self.latest(project_id, workflow_id, step_id)
            if attempt and attempt['run_id']:
                raise ValueError('Prepared runs are immutable. Start a follow-up or a repair attempt.')
            attempt = attempt or self._new(project_id, workflow_id, step)
            attempt = self._save(attempt, state='planning', approved_digest=None, last_error=None)
            try:
                config = self._config(project_id, step, attempt)
                previous_messages, previous_goal = [], ''
                if attempt['conversation_path']:
                    try:
                        self._check_source(project_id, workflow_id, step, attempt)
                        if attempt.get('conversation_error'):
                            raise ValueError('The saved step conversation is unavailable.')
                    except (ValueError, OSError):
                        if attempt['source_context'].get('kind') in {'retry', 'continuation'}:
                            raise ValueError('The source run changed. Request a fresh follow-up plan.') from None
                        # A new source needs a new frozen copy. Keep the old
                        # attempt and carry only the user's stated requirements.
                        if not attempt.get('conversation_error'):
                            old = conversation_store.load(self.workspace.resolve_path(project_id, attempt['conversation_path']))
                            previous_messages = [item for item in old['messages'] if item['role'] == 'user']
                            previous_goal = old.get('goal', '')
                        previous_environment = attempt['environment_id']
                        self._save(attempt, state='superseded', approved_digest=None)
                        attempt = self._new(project_id, workflow_id, step)
                        attempt = self._save(attempt, environment_id=previous_environment)
                if not attempt['conversation_path']:
                    source, receipt = self._source(project_id, workflow_id, step, attempt)
                    folder = conversation.start(self.workspace.project_paths(project_id)['runs'], step['engine'], source)
                    if previous_messages:
                        record = conversation_store.load(folder)
                        record.update(messages=previous_messages, goal=previous_goal)
                        conversation_store.save(folder, record, expected_revision=record['revision'])
                    relative = folder.relative_to(self.workspace.project_paths(project_id)['root']).as_posix()
                    attempt = self._save(attempt, conversation_path=relative, source_context=receipt)
                else:
                    self._check_source(project_id, workflow_id, step, attempt)
                    folder = self.workspace.resolve_path(project_id, attempt['conversation_path'], must_exist=True)
                self._source_history(project_id, workflow_id, step, attempt, folder)
                followup = {'retry': 'repair', 'continuation': 'continue'}.get(attempt['source_context'].get('kind'))
                result = conversation.send(folder, text, settings, config=config, requested_action=followup)
                pending = result.get('pending')
                state = 'ready' if pending and pending.get('status') == 'ready' else 'needs_input'
                return self._save(attempt, state=state)
            except Exception as error:
                self._save(attempt, state='needs_attention', last_error=str(error))
                raise

    def prepare_step(self, project_id, workflow_id, step_id):
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            attempt = self.latest(project_id, workflow_id, step_id)
            if not attempt:
                raise ValueError('Review a plan before preparing inputs.')
            if attempt['run_id']:
                return attempt
            config = self._config(project_id, step, attempt, required=True)
            self._check_source(project_id, workflow_id, step, attempt)
            if attempt['state'] != 'ready':
                raise ValueError('Answer the open questions and review the plan first.')
            attempt = self._save(attempt, state='preparing', approved_digest=None, last_error=None)
            run_dir = self.workspace.project_paths(project_id)['runs'] / attempt['id']
            try:
                with self.workspace.edit_project(project_id) as event:
                    if step['engine'] in {'python', 'wannier90'}:
                        from .script_workflow import prepare_run
                        inputs, source = self._script_source(project_id, workflow_id, step)
                        environment = self.registry.environment(attempt['environment_id'] or step['spec'].get('environment_id'))
                        with tempfile.TemporaryDirectory(prefix='dft-inputs-') as temporary:
                            if source['input_artifacts']:
                                from .project_input_artifacts import PROVENANCE_NAME
                                proof = Path(temporary) / PROVENANCE_NAME
                                proof.write_text(json.dumps({'schema_version': 1, 'inputs': source['input_artifacts']},
                                    sort_keys=True, indent=2, allow_nan=False) + '\n')
                                proof.chmod(0o600)
                                inputs[PROVENANCE_NAME] = proof
                            state = prepare_run(run_dir, config, engine=step['engine'], files=inputs,
                                entrypoint=source['entrypoint'], arguments=source['arguments'], outputs=source['outputs'],
                                command=environment['config']['command'], validation=source['validation'])
                        frozen = state['stages'][0]['metadata']['input_sha256']
                        if any(frozen[item['name']] != item['sha256'] for item in [*source['files'], *source['input_artifacts']]):
                            raise ValueError('The program inputs changed while preparing. Review the draft again.')
                        self._check_source(project_id, workflow_id, step, attempt)
                    else:
                        folder = self.workspace.resolve_path(project_id, attempt['conversation_path'], must_exist=True)
                        conversation.prepare(folder, run_dir, config)
                    event('step.prepared', {'workflow_id': workflow_id, 'step_id': step_id, 'attempt_id': attempt['id']})
                record = self.workspace.register_run(project_id, run_dir, workflow_id, step_id)
                return self._save(attempt, run_id=record['id'], state='prepared')
            except Exception as error:
                self._save(attempt, state='needs_attention', last_error=str(error))
                raise

    def recover_step(self, project_id, workflow_id, step_id):
        """Recover a local interrupted operation without submitting any job."""
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            attempt = self.latest(project_id, workflow_id, step_id)
            if not attempt:
                raise ValueError('This step has no operation to recover.')
            with self.workspace.edit_project(project_id):
                pass
            if attempt['run_id']:
                return attempt
            run_dir = self.workspace.project_paths(project_id)['runs'] / attempt['id']
            if (run_dir / 'run.json').is_file() or (run_dir / 'batch.json').is_file():
                if (run_dir / 'batch.json').is_file():
                    from .batch import _children
                    _children(run_dir, json.loads((run_dir / 'batch.json').read_text()))
                else:
                    state = workflow.read_state(run_dir)
                    if state.get('engine') == 'qe':
                        from .qe_workflow import _check_config
                    else:
                        from .workflow import _check_config
                    _check_config(run_dir, state)
                record = self.workspace.register_run(project_id, run_dir, workflow_id, step_id)
                state = record['state']
                stages = state.get('stages', [])
                if record['kind'] == 'batch':
                    stages = [stage for _, _, child in _children(run_dir, state) for stage in child['stages']]
                prepared = state['status'] == 'planned' and not any(stage.get('job_id') or stage.get('staged') for stage in stages)
                return self._save(attempt, run_id=record['id'], state='prepared' if prepared else state['status'],
                                  approved_digest=None, last_error=None)
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError('Partial preparation remains. Inspect it before choosing a new attempt.')
            pending = attempt['proposal']
            return self._save(attempt, state='ready' if pending and pending.get('status') == 'ready' else 'needs_input',
                              approved_digest=None, last_error=None)

    def retry_step(self, project_id, workflow_id, step_id, settings, message):
        """Draft a separate repair; never overwrite or relaunch the failed run."""
        return self._followup_step(project_id, workflow_id, step_id, settings, message, 'repair')

    def continue_step(self, project_id, workflow_id, step_id, settings, message):
        """Draft a new calculation from a selected accepted stage, preserving old runs."""
        return self._followup_step(project_id, workflow_id, step_id, settings, message, 'continue')

    def _followup_step(self, project_id, workflow_id, step_id, settings, message, action):
        reject_ordinary_text(message)
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            if step['engine'] not in {'vasp', 'qe', 'abinit', 'cp2k', 'elk'}:
                raise ValueError('Review and edit the program inputs, then create a follow-up workflow.')
            previous = self.latest(project_id, workflow_id, step_id)
            source_kind = 'retry' if action == 'repair' else 'continuation'
            pending_retry = previous if previous and not previous['run_id'] and previous['source_context'].get('kind') == source_kind else None
            original_run = previous['run_id'] if previous else None
            if pending_retry:
                original_run = pending_retry['source_context']['run_id']
            if not original_run:
                raise ValueError('Select a finished run before planning a follow-up.')
            run = self.workspace.get_run(project_id, original_run)
            if run['kind'] == 'batch':
                raise ValueError('Open the comparison member before planning its follow-up.')
            state = run['state']
            allowed = {'failed', 'needs_attention', 'cancelled'} if action == 'repair' else {'succeeded', 'failed', 'needs_attention'}
            if state is None or state['status'] not in allowed:
                raise ValueError('The run is not ready for this follow-up. Refresh its status first.')
            from .scheduler import TERMINAL_STATES
            for stage in state.get('stages', []):
                if (stage.get('job_id') or stage.get('staged')) and stage.get('scheduler_state') not in TERMINAL_STATES:
                    raise ValueError('The submission is uncertain or still active. Reconnect before retrying.')
            from .explanation import load_run_context
            evidence = load_run_context(run['run_dir'])
            if action == 'continue' and not any(key.endswith('.accepted') and value.get('value') is True
                                               for key, value in evidence['facts'].items()):
                raise ValueError('This run has no accepted stage to continue from.')
            config = self._config(project_id, step, previous, required=True)
            folder = conversation.open_run(run['run_dir'], self.workspace.project_paths(project_id)['runs'])
            attempt = pending_retry or self._new(project_id, workflow_id, step)
            relative = folder.relative_to(self.workspace.project_paths(project_id)['root']).as_posix()
            attempt = self._save(attempt, conversation_path=relative, environment_id=previous['environment_id'],
                                 state='planning', approved_digest=None,
                                 source_context={'kind': source_kind, 'run_id': original_run, 'context_sha256': evidence['context_sha256']})
            try:
                result = conversation.send(folder, message, settings, config=config, requested_action=action)
                pending = result.get('pending')
                return self._save(attempt, state='ready' if pending and pending.get('status') == 'ready' else 'needs_input')
            except Exception as error:
                self._save(attempt, state='needs_attention', last_error=str(error))
                raise

    def review(self, project_id, workflow_id, step_id):
        attempt = self.latest(project_id, workflow_id, step_id)
        if not attempt or not attempt['run_id']:
            raise ValueError('Prepare the inputs first.')
        run = self.workspace.get_run(project_id, attempt['run_id'])
        if run['state'] is None:
            raise ValueError('The run is unavailable.')
        root = Path(run['run_dir'])
        if run['kind'] == 'batch':
            from .batch import _children
            states = [(path, state) for _, path, state in _children(root, run['state'])]
        else:
            states = [(root, run['state'])]
        for folder, state in states:
            if state.get('engine') == 'qe':
                from .qe_workflow import _check_config
            else:
                from .workflow import _check_config
            _check_config(folder, state)
        records = {}
        plan_file = 'batch-plan.json' if run['kind'] == 'batch' else 'plan.json'
        for relative in (plan_file, 'config.json', 'proposal.json', 'parent.json'):
            path = root / relative
            if path.exists():
                if path.is_symlink():
                    raise ValueError('Prepared files cannot be symbolic links.')
                records[relative] = _digest(path)
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Prepared files cannot be symbolic links.')
            if path.is_file() and ('inputs' in path.relative_to(root).parts or path.name in {'plan.json', 'config.json', 'proposal.json'}):
                records[path.relative_to(root).as_posix()] = _digest(path)
        if not {plan_file, 'config.json'} <= records.keys():
            raise ValueError('The prepared plan or configuration is missing.')
        payload = {'attempt_id': attempt['id'], 'run_id': run['id'], 'files': records}
        fingerprint = hashlib.sha256(_object(payload).encode()).hexdigest()
        return {**payload, 'digest': fingerprint, 'state': run['state'], 'run_dir': str(root), 'kind': run['kind']}

    def authorize_step(self, project_id, workflow_id, step_id, expected_digest):
        with self._lock(project_id, workflow_id):
            with self.workspace.edit_project(project_id):
                attempt = self.latest(project_id, workflow_id, step_id)
                if not attempt or attempt['state'] not in {'prepared', 'authorized'}:
                    raise ValueError('Review the prepared inputs before authorizing this run.')
                _, step = self._step(project_id, workflow_id, step_id)
                self._check_source(project_id, workflow_id, step, attempt)
                review = self.review(project_id, workflow_id, step_id)
                if expected_digest != review['digest']:
                    raise ValueError('The inputs changed. Review them before authorizing.')
            return self._save(attempt, state='authorized', approved_digest=review['digest'])

    def launch_step(self, project_id, workflow_id, step_id):
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            attempt = self.latest(project_id, workflow_id, step_id)
            if attempt and attempt['state'] in {'starting', 'active', 'succeeded'}:
                return attempt
            if not attempt or attempt['state'] != 'authorized' or not attempt['approved_digest']:
                raise ValueError('Authorize the reviewed inputs before submitting.')
            self._config(project_id, step, attempt, required=True)
            self._check_source(project_id, workflow_id, step, attempt)
            review = self.review(project_id, workflow_id, step_id)
            if review['digest'] != attempt['approved_digest']:
                self._save(attempt, state='prepared', approved_digest=None)
                raise ValueError('The inputs changed after approval. Review them again.')
            with self.workspace.edit_project(project_id):
                pass
            attempt = self._save(attempt, state='starting', last_error=None)
            try:
                if review['kind'] == 'batch':
                    from .batch import start_batch_worker
                    start_batch_worker(review['run_dir'])
                elif step['engine'] in {'python', 'wannier90'}:
                    from .script_workflow import start_worker
                    start_worker(review['run_dir'])
                else:
                    workflow.start_worker(review['run_dir'])
                return self._save(attempt, state='active')
            except Exception as error:
                # A lost launch acknowledgement is not permission to launch anew.
                self._save(attempt, state='needs_attention', last_error=str(error))
                raise

    def refresh(self, project_id, workflow_id, *, reconnect=False):
        from .project_artifacts import inspect_run
        with self._lock(project_id, workflow_id):
            result = []
            item = self.workspace.get_workflow(project_id, workflow_id)
            for step in item['steps']:
                attempt = self.latest(project_id, workflow_id, step['id'])
                if not attempt or not attempt['run_id']:
                    result.append({'step_id': step['id'], 'attempt': attempt})
                    continue
                run = self.workspace.get_run(project_id, attempt['run_id'])
                if reconnect and run['state'] is not None:
                    if run['kind'] == 'batch':
                        from .batch import resume_batch, _children, read_batch
                        self._config(project_id, step, attempt, required=True)
                        resume_batch(run['run_dir'])
                        children = _children(Path(run['run_dir']), read_batch(run['run_dir']))
                        for _, child, child_state in children:
                            stage = child_state['stages'][child_state['current_stage']]
                            if stage.get('job_id') and stage['status'] not in {'planned', 'submitting'}:
                                workflow.advance(child, poll_only=True)
                    state = run['state']
                    stage = state['stages'][state['current_stage']] if state.get('stages') else {}
                    if stage.get('job_id') or stage.get('staged'):
                        self._config(project_id, step, attempt, required=True)
                        workflow.resume(run['run_dir'])
                        state = workflow.read_state(run['run_dir'])
                        stage = state['stages'][state['current_stage']]
                        # Poll/collect only a known submitted stage; a dependency
                        # becoming ready does not submit it during reconnection.
                        if stage.get('job_id') and stage['status'] not in {'planned', 'submitting'}:
                            workflow.advance(run['run_dir'], poll_only=True)
                report = inspect_run(self.workspace, project_id, attempt['run_id'])
                status = report.get('status', run['status'])
                if report.get('accepted'):
                    status = 'succeeded'
                elif status == 'planned':
                    status = attempt['state']
                attempt = self._save(attempt, state=status, last_error=report.get('error'))
                result.append({'step_id': step['id'], 'attempt': attempt, 'result': report})
            return {'workflow': self.workspace.get_workflow(project_id, workflow_id), 'steps': result}

    def cancel_step(self, project_id, workflow_id, step_id, *, confirmed=False):
        if confirmed is not True:
            raise ValueError('Confirm cancellation for this step.')
        with self._lock(project_id, workflow_id):
            _, step = self._step(project_id, workflow_id, step_id)
            attempt = self.latest(project_id, workflow_id, step_id)
            if not attempt or not attempt['run_id']:
                raise ValueError('This step has no submitted run.')
            self._config(project_id, step, attempt, required=True)
            run = self.workspace.get_run(project_id, attempt['run_id'])
            self._save(attempt, state='cancelling')
            try:
                if run['kind'] == 'batch':
                    from .batch import cancel_batch
                    state = cancel_batch(run['run_dir'])
                else:
                    state = workflow.cancel(run['run_dir'])
                return self._save(attempt, state=state['status'])
            except Exception as error:
                self._save(attempt, state='needs_attention', last_error=str(error))
                raise

    def replace_workflow(self, project_id, workflow_id, title, goal, steps):
        with self._lock(project_id, workflow_id):
            item = self.workspace.get_workflow(project_id, workflow_id)
            attempts = self.attempts(project_id, workflow_id)
            if any(a['run_id'] or a['approved_digest'] or a['state'] in _BUSY for a in attempts):
                raise ValueError('This workflow has prepared or active work. Create a follow-up workflow instead.')
            result = self.workspace.create_workflow(project_id, item['session_id'], title, goal, steps)
            with self.workspace._transaction() as db:
                self.workspace._project_row(db, project_id, writable=True)
                db.execute("UPDATE workflows SET status='superseded',updated_at=? WHERE project_id=? AND id=?", (_now(), project_id, workflow_id))
                self.workspace._event(db, project_id, 'workflow.revised', {'previous': workflow_id, 'workflow_id': result['id']})
            return result
