"""Project-owned, evidence-checked outputs used as explicit program inputs."""

from pathlib import Path, PurePosixPath
import hashlib

from . import explanation, script_workflow
from .project_artifacts import inspect_run
from .project_context import _private_file, _run_path
from .project_files import MAX_UPLOAD_BYTES, _safe_content
from .secret_guard import reject_ordinary_text


PROVENANCE_NAME = 'input_provenance.json'


def validate_reference(reference, dependencies=None):
    if not isinstance(reference, dict) or set(reference) != {'step_id', 'path', 'name'}:
        raise ValueError('Select an output step, path and input filename.')
    step_id, relative, name = (reference[key] for key in ('step_id', 'path', 'name'))
    if not isinstance(step_id, str) or not step_id or dependencies is not None and step_id not in dependencies:
        raise ValueError('An input artifact must come from a declared predecessor step.')
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Choose a retained output from the predecessor run.')
    parts = PurePosixPath(relative).parts
    if (len(parts) not in {1, 3} or len(parts) == 3 and parts[1] != 'outputs'
            or any(part in {'.', '..'} for part in parts)):
        raise ValueError('Choose a declared output name or retained output path from the predecessor.')
    script_workflow._name(parts[0])
    script_workflow._name(parts[-1])
    script_workflow._name(name)
    if _private_file(relative) or _private_file(name) or name in script_workflow._INTERNAL | script_workflow._RECEIPTS | {PROVENANCE_NAME}:
        raise ValueError('Private files and execution receipts cannot be program inputs.')
    if parts[-1] in script_workflow._INTERNAL | script_workflow._RECEIPTS | {PROVENANCE_NAME}:
        raise ValueError('Choose a declared scientific output, not an execution receipt.')
    return reference


def _snapshot(workspace, project_id, run_id):
    record = workspace.get_run(project_id, run_id)
    root = _run_path(workspace, project_id, record, workspace.project_paths(project_id)['runs'])
    if record['state'] is None or record['kind'] != 'run':
        raise ValueError('Choose a retained result from one workflow step.')
    report = inspect_run(workspace, project_id, run_id)
    if not (report['process_success'] if record['engine'] == 'python' else report['accepted']):
        raise ValueError('The source has no verified completed output. Review its evidence first.')
    context = explanation.load_run_context(root)
    if context['context_sha256'] != report['context_sha256']:
        raise ValueError('The source evidence changed. Inspect it again.')
    declared = {}
    if record['engine'] in {'python', 'wannier90'}:
        for stage in record['state']['stages']:
            for item in stage['outputs']:
                declared[stage['folder'] + '/outputs/' + item['path']] = item['kind']
    else:
        # The native parsers declare their validated, retained artifacts.
        declared = {item['path']: item.get('kind', 'native_artifact') for item in context['artifacts']}
    available = []
    for item in context['artifacts']:
        if item.get('path') not in declared or item.get('status') != 'verified' or not item.get('sha256'):
            continue
        try:
            validate_reference({'step_id': 'source', 'path': item['path'], 'name': Path(item['path']).name})
        except ValueError:
            continue
        available.append({'path': item['path'], 'name': Path(item['path']).name,
                          'kind': declared[item['path']], 'sha256': item['sha256']})
    return record, root, report, available


def available_inputs(workspace, project_id, run_id):
    """Return only source-bounded metadata, never file contents or cluster settings."""
    _, _, report, available = _snapshot(workspace, project_id, run_id)
    counts = {item['name']: sum(other['name'] == item['name'] for other in available) for item in available}
    public = [{**item, 'path': item['name'] if counts[item['name']] == 1 else item['path']} for item in available]
    return {'artifacts': public, 'accepted': report['accepted'],
            'process_success': report['process_success'], 'limits': report['limits'],
            'context_sha256': report['context_sha256']}


def resolve_input(workspace, project_id, run_id, reference):
    """Resolve one explicit output and recheck its bytes; no copying or execution."""
    validate_reference(reference)
    record, root, report, available = _snapshot(workspace, project_id, run_id)
    matches = [item for item in available if item['path'] == reference['path']
               or '/' not in reference['path'] and item['name'] == reference['path']]
    if len(matches) != 1:
        raise ValueError('Choose one declared, verified output; use its retained path when the name is ambiguous.')
    selected = matches[0]
    path = workspace.resolve_path(project_id, record['relative_path'] + '/' + selected['path'], must_exist=True)
    if not path.is_file() or path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError('Choose a retained output smaller than 512 MiB.')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != selected['sha256']:
        raise ValueError('The selected output changed. Inspect the predecessor again.')
    _safe_content(data)
    try:
        reject_ordinary_text(data.decode('utf-8'))
    except UnicodeDecodeError:
        pass
    if explanation.load_run_context(root)['context_sha256'] != report['context_sha256']:
        raise ValueError('The source evidence changed while selecting its output.')
    receipt = {**reference, 'resolved_path': selected['path'], 'project_id': project_id, 'source_run_id': run_id,
        'source_engine': record['engine'], 'kind': selected['kind'], 'sha256': selected['sha256'],
        'size': len(data), 'source_context_sha256': report['context_sha256'],
        'source_accepted': report['accepted'], 'source_process_success': report['process_success'],
        'limits': report['limits']}
    return path, receipt
