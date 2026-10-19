"""Read-only analysis of explicitly selected results in one project."""

from copy import deepcopy
import hashlib
import re

from . import explanation
from .project_context import _run_path, _value


_COMPARISON_LIMIT = 'Cross-run total-energy comparability has not been established. Do not rank these results by total energy.'
_INSTRUCTIONS = '''
These are separate selected runs with namespaced fact IDs. Identify each run by
its engine and namespace, and cite a fact from every selected run. Report the
verified values separately. Shared composition, units or functional do not
establish total-energy comparability across runs or engines. Never rank total
energies, calculate their difference, infer greater stability, or claim they agree.
Do not derive any new quantities. Python process completion is not scientific
validation. Preserve each run's validation scope and relevant limitations.
'''


def load_results(workspace, project_id, run_ids):
    if not isinstance(run_ids, (list, tuple)) or not 2 <= len(run_ids) <= 3 or len(set(run_ids)) != len(run_ids):
        raise ValueError('Select two or three different results from this project.')
    project = workspace.get_project(project_id)
    if project['archived'] or project['deleted']:
        raise ValueError('Open an active project before analyzing results.')
    roots = []
    # Resolve every identity before opening any solver evidence.
    for run_id in run_ids:
        record = workspace.get_run(project_id, run_id)
        if record['state'] is None:
            raise ValueError('A selected run is missing or its identity changed.')
        root = _run_path(workspace, project_id, record, workspace.project_paths(project_id)['runs'])
        roots.append((record, root))
    result = {'engine': 'mixed', 'workflow': 'project_results', 'goal': 'Analyze the selected project results.',
              'dialogue': [], 'status': 'selected_results', 'outcome': 'Validation scope is recorded separately for each run.',
              'facts': {}, 'artifacts': [], 'limits': [], 'mandatory_limits': [_COMPARISON_LIMIT], 'runs': []}
    for letter, (record, root) in zip('abc', roots):
        context = explanation.load_run_context(root)
        namespace = 'run_' + letter
        item = {'id': record['id'], 'namespace': namespace, 'engine': record['engine'],
                'status': context['status'], 'context_sha256': context['context_sha256'],
                'goal': _value(context.get('goal', '')), 'limits': _value(context.get('limits', []))}
        result['runs'].append(item)
        for identifier, fact in context['facts'].items():
            current = deepcopy(fact)
            current['label'] = record['engine'].upper() + ' (' + namespace + '): ' + current['label']
            current['source'] = namespace + ':' + current.get('source', '')
            result['facts'][namespace + '.' + identifier] = current
        result['facts'][namespace + '.engine'] = {'label': 'Program for ' + namespace, 'value': record['engine'],
                                                  'unit': '', 'source': 'project run registry'}
        for artifact in context['artifacts']:
            result['artifacts'].append({**artifact, 'id': namespace + '.' + artifact['id'],
                'label': record['engine'].upper() + ' (' + namespace + '): ' + artifact['label'],
                'path': namespace + '/' + artifact['path']})
        limits = [namespace + ': ' + limit for limit in context['limits']]
        result['limits'].extend(limits)
        if record['engine'] == 'python':
            result['mandatory_limits'].append(namespace + ': Python execution and readable outputs do not establish scientific correctness.')
        elif record['engine'] == 'wannier90':
            result['mandatory_limits'].append(namespace + ': Wannier checks cover localization and Hamiltonian consistency, not material accuracy.')
    result['limits'].extend(result['mandatory_limits'])
    # No earlier member can change while a later member is being parsed.
    for (_, root), item in zip(roots, result['runs']):
        if explanation.load_run_context(root)['context_sha256'] != item['context_sha256']:
            raise ValueError('The selected results changed during inspection. Ask again.')
    result['context_sha256'] = hashlib.sha256(explanation._canonical(
        {'project_id': project_id, 'context': result}).encode()).hexdigest()
    return result


def _validate_comparison(answer, context):
    evidence = answer['evidence']
    if any(not any(identifier.startswith(item['namespace'] + '.') for identifier in evidence) for item in context['runs']):
        raise ValueError('Support the answer with evidence from every selected run.')
    text = ' '.join([answer['answer'], *answer['limits'], *answer['next_steps']])
    # Negative scope statements remain allowed; positive cross-run energy
    # rankings are not supplied by the individual solver parsers.
    checked = re.sub(r'\b(?:not|never)\s+(?:directly\s+)?(?:comparable|consistent|validated|verified)\b', '', text, flags=re.I)
    patterns = [
        r'\b(?:lower|higher|lowest|highest|more negative|less negative)\b[^.;!?]{0,50}\benerg(?:y|ies)\b',
        r'\benerg(?:y|ies)\b[^.;!?]{0,45}\b(?:lower|higher|lowest|highest|agree|agreement|comparable|match|consistent)\b',
        r'\b(?:more|less|most|least)\s+(?:energetically\s+)?stable\b',
        r'\b(?:energy difference|difference in (?:total )?energ(?:y|ies))\s+(?:is|equals|of|=|:)\s*[-+\d{]',
    ]
    if any(re.search(pattern, checked, re.I) for pattern in patterns):
        raise ValueError('Cross-run energy comparisons are not established.')
    for item in context['runs']:
        if item['engine'] != 'python':
            continue
        subject = r'(?:Python|' + re.escape(item['namespace']) + ')'
        if re.search(subject + r'[^.;!?]{0,80}\b(?:scientifically (?:correct|validated|verified)|scientific (?:validity|accuracy) (?:is )?(?:verified|validated)|results? (?:is |are )?(?:accepted|validated))\b', checked, re.I):
            raise ValueError('Python process completion does not validate scientific results.')


def explain_results(workspace, project_id, run_ids, settings, question, history=None):
    selected = tuple(run_ids)
    latest = {}
    def loader():
        context = load_results(workspace, project_id, selected)
        latest['context'] = context
        return context
    answer = explanation._explain_context(loader, settings, question, history,
        extra_instructions=_INSTRUCTIONS, validate_answer=_validate_comparison)
    answer['sources'] = [{key: item[key] for key in ('id', 'namespace', 'engine', 'status', 'context_sha256')}
                         for item in latest['context']['runs']]
    return answer
