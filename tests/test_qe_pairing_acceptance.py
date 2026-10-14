"""Acceptance checks across reviewed plans, displayed facts and exported evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import qe_pairing_agent, qe_pairing_workflow as workflow
from vasp_slurm_agent.config import ClusterConfig
from test_qe_pairing_workflow import complete, prepared


def reviewed_run(prepared, target):
    old, transport, state = prepared
    source = old / '01_pairing/inputs/hamiltonian_hr.dat'
    plan = dict(kind='qe_pairing', status='ready', summary='Compare pairing channels.', questions=[],
                unsupported=[], limits=['Model only.'], parameters=state['stages'][0]['parameters'],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), origin='model')
    state = qe_pairing_agent.prepare_task(source, target, ClusterConfig.load(old/'config.json'),
                                         plan, python_executable="python3")
    return target, transport, state


def test_reviewed_proposal_is_frozen_and_tampering_prevents_submission(prepared, tmp_path):
    root, transport, state = reviewed_run(prepared, tmp_path/'reviewed')
    proposal = root/'proposal.json'
    plan = json.loads((root/'plan.json').read_text())
    assert plan['sources']['proposal.json'] == hashlib.sha256(proposal.read_bytes()).hexdigest()
    assert state['plan_sha256'] == hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
    contents = json.loads(proposal.read_text())
    contents['summary'] = 'Changed after review'
    proposal.write_text(json.dumps(contents))
    assert workflow.advance(root, transport)['status'] == 'needs_attention'
    assert not transport.commands


def test_pairing_screen_uses_verified_native_facts_not_cached_numbers(prepared):
    root, _, state = complete(prepared)
    facts = workflow.load_run_context(root)['facts']
    state['stages'][0]['result']['model_result']['channels'][0]['eigenvalue'] = 99
    state['stages'][0]['result']['model_result']['temperature_k'] = 99999
    (root/'run.json').write_text(json.dumps(state))
    app = AppTest.from_string(f'''
from pathlib import Path
from vasp_slurm_agent.qe_pairing_app import run_panel
run_panel(Path({str(root)!r}), None)
''').run()
    assert not app.exception
    table = app.dataframe[0].value.set_index('Channel')
    assert table.loc['singlet', 'Eigenvalue'] == facts['stage_1.singlet.eigenvalue']['value']
    assert table.loc['triplet', 'Eigenvalue'] == facts['stage_1.triplet.eigenvalue']['value']
    assert '99999' not in '\n'.join(x.value for x in app.markdown)


def test_pairing_screen_hides_numbers_when_native_evidence_changes(prepared):
    root, _, _ = complete(prepared)
    result = root/'01_pairing/outputs/result.json'
    value = json.loads(result.read_text())
    value['channels'][0]['eigenvalue'] = 99
    result.write_text(json.dumps(value))
    app = AppTest.from_string(f'''
from pathlib import Path
from vasp_slurm_agent.qe_pairing_app import run_panel
run_panel(Path({str(root)!r}), None)
''').run()
    assert not app.exception
    assert not app.dataframe
    assert any('needs attention' in x.value for x in app.markdown)


def test_pairing_download_preserves_the_reviewed_proposal(prepared, tmp_path):
    root, _, _ = reviewed_run(prepared, tmp_path/'reviewed')
    (root/'doctor.json').write_text('{"success": true}')
    (root/'explanations.json').write_text('{"exchanges": []}')
    with zipfile.ZipFile(workflow.bundle_run(root)) as bundle:
        assert 'proposal.json' in bundle.namelist()
        assert bundle.read('proposal.json') == (root/'proposal.json').read_bytes()
        assert bundle.read('doctor.json') == (root/'doctor.json').read_bytes()
        assert bundle.read('explanations.json') == (root/'explanations.json').read_bytes()
        assert 'config.json' not in bundle.namelist()
