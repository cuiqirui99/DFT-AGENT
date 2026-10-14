import json
import sys

import pytest

from vasp_slurm_agent import agent, cli
from test_qe_pairing_workflow import prepared, complete


@pytest.fixture
def invoke(monkeypatch):
    monkeypatch.setattr(cli, 'migrate_legacy_config', lambda: None)
    def run(*arguments):
        monkeypatch.setattr(sys, 'argv', ['dft-agent', *map(str, arguments)])
        return cli.main()
    return run


def plan_arguments(prepared, tmp_path, monkeypatch):
    root, _, state = prepared
    source = root/'01_pairing/inputs/hamiltonian_hr.dat'
    reply = dict(status='ready', summary='Compare the two channels.', questions=[], unsupported=[],
                 limits=['Model only.'], parameters=state['stages'][0]['parameters'])
    monkeypatch.setattr(agent, '_request_structured', lambda *args, **kwargs: json.dumps(reply))
    return source, tmp_path/'pairing-plan.json', root/'config.json'


def test_cli_pairing_plan_revision_and_frozen_preparation(prepared, tmp_path, monkeypatch, invoke):
    source, plan, config = plan_arguments(prepared, tmp_path, monkeypatch)
    assert invoke('plan', source, 'Compare channels', '--engine', 'qe', '--pairing', '--origin', 'model',
                  '--provider', 'codex', '--output', plan) == 0
    value = json.loads(plan.read_text())
    assert value['kind'] == 'qe_pairing' and value['origin'] == 'model'
    revised = tmp_path/'revised.json'
    assert invoke('task', source, 'Keep these settings', '--engine', 'qe', '--pairing', '--origin', 'model',
                  '--provider', 'codex', '--previous', plan, '--output', revised) == 0
    assert len(json.loads(revised.read_text())['dialogue']) > len(value['dialogue'])
    target = tmp_path/'cli-run'
    assert invoke('prepare', source, target, '--config', config, '--plan', revised, '--cluster-python', 'python3') == 0
    state = json.loads((target/'run.json').read_text())
    assert state['workflow'] == 'pairing' and state['status'] == 'planned'
    assert state['stages'][0]['job_id'] is None
    manifest = json.loads((target/'plan.json').read_text())
    assert 'proposal.json' in manifest['sources']


def test_cli_pairing_missing_runtime_is_actionable(prepared, tmp_path, monkeypatch, invoke, capsys):
    source, plan, config = plan_arguments(prepared, tmp_path, monkeypatch)
    assert invoke('plan', source, 'Compare', '--engine', 'qe', '--pairing', '--origin', 'model',
                  '--provider', 'codex', '--output', plan) == 0
    capsys.readouterr()
    assert invoke('prepare', source, tmp_path/'not-created', '--plan', plan, '--config', config) == 1
    assert '--cluster-python' in capsys.readouterr().err
    assert not (tmp_path/'not-created').exists()


def test_cli_pairing_files_lists_only_verified_collected_artifacts(prepared, invoke, capsys):
    root, _, _ = complete(prepared)
    assert invoke('files', root, '01_pairing') == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'succeeded'
    assert any(item['label'] == 'pairing.npz' for item in report['artifacts'])
    assert all(item['status'] == 'verified' for item in report['artifacts'])
    assert invoke('files', root, '01_pairing', '--fetch', 'pairing.npz') == 1
    assert 'bundle' in capsys.readouterr().err
    (root/'01_pairing/outputs/result.json').write_text('{}')
    assert invoke('files', root, '01_pairing') == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'needs_attention' and not report['artifacts']


def test_cli_pairing_route_requires_explicit_engine_and_origin(prepared, invoke, capsys):
    root, _, _ = prepared
    source = root/'01_pairing/inputs/hamiltonian_hr.dat'
    assert invoke('plan', source, 'Compare', '--pairing', '--engine', 'qe') == 1
    assert '--origin' in capsys.readouterr().err
