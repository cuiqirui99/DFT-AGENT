"""Mocked model routing with real project storage, file tools and result gates."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import agent, project_agent, project_files, workflow, qe_workflow
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.model_usage import ModelResponse, usage_record
from vasp_slurm_agent.workspace import Workspace
from test_explanation import accepted_run, model_reply


@pytest.fixture
def project(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Project chat cannot contact a cluster or submit jobs")
    monkeypatch.setattr(workflow, "SSHTransport", blocked)
    monkeypatch.setattr(qe_workflow, "SSHTransport", blocked)
    store = Workspace(tmp_path / "workspace")
    registry = ConnectionRegistry(store)
    pid = store.create_project("Silicon")["id"]
    sid = store.create_session(pid)["id"]
    connection = registry.create("model", "Codex", {"provider": "codex"})["id"]
    registry.bind(pid, connection)
    project_files.write_file(store, pid, "files/Si.cif", "structure draft")
    return store, registry, pid, sid, connection


def reply(action="clarify", **changes):
    return {"action": action, "message": "Which method should this step use?", "workflow_id": None,
        "step_id": None, "run_id": None, "workflow": None, "file": None, "memory": None, **changes}


def steps():
    return [
        {"id": "relax", "name": "Relax", "engine": "vasp", "goal": "Relax at PBE without SOC.",
         "source": {"kind": "file", "path": "files/Si.cif", "step_id": None, "stage": None},
         "environment_id": None, "depends_on": [], "inputs": [], "input_artifacts": [], "arguments": [], "outputs": []},
        {"id": "phonons", "name": "Phonons", "engine": "qe", "goal": "Calculate phonons of the relaxed structure.",
         "source": {"kind": "step", "path": None, "step_id": "relax", "stage": None},
         "environment_id": None, "depends_on": ["relax"], "inputs": [], "input_artifacts": [], "arguments": [], "outputs": []},
    ]


def graph(**changes):
    return {"title": "Relax and phonons", "goal": "VASP relaxation, then QE phonons.",
            "replace_selected": False, "steps": steps(), **changes}


def provider(monkeypatch, replies):
    remaining = list(replies)
    calls = []
    def request(payload, schema, settings, **kwargs):
        calls.append((kwargs["name"], json.loads(payload)))
        response = remaining.pop(0)
        if callable(response):
            response = response(calls[-1][1])
        if isinstance(response, Exception):
            raise response
        return ModelResponse(json.dumps(response), usage_record("codex", "test", reported={"input_tokens": 5, "output_tokens": 4}))
    monkeypatch.setattr(agent, "_request_structured", request)
    return calls, remaining


def send(project, message, **kwargs):
    store, _, pid, sid, cid = project
    return project_agent.send(store, pid, sid, message, cid, **kwargs)


def test_mixed_workflow_persists_and_followup_restores_project_history(project, monkeypatch):
    calls, remaining = provider(monkeypatch, [reply("workflow", workflow=graph()), reply("status")])
    first = send(project, "Relax with VASP, then use QE for phonons.")
    assert first["action"] == "workflow" and first["result"]["status"] == "planned"
    store, registry, pid, sid, cid = project
    saved = Workspace(store.home).get_workflow(pid, first["workflow_id"])
    assert saved["steps"][1]["spec"]["source"] == {"kind": "step", "step_id": "relax", "stage": None}
    assert saved["steps"][1]["depends_on"] == ["relax"]
    restored = (Workspace(store.home), registry, pid, sid, cid)
    second = send(restored, "Show the plan status.", workflow_id=saved["id"])
    assert second["action"] == "status" and "planned" in second["message"]
    assert "Relax with VASP" in json.dumps(calls[1][1]["context"]["session"]["history"])
    assert [m["role"] for m in second["session"]["messages"]] == ["user", "assistant", "user", "assistant"]
    assert not remaining and store.list_runs(pid) == []


@pytest.mark.parametrize("change,match", [
    ("missing_file", "unavailable"), ("dependency", "dependency"),
    ("foreign_environment", "matching environment"), ("cycle", "cycle"),
])
def test_invalid_workflow_never_becomes_executable(project, monkeypatch, change, match):
    draft = graph()
    if change == "missing_file":
        draft["steps"][0]["source"]["path"] = "files/missing.cif"
    elif change == "dependency":
        draft["steps"][1]["depends_on"] = []
    elif change == "foreign_environment":
        draft["steps"][0]["environment_id"] = "not-project-bound"
    else:
        draft["steps"][0]["depends_on"] = ["phonons"]
    provider(monkeypatch, [reply("workflow", workflow=draft)])
    answer = send(project, "Create the workflow.")
    assert answer["action"] == "error" and match in answer["message"]
    assert project[0].list_workflows(project[2]) == []


@pytest.mark.parametrize("selection", ["session", "model", "workflow", "run"])
def test_foreign_selection_is_rejected_before_model_or_history(project, monkeypatch, selection):
    store, registry, pid, sid, cid = project
    other = store.create_project("Private project")["id"]
    other_session = store.create_session(other)["id"]
    kwargs = {}
    if selection == "session":
        sid = other_session
    elif selection == "model":
        cid = registry.create("model", "Foreign model", {"provider": "codex"})["id"]
        registry.bind(other, cid)
    elif selection == "workflow":
        kwargs["workflow_id"] = store.create_workflow(other, other_session, "Other", "Other",
            [{"id": "scf", "name": "SCF", "engine": "qe"}])["id"]
    else:
        kwargs["run_ids"] = ["foreign-run"]
    provider(monkeypatch, [])
    with pytest.raises(ValueError):
        project_agent.send(store, pid, sid, "Inspect this.", cid, **kwargs)
    assert store.get_session(pid, project[3])["messages"] == []


def test_context_contains_only_bound_environment_metadata(project, monkeypatch):
    store, registry, pid, _, _ = project
    hpc = registry.create("hpc", "Own HPC", {"host": "never-export-host", "user": "private-user",
        "remote_base": "/private/scratch"})["id"]
    own = registry.create_environment(hpc, "Own VASP", "vasp", {"vasp_command": "srun vasp_std", "potcar_root": "/private/potentials", "partition": "cpu"})
    registry.bind(pid, hpc)
    foreign = registry.create("hpc", "Other HPC", {"host": "foreign-host", "user": "foreign-user"})["id"]
    registry.create_environment(foreign, "Foreign environment", "qe", {"partition": "cpu", "qe_pseudo_root": "/private/pseudos"})
    calls, _ = provider(monkeypatch, [reply()])
    send(project, "Help me plan.")
    context = calls[0][1]["context"]
    assert context["environments"] == [{"id": own["id"], "name": "Own VASP", "engine": "vasp"}]
    text = json.dumps(context)
    assert all(value not in text for value in ("private-user", "never-export-host", "/private", "foreign", "Foreign environment"))


def test_provider_error_and_project_change_keep_turn_without_mutation(project, monkeypatch):
    provider(monkeypatch, [agent.AgentError("The model reached its rate limit.")])
    answer = send(project, "Plan a relaxation.")
    assert answer["action"] == "error" and "rate limit" in answer["message"]
    store, _, pid, sid, _ = project
    assert len(store.get_session(pid, sid)["messages"]) == 2
    def changed(_):
        store.save_memory(pid, "new preference", "Keep the cell fixed")
        return reply("workflow", workflow=graph())
    provider(monkeypatch, [changed])
    answer = send(project, "Try again.")
    assert answer["action"] == "error" and "project changed" in answer["message"]
    assert store.list_workflows(pid) == []


def test_session_operation_lock_blocks_duplicate_requests(project, monkeypatch):
    store, _, pid, sid, _ = project
    provider(monkeypatch, [])
    with project_agent._operation(store, pid, sid):
        with pytest.raises(agent.AgentError, match="busy"):
            send(project, "Plan a relaxation.")
    assert store.get_session(pid, sid)["messages"] == []


def test_secrets_are_rejected_before_persistence_or_file_model_context(project, monkeypatch):
    provider(monkeypatch, [])
    with pytest.raises(ValueError, match="secure"):
        send(project, "API_KEY=do-not-save-this")
    store, _, pid, sid, _ = project
    assert store.get_session(pid, sid)["messages"] == []
    (store.project_paths(pid)["files"] / "notes.txt").write_text("password: private-value")
    calls, _ = provider(monkeypatch, [reply("read_file", file={"path": "files/notes.txt", "content": None})])
    answer = send(project, "Open notes.")
    assert answer["action"] == "error" and "private-value" not in json.dumps(answer)
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["../outside.txt", "config/cluster.json", "files/POTCAR", "files/Si.UPF"])
def test_private_and_external_files_are_never_sent_or_written(project, monkeypatch, path):
    provider(monkeypatch, [reply("write_file", file={"path": path, "content": "ordinary text"})])
    answer = send(project, "Save a note.")
    assert answer["action"] == "error"


def test_file_read_edit_review_and_restart_keep_original_until_apply(project, monkeypatch):
    store, _, pid, sid, _ = project
    original = project_files.write_file(store, pid, "files/analysis.py", "print('old')\n")
    calls, remaining = provider(monkeypatch, [
        reply("read_file", file={"path": "files/analysis.py", "content": None}),
        reply("write_file", file={"path": "files/analysis.py", "content": "print('new')\n"}),
    ])
    answer = send(project, "Change old to new in analysis.py.")
    assert calls[1][1]["files_read"][0]["content"] == "print('old')\n"
    assert answer["action"] == "file_edit" and not remaining
    assert project_files.read_file(store, pid, "files/analysis.py")["content"] == "print('old')\n"
    proposal = Workspace(store.home).get_session(pid, sid)["messages"][-1]["metadata"]["file_edit"]
    assert proposal["expected_sha256"] == original["sha256"]
    project_files.write_file(store, pid, proposal["path"], proposal["content"], expected_sha256=proposal["expected_sha256"])
    assert project_files.read_file(store, pid, "files/analysis.py")["content"] == "print('new')\n"


def test_same_length_file_change_rejects_stale_edit(project, monkeypatch):
    store, _, pid, _, _ = project
    original = project_files.write_file(store, pid, "files/note.txt", "old")
    def change(_):
        project_files.write_file(store, pid, "files/note.txt", "now", expected_sha256=original["sha256"])
        return reply("write_file", file={"path": "files/note.txt", "content": "new"})
    provider(monkeypatch, [reply("read_file", file={"path": "files/note.txt", "content": None}), change])
    answer = send(project, "Edit the note.")
    assert answer["action"] == "error" and "file changed" in answer["message"]
    assert project_files.read_file(store, pid, "files/note.txt")["content"] == "now"


def test_new_file_and_explicit_memory_are_local_only(project, monkeypatch):
    provider(monkeypatch, [reply("write_file", file={"path": "files/check.py", "content": "print('draft')\n"}),
        reply("remember", memory={"key": "structure", "text": "Keep the cell fixed"})])
    first = send(project, "Write a small analysis script.")
    assert first["action"] == "write_file"
    second = send(project, "Remember to keep the cell fixed.")
    assert second["action"] == "remember"
    assert project[0].list_memory(project[2])[0]["text"] == "Keep the cell fixed"
    assert project[0].list_runs(project[2]) == []


def test_selected_step_dispatches_only_draft_and_preserves_full_message(project, monkeypatch):
    store, _, pid, sid, _ = project
    created = store.create_workflow(pid, sid, "SCF", "SCF", [
        {"id": "scf", "name": "SCF", "engine": "vasp", "spec": {"goal": "SCF"}}])
    calls = []
    class Execution:
        def latest(self, *args):
            return None
        def __init__(self, workspace, registry):
            assert workspace is store
        def draft_step(self, project_id, workflow_id, step_id, settings, message=None):
            calls.append((project_id, workflow_id, step_id, settings.connection_id, message))
            return {"id": "attempt", "reply": "Review the revised step.", "state": "drafted"}
        def launch_step(self, *args):
            pytest.fail("No model-triggered submission")
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.project_execution", SimpleNamespace(ProjectExecution=Execution))
    provider(monkeypatch, [reply("draft_step")])
    answer = send(project, "Use 3x1x1 instead. Keep SOC off.", workflow_id=created["id"], step_id="scf")
    assert answer["action"] == "draft_step"
    assert calls == [(pid, created["id"], "scf", project[4], "Use 3x1x1 instead. Keep SOC off.")]


@pytest.mark.parametrize("action,method", [("repair_step", "retry_step"), ("recover_step", "recover_step")])
def test_repair_and_local_recovery_never_launch(project, monkeypatch, action, method):
    store, _, pid, sid, _ = project
    created = store.create_workflow(pid, sid, "SCF", "SCF", [{"id": "scf", "name": "SCF", "engine": "qe"}])
    calls = []
    class Execution:
        def latest(self, *args):
            return None
        def __init__(self, *args):
            pass
    setattr(Execution, method, lambda self, *args, **kwargs:
            calls.append((args, kwargs)) or {"reply": "Review this step.", "state": "needs_input"})
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.project_execution", SimpleNamespace(ProjectExecution=Execution))
    provider(monkeypatch, [reply(action)])
    result = send(project, "Help recover this step.", workflow_id=created['id'], step_id='scf')
    assert result['action'] == action and len(calls) == 1
    assert calls[0][0][:3] == (pid, created['id'], 'scf')


def test_known_connection_secrets_are_not_saved_or_sent(project, monkeypatch):
    monkeypatch.setattr(agent, '_secrets', lambda settings: ['opaque-private-value'])
    store, _, pid, sid, _ = project
    with pytest.raises(ValueError, match='secure'):
        send(project, 'Here is opaque-private-value')
    assert store.get_session(pid, sid)['messages'] == []
    store.save_memory(pid, 'old note', 'opaque-private-value')
    calls, _ = provider(monkeypatch, [reply()])
    send(project, 'Help plan this project.')
    assert 'opaque-private-value' not in json.dumps(calls)


def test_results_use_evidence_explainer_and_reject_invented_numbers(project, accepted_run, monkeypatch):
    store, _, pid, _, _ = project
    target = store.project_paths(pid)["runs"] / "accepted"
    shutil.copytree(accepted_run[0], target)
    run = store.register_run(pid, target)
    calls, _ = provider(monkeypatch, [reply("explain", run_id=run["id"]),
        model_reply("The energy is -5.0 eV per atom.", ["stage_1.energy_per_atom"]),
        reply("explain", run_id=run["id"]),
        model_reply("The energy is -99.0 eV per atom.", ["stage_1.energy_per_atom"])])
    valid = send(project, "What is the energy?", run_ids=[run["id"]])
    assert valid["action"] == "explain" and valid["result"]["evidence"] == ["stage_1.energy_per_atom"]
    invalid = send(project, "And the energy again?", run_ids=[run["id"]])
    assert invalid["action"] == "error" and "could not be verified" in invalid["message"]
    assert [call[0] for call in calls] == ["dft_project_action", "dft_explanation"] * 2
    assert "private-host" not in json.dumps(calls)


def test_generic_workflow_drafts_real_owned_inputs_without_submission(project, monkeypatch):
    store, _, pid, _, _ = project
    project_files.write_file(store, pid, 'files/analyze.py', "print('analysis')\n")
    project_files.write_file(store, pid, 'files/input.csv', 'x,y\n1,2\n')
    item = steps()[0]
    item.update(engine='python', goal='Analyze the supplied CSV',
        source={'kind': 'file', 'path': 'files/analyze.py', 'step_id': None, 'stage': None},
        inputs=['files/input.csv'], arguments=['input.csv'],
        outputs=[{'path': 'result.json', 'kind': 'json', 'required': True}])
    provider(monkeypatch, [reply('workflow', workflow=graph(steps=[item])), reply('draft_step')])
    planned = send(project, 'Use the reviewed Python program for this table.')
    assert planned['action'] == 'workflow'
    drafted = send(project, 'Review these inputs.', workflow_id=planned['workflow_id'], step_id=item['id'])
    assert drafted['action'] == 'draft_step' and drafted['result']['state'] == 'ready'
    assert {entry['path'] for entry in drafted['result']['proposal']['files']} == {'files/analyze.py', 'files/input.csv'}
    assert drafted['result']['proposal']['arguments'] == ['input.csv']
    assert store.list_runs(pid) == []


def test_generic_workflow_cannot_infer_structure_from_another_program(project, monkeypatch):
    item = steps()[1]
    item['engine'] = 'python'
    provider(monkeypatch, [reply('workflow', workflow=graph(steps=[steps()[0], item]))])
    result = send(project, 'Use the previous result as a Python program.')
    assert result['action'] == 'error' and 'reviewed project files' in result['message']
    assert project[0].list_workflows(project[2]) == []


def test_native_workflow_rejects_ignored_generic_arguments(project, monkeypatch):
    item = steps()[0]
    item['arguments'] = ['--secret-option']
    provider(monkeypatch, [reply('workflow', workflow=graph(steps=[item]))])
    result = send(project, 'Run this.')
    assert result['action'] == 'error' and 'scientific adapters' in result['message']


def test_model_can_connect_declared_outputs_to_generic_inputs(project, monkeypatch):
    store, _, pid, _, _ = project
    project_files.write_file(store, pid, 'files/source.py', "print('source')\n")
    project_files.write_file(store, pid, 'files/consumer.py', "print('consumer')\n")
    first = steps()[0]
    first.update(id='source', engine='python', goal='Generate the data',
        source={'kind': 'file', 'path': 'files/source.py', 'step_id': None, 'stage': None},
        outputs=[{'path': 'data.json', 'kind': 'json', 'required': True}])
    second = deepcopy(first)
    second.update(id='consumer', goal='Analyze the prior output', depends_on=['source'],
        source={'kind': 'file', 'path': 'files/consumer.py', 'step_id': None, 'stage': None},
        input_artifacts=[{'step_id': 'source', 'path': 'data.json', 'name': 'input.json'}],
        outputs=[{'path': 'analysis.json', 'kind': 'json', 'required': True}])
    provider(monkeypatch, [reply('workflow', workflow=graph(steps=[first, second]))])
    answer = send(project, 'Pass the generated JSON to the analysis program.')
    assert answer['action'] == 'workflow'
    saved = Workspace(store.home).get_workflow(pid, answer['workflow_id'])
    assert saved['steps'][1]['spec']['input_artifacts'] == second['input_artifacts']
    assert store.list_runs(pid) == []


def test_model_artifact_reference_requires_dependency(project, monkeypatch):
    store, _, pid, _, _ = project
    project_files.write_file(store, pid, 'files/consumer.py', "print('consumer')\n")
    item = steps()[0]
    item.update(engine='python', source={'kind': 'file', 'path': 'files/consumer.py', 'step_id': None, 'stage': None},
        input_artifacts=[{'step_id': 'other', 'path': '01_python/outputs/data.json', 'name': 'input.json'}])
    provider(monkeypatch, [reply('workflow', workflow=graph(steps=[item]))])
    answer = send(project, 'Read the output from the other step.')
    assert answer['action'] == 'error' and 'declared predecessor' in answer['message']
    assert store.list_workflows(pid) == []
