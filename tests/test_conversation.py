"""A saved chat keeps its source, revisions and verified results together."""

import hashlib
from importlib.resources import files
import json
from pathlib import Path

import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar

from vasp_slurm_agent import agent, conversation, conversation_store as store, qe_workflow, workflow
from vasp_slurm_agent.config import ClusterConfig
from test_continuation import parent
from test_explanation import accepted_run
from test_qe_context import accepted_qe
from test_qe_agent import response as qe_response
from test_qe_reasoning import decision as qe_decision
from test_recovery import failed_run
from test_task_agent import response as vasp_response, stage


SETTINGS = agent.ModelSettings("codex")


def config(engine):
    return ClusterConfig(host="test.invalid", user="researcher", remote_root="/scratch/test",
        partition="cpu", engine=engine, vasp_command="srun vasp_std", potcar_root="/licensed",
        qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})


def route(action="plan", source_stage=None, original_source=False):
    return {"action": action, "reply": "Review the plan.", "source_stage": source_stage,
            "original_source": original_source}


def plan(engine, size=2):
    operations = [{"type": "supercell", "matrix": [size, 1, 1]}]
    return qe_response(("scf",), operations=operations) if engine == "qe" else vasp_response(
        [stage("scf")], operations=operations)


def provider(monkeypatch, replies):
    pending = list(replies)
    seen = []
    def request(payload, schema, settings, **kwargs):
        name = kwargs["name"]
        seen.append((name, json.loads(payload)))
        expected, response = pending.pop(0)
        assert name == expected
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)
    monkeypatch.setattr(agent, "_request_structured", request)
    return seen, pending


@pytest.fixture(autouse=True)
def no_remote(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Conversation planning must not contact a cluster or submit jobs")
    for module in (workflow, qe_workflow):
        monkeypatch.setattr(module, "SSHTransport", blocked)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "Si.cif"
    path.write_bytes(files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    return path


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_repeated_edits_reload_original_once_and_prepare_without_submission(engine, source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", engine, source)
    original = source.read_bytes()
    source.unlink()
    name = "qe_task" if engine == "qe" else "dft_task"
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route()), (name, plan(engine, 2)),
        ("dft_conversation", route()), (name, plan(engine, 3)),
    ])
    first = conversation.send(folder, "Make a 2x1x1 supercell and run SCF only, without SOC.", SETTINGS, config=config(engine))
    assert first["pending"]["status"] == "ready"
    assert seen[0][1]["has_structure"] is True
    assert seen[0][1]["has_run"] is False
    assert store.load(folder)["messages"] == first["messages"]
    second = conversation.send(Path(str(folder)), "Use 3x1x1 instead. Keep the other settings.", SETTINGS, config=config(engine))
    assert second["pending"]["status"] == "ready"
    assert not remaining
    assert "2x1x1" in json.dumps(seen[-1][1]["history"])
    assert [message["content"] for message in second["messages"] if message["role"] == "user"] == [
        "Make a 2x1x1 supercell and run SCF only, without SOC.", "Use 3x1x1 instead. Keep the other settings."]
    destination = tmp_path / "runs" / "final"
    prepared = conversation.prepare(folder, destination, config(engine))
    state = json.loads((destination / "run.json").read_text())
    assert state["status"] == "planned"
    assert not any(item["job_id"] for item in state["stages"])
    structure = destination / "01_scf/inputs" / ("structure.cif" if engine == "qe" else "POSCAR")
    assert len(Structure.from_file(structure)) == 6
    assert (folder / prepared["source"]).read_bytes() == original
    assert prepared["pending"] is None and prepared["active_run"] == str(destination)
    assert conversation.open_run(destination, tmp_path / "runs") == folder
    with pytest.raises(ValueError, match="review a plan"):
        conversation.prepare(folder, tmp_path / "runs" / "duplicate", config(engine))


@pytest.mark.parametrize("action", ["discuss", "answer", "clarify"])
@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_answer_keeps_draft_but_clarification_waits_for_user(action, engine, source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", engine, source)
    name = "qe_task" if engine == "qe" else "dft_task"
    provider(monkeypatch, [("dft_conversation", route()), (name, plan(engine)),
                           ("dft_conversation", route(action))])
    first = conversation.send(folder, "SCF in a doubled cell", SETTINGS, config=config(engine))
    second = conversation.send(folder, "Why did you choose those settings?", SETTINGS, config=config(engine))
    assert second["pending"] == (None if action == "clarify" else first["pending"])
    assert second["messages"][-1]["action"] == action


@pytest.mark.parametrize("failure_at", ["route", "planner"])
def test_failed_revision_disables_stale_plan(failure_at, source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", "vasp", source)
    replies = [("dft_conversation", route()), ("dft_task", plan("vasp")),
               ("dft_conversation", agent.AgentError("Unavailable") if failure_at == "route" else route())]
    if failure_at == "planner":
        replies.append(("dft_task", agent.AgentError("Unavailable")))
    provider(monkeypatch, replies)
    conversation.send(folder, "Double the cell and run SCF", SETTINGS, config=config("vasp"))
    revised = conversation.send(folder, "Triple it instead", SETTINGS, config=config("vasp"))
    assert revised["pending"] is None
    assert revised["messages"][-1]["error"]
    assert revised["messages"][-2]["content"] == "Triple it instead"
    with pytest.raises(ValueError, match="review a plan"):
        conversation.prepare(folder, tmp_path / "runs" / "stale", config("vasp"))


def test_engine_mismatch_fails_before_saving_or_calling_model(source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", "vasp", source)
    before = (folder / "conversation.json").read_bytes()
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **k: pytest.fail("No model call"))
    with pytest.raises(ValueError, match="engine"):
        conversation.send(folder, "SCF", SETTINGS, config=config("qe"))
    assert (folder / "conversation.json").read_bytes() == before


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_repeated_result_answers_remain_grounded_and_keep_history(engine, request, tmp_path, monkeypatch):
    fixture = request.getfixturevalue("accepted_qe" if engine == "qe" else "accepted_run")
    root = fixture if engine == "qe" else fixture[0]
    folder = conversation.open_run(root, tmp_path)
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and p.name != "conversation.json"}
    answer = {"answer": "The energy is {{stage_1.energy}}.", "evidence": ["stage_1.energy"],
              "limits": [], "next_steps": []}
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route("answer")), ("dft_explanation", answer),
        ("dft_conversation", route("answer")), ("dft_explanation", answer),
    ])
    first = conversation.send(folder, "What is the energy?", SETTINGS)
    second = conversation.send(folder, "And how should I interpret that energy?", SETTINGS)
    assert not remaining and second["messages"][-1]["action"] == "answer"
    assert second["messages"][-1]["context_sha256"] == first["messages"][-1]["context_sha256"]
    assert "What is the energy?" in json.dumps(seen[-1][1]["history"])
    assert before == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p.name != "conversation.json"}
    assert conversation.open_run(root, tmp_path) == folder


def test_vasp_continuation_revision_keeps_parent_and_new_original_escape(parent, tmp_path, monkeypatch):
    root, _ = parent
    folder = conversation.open_run(root, tmp_path)
    provider(monkeypatch, [
        ("dft_conversation", route("continue", 0)), ("dft_task", plan("vasp", 2)),
        ("dft_conversation", route("plan")), ("dft_task", plan("vasp", 3)),
        ("dft_conversation", route("plan", original_source=True)), ("dft_task", plan("vasp", 4)),
    ])
    first = conversation.send(folder, "Double the accepted structure and calculate SCF", SETTINGS)
    assert first["pending"]["kind"] == "vasp_continuation"
    second = conversation.send(folder, "Triple instead", SETTINGS)
    assert second["pending"]["kind"] == "vasp_continuation"
    assert second["pending"]["source_stage"] == 0
    assert second["pending"]["source_run"] == str(root)
    third = conversation.send(folder, "Start fresh from the upload, quadruple and SCF", SETTINGS)
    assert third["pending"]["kind"] == "plan" and third["planning_base"] is None


def test_vasp_continuation_changed_evidence_refuses_prepare(parent, tmp_path, monkeypatch):
    root, outputs = parent
    folder = conversation.open_run(root, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("continue", 0)), ("dft_task", plan("vasp"))])
    record = conversation.send(folder, "Double the accepted structure then SCF", SETTINGS)
    assert record["pending"]["status"] == "ready"
    (outputs / "OUTCAR").write_text("changed evidence")
    destination = tmp_path / "stale"
    with pytest.raises(ValueError, match="accepted|changed"):
        conversation.prepare(folder, destination, config("vasp"))
    assert not destination.exists()


def test_changed_result_context_does_not_reuse_old_answer(parent, tmp_path, monkeypatch):
    root, outputs = parent
    folder = conversation.open_run(root, tmp_path)
    answer = {"answer": "The SCF result passed.", "evidence": ["stage_1.accepted"], "limits": [], "next_steps": []}
    seen, _ = provider(monkeypatch, [
        ("dft_conversation", route("answer")), ("dft_explanation", answer),
        ("dft_conversation", route("clarify")),
    ])
    conversation.send(folder, "Was that accepted?", SETTINGS)
    (outputs / "OUTCAR").write_text("changed evidence")
    conversation.send(folder, "Remind me of the result", SETTINGS)
    assert "Its evidence has changed" in json.dumps(seen[-1][1]["history"])
    assert "The SCF result passed." not in json.dumps(seen[-1][1]["history"])


def test_qe_continuation_prepares_only_verified_followup(accepted_qe, tmp_path, monkeypatch):
    root = accepted_qe
    folder = conversation.open_run(root, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("continue", 0)), ("qe_next_action", qe_decision())])
    record = conversation.send(folder, "Continue with bands without repeating the SCF", SETTINGS)
    assert record["pending"]["kind"] == "qe_next_action" and record["pending"]["status"] == "ready"
    destination = tmp_path / "bands"
    prepared = conversation.prepare(folder, destination)
    state = qe_workflow.read_state(destination)
    assert [item["name"] for item in state["stages"]] == ["bands"]
    assert state["stages"][0]["external_checkpoint"] and not state["stages"][0]["job_id"]
    assert prepared["active_run"] == str(destination)
    assert conversation.open_run(root, tmp_path) == folder
    assert store.load(folder)["active_run"] == str(root)


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_missing_settings_can_be_answered_after_reload(engine, source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", engine, source)
    name = "qe_task" if engine == "qe" else "dft_task"
    incomplete = plan(engine)
    incomplete.update(status="needs_input", questions=["Which supercell size should be used?"])
    seen, _ = provider(monkeypatch, [("dft_conversation", route()), (name, incomplete),
        ("dft_conversation", route()), (name, plan(engine, 3))])
    first = conversation.send(folder, "Make a supercell and run SCF", SETTINGS, config=config(engine))
    assert first["pending"]["status"] == "needs_input"
    with pytest.raises(ValueError, match="review a plan"):
        conversation.prepare(folder, tmp_path / "runs" / "too-early", config(engine))
    second = conversation.send(Path(str(folder)), "Use 3x1x1", SETTINGS, config=config(engine))
    assert second["pending"]["status"] == "ready"
    assert "Which supercell size" in json.dumps(seen[-1][1]["history"])


def test_qe_changed_evidence_refuses_prepared_followup(accepted_qe, tmp_path, monkeypatch):
    folder = conversation.open_run(accepted_qe, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("continue", 0)), ("qe_next_action", qe_decision())])
    record = conversation.send(folder, "Continue with bands", SETTINGS)
    assert record["pending"]["status"] == "ready"
    output = accepted_qe / "01_scf/outputs/pw.out"
    output.write_text(output.read_text() + "\nchanged evidence")
    with pytest.raises(ValueError, match="changed"):
        conversation.prepare(folder, tmp_path / "stale-followup")
    assert not (tmp_path / "stale-followup").exists()


def test_qe_repair_preserves_user_constraints_and_stays_local(accepted_qe, tmp_path, monkeypatch):
    from test_qe_continuation import fail_scf
    fail_scf(accepted_qe)
    folder = conversation.open_run(accepted_qe, tmp_path)
    repair = qe_decision("repair")
    repair.update(repair_action="reduce_mixing", evidence=["failure.reason.1"], summary="Reduce mixing and retry.")
    seen, _ = provider(monkeypatch, [("dft_conversation", route("repair")), ("qe_next_action", repair)])
    message = "Repair convergence but preserve the energy tolerance, functional and cutoff."
    record = conversation.send(folder, message, SETTINGS)
    assert record["pending"]["status"] == "ready"
    assert message in json.dumps(seen[-1][1]["history"])
    destination = tmp_path / "repaired"
    conversation.prepare(folder, destination)
    state = qe_workflow.read_state(destination)
    assert state["status"] == "planned" and not any(s.get("job_id") for s in state["stages"])
    assert state["stages"][0]["parameters"]["namelists"]["electrons"]["mixing_beta"] == .3


def test_vasp_repair_uses_current_question_and_keeps_tolerance(tmp_path, monkeypatch, request):
    create, _, _ = request.getfixturevalue("failed_run")
    root = create()
    folder = conversation.open_run(root, tmp_path)
    reply = {"action": "increase_nelm", "diagnosis": "Allow more SCF iterations.", "questions": []}
    seen, _ = provider(monkeypatch, [("dft_conversation", route("repair")), ("dft_repair", reply)])
    message = "Keep the electronic tolerance and only increase the iteration limit."
    record = conversation.send(folder, message, SETTINGS)
    assert record["pending"]["status"] == "ready"
    assert seen[-1][1]["question"] == message and message in json.dumps(seen[-1][1]["history"])
    destination = tmp_path / "repair"
    conversation.prepare(folder, destination)
    before, after = workflow.read_state(root), workflow.read_state(destination)
    original = Incar.from_file(root / "01_scf/inputs/INCAR")
    repaired = Incar.from_file(destination / "01_scf/inputs/INCAR")
    assert original["EDIFF"] == repaired["EDIFF"]
    assert {key for key in original if original[key] != repaired[key]} == {"NELM"}
    assert after["status"] == "planned" and after["stages"][0]["job_id"] is None


def test_secret_in_user_turn_rejected_before_history_or_model(source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / 'runs', 'vasp', source)
    before = store.load(folder)
    monkeypatch.setattr(agent, '_request_structured', lambda *a, **k: pytest.fail('Credential sent to model'))
    with pytest.raises(ValueError, match='secure'):
        conversation.send(folder, 'API_KEY=not-a-real-secret', SETTINGS, config=config('vasp'))
    assert store.load(folder) == before
