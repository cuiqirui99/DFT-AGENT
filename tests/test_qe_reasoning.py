import json

import pytest

from vasp_slurm_agent import agent, qe_reasoning, qe_workflow
from test_qe_agent import response
from test_qe_context import accepted_qe
from test_qe_continuation import fail_scf


def decision(action="continue", parameters=None):
    plan = response(("bands",), parameters)
    return {"action": action, "summary": "Continue with bands.", "questions": [],
            "evidence": ["stage_1.accepted"], "limits": [], "repair_action": None,
            "continuation": {"source_stage": 0, "intent": plan["intent"], "stages": plan["stages"]}
            if action == "continue" else None}


def mock(monkeypatch, data):
    def reply(*a, **kw):
        return json.dumps(data)
    monkeypatch.setattr(agent, "_request_structured", reply)


def test_model_continuation_checks_then_prepares_without_submission(accepted_qe, tmp_path, monkeypatch):
    state = qe_workflow.read_state(accepted_qe)
    original_dialogue = [{"role": "user", "content": "Study the bands of this silicon structure."}]
    qe_workflow._write(accepted_qe / "proposal.json", {"goal": "Study silicon bands.", "dialogue": original_dialogue,
        "execution": {"plan_sha256": state["plan_sha256"]}})
    before = {str(p): qe_workflow._digest(p) for p in accepted_qe.rglob("*") if p.is_file()}
    mock(monkeypatch, decision())
    proposal = qe_reasoning.draft_next_action("Now calculate bands", accepted_qe, agent.ModelSettings("codex"))
    assert proposal["status"] == "ready" and proposal["action"] == "continue"
    assert proposal["model_usage"] and not proposal["submits_jobs"]
    state = qe_reasoning.prepare_next_action(accepted_qe, tmp_path / "next", proposal)
    assert [stage["name"] for stage in state["stages"]] == ["bands"]
    assert state["status"] == "planned" and not any(stage["job_id"] for stage in state["stages"])
    assert (tmp_path / "next/reasoning.json").exists()
    saved = json.loads((tmp_path / "next/proposal.json").read_text())
    assert saved["goal"] == "Study silicon bands." and saved["dialogue"][0] == original_dialogue[0]
    assert saved["execution"]["plan_sha256"] == state["plan_sha256"]
    assert before == {str(p): qe_workflow._digest(p) for p in accepted_qe.rglob("*") if p.is_file()}


def test_evidence_changed_after_model_return_rejects_proposal(accepted_qe, monkeypatch):
    def reply(*a, **kw):
        path = accepted_qe / "01_scf/outputs/pw.out"
        path.write_text(path.read_text() + "\nchanged")
        return json.dumps(decision("explain"))
    monkeypatch.setattr(agent, "_request_structured", reply)
    with pytest.raises(agent.AgentError, match="changed"):
        qe_reasoning.draft_next_action("What next?", accepted_qe, agent.ModelSettings("codex"))


def test_evidence_changed_before_prepare_rejects_proposal(accepted_qe, tmp_path, monkeypatch):
    mock(monkeypatch, decision())
    proposal = qe_reasoning.draft_next_action("Bands", accepted_qe, agent.ModelSettings("codex"))
    path = accepted_qe / "01_scf/outputs/pw.out"
    path.write_text(path.read_text() + "\nchanged")
    with pytest.raises(ValueError, match="changed"):
        qe_reasoning.prepare_next_action(accepted_qe, tmp_path / "next", proposal)
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize("parameters", [{"soc": True}, {"hubbard": [{"label": "Si", "manifold": "3p"}]}, {"ENCUT": 400}])
def test_missing_or_invalid_science_becomes_question(accepted_qe, monkeypatch, parameters):
    mock(monkeypatch, decision(parameters=parameters))
    proposal = qe_reasoning.draft_next_action("Continue", accepted_qe, agent.ModelSettings("codex"))
    assert proposal["action"] == proposal["status"] == "needs_input"
    assert proposal["questions"] and proposal["continuation"] is None


def test_model_can_only_select_verified_failure_repairs(accepted_qe, tmp_path, monkeypatch):
    fail_scf(accepted_qe)
    choice = decision("repair")
    choice.update(repair_action="reduce_mixing", evidence=["failure.reason.1"], summary="Reduce mixing and retry.")
    mock(monkeypatch, choice)
    proposal = qe_reasoning.draft_next_action("Repair this failure", accepted_qe, agent.ModelSettings("codex"))
    state = qe_reasoning.prepare_next_action(accepted_qe, tmp_path / "repair", proposal)
    assert state["status"] == "planned"
    assert state["stages"][0]["parameters"]["namelists"]["electrons"]["mixing_beta"] == 0.3
    assert json.loads((tmp_path / "repair/repair.json").read_text())["attempt"] == 1


def test_repair_cannot_invent_resource_failure(accepted_qe, monkeypatch):
    fail_scf(accepted_qe)
    choice = decision("repair")
    choice["repair_action"] = "increase_walltime"
    mock(monkeypatch, choice)
    with pytest.raises(agent.AgentError, match="verified"):
        qe_reasoning.draft_next_action("Repair", accepted_qe, agent.ModelSettings("codex"))


def test_cg_repair_is_selectable_only_when_verified_and_offered():
    choice = decision("repair")
    choice.update(repair_action="use_cg_diagonalization", evidence=["failure.reason.0"])
    context = {"facts": {"failure.reason.0": {"value": "Some band eigenvalues did not converge."}}}
    offered = {"actions": [{"action": "use_cg_diagonalization"}]}
    qe_reasoning._validate(choice, context, [], offered)
    with pytest.raises(ValueError, match="unavailable"):
        qe_reasoning._validate(choice, context, [], {"actions": []})


def test_unavailable_methods_are_not_ordinary_calculations(accepted_qe, monkeypatch):
    choice = decision("unsupported")
    choice.update(summary="This pairing solver is unavailable.", questions=["Which interaction model and pairing method should be used?"])
    mock(monkeypatch, choice)
    proposal = qe_reasoning.draft_next_action("Find unconventional superconductivity", accepted_qe, agent.ModelSettings("codex"))
    assert proposal["status"] == "unsupported" and proposal["continuation"] is None


def test_model_payload_omits_cluster_paths_and_credentials(accepted_qe, monkeypatch):
    seen = []
    def reply(payload, *a, **kw):
        seen.append(payload)
        return json.dumps(decision("explain"))
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "private-secret-value")
    monkeypatch.setattr(agent, "_request_structured", reply)
    proposal = qe_reasoning.draft_next_action("Explain private-secret-value", accepted_qe, agent.ModelSettings("codex"))
    text = seen[0] + json.dumps(proposal)
    assert "private-secret-value" not in text
    assert "/scratch/jobs" not in text and "/pseudo" not in text and "test.invalid" not in text


def test_native_override_preserves_hamiltonian_and_accuracy():
    original = {"namelists": {"SYSTEM": {"input_dft": "LDA", "nspin": 2, "starting_magnetization(1)": 0.5},
                             "electrons": {"conv_thr": 1e-12}}}
    choice = decision(parameters={"namelists": {"ELECTRONS": {"electron_maxstep": 200}}})
    _, recipes = qe_reasoning._continuation_plan(choice, {"parameters": original})
    native = recipes[0]["parameters"]["namelists"]
    assert native["system"] == original["namelists"]["SYSTEM"]
    assert native["electrons"] == {"conv_thr": 1e-12, "electron_maxstep": 200}
    assert "ELECTRONS" not in native


def test_continuation_payload_filters_native_free_text(accepted_qe, monkeypatch):
    original = qe_reasoning.qe_continuation.inspect_continuation
    def inspect(*args, **kwargs):
        result = original(*args, **kwargs)
        result["parameters"]["namelists"] = {"control": {"title": "private.customer.host"},
            "system": {"input_dft": "PBE"}, "electrons": {"conv_thr": 1e-12}}
        return result
    monkeypatch.setattr(qe_reasoning.qe_continuation, "inspect_continuation", inspect)
    captured = []
    def reply(payload, *args, **kwargs):
        captured.append(json.loads(payload))
        return json.dumps(decision("explain"))
    monkeypatch.setattr(agent, "_request_structured", reply)
    qe_reasoning.draft_next_action("Explain", accepted_qe, agent.ModelSettings("codex"))
    assert "private.customer.host" not in json.dumps(captured)
    native = captured[0]["continuations"][0]["parameters"]["namelists"]
    assert native["system"]["input_dft"] == "PBE" and native["electrons"]["conv_thr"] == 1e-12


def test_next_step_retrieves_qe_guidance_excluding_current_run(accepted_qe, monkeypatch, tmp_path):
    from vasp_slurm_agent import qe_guidance
    requested, payloads = [], []
    guidance = {"guidance": [{"id": "qe-checkpoints", "checks": ["Recheck convergence after a method change."]}],
                "experience": [], "limits": []}
    def build(structure_path, **kwargs):
        requested.append((structure_path, kwargs))
        return guidance
    def reply(payload, *a, **kw):
        payloads.append(json.loads(payload))
        return json.dumps(decision("explain"))
    monkeypatch.setattr(qe_guidance, "build_context", build)
    monkeypatch.setattr(agent, "_request_structured", reply)
    result = qe_reasoning.draft_next_action("Explain", accepted_qe, agent.ModelSettings("codex"), runs_root=tmp_path)
    assert requested[0][0].is_file()
    assert requested[0][1]["exclude_run"] == accepted_qe and requested[0][1]["runs_root"] == tmp_path
    assert payloads[0]["scientific_context"] == result["scientific_report"] == guidance
def test_next_step_schema_limits_evidence_to_available_fact_ids():
    from vasp_slurm_agent import qe_reasoning
    schema = qe_reasoning._schema({"failure.reason.0": {}, "stage_2.accepted": {}})
    assert schema["properties"]["evidence"]["items"]["enum"] == ["failure.reason.0", "stage_2.accepted"]

