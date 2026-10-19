from copy import deepcopy
import json
import zipfile

import pytest

from vasp_slurm_agent import agent, batch, batch_context, explanation, workflow
from test_batch import setup, accept_child


@pytest.fixture
def accepted_batch(setup, monkeypatch):
    source, root, config = setup
    state = batch.prepare_batch(source, root, config, ["scf"], [
        {"label": "PBE"}, {"label": "U4", "parameters": {"hubbard_u": {"Fe": {"l": 2, "u": 4, "j": 0}}}},
    ])
    results = {}
    for index, member in enumerate(state["runs"]):
        accept_child(root / member["folder"], -10 - index, results)
    monkeypatch.setattr(explanation, "analyze_outputs", lambda path, *args: deepcopy(
        results[(path / "vasprun.xml").read_text()]))
    batch._refresh(root, batch.read_batch(root))
    return root


def files(root):
    return {str(path.relative_to(root)): workflow._digest(path) for path in root.rglob("*") if path.is_file()}


def test_batch_context_reparses_members_without_writing_or_ranking(accepted_batch, monkeypatch):
    root = accepted_batch
    before = files(root)
    monkeypatch.setattr(batch, "summarize_batch", lambda *a: pytest.fail("Reading must not refresh a batch"))
    context = explanation.load_run_context(root)
    assert context["workflow"] == "batch" and context["engine"] == "vasp"
    assert context["status"] == "succeeded"
    assert context["facts"]["batch.accepted_members"]["value"] == 2
    assert context["facts"]["member_1.stage_1.energy"]["value"] == -10
    assert context["facts"]["member_2.stage_1.energy"]["value"] == -11
    assert context["facts"]["member_1.stage_1.band_gap"]["value"] == 1.2
    assert any("convergence" in limit for limit in context["limits"])
    assert "No cross-variant energy ranking" in context["facts"]["batch.comparison_scope"]["value"]
    assert not any("rank" in key or "relative_energy" in key for key in context["facts"])
    assert all(item["id"].startswith("member_") for item in context["artifacts"])
    assert files(root) == before


def test_changed_member_evidence_is_removed_and_invalidates_batch_context(accepted_batch):
    before = explanation.load_run_context(accepted_batch)
    state = batch.read_batch(accepted_batch)
    member = accepted_batch / state["runs"][0]["folder"]
    path = member / "01_scf/outputs/OUTCAR"
    path.write_text(path.read_text() + "\nchanged")
    after = explanation.load_run_context(accepted_batch)
    assert after["context_sha256"] != before["context_sha256"]
    assert after["status"] == "needs_attention"
    assert after["facts"]["batch.accepted_members"]["value"] == 1
    assert not after["facts"]["member_1.stage_1.accepted"]["value"]
    assert "member_1.stage_1.energy" not in after["facts"]
    assert after["facts"]["member_2.stage_1.energy"]["value"] == -11


def test_cached_summary_cannot_supply_fabricated_results(accepted_batch):
    before = explanation.load_run_context(accepted_batch)
    workflow._write(accepted_batch / "summary.json", {"rows": [{"energy_ev": -99999, "rank": 1}]})
    assert explanation.load_run_context(accepted_batch) == before


@pytest.mark.parametrize("target", ["batch-plan.json", "source/POSCAR", "config.json"])
def test_changed_batch_inputs_fail_closed(accepted_batch, target):
    path = accepted_batch / target
    path.write_text(path.read_text() + "\nchanged")
    with pytest.raises(ValueError):
        explanation.load_run_context(accepted_batch)


def test_member_list_is_checked_against_manifest(accepted_batch):
    state = batch.read_batch(accepted_batch)
    state["runs"][0]["folder"] = "../other-run"
    workflow._write(accepted_batch / "batch.json", state)
    with pytest.raises(ValueError, match="frozen plan"):
        batch_context.load_batch_context(accepted_batch)


def test_batch_state_changes_invalidate_context_hash(accepted_batch):
    before = explanation.load_run_context(accepted_batch)
    state = batch.read_batch(accepted_batch)
    state["cancel_requested"] = True
    workflow._write(accepted_batch / "batch.json", state)
    after = explanation.load_run_context(accepted_batch)
    assert before["context_sha256"] != after["context_sha256"]
    assert before["facts"] == after["facts"]


def test_result_explanation_accepts_prefixed_verified_facts(accepted_batch, monkeypatch):
    seen = []

    def reply(payload, *a, **kw):
        seen.append(json.loads(payload))
        return json.dumps({"answer": "The PBE energy is {{member_1.stage_1.energy_per_atom}}.",
                           "evidence": ["member_1.stage_1.energy_per_atom"], "limits": [], "next_steps": []})

    monkeypatch.setattr(agent, "_request_structured", reply)
    response = explanation.explain_run(accepted_batch, agent.ModelSettings("codex"), "What is the PBE energy?")
    assert "-5.0" in response["answer"]
    assert seen[0]["context"]["workflow"] == "batch"
    assert "/scratch" not in json.dumps(seen)


def test_batch_change_during_model_request_rejects_answer(accepted_batch, monkeypatch):
    def reply(*a, **kw):
        state = batch.read_batch(accepted_batch)
        state["cancel_requested"] = True
        workflow._write(accepted_batch / "batch.json", state)
        return json.dumps({"answer": "The variant results passed their checks.",
                           "evidence": ["batch.accepted_members"], "limits": [], "next_steps": []})

    monkeypatch.setattr(agent, "_request_structured", reply)
    with pytest.raises(agent.AgentError, match="changed"):
        explanation.explain_run(accepted_batch, agent.ModelSettings("codex"))


def test_batch_bundle_retains_conversation(accepted_batch):
    transcript = {"schema_version": 1, "messages": [{"role": "user", "content": "Compare the variants"}]}
    workflow._write(accepted_batch / "conversation.json", transcript)
    with zipfile.ZipFile(batch.bundle_batch(accepted_batch)) as archive:
        assert json.loads(archive.read("conversation.json")) == transcript
        assert "config.json" not in archive.namelist()
