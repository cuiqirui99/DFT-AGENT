"""Regression checks for recoverable chat errors and continuation boundaries."""

import json

import pytest

from vasp_slurm_agent import agent, conversation, conversation_store as store, workflow
from test_conversation import SETTINGS, config, no_remote, plan, provider, route, source
from test_continuation import parent
from test_explanation import accepted_run


@pytest.mark.parametrize("settings", [None, agent.ModelSettings("responses")])
def test_missing_model_preserves_revision_and_disables_previous_draft(source, tmp_path, monkeypatch, settings):
    monkeypatch.delenv("DFT_AGENT_MODEL", raising=False)
    folder = conversation.start(tmp_path / "runs", "vasp", source)
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route()), ("dft_task", plan("vasp", 2)),
        ("dft_conversation", route()), ("dft_task", plan("vasp", 3)),
    ])
    first = conversation.send(folder, "Double the cell and run SCF.", SETTINGS)
    assert first["pending"]["status"] == "ready"
    failed = conversation.send(folder, "Triple it instead.", settings)
    assert failed["pending"] is None
    assert failed["messages"][-1]["error"]
    assert failed["messages"][-2]["content"] == "Triple it instead."
    assert len(seen) == 2
    with pytest.raises(ValueError, match="review a plan"):
        conversation.prepare(folder, tmp_path / "runs" / "stale", config("vasp"))
    retried = conversation.send(folder, "Retry that change.", SETTINGS)
    assert not remaining
    assert retried["pending"]["status"] == "ready"
    assert "Triple it instead." in json.dumps(seen[-1][1]["history"])


def test_discussing_pending_plan_after_run_does_not_use_old_result_reader(parent, tmp_path, monkeypatch):
    root, _ = parent
    folder = conversation.open_run(root, tmp_path)
    discussion = {**route("discuss"), "reply": "The proposed calculation uses a doubled cell."}
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route("continue", 0)), ("dft_task", plan("vasp", 2)),
        ("dft_conversation", discussion),
    ])
    proposed = conversation.send(folder, "Double the accepted structure and run SCF.", SETTINGS)
    def rejected_reader(*args, **kwargs):
        pytest.fail("Draft questions must not be answered from the old run's result facts")
    monkeypatch.setattr(conversation.explanation, "explain_run", rejected_reader)
    answered = conversation.send(folder, "Does the proposed calculation use the doubled cell?", SETTINGS)
    assert not remaining
    assert answered["pending"] == proposed["pending"]
    assert answered["messages"][-1]["action"] == "discuss"
    assert answered["messages"][-1]["content"] == discussion["reply"]
    assert seen[-1][1]["pending"]["operations"] == [{"type": "supercell", "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}]


@pytest.mark.parametrize("action", ["send", "prepare"])
def test_busy_conversation_rejects_duplicate_actions_without_state_change(action, source, tmp_path, monkeypatch):
    folder = conversation.start(tmp_path / "runs", "vasp", source)
    before = (folder / "conversation.json").read_bytes()
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: pytest.fail("No duplicate model call"))
    with (folder / ".action.lock").open("a+b") as stream:
        workflow._acquire(stream, blocking=False)
        try:
            with pytest.raises(agent.AgentError, match="busy"):
                if action == "send":
                    conversation.send(folder, "SCF", SETTINGS)
                else:
                    conversation.prepare(folder, tmp_path / "runs" / "new")
        finally:
            workflow._release(stream)
    assert (folder / "conversation.json").read_bytes() == before
    assert not (tmp_path / "runs" / "new").exists()


def test_new_continuation_excludes_completed_edits_but_keeps_its_own_revisions(parent, tmp_path, monkeypatch):
    root, _ = parent
    folder = conversation.open_run(root, tmp_path)
    record = store.load(folder)
    record["messages"] = [
        {"role": "user", "content": "Make the old 3x3x3 supercell and replace site 0 with Ge."},
        {"role": "assistant", "content": "Inputs prepared.", "action": "prepared"},
    ]
    record["latest_plan"] = plan("vasp", 3)
    store.save(folder, record)
    seen, remaining = provider(monkeypatch, [
        ("dft_conversation", route("continue", 0)), ("dft_task", plan("vasp", 2)),
        ("dft_conversation", route("plan")), ("dft_task", plan("vasp", 4)),
    ])
    conversation.send(folder, "From the accepted result, make an additional 2x1x1 cell and SCF.", SETTINGS)
    first_history = json.dumps(seen[1][1]["history"])
    assert "old 3x3x3" not in first_history and "replace site 0" not in first_history
    assert "already contains all completed edits" in first_history
    conversation.send(folder, "Make that additional cell 4x1x1 instead.", SETTINGS)
    revised_history = json.dumps(seen[-1][1]["history"])
    assert "additional 2x1x1" in revised_history
    assert "old 3x3x3" not in revised_history
    assert not remaining
