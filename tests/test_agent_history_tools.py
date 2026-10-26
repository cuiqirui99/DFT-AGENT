"""Conversation retrieval is bound to the current project and preserves requirements."""

import json

import pytest

from vasp_slurm_agent.mcp_server import ProjectTools
from vasp_slurm_agent.workspace import Workspace


@pytest.fixture
def history(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create_project("History")
    session = workspace.create_session(project["id"])
    tools = ProjectTools(workspace, project["id"], session["id"])
    return workspace, project["id"], session["id"], tools


def test_history_pages_are_chronological_and_complete(history):
    workspace, pid, sid, tools = history
    requirement = "context " * 350 + "Use QE only, ecutwfc=73 Ry."
    workspace.append_message(pid, sid, "user", requirement)
    for index in range(1, 25):
        workspace.append_message(pid, sid, "assistant" if index % 2 else "user", str(index))
    latest = tools.call("conversation_history", {})
    assert [item["index"] for item in latest["messages"]] == list(range(15, 25))
    assert latest["total"] == 25 and latest["next_before"] == 15
    older = tools.call("conversation_history", {"before": latest["next_before"], "limit": 20})
    assert [item["index"] for item in older["messages"]] == list(range(15))
    assert older["messages"][0] == {"index": 0, "role": "user", "content": requirement}
    assert older["next_before"] is None
    assert tools.conversation_history(before=0)["messages"] == []


def test_history_character_budget_keeps_whole_messages(history):
    workspace, pid, sid, tools = history
    for text in ("first", "a" * 25000, "b" * 25000):
        workspace.append_message(pid, sid, "user", text)
    page = tools.conversation_history(limit=20)
    assert len(page["messages"]) == 1 and len(page["messages"][0]["content"]) == 25000
    assert page["next_before"] == 2
    older = tools.conversation_history(before=2)
    assert [item["index"] for item in older["messages"]] == [0, 1]
    workspace.append_message(pid, sid, "user", "c" * 45000)
    assert len(tools.conversation_history()["messages"][0]["content"]) == 45000


@pytest.mark.parametrize("arguments", [{"before": -1}, {"before": 1}, {"limit": 0}, {"limit": 21},
                                      {"limit": True}, {"session_id": "other"}])
def test_history_rejects_invalid_or_foreign_selection(history, arguments):
    with pytest.raises(ValueError):
        history[-1].call("conversation_history", arguments)


def test_history_session_binding_and_redaction(history):
    workspace, pid, sid, tools = history
    workspace.append_message(pid, sid, "user", "Legacy imported message")
    with workspace._transaction() as db:
        db.execute("UPDATE messages SET content=? WHERE project_id=? AND session_id=?",
                   ("API_KEY=sk-test-1234567890abcdefghijk", pid, sid))
    assert "1234567890abcdefghijk" not in json.dumps(tools.conversation_history())
    with pytest.raises(ValueError, match="bound"):
        ProjectTools(workspace, pid).conversation_history()
    other = workspace.create_project("Other")
    with pytest.raises(ValueError):
        ProjectTools(workspace, other["id"], sid).conversation_history()
