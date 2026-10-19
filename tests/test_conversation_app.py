"""Chat UI actions stay explicit and reopen the same local conversation."""

from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import conversation, conversation_store
from test_app import silicon_bytes, workbench


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def chat_app(runs_root, engine="vasp"):
    return AppTest.from_string(f"""
from pathlib import Path
from vasp_slurm_agent.agent import ModelSettings
from vasp_slurm_agent.conversation_app import panel
panel(Path({str(runs_root)!r}), {engine!r}, ModelSettings('codex'))
""", default_timeout=20).run()


@pytest.mark.parametrize("engine,example", [("vasp", "Si"), ("qe", "MgB2")])
def test_start_chat_copies_structure_and_reopens_without_a_model_call(tmp_path, monkeypatch, engine, example):
    runs = tmp_path / "runs"
    send = Mock()
    monkeypatch.setattr(conversation, "send", send)
    app = chat_app(runs, engine)
    assert not app.exception
    assert not app.file_uploader
    widget(app.button, "New conversation").click().run()
    widget(app.radio, "Chat structure").set_value("Example").run()
    assert widget(app.selectbox, "Chat example").value == example
    widget(app.button, "Start conversation").click().run()
    assert not app.exception
    folder = Path(app.session_state["active_conversation"])
    record = conversation_store.load(folder)
    assert record["engine"] == engine
    assert not record["messages"] and not record["runs"]
    assert record["source_sha256"]
    send.assert_not_called()
    reopened = chat_app(runs, engine)
    widget(reopened.selectbox, "Conversation").select(str(folder)).run()
    assert not reopened.exception
    assert reopened.chat_input
    send.assert_not_called()


def test_revision_history_survives_reopen_and_prepare_never_submits(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = files("vasp_slurm_agent").joinpath("examples", "Si.cif")
    folder = conversation_store.create(runs, "vasp", source)
    histories = []

    def respond(folder, message, settings, *, config=None):
        record = conversation_store.load(folder)
        histories.append(list(record["messages"]))
        ready = bool(record["messages"])
        record["messages"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Keep the cell fixed?" if not ready else "Ready to relax."},
        ])
        record["pending"] = {"kind": "vasp_task", "status": "ready" if ready else "needs_input",
                             "summary": "Relaxation", "questions": [] if ready else ["Keep the cell fixed?"],
                             "plan": {"operations": [], "stages": [{"task": "relax"}]}}
        return conversation_store.save(folder, record)

    def prepare(folder, run_dir, config=None):
        record = conversation_store.load(folder)
        run_dir.mkdir()
        record.update(pending=None, active_run=str(run_dir), runs=[str(run_dir)])
        return conversation_store.save(folder, record)

    monkeypatch.setattr(conversation, "send", Mock(side_effect=respond))
    prepared = Mock(side_effect=prepare)
    monkeypatch.setattr(conversation, "prepare", prepared)
    app = chat_app(runs)
    widget(app.selectbox, "Conversation").select(str(folder)).run()
    app.chat_input[0].set_value("Relax this structure.").run()
    assert not app.exception
    assert widget(app.button, "Prepare inputs").disabled
    assert len(app.chat_message) == 2
    prepared.assert_not_called()
    app = chat_app(runs)
    widget(app.selectbox, "Conversation").select(str(folder)).run()
    app.chat_input[0].set_value("Yes, keep the cell fixed.").run()
    assert not app.exception
    assert len(histories[-1]) == 2
    assert len(app.chat_message) == 4
    assert not widget(app.button, "Prepare inputs").disabled
    assert not any(path.name != ".conversations" for path in runs.iterdir())
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    prepared.assert_called_once()
    record = conversation_store.load(folder)
    assert Path(app.session_state["active_run"]) == Path(record["active_run"])
    assert record["pending"] is None
    assert not app.button or all(button.label != "Submit calculation" for button in app.button)


def test_engine_selector_does_not_mix_saved_conversations(tmp_path):
    source = files("vasp_slurm_agent").joinpath("examples", "Si.cif")
    vasp = conversation_store.create(tmp_path, "vasp", source)
    qe = conversation_store.create(tmp_path, "qe", source)
    for engine, expected, excluded in (("vasp", vasp, qe), ("qe", qe, vasp)):
        app = chat_app(tmp_path, engine)
        widget(app.selectbox, "Conversation").select(str(expected)).run()
        assert not app.exception
        assert app.session_state["active_conversation"] == str(expected)
        assert str(excluded) != widget(app.selectbox, "Conversation").value


def test_select_saved_chat_closes_new_conversation_form(tmp_path):
    source = files("vasp_slurm_agent").joinpath("examples", "Si.cif")
    folder = conversation_store.create(tmp_path, "vasp", source)
    app = chat_app(tmp_path)
    widget(app.button, "New conversation").click().run()
    assert app.file_uploader
    widget(app.selectbox, "Conversation").select(str(folder)).run()
    assert not app.exception
    assert not app.file_uploader
    assert app.chat_input


@pytest.mark.parametrize("outside_root", [False, True])
def test_existing_run_opens_chat_without_changing_inputs(workbench, outside_root):
    app, worker, runs_root = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    root = Path(app.session_state["active_run"])
    original = {name: (root / name).read_bytes() for name in ("run.json", "plan.json")}
    if outside_root:
        widget(app.text_input, "Run folder").set_value(str(runs_root.parent / "other-runs")).run()
    widget(app.button, "Open chat").click().run()
    assert not app.exception
    assert app.session_state["workspace_tab"] == "Chat"
    folder = Path(app.session_state["active_conversation"])
    assert folder.parent == runs_root / ".conversations"
    assert Path(widget(app.text_input, "Run folder").value) == runs_root
    record = conversation_store.load(folder)
    assert record["active_run"] == str(root)
    assert record["pending"] is None
    assert app.chat_input
    assert {name: (root / name).read_bytes() for name in original} == original
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_old_result_messages_remain_visible_as_earlier_results(tmp_path, monkeypatch):
    from vasp_slurm_agent import explanation

    source = files("vasp_slurm_agent").joinpath("examples", "Si.cif")
    folder = conversation_store.create(tmp_path, "vasp", source)
    run = tmp_path / "run"
    run.mkdir()
    record = conversation_store.load(folder)
    record.update(active_run=str(run), runs=[str(run)], messages=[
        {"role": "assistant", "content": "The earlier stage completed.", "context_sha256": "old"},
        {"role": "assistant", "content": "The current output needs review.", "context_sha256": "current"},
    ])
    conversation_store.save(folder, record)
    monkeypatch.setattr(explanation, "load_run_context", lambda root: {"context_sha256": "current"})
    app = chat_app(tmp_path)
    widget(app.selectbox, "Conversation").select(str(folder)).run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert len([item for item in app.caption if item.value == "Earlier results"]) == 1
