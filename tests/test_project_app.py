"""Project UI uses real local storage; model and process calls are explicit mocks."""

from importlib.resources import files
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import project_files, workflow
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_execution import ProjectExecution
from vasp_slurm_agent.workspace import Workspace
from test_project_agent import provider, reply, graph


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def workspace(monkeypatch):
    monkeypatch.delenv("DFT_AGENT_UI")
    monkeypatch.setattr(workflow, "SSHTransport", Mock(side_effect=AssertionError("No implicit SSH")))
    monkeypatch.setattr(workflow, "start_worker", Mock(side_effect=AssertionError("No implicit launch")))
    return Workspace()


def app():
    return AppTest.from_string("from vasp_slurm_agent.app import main\nmain()", default_timeout=30).run()


def seed(workspace, name="Silicon", model=True):
    pid = workspace.create_project(name)["id"]
    sid = workspace.create_session(pid)["id"]
    registry = ConnectionRegistry(workspace)
    if model:
        connection = registry.create("model", name + " model", {"provider": "codex"})
        registry.bind(pid, connection["id"])
    return pid, sid, registry


def test_default_entrypoint_project_lifecycle_preserves_files(workspace):
    ui = app()
    assert not ui.exception and not ui.chat_input
    widget(ui.button, "＋ New project").click().run()
    widget(ui.text_input, "Project name").set_value("Silicon").run()
    widget(ui.button, "Create project").click().run()
    assert not ui.exception and ui.chat_input[0].disabled
    pid = workspace.list_projects()[0]["id"]
    project_files.write_file(workspace, pid, "files/note.txt", "Keep this file.")
    widget(ui.text_input, "Project name").set_value("Silicon study").run()
    widget(ui.button, "Rename project").click().run()
    assert workspace.get_project(pid)["name"] == "Silicon study"
    widget(ui.button, "Archive project").click().run()
    assert not ui.exception and workspace.get_project(pid)["archived"]
    assert ui.chat_input[0].disabled
    widget(ui.button, "Restore project").click().run()
    widget(ui.text_input, "Type the project name to delete").set_value("Silicon study").run()
    widget(ui.button, "Delete project").click().run()
    assert not ui.exception and workspace.get_project(pid)["deleted"]
    assert project_files.read_file(workspace, pid, "files/note.txt")["content"] == "Keep this file."
    widget(ui.button, "Restore project").click().run()
    assert not ui.exception and not workspace.get_project(pid)["deleted"]


def test_switch_projects_isolates_sessions_files_and_model(workspace):
    first, sid, registry = seed(workspace)
    workspace.append_message(first, sid, "user", "First project question")
    project_files.write_file(workspace, first, "files/first.txt", "first")
    second, other, _ = seed(workspace, "Other", model=False)
    workspace.append_message(second, other, "user", "Other project question")
    ui = app()
    ui.button(key="project_open_" + first).click().run()
    widget(ui.button, "Files and runs").click().run()
    assert widget(ui.selectbox, "File").value == "files/first.txt"
    assert not ui.chat_input[0].disabled
    assert "First project question" in [item.value for item in ui.markdown]
    ui.button(key="project_open_" + second).click().run()
    assert not ui.exception and ui.chat_input[0].disabled
    assert not any(item.label in {"File", "Model"} for item in ui.selectbox)
    assert "First project question" not in [item.value for item in ui.markdown]
    assert "Other project question" in [item.value for item in ui.markdown]
    ui.button(key="project_open_" + first).click().run()
    assert ui.session_state[f"project_{first}_session"] == sid
    assert not ui.chat_input[0].disabled


def test_conversation_search_new_and_restart(workspace):
    pid, sid, _ = seed(workspace)
    workspace.append_message(pid, sid, "user", "Calculate phonons")
    second = workspace.create_session(pid, "Magnetism")["id"]
    workspace.append_message(pid, second, "user", "Compare magnetic order")
    ui = app()
    widget(ui.text_input, "Search conversations").set_value("phonons").run()
    assert ui.session_state[f"project_{pid}_session"] == sid
    widget(ui.button, "＋ New conversation").click().run()
    assert not ui.exception
    assert widget(ui.text_input, "Search conversations").value == ""
    new_id = ui.session_state[f"project_{pid}_session"]
    assert new_id not in {sid, second}
    restored = app()
    restored.button(key=f"project_{pid}_open_session_{sid}").click().run()
    assert "Calculate phonons" in [item.value for item in restored.markdown]


def test_all_projects_opens_project_outside_recent_twelve(workspace):
    first, sid, _ = seed(workspace, "Earlier silicon study", model=False)
    workspace.append_message(first, sid, "user", "An older project conversation")
    for number in range(12):
        seed(workspace, f"Recent study {number}", model=False)
    ui = app()
    assert not any(button.key == "project_open_" + first for button in ui.button)
    widget(ui.text_input, "Find a project").set_value("Earlier silicon").run()
    widget(ui.selectbox, "Project").select(first).run()
    widget(ui.button, "Open project").click().run()
    assert not ui.exception and not ui.error
    assert ui.session_state["project_selected"] == first
    assert ui.session_state[f"project_{first}_session"] == sid
    assert "An older project conversation" in [item.value for item in ui.markdown]


def test_saved_analysis_limits_and_evidence_survive_restart(workspace):
    pid, sid, _ = seed(workspace)
    limits = ["Do not rank energies from these different solver setups.",
              "Python execution completed; scientific results remain unverified."]
    workspace.append_message(pid, sid, "assistant", "Both calculations produced output.", metadata={"analysis": {
        "limits": limits, "evidence": ["run_a.run.process_success", "run_b.run.process_success"],
        "sources": [{"id": "first", "namespace": "run_a", "engine": "vasp", "status": "completed"},
                    {"id": "second", "namespace": "run_b", "engine": "python", "status": "completed_unverified"}]}})
    for ui in (app(), app()):
        assert not ui.exception and not ui.error
        assert all(limit in [item.value for item in ui.caption] for limit in limits)
        assert any(item.label == "Evidence" for item in ui.expander)
        assert "run_a.run.process_success" in ui.code[0].value
        assert "completed_unverified" in ui.dataframe[0].value["Status"].tolist()


def test_real_agent_route_persists_mixed_workflow_without_submission(workspace, monkeypatch):
    pid, sid, _ = seed(workspace)
    project_files.write_file(workspace, pid, "files/Si.cif", files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    # After the workflow is created, the first step is planned at once: its router asks a question here.
    clarify = {"action": "clarify", "reply": "Which exchange-correlation functional should the relaxation use?",
               "source_stage": None, "original_source": False}
    calls, remaining = provider(monkeypatch, [reply("workflow", workflow=graph()), clarify, reply("status")])
    ui = app()
    widget(ui.pills, "Mode").set_value("guided").run()
    ui.chat_input[0].set_value("Relax with VASP, then QE phonons.").run()
    assert not ui.exception and not ui.error
    result = workspace.list_workflows(pid)[0]
    assert [step["engine"] for step in result["steps"]] == ["vasp", "qe"]
    assert widget(ui.selectbox, "Workflow").value == result["id"]
    assert widget(ui.selectbox, "Step").value == "relax"
    assert [call[0] for call in calls] == ["dft_project_action", "dft_conversation"]
    assert any("exchange-correlation" in item.value for item in ui.markdown)
    ui.chat_input[0].set_value("What is the status?").run()
    assert not ui.exception and not remaining and len(calls) == 3
    assert len(workspace.get_session(pid, sid)["messages"]) == 4
    workflow.start_worker.assert_not_called()
    assert not workspace.list_runs(pid)


def test_first_run_checklist_guides_model_then_cluster(workspace, monkeypatch):
    pid, sid, registry = seed(workspace, model=False)
    ui = app()
    assert not ui.exception and ui.chat_input[0].disabled
    assert "Connect a model to plan calculations" in ui.chat_input[0].placeholder
    assert widget(ui.button, "Connect a model") and widget(ui.button, "Connect a cluster")
    widget(ui.button, "Connect a model").click().run()
    assert not ui.exception and ui.pills[0].value == "model"
    connection = registry.create("model", "Codex", {"provider": "codex"})
    registry.bind(pid, connection["id"])
    monkeypatch.setattr(ConnectionRegistry, "_probe_model", lambda self, item: {
        "auth_status": "available", "connection_status": "connected", "detail": "ok"})
    registry.probe(connection["id"])
    ui = app()
    assert not ui.chat_input[0].disabled
    # The model step is done: its button disappears and runnable examples appear.
    assert not any(button.label == "Connect a model" for button in ui.button)
    assert any(button.label.startswith("Relax silicon") for button in ui.button)
    widget(ui.button, "Connect a cluster").click().run()
    assert not ui.exception and ui.pills[0].value == "hpc"


def test_task_card_uses_the_only_matching_environment_without_asking(workspace):
    pid, sid, registry = seed(workspace)
    hpc = registry.create("hpc", "Fixture cluster", {"host": "fixture.invalid", "user": "researcher",
                          "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    registry.bind(pid, hpc["id"])
    environment = registry.create_environment(hpc["id"], "VASP", "vasp", {
        "vasp_command": "srun vasp_std", "potcar_root": "/licensed", "partition": "cpu"})
    project_files.write_file(workspace, pid, "files/Si.cif", files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    graph = workspace.create_workflow(pid, sid, "SCF", "Calculate SCF", [{"id": "scf", "name": "SCF", "engine": "vasp",
        "spec": {"goal": "Calculate SCF", "source": {"kind": "file", "path": "files/Si.cif"}}}])
    ui = app()
    assert not ui.exception
    assert not any(button.label == "Use environment" for button in ui.button)
    assert widget(ui.button, "Plan step") and not widget(ui.button, "Plan step").disabled
    attempt = ProjectExecution(workspace, registry).latest(pid, graph["id"], "scf")
    assert attempt["environment_id"] == environment["id"] and attempt["state"] == "draft"
    workflow.start_worker.assert_not_called()


def test_revoked_model_and_archive_remove_chat_capability(workspace):
    pid, sid, registry = seed(workspace)
    ui = app()
    registry.revoke(registry.project_connections(pid)[0]["id"])
    ui.run()
    assert not ui.exception and ui.chat_input[0].disabled
    workspace.archive_project(pid)
    ui.run()
    assert not ui.exception and ui.chat_input[0].disabled
    assert widget(ui.button, "＋ New conversation").disabled
    assert not workspace.get_session(pid, sid)["messages"]


def test_file_editor_rejects_stale_write_and_reloads(workspace):
    pid, _, _ = seed(workspace)
    first = project_files.write_file(workspace, pid, "files/note.txt", "version one")
    ui = app()
    widget(ui.button, "Files and runs").click().run()
    widget(ui.text_area, "Contents").set_value("user edit").run()
    project_files.write_file(workspace, pid, "files/note.txt", "external edit", expected_sha256=first["sha256"])
    widget(ui.button, "Save file").click().run()
    assert not ui.exception and any("file changed" in item.value.lower() for item in ui.error)
    assert project_files.read_file(workspace, pid, "files/note.txt")["content"] == "external edit"
    widget(ui.button, "Reload file").click().run()
    assert widget(ui.text_area, "Contents").value == "external edit"
    widget(ui.text_area, "Contents").set_value("final edit").run()
    widget(ui.button, "Save file").click().run()
    assert not ui.exception and project_files.read_file(workspace, pid, "files/note.txt")["content"] == "final edit"


def test_saved_file_edit_requires_explicit_apply(workspace):
    pid, sid, _ = seed(workspace)
    saved = project_files.write_file(workspace, pid, "files/note.txt", "before")
    workspace.append_message(pid, sid, "assistant", "Review the file change.", metadata={"file_edit": {
        "path": "files/note.txt", "content": "after", "expected_sha256": saved["sha256"]}})
    ui = app()
    assert project_files.read_file(workspace, pid, "files/note.txt")["content"] == "before"
    widget(ui.button, "Apply change").click().run()
    assert not ui.exception
    assert project_files.read_file(workspace, pid, "files/note.txt")["content"] == "after"
    assert not any(button.label == "Apply change" for button in ui.button)


def test_identical_files_keep_separate_unsaved_edits(workspace):
    pid, _, _ = seed(workspace)
    project_files.write_file(workspace, pid, "files/first.txt", "same text")
    project_files.write_file(workspace, pid, "files/second.txt", "same text")
    ui = app()
    widget(ui.button, "Files and runs").click().run()
    widget(ui.selectbox, "File").select("files/first.txt").run()
    widget(ui.text_area, "Contents").set_value("first file draft").run()
    widget(ui.selectbox, "File").select("files/second.txt").run()
    assert widget(ui.text_area, "Contents").value == "same text"
    assert widget(ui.button, "Save file").disabled
    assert project_files.read_file(workspace, pid, "files/first.txt")["content"] == "same text"


def test_upload_adds_inputs_only_to_current_project_and_rejects_potcar(workspace):
    pid, _, _ = seed(workspace)
    ui = app()
    widget(ui.button, "Files and runs").click().run()
    ui.file_uploader[0].upload("Si.cif", files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes()).run()
    widget(ui.button, "Import files").click().run()
    assert not ui.exception
    assert project_files.read_file(workspace, pid, "files/Si.cif")["content"]
    ui = app()
    widget(ui.button, "Files and runs").click().run()
    ui.file_uploader[0].upload("POTCAR", b"private potential").run()
    widget(ui.button, "Import files").click().run()
    assert not ui.exception and ui.error
    assert not (workspace.project_paths(pid)["files"] / "POTCAR").exists()


def test_prepared_run_requires_review_and_explicit_submit(workspace, monkeypatch):
    from test_conversation import provider as plan_provider, route, plan, SETTINGS

    pid, sid, registry = seed(workspace)
    hpc = registry.create("hpc", "Fixture cluster", {"host": "fixture.invalid", "user": "researcher",
                          "remote_base": "/scratch/researcher", "scheduler": "slurm"})
    registry.bind(pid, hpc["id"])
    environment = registry.create_environment(hpc["id"], "VASP", "vasp", {
        "vasp_command": "srun vasp_std", "potcar_root": "/licensed", "partition": "cpu"})
    project_files.write_file(workspace, pid, "files/Si.cif", files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    graph = workspace.create_workflow(pid, sid, "SCF", "Calculate SCF", [{"id": "scf", "name": "SCF", "engine": "vasp",
        "spec": {"goal": "Calculate SCF", "source": {"kind": "file", "path": "files/Si.cif"}, "environment_id": environment["id"]}}])
    execution = ProjectExecution(workspace, registry)
    plan_provider(monkeypatch, [("dft_conversation", route()), ("dft_task", plan("vasp"))])
    execution.draft_step(pid, graph["id"], "scf", SETTINGS)
    ui = app()
    widget(ui.button, "Prepare inputs").click().run()
    assert not ui.exception and widget(ui.button, "Submit").disabled
    workflow.start_worker.assert_not_called()
    attempt = execution.latest(pid, graph["id"], "scf")
    assert attempt["state"] == "prepared"
    widget(ui.button, "Files and runs").click().run()
    widget(ui.radio, "Context").set_value("Results").run()
    assert not ui.exception
    assert not any(button.label in {"Submit calculation", "Resume monitoring", "Open chat"} for button in ui.button)
    widget(ui.checkbox, "I reviewed the inputs and resources.").check().run()
    assert not widget(ui.button, "Submit").disabled
    workflow.start_worker.assert_not_called()
    prepared_run = workspace.get_run(pid, attempt["run_id"])
    incar = workspace.resolve_path(pid, prepared_run["relative_path"] + "/01_scf/inputs/INCAR")
    incar.write_text(incar.read_text() + "\n")
    ui.run()
    assert widget(ui.button, "Submit").disabled
    assert not widget(ui.checkbox, "I reviewed the inputs and resources.").value
    widget(ui.checkbox, "I reviewed the inputs and resources.").check().run()
    launched = Mock(return_value=1234)
    monkeypatch.setattr(workflow, "start_worker", launched)
    widget(ui.button, "Submit").click().run()
    assert not ui.exception and not ui.error
    launched.assert_called_once()
    ui.run()
    launched.assert_called_once()
    assert execution.latest(pid, graph["id"], "scf")["state"] == "active"


def test_context_can_hide_and_connections_use_secure_panel(workspace):
    pid, _, _ = seed(workspace)
    project_files.write_file(workspace, pid, "files/note.txt", "text")
    ui = app()
    assert not any(item.label == "File" for item in ui.selectbox)
    widget(ui.button, "Files and runs").click().run()
    widget(ui.button, "Files and runs").click().run()
    assert not any(item.label == "File" for item in ui.selectbox)
    widget(ui.button, "Files and runs").click().run()
    assert widget(ui.selectbox, "File").value == "files/note.txt"
    widget(ui.button, "Connections").click().run()
    assert not ui.exception and ui.pills[0].value == "model"
    assert not ui.chat_input


def test_legacy_route_and_back_preserve_default(workspace):
    seed(workspace)
    ui = app()
    widget(ui.button, "Settings").click().run()
    widget(ui.button, "Open classic workbench").click().run()
    assert not ui.exception and widget(ui.radio, "Engine").value == "vasp"
    widget(ui.button, "Back to projects").click().run()
    assert not ui.exception and any(item.label == "＋ New project" for item in ui.button)


def test_python_results_show_execution_scope_and_raw_downloads(workspace, tmp_path):
    import shlex
    import sys
    from vasp_slurm_agent import script_workflow
    from test_script_workflow import config, Transport, complete

    pid, _, _ = seed(workspace)
    project_files.write_file(workspace, pid, "files/test.py", "from pathlib import Path\nPath('result.json').write_text('{\"sum\": 45}')\n")
    source = workspace.resolve_path(pid, "files/test.py")
    root = workspace.project_paths(pid)["runs"] / "python-test"
    state = script_workflow.prepare_run(root, config(), engine="python", files={"test.py": source},
        entrypoint="test.py", command=shlex.quote(sys.executable), outputs=[{"path": "result.json", "kind": "json"}])
    complete((root, Transport(tmp_path / "remote"), state))
    workspace.register_run(pid, root)
    ui = app()
    widget(ui.button, "Files and runs").click().run()
    widget(ui.radio, "Context").set_value("Results").run()
    assert not ui.exception and not ui.error
    assert "completed unverified" in [item.value for item in ui.markdown]
    assert any("scientific results remain unverified" in item.value for item in ui.caption)
    assert any(item.label == "result.json" for item in ui.download_button)
    assert not any(item.label == "Submit calculation" for item in ui.button)
