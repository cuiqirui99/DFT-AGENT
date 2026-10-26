"""Project workspace over the existing planning and calculation tools."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import streamlit as st

from . import __version__, agent_runner, project_agent, project_files
from .connections import ConnectionRegistry
from .i18n import LANGUAGE_NAMES, set_language, t
from .project_execution import ProjectExecution
from .secret_guard import redact
from .settings import load_settings, save_settings
from .ui_style import LOGO, chip, tag
from .workspace import Workspace


def _error(error):
    st.error(redact(str(error)))


_STATE_LABELS = {"planning": "planning", "needs_input": "needs your answer", "ready": "plan ready",
                 "draft": "draft", "preparing": "preparing", "prepared": "inputs prepared, review them",
                 "authorized": "authorized", "starting": "starting", "active": "running", "succeeded": "finished",
                 "failed": "failed", "needs_attention": "needs attention", "cancelled": "cancelled",
                 "superseded": "superseded", "pending": "pending"}


# Codex CLI and Claude Code connections start in agent mode: the CLI's own loop
# calls the project's tools. API providers always use the guided mode.
_DEFAULT_MODE = "agent"


def _agent_actions(message):
    metadata = message.get("metadata") or {}
    calls = metadata.get("tool_calls") or []
    if metadata.get("mode") != "agent" or not calls:
        return
    with st.expander(t("Agent actions") + f" ({len(calls)})"):
        for call in calls:
            status = call.get("status") or ""
            st.markdown(f"`{call.get('tool', '')}` {redact(str(call.get('arguments') or ''))[:200]}"
                        + (f" · {t(status)}" if status else ""))
            if call.get("result"):
                st.caption(redact(str(call["result"]))[:300])


def _step_elements(workspace, identifier, step):
    source = step["spec"].get("source") or {}
    if source.get("kind") != "file" or not source.get("path"):
        return []
    try:
        from .structures import _load
        _, _, structure, _ = _load(workspace.resolve_path(identifier, source["path"], must_exist=True))
        return sorted({site.specie.symbol for site in structure})
    except Exception:
        return []


def _pseudopotential_check(workspace, registry, identifier, step, environment_id, prefix, proposal=None):
    """A QE step needs a UPF for every element; public libraries can be installed here."""
    try:
        environment = registry.environment(environment_id)
        requirements = _library_requirements(workspace, identifier, step, proposal)
        plan = (proposal or {}).get("plan") or proposal or {}
        stages = plan.get("stages", [])
        soc = any(item[1] for item in requirements) or any(
            (json.loads(stage.get("parameters_json", "{}")).get("namelists") or {}).get("system", {}).get("lspinorb") is True
            for stage in stages)
    except (ValueError, TypeError) as error:
        st.caption(redact(str(error)))
        return False
    elements = sorted({element for symbols, _, _ in requirements for element in symbols})
    present = environment["config"].get("qe_pseudopotentials") or {}
    missing = [element for element in elements if element not in present or soc and not (
        isinstance(present[element], dict) and (present[element].get("fully_relativistic") is True
                                               or present[element].get("relativistic") == "full"))]
    if not missing:
        return True
    st.info(t("This step needs pseudopotentials for {elements}. Install the public PBE library into the environment's UPF folder.").format(elements=", ".join(missing)))
    options = ["pseudodojo-fr"] if soc else ["sssp", "pseudodojo", "pseudodojo-fr"]
    library = st.selectbox(t("Library"), options, key=prefix + "upf_library_" + ("soc" if soc else "scalar"),
                           format_func=lambda value: {"sssp": "SSSP 1.3 PBE efficiency", "pseudodojo": "PseudoDojo 0.4 PBE (norm-conserving)",
                                                      "pseudodojo-fr": "PseudoDojo 0.4 PBE fully relativistic (spin-orbit)"}[value])
    if st.button(t("Install public UPFs"), key=prefix + "upf_install"):
        try:
            with st.spinner(t("Installing pseudopotentials on the cluster…")):
                registry.install_pseudopotentials(environment_id, missing, library=library)
            st.rerun()
        except Exception as error:
            st.error(redact(str(error)))
    return False


def _library_requirements(workspace, identifier, step, proposal=None):
    plan = (proposal or {}).get("plan") or proposal or {}
    sites = ((plan.get("preview") or {}).get("output") or {}).get("sites") or []
    elements = sorted({site["element"] for site in sites}) if sites else _step_elements(workspace, identifier, step)
    methods = set()
    for stage in plan.get("stages", []):
        parameters = json.loads(stage.get("parameters_json", "{}"))
        methods.add((bool(parameters.get("soc", False)), parameters.get("functional", "PBE")))
    return [(elements, soc, functional) for soc, functional in sorted(methods or {(False, "PBE")})]


def _library_check(workspace, registry, identifier, step, environment_id, prefix, proposal=None):
    """ABINIT, CP2K and Elk steps need their pinned library files in the environment's library folder."""
    try:
        missing = []
        for elements, soc, functional in _library_requirements(workspace, identifier, step, proposal):
            status = registry.library_status(environment_id, elements, soc=soc, functional=functional)
            if status["missing"]:
                missing.append((elements, soc, functional, status))
    except ValueError as error:
        st.caption(redact(str(error)))
        return False
    if not missing:
        return True
    label = {"abinit": t("pseudopotentials"), "cp2k": t("basis sets and pseudopotentials"), "elk": t("species files")}[step["engine"]]
    st.info(t("This step needs {library} for {elements}. Install the pinned files into the environment's library folder.").format(
        library=label, elements=", ".join(sorted({element for _, _, _, status in missing for element in status["missing"]}))))
    if step["engine"] == "abinit":
        st.caption(" · ".join(functional + (" + SOC" if soc else "") for _, soc, functional, _ in missing))
    if st.button(t("Install library files"), key=prefix + "library_install"):
        try:
            with st.spinner(t("Installing library files on the cluster…")):
                for elements, soc, functional, _ in missing:
                    options = {"soc": soc, "functional": functional} if soc or functional != "PBE" else {}
                    registry.install_engine_library(environment_id, elements, **options)
            st.rerun()
        except Exception as error:
            st.error(redact(str(error)))
    return False


def _setup_status(registry, project_id):
    """What the first-run checklist shows: a model, a cluster and its software."""
    connections = [item for item in registry.project_connections(project_id) if not item["revoked"] and not item["deleted"]]
    models = [item for item in connections if item["kind"] == "model"]
    clusters = [item for item in connections if item["kind"] == "hpc"]
    bound = {item["id"] for item in clusters}
    environments = [item for item in registry.environments() if item["connection_id"] in bound]
    return {"models": models, "clusters": clusters, "environments": environments}


_STATE_KIND = {"succeeded": "ok", "finished": "ok", "ready": "accent", "prepared": "accent", "authorized": "accent",
               "active": "accent", "running": "accent", "failed": "bad", "cancelled": "bad", "superseded": "bad"}


def _step_card(number, title, status, *, done=False, current=False):
    import html
    kind = "done" if done else "now" if current else ""
    mark = "✓" if done else str(number)
    st.html(f'<div class="dfa-step {kind}"><div class="n">{mark}</div><div><div class="t">{html.escape(title)}</div>'
            f'<div class="s">{html.escape(status)}</div></div></div>')


def _onboarding(registry, project, status):
    identifier = project["id"]
    prefix = f"project_{identifier}_welcome_"
    models = [item for item in status["models"] if item["connection_status"] == "connected"]
    clusters = [item for item in status["clusters"] if item["connection_status"] == "connected"]
    st.html('<div class="dfa-hello">' + t("Set up this project") + '</div><div class="dfa-sub">'
            + t("Connect a model and a cluster, then add a structure.") + "</div>")
    first, second, third = st.columns(3, gap="medium")
    with first:
        if models:
            _step_card(1, t("Model"), models[0]["name"], done=True)
        elif status["models"]:
            _step_card(1, t("Model"), status["models"][0]["name"] + " · " + t("not tested yet"), current=True)
            _go_to_connections("model", identifier, prefix + "model", t("Check model"))
        else:
            _step_card(1, t("Model"), t("Codex CLI, Claude Code CLI or an API key. Uses your own quota."), current=True)
            _go_to_connections("model", identifier, prefix + "model", t("Connect a model"))
    with second:
        if clusters:
            software = ", ".join(sorted({item["engine"].upper() for item in status["environments"]}))
            _step_card(2, t("Cluster"), clusters[0]["settings"]["host"] + (f" · {software}" if software else ""), done=True)
            if not status["environments"]:
                _go_to_connections("hpc", identifier, prefix + "software", t("Set up software"))
        elif status["clusters"]:
            _step_card(2, t("Cluster"), status["clusters"][0]["settings"]["host"] + " · " + t("not connected yet"), current=bool(models))
            _go_to_connections("hpc", identifier, prefix + "cluster", t("Finish cluster sign-in"))
        else:
            _step_card(2, t("Cluster"), t("SSH address and password or key. Optional for planning."), current=bool(models))
            _go_to_connections("hpc", identifier, prefix + "cluster", t("Connect a cluster"))
    with third:
        _step_card(3, t("Structure"), t("Upload a structure file, then describe the calculation below."),
                   current=bool(models and clusters))
        if st.button(t("Add files"), key=prefix + "files", use_container_width=True, icon=":material/upload_file:"):
            st.session_state[f"project_{identifier}_show_context"] = True
            st.session_state[f"project_{identifier}_context_page"] = "Files"
            st.rerun()
    if models:
        st.caption(t("Try an example:"))
        examples = [t("Relax silicon with PBE, then calculate its band structure and DOS."),
                    t("Compare the ferromagnetic and antiferromagnetic states of this structure."),
                    t("What can you do with this project?")]
        columns = st.columns(len(examples))
        for column, example in zip(columns, examples):
            if column.button(example, key=prefix + "example_" + str(abs(hash(example)) % 10000), use_container_width=True):
                st.session_state[f"project_{identifier}_send"] = example
                st.rerun()


def _select(label, records, key, *, title="name", preferred=None, optional=False):
    values = {item["id"]: item for item in records}
    if not values:
        return None
    options = ([None] if optional else []) + list(values)
    if not options:
        return None
    requested = st.session_state.pop(key + "_requested", None)
    if requested in values:
        st.session_state[key] = requested
    elif key not in st.session_state or st.session_state.get(key) not in options:
        st.session_state[key] = preferred if preferred in options else options[0]
    return st.selectbox(label, options, key=key,
                        format_func=lambda item: values[item][title] if item else "None")


def _project_controls(workspace, project):
    identifier = project["id"]
    prefix = f"project_{identifier}_"
    with st.expander(t("Manage project")):
        if project["archived"] or project["deleted"]:
            st.caption(t("Deleted") if project["deleted"] else t("Archived"))
            if st.button(t("Restore project"), key=prefix + "restore"):
                workspace.restore_project(identifier)
                st.rerun()
            return
        name = st.text_input(t("Project name"), project["name"], key=prefix + "name")
        if st.button(t("Rename project"), key=prefix + "rename", disabled=not name.strip() or name == project["name"]):
            workspace.rename_project(identifier, name)
            st.rerun()
        if st.button(t("Archive project"), key=prefix + "archive"):
            workspace.archive_project(identifier)
            st.session_state["project_show_inactive_requested"] = True
            st.rerun()
        confirmation = st.text_input(t("Type the project name to delete"), key=prefix + "confirm_delete")
        st.caption(t("Files are kept. Deleted projects can be restored."))
        if st.button(t("Delete project"), key=prefix + "delete", disabled=confirmation != project["name"]):
            workspace.delete_project(identifier, confirmation=confirmation)
            st.session_state["project_show_inactive_requested"] = True
            st.rerun()


def _sidebar(workspace):
    st.html(f'<div class="dfa-brand">{LOGO}<span>DFT Agent</span><small>{__version__}</small></div>')
    if st.button(t("＋ New project"), key="project_create_page", use_container_width=True):
        st.session_state["project_page"] = "New project"
        st.rerun()
    projects = workspace.list_projects()
    if requested := st.session_state.pop("project_selected_requested", None):
        st.session_state["project_selected"] = requested
    identifier = st.session_state.get("project_selected")
    if identifier is None and projects:
        identifier = projects[0]["id"]
        st.session_state["project_selected"] = identifier
    st.caption(t("PROJECTS"))
    for item in projects[:12]:
        if st.button(item["name"], key="project_open_" + item["id"],
                     type="primary" if item["id"] == identifier else "secondary", use_container_width=True):
            st.session_state["project_selected"] = item["id"]
            st.session_state["project_page"] = "Chat"
            st.rerun()
    if len(projects) > 12:
        with st.popover(t("All projects"), use_container_width=True):
            query = st.text_input(t("Find a project"), key="project_find")
            matches = [item for item in projects if query.casefold() in item["name"].casefold()]
            selected = _select("Project", matches, "project_all_selected", optional=True)
            if not matches:
                st.caption("No matching projects.")
            if st.button(t("Open project"), key="project_all_open", disabled=not selected):
                st.session_state["project_selected_requested"] = selected
                st.session_state["project_page"] = "Chat"
                st.rerun()
    project = workspace.get_project(identifier) if identifier else None
    session_id = None
    if project:
        if st.session_state.get("project_opened") != identifier:
            if not project["archived"] and not project["deleted"]:
                workspace.open_project(identifier)
            st.session_state["project_opened"] = identifier
        prefix = f"project_{identifier}_"
        st.divider()
        st.caption(t("CONVERSATIONS"))
        with st.popover(t("Search"), use_container_width=True):
            query = st.text_input(t("Search conversations"), key=prefix + "search", label_visibility="collapsed",
                                  placeholder=t("Search conversations"))
        sessions = workspace.list_sessions(identifier, query=query)
        session_key = prefix + "session"
        if requested := st.session_state.pop(session_key + "_requested", None):
            st.session_state[session_key] = requested
        session_id = st.session_state.get(session_key)
        if session_id not in {item["id"] for item in sessions}:
            session_id = sessions[0]["id"] if sessions else None
            st.session_state[session_key] = session_id
        for item in sessions[:20]:
            title = t("Untitled") if item["title"] == "New conversation" else item["title"]
            if st.button(title, key=prefix + "open_session_" + item["id"], use_container_width=True):
                st.session_state[session_key] = item["id"]
                st.session_state["project_page"] = "Chat"
                st.rerun()
        readonly = project["archived"] or project["deleted"]
        if st.button(t("＋ New conversation"), key=prefix + "new_session", disabled=readonly, use_container_width=True):
            session = workspace.create_session(identifier)
            st.session_state[prefix + "session_requested"] = session["id"]
            st.session_state[prefix + "search_requested"] = ""
            st.rerun()
    with st.container(key="project_sidebar_bottom"):
        for page_name in ("Connections", "Settings"):
            if st.button(t(page_name), key="project_nav_" + page_name.lower(), use_container_width=True,
                         icon=":material/hub:" if page_name == "Connections" else ":material/tune:"):
                st.session_state["project_page"] = page_name
                st.rerun()
    page = st.session_state.get("project_page", "Chat")
    return project, session_id, page


def _new_project(workspace):
    st.subheader(t("New project"))
    name = st.text_input(t("Project name"), key="project_new_name", placeholder=t("For example, silicon optics"))
    if st.button(t("Create project"), disabled=not name.strip(), key="project_create", type="primary"):
        project = workspace.create_project(name)
        workspace.create_session(project["id"])
        st.session_state["project_selected_requested"] = project["id"]
        st.session_state["project_page"] = "Chat"
        st.rerun()


def _all_projects(workspace):
    st.subheader("Projects")
    query = st.text_input("Find a project", key="project_find")
    for project in workspace.list_projects():
        if query.casefold() not in project["name"].casefold():
            continue
        if st.button(project["name"], key="all_project_" + project["id"], use_container_width=True):
            st.session_state["project_selected_requested"] = project["id"]
            st.session_state["project_page"] = "Chat"
            st.rerun()


def _connections(workspace, registry, project):
    from .setup_app import chat_panel

    st.header(t("Connections"))
    identifier = project["id"] if project else None
    if project and not project["archived"] and not project["deleted"]:
        bound = {item["id"] for item in registry.project_connections(identifier)}
        available = [item for item in registry.list() if item["id"] not in bound and not item["revoked"] and not item["deleted"]]
        if available:
            with st.expander(t("Use an existing connection")):
                selected = _select(t("Saved connection"), available, f"project_{identifier}_bind")
                if st.button(t("Connect to project"), key=f"project_{identifier}_bind_save"):
                    item = registry.get(selected)
                    registry.bind(identifier, selected, role=item["kind"])
                    st.rerun()
    chat_panel(registry, identifier)


def _language_selector():
    prefs = st.session_state.get("prefs") or load_settings()
    options = list(LANGUAGE_NAMES)
    current = st.session_state.get("language", prefs.get("language", "en"))
    if current not in options:
        current = "en"
    choice = st.selectbox("Language / 语言", options, index=options.index(current), format_func=LANGUAGE_NAMES.get,
                          key="project_language_choice")
    if choice != current:
        st.session_state["language"] = choice
        save_settings({"language": choice})
        st.session_state["prefs"] = load_settings()
        set_language(choice)
        st.rerun()


def _settings(workspace, project):
    st.header(t("Settings"))
    _language_selector()
    if project:
        _project_controls(workspace, project)
        notes = workspace.list_memory(project["id"])
        if notes:
            with st.expander(t("Project notes")):
                for item in notes:
                    st.markdown("**" + item["key"] + "**")
                    st.write(item["text"])
    inactive = [item for item in workspace.list_projects(include_archived=True, include_deleted=True) if item["archived"] or item["deleted"]]
    with st.expander(t("Archived and deleted projects")):
        if not inactive:
            st.caption(t("None."))
        for item in inactive:
            st.write(item["name"])
            if st.button(t("Restore"), key="restore_project_" + item["id"]):
                workspace.restore_project(item["id"])
                st.session_state["project_selected_requested"] = item["id"]
                st.rerun()
    with st.expander(t("Storage")):
        st.code(str(workspace.home), language="text")
    with st.expander(t("Classic workbench")):
        st.caption(t("Single calculations without projects, as in earlier versions."))
        if st.button(t("Open classic workbench"), key="project_legacy"):
            st.query_params["view"] = "legacy"
            st.rerun()


def _upload(workspace, identifier, prefix, readonly):
    with st.expander(t("Add files")):
        st.caption(t("Structures (CIF, POSCAR), scripts and notes for this project."))
        uploaded = st.file_uploader(t("Files"), accept_multiple_files=True, key=prefix + "upload", disabled=readonly)
        if st.button(t("Import files"), key=prefix + "import", disabled=readonly or not uploaded):
            for item in uploaded:
                name = Path(item.name.replace("\\", "/")).name
                project_files.write_file(workspace, identifier, "files/" + name, item.getvalue())
            st.success(t("Files imported."))


def _file_editor(workspace, identifier, prefix, readonly):
    _upload(workspace, identifier, prefix, readonly)
    files = project_files.list_files(workspace, identifier)
    if not files:
        st.caption(t("No files yet."))
        return
    names = [item["path"] for item in files]
    key = prefix + "file"
    if st.session_state.get(key) not in names:
        st.session_state[key] = names[0]
    relative = st.selectbox("File", names, key=key)
    path = workspace.resolve_path(identifier, relative, must_exist=True)
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"} and path.stat().st_size <= 10 * 1024 * 1024:
        st.image(str(path))
    snapshot_key = prefix + "file_snapshot"
    saved = st.session_state.get(snapshot_key)
    if not saved or saved["path"] != relative:
        try:
            saved = project_files.read_file(workspace, identifier, relative)
        except ValueError as error:
            st.caption(str(error))
            saved = {"path": relative, "content": None}
        st.session_state[snapshot_key] = saved
    if saved["content"] is not None:
        if saved["editable"] and not readonly:
            text = st.text_area("Contents", saved["content"], height=380,
                                key=prefix + "edit_" + saved["path"] + "_" + saved["sha256"])
            left, right = st.columns(2)
            if left.button("Save file", key=prefix + "save_file", disabled=text == saved["content"]):
                try:
                    project_files.write_file(workspace, identifier, relative, text, expected_sha256=saved["sha256"])
                    st.session_state.pop(snapshot_key, None)
                    st.rerun()
                except (ValueError, OSError) as error:
                    _error(error)
            if right.button("Reload file", key=prefix + "reload_file"):
                st.session_state.pop(snapshot_key, None)
                st.rerun()
        else:
            st.code(saved["content"], language="text")
    if path.stat().st_size <= 100 * 1024 * 1024:
        st.download_button("Download file", path.read_bytes(), file_name=path.name, key=prefix + "download_file")


def _file_proposal(workspace, identifier, message, readonly):
    proposal = message.get("metadata", {}).get("file_edit")
    if not proposal:
        return
    with st.expander("Review file change · " + proposal["path"]):
        try:
            current = project_files.read_file(workspace, identifier, proposal["path"])
            if current["sha256"] != proposal["expected_sha256"]:
                st.caption("This file has changed. Ask for a new edit.")
                return
            diff = "\n".join(difflib.unified_diff(current["content"].splitlines(), proposal["content"].splitlines(),
                                               fromfile=proposal["path"], tofile="Proposed", lineterm=""))
            st.code(diff, language="diff")
            if st.button("Apply change", disabled=readonly, key=f"project_{identifier}_apply_{message['id']}"):
                project_files.write_file(workspace, identifier, proposal["path"], proposal["content"],
                                         expected_sha256=proposal["expected_sha256"])
                st.session_state.pop(f"project_{identifier}_context_file_snapshot", None)
                st.rerun()
        except (ValueError, OSError) as error:
            _error(error)


def _current_task_label(attempt):
    if not attempt or attempt.get("source_context", {}).get("kind") not in {"continuation", "retry"}:
        return ""
    return " → ".join(task.upper() for task in attempt.get("tasks", []))


def _workflow_graph(item, attempts=None):
    nodes = {step["id"]: f"n{index}" for index, step in enumerate(item["steps"])}
    lines = ['digraph { rankdir=LR; node [shape=box, style="rounded", fontname="Arial"];']
    for step in item["steps"]:
        label = f"{step['name']}\n{step['engine'].upper()} · {step['status']}"
        current = _current_task_label((attempts or {}).get(step["id"]))
        if current:
            label += "\nCurrent: " + current
        lines.append(f'{nodes[step["id"]]} [label={json.dumps(label)}];')
        lines.extend(f'{nodes[parent]} -> {nodes[step["id"]]};' for parent in step["depends_on"])
    lines.append("}")
    st.graphviz_chart("\n".join(lines))


def _review_inputs(workspace, identifier, review, prefix):
    st.caption("Inputs and resources")
    state = review["state"]
    st.json(state.get("parameters", state.get("task", {})))
    root = workspace.project_paths(identifier)["root"]
    run = Path(review["run_dir"])
    paths = list(review["files"])
    if paths:
        selected = st.selectbox("Prepared file", paths, key=prefix + "prepared_file")
        relative = (run / selected).relative_to(root).as_posix()
        try:
            st.code(project_files.read_file(workspace, identifier, relative)["content"], language="text")
        except (ValueError, OSError) as error:
            st.caption(str(error))


def _go_to_connections(kind, identifier, key, label):
    if st.button(label, key=key):
        st.session_state["setup_chat_" + identifier + "_kind_requested"] = kind
        st.session_state["project_page"] = "Connections"
        st.rerun()


def _step_controls(workspace, registry, execution, identifier, item, step_id, model_id, readonly):
    if not step_id:
        return
    step = next(step for step in item["steps"] if step["id"] == step_id)
    prefix = f"project_{identifier}_{item['id']}_{step_id}_"
    attempt = execution.latest(identifier, item["id"], step_id)
    state = attempt["state"] if attempt else step["status"]
    import html as _html
    st.html(f'<div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.2rem">'
            f'<strong style="font-size:1rem">{_html.escape(step["name"])}</strong>{tag(step["engine"].upper())}'
            f'{chip(t(_STATE_LABELS.get(state, state)), _STATE_KIND.get(state, "warn"))}</div>')
    if current := _current_task_label(attempt):
        st.caption("Current: " + current)
    if attempt and attempt.get("last_error"):
        st.warning(redact(attempt["last_error"]))
    if readonly:
        return
    bound = {connection["id"] for connection in registry.project_connections(identifier)
             if connection["kind"] == "hpc" and not connection["revoked"] and not connection["deleted"]}
    environments = [environment for environment in registry.environments()
                    if environment["connection_id"] in bound and environment["engine"] == step["engine"]]
    configured = (attempt or {}).get("environment_id") or step["spec"].get("environment_id")
    if not attempt or not attempt.get("run_id"):
        if environments and not configured and len(environments) == 1:
            # One matching environment: use it without asking.
            try:
                attempt = execution.configure_step(identifier, item["id"], step_id, environments[0]["id"])
                configured = environments[0]["id"]
            except ValueError as error:
                st.caption(redact(str(error)))
        if environments and (len(environments) > 1 or configured not in {environment["id"] for environment in environments}):
            preferred = configured
            environment = _select(t("Environment"), environments, prefix + "environment", preferred=preferred)
            if environment != configured and st.button(t("Use environment"), key=prefix + "configure"):
                execution.configure_step(identifier, item["id"], step_id, environment)
                st.rerun()
        elif not environments:
            if bound:
                st.caption(t("No {engine} environment is set up on the connected cluster yet. Planning works without it; preparing inputs needs it.").format(engine=step["engine"].upper()))
                _go_to_connections("software", identifier, prefix + "add_environment", t("Add software environment"))
            else:
                st.caption(t("Connect a cluster to prepare and submit this step. Planning works without it."))
                _go_to_connections("hpc", identifier, prefix + "add_cluster", t("Connect a cluster"))
        if not attempt or not attempt.get("proposal") or attempt["state"] == "draft":
            if st.button(t("Plan step"), key=prefix + "plan", disabled=not model_id, type="primary"):
                with st.spinner(t("Planning…")):
                    try:
                        execution.draft_step(identifier, item["id"], step_id, registry.model_settings(model_id))
                    except Exception as error:
                        st.session_state[prefix + "notice"] = redact(str(error))
                st.rerun()
    if notice := st.session_state.pop(prefix + "notice", None):
        st.error(notice)
    if attempt and attempt.get("reply") and not attempt.get("run_id"):
        st.write(attempt["reply"])
    if attempt and attempt.get("proposal"):
        proposal = attempt["proposal"]
        summary = proposal.get("summary") or ""
        if summary and summary not in (attempt.get("reply") or ""):
            st.write(summary)
        for question in proposal.get("questions", []):
            st.info(question)
        if proposal.get("questions"):
            st.caption(t("Answer in the message box below; the plan is revised in place."))
        with st.expander(t("Plan details")):
            plan = proposal.get("plan") or proposal
            st.json({key: value for key, value in plan.items()
                     if key in {"operations", "stages", "variants", "parameters", "task", "intent", "notes", "continuation",
                                "inputs", "input_sha256", "entrypoint", "outputs", "arguments", "scope", "files",
                                "validation", "dependencies", "input_artifacts", "limits", "references"}})
    library_ready = True
    if step["engine"] == "qe" and configured and (not attempt or not attempt.get("run_id")):
        library_ready = _pseudopotential_check(workspace, registry, identifier, step, configured, prefix,
                                              (attempt or {}).get("proposal"))
    if step["engine"] in {"abinit", "cp2k", "elk"} and configured and (not attempt or not attempt.get("run_id")):
        library_ready = _library_check(workspace, registry, identifier, step, configured, prefix,
                                       (attempt or {}).get("proposal"))
    if attempt and attempt["state"] == "ready":
        if st.button(t("Prepare inputs"), key=prefix + "prepare", type="primary", disabled=not library_ready,
                     help=t("Writes the input files locally for review. Nothing is submitted yet.")):
            try:
                execution.prepare_step(identifier, item["id"], step_id)
            except Exception as error:
                st.session_state[prefix + "notice"] = redact(str(error))
            st.rerun()
    if attempt and attempt["state"] in {"prepared", "authorized"}:
        with st.expander(t("Review inputs"), expanded=True):
            review = execution.review(identifier, item["id"], step_id)
            _review_inputs(workspace, identifier, review, prefix)
            reviewed = st.checkbox(t("I reviewed the inputs and resources."), key=prefix + "review_" + review["digest"])
            if st.button(t("Submit"), type="primary", disabled=not reviewed, key=prefix + "submit_" + review["digest"]):
                try:
                    execution.authorize_step(identifier, item["id"], step_id, review["digest"])
                    execution.launch_step(identifier, item["id"], step_id)
                except Exception as error:
                    st.session_state[prefix + "notice"] = redact(str(error))
                st.rerun()
    if attempt and attempt.get("run_id") and attempt["state"] not in {"succeeded", "completed_unverified", "cancelled"}:
        with st.expander(t("Job controls")):
            if attempt.get("approved_digest") and attempt["state"] not in {"prepared", "authorized"}:
                if st.button(t("Resume monitoring"), key=prefix + "resume",
                             help=t("Resume this run, including its remaining approved stages.")):
                    try:
                        execution.resume_monitoring(identifier, item["id"], step_id, confirmed=True)
                    except Exception as error:
                        st.session_state[prefix + "notice"] = redact(str(error))
                    st.rerun()
            if st.button(t("Reconnect"), key=prefix + "reconnect"):
                execution.refresh(identifier, item["id"], reconnect=True)
                st.rerun()
            confirmed = st.checkbox(t("Cancel this calculation"), key=prefix + "cancel_confirm")
            if st.button(t("Cancel job"), disabled=not confirmed, key=prefix + "cancel"):
                execution.cancel_step(identifier, item["id"], step_id, confirmed=True)
                st.rerun()


def _context_panel(workspace, registry, execution, project, workflow):
    identifier = project["id"]
    prefix = f"project_{identifier}_context_"
    readonly = project["archived"] or project["deleted"]
    page = st.radio("Context", ["Files", "Workflow", "Jobs", "Results"], horizontal=True, key=prefix + "page",
                    format_func=t, label_visibility="collapsed")
    if page == "Files":
        _file_editor(workspace, identifier, prefix, readonly)
    elif page == "Workflow":
        if workflow:
            attempts = {attempt["step_id"]: attempt for attempt in execution.attempts(identifier, workflow["id"])}
            _workflow_graph(workflow, attempts)
            rows = [{field: step[field] for field in ("name", "engine", "status")} for step in workflow["steps"]]
            if any(_current_task_label(attempt) for attempt in attempts.values()):
                for row, step in zip(rows, workflow["steps"]):
                    row["current"] = _current_task_label(attempts.get(step["id"]))
            st.dataframe(rows, hide_index=True)
        else:
            st.caption("Describe a calculation to create a workflow.")
    elif page in {"Jobs", "Results"}:
        runs = workspace.list_runs(identifier)
        if not runs:
            st.caption("No calculations yet.")
            return
        if page == "Jobs":
            st.dataframe([{"Run": run["run_id"], "Engine": run["engine"], "Status": run["status"]} for run in runs], hide_index=True)
        selected = _select("Run", runs, prefix + "run", title="run_id")
        run = workspace.get_run(identifier, selected)
        if page == "Jobs":
            for stage in (run.get("state") or {}).get("stages", []):
                st.write(f"{stage.get('name', '')} · {stage.get('status', '')} · {stage.get('job_id') or 'No job ID'}")
            logs = [item for item in project_files.list_files(workspace, identifier)
                    if item["path"].startswith(run["relative_path"] + "/")
                    and (Path(item["path"]).suffix in {".log", ".err", ".out"} or Path(item["path"]).name == "worker.log")]
            if logs:
                path = st.selectbox("Log", [log["path"] for log in logs], key=prefix + "log")
                try:
                    st.code(project_files.read_file(workspace, identifier, path)["content"], language="text")
                except (ValueError, OSError) as error:
                    st.caption(str(error))
        else:
            from .app import _batch_panel, _run_panel

            folder = workspace.resolve_path(identifier, run["relative_path"], must_exist=True)
            if run["engine"] in {"python", "wannier90"}:
                _script_results(workspace, identifier, run, prefix)
            elif run["kind"] == "batch":
                _batch_panel(folder, None, read_only=True)
            else:
                _run_panel(folder, read_only=True)


def _script_results(workspace, identifier, run, prefix):
    from .script_workflow import bundle_run, load_run_context

    folder = workspace.resolve_path(identifier, run["relative_path"], must_exist=True)
    context = load_run_context(folder)
    st.subheader("Python" if run["engine"] == "python" else "Wannier90")
    st.write(context["status"].replace("_", " "))
    st.caption(context["outcome"])
    for limitation in context["limits"]:
        st.caption(limitation)
    with st.expander("Checks"):
        st.dataframe([{ "Check": fact["label"], "Value": str(fact["value"])} for fact in context["facts"].values()], hide_index=True)
    for artifact in context["artifacts"]:
        path = workspace.resolve_path(identifier, run["relative_path"] + "/" + artifact["path"], must_exist=True)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and path.stat().st_size <= 10 * 1024 * 1024:
            st.image(str(path))
        if path.stat().st_size <= 100 * 1024 * 1024:
            st.download_button(artifact["label"], path.read_bytes(), file_name=path.name,
                               key=prefix + run["id"] + "_artifact_" + artifact["id"])
    if st.button("Prepare download", key=prefix + run["id"] + "_bundle"):
        bundle_run(folder)
    archive = folder / "results.zip"
    if archive.is_file() and archive.stat().st_size <= 100 * 1024 * 1024:
        st.download_button("Download results", archive.read_bytes(), file_name="results.zip", key=prefix + run["id"] + "_zip")


def _chat(workspace, registry, project, session_id):
    identifier = project["id"]
    prefix = f"project_{identifier}_"
    readonly = project["archived"] or project["deleted"]
    if not session_id:
        st.info("Start a conversation in the sidebar.")
        return
    session = workspace.get_session(identifier, session_id)
    models = [item for item in registry.project_connections(identifier)
              if item["kind"] == "model" and not item["revoked"] and not item["deleted"]]
    # One header row: the model and mode are settings of the conversation, not its headline.
    header, mode_menu, model_menu, context_menu, project_menu = st.columns([3.4, 2.2, 1.9, 1.9, .8], vertical_alignment="center")
    with header:
        st.subheader(project["name"])
    selected_model = st.session_state.get(prefix + "model")
    current_model = next((model for model in models if model["id"] == selected_model), models[0] if models else None)
    agent_capable = bool(current_model) and current_model["settings"].get("provider") in {"codex", "claude_cli"}
    mode_key = prefix + "mode"
    with mode_menu:
        if agent_capable:
            mode = st.pills(t("Mode"), ["guided", "agent"], default=st.session_state.get(mode_key, _DEFAULT_MODE), key=mode_key,
                            format_func=lambda value: t("Automatic") if value == "agent" else t("Step by step"), label_visibility="collapsed",
                            help=t("Automatic: the CLI reads the project, then plans, prepares and explains with its own tool calls. Step by step: one checked action per message."))
            mode = mode or _DEFAULT_MODE
        else:
            mode = "guided"
    with model_menu:
        selected_model = st.session_state.get(prefix + "model")
        model_name = next((model["name"] for model in models if model["id"] == selected_model),
                          models[0]["name"] if models else t("Connect a model"))
        with st.popover(model_name, use_container_width=True):
            model_id = _select(t("Model"), models, prefix + "model")
            usage = next((message.get("metadata", {}).get("model_usage") for message in reversed(session["messages"])
                          if message.get("metadata", {}).get("model_usage")), None)
            if usage:
                from .app import _model_usage
                with st.expander(t("Last response usage")):
                    for call in usage:
                        _model_usage(call)
            if st.button(t("Model settings"), key=prefix + "model_settings"):
                st.session_state["setup_chat_" + identifier + "_kind_requested"] = "model"
                st.session_state["project_page"] = "Connections"
                st.rerun()
    with context_menu:
        if st.button(t("Files and runs"), key=prefix + "context_toggle", use_container_width=True):
            st.session_state[prefix + "show_context"] = not st.session_state.get(prefix + "show_context", False)
            st.rerun()
    with project_menu:
        with st.popover("···", use_container_width=True):
            _project_controls(workspace, project)
            title = st.text_input(t("Conversation title"), session["title"], key=prefix + session_id + "_title")
            if st.button(t("Rename conversation"), disabled=readonly or not title.strip(), key=prefix + session_id + "_rename"):
                workspace.update_session_title(identifier, session_id, title, expected_revision=session["revision"])
                st.rerun()
    if readonly:
        st.info(t("Restore this project to continue working."))
    show_context = st.session_state.get(prefix + "show_context", False)
    if show_context:
        width = st.session_state.get(prefix + "context_width", "Medium")
        center, side = st.columns({"Narrow": [3, 1], "Medium": [2, 1], "Wide": [1, 1]}[width], gap="large")
    else:
        center, side = st.container(key="project_chat"), None
    execution = ProjectExecution(workspace, registry)
    workflows = [item for item in workspace.list_workflows(identifier) if item["status"] != "superseded"]
    workflow_key = prefix + session_id + "_workflow"
    preferred = next((item["id"] for item in workflows if item["session_id"] == session_id), None)
    setup = _setup_status(registry, identifier)
    with center:
        if not session["messages"]:
            _onboarding(registry, project, setup)
        elif not models:
            st.warning(t("This project has no model connection. Connect one to continue the conversation."))
            _go_to_connections("model", identifier, prefix + "reconnect_model", t("Connect a model"))
        for message in session["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                _agent_actions(message)
                analysis = message.get("metadata", {}).get("analysis")
                if analysis:
                    for limitation in analysis.get("limits") or []:
                        st.caption(limitation)
                    if analysis.get("sources") or analysis.get("evidence"):
                        with st.expander(t("Evidence")):
                            if analysis.get("sources"):
                                st.dataframe([{"Source": source["namespace"], "Engine": source["engine"],
                                               "Status": source["status"]} for source in analysis["sources"]], hide_index=True)
                            if analysis.get("evidence"):
                                st.code("\n".join(analysis["evidence"]), language="text")
                _file_proposal(workspace, identifier, message, readonly)
        item, step_id, workflow_id = None, None, None
        if workflows:
            with st.expander(t("Tasks"), expanded=False):
                workflow_id = _select(t("Workflow"), workflows, workflow_key, title="title", preferred=preferred, optional=True)
                item = workspace.get_workflow(identifier, workflow_id) if workflow_id else None
                if item:
                    step_id = _select(t("Step"), item["steps"], workflow_key + "_step")
                    if st.button(t("Refresh status"), key=workflow_key + "_refresh", disabled=readonly):
                        execution.refresh(identifier, workflow_id)
                        st.rerun()
            if item and step_id:
                # The current step is the main thing to act on, so it sits in the conversation, not in a menu.
                with st.container(border=True, key=prefix + session_id + "_task_card"):
                    _step_controls(workspace, registry, execution, identifier, item, step_id, model_id, readonly)
        run_records = workspace.list_runs(identifier)
        selected_runs = []
        if run_records:
            with st.expander(t("Attach results")):
                selected_runs = st.multiselect(t("Results to discuss"), [run["id"] for run in run_records],
                    format_func=lambda value: next(run["run_id"] for run in run_records if run["id"] == value),
                    key=prefix + session_id + "_analysis_runs")
        if not models and session["messages"]:
            st.caption(t("Connect a model to plan calculations."))
    if side:
        with side:
            with st.popover(t("Panel size")):
                width = st.selectbox(t("Panel width"), ["Narrow", "Medium", "Wide"], index=1, key=prefix + "context_width", format_func=t)
            _context_panel(workspace, registry, execution, project, item)
    # At the page root, Streamlit keeps this composer fixed to the viewport.
    placeholder = t("Describe a calculation or ask about the results") if model_id else t("Connect a model to plan calculations")
    message = st.chat_input(placeholder, key=prefix + session_id + "_chat", disabled=readonly or not model_id)
    if not message and model_id and not readonly:
        message = st.session_state.pop(prefix + "send", None)
    if message:
        try:
            if mode == "agent":
                known = {item["id"] for item in workspace.list_workflows(identifier)}
                with st.spinner(t("Working…")):
                    outcome = agent_runner.run_turn(workspace, registry, identifier, session_id, message, model_id,
                                                    workflow_id=workflow_id, step_id=step_id, run_ids=selected_runs)
                created = [item for item in workspace.list_workflows(identifier) if item["id"] not in known]
                if created:
                    st.session_state[workflow_key + "_requested"] = created[-1]["id"]
                    first = next((step for step in created[-1]["steps"] if not step["depends_on"]), None)
                    if first:
                        st.session_state[workflow_key + "_step_requested"] = first["id"]
                if session["title"] == "New conversation" and not session["messages"]:
                    workspace.update_session_title(identifier, session_id, message.splitlines()[0][:80],
                                                   expected_revision=outcome["session"]["revision"])
                st.rerun()
            with st.spinner(t("Working…")):
                result = project_agent.send(workspace, identifier, session_id, message, model_id,
                                            workflow_id=workflow_id, step_id=step_id, run_ids=selected_runs)
            if result.get("workflow_id"):
                st.session_state[workflow_key + "_requested"] = result["workflow_id"]
            if result.get("step_id"):
                st.session_state[workflow_key + "_step_requested"] = result["step_id"]
            if result.get("action") == "workflow" and result.get("workflow_id"):
                # Plan the first runnable step right away, so the reply is a plan, not a menu.
                graph = workspace.get_workflow(identifier, result["workflow_id"])
                first = next((step for step in graph["steps"] if not step["depends_on"]), None)
                if first:
                    st.session_state[workflow_key + "_step_requested"] = first["id"]
                    with st.spinner(t("Planning the first step…")):
                        try:
                            execution.draft_step(identifier, result["workflow_id"], first["id"],
                                                 registry.model_settings(model_id), message=message)
                        except Exception as error:
                            st.session_state[f"project_{identifier}_{result['workflow_id']}_{first['id']}_notice"] = redact(str(error))
            if session["title"] == "New conversation" and not session["messages"]:
                workspace.update_session_title(identifier, session_id, message.splitlines()[0][:80],
                                               expected_revision=result["session"]["revision"])
            st.rerun()
        except (ValueError, OSError, agent_runner.agent.AgentError) as error:
            _error(error)


def main():
    st.set_page_config(page_title="DFT Agent", page_icon="⚛", layout="wide")
    from .ui_style import apply_style
    apply_style()
    st.html("""<style>
    [data-testid="stMainBlockContainer"] {max-width:1380px;padding:1.3rem 2.4rem 3rem;}
    [data-testid="stMainBlockContainer"]:has(#connections) {max-width:960px;}
    [data-testid="stSidebarHeader"] {height:40px;}
    [data-testid="stSidebarUserContent"] {padding:.6rem .7rem .6rem;}
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {gap:.3rem;}
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        display:block;text-align:left;justify-content:flex-start;border-color:transparent;background:transparent;
        font-weight:450;padding:.45rem .65rem;min-height:2.1rem;border-radius:9px;}
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {background:var(--dfa-code);color:var(--dfa-ink);border-color:transparent;}
    [data-testid="stSidebar"] [data-testid="stButton"] [data-testid="stMarkdownContainer"] {width:100%;}
    [data-testid="stSidebar"] [data-testid="stButton"] button > div {justify-content:flex-start;}
    [data-testid="stSidebar"] [data-testid="stButton"] button > div > span {width:100%;justify-content:flex-start;}
    [data-testid="stSidebar"] [data-testid="stButton"] button p {overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:left;font-size:.92rem;}
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        color:var(--dfa-accent-strong)!important;background:var(--dfa-accent-soft)!important;border-color:transparent!important;font-weight:600;}
    [data-testid="stSidebar"] [data-testid="stPopover"] button {border-color:var(--dfa-line);background:var(--dfa-surface);text-align:left;}
    [data-testid="stSidebarUserContent"] > div {min-height:calc(100vh - 80px);}
    [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {min-height:calc(100vh - 96px);}
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:has(> .st-key-project_sidebar_bottom) {margin-top:auto;}
    .st-key-project_sidebar_bottom {margin-top:auto;padding-top:1.5rem;border-top:1px solid var(--dfa-line);}
    .st-key-project_chat {max-width:820px;margin:.5rem auto 0;}
    [data-testid="stBottom"] > div {max-width:980px;margin:auto;}
    [data-testid="stChatInput"] {max-width:820px;margin:0 auto;}
    [class*="_task_card"] > div {border-left:3px solid var(--dfa-accent) !important;padding:1rem 1.1rem;}
    [data-testid="stMainBlockContainer"] h3 {margin:0;}
    @media(max-width:900px) {[data-testid="stMainBlockContainer"] {padding:1.2rem .9rem;} .dfa-hello {margin-top:1rem;}}
    </style>""")
    prefs = st.session_state.get("prefs") or load_settings()
    st.session_state["prefs"] = prefs
    set_language(st.session_state.get("language", prefs.get("language", "en")))
    workspace = Workspace()
    registry = ConnectionRegistry(workspace)
    try:
        # Deferred resets happen before their widgets are created.
        for key in list(st.session_state):
            if key.startswith("project_") and key.endswith("_search_requested"):
                st.session_state[key.removesuffix("_requested")] = st.session_state.pop(key)
        with st.sidebar:
            project, session_id, page = _sidebar(workspace)
        if page == "New project":
            _new_project(workspace)
        elif page == "Projects":
            _all_projects(workspace)
        elif page == "Connections":
            _connections(workspace, registry, project)
        elif page == "Settings":
            _settings(workspace, project)
        elif project:
            _chat(workspace, registry, project, session_id)
        else:
            st.title("DFT Agent")
            st.write(t("A place for your calculations."))
            _language_selector()
            if st.button(t("Create your first project"), type="primary"):
                st.session_state["project_page"] = "New project"
                st.rerun()
    except (ValueError, OSError) as error:
        _error(error)
