"""Persistent conversations shared by the VASP and QE interfaces."""

from datetime import datetime
from importlib.resources import files
from pathlib import Path
import tempfile
import uuid

import streamlit as st

from .i18n import t


def _activate(folder, record):
    engine = record["engine"]
    st.session_state["active_conversation"] = str(folder)
    st.session_state[f"chat_requested_{engine}"] = str(folder)
    st.session_state[f"chat_new_{engine}"] = False
    active = record.get("active_run")
    if active:
        st.session_state["active_run"] = str(active)
    else:
        st.session_state.pop("active_run", None)


def open_run_chat(run_dir):
    """Open an existing run without changing its inputs or execution state."""
    from . import conversation, conversation_store

    try:
        folder = conversation.open_run(run_dir)
        record = conversation_store.load(folder)
        _activate(folder, record)
        st.session_state["requested_chat_runs_root"] = str(Path(folder).parent.parent)
        st.session_state["requested_chat_engine"] = record["engine"]
        st.session_state["requested_workspace_tab"] = t("Chat")
        st.rerun(scope="app")
    except Exception as exc:
        st.error(str(exc))


def _start_panel(runs_root, engine):
    from . import conversation, conversation_store

    source_kind = st.radio(t("Chat structure"), ["Upload", "Example"], horizontal=True,
                           format_func=t, key=f"chat_source_{engine}")
    content, name = None, "structure.vasp"
    if source_kind == "Upload":
        upload = st.file_uploader(t("Structure for this conversation"), key=f"chat_upload_{engine}",
                                  help=t("CIF or POSCAR; no extension needed for POSCAR."))
        if upload is not None:
            content, name = upload.getvalue(), upload.name
    else:
        examples = files("vasp_slurm_agent").joinpath("examples")
        choices = {}
        for entry in sorted(examples.iterdir(), key=lambda item: item.name):
            if entry.name.endswith((".cif", ".vasp")):
                choices[Path(entry.name).stem] = entry.name
        labels = sorted(choices)
        default = "MgB2" if engine == "qe" else "Si"
        label = st.selectbox(t("Chat example"), labels, index=labels.index(default),
                             key=f"chat_example_{engine}")
        name = choices[label]
        content = examples.joinpath(name).read_bytes()
    endpoints = []
    if engine == "qe":
        with st.expander(t("NEB images")):
            endpoints = st.file_uploader(t("Additional structures in path order"),
                                         accept_multiple_files=True, key="chat_neb_images")
    if st.button(t("Start conversation"), disabled=not content, type="primary", key=f"chat_start_{engine}"):
        try:
            with tempfile.TemporaryDirectory(prefix="dft-chat-input-") as temporary:
                source = Path(temporary) / ("structure.cif" if name.lower().endswith(".cif") else "structure.vasp")
                source.write_bytes(content)
                images = [source] if endpoints else None
                for index, upload in enumerate(endpoints or []):
                    image = Path(temporary) / (f"image-{index}.cif" if upload.name.lower().endswith(".cif") else f"image-{index}.vasp")
                    image.write_bytes(upload.getvalue())
                    images.append(image)
                folder = conversation.start(runs_root, engine, source, images=images)
            _activate(folder, conversation_store.load(folder))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _model_note(settings):
    from .providers import effective_key

    if settings is None or (settings.provider != "codex" and
                            (not settings.model or not effective_key(settings.provider, settings.api_key))):
        st.caption(t("Choose a model and add your API key in Model."))


def panel(runs_root, engine, settings, config=None):
    from . import conversation, conversation_store
    from .app import _model_usage

    selected_key = f"chat_selected_{engine}"
    new_key = f"chat_new_{engine}"
    try:
        saved = conversation_store.list_conversations(runs_root, engine=engine)
    except Exception as exc:
        st.error(str(exc))
        return
    choices = {str(item["folder"]): item for item in saved}
    if requested := st.session_state.pop(f"chat_requested_{engine}", None):
        st.session_state[selected_key] = requested
    selected = st.session_state.get(selected_key)
    if selected not in choices:
        st.session_state[selected_key] = None
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with left:
        selected = st.selectbox(t("Conversation"), [None, *choices], key=selected_key,
                                format_func=lambda value: choices[value].get("title") or Path(value).name if value else t("Choose a conversation"),
                                on_change=lambda: st.session_state.update({new_key: False}))
    with right:
        if st.button(t("New conversation"), key=f"chat_new_button_{engine}"):
            st.session_state[new_key] = True
            st.session_state.pop("active_run", None)
    if st.session_state.get(new_key):
        _start_panel(runs_root, engine)
        return
    if not selected:
        st.caption(t("Start a conversation or open one from Runs."))
        return
    folder = Path(selected)
    try:
        record = conversation_store.load(folder)
    except Exception as exc:
        st.error(str(exc))
        return
    st.session_state["active_conversation"] = selected
    if st.session_state.get("chat_displayed") != selected:
        st.session_state["chat_displayed"] = selected
        active = record.get("active_run")
        if active:
            st.session_state["active_run"] = str(active)
        else:
            st.session_state.pop("active_run", None)
    _model_note(settings)
    current_context = None
    if record.get("active_run") and any(item.get("context_sha256") for item in record["messages"]):
        from .explanation import load_run_context
        try:
            current_context = load_run_context(record["active_run"])["context_sha256"]
        except (OSError, ValueError, KeyError):
            pass
    for message in record.get("messages", []):
        with st.chat_message(message["role"]):
            if message.get("context_sha256") and message["context_sha256"] != current_context:
                st.caption(t("Earlier results"))
            if message.get("error"):
                st.error(message["content"])
            else:
                st.write(message["content"])
            _model_usage(message.get("model_usage"))
    pending = record.get("pending")
    if pending:
        plan = pending.get("plan", pending)
        if pending.get("summary"):
            st.write(pending["summary"])
        for question in pending.get("questions", []):
            st.info(question)
        with st.expander(t("Plan details")):
            st.json({key: plan[key] for key in ("operations", "stages", "variants", "tasks", "parameters",
                                               "stage_parameters", "continuation", "repair_action", "changes",
                                               "questions", "notes", "intent") if key in plan})
        if st.button(t("Prepare inputs"), key="chat_prepare", disabled=pending.get("status") != "ready"):
            try:
                run_dir = Path(runs_root) / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
                record = conversation.prepare(folder, run_dir, config=config)
                st.session_state["active_run"] = str(record.get("active_run") or run_dir)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    message = st.chat_input(t("Ask, revise, or plan the next step"), key=f"chat_message_{engine}")
    if message:
        try:
            with st.spinner(t("Planning…")):
                conversation.send(folder, message, settings, config=config)
            st.session_state.pop(f"chat_error_{selected}", None)
            st.rerun()
        except Exception as exc:
            st.session_state[f"chat_error_{selected}"] = str(exc)
    error = st.session_state.get(f"chat_error_{selected}")
    if error:
        st.error(error)
