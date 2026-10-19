"""Guided connection setup for the project interface.

Credentials only ever pass through secure fields and on-click callbacks that
remove them from the session before the page renders again. Nothing here
submits a job; cluster access is limited to sign-in and read-only probes.
"""

import json
import re
import shlex

import streamlit as st

from . import setup_assistant as assistant
from .i18n import t
from .secret_guard import redact

_PROVIDER_ORDER = ("responses", "anthropic", "qwen", "deepseek", "glm", "grok", "chat_completions", "local")
_CLI_PROVIDERS = {"codex", "claude_cli"}
_SIGN_IN_COMMANDS = {"codex": "codex login", "claude_cli": "claude"}
_INSTALL_HINTS = {"codex": "npm install -g @openai/codex", "claude_cli": "npm install -g @anthropic-ai/claude-code"}


def _option_label(field, option):
    if field == "provider":
        from .providers import PROVIDERS

        return PROVIDERS.get(option, {}).get("label", {"claude_cli": "Claude Code", "local": "Local model"}.get(option, option))
    return {"auth_mode": {"agent": t("SSH agent"), "key": t("SSH key"), "password": t("Password"),
                          "code": t("Password and login code"), "interactive": t("Interactive SSH")},
            "engine": {"vasp": "VASP", "qe": "Quantum ESPRESSO", "wannier90": "Wannier90", "python": "Python"},
            "scheduler": {"auto": t("Detect"), "slurm": "Slurm", "pbs": "PBS"}}.get(field, {}).get(option, option)


def _active_project(registry, project_id):
    if project_id is not None:
        project = registry.workspace.get_project(project_id)
        if project["archived"] or project["deleted"]:
            raise ValueError("Restore the project before changing its setup.")


def _status_label(record):
    status = record.get("connection_status", "untested")
    return {"connected": t("connected"), "untested": t("not tested yet"), "error": t("error"),
            "revoked": t("revoked"), "unavailable": t("unavailable")}.get(status, status)


def _probe_detail(record):
    return redact(str((record.get("probe") or {}).get("detail") or ""))


def _notice(key):
    if key in st.session_state:
        ok, message = st.session_state.pop(key)
        (st.success if ok else st.error)(redact(message))


def _credential_controls(registry, record, prefix, project_id=None):
    from .connections import PASSWORD_MODES

    settings = record["settings"]
    needs_secret = ((record["kind"] == "model" and settings.get("provider") not in {"codex", "claude_cli", "local"})
                    or (record["kind"] == "hpc" and settings.get("auth_mode") in PASSWORD_MODES))
    if not needs_secret or record.get("revoked"):
        return
    key, notice = prefix + "secret", prefix + "secret_notice"

    def save_secret():
        value = st.session_state.pop(key, "")
        try:
            _active_project(registry, project_id)
            registry.set_secret(record["id"], value)
            st.session_state[notice] = (True, t("Credential saved in the system keychain."))
        except Exception:
            st.session_state[notice] = (False, t("Could not save the credential. Check the system keychain."))

    st.text_input(t("API key") if record["kind"] == "model" else t("SSH password"), type="password", key=key)
    st.button(t("Save credential"), key=prefix + "save_secret", on_click=save_secret)
    if notice in st.session_state:
        ok, message = st.session_state[notice]
        (st.success if ok else st.error)(message)


def _connection_controls(registry, record, prefix, project_id=None):
    st.caption("Authentication: " + str(record.get("auth_status", "unchecked"))
               + " · Connection: " + str(record.get("connection_status", "unchecked")))
    if record.get("revoked"):
        st.info("Connection revoked.")
        return
    if record["kind"] == "hpc" and record["settings"].get("auth_mode") == "interactive":
        st.info("Sign in through your SSH terminal. Enter login codes there, then test the connection here.")
        if st.button("Open SSH sign-in", key=prefix + "ssh_signin"):
            from .ssh_auth import launch

            try:
                st.session_state[prefix + "ssh_signin_result"] = launch(registry, record["id"])
            except Exception as exc:
                st.error(redact(exc))
        if result := st.session_state.get(prefix + "ssh_signin_result"):
            st.info(result["message"])
            if result["status"] == "manual":
                st.code(result["command"], language="powershell" if not result["reusable"] else "bash")
    _credential_controls(registry, record, prefix, project_id)
    left, right = st.columns(2)
    if left.button("Test connection", key=prefix + "probe"):
        try:
            with st.spinner("Checking connection…"):
                result = registry.probe(record["id"])
            st.json(result)
        except Exception as exc:
            st.error(redact(exc))
    confirmed = right.checkbox("Revoke this connection", key=prefix + "revoke_confirm")
    if right.button("Revoke", disabled=not confirmed, key=prefix + "revoke"):
        try:
            registry.revoke(record["id"])
            st.rerun()
        except Exception as exc:
            st.error(redact(exc))


def panel(registry, project_id=None):
    """Render setup without accessing credentials through ordinary chat."""
    try:
        _active_project(registry, project_id)
    except ValueError as exc:
        st.info(str(exc))
        return
    prefix = "setup_" + (project_id or "global") + "_"
    labels = {"model": "Model", "hpc": "HPC", "software": "Software"}
    kind = st.selectbox("Set up", assistant.KINDS, format_func=labels.get, key=prefix + "kind")
    records = registry.environments() if kind == "software" else registry.list(kind=kind)
    records = {item["id"]: item for item in records if not item.get("deleted")}
    if requested := st.session_state.pop(prefix + kind + "next_selection", None):
        st.session_state[prefix + kind + "selected"] = requested
    selected = st.selectbox("Connection" if kind != "software" else "Environment", [None, *records],
                            format_func=lambda identifier: records[identifier]["name"] if identifier else "New",
                            key=prefix + kind + "selected")
    state_key = prefix + "state"
    record = records.get(selected)
    identity = (kind, selected, record.get("revision") if record else None)
    if st.session_state.get(prefix + "identity") != identity:
        st.session_state[prefix + "identity"] = identity
        st.session_state.pop(state_key, None)
        for key in list(st.session_state):
            if key.startswith(prefix + kind + "answer_") or key in {prefix + kind + "advanced", prefix + kind + "edit"}:
                st.session_state.pop(key, None)
    if record is not None:
        if kind == "software":
            st.json(record["config"])
            return
        _connection_controls(registry, record, prefix + record["id"] + "_", project_id)
        if record.get("revoked"):
            return
    if state_key not in st.session_state:
        st.session_state[state_key] = assistant.begin(kind, record)
    state = st.session_state[state_key]
    for message in state["history"][-12:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = assistant.next_question(state)
    if question is None:
        st.json({"name": state["name"], **state["settings"]})
        if st.button("Save environment" if kind == "software" else "Save connection", key=prefix + "save"):
            try:
                result = assistant.save(state, registry, project_id=project_id)
                st.session_state[prefix + kind + "next_selection"] = result["id"]
                st.rerun()
            except Exception as exc:
                st.error(redact(exc))

        fields = {item["field"]: item for item in assistant.questions(state)}
        field = st.selectbox("Edit", [None, *fields], format_func=lambda item: fields[item]["label"] if item else "Choose a field",
                             key=prefix + kind + "edit")
        question = fields.get(field)
    if question is not None:
        field = question["field"]
        st.write(question["label"])
        widget_key = prefix + kind + "answer_" + field
        options = question["options"]
        if field == "connection_id":
            available = {item["id"]: item for item in registry.list(kind="hpc") if not item.get("revoked") and not item.get("deleted")}
            if not available:
                st.info("Set up an HPC connection first.")
                return
            response = st.selectbox("HPC connection", list(available), format_func=lambda item: available[item]["name"], key=widget_key)
        elif options:
            current = assistant.value(state, field)
            response = st.selectbox(question["label"], options, index=options.index(current) if current in options else 0,
                                    format_func=lambda option: _option_label(field, option),
                                    key=widget_key, label_visibility="collapsed")
        else:
            response = st.text_input(question["label"], value=assistant.value(state, field) or question["default"] or "",
                                     key=widget_key, label_visibility="collapsed")
        if st.button("Continue", key=prefix + "continue"):
            try:
                st.session_state[state_key] = assistant.answer(state, field, response)
                st.rerun()
            except Exception as exc:
                st.error(redact(exc))
    with st.expander("Advanced"):
        raw = st.text_area("Additional settings (JSON)", "{}", key=prefix + kind + "advanced")
        if st.button("Apply settings", key=prefix + "apply_advanced"):
            try:
                st.session_state[state_key] = assistant.advanced(state, json.loads(raw))
                st.rerun()
            except Exception as exc:
                st.error(redact(exc))


# ---------------------------------------------------------------------------
# Conversational setup used by the project workspace
# ---------------------------------------------------------------------------

_PROMPTS = {
    "connection_id": "Which cluster should run this software?",
    "engine": "Which software do you want to use?",
    "config.partition": "Which partition or queue should it use?",
    "config.vasp_command": "What command starts VASP on this cluster?",
    "config.potcar_root": "Where is your licensed POTCAR library?",
    "config.qe_command": "What command starts Quantum ESPRESSO?",
    "config.qe_pseudo_root": "Where are your UPF pseudopotentials?",
    "config.command": "What command starts this program?",
    "config.tasks": "How many MPI tasks should a job use?",
    "config.nodes": "How many nodes?",
    "config.walltime": "What time limit should a job have (HH:MM:SS)?",
}


def parse_address(text):
    """Read `user@host`, `ssh -p 2222 user@host`, `user@host:2222` or a bare host."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter the cluster's SSH address.")
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        raise ValueError("Enter the cluster's SSH address.") from None
    if parts and parts[0] == "ssh":
        parts = parts[1:]
    port, user, positional = 22, None, []
    index = 0
    while index < len(parts):
        item = parts[index]
        if item in {"-p", "-l"} and index + 1 < len(parts):
            if item == "-p":
                port = parts[index + 1]
            else:
                user = parts[index + 1]
            index += 2
            continue
        if item.startswith("-"):
            raise ValueError("Only the -p and -l ssh options are understood here. Enter name@host, or set other options in your SSH config.")
        positional.append(item)
        index += 1
    if len(positional) != 1:
        raise ValueError("Enter one SSH address, for example ssh name@cluster.example.org.")
    host = positional[0]
    if "@" in host:
        user, host = host.rsplit("@", 1)
    if re.fullmatch(r"[^:\[\]]+:\d{1,5}", host):
        host, port = host.rsplit(":", 1)
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError("Use a numeric SSH port.") from None
    if not 1 <= port <= 65535:
        raise ValueError("Use an SSH port between 1 and 65535.")
    if not (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", host) or re.fullmatch(r"\[[0-9A-Fa-f:.%]+\]", host)):
        raise ValueError("Enter a valid SSH host name or address.")
    if user is not None and (not user or re.search(r"[\s\x00-\x1f]", user) or user.startswith("-")):
        raise ValueError("Enter a valid SSH user name.")
    return {"host": host, "user": user, "port": port}


def _chat_begin(kind):
    if kind in {"model", "hpc"}:
        return {"schema_version": 2, "kind": kind, "record_id": None, "detect": None, "needs_code": None,
                "attempted": False, "discovered": False, "pending_host": None, "history": []}
    state = assistant.begin(kind)
    state["name"] = t("Software")
    return state


def _bind_button(registry, record, project_id, key):
    if project_id and record["id"] not in {item["id"] for item in registry.project_connections(project_id)}:
        if st.button(t("Use in this project"), key=key):
            registry.bind(project_id, record["id"], role=record["kind"])
            st.rerun()


def _manage_popover(registry, record, prefix):
    with st.popover(t("Manage connection")):
        name = st.text_input(t("Connection name"), record["name"], key=prefix + "name")
        if st.button(t("Rename"), key=prefix + "rename", disabled=not name.strip()):
            registry.update(record["id"], name=name)
            st.rerun()
        confirmed = st.checkbox(t("Revoke this connection"), key=prefix + "revoke_confirm")
        if st.button(t("Revoke"), key=prefix + "revoke", disabled=not confirmed):
            registry.revoke(record["id"])
            st.rerun()


def _back_to_chat(project_id, key):
    if project_id and st.button(t("Back to chat"), key=key, type="primary"):
        st.session_state["project_page"] = "Chat"
        st.rerun()


def _saved_records(registry, kind, state_key, current):
    records = registry.environments() if kind == "software" else registry.list(kind=kind)
    records = [item for item in records if not item.get("deleted")]
    if not records:
        return
    with st.popover(t("Saved connections") if kind != "software" else t("Saved environments")):
        for record in records:
            if st.button(record["name"], key=current + "open_" + record["id"], use_container_width=True):
                if record.get("revoked"):
                    st.info(t("This connection has been revoked."))
                elif kind == "software":
                    st.session_state[state_key] = assistant.begin(kind, record)
                    st.rerun()
                else:
                    st.session_state[state_key] = {**_chat_begin(kind), "record_id": record["id"]}
                    st.rerun()
        if st.button(t("Add new"), key=current + "new", use_container_width=True):
            st.session_state[state_key] = _chat_begin(kind)
            st.rerun()


# --- Model -----------------------------------------------------------------

def _cli_detection(prefix):
    key = prefix + "cli_detect"
    if key not in st.session_state:
        from . import cli_tools

        st.session_state[key] = cli_tools.detect()
    return st.session_state[key]


def _cli_caption(item):
    from .cli_tools import version_text

    if not item["installed"]:
        return t("Not installed")
    version = version_text(item["version"])
    if item["logged_in"]:
        return f"{version} · " + t("Signed in") + (f" ({item['detail']})" if item["detail"] and item["detail"] != "Signed in" else "")
    if item["logged_in"] is None:
        return f"{version} · " + t("Sign-in state unknown")
    return f"{version} · " + t("Not signed in")


def _model_setup(registry, project_id, prefix, state_key):
    from .providers import PROVIDERS

    state = st.session_state[state_key]
    _saved_records(registry, "model", state_key, prefix)
    if state["record_id"]:
        _model_record(registry, registry.get(state["record_id"]), prefix + state["record_id"] + "_", project_id, state)
        return
    with st.chat_message("assistant"):
        st.write(t("Which model would you like to use?"))
        st.caption(t("A signed-in Codex or Claude Code CLI uses your existing subscription; no API key is needed."))
    detected = _cli_detection(prefix)
    columns = st.columns(len(detected))
    for column, item in zip(columns, detected):
        with column:
            if st.button(item["label"], key=prefix + "cli_" + item["provider"], use_container_width=True,
                         disabled=not item["installed"], type="primary" if item["logged_in"] else "secondary"):
                record = registry.create("model", item["label"], {"provider": item["provider"]})
                if project_id:
                    registry.bind(project_id, record["id"], role="model")
                state["record_id"] = record["id"]
                if item["logged_in"]:
                    with st.spinner(t("Testing the connection…")):
                        registry.probe(record["id"])
                st.rerun()
            st.caption(_cli_caption(item))
            if not item["installed"]:
                st.code(_INSTALL_HINTS[item["provider"]], language="bash")
    if st.button(t("Check again"), key=prefix + "cli_refresh"):
        st.session_state.pop(prefix + "cli_detect", None)
        st.rerun()
    st.caption(t("Or use an API key"))
    provider = st.pills(t("Provider"), _PROVIDER_ORDER, format_func=lambda option: _option_label("provider", option),
                        key=prefix + "provider", label_visibility="collapsed")
    if not provider:
        return
    info = PROVIDERS[provider]
    model = st.text_input(t("Model name"), value=info["model_example"], key=prefix + "model_" + provider,
                          help=t("The exact model name offered by this provider."))
    base_url = info["base_url"] or ""
    if provider in {"qwen", "chat_completions", "local"}:
        base_url = st.text_input(t("API address"), value=info["base_url"] or "", key=prefix + "base_url_" + provider,
                                 placeholder="https://…/v1")
    key = prefix + "api_key"
    if provider != "local":
        st.text_input(t("API key"), type="password", key=key, help=t("Stored in the system keychain, never in project files."))

    def connect():
        secret = st.session_state.pop(key, "")
        try:
            _active_project(registry, project_id)
            if provider != "local" and not secret:
                st.session_state[prefix + "notice"] = (False, t("Enter the API key, then connect."))
                return
            settings = {"provider": provider, "model": model.strip()}
            if provider in {"qwen", "chat_completions", "local"}:
                settings["base_url"] = base_url.strip()
            record = registry.create("model", _option_label("provider", provider), settings)
            if project_id:
                registry.bind(project_id, record["id"], role="model")
            state["record_id"] = record["id"]
            if provider != "local":
                registry.set_secret(record["id"], secret)
            registry.probe(record["id"])
        except Exception as error:
            st.session_state[prefix + "notice"] = (False, str(error))

    st.button(t("Connect"), key=prefix + "api_connect", type="primary", on_click=connect)
    _notice(prefix + "notice")


def _model_record(registry, record, prefix, project_id, state):
    provider = record["settings"]["provider"]
    status = record["connection_status"]
    st.caption(f"{record['name']} · {_status_label(record)}")
    _notice(prefix[:-len(record["id"]) - 1] + "notice")
    if record.get("revoked"):
        st.info(t("This connection has been revoked."))
        return
    detail = _probe_detail(record)
    if status == "connected":
        st.success(t("The model answered. You can start chatting."))
    elif detail:
        (st.warning if record["auth_status"] == "required" else st.error)(detail)
    if provider in _CLI_PROVIDERS and status != "connected":
        st.write(t("Sign in with the CLI in a terminal, then test the connection again:"))
        st.code(_SIGN_IN_COMMANDS[provider], language="bash")
    if provider in _CLI_PROVIDERS and status == "error" and "too old" in detail:
        st.code(_INSTALL_HINTS[provider], language="bash")
    _credential_controls(registry, record, prefix, project_id)
    if st.button(t("Test connection"), key=prefix + "test", type="secondary" if status == "connected" else "primary"):
        with st.spinner(t("Testing the connection…")):
            registry.probe(record["id"])
        st.rerun()
    _bind_button(registry, record, project_id, prefix + "bind")
    _manage_popover(registry, record, prefix)
    if status == "connected":
        _back_to_chat(project_id, prefix + "back")


# --- Cluster ---------------------------------------------------------------

def _cluster_setup(registry, project_id, prefix, state_key):
    state = st.session_state[state_key]
    _saved_records(registry, "hpc", state_key, prefix)
    if state["record_id"]:
        _cluster_record(registry, registry.get(state["record_id"]), prefix, project_id, state)
        return
    with st.chat_message("assistant"):
        if state["pending_host"]:
            st.write(t("What is your username on this cluster?"))
        else:
            st.write(t("What is your cluster's SSH address? You can paste ssh user@host."))
            st.caption(t("The app first tries your SSH keys, then asks for a password or login code only if the cluster requires one."))
    answer = st.chat_input(t("Username") if state["pending_host"] else "ssh name@cluster.example.org", key=prefix + "answer")
    if not answer:
        return
    try:
        if state["pending_host"]:
            address = {**state["pending_host"], "user": answer.strip()}
            parse_address(f"{address['user']}@{address['host']}:{address['port']}")
        else:
            address = parse_address(answer)
            if not address["user"]:
                state["pending_host"] = address
                st.rerun()
        record = registry.create("hpc", address["host"], {"host": address["host"], "user": address["user"],
                                                            "port": address["port"], "auth_mode": "agent"})
        if project_id:
            registry.bind(project_id, record["id"], role="hpc")
        state.update(record_id=record["id"], pending_host=None, detect=None, attempted=False)
        st.rerun()
    except ValueError as error:
        st.error(redact(str(error)))


def _sign_in(registry, record_id, prefix, state, password=None, code=None):
    """Save a freshly typed password, then sign in. Runs inside a button callback."""
    from .connections import PASSWORD_MODES
    from .transport import AuthenticationError, VerificationRequired

    try:
        record = registry.get(record_id)
        if password is not None:
            if not password:
                st.session_state[prefix + "notice"] = (False, t("Enter your cluster password."))
                return
            if record["settings"]["auth_mode"] not in PASSWORD_MODES:
                registry.update(record_id, settings={"auth_mode": "password"})
            registry.set_secret(record_id, password)
        if code is not None and not code.strip():
            st.session_state[prefix + "notice"] = (False, t("Enter the login code."))
            return
        registry.connect(record_id, code=(code or "").strip())
        state["needs_code"] = None
        state["attempted"] = True
        registry.probe(record_id)
        if registry.get(record_id)["connection_status"] != "connected":
            st.session_state[prefix + "notice"] = (False, _probe_detail(registry.get(record_id)) or t("The cluster test did not finish."))
    except VerificationRequired as error:
        state["needs_code"] = error.prompt or t("Login code")
        state["attempted"] = True
    except AuthenticationError as error:
        state["attempted"] = True
        if code is not None:
            st.session_state[prefix + "notice"] = (False, t("The cluster rejected the login code. Enter a fresh code."))
        else:
            state["detect"] = {"status": "password", "detail": str(error)}
            st.session_state[prefix + "notice"] = (False, t("The cluster rejected the password. Check it and try again."))
    except Exception as error:
        state["attempted"] = True
        st.session_state[prefix + "notice"] = (False, str(error))


def _password_form(registry, record, prefix, state):
    key = prefix + "password"
    with st.chat_message("assistant"):
        st.write(t("Enter your password for this cluster."))
        st.caption(t("It is stored in the system keychain and used only for this connection."))
    st.text_input(t("SSH password"), type="password", key=key, label_visibility="collapsed")

    def submit():
        _sign_in(registry, record["id"], prefix, state, password=st.session_state.pop(key, ""))

    st.button(t("Sign in"), key=prefix + "sign_in", type="primary", on_click=submit)


def _code_form(registry, record, prefix, state):
    key = prefix + "code"
    with st.chat_message("assistant"):
        st.write(t("This cluster also asks for a login code."))
        if state["needs_code"]:
            st.code(state["needs_code"], language="text")
        from .transport import LOGIN_PERSIST_SECONDS
        st.caption(t("Enter the code from your authenticator or message. The signed-in session is reused for about {hours} hours of activity, so the code is not needed for every job.").format(hours=LOGIN_PERSIST_SECONDS // 3600))
    st.text_input(t("Login code"), type="password", key=key, label_visibility="collapsed", placeholder="123456")

    def submit():
        _sign_in(registry, record["id"], prefix, state, code=st.session_state.pop(key, ""))

    st.button(t("Submit code"), key=prefix + "submit_code", type="primary", on_click=submit)


def _terminal_sign_in(registry, record, prefix):
    """The visible-terminal path remains for clusters with unusual login flows."""
    with st.expander(t("Sign in through a terminal instead")):
        st.caption(t("Opens your terminal with the right ssh command. Passwords and codes stay in the terminal; the app reuses the session afterwards."))
        if st.button(t("Open terminal sign-in"), key=prefix + "terminal"):
            from .ssh_auth import launch

            try:
                st.session_state[prefix + "terminal_result"] = launch(registry, record["id"])
            except Exception as error:
                st.error(redact(str(error)))
        if result := st.session_state.get(prefix + "terminal_result"):
            st.info(result["message"])
            if result["status"] == "manual":
                st.code(result["command"], language="powershell" if not result["reusable"] else "bash")
            if st.button(t("I have signed in, test the connection"), key=prefix + "terminal_test"):
                with st.spinner(t("Testing the connection…")):
                    registry.probe(record["id"])
                st.rerun()


def _environments_panel(registry, record, prefix, state):
    environments = registry.environments(record["id"])
    discovery = (record.get("probe") or {}).get("discovery")
    if not environments and discovery is None and not state["discovered"]:
        state["discovered"] = True
        try:
            with st.spinner(t("Reading partitions, modules and potentials from the cluster…")):
                registry.discover_environments(record["id"])
        except ValueError as error:
            st.warning(redact(str(error)))
        st.rerun()
    if environments:
        st.write(t("Software found on this cluster:"))
        for environment in environments:
            st.caption(f"• {environment['name']} · " + t("partition") + f" {environment['config'].get('partition', '')}")
    if discovery:
        missing = {}
        if discovery.get("vasp_found") and not any(item["engine"] == "vasp" for item in environments):
            missing["vasp"] = True
        if missing.get("vasp"):
            st.info(t("VASP was found, but no POTCAR folder. Enter the path of your licensed PAW potentials to finish the VASP environment."))
            potcar = st.text_input("POTCAR", key=prefix + "potcar", placeholder="/home/name/potpaw_PBE", label_visibility="collapsed")
            if st.button(t("Add VASP environment"), key=prefix + "add_vasp", disabled=not potcar.strip()):
                try:
                    registry.discover_environments(record["id"], potcar_root=potcar.strip())
                    st.rerun()
                except ValueError as error:
                    st.error(redact(str(error)))
        if not environments and not discovery.get("vasp_found") and not discovery.get("qe_found"):
            st.info(t("Neither VASP nor Quantum ESPRESSO was found on the login node. Quantum ESPRESSO is open source: the app can build it in your cluster storage below."))
        if (record.get("probe") or {}).get("qe_install") or (not discovery.get("qe_found") and not any(item["engine"] == "qe" for item in environments)):
            _qe_install_panel(registry, record, prefix)
    if st.button(t("Detect again"), key=prefix + "rediscover"):
        try:
            with st.spinner(t("Reading partitions, modules and potentials from the cluster…")):
                registry.discover_environments(record["id"])
        except ValueError as error:
            st.error(redact(str(error)))
        st.rerun()


def _qe_install_panel(registry, record, prefix):
    """Managed Quantum ESPRESSO build: detect a toolchain, review the plan, submit one build job, follow it."""
    install = (record.get("probe") or {}).get("qe_install") or {}
    status = install.get("status")
    with st.container(border=True):
        st.markdown("**" + t("Install Quantum ESPRESSO") + "**")
        if status in {None, "planned"} or (status == "failed" and st.session_state.get(prefix + "qe_replan")):
            st.caption(t("Builds QE from the official source in your own cluster storage with a single scheduler job. No administrator access is needed."))
            partition = st.text_input(t("Partition for the build job"), install.get("partition") or ((record.get("probe") or {}).get("discovery") or {}).get("suggested", {}).get("partition", ""),
                                      key=prefix + "qe_partition")
            account = st.text_input(t("Account (optional)"), install.get("account") or ((record.get("probe") or {}).get("discovery") or {}).get("suggested", {}).get("account", ""),
                                    key=prefix + "qe_account")
            modules = st.text_area(t("Module commands (optional, one per line)"), "\n".join(install.get("setup_commands") or []),
                                   key=prefix + "qe_modules", placeholder="module load gcc openmpi")
            cpus = st.number_input(t("Build CPUs"), 1, 64, int(install.get("plan", {}).get("build_jobs", 8)), key=prefix + "qe_cpus")
            if st.button(t("Plan the build"), key=prefix + "qe_plan"):
                try:
                    with st.spinner(t("Detecting compilers and MPI on the cluster…")):
                        registry.qe_install_plan(record["id"], build_cpus=int(cpus), partition=partition.strip() or None, account=account.strip(),
                                                 setup_commands=[line.strip() for line in modules.splitlines() if line.strip()] or None)
                    st.session_state.pop(prefix + "qe_replan", None)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(redact(str(error)))
            if status == "planned":
                plan = install["plan"]
                st.write(f"QE {plan['version']} · {plan['build_jobs']} CPUs · {plan['walltime']} · {install['partition']}"
                         + (f" · {install['account']}" if install.get("account") else ""))
                st.code("\n".join(install.get("setup_commands") or [t("no module commands")]) + "\n" + plan["prefix"], language="text")
                for note in install.get("toolchain", {}).get("notes", []):
                    st.caption(note)
                if install.get("toolchain", {}).get("ready"):
                    agreed = st.checkbox(t("Use this account and allocation to build QE."), key=prefix + "qe_agree")
                    if st.button(t("Start the build"), key=prefix + "qe_start", type="primary", disabled=not agreed):
                        try:
                            with st.spinner(t("Staging sources and submitting the build job…")):
                                registry.qe_install_start(record["id"], install["plan_id"])
                            st.rerun()
                        except (ValueError, OSError) as error:
                            st.error(redact(str(error)))
                else:
                    st.warning(t("No MPI Fortran compiler was found. Enter the module commands your cluster documents and plan again."))
            return
        st.write(t("Build status: ") + t(str(status)) + (f" · job {install.get('job_id')}" if install.get("job_id") else "")
                 + (f" · {install.get('scheduler_state')}" if install.get("scheduler_state") else ""))
        if install.get("error"):
            st.error(redact(str(install["error"])))
        if status == "installed":
            st.success(t("Quantum ESPRESSO is installed and its environment is ready."))
            st.code(install.get("bin_dir") or "", language="text")
            return
        left, right = st.columns(2)
        if left.button(t("Check the build"), key=prefix + "qe_check"):
            try:
                with st.spinner(t("Reading the build status…")):
                    registry.qe_install_status(record["id"])
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(redact(str(error)))
        if status == "failed" and right.button(t("Plan again"), key=prefix + "qe_replan_button"):
            st.session_state[prefix + "qe_replan"] = True
            st.rerun()
        if install.get("logs"):
            st.caption(t("Build logs on the cluster: ") + ", ".join(install["logs"].values()))


def _cluster_record(registry, record, prefix, project_id, state):
    from .connections import PASSWORD_MODES

    settings = record["settings"]
    current = prefix + record["id"] + "_"
    st.caption(f"{settings['user']}@{settings['host']}" + (f":{settings['port']}" if settings["port"] != 22 else "")
               + f" · {_status_label(record)}")
    _notice(prefix + "notice")
    _notice(current + "notice")
    if record.get("revoked"):
        st.info(t("This connection has been revoked."))
        return
    if record["connection_status"] == "connected":
        st.success(t("Connected to the cluster."))
        with st.popover(t("Work folder: ") + (settings["remote_base"] or t("not set"))):
            folder = st.text_input(t("Work folder on the cluster"), settings["remote_base"] or "", key=current + "remote_base")
            if st.button(t("Save folder"), key=current + "save_remote_base", disabled=not folder.strip()):
                try:
                    registry.record_settings(record["id"], remote_base=folder.strip())
                    st.rerun()
                except ValueError as error:
                    st.error(redact(str(error)))
        with st.popover(t("Scheduler")):
            scheduler = st.selectbox(t("Scheduler"), ["auto", "slurm", "pbs"],
                                     index=["auto", "slurm", "pbs"].index(settings["scheduler"]),
                                     format_func=lambda value: _option_label("scheduler", value), key=current + "scheduler")
            if st.button(t("Save scheduler"), key=current + "save_scheduler"):
                registry.record_settings(record["id"], scheduler=scheduler)
                state["discovered"] = False
                st.rerun()
        if settings["scheduler"] == "auto" and len((record.get("probe") or {}).get("schedulers", [])) != 1:
            st.warning(t("Choose Slurm or PBS in the cluster settings before continuing."))
        else:
            _environments_panel(registry, record, current, state)
        left, right = st.columns(2)
        if left.button(t("Test connection"), key=current + "test"):
            with st.spinner(t("Testing the connection…")):
                registry.probe(record["id"])
            st.rerun()
        with right:
            _manage_popover(registry, record, current)
        _bind_button(registry, record, project_id, current + "bind")
        _back_to_chat(project_id, current + "back")
        return
    # Not connected yet: find out what the cluster wants, asking for as little as possible.
    probe = record.get("probe") or {}
    if state["needs_code"] or probe.get("needs") == "code":
        if not state["needs_code"]:
            state["needs_code"] = probe.get("prompt") or t("Login code")
        _code_form(registry, record, current, state)
    elif (settings["auth_mode"] == "code" or (settings["auth_mode"] == "password" and record["has_secret"])) and not state["attempted"]:
        with st.spinner(t("Signing in…")):
            _sign_in(registry, record["id"], prefix, state)
        st.rerun()
    elif settings["auth_mode"] in PASSWORD_MODES or (state["detect"] or {}).get("status") == "password" or probe.get("needs") == "password":
        _password_form(registry, record, current, state)
    elif settings["auth_mode"] == "interactive":
        st.info(t("This connection signs in through a terminal."))
    elif state["detect"] is None:
        with st.spinner(t("Checking how this cluster signs you in…")):
            try:
                state["detect"] = registry.detect_auth(record["id"])
            except Exception as error:
                state["detect"] = {"status": "error", "detail": str(error)}
        if state["detect"]["status"] == "connected":
            with st.spinner(t("Reading the cluster…")):
                registry.probe(record["id"])
        elif state["detect"]["status"] == "code":
            state["needs_code"] = state["detect"].get("prompt") or t("Login code")
        st.rerun()
    else:
        detect = state["detect"]
        if detect["status"] == "connected":
            st.info(t("Your SSH key signed in, but the cluster test did not finish.") + " " + _probe_detail(record))
        elif detect["status"] == "key":
            st.warning(redact(detect["detail"]))
            identity = st.text_input(t("Private key file"), settings.get("identity_file") or "", key=current + "identity",
                                     placeholder="~/.ssh/id_ed25519")
            if st.button(t("Use this key"), key=current + "save_identity", disabled=not identity.strip()):
                try:
                    registry.update(record["id"], settings={"auth_mode": "key", "identity_file": identity.strip()})
                    state.update(detect=None)
                    st.rerun()
                except ValueError as error:
                    st.error(redact(str(error)))
        else:
            st.error(redact(detect["detail"]))
        if st.button(t("Try again"), key=current + "retry"):
            state.update(detect=None, attempted=False)
            st.rerun()
    _terminal_sign_in(registry, record, current)
    _manage_popover(registry, record, current)


# --- Software --------------------------------------------------------------

def _software_defaults(registry, state):
    """Prefill answers from the cluster's own discovery when it exists."""
    connection_id = state["settings"].get("connection_id")
    if not connection_id:
        return {}
    try:
        record = registry.get(connection_id)
    except ValueError:
        return {}
    discovery = (record.get("probe") or {}).get("discovery") or {}
    suggested = discovery.get("suggested", {})
    base = record["settings"].get("remote_base") or ""
    launcher = "mpiexec" if record["settings"].get("scheduler") == "pbs" else "srun"
    return {"config.partition": suggested.get("partition", ""), "config.potcar_root": suggested.get("potcar_root", ""),
            "config.vasp_command": suggested.get("vasp_command", launcher + " vasp_std"), "config.qe_command": launcher + " pw.x",
            "config.qe_pseudo_root": (base.rstrip("/") + "/pseudos") if base else "", "config.command": "python3"}


def _software_setup(registry, project_id, prefix, state_key):
    state = st.session_state[state_key]
    _saved_records(registry, "software", state_key, prefix)
    if state["record_id"]:
        record = registry.environment(state["record_id"])
        st.success(record["name"] + " " + t("is saved."))
        st.caption(_option_label("engine", record["engine"]) + " · " + record["config"].get("partition", ""))
        st.json(record["config"])
        _back_to_chat(project_id, prefix + "back")
        return
    question = assistant.next_question(state)
    if question is None:
        with st.chat_message("assistant"):
            st.write(t("Save this software environment?"))
            st.json({"name": state["name"], **state["settings"]})
        if st.button(t("Save environment"), key=prefix + "save", type="primary"):
            try:
                assistant.save(state, registry, project_id)
                st.rerun()
            except Exception as error:
                st.error(redact(str(error)))
        with st.popover(t("Options")):
            raw = st.text_area(t("Additional settings (JSON)"), "{}", key=prefix + "advanced")
            if st.button(t("Apply settings"), key=prefix + "advanced_save"):
                try:
                    st.session_state[state_key] = assistant.advanced(state, json.loads(raw))
                    st.rerun()
                except Exception as error:
                    st.error(redact(str(error)))
        return
    for message in state["history"][-6:]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
    field = question["field"]
    with st.chat_message("assistant"):
        st.write(t(_PROMPTS.get(field, question["label"])))
    options = question["options"]
    available = {}
    if field == "connection_id":
        available = {item["id"]: item for item in registry.list(kind="hpc") if not item["revoked"] and not item["deleted"]}
        options = list(available)
        if not options:
            st.caption(t("Connect a cluster first."))
            return
    if options:
        columns = st.columns(min(3, len(options)))
        for index, option in enumerate(options):
            label = available[option]["name"] if field == "connection_id" else _option_label(field, option)
            if columns[index % len(columns)].button(label, key=prefix + field + "_" + option, use_container_width=True):
                updated = assistant.answer(state, field, option)
                if field == "engine":
                    updated["name"] = _option_label(field, option)
                st.session_state[state_key] = updated
                st.rerun()
        return
    default = _software_defaults(registry, state).get(field) or question["default"] or ""
    if default:
        st.caption(t("Suggested: ") + str(default))
    if response := st.chat_input(str(default) if default else t(_PROMPTS.get(field, question["label"])), key=prefix + "answer"):
        try:
            st.session_state[state_key] = assistant.answer(state, field, response)
            st.rerun()
        except ValueError as error:
            st.error(redact(str(error)))
    if default and st.button(t("Use suggested value"), key=prefix + "use_default_" + field):
        st.session_state[state_key] = assistant.answer(state, field, str(default))
        st.rerun()


def chat_panel(registry, project_id=None):
    """Guided setup: detect what is available, then ask only for what is missing."""
    try:
        _active_project(registry, project_id)
        prefix = "setup_chat_" + (project_id or "global") + "_"
        kinds = {"model": t("Model"), "hpc": t("Cluster"), "software": t("Software")}
        if requested := st.session_state.pop(prefix + "kind_requested", None):
            st.session_state[prefix + "kind"] = requested
        kind = st.pills(t("Connect"), list(kinds), default="model", format_func=kinds.get, key=prefix + "kind",
                        label_visibility="collapsed") or "model"
        current = prefix + kind + "_"
        state_key = current + "state"
        if state_key not in st.session_state:
            st.session_state[state_key] = _chat_begin(kind)
        if kind == "model":
            _model_setup(registry, project_id, current, state_key)
        elif kind == "hpc":
            _cluster_setup(registry, project_id, current, state_key)
        else:
            _software_setup(registry, project_id, current, state_key)
    except Exception as error:
        st.error(redact(str(error)))
