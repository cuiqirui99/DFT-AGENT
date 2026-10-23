"""Guided connection setup for the project interface.

Credentials only ever pass through secure fields and on-click callbacks that
remove them from the session before the page renders again. Nothing here
submits a job; cluster access is limited to sign-in and read-only probes.
"""

import json
import re
import shlex
import time

import streamlit as st

from . import setup_assistant as assistant
from .i18n import t
from .secret_guard import redact

_PROVIDER_ORDER = ("responses", "anthropic", "qwen", "deepseek", "glm", "grok", "chat_completions", "local")
_CLI_PROVIDERS = {"codex", "claude_cli"}


def _option_label(field, option):
    if field == "provider":
        from .providers import PROVIDERS

        label = PROVIDERS.get(option, {}).get("label", {"claude_cli": "Claude Code", "local": "Local model"}.get(option, option))
        # Product names stay as they are; the two generic entries are translated.
        return t(label) if option in {"chat_completions", "local"} else label
    return {"auth_mode": {"agent": t("SSH agent"), "key": t("SSH key"), "password": t("Password"),
                          "code": t("Password and login code"), "interactive": t("Interactive SSH")},
            "engine": {"vasp": "VASP", "qe": "Quantum ESPRESSO", "abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk",
                       "wannier90": "Wannier90", "python": "Python"},
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
            if record["kind"] == "model":
                from .model_connect import clean_key

                value = clean_key(value)
            registry.set_secret(record["id"], value)
            st.session_state[notice] = (True, t("Credential saved in the system keychain."))
        except Exception:
            st.session_state[notice] = (False, t("Could not save the credential. Check the system keychain."))

    # A form sends a key typed but not yet confirmed with Enter.
    with st.form(prefix + "secret_form", border=False):
        st.text_input(t("API key") if record["kind"] == "model" else t("SSH password"), type="password", key=key)
        st.form_submit_button(t("Save credential"), key=prefix + "save_secret", on_click=save_secret)
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
    "connection_id": "Cluster",
    "engine": "Program",
    "config.partition": "Partition or queue",
    "config.vasp_command": "VASP launch command",
    "config.potcar_root": "POTCAR folder",
    "config.qe_command": "Quantum ESPRESSO launch command",
    "config.qe_pseudo_root": "UPF folder",
    "config.command": "Launch command",
    "config.tasks": "MPI tasks per job",
    "config.nodes": "Nodes",
    "config.walltime": "Time limit (HH:MM:SS)",
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

# Where each API provider issues keys.
_KEY_PAGES = {"responses": "https://platform.openai.com/api-keys", "anthropic": "https://platform.claude.com/settings/keys",
              "qwen": "https://www.alibabacloud.com/help/en/model-studio/get-api-key",
              "deepseek": "https://platform.deepseek.com/api_keys", "glm": "https://z.ai/manage-apikey/apikey-list",
              "grok": "https://console.x.ai/"}
_STEPS = (("install", "Install"), ("sign_in", "Sign in"), ("connect", "Connect"))
# After a terminal window opens, its CLI is checked again every few seconds for this long.
_WATCH_SECONDS = 900


def _cli_detection(prefix, providers=()):
    """The detected CLIs, kept for the page; providers are checked again."""
    from . import cli_tools

    key = prefix + "cli_detect"
    if key not in st.session_state:
        st.session_state[key] = cli_tools.detect()
    elif providers:
        fresh = {item["provider"]: item for item in cli_tools.detect(tuple(providers))}
        st.session_state[key] = [fresh.get(item["provider"], item) for item in st.session_state[key]]
    return st.session_state[key]


def _terminal_messages(provider):
    from .cli_tools import LABELS

    name, account = LABELS[provider], {"codex": t("your ChatGPT account"), "claude_cli": t("your Claude account")}[provider]
    return {"title": t("DFT Agent: setting up {name}").format(name=name),
            "installing": t("Installing {name} with its official installer…").format(name=name),
            "failed": t("The installation did not finish. Check the internet connection and try again from DFT Agent, or use an API key instead."),
            "missing": t("{name} is not where its installer puts it. Try again from DFT Agent.").format(name=name),
            "signing_in": t("Signing in: a page opens in your browser. Sign in there with {account}.").format(account=account),
            "sign_in_failed": t("Sign-in did not finish. Try again from DFT Agent."),
            "done": t("Done. Close this window and go back to DFT Agent; it notices by itself."),
            "close": t("Press Return to close this window.")}


def _cli_actions(prefix, item, *, update=False, watched=True):
    """The next step of a CLI that is missing, signed out or too old: one button that opens a
    terminal doing it, and the same commands to paste by hand. watched: the page checks the CLI
    again by itself, else the person tests the connection."""
    from . import cli_setup

    provider = item["provider"]
    step = "update" if update else cli_setup.next_step(item)
    label = {"install": t("Install and sign in"), "sign_in": t("Sign in"), "update": t("Update {name}").format(name=item["label"])}[step]
    if st.button(label, key=prefix + "terminal_" + provider, type="primary", use_container_width=True):
        try:
            cli_setup.open_terminal(provider, item, _terminal_messages(provider), update=update)
            st.session_state[prefix + "cli_watch"] = {"provider": provider, "until": time.time() + _WATCH_SECONDS}
        except OSError as error:
            st.session_state[prefix + "cli_notice"] = (False, t(str(error)))
        st.rerun()
    if (st.session_state.get(prefix + "cli_watch") or {}).get("provider") != provider:
        st.caption(t("Opens a terminal window that does this for you. You only sign in on the page it opens in your browser."))
    elif watched:
        st.caption(t("A terminal window is open. Follow what it says; this page updates by itself when you are done."))
    else:
        st.caption(t("A terminal window is open. Follow what it says, then click Test connection."))
    name = cli_setup.system()
    with st.expander(t("Do it yourself in a terminal")):
        st.caption({"macos": t("Open Terminal: press Command and Space, type Terminal, press Return."),
                    "windows": t("Open PowerShell: open the Start menu, type PowerShell, press Enter."),
                    "linux": t("Open a terminal, often with Ctrl+Alt+T.")}[name]
                   + " " + t("Paste each line, press Return and follow what it asks."))
        for line in cli_setup.commands(provider, item, update=update):
            st.code(line, language="powershell" if name == "windows" else "bash", wrap_lines=True)


def _cli_card(registry, project_id, prefix, state, item):
    from . import cli_setup
    from .cli_tools import version_text

    step = cli_setup.next_step(item)
    order = [name for name, _ in _STEPS]
    with st.container(border=True):
        st.markdown(f"**{item['label']}**")
        st.caption(t(cli_setup.ACCOUNTS[item["provider"]]))
        lines = []
        for name, label in _STEPS:
            detail = ""
            if name == "install" and item["installed"]:
                detail = " · " + version_text(item["version"])
            elif name == "sign_in" and item["logged_in"] and item.get("detail") not in (None, "Signed in"):
                detail = " · " + item["detail"]
            if order.index(name) < order.index(step):
                lines.append(f":green[:material/check_circle:] {t(label)}:gray[{detail}]")
            elif name == step:
                lines.append(f":material/arrow_circle_right: **{t(label)}**")
            else:
                lines.append(f":gray[:material/radio_button_unchecked: {t(label)}]")
        st.markdown("  \n".join(lines))
        if step != "connect":
            _cli_actions(prefix, item)
        elif st.button(t("Connect {name}").format(name=item["label"]), key=prefix + "cli_" + item["provider"], type="primary",
                       use_container_width=True):
            record = registry.create("model", item["label"], {"provider": item["provider"]})
            if project_id:
                registry.bind(project_id, record["id"], role="model")
            state["record_id"] = record["id"]
            with st.spinner(t("Testing the connection…")):
                registry.probe(record["id"])
            st.rerun()


def _cli_cards(registry, project_id, prefix, state):
    """One card per CLI with its three steps. While a terminal window works on one, the card
    checks it again every few seconds instead of waiting for Check again."""
    watch = st.session_state.get(prefix + "cli_watch")
    if watch and watch["until"] < time.time():
        st.session_state.pop(prefix + "cli_watch")
        watch = None

    @st.fragment(run_every=4 if watch else None)
    def cards():
        current = st.session_state.get(prefix + "cli_watch")
        if current and current["until"] < time.time():
            st.session_state.pop(prefix + "cli_watch")
            st.rerun()
        detected = _cli_detection(prefix, [current["provider"]] if current else ())
        if current and next(item for item in detected if item["provider"] == current["provider"])["logged_in"]:
            st.session_state.pop(prefix + "cli_watch")
            st.rerun()
        _notice(prefix + "cli_notice")
        for column, item in zip(st.columns(len(detected)), detected):
            with column:
                _cli_card(registry, project_id, prefix, state, item)
        if current:
            if st.button(t("Stop checking"), key=prefix + "cli_stop"):
                st.session_state.pop(prefix + "cli_watch", None)
                st.rerun()
            st.caption(t("Close the terminal window to cancel sign-in."))
    cards()
    if st.button(t("Check again"), key=prefix + "cli_refresh"):
        st.session_state.pop(prefix + "cli_detect", None)
        st.rerun()


def _model_setup(registry, project_id, prefix, state_key):
    state = st.session_state[state_key]
    _saved_records(registry, "model", state_key, prefix)
    if state["record_id"]:
        _model_record(registry, registry.get(state["record_id"]), prefix + state["record_id"] + "_", project_id, state)
        return
    st.caption(t("DFT Agent understands your requests through an AI model. The easiest way is a ChatGPT or Claude plan you "
                 "already have: install its official command-line tool and sign in once."))
    _cli_cards(registry, project_id, prefix, state)
    st.caption(t("No ChatGPT or Claude plan, or the installation does not work? Use an API key instead; the provider bills by use."))
    provider = st.pills(t("Provider"), _PROVIDER_ORDER, format_func=lambda option: _option_label("provider", option),
                        key=prefix + "provider", label_visibility="collapsed")
    if provider == "local":
        _local_setup(registry, project_id, prefix, state)
    elif provider:
        _api_setup(registry, project_id, prefix, state, provider)


def _model_advice(provider, code, *, owner=None, host="", model=""):
    """A failed model connection in plain words, with the links that fix it."""
    from .model_connect import GUIDES

    label, guide = _option_label("provider", provider), GUIDES.get(provider, {})
    links = []
    if code == "empty":
        text = t("Paste the API key first.")
    elif code == "address":
        text = t("Check the API address. Use an HTTP(S) URL without credentials or query parameters; local models must run on this computer.")
    elif code == "key":
        text = t("{provider} did not accept this key. Copy it again in full; a key works only on the platform where it was created.").format(provider=label)
        if owner:
            text += " " + t("It looks like a {owner} key: choose {owner} above.").format(owner=_option_label("provider", owner))
        links = [(t("Create a new key"), guide.get("keys"))]
    elif code == "credit":
        text = t("The key works, but the account has no API credit. Top it up, then test again; chat memberships do not include API use.")
        links = [(t("Top up"), guide.get("billing"))]
    elif code == "model":
        text = (t("This account cannot use the model {model}. Choose another under Change the model.").format(model=model) if model
                else t("Enter the model name your provider gave you."))
    elif code == "denied":
        text = t("The account may not use this model. Choose another under Change the model, or check the account's permissions.")
    elif code in {"network", "timeout"}:
        text = (t("Nothing answered at this address on your computer. Start the model server, then try again.") if provider == "local"
                else t("Cannot reach {host}. Check the internet connection; some networks need a proxy for this service, and a provider closer to you may work better.").format(
                    host=host or label))
    elif code == "rate":
        text = t("The service is limiting requests for now. Wait a minute, then test again.")
    elif code == "format":
        text = t("This model does not give the structured answers DFT Agent needs. Choose another under Change the model.")
    elif code == "server":
        text = t("The service has a temporary problem. Try again in a few minutes.")
    else:
        text = t("The connection did not work. Check the key and the model name, then try again.")
    return text, [(name, url) for name, url in links if url]


def _show_advice(provider, advice):
    text, links = _model_advice(provider, advice["code"], owner=advice.get("owner"), host=advice.get("host", ""),
                                model=advice.get("model", ""))
    st.error(text)
    for column, (name, url) in zip(st.columns(max(1, len(links))), links):
        with column:
            st.link_button(name, url)


def _api_connect(registry, project_id, prefix, state, provider, secret, model, base_url):
    """Check the key for free, then save the connection and test it; False with advice when the key cannot work."""
    from . import model_connect
    from .i18n import current_language

    st.session_state.pop(prefix + "advice", None)
    try:
        _active_project(registry, project_id)
        if provider != "local" and not secret:
            raise model_connect.ConnectFailure("empty", "No key was entered.")
        found = model_connect.find_endpoint(provider, secret, base_url=base_url, model=model, language=current_language())
    except model_connect.ConnectFailure as failure:
        st.session_state[prefix + "advice"] = {"code": failure.code, "provider": provider, **failure.details}
        return False
    settings = {"provider": provider, "model": found["model"]}
    if found["base_url"]:
        settings["base_url"] = found["base_url"]
    try:
        record = registry.create("model", _option_label("provider", provider), settings)
    except ValueError as error:
        st.error(redact(str(error)))
        return False
    state["record_id"] = record["id"]
    st.session_state[prefix + "models_" + record["id"]] = found["models"]
    if provider != "local":
        try:
            registry.set_secret(record["id"], secret)
        except Exception:
            st.session_state[prefix + "notice"] = (False, t("Could not save the credential. Check the system keychain."))
            return True
    if project_id:
        registry.bind(project_id, record["id"], role="model")
    record = registry.probe(record["id"])
    # A listed model that cannot answer in the required format gives way to the next best, twice at most.
    tried = {found["model"]}
    while (record["connection_status"] != "connected" and (record.get("probe") or {}).get("error_code") in {"model", "format"}
           and len(tried) < 3):
        alternative = model_connect.choose_model(provider, [name for name in found["models"] if name not in tried])
        if not alternative or alternative in tried:
            break
        tried.add(alternative)
        registry.update(record["id"], settings={"model": alternative})
        record = registry.probe(record["id"])
    return True


def _connect_button(prefix, key):
    """Connect inside the key's form, so a key typed but not yet confirmed (Tab moves focus to the
    password field's eye button without confirming it) is still sent. The key leaves the widget at
    once and is used in the next run only."""
    def submit():
        from .model_connect import clean_key

        st.session_state[prefix + "pending"] = clean_key(st.session_state.pop(key, ""))
    st.form_submit_button(t("Connect"), key=prefix + "api_connect", type="primary", on_click=submit)


def _api_setup(registry, project_id, prefix, state, provider):
    from .model_connect import GUIDES, REGIONS
    from .providers import PROVIDERS

    info, guide, label = PROVIDERS[provider], GUIDES.get(provider), _option_label("provider", provider)
    # One field per provider, so a key pasted for one is never sent to another.
    key = prefix + "api_key_" + provider
    with st.container(border=True):
        st.markdown("**" + t("1. Get an API key") + "**")
        if guide:
            steps = [t("Open {site} and sign in; create an account if you have none.").format(site=guide["site"]),
                     t("Add API credit there. API use is paid separately from chat memberships."),
                     t("Create an API key and copy it.")]
            st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1)))
            st.link_button(t("Open the {provider} key page").format(provider=label), guide["keys"])
            if "abroad" in guide:
                site, url = guide["abroad"]
                st.caption(t("Account outside mainland China? Create the key on {site} instead; DFT Agent tries known regions.").format(
                    site=f"[{site}]({url})"))
        else:
            st.markdown(t("Your provider gives you three things: the API address, which usually ends in /v1, a key and a model name."))
        with st.form(prefix + "api_form_" + provider, border=False):
            model, base_url = info["model_example"], ""
            if not guide:
                base_url = st.text_input(t("API address"), key=prefix + "base_url_" + provider, placeholder="https://…/v1")
                model = st.text_input(t("Model name"), key=prefix + "model_" + provider,
                                      help=t("The exact model name offered by this provider."))
            st.markdown("**" + t("2. Paste the key") + "**")
            st.text_input(t("API key"), type="password", key=key, placeholder=t("Paste the key here"),
                          help=t("Stored in the system keychain, never in project files."))
            st.markdown("**" + t("3. Connect") + "**")
            st.caption(t("DFT Agent checks the key and region, then tests a model. Test requests use your API tokens."))
            if guide:
                with st.expander(t("Advanced settings (usually not needed)")):
                    model = st.text_input(t("Model name"), value=info["model_example"], key=prefix + "model_" + provider,
                                          help=t("The exact model name offered by this provider."))
                    if provider in REGIONS:
                        base_url = st.text_input(t("API address"), key=prefix + "base_url_" + provider, placeholder="https://…/v1",
                                                 help=t("Leave empty to try known regions."))
            _connect_button(prefix, key)
        if prefix + "pending" in st.session_state:
            secret = st.session_state.pop(prefix + "pending")
            with st.spinner(t("Checking the key and choosing a model…")):
                connected = _api_connect(registry, project_id, prefix, state, provider, secret, model.strip(), base_url.strip())
            del secret
            if connected:
                st.rerun()
        advice = st.session_state.get(prefix + "advice")
        if advice and advice.get("provider") == provider:
            _show_advice(provider, advice)


def _local_setup(registry, project_id, prefix, state):
    from . import model_connect

    found_key = prefix + "local_servers"
    if found_key not in st.session_state:
        st.session_state[found_key] = model_connect.local_servers()
    servers = st.session_state[found_key]
    with st.container(border=True):
        st.markdown("**" + t("1. Start a model server on this computer") + "**")
        st.markdown(t("For example, install Ollama from ollama.com, then download a model in a terminal:"))
        st.code("ollama pull qwen3", language="bash")
        st.caption(t("LM Studio, llama.cpp and vLLM servers are found as well. Small models may plan poorly."))
        st.markdown("**" + t("2. Choose a model") + "**")
        options = [(server["name"], server["base_url"], name) for server in servers for name in server["models"]]
        choice = None
        if options:
            choice = st.selectbox(t("Model"), options, format_func=lambda item: f"{item[2]} · {item[0]}", key=prefix + "local_model",
                                  label_visibility="collapsed")
        else:
            st.info(t("No model server is running on this computer yet. Start one, then look again."))
        if st.button(t("Look for local servers again"), key=prefix + "local_refresh"):
            st.session_state.pop(found_key, None)
            st.rerun()
        st.markdown("**" + t("3. Connect") + "**")
        if st.button(t("Connect"), key=prefix + "local_connect", type="primary", disabled=choice is None):
            with st.spinner(t("Testing the connection…")):
                if _api_connect(registry, project_id, prefix, state, "local", "", choice[2], choice[1]):
                    st.rerun()
        with st.expander(t("Other local server")):
            with st.form(prefix + "local_manual_form", border=False):
                address = st.text_input(t("API address"), key=prefix + "local_address", placeholder="http://127.0.0.1:9000/v1")
                model = st.text_input(t("Model name"), key=prefix + "local_name")
                submitted = st.form_submit_button(t("Connect local server"))
            if submitted:
                if not address.strip():
                    st.error(t("Enter the local server's API address."))
                else:
                    with st.spinner(t("Testing the connection…")):
                        if _api_connect(registry, project_id, prefix, state, "local", "", model.strip(), address.strip()):
                            st.rerun()
        advice = st.session_state.get(prefix + "advice")
        if advice and advice.get("provider") == "local":
            _show_advice("local", advice)


def _model_summary(record):
    from .model_connect import REGIONS

    settings = record["settings"]
    region = next((name for name, address in REGIONS.get(settings["provider"], ()) if address == settings.get("base_url")), None)
    return t("Model: {model}").format(model=settings.get("model", "")) + (" · " + t("Region: {region}").format(region=t(region)) if region else "")


def _change_model(registry, record, prefix):
    """Another model for a saved API connection, from the account's list when it was read."""
    models = st.session_state.get(prefix[:-len(record["id"]) - 1] + "models_" + record["id"]) or []
    with st.expander(t("Change the model")):
        current = record["settings"].get("model", "")
        if models:
            from .model_connect import _NOT_CHAT

            listed = sorted(name for name in models if not _NOT_CHAT.search(name)) or sorted(models)
            choice = st.selectbox(t("Model name"), listed, index=listed.index(current) if current in listed else 0, key=prefix + "model_choice")
        else:
            choice = st.text_input(t("Model name"), value=current, key=prefix + "model_choice")
        if st.button(t("Save and test"), key=prefix + "model_save", disabled=not str(choice).strip() or choice == current):
            try:
                registry.update(record["id"], settings={"model": str(choice).strip()})
                with st.spinner(t("Testing the connection…")):
                    registry.probe(record["id"])
            except ValueError as error:
                st.session_state[prefix + "model_notice"] = (False, str(error))
            st.rerun()


def _model_record(registry, record, prefix, project_id, state):
    provider = record["settings"]["provider"]
    status = record["connection_status"]
    st.caption(f"{record['name']} · {_status_label(record)}")
    _notice(prefix[:-len(record["id"]) - 1] + "notice")
    _notice(prefix + "model_notice")
    if record.get("revoked"):
        st.info(t("This connection has been revoked."))
        return
    detail = _probe_detail(record)
    code = (record.get("probe") or {}).get("error_code")
    if status == "connected":
        st.success(t("Model connected."))
        if provider not in _CLI_PROVIDERS:
            st.caption(_model_summary(record))
    elif code and provider not in _CLI_PROVIDERS:
        from urllib.parse import urlsplit

        _show_advice(provider, {"code": code, "model": record["settings"].get("model", ""),
                                "host": urlsplit(record["settings"].get("base_url") or "").hostname or ""})
        with st.expander(t("Technical details")):
            st.caption(detail)
    elif detail:
        (st.warning if record["auth_status"] == "required" else st.error)(detail)
    if provider not in _CLI_PROVIDERS:
        _change_model(registry, record, prefix)
    if provider in _CLI_PROVIDERS and status != "connected":
        from .cli_setup import next_step

        item = next(item for item in _cli_detection(prefix) if item["provider"] == provider)
        outdated = status == "error" and "too old" in detail
        if outdated or next_step(item) != "connect":
            _notice(prefix + "cli_notice")
            _cli_actions(prefix, item, update=outdated, watched=False)
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
    if not state["pending_host"]:
        st.caption(t("The app first tries your SSH keys, then asks for a password or login code only if the cluster requires one."))
    with st.form(prefix + "address_form", border=False):
        answer = st.text_input(t("Username on this cluster") if state["pending_host"] else t("SSH address"),
                               key=prefix + ("username" if state["pending_host"] else "answer"),
                               placeholder="" if state["pending_host"] else "user@cluster.example.org")
        submitted = st.form_submit_button(t("Next"), type="primary")
    if not submitted or not answer.strip():
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
    st.text_input(t("SSH password"), type="password", key=key,
                  help=t("It is stored in the system keychain and used only for this connection."))

    def submit():
        _sign_in(registry, record["id"], prefix, state, password=st.session_state.pop(key, ""))

    st.button(t("Sign in"), key=prefix + "sign_in", type="primary", on_click=submit)


def _code_form(registry, record, prefix, state):
    key = prefix + "code"
    from .transport import LOGIN_PERSIST_SECONDS
    if state["needs_code"]:
        st.code(state["needs_code"], language="text")
    st.text_input(t("Login code"), type="password", key=key, placeholder="123456",
                  help=t("Enter the code from your authenticator or message. The signed-in session is reused for about {hours} hours of activity, so the code is not needed for every job.").format(hours=LOGIN_PERSIST_SECONDS // 3600))

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


def _reference_panel(registry, environments, prefix):
    """Official manuals and examples the planner reads; CP2K's keyword list comes from the program itself."""
    from . import references

    missing, ready = [], []
    for environment in environments:
        if environment["engine"] not in references.ENGINES:
            continue
        parts = references.available(environment["engine"], registry._environment_config(environment))
        label = references.LABELS[environment["engine"]] + (f" {parts['version']}" if parts.get("version") else "")
        (ready if all(value for key, value in parts.items() if key != "version") else missing).append((environment, label))
    if ready:
        st.caption(t("Manual and examples ready: ") + ", ".join(label for _, label in ready))
    if not missing:
        return
    if st.button(t("Get manuals and examples"), key=prefix + "references",
                 help=t("Downloads the pinned official manual and examples for the planner. For CP2K it also runs cp2k --xml on the login node to read that build's keywords.")):
        errors = []
        with st.spinner(t("Reading the official manuals and examples…")):
            for environment, label in missing:
                try:
                    registry.build_reference(environment["id"])
                except (ValueError, OSError) as error:
                    errors.append(f"{label}: {redact(str(error))}")
        for error in errors:
            st.error(error)
        if not errors:
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
        _reference_panel(registry, environments, prefix)
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
        _engine_install_panel(registry, record, prefix, environments)
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


_CODES = {"abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk"}
_CODE_NOTES = {"abinit": "Plane waves with PseudoDojo norm-conserving pseudopotentials. Spin-orbit coupling supported.",
               "cp2k": "Gaussian basis sets (GPW). Efficient for large cells, molecules and surfaces.",
               "elk": "All-electron full-potential LAPW. Reference-quality bands, magnetism and spin-orbit coupling."}


def _engine_install_panel(registry, record, prefix, environments):
    """Pinned conda-forge installs of ABINIT, CP2K and Elk: review the plan, start one job, follow it."""
    probe = record.get("probe") or {}
    with st.container(border=True):
        st.markdown("**" + t("Install open-source codes") + "**")
        st.caption(t("ABINIT, CP2K and Elk are open source. The app installs pinned, checksum-verified builds in your cluster storage and checks each with a small silicon calculation. No compiler or administrator access is needed."))
        for engine, label in _CODES.items():
            install = probe.get(f"{engine}_install") or {}
            present = [item for item in environments if item["engine"] == engine]
            status = install.get("status")
            line = f"{label} · " + (t("ready") if present else t(str(status)) if status else t("not installed"))
            if install.get("job_id"):
                line += f" · job {install['job_id']}"
            if install.get("scheduler_state"):
                line += f" · {install['scheduler_state']}"
            st.caption(line + " — " + t(_CODE_NOTES[engine]))
            if install.get("error"):
                st.error(redact(str(install["error"])))
            if status == "installed" and (install.get("smoke") or {}).get("success"):
                smoke = install["smoke"]
                if smoke.get("final_energy_ev") is not None:
                    st.caption(t("Silicon test passed: total energy {energy:.4f} eV.").format(energy=smoke["final_energy_ev"]))
        choices = [engine for engine in _CODES if not any(item["engine"] == engine for item in environments)
                   and (probe.get(f"{engine}_install") or {}).get("status") != "installed"]
        if not choices:
            return
        engine = st.radio(t("Code"), choices, horizontal=True, key=prefix + "code_choice", format_func=lambda value: _CODES[value])
        install = probe.get(f"{engine}_install") or {}
        status = install.get("status")
        suggested = (probe.get("discovery") or {}).get("suggested", {})
        if status in {None, "planned"} or (status == "failed" and st.session_state.get(prefix + engine + "_replan")):
            partition = st.text_input(t("Partition for the installation job"), install.get("partition") or suggested.get("partition", ""),
                                      key=prefix + engine + "_partition")
            account = st.text_input(t("Account (optional)"), install.get("account") or suggested.get("account", ""),
                                    key=prefix + engine + "_account")
            if st.button(t("Plan the installation"), key=prefix + engine + "_plan"):
                try:
                    with st.spinner(t("Reading the cluster's architecture…")):
                        registry.engine_install_plan(record["id"], engine, partition=partition.strip() or None, account=account.strip())
                    st.session_state.pop(prefix + engine + "_replan", None)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(redact(str(error)))
            if status == "planned":
                plan = install["plan"]
                st.write(f"{_CODES[engine]} {plan['version']} · {plan['platform']} · glibc {plan['glibc']} · "
                         f"{round(plan['download_size'] / 1e6)} MB · {install['partition']}" + (f" · {install['account']}" if install.get("account") else ""))
                st.code(plan["prefix"] + "\n" + plan["command"], language="text")
                agreed = st.checkbox(t("Use this account and allocation for a short installation job."), key=prefix + engine + "_agree")
                if st.button(t("Start the installation"), key=prefix + engine + "_start", type="primary", disabled=not agreed):
                    try:
                        with st.spinner(t("Downloading pinned packages on the cluster and submitting the job…")):
                            registry.engine_install_start(record["id"], engine, install["plan_id"])
                        st.rerun()
                    except (ValueError, OSError) as error:
                        st.error(redact(str(error)))
            return
        left, right = st.columns(2)
        if left.button(t("Check the installation"), key=prefix + engine + "_check"):
            try:
                with st.spinner(t("Reading the installation status…")):
                    registry.engine_install_status(record["id"], engine)
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(redact(str(error)))
        if status == "failed" and right.button(t("Plan again"), key=prefix + engine + "_replan_button"):
            st.session_state[prefix + engine + "_replan"] = True
            st.rerun()
        if install.get("logs"):
            st.caption(t("Installation logs on the cluster: ") + ", ".join(install["logs"].values()))


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
        st.write(t("Review the environment, then save it."))
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
    answered = [message["content"] for message in state["history"] if message["role"] == "user"]
    if answered:
        st.caption(" · ".join(str(value) for value in answered[-6:]))
    field = question["field"]
    options = question["options"]
    available = {}
    if field == "connection_id":
        available = {item["id"]: item for item in registry.list(kind="hpc") if not item["revoked"] and not item["deleted"]}
        options = list(available)
        if not options:
            st.caption(t("Add a cluster in the Cluster tab first."))
            return
    st.markdown("**" + t(_PROMPTS.get(field, question["label"])) + "**")
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
    with st.form(prefix + "answer_form_" + field, border=False):
        response = st.text_input(t(_PROMPTS.get(field, question["label"])), key=prefix + "answer_" + field,
                                 placeholder=str(default) if default else "", label_visibility="collapsed")
        submitted = st.form_submit_button(t("Next"), type="primary")
    if submitted and response.strip():
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
