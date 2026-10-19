"""Guided connection setup. Credentials never enter the conversation state."""

from copy import deepcopy
from pathlib import PurePosixPath
import re

from .secret_guard import reject_ordinary_text


KINDS = ("model", "hpc", "software")


def _question(field, label, *, options=None, default=None, optional=False, integer=False):
    return {"field": field, "label": label, "options": options, "default": default,
            "optional": optional, "integer": integer}


def questions(state):
    from .providers import PROVIDERS

    settings = state["settings"]
    fields = [_question("name", "Name")]
    if state["kind"] == "model":
        providers = list(dict.fromkeys([*PROVIDERS, "claude_cli", "local"]))
        fields.append(_question("provider", "Model provider", options=providers))
        if settings.get("provider") not in {"codex", "claude_cli"}:
            fields.append(_question("model", "Model name"))
        if settings.get("provider") in {"qwen", "chat_completions", "local"}:
            fields.append(_question("base_url", "API address"))
    elif state["kind"] == "hpc":
        fields.extend([
            _question("host", "SSH host"), _question("user", "SSH user"),
            _question("auth_mode", "Sign in with", options=["agent", "key", "password", "interactive"]),
            _question("remote_base", "Remote work folder"),
            _question("scheduler", "Scheduler", options=["auto", "slurm", "pbs"], default="auto"),
        ])
        if settings.get("auth_mode") == "key":
            fields.append(_question("identity_file", "Private key file", optional=True, default=""))
        if settings.get("auth_mode") == "interactive":
            fields.append(_question("ssh_control_path", "SSH control socket", optional=True, default=""))
    elif state["kind"] == "software":
        fields.extend([
            _question("connection_id", "HPC connection"),
            _question("engine", "Software", options=["vasp", "qe", "wannier90", "python"]),
            _question("config.partition", "Partition or queue"),
        ])
        if settings.get("engine") == "vasp":
            fields.extend([_question("config.vasp_command", "VASP command"),
                           _question("config.potcar_root", "POTCAR folder")])
        elif settings.get("engine") == "qe":
            fields.extend([_question("config.qe_command", "QE command"),
                           _question("config.qe_pseudo_root", "UPF folder")])
        elif settings.get("engine") in {"wannier90", "python"}:
            fields.append(_question("config.command", "Program command"))
        fields.extend([_question("config.tasks", "Tasks", integer=True),
                       _question("config.nodes", "Nodes", integer=True),
                       _question("config.walltime", "Time limit")])
    else:
        raise ValueError("Choose model, hpc or software setup.")
    return fields


def begin(kind, record=None):
    if kind not in KINDS:
        raise ValueError("Choose model, hpc or software setup.")
    state = {"schema_version": 1, "kind": kind, "name": "", "settings": {},
             "record_id": None, "history": []}
    if kind == "hpc":
        state["settings"]["port"] = 22
    elif kind == "software":
        state["settings"]["config"] = {"tasks": 8, "nodes": 1, "walltime": "00:30:00"}
    if record is not None:
        if record.get("kind", kind) != kind or record.get("deleted") or record.get("revoked"):
            raise ValueError("Choose an active connection of the same kind.")
        state.update(name=record["name"], record_id=record["id"])
        state["settings"] = deepcopy(record["settings"] if kind != "software" else {
            "connection_id": record["connection_id"], "engine": record["engine"], "config": record["config"]})
        _check_text(state)
    return state


def value(state, field):
    if field == "name":
        return state["name"]
    if field.startswith("config."):
        return state["settings"].get("config", {}).get(field.removeprefix("config."))
    return state["settings"].get(field)


def next_question(state):
    for question in questions(state):
        response = value(state, question["field"])
        if response is None or (not question["optional"] and response == ""):
            return question
    return None


def _check_text(item):
    if isinstance(item, str):
        reject_ordinary_text(item)
        if "\x00" in item or len(item) > 100000:
            raise ValueError("The setup value is too long or contains invalid characters.")
    elif isinstance(item, dict):
        for key, child in item.items():
            normalized = re.sub(r"[^a-z]", "", str(key).lower())
            if normalized in {"secret", "credential", "credentials", "otp", "totp", "devicecode", "passphrase", "authorization"} or any(
                    normalized.endswith(suffix) for suffix in ("password", "apikey", "privatekey", "accesstoken", "refreshtoken")):
                raise ValueError("Use the secure credential field, not setup answers.")
            reject_ordinary_text(str(key))
            _check_text(child)
    elif isinstance(item, list):
        for child in item:
            _check_text(child)


def answer(state, field, response):
    """Validate an ordinary answer before adding it to setup history."""
    allowed = {item["field"]: item for item in questions(state)}
    if field not in allowed:
        raise ValueError("Choose a field from this setup step.")
    question = allowed[field]
    _check_text(response)
    if not isinstance(response, str) or (not response.strip() and not question["optional"]):
        raise ValueError("Enter a value.")
    response = response.strip()
    if any(ord(character) < 32 for character in response):
        raise ValueError("Use a single line.")
    if question["options"] and response not in question["options"]:
        raise ValueError("Choose one of the listed options.")
    if field in {"remote_base", "config.potcar_root", "config.qe_pseudo_root"} and not PurePosixPath(response).is_absolute():
        raise ValueError("Use an absolute folder path on the cluster.")
    stored_response = response
    if question["integer"]:
        if not response.isdecimal() or not 1 <= int(response) <= 100000:
            raise ValueError("Enter a positive integer up to 100000.")
        stored_response = int(response)
    updated = deepcopy(state)
    settings = updated["settings"]
    if field == "name":
        updated["name"] = response
    elif field.startswith("config."):
        settings.setdefault("config", {})[field.removeprefix("config.")] = stored_response
    else:
        if field == "provider" and settings.get(field) != response:
            settings.pop("model", None)
            settings.pop("base_url", None)
        if field == "engine" and settings.get(field) != response:
            for key in ("vasp_command", "potcar_root", "vasp_ncl_command", "qe_command", "qe_bin_dir", "qe_pseudo_root", "qe_pseudopotentials", "command"):
                settings.setdefault("config", {}).pop(key, None)
        settings[field] = response
    updated["history"].extend([{"role": "assistant", "content": question["label"]},
                               {"role": "user", "content": response}])
    return updated


def advanced(state, settings):
    """Merge reviewed non-secret settings; the registry validates field names."""
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object.")
    _check_text(settings)
    updated = deepcopy(state)
    target = updated["settings"].setdefault("config", {}) if state["kind"] == "software" else updated["settings"]
    target.update(deepcopy(settings))
    return updated


def save(state, registry, project_id=None):
    """Save settings, without probing, logging in or launching a calculation.

    The saved ID is retained before binding, so a failed bind can be retried
    without creating a duplicate connection.
    """
    if next_question(state) is not None:
        raise ValueError("Complete the setup questions first.")
    _check_text(state)
    if state["kind"] == "software":
        if state["record_id"]:
            raise ValueError("This environment is already saved. Start another setup to create a new one.")
        result = registry.create_environment(state["settings"]["connection_id"], state["name"],
                                             state["settings"]["engine"], state["settings"]["config"])
    elif state["record_id"]:
        result = registry.update(state["record_id"], settings=state["settings"], name=state["name"])
    else:
        result = registry.create(state["kind"], state["name"], state["settings"])
    state["record_id"] = result["id"]
    if project_id is not None and state["kind"] != "software":
        registry.bind(project_id, result["id"], role=state["kind"])
    return result
