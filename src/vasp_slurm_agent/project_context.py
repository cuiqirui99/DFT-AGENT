"""Build model context from one project and freshly checked run evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .explanation import load_run_context
from .project_files import list_files
from .prompt_context import compact_history
from .secret_guard import redact


_LIMITS = {"messages": None, "workflows": 12, "files": 40, "artifacts": 40, "memory": 24,
           "runs": 3, "characters": 80000}
_LOCAL_PATH = re.compile(r"(?<![\w:/])(?:~?/[^\s,;\"']+|[A-Za-z]:\\[^\s,;\"']+)")


def _text(value):
    return _LOCAL_PATH.sub("[path omitted]", redact(value))


def _value(value):
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, list):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {_text(key): _value(item) for key, item in value.items()}
    return value


def check_context_budget(context, characters=None):
    serialized = json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True)
    if len(serialized) > (characters if characters is not None else _LIMITS["characters"]):
        raise ValueError("Project context is too large. Save the requirements in a workflow or project note, "
                         "then start a shorter session or select fewer records.")
    return serialized


def _owned(record, project_id):
    if record.get("project_id") != project_id:
        raise ValueError("The selected record belongs to another project.")
    return record


def _select(records, identifiers, project_id, limit, kind):
    records = [_owned(item, project_id) for item in records]
    if identifiers is not None:
        identifiers = list(dict.fromkeys(identifiers))
        by_id = {item["id"]: item for item in records}
        if any(identifier not in by_id for identifier in identifiers):
            raise ValueError(f"A selected {kind} does not belong to this project.")
        records = [by_id[identifier] for identifier in identifiers]
        if len(records) > limit:
            raise ValueError(f"Select at most {limit} {kind} records.")
    return records[:limit], max(0, len(records) - limit)


def _private_file(relative):
    path = Path(relative)
    parts = {part.lower() for part in path.parts}
    return bool(parts & {"config", ".ssh", "secrets", "credentials", "pseudos", "pseudopotentials"}
                or path.name.lower() in {"vasp_license", "auth.json", "credentials.json"}
                or path.name.lower().startswith(("potcar", ".env", "id_rsa", "id_ed25519"))
                or path.suffix.lower() in {".upf", ".pem", ".key"})


def _run_path(workspace, project_id, record, runs_root):
    candidate = workspace.resolve_path(project_id, record["relative_path"])
    if not candidate.is_dir() or not candidate.resolve().is_relative_to(runs_root.resolve()):
        raise ValueError("The run must be inside this project's runs folder.")
    # Solver parsers inspect several linked files. Reject redirects before any
    # parser can read data from outside the selected project.
    for index, path in enumerate(candidate.rglob("*")):
        if path.is_symlink():
            raise ValueError("Run evidence cannot contain symbolic links.")
        if index >= 20000:
            raise ValueError("This run has too many files for model context.")
    return candidate


def build_project_context(workspace, project_id, *, session_id=None,
                          workflow_ids=None, artifact_ids=None, run_ids=(),
                          question="", limits=None):
    """Return bounded, path-free context; no model call or cluster access occurs.

    Full conversation history is included by default within the character
    budget. Explicit record limits report omissions; run facts stay complete.
    """
    bounds = dict(_LIMITS)
    if limits:
        if set(limits) - set(bounds) or any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Invalid project context limits.")
        bounds.update(limits)
    project = workspace.get_project(project_id)
    if project["id"] != project_id or project.get("deleted") or project.get("archived"):
        raise ValueError("Open an active project before building model context.")
    paths = workspace.project_paths(project_id)
    context = {"project": {"id": project_id, "name": _text(project["name"]),
                           "revision": project["revision"]},
               "question": _text(question), "session": None, "workflows": [], "files": [],
               "artifacts": [], "memory": [], "runs": [], "omitted": {}}
    inventory = [item for item in list_files(workspace, project_id, limit=min(bounds["files"] + 1, 5000))
                 if Path(item["path"]).parts[0] == "files" and not _private_file(item["path"])]
    context["files"] = [{"path": _text(item["path"]), "size": item["size"]}
                        for item in inventory[:bounds["files"]]]
    context["omitted"]["more_files"] = len(inventory) > bounds["files"]
    if session_id is not None:
        session = _owned(workspace.get_session(project_id, session_id), project_id)
        messages = session.get("messages", [])
        selected = messages if bounds["messages"] is None else messages[-bounds["messages"]:]
        context["session"] = {"id": session["id"], "title": _text(session["title"]),
                              "revision": session["revision"], "history": compact_history([
                                  {"role": item["role"], "content": _text(item["content"])}
                                  for item in selected if item["role"] in {"user", "assistant"}
                              ])}
        context["omitted"]["messages"] = max(0, len(messages) - len(selected))
    workflows, omitted = _select(workspace.list_workflows(project_id), workflow_ids,
                                 project_id, bounds["workflows"], "workflow")
    context["omitted"]["workflows"] = omitted
    for selected in workflows:
        item = _owned(workspace.get_workflow(project_id, selected["id"]), project_id)
        with workspace._transaction(write=False) as db:
            attempts = {row["step_id"]: dict(row) for row in db.execute(
                "SELECT step_id,attempt,state,run_id FROM step_attempts WHERE project_id=? AND workflow_id=? ORDER BY attempt",
                (project_id, item["id"]))}
        steps = []
        for step in item.get("steps", []):
            entry = {key: _value(step[key]) for key in ("id", "name", "engine", "status", "depends_on")}
            attempt = attempts.get(step["id"])
            if attempt:
                state = (workspace.get_run(project_id, attempt["run_id"])["state"] or {}) if attempt["run_id"] else {}
                tasks = [stage["name"] for stage in state.get("stages", []) if stage.get("name")]
                entry["current_attempt"] = {"number": attempt["attempt"], "tasks": _value(tasks),
                                            "status": attempt["state"], "run_id": attempt["run_id"]}
            steps.append(entry)
        context["workflows"].append({"id": item["id"], "title": _text(item["title"]),
                                     "goal": _text(item["goal"]), "status": item["status"],
                                     "steps": steps})
    artifacts, omitted = _select(workspace.list_artifacts(project_id), artifact_ids,
                                project_id, bounds["artifacts"], "artifact")
    context["omitted"]["artifacts"] = omitted
    for item in artifacts:
        relative = item["relative_path"]
        workspace.resolve_path(project_id, relative)
        if _private_file(relative):
            context["omitted"]["artifacts"] += 1
            continue
        context["artifacts"].append({"id": item["id"], "name": _text(Path(relative).name),
                                     "kind": _text(item["kind"]), "size": item["size"],
                                     "registered_sha256": item["sha256"],
                                     "available": bool(item.get("available", True)),
                                     "run_id": item.get("run_id")})
    notes = [_owned(item, project_id) for item in workspace.list_memory(project_id)]
    context["memory"] = [{"key": _text(item["key"]), "text": _text(item["text"]),
                           "source": "project_note", "verified": False}
                          for item in notes[:bounds["memory"]]]
    context["omitted"]["memory"] = max(0, len(notes) - bounds["memory"])
    runs, _ = _select(workspace.list_runs(project_id), run_ids, project_id, bounds["runs"], "run")
    for item in runs:
        if item.get("state") is None:
            raise ValueError("The selected run is missing or its identity has changed.")
        path = _run_path(workspace, project_id, item, paths["runs"])
        evidence = load_run_context(path)
        facts = {identifier: {key: _value(fact[key]) for key in ("label", "value", "unit") if key in fact}
                 for identifier, fact in evidence.get("facts", {}).items()}
        context["runs"].append({"id": item["id"], "engine": item["engine"],
                                "status": evidence["status"], "outcome": _text(evidence["outcome"]),
                                "context_sha256": evidence["context_sha256"], "facts": facts,
                                "limits": _value(evidence.get("limits", []))})
    serialized = check_context_budget(context, bounds["characters"])
    context["context_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    return context
