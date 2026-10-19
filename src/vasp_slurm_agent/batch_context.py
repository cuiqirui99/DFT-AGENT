"""Read verified variant results without refreshing or writing batch records."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def load_batch_context(batch_dir):
    from . import batch, explanation, workflow

    root = Path(batch_dir).expanduser().resolve()

    def path(relative):
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Invalid batch artifact path.")
        candidate = root / relative
        if not candidate.resolve().is_relative_to(root) or candidate.resolve() == root:
            raise ValueError("Invalid batch artifact path.")
        if any(item.is_symlink() for item in (candidate, *candidate.parents) if item.is_relative_to(root)):
            raise ValueError("Batch artifacts cannot be symbolic links.")
        return candidate

    fingerprints = {}

    def read(name, *, optional=False):
        item = path(name)
        if optional and not item.exists():
            fingerprints[name] = None
            return {}
        content = item.read_bytes()
        fingerprints[name] = hashlib.sha256(content).hexdigest()
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Invalid batch record.")
        return value

    state = read("batch.json")
    plan = read("batch-plan.json")
    proposal = read("proposal.json", optional=True)
    config_file = path("config.json")
    fingerprints["config.json"] = workflow._digest(config_file) if config_file.is_file() else None
    if (fingerprints["batch-plan.json"] != state.get("plan_sha256")
            or not isinstance(plan.get("runs"), list) or not 1 <= len(plan["runs"]) <= 256):
        raise ValueError("The batch plan changed. Prepare a new batch.")
    source = path(plan.get("source"))
    fingerprints[plan["source"]] = workflow._digest(source)
    if fingerprints[plan["source"]] != plan.get("source_sha256"):
        raise ValueError("The frozen batch source changed.")
    for member in plan["runs"]:
        path(member.get("folder"))
    children = batch._children(root, state)
    facts, artifacts, limits, members, member_hashes = {}, [], [], [], {}

    def fact(identifier, label, value, source="batch-plan.json", unit=""):
        facts[identifier] = {"label": label, "value": value, "source": source, "unit": unit}

    for index, (member, folder, _) in enumerate(children, 1):
        prefix = f"member_{index}"
        context = explanation.load_run_context(folder)
        label = explanation._safe_text(member["label"])
        member_hashes[member["folder"]] = context["context_sha256"]
        members.append({"id": prefix, "label": label, "status": context["status"],
                        "final_plan": deepcopy(context.get("final_plan", {}))})
        fact(prefix + ".label", "Variant", label)
        fact(prefix + ".status", f"{label}: result status", context["status"], member["folder"] + "/run.json")
        for identifier, item in context["facts"].items():
            facts[prefix + "." + identifier] = {
                **deepcopy(item), "label": f"{label}: {item['label']}",
                "source": member["folder"] + "/" + item.get("source", "run.json"),
            }
        for item in context.get("artifacts", []):
            artifacts.append({**deepcopy(item), "id": prefix + "." + item["id"],
                              "label": f"{label}: {item['label']}",
                              "path": member["folder"] + "/" + item["path"]})
        limits.extend(f"{label}: {limit}" for limit in context.get("limits", []))
    statuses = [member["status"] for member in members]
    if all(status == "succeeded" for status in statuses):
        status = "succeeded"
    elif all(status == "planned" for status in statuses):
        status = "planned"
    elif any(status not in workflow.TERMINAL | {"planned"} for status in statuses):
        status = "running"
    elif any(status in {"failed", "needs_attention"} for status in statuses):
        status = "needs_attention"
    elif any(status == "cancelled" for status in statuses):
        status = "cancelled"
    else:
        status = "running"
    succeeded = statuses.count("succeeded")
    comparison_scope = ("Each result belongs to its variant's method and settings. No cross-variant energy ranking "
                        "is supplied by this context; do not infer a common energy ordering or a magnetic ground state.")
    fact("batch.status", "Batch result status", status, "batch.json")
    fact("batch.members", "Variants", len(members))
    fact("batch.accepted_members", "Variants with accepted complete results", succeeded, "verified member results")
    fact("batch.comparison_scope", "Comparison scope", comparison_scope)
    limits.append(comparison_scope)
    linked = (proposal.get("execution") or {}).get("plan_sha256") == fingerprints["batch-plan.json"]
    goal = explanation._safe_text(proposal.get("goal")) if linked else ""
    dialogue = [{"role": item["role"], "content": explanation._safe_text(item.get("content"))}
                for item in proposal.get("dialogue", []) if linked and isinstance(item, dict)
                and item.get("role") in {"user", "assistant"}]
    context = {"engine": "vasp", "workflow": "batch", "goal": goal, "dialogue": dialogue,
               "status": status, "outcome": f"{succeeded} of {len(members)} variants have accepted complete results.",
               "final_plan": {"tasks": list(plan.get("tasks", [])), "variants": len(members)},
               "members": members, "facts": facts, "artifacts": artifacts,
               "limits": list(dict.fromkeys(limits))}
    for name, expected in fingerprints.items():
        item = path(name)
        if (workflow._digest(item) if item.is_file() else None) != expected:
            raise ValueError("The batch changed while reading its results. Try again.")
    context["context_sha256"] = hashlib.sha256(explanation._canonical({
        "context": context, "files": fingerprints, "members": member_hashes,
    }).encode()).hexdigest()
    return context
