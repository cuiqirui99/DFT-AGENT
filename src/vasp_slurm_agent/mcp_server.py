"""A Model Context Protocol server over one project's reviewed tools.

Codex CLI and Claude Code launch this server over stdio during an agent turn
and call its tools: reading and writing project files, describing structures,
creating workflows, running the engine planners, preparing and reviewing
inputs, checking status, explaining verified results, and setting up Quantum
ESPRESSO. The tools reuse the same validation as the guided interface. Nothing
here submits a calculation; submission stays an explicit action in the app.

Environment: DFT_AGENT_HOME (workspace), DFT_AGENT_PROJECT, DFT_AGENT_SESSION
and DFT_AGENT_MODEL_CONNECTION (for nested planner and explanation calls).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import __version__
from .secret_guard import redact, reject_ordinary_text

PROTOCOL_VERSION = "2025-06-18"
INSTRUCTIONS = (
    "Tools act on the user's current DFT Agent project. Read files and describe structures before planning. "
    "create_workflow records steps; plan_step runs the engine planner, which proposes and validates settings and may ask "
    "questions; continue_step starts a separate calculation from an accepted stage of a finished step; repair_step "
    "drafts a bounded retry of a failed step. Old runs are preserved. prepare_inputs writes reviewable input files; "
    "review_inputs shows them. Submission is not a tool: after "
    "preparing and reviewing, ask the user to click Submit in the step card. Use explain_results for any number taken "
    "from a calculation. search_reference and read_reference return the official QE, Elk, ABINIT and CP2K manual entries and "
    "example inputs for an environment's version; check them before saying what a program can do or what a setting "
    "means. Examples demonstrate features and are not converged settings. "
    "Open-source codes (QE, ABINIT, CP2K, Elk) can be installed on the cluster: plan with "
    "qe_install_plan or engine_install_plan, then the user starts the installation in Connections. Never ask for "
    "passwords or keys."
)


def _text(value):
    return {"type": "string"} if value is None else {"type": "string", "description": value}


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


class ProjectTools:
    """Tools bound to one project. Every call re-reads storage; nothing is cached."""

    def __init__(self, workspace, project_id, session_id=None, model_connection_id=None):
        from .connections import ConnectionRegistry
        from .project_execution import ProjectExecution

        self.workspace = workspace
        self.registry = ConnectionRegistry(workspace)
        self.execution = ProjectExecution(workspace, self.registry)
        self.project_id = project_id
        self.session_id = session_id
        self.model_connection_id = model_connection_id
        workspace.get_project(project_id)

    @classmethod
    def from_environment(cls):
        from .workspace import Workspace

        project = os.environ.get("DFT_AGENT_PROJECT")
        if not project:
            raise ValueError("DFT_AGENT_PROJECT names the project this server works on.")
        return cls(Workspace(), project, os.environ.get("DFT_AGENT_SESSION") or None,
                   os.environ.get("DFT_AGENT_MODEL_CONNECTION") or None)

    # --- definitions -------------------------------------------------------

    def definitions(self):
        from .project_agent import _schema as project_schema

        draft_fields = project_schema()["properties"]["workflow"]["anyOf"][0]["properties"]["steps"]["items"]["properties"]
        step = {"type": "object", "additionalProperties": False, "required": ["name", "engine", "goal"], "properties": {
            "id": _text("Short unique step id, letters, digits, - and _"), "name": _text("Step name"),
            "engine": {"type": "string", "enum": ["vasp", "qe", "abinit", "cp2k", "elk", "wannier90", "python"]},
            "goal": _text("Scientific goal of this step in one or two sentences"),
            "source_path": _text("Project file used as the input structure, for example files/Si.cif"),
            "source_step": _text("Id of an earlier step whose accepted structure this step continues from"),
            "depends_on": {"type": "array", "items": {"type": "string"}},
            **{name: draft_fields[name] for name in ("environment_id", "inputs", "input_artifacts", "arguments", "outputs")}}}
        return [
            {"name": "project_overview", "description": "Project files, connections, software environments, workflows with step states, and runs.", "inputSchema": _schema()},
            {"name": "read_file", "description": "Read a project text file (structures, scripts, notes). Private files are excluded.", "inputSchema": _schema({"path": _text("Relative path such as files/Si.cif")}, ["path"])},
            {"name": "write_file", "description": "Create or replace a draft file under files/ (structures, scripts, notes). Never credentials.", "inputSchema": _schema({"path": _text("Relative path under files/"), "content": _text("Full file content")}, ["path", "content"])},
            {"name": "describe_structure", "description": "Formula, lattice, sites, elements and symmetry of a structure file in the project.", "inputSchema": _schema({"path": _text("Relative path of a CIF or POSCAR")}, ["path"])},
            {"name": "list_examples", "description": "Bundled example structures that can be imported into the project.", "inputSchema": _schema()},
            {"name": "import_example", "description": "Copy a bundled example structure into files/.", "inputSchema": _schema({"name": _text("Example file name from list_examples")}, ["name"])},
            {"name": "create_workflow", "description": "Record an ordered set of calculation steps for this project. Does not run anything.", "inputSchema": _schema({
                "title": _text("Short workflow title"), "goal": _text("Overall goal"),
                "steps": {"type": "array", "minItems": 1, "maxItems": 32, "items": step}}, ["title", "goal", "steps"])},
            {"name": "plan_step", "description": "Run the engine planner for a step. It proposes validated settings from the goal and may return questions; answer them by calling plan_step again with instructions.", "inputSchema": _schema({
                "workflow_id": _text(None), "step_id": _text(None), "instructions": _text("The user's requirements or answers to the planner's questions")}, ["workflow_id", "step_id"])},
            {"name": "continue_step", "description": "Draft a separate calculation from an accepted stage of a finished step. Describe the source stage and new task. Preserves old runs; nothing is submitted.", "inputSchema": _schema({
                "workflow_id": _text(None), "step_id": _text(None), "instructions": _text("The source stage and requested follow-up calculation")}, ["workflow_id", "step_id", "instructions"])},
            {"name": "repair_step", "description": "Diagnose a failed step and draft a separate bounded repair. Keeps approved physics and previous runs; nothing is submitted.", "inputSchema": _schema({
                "workflow_id": _text(None), "step_id": _text(None), "instructions": _text("The user's repair request or answers to its questions")}, ["workflow_id", "step_id", "instructions"])},
            {"name": "prepare_inputs", "description": "Write the input files of a planned step locally for review. Nothing is submitted.", "inputSchema": _schema({"workflow_id": _text(None), "step_id": _text(None)}, ["workflow_id", "step_id"])},
            {"name": "review_inputs", "description": "Prepared input files, their contents and the requested resources for a step.", "inputSchema": _schema({"workflow_id": _text(None), "step_id": _text(None)}, ["workflow_id", "step_id"])},
            {"name": "request_submission", "description": "Report that a step is ready to submit. The user submits it in the app after reviewing.", "inputSchema": _schema({"workflow_id": _text(None), "step_id": _text(None)}, ["workflow_id", "step_id"])},
            {"name": "refresh_status", "description": "Current state of every step and job in a workflow, reading the cluster for active jobs.", "inputSchema": _schema({"workflow_id": _text(None)}, ["workflow_id"])},
            {"name": "explain_results", "description": "Answer a question about finished calculations from their verified outputs. The only source of numerical results.", "inputSchema": _schema({
                "run_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}, "question": _text(None)}, ["run_ids", "question"])},
            {"name": "cluster_status", "description": "Connected clusters, their work folders, schedulers, detected software and installation state.", "inputSchema": _schema()},
            {"name": "detect_software", "description": "Read partitions, modules, VASP and QE from a connected cluster and create default software environments.", "inputSchema": _schema({"connection_id": _text(None)}, ["connection_id"])},
            {"name": "qe_install_plan", "description": "Detect a compiler toolchain on the cluster and plan a managed Quantum ESPRESSO build in the user's storage. Nothing is submitted.", "inputSchema": _schema({
                "connection_id": _text(None), "build_cpus": {"type": "integer", "minimum": 1, "maximum": 64}, "partition": _text("Partition for the build job when the detected one is wrong")}, ["connection_id"])},
            {"name": "qe_install_status", "description": "Progress of the QE build; creates the QE environment when the build finished.", "inputSchema": _schema({"connection_id": _text(None)}, ["connection_id"])},
            {"name": "install_pseudopotentials", "description": "Install public PBE pseudopotentials (SSSP, PseudoDojo, or fully relativistic PseudoDojo FR for spin-orbit coupling) for elements into a QE environment's UPF folder.", "inputSchema": _schema({
                "environment_id": _text(None), "elements": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 40},
                "library": {"type": "string", "enum": ["sssp", "pseudodojo", "pseudodojo-fr"]}}, ["environment_id", "elements"])},
            {"name": "engine_install_plan", "description": "Read the cluster's CPU architecture and glibc and plan a pinned ABINIT, CP2K or Elk installation (conda-forge packages, offline micromamba, silicon test job). Nothing is downloaded or submitted.", "inputSchema": _schema({
                "connection_id": _text(None), "engine": {"type": "string", "enum": ["abinit", "cp2k", "elk"]},
                "build_cpus": {"type": "integer", "minimum": 1, "maximum": 64}, "partition": _text("Partition for the installation job when the detected one is wrong")}, ["connection_id", "engine"])},
            {"name": "engine_install_status", "description": "Progress of an ABINIT, CP2K or Elk installation; creates the software environment once its silicon test passed.", "inputSchema": _schema({
                "connection_id": _text(None), "engine": {"type": "string", "enum": ["abinit", "cp2k", "elk"]}}, ["connection_id", "engine"])},
            {"name": "install_library", "description": "Install pinned library files into an ABINIT (PseudoDojo psp8), CP2K (MOLOPT basis and GTH potentials) or Elk (species) environment.", "inputSchema": _schema({
                "environment_id": _text(None), "elements": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                "soc": {"type": "boolean"}, "functional": {"type": "string", "enum": ["PBE", "PBEsol"]}}, ["environment_id", "elements"])},
            {"name": "search_reference", "description": "Search the official QE, Elk, ABINIT or CP2K manual, input variables or keywords, tutorials and example inputs for the version an environment runs, and DFT Agent's verified cluster checks. Local search; returns ids for read_reference.", "inputSchema": _schema({
                "engine": {"type": "string", "enum": ["qe", "elk", "abinit", "cp2k"]},
                "query": _text("Words or a question, such as 'bootstrap kernel exciton' or 'EPS_SCF'"),
                "kind": {"type": "string", "enum": ["any", "keyword", "section", "task", "guide", "example", "lesson"]},
                "environment_id": _text("Software environment whose program version to use")}, ["engine", "query"])},
            {"name": "read_reference", "description": "One reference entry in full: an input variable or keyword with its type and default, a task, a manual section, an official example input or a verified check.", "inputSchema": _schema({
                "engine": {"type": "string", "enum": ["qe", "elk", "abinit", "cp2k"]},
                "id": _text("Entry id from search_reference, a variable name or a CP2K keyword path"),
                "environment_id": _text(None)}, ["engine", "id"])},
            {"name": "build_reference", "description": "Prepare the reference of a QE, Elk, ABINIT or CP2K environment: download the pinned manual and examples and, for CP2K, run cp2k --xml on the login node to read that build's keywords. Nothing is submitted.", "inputSchema": _schema({
                "environment_id": _text(None)}, ["environment_id"])},
            {"name": "remember", "description": "Save a project note the user explicitly asked to keep.", "inputSchema": _schema({"key": _text(None), "text": _text(None)}, ["key", "text"])},
        ]

    # --- helpers -----------------------------------------------------------

    def _settings(self):
        if not self.model_connection_id:
            raise ValueError("No model connection is bound to this agent turn.")
        self._require_connection(self.model_connection_id, "model")
        return self.registry.model_settings(self.model_connection_id)

    def _require_connection(self, identifier, kind="hpc"):
        project = self.workspace.get_project(self.project_id)
        if project["archived"] or project["deleted"]:
            raise ValueError("Open an active project first.")
        for item in self.registry.project_connections(self.project_id):
            if item["id"] == identifier and item["kind"] == kind and not item["revoked"] and not item["deleted"]:
                return item
        raise ValueError("Choose an active connection bound to this project.")

    def _environments(self):
        from .project_agent import _environments

        return _environments(self.registry, self.project_id)

    def _steps_for(self, workflow_id):
        return self.workspace.get_workflow(self.project_id, workflow_id)

    def _attempt(self, workflow_id, step_id):
        attempt = self.execution.latest(self.project_id, workflow_id, step_id)
        if attempt is None:
            return None
        proposal = attempt.get("proposal") or {}
        plan = proposal.get("plan") or proposal
        return {"state": attempt["state"], "run_id": attempt.get("run_id"), "environment_id": attempt.get("environment_id"),
                "reply": redact(attempt.get("reply") or ""), "last_error": redact(attempt.get("last_error") or ""),
                "questions": list(proposal.get("questions") or []), "summary": redact(str(proposal.get("summary") or "")),
                "plan": {key: value for key, value in plan.items() if key in {
                    "operations", "stages", "tasks", "variants", "parameters", "intent", "notes", "continuation",
                    "entrypoint", "outputs", "arguments", "inputs", "limits", "changes", "references"}} if isinstance(plan, dict) else None}

    # --- tools -------------------------------------------------------------

    def project_overview(self):
        from .project_files import list_files
        from .project_context import _private_file

        project = self.workspace.get_project(self.project_id)
        files = [{"path": item["path"], "size": item["size"]} for item in list_files(self.workspace, self.project_id, limit=400)
                 if Path(item["path"]).parts[0] == "files" and not _private_file(item["path"])]
        connections = [{"id": item["id"], "kind": item["kind"], "name": item["name"], "status": item["connection_status"],
                        **({"provider": item["settings"]["provider"]} if item["kind"] == "model" else
                           {"host": item["settings"]["host"], "remote_base": item["settings"].get("remote_base"),
                            "scheduler": item["settings"].get("scheduler")})}
                       for item in self.registry.project_connections(self.project_id) if not item["revoked"] and not item["deleted"]]
        bound = {item["id"] for item in connections if item["kind"] == "hpc"}
        environments = [{"id": item["id"], "name": item["name"], "engine": item["engine"], "connection_id": item["connection_id"],
                         "partition": item["config"].get("partition"),
                         "pseudopotentials": sorted((item["config"].get("qe_pseudopotentials") or {}).keys()) if item["engine"] == "qe" else None,
                         "library_files": len(item["config"].get("engine_library") or {}) if item["engine"] in {"abinit", "cp2k", "elk"} else None}
                        for item in self.registry.environments() if item["connection_id"] in bound]
        workflows = []
        for item in self.workspace.list_workflows(self.project_id):
            if item["status"] == "superseded":
                continue
            steps = []
            for step in item["steps"]:
                attempt = self._attempt(item["id"], step["id"])
                steps.append({"id": step["id"], "name": step["name"], "engine": step["engine"], "depends_on": step["depends_on"],
                              "goal": step["spec"].get("goal", ""), "state": attempt["state"] if attempt else step["status"],
                              "run_id": attempt["run_id"] if attempt else None})
            workflows.append({"id": item["id"], "title": item["title"], "goal": item["goal"], "status": item["status"], "steps": steps})
        runs = [{"id": item["id"], "run_id": item["run_id"], "engine": item["engine"], "status": item["status"],
                 "workflow_id": item.get("workflow_id"), "step_id": item.get("step_id")} for item in self.workspace.list_runs(self.project_id)]
        notes = [{"key": item["key"], "text": item["text"]} for item in self.workspace.list_memory(self.project_id)][:24]
        return {"project": {"id": project["id"], "name": project["name"]}, "files": files, "connections": connections,
                "environments": environments, "workflows": workflows, "runs": runs, "notes": notes}

    def read_file(self, path):
        from .project_agent import _read

        return _read(self.workspace, self.project_id, path)

    def write_file(self, path, content):
        from . import project_files
        from .project_agent import _draft_path

        if not isinstance(content, str) or len(content) > 200000:
            raise ValueError("Supply text content of at most 200,000 characters.")
        reject_ordinary_text(content)
        target = _draft_path(self.workspace, self.project_id, path)
        if not path.startswith("files/"):
            raise ValueError("Write drafts under files/.")
        created = not target.exists()
        result = project_files.write_file(self.workspace, self.project_id, path, content)
        return {"path": result["path"], "sha256": result["sha256"], "created": created}

    def describe_structure(self, path):
        from .project_context import _private_file
        from .structures import _load, _summary

        if _private_file(path):
            raise ValueError("Private files are excluded.")
        resolved = self.workspace.resolve_path(self.project_id, path, must_exist=True)
        _, _, structure, fmt = _load(resolved)
        summary = _summary(structure)
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
            summary["spacegroup"] = {"symbol": analyzer.get_space_group_symbol(), "number": analyzer.get_space_group_number()}
        except Exception:
            summary["spacegroup"] = None
        lattice = structure.lattice
        summary.update(format=fmt, path=path, elements=sorted({site.specie.symbol for site in structure}),
                       lattice={"a": round(lattice.a, 4), "b": round(lattice.b, 4), "c": round(lattice.c, 4),
                                "alpha": round(lattice.alpha, 3), "beta": round(lattice.beta, 3), "gamma": round(lattice.gamma, 3)})
        return summary

    def list_examples(self):
        from importlib.resources import files

        folder = files("vasp_slurm_agent").joinpath("examples")
        return {"examples": sorted(item.name for item in folder.iterdir() if item.name.endswith((".cif", ".vasp")))}

    def import_example(self, name):
        from importlib.resources import files
        from . import project_files

        if not isinstance(name, str) or name not in self.list_examples()["examples"]:
            raise ValueError("Choose an example from list_examples.")
        data = files("vasp_slurm_agent").joinpath("examples", name).read_bytes()
        result = project_files.write_file(self.workspace, self.project_id, "files/" + name, data)
        return {"path": result["path"], "sha256": result["sha256"]}

    def create_workflow(self, title, goal, steps):
        from .project_agent import _steps

        if not self.session_id:
            raise ValueError("No conversation is bound to this agent turn.")
        environments = self._environments()
        proposed = []
        for index, item in enumerate(steps):
            if not isinstance(item, dict):
                raise ValueError("Each step must be an object.")
            identifier = item.get("id") or f"step{index + 1}"
            if item.get("source_step"):
                source = {"kind": "step", "path": None, "step_id": item["source_step"], "stage": None}
            else:
                source = {"kind": "file", "path": item.get("source_path"), "step_id": None, "stage": None}
            depends_on = list(item.get("depends_on") or [])
            if item.get("source_step") and item["source_step"] not in depends_on:
                depends_on.append(item["source_step"])
            matching = [environment["id"] for environment in environments if environment["engine"] == item.get("engine")]
            proposed.append({"id": identifier, "name": item.get("name", identifier), "engine": item.get("engine"),
                             "goal": item.get("goal", ""), "source": source, "depends_on": depends_on,
                             "environment_id": item.get("environment_id") or (matching[0] if len(matching) == 1 else None),
                             **{name: item.get(name, []) for name in ("inputs", "input_artifacts", "arguments", "outputs")}})
            if source["kind"] == "file" and not source["path"]:
                raise ValueError(f"Step {identifier} needs source_path (a project file) or source_step.")
        checked = _steps(self.workspace, self.project_id, proposed, environments)
        result = self.workspace.create_workflow(self.project_id, self.session_id, title, goal, checked)
        return {"workflow_id": result["id"], "title": result["title"],
                "steps": [{"id": step["id"], "name": step["name"], "engine": step["engine"], "depends_on": step["depends_on"],
                           "environment_id": step["spec"].get("environment_id")} for step in result["steps"]]}

    def plan_step(self, workflow_id, step_id, instructions=""):
        self.execution.draft_step(self.project_id, workflow_id, step_id, self._settings(), message=instructions or None)
        return self._attempt(workflow_id, step_id)

    def continue_step(self, workflow_id, step_id, instructions):
        self.execution.continue_step(self.project_id, workflow_id, step_id, self._settings(), message=instructions)
        return self._attempt(workflow_id, step_id)

    def repair_step(self, workflow_id, step_id, instructions):
        self.execution.retry_step(self.project_id, workflow_id, step_id, self._settings(), message=instructions)
        return self._attempt(workflow_id, step_id)

    def prepare_inputs(self, workflow_id, step_id):
        attempt = self.execution.prepare_step(self.project_id, workflow_id, step_id)
        review = self.execution.review(self.project_id, workflow_id, step_id)
        return {"state": attempt["state"], "run_id": attempt["run_id"], "files": sorted(review["files"]),
                "next": "Call review_inputs, then ask the user to review and click Submit in the step card."}

    def review_inputs(self, workflow_id, step_id):
        from .project_files import read_file

        review = self.execution.review(self.project_id, workflow_id, step_id)
        root = self.workspace.project_paths(self.project_id)["root"]
        run = Path(review["run_dir"])
        contents = {}
        for relative in sorted(review["files"]):
            name = Path(relative).name
            if name in {"config.json", "plan.json", "proposal.json", "parent.json", "batch-plan.json"} or name.endswith((".json", ".cif")):
                continue
            try:
                text = read_file(self.workspace, self.project_id, (run / relative).relative_to(root).as_posix())["content"]
            except (ValueError, OSError):
                continue
            contents[relative] = text if len(text) <= 6000 else text[:6000] + "\n… (truncated)"
        state = review["state"]
        return {"digest": review["digest"], "files": sorted(review["files"]), "contents": contents,
                "resources": state.get("parameters") or state.get("task") or {},
                "stages": [{"name": stage.get("name"), "status": stage.get("status")} for stage in state.get("stages", [])]}

    def request_submission(self, workflow_id, step_id):
        attempt = self.execution.latest(self.project_id, workflow_id, step_id)
        if not attempt or attempt["state"] not in {"prepared", "authorized"}:
            raise ValueError("Prepare the inputs first; only prepared steps can be submitted.")
        self.workspace.record_event(self.project_id, "agent.requested_submission", {"workflow_id": workflow_id, "step_id": step_id})
        return {"status": "awaiting_user_approval",
                "message": "The step is prepared. The user reviews the inputs in the step card and clicks Submit; tools cannot submit."}

    def refresh_status(self, workflow_id):
        result = self.execution.refresh(self.project_id, workflow_id)
        return {"workflow": {"id": result["workflow"]["id"], "title": result["workflow"]["title"], "status": result["workflow"]["status"]},
                "steps": [{"step_id": item["step_id"], "state": item["attempt"]["state"] if item.get("attempt") else "pending",
                           "run_id": (item.get("attempt") or {}).get("run_id"),
                           "report": {key: (item.get("result") or {}).get(key) for key in ("status", "accepted", "error", "limits")}}
                          for item in result["steps"]]}

    def explain_results(self, run_ids, question):
        from . import explanation
        from .project_analysis import explain_results

        reject_ordinary_text(question)
        settings = self._settings()
        if len(run_ids) > 1:
            answer = explain_results(self.workspace, self.project_id, list(run_ids), settings, question)
        else:
            from .project_context import build_project_context

            context = build_project_context(self.workspace, self.project_id, run_ids=run_ids)
            record = self.workspace.get_run(self.project_id, run_ids[0])
            answer = explanation.explain_run(self.workspace.resolve_path(self.project_id, record["relative_path"]), settings, question=question)
            if answer["context_sha256"] != context["runs"][0]["context_sha256"]:
                raise ValueError("The calculation evidence changed. Ask again with its current results.")
        return {key: answer.get(key) for key in ("answer", "evidence", "limits", "next_steps")}

    def cluster_status(self):
        from . import references

        connections = [item for item in self.registry.project_connections(self.project_id)
                       if item["kind"] == "hpc" and not item["revoked"] and not item["deleted"]]
        result = []
        for item in connections:
            probe = item.get("probe") or {}
            environments = [{"id": environment["id"], "name": environment["name"], "engine": environment["engine"],
                             "partition": environment["config"].get("partition"),
                             "pseudopotentials": sorted((environment["config"].get("qe_pseudopotentials") or {}).keys())
                             if environment["engine"] == "qe" else None,
                             "library_files": sorted(environment["config"].get("engine_library") or {})[:40]
                             if environment["engine"] in {"abinit", "cp2k", "elk"} else None,
                             "reference": references.available(environment["engine"], self.registry._environment_config(environment))
                             if environment["engine"] in references.ENGINES else None}
                            for environment in self.registry.environments(item["id"])]
            install = probe.get("qe_install") or {}
            engine_installs = {engine: {key: (probe.get(f"{engine}_install") or {}).get(key)
                                        for key in ("status", "plan_id", "job_id", "error", "environment_id")}
                               for engine in ("abinit", "cp2k", "elk") if probe.get(f"{engine}_install")}
            result.append({"connection_id": item["id"], "name": item["name"], "host": item["settings"]["host"],
                           "status": item["connection_status"], "remote_base": item["settings"].get("remote_base"),
                           "scheduler": item["settings"].get("scheduler"), "schedulers_found": probe.get("schedulers"),
                           "environments": environments, "discovery": {key: probe.get("discovery", {}).get(key) for key in
                                                                        ("vasp_found", "qe_found", "notes", "suggested")},
                           "qe_install": {key: install.get(key) for key in ("status", "plan_id", "job_id", "bin_dir", "error", "environment_id")},
                           "engine_installs": engine_installs})
        return {"clusters": result}

    def detect_software(self, connection_id):
        self._require_connection(connection_id)
        result = self.registry.discover_environments(connection_id)
        return {"created": [{"id": item["id"], "name": item["name"], "engine": item["engine"]} for item in result["created"]],
                "missing": result["missing"], "notes": result["notes"], "vasp_found": result["vasp_found"], "qe_found": result["qe_found"],
                "open_source_codes_found": result.get("codes_found", [])}

    def qe_install_plan(self, connection_id, build_cpus=8, partition=None):
        self._require_connection(connection_id)
        result = self.registry.qe_install_plan(connection_id, build_cpus=build_cpus, partition=partition or None)
        return {**result, "next": "Open Connections, review this cluster's build plan and click Start the build. Only the user can submit it."}

    def qe_install_start(self, connection_id, plan_id, user_confirmed=False):
        raise ValueError("Start the build in Connections after reviewing its account and allocation. A model cannot authorize this job.")

    def qe_install_status(self, connection_id):
        self._require_connection(connection_id)
        return self.registry.qe_install_status(connection_id)

    def install_pseudopotentials(self, environment_id, elements, library="sssp"):
        self._require_connection(self.registry.environment(environment_id)["connection_id"])
        return self.registry.install_pseudopotentials(environment_id, list(elements), library=library or "sssp")

    def engine_install_plan(self, connection_id, engine, build_cpus=4, partition=None):
        self._require_connection(connection_id)
        result = self.registry.engine_install_plan(connection_id, engine, build_cpus=build_cpus, partition=partition or None)
        return {**result, "next": "Open Connections, review this installation and click Start. Only the user can start it."}

    def engine_install_status(self, connection_id, engine):
        self._require_connection(connection_id)
        return self.registry.engine_install_status(connection_id, engine)

    def install_library(self, environment_id, elements, soc=False, functional="PBE"):
        self._require_connection(self.registry.environment(environment_id)["connection_id"])
        return self.registry.install_engine_library(environment_id, list(elements), soc=soc, functional=functional)

    def _reference_config(self, engine, environment_id):
        if not environment_id:
            return None
        environment = self.registry.environment(environment_id)
        self._require_connection(environment["connection_id"])
        if environment["engine"] != engine:
            raise ValueError(f"That environment runs {environment['engine']}, not {engine}.")
        return self.registry._environment_config(environment)

    def search_reference(self, engine, query, kind="any", environment_id=None):
        from . import references

        config = self._reference_config(engine, environment_id)
        return references.search(engine, query, config=config, kinds=None if kind == "any" else [kind])

    def read_reference(self, engine, id, environment_id=None):  # noqa: A002 - the tool's argument name
        from . import references

        return references.read(engine, id, config=self._reference_config(engine, environment_id))

    def build_reference(self, environment_id):
        self._require_connection(self.registry.environment(environment_id)["connection_id"])
        return self.registry.build_reference(environment_id)

    def remember(self, key, text):
        result = self.workspace.save_memory(self.project_id, key, text)
        return {"key": result["key"], "text": result["text"]}

    def call(self, name, arguments):
        definitions = {item["name"]: item for item in self.definitions()}
        if name not in definitions:
            raise ValueError(f"Unknown tool {name}.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        _validate_arguments(arguments, definitions[name]["inputSchema"])
        return getattr(self, name)(**arguments)


# --- JSON-RPC over stdio -----------------------------------------------------

def _validate_arguments(value, schema):
    """Enforce the same bounded schemas advertised to the CLI before any action."""
    from .agent import AgentError, _check_schema

    if schema.get("type") == "object":
        if (not isinstance(value, dict) or not set(schema.get("required", ())) <= set(value)
                or not set(value) <= set(schema["properties"])):
            raise ValueError("Tool arguments do not match its schema.")
        for key, item in value.items():
            _validate_arguments(item, schema["properties"][key])
    elif schema.get("type") == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10000):
            raise ValueError("Tool arguments do not match its schema.")
        for item in value:
            _validate_arguments(item, schema["items"])
    else:
        try:
            _check_schema(value, schema)
        except AgentError:
            raise ValueError("Tool arguments do not match its schema.") from None

def _result(identifier, value):
    return {"jsonrpc": "2.0", "id": identifier, "result": value}


def _error(identifier, code, message):
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


def handle(tools, message):
    """One JSON-RPC message in, one response (or None for notifications) out."""
    if not isinstance(message, dict) or not isinstance(message.get("method"), str):
        return None
    method, identifier, params = message["method"], message.get("id"), message.get("params") or {}
    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        return _result(identifier, {"protocolVersion": requested if isinstance(requested, str) and requested else PROTOCOL_VERSION,
                                    "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": "dft-agent", "version": __version__}, "instructions": INSTRUCTIONS})
    if method == "ping":
        return _result(identifier, {})
    if method == "tools/list":
        return _result(identifier, {"tools": tools.definitions()})
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        arguments = params.get("arguments") if isinstance(params, dict) else None
        try:
            value = tools.call(name, {} if arguments is None else arguments)
            text = json.dumps(value, ensure_ascii=False, indent=1, default=str)
            return _result(identifier, {"content": [{"type": "text", "text": redact(text)}], "isError": False})
        except Exception as error:  # the model sees a short, redacted reason and can continue
            if isinstance(error, (ValueError, RuntimeError, OSError, TypeError, KeyError)):
                detail = redact(str(error)) or type(error).__name__
            else:
                detail = "The tool failed unexpectedly."
            print(f"{type(error).__name__}: {detail}", file=sys.stderr)
            return _result(identifier, {"content": [{"type": "text", "text": detail}], "isError": True})
    return _error(identifier, -32601, "Method not found")


def serve(stdin=None, stdout=None, tools=None):
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    tools = tools or ProjectTools.from_environment()
    for raw in iter(stdin.readline, b""):
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            response = _error(None, -32700, "Parse error")
        else:
            response = handle(tools, message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()
    return 0


def main():
    # Never let library warnings reach the protocol stream.
    sys.stdout = sys.stderr
    return serve(stdout=sys.__stdout__.buffer)


if __name__ == "__main__":
    sys.exit(main())
