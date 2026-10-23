"""ABINIT, CP2K and Elk across the workspace, connections, discovery and agent tools."""
import sqlite3
from unittest.mock import Mock

import pytest

from test_agent_mode import call, keychain, project, tools_for  # noqa: F401  (fixtures)
from vasp_slurm_agent import discovery, engine_setup, transport, workspace as workspace_module
from vasp_slurm_agent.connections import ConnectionRegistry, site_library_root
from vasp_slurm_agent.workspace import Workspace


def _step(identifier, engine):
    return {"id": identifier, "name": identifier.upper(), "engine": engine, "depends_on": [], "spec": {"goal": "Check silicon"}}


def test_older_workspaces_migrate_and_keep_their_rows(tmp_path, monkeypatch, keychain):
    monkeypatch.setattr(workspace_module, "_MIGRATIONS", workspace_module._MIGRATIONS[:4])
    old = Workspace(tmp_path / "home")
    registry = ConnectionRegistry(old)
    pid = old.create_project("Silicon")["id"]
    sid = old.create_session(pid)["id"]
    workflow = old.create_workflow(pid, sid, "QE", "Bands", [_step("scf", "qe")])
    cid = registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})["id"]
    environment = registry.create_environment(cid, "QE", "qe", {"partition": "cpu", "qe_command": "srun pw.x", "qe_pseudo_root": "/p"})
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        old.create_workflow(pid, sid, "ABINIT", "Bands", [_step("scf", "abinit")])
    monkeypatch.undo()
    upgraded = Workspace(tmp_path / "home")
    with upgraded._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 5
        assert db.execute("PRAGMA foreign_key_check").fetchone() is None
    assert upgraded.get_workflow(pid, workflow["id"])["steps"][0]["engine"] == "qe"
    registry = ConnectionRegistry(upgraded)
    assert registry.environment(environment["id"])["config"]["qe_command"] == "srun pw.x"
    for engine in ("abinit", "cp2k", "elk"):
        created = upgraded.create_workflow(pid, sid, engine, "Bands", [_step("scf", engine)])
        assert created["steps"][0]["engine"] == engine
        registry.create_environment(cid, engine, engine, {"partition": "cpu", "engine_command": "srun " + {"abinit": "abinit", "cp2k": "cp2k.psmp", "elk": "elk"}[engine],
                                                          "engine_library_root": site_library_root(engine, "/scratch/u")})
    assert {item["engine"] for item in registry.environments(cid)} == {"qe", "abinit", "cp2k", "elk"}


def test_probe_reports_open_source_programs_and_modules():
    output = "\n".join([f"{discovery.MARK} codes", "exe cp2k.psmp=/opt/cp2k/bin/cp2k.psmp", "module abinit/9.10.3",
                        "module ABINIT/10.0.3-foss-2023a", "module Elk/9.6.8", "exe elk=relative/elk", f"{discovery.MARK} scratch", "/scratch/u"])
    codes = discovery.parse_probe(output)["codes"]
    assert codes["cp2k"]["executables"] == {"cp2k.psmp": "/opt/cp2k/bin/cp2k.psmp"}
    assert codes["abinit"]["modules"] == ["abinit/9.10.3", "ABINIT/10.0.3-foss-2023a"]
    assert codes["elk"] == {"executables": {}, "modules": ["Elk/9.6.8"]}


class _Remote:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


@pytest.fixture
def connected(tmp_path, keychain, monkeypatch):
    registry = ConnectionRegistry(tmp_path / "home")
    cid = registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})["id"]
    monkeypatch.setattr(registry, "_probe_hpc", lambda item: {"auth_status": "available", "connection_status": "connected",
                                                              "schedulers": ["slurm"], "detail": "SSH connected."})
    registry.probe(cid)
    monkeypatch.setattr(transport, "SSHTransport", _Remote)
    monkeypatch.setattr(registry, "_qe_transport", lambda item: _Remote())
    return registry, cid


def test_site_programs_become_environments_with_pinned_library_folders(connected, monkeypatch):
    registry, cid = connected
    facts = {"partitions": [{"name": "cpu", "default": True, "timelimit": "1-00:00:00", "gpus_per_node": 0}], "accounts": [],
             "modules": [], "executables": {}, "potcar_roots": [], "scratch_dirs": [], "slurm": ["sbatch", "squeue", "sacct", "scancel"],
             "codes": {"abinit": {"executables": {}, "modules": ["abinit/9.10.3", "abinit/10.0.3"]},
                       "cp2k": {"executables": {"cp2k.psmp": "/opt/cp2k.psmp"}, "modules": []},
                       "elk": {"executables": {}, "modules": []}}}
    monkeypatch.setattr(discovery, "probe_cluster", lambda config, transport=None, timeout=90: {"facts": facts, **discovery.suggest(facts)})
    result = registry.discover_environments(cid)
    assert result["codes_found"] == ["abinit", "cp2k"]
    created = {item["engine"]: item["config"] for item in result["created"]}
    assert created["abinit"]["setup_commands"] == ["module load abinit/10.0.3"] and created["abinit"]["engine_command"] == "srun abinit"
    assert created["cp2k"]["engine_command"] == "srun cp2k.psmp" and created["cp2k"]["engine_library_root"] == "/scratch/u/libraries/cp2k-2026.2"
    assert registry.discover_environments(cid)["created"] == []


def test_finished_install_creates_one_environment_with_recorded_library(connected, monkeypatch):
    registry, cid = connected
    registry._store_probe(cid, "discovery", {"suggested": {"partition": "cpu"}})
    monkeypatch.setattr(engine_setup, "detect_platform", lambda config, transport=None: {"platform": "linux-64", "glibc": "2.28"})
    plan = registry.engine_install_plan(cid, "elk")
    assert plan["status"] == "planned" and plan["platform"] == "linux-64" and plan["download_mb"] > 50
    with pytest.raises(ValueError, match="reviewed"):
        registry.engine_install_start(cid, "elk", "another-plan")
    record = registry.get(cid)["probe"]["elk_install"]
    registry._store_probe(cid, "elk_install", {**record, "status": "submitted", "receipt": {"job_id": "41", "status": "submitted"}})
    poll = Mock(return_value={"status": "installed", "scheduler_state": "COMPLETED",
                              "smoke": {"success": True, "final_energy_ev": -15783.9}})
    monkeypatch.setattr(engine_setup, "poll_install", poll)
    first = registry.engine_install_status(cid, "elk")
    environment = registry.environment(first["environment_id"])
    assert environment["engine"] == "elk" and environment["config"]["engine_command"].endswith("/bin/elk")
    assert environment["config"]["tasks"] == 1 and environment["config"]["cpus_per_task"] == 8
    assert "Si.in" in environment["config"]["engine_library"]
    assert registry.engine_install_status(cid, "elk")["environment_id"] == first["environment_id"]
    assert poll.call_count == 1
    assert registry.library_status(first["environment_id"], ["Si", "Fe"])["missing"] == []


def test_failed_install_is_planned_again_as_a_new_attempt(connected, monkeypatch):
    registry, cid = connected
    registry._store_probe(cid, "discovery", {"suggested": {"partition": "cpu"}})
    monkeypatch.setattr(engine_setup, "detect_platform", lambda config, transport=None: {"platform": "linux-64", "glibc": "2.34"})
    first = registry.engine_install_plan(cid, "cp2k")
    assert registry.engine_install_plan(cid, "cp2k")["plan_id"] == first["plan_id"]
    record = registry.get(cid)["probe"]["cp2k_install"]
    registry._store_probe(cid, "cp2k_install", {**record, "status": "failed", "receipt": {"job_id": "41", "status": "submitted"}})
    second = registry.engine_install_plan(cid, "cp2k")
    retry = registry.get(cid)["probe"]["cp2k_install"]
    # A new plan, job and folder; the same environment prefix, which the installer reuses.
    assert second["plan_id"] != first["plan_id"] and retry["attempt"] == 1 and retry["status"] == "planned"
    assert retry["plan"]["prefix"] == first["prefix"] and retry["plan"]["work_dir"].endswith("cp2k-" + second["plan_id"])
    assert registry.engine_install_plan(cid, "cp2k")["plan_id"] == second["plan_id"]


def test_library_installation_records_hashes(connected, monkeypatch):
    registry, cid = connected
    environment = registry.create_environment(cid, "ABINIT", "abinit", {"partition": "cpu", "engine_command": "srun abinit",
                                                                         "engine_library_root": "/scratch/u/pseudos/abinit"})
    assert registry.library_status(environment["id"], ["Si", "O"])["missing"] == ["O", "Si"]

    def ensure(config, elements, transport=None, **options):
        assert config.engine == "abinit" and options == {"soc": False, "functional": "PBE"}
        return {"library_root": config.engine_library_root,
                "files": {item["name"]: {"sha256": item["sha256"], "element": item["element"], "library": item["library"]}
                          for item in engine_setup.plan_library("abinit", elements)}}

    monkeypatch.setattr(engine_setup, "ensure_library", ensure)
    result = registry.install_engine_library(environment["id"], ["Si", "O", "Si"])
    assert result["installed"] == ["O.pbe_sr.psp8", "Si.pbe_sr.psp8"]
    assert registry.library_status(environment["id"], ["Si", "O"])["missing"] == []
    with pytest.raises(ValueError, match="element symbols"):
        registry.install_engine_library(environment["id"], ["silicon"])


def test_agent_tools_plan_installs_but_cannot_start_them(project, monkeypatch):
    tools = tools_for(project)
    names = {item["name"] for item in tools.definitions()}
    assert {"engine_install_plan", "engine_install_status", "install_library"} <= names
    assert "engine_install_start" not in names
    hpc = project.registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    planned = Mock(return_value={"plan_id": "p1", "status": "planned"})
    monkeypatch.setattr(ConnectionRegistry, "engine_install_plan", planned)
    error, reason = call(tools, "engine_install_plan", connection_id=hpc["id"], engine="cp2k")
    assert error and "bound to this project" in reason and not planned.called
    project.registry.bind(project.pid, hpc["id"])
    error, value = call(tools, "engine_install_plan", connection_id=hpc["id"], engine="cp2k")
    assert not error and value["plan_id"] == "p1" and "Only the user can start it" in value["next"]
    error, reason = call(tools, "engine_install_plan", connection_id=hpc["id"], engine="siesta")
    assert error


def test_agent_can_record_a_cp2k_workflow_step(project):
    tools = tools_for(project)
    error, value = call(tools, "create_workflow", title="Si with CP2K", goal="Bands", steps=[
        {"id": "bands", "name": "Bands", "engine": "cp2k", "goal": "Silicon band structure", "source_path": "files/Si.cif"}])
    assert not error and value["steps"][0]["engine"] == "cp2k"


def test_connections_screen_offers_reviewed_open_source_installs(keychain, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from vasp_slurm_agent.i18n import set_language

    registry = ConnectionRegistry()
    cid = registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})["id"]
    registry._store_probe(cid, "discovery", {"suggested": {"partition": "cpu", "account": "naiss"}, "qe_found": True})
    plan = engine_setup.install_plan(registry._engine_base_config(registry.get(cid), "elk", partition="cpu", account="naiss"), "elk",
                                     {"platform": "linux-64", "glibc": "2.28"})
    registry._store_probe(cid, "elk_install", {"status": "planned", "plan": plan, "plan_id": plan["id"], "partition": "cpu", "account": "naiss"})
    registry.create_environment(cid, "ABINIT", "abinit", {"partition": "cpu", "engine_command": "srun abinit",
                                                          "engine_library_root": "/scratch/u/pseudos/abinit"})
    script = f"""
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.i18n import set_language
from vasp_slurm_agent.setup_app import _environments_panel
set_language(st.session_state.get("language", "en"))
registry = ConnectionRegistry()
_environments_panel(registry, registry.get({cid!r}), "t_", {{"discovered": True}})
"""
    view = AppTest.from_string(script, default_timeout=30).run()
    assert not view.exception, view.exception
    assert any("Install open-source codes" in item.value for item in view.markdown)
    choices = view.radio[0]
    assert choices.options == ["CP2K", "Elk"] and choices.value == "cp2k"
    choices.set_value("elk").run()
    start = next(button for button in view.button if button.label == "Start the installation")
    assert start.disabled
    assert any(plan["prefix"] in item.value for item in view.code)
    started = Mock()
    monkeypatch.setattr(ConnectionRegistry, "engine_install_start", started)
    view.checkbox[0].check().run()
    next(button for button in view.button if button.label == "Start the installation").click().run()
    assert not view.exception
    started.assert_called_once()
    assert started.call_args.args == (cid, "elk", plan["id"])
    view.session_state["language"] = "zh"
    view.run()
    assert any("安装开源程序" in item.value for item in view.markdown)
    set_language("en")


def test_step_card_offers_missing_library_files(project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    hpc = project.registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    environment = project.registry.create_environment(hpc["id"], "ABINIT", "abinit", {
        "partition": "cpu", "engine_command": "srun abinit", "engine_library_root": "/scratch/u/pseudos/abinit"})
    installed = Mock(return_value={"installed": ["Si.pbe_sr.psp8"]})
    monkeypatch.setattr(ConnectionRegistry, "install_engine_library", installed)
    script = f"""
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_app import _library_check
from vasp_slurm_agent.workspace import Workspace
workspace = Workspace({str(project.workspace.home)!r})
step = {{"engine": "abinit", "spec": {{"source": {{"kind": "file", "path": "files/Si.cif"}}}}}}
_library_check(workspace, ConnectionRegistry(workspace), {project.pid!r}, step, {environment["id"]!r}, "p_")
"""
    view = AppTest.from_string(script, default_timeout=30).run()
    assert not view.exception, view.exception
    assert any("pseudopotentials for Si" in item.value for item in view.info)
    next(button for button in view.button if button.label == "Install library files").click().run()
    assert not view.exception
    installed.assert_called_once_with(environment["id"], ["Si"])


@pytest.mark.parametrize("engine", ["abinit", "cp2k", "elk"])
def test_project_step_plans_and_prepares_through_the_conversation(project, monkeypatch, engine):
    import json
    from vasp_slurm_agent import agent
    from vasp_slurm_agent.project_execution import ProjectExecution
    from test_engine_adapters import _plan

    hpc = project.registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    project.registry.bind(project.pid, hpc["id"])
    command = {"abinit": "srun abinit", "cp2k": "srun cp2k.psmp", "elk": "elk"}[engine]
    environment = project.registry.create_environment(hpc["id"], engine, engine, {
        "partition": "cpu", "engine_command": command, "engine_library_root": site_library_root(engine, "/scratch/u")})
    workflow = project.workspace.create_workflow(project.pid, project.sid, "Si", "Bands", [
        {"id": "bands", "name": "Bands", "engine": engine, "depends_on": [],
         "spec": {"goal": "Silicon band structure", "source": {"kind": "file", "path": "files/Si.cif"}}}])

    def model(payload, schema, settings, *, instructions, name="dft_response"):
        if name == "dft_conversation":
            return json.dumps({"action": "plan", "reply": "", "source_stage": None, "original_source": False})
        assert name == f"{engine}_task"
        return json.dumps(_plan(["bands"], {"mesh": [2, 2, 2], "line_density": 5}))

    monkeypatch.setattr(agent, "_request_structured", model)
    execution = ProjectExecution(project.workspace, project.registry)
    execution.configure_step(project.pid, workflow["id"], "bands", environment["id"])
    attempt = execution.draft_step(project.pid, workflow["id"], "bands", project.registry.model_settings(project.model))
    assert attempt["state"] == "ready", attempt.get("last_error")
    prepared = execution.prepare_step(project.pid, workflow["id"], "bands")
    assert prepared["state"] == "prepared" and prepared["run_id"]
    review = execution.review(project.pid, workflow["id"], "bands")
    assert [stage["name"] for stage in review["state"]["stages"]] == ["scf", "bands"]
    assert review["state"]["engine"] == engine and review["state"]["remote_root"].startswith("/scratch/u/projects/")


def test_module_choice_follows_the_real_arrhenius_listing():
    # Lmod's terse listing on Arrhenius: folder entries, a recommendation alias and versions.
    lines = ["ABINIT/", "ABINIT/10.6.7-gcc-2026.03-mpich-eb", "ABINIT/recommendation", "Elk/", "Elk/11.0.2-gcc-2026.03-mpich-eb",
             "Elk/8.5.2-gcc-2025b-eb", "Elk/9.6.8-intel-2025b-eb", "Elk/recommendation"]
    output = "\n".join([f"{discovery.MARK} codes", *("module " + line for line in lines),
                        f"{discovery.MARK} modules", "VASP/", "VASP/5.4.4-gcc-2026.03-mpich", "VASP/6.5.1-gcc-2026.03-mpich"])
    facts = discovery.parse_probe(output)
    assert "ABINIT/" not in facts["codes"]["abinit"]["modules"] and "VASP/" not in facts["modules"]
    assert discovery.preferred_module(facts["codes"]["elk"]["modules"]) == "Elk/11.0.2-gcc-2026.03-mpich-eb"
    assert discovery.preferred_module(facts["codes"]["abinit"]["modules"]) == "ABINIT/10.6.7-gcc-2026.03-mpich-eb"
    assert discovery.preferred_module(["CP2K/recommendation"]) == "CP2K/recommendation"
    assert discovery.suggest({**facts, "partitions": []})["suggested"]["setup_commands"] == ["module load VASP/6.5.1-gcc-2026.03-mpich"]
