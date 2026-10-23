import json
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from test_engine_integration import connected, keychain, project  # noqa: F401
from vasp_slurm_agent import engine_setup
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_app import _library_requirements


def test_library_status_checks_the_requested_method(connected):
    registry, cid = connected
    files = engine_setup.plan_library("abinit", ["Si"])
    environment = registry.create_environment(cid, "ABINIT", "abinit", {
        "partition": "cpu", "engine_command": "srun abinit", "engine_library_root": "/library",
        "engine_library": {item["name"]: {"sha256": item["sha256"]} for item in files}})
    assert registry.library_status(environment["id"], ["Si"])["missing"] == []
    for options, filename in [({"soc": True}, "Si.pbe_fr.psp8"),
                              ({"functional": "PBEsol"}, "Si.pbesol_sr.psp8")]:
        status = registry.library_status(environment["id"], ["Si"], **options)
        assert status == {"missing": ["Si"], "files": [filename]}


def test_library_requirements_follow_edited_structure_and_each_stage(project):
    proposal = {"plan": {"preview": {"output": {"sites": [{"element": "Si"}, {"element": "Ge"}]}},
                         "stages": [{"parameters_json": json.dumps(params)} for params in
                                    [{"functional": "PBEsol"}, {"functional": "PBE", "soc": True}]]}}
    step = {"engine": "abinit", "spec": {"source": {"kind": "file", "path": "files/Si.cif"}}}
    assert _library_requirements(project.workspace, project.pid, step, proposal) == [
        (["Ge", "Si"], False, "PBEsol"), (["Ge", "Si"], True, "PBE")]


def test_soc_library_install_button_remains_when_scalar_library_exists(project, monkeypatch):
    hpc = project.registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    files = engine_setup.plan_library("abinit", ["Si"])
    environment = project.registry.create_environment(hpc["id"], "ABINIT", "abinit", {
        "partition": "cpu", "engine_command": "srun abinit", "engine_library_root": "/library",
        "engine_library": {item["name"]: {"sha256": item["sha256"]} for item in files}})
    installed = Mock()
    monkeypatch.setattr(ConnectionRegistry, "install_engine_library", installed)
    proposal = {"plan": {"stages": [{"parameters_json": '{"soc": true}'}]}}
    script = f'''
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_app import _library_check
from vasp_slurm_agent.workspace import Workspace
w = Workspace({str(project.workspace.home)!r})
step = {{"engine": "abinit", "spec": {{"source": {{"kind": "file", "path": "files/Si.cif"}}}}}}
_library_check(w, ConnectionRegistry(w), {project.pid!r}, step, {environment["id"]!r}, "soc_", {proposal!r})
'''
    view = AppTest.from_string(script).run()
    assert not view.exception
    assert any("PBE + SOC" in caption.value for caption in view.caption)
    next(button for button in view.button if button.label == "Install library files").click().run()
    assert not view.exception
    installed.assert_called_once_with(environment["id"], ["Si"], soc=True, functional="PBE")


@pytest.mark.parametrize("parameters", [{"soc": True}, {"namelists": {"system": {"lspinorb": True}}}])
def test_qe_soc_install_uses_edited_elements_and_relativistic_library(project, monkeypatch, parameters):
    hpc = project.registry.create("hpc", "Cluster", {"host": "h.invalid", "user": "u", "remote_base": "/scratch/u", "scheduler": "slurm"})
    environment = project.registry.create_environment(hpc["id"], "QE", "qe", {
        "partition": "cpu", "qe_command": "srun pw.x", "qe_pseudo_root": "/library",
        "qe_pseudopotentials": {"Si": {"filename": "Si.UPF", "relativistic": "scalar"}}})
    installed = Mock()
    monkeypatch.setattr(ConnectionRegistry, "install_pseudopotentials", installed)
    proposal = {"plan": {"preview": {"output": {"sites": [{"element": "Si"}, {"element": "Ge"}]}},
                         "stages": [{"task": "scf", "parameters_json": json.dumps(parameters)}]}}
    script = f'''
import streamlit as st
from vasp_slurm_agent.connections import ConnectionRegistry
from vasp_slurm_agent.project_app import _pseudopotential_check
from vasp_slurm_agent.workspace import Workspace
w = Workspace({str(project.workspace.home)!r})
step = {{"engine": "qe", "spec": {{"source": {{"kind": "file", "path": "files/Si.cif"}}}}}}
ready = _pseudopotential_check(w, ConnectionRegistry(w), {project.pid!r}, step, {environment['id']!r}, "soc_", {proposal!r})
st.button("Prepare inputs", disabled=not ready)
'''
    view = AppTest.from_string(script).run()
    assert not view.exception
    assert view.selectbox[0].value == "pseudodojo-fr" and len(view.selectbox[0].options) == 1
    assert next(button for button in view.button if button.label == "Prepare inputs").disabled
    next(button for button in view.button if button.label == "Install public UPFs").click().run()
    assert not view.exception
    installed.assert_called_once_with(environment["id"], ["Ge", "Si"], library="pseudodojo-fr")


def test_planning_again_preserves_active_receipt(connected, monkeypatch):
    registry, cid = connected
    monkeypatch.setattr(engine_setup, "detect_platform", lambda *a, **k: {"platform": "linux-64", "glibc": "2.34"})
    first = registry.engine_install_plan(cid, "elk", partition="cpu")
    record = registry.get(cid)["probe"]["elk_install"]
    record.update(status="submitted", receipt={"job_id": "41"}, job_id="41")
    registry._store_probe(cid, "elk_install", record)
    again = registry.engine_install_plan(cid, "elk", partition="cpu")
    assert again["status"] == "submitted" and again["plan_id"] == first["plan_id"]
    assert registry.get(cid)["probe"]["elk_install"]["receipt"] == {"job_id": "41"}
    with pytest.raises(ValueError, match="already exists"):
        registry.engine_install_plan(cid, "elk", partition="other")


def test_qe_module_is_discovered_without_pw_on_login_path(connected, monkeypatch):
    from vasp_slurm_agent import discovery
    registry, cid = connected
    facts = discovery.parse_probe(f"{discovery.MARK} codes\nmodule QuantumESPRESSO/recommendation\n"
                                  "module QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb\n")
    monkeypatch.setattr(discovery, "probe_cluster", lambda *a, **k: {
        "facts": facts, "suggested": {"partition": "cpu"}, "notes": []})
    result = registry.discover_environments(cid)
    assert result["qe_found"]
    environment = next(item for item in result["created"] if item["engine"] == "qe")
    assert environment["config"]["setup_commands"] == ["module load QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb"]
