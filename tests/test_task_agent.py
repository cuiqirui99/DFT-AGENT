import hashlib
from importlib.resources import files
import json
import sys
import zipfile

import pytest
from pymatgen.core import Structure

from vasp_slurm_agent import agent, cli, task_agent, workflow
from vasp_slurm_agent.config import ClusterConfig


def stage(task, **parameters):
    return {"task": task, "parameters": {**dict.fromkeys(agent.DEFAULTS), "spin": "none", "soc": False,
            "saxis": [0, 0, 1], "magmom": None, "functional": "PBE", "hubbard_u": [], **parameters},
            "requirements": {"spin": None, "soc": None, "functional": None, "hubbard_u": None}}


def response(stages=None, operations=None, variants=None):
    stages = stages or [stage("relax"), stage("bands", soc=True)]
    return {"status": "ready", "summary": "Relax, then calculate bands.", "operations": operations or [],
            "stages": stages, "variants": variants or [], "questions": [], "notes": [],
            "intent": {"requested_tasks": [item["task"] for item in stages], "forbidden_tasks": [], "unsupported": []}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(host="cluster.invalid", user="test", remote_root="/scratch/test",
        vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl", potcar_root="/licensed/pbe", partition="cpu")
    def no_network(*args, **kwargs):
        pytest.fail("Drafting and preparation must not submit jobs")
    monkeypatch.setattr(workflow, "SSHTransport", no_network)
    return source, tmp_path / "run", config


def mock_response(monkeypatch, result):
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(result))


def test_one_task_edits_relaxes_then_soc(setup, monkeypatch):
    source, root, config = setup
    original = source.read_bytes()
    result = response(operations=[{"type": "supercell", "matrix": [2, 1, 1]},
                                 {"type": "replace_sites", "indices": [0], "species": "Ge"}])
    mock_response(monkeypatch, result)
    plan = task_agent.draft_task("Make a 2x1x1 cell; replace site 0 with Ge; relax then SOC bands, axis z.", source, agent.ModelSettings("codex"))
    assert plan["status"] == "ready", plan["questions"]
    assert plan["preview"]["output"]["number_of_sites"] == 4
    state = task_agent.prepare_task(source, root, config, plan)
    assert source.read_bytes() == original
    assert [item["name"] for item in state["stages"]] == ["relax", "scf", "bands"]
    assert [item["parameters"]["soc"] for item in state["stages"]] == [False, True, True]
    assert state["stages"][2]["charge_from"] == 1
    structure = Structure.from_file(root / "01_relax/inputs/POSCAR")
    assert structure.composition.get_el_amt_dict() == {"Si": 3, "Ge": 1}
    saved = json.loads((root / "proposal.json").read_text())
    assert saved["goal"] == plan["goal"] and saved["dialogue"]
    with zipfile.ZipFile(workflow.bundle_run(root)) as bundle:
        assert "structure_edit/source.cif" in bundle.namelist()
        assert "proposal.json" in bundle.namelist()
    assert not any(item["job_id"] for item in state["stages"])


def test_revision_replaces_edits_not_double_applied(setup, monkeypatch):
    source, root, config = setup
    first = response(operations=[{"type": "supercell", "matrix": [2, 1, 1]}])
    mock_response(monkeypatch, first)
    plan = task_agent.draft_task("Make a supercell then relax and SOC bands, axis z.", source, agent.ModelSettings("codex"))
    seen = []
    def revised(payload, *args, **kwargs):
        seen.append(json.loads(payload))
        return json.dumps(response(operations=[{"type": "supercell", "matrix": [3, 1, 1]}]))
    monkeypatch.setattr(agent, "_request_structured", revised)
    history = plan["dialogue"] + [{"role": "user", "content": "Use 3x1x1 instead."}]
    plan = task_agent.draft_task(plan["goal"], source, agent.ModelSettings("codex"), history)
    assert plan["preview"]["output"]["number_of_sites"] == 6
    assert seen[0]["history"][-1] == history[-1]
    assert len(plan["dialogue"]) == 3
    task_agent.prepare_task(source, root, config, plan)
    assert len(Structure.from_file(root / "01_relax/inputs/POSCAR")) == 6


@pytest.mark.parametrize("issue", ["site", "axis", "u", "moments", "exclusion"])
def test_incomplete_task_never_prepares(setup, monkeypatch, issue):
    source, root, config = setup
    result = response()
    if issue == "site":
        result["operations"] = [{"type": "replace_sites", "indices": None, "species": "Ge"}]
    elif issue == "axis":
        result["stages"][1]["parameters"]["saxis"] = None
    elif issue == "u":
        result["stages"][0]["parameters"]["hubbard_u"] = [{"element": "Si", "l": 1, "u": 2, "j": None}]
    elif issue == "moments":
        result["stages"][0]["parameters"]["spin"] = "collinear"
        result["stages"][0]["parameters"]["magmom"] = [1]
    else:
        result["intent"]["forbidden_tasks"] = ["relax"]
    mock_response(monkeypatch, result)
    plan = task_agent.draft_task("Requested task", source, agent.ModelSettings("codex"))
    assert plan["status"] == "needs_input" and plan["questions"]
    with pytest.raises(ValueError, match="questions"):
        task_agent.prepare_task(source, root, config, plan)
    assert not root.exists()


def test_hse_and_strain_batch_cli(setup, monkeypatch, capsys):
    source, root, config = setup
    stages = [stage("relax", cell_relax=False), stage("bands", functional="HSE06")]
    variants = [{"label": label, "strain": [strain, 0, 0], "stages": stages}
                for label, strain in [("compressed", -.01), ("expanded", .01)]]
    mock_response(monkeypatch, response(stages=stages, variants=variants))
    config_path = config.save(root.parent / "cluster.json")
    plan_path = root.parent / "task.json"
    monkeypatch.setattr(sys, "argv", ["dft-agent", "task", str(source), "Compare strains, PBE relax then HSE06 bands.",
                                    "--provider", "codex", "--output", str(plan_path)])
    assert cli.main() == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["dft-agent", "prepare", str(source), str(root), "--plan", str(plan_path), "--config", str(config_path)])
    assert cli.main() == 0
    state = json.loads(capsys.readouterr().out)
    assert len(state["runs"]) == 2
    volumes = []
    for member in state["runs"]:
        child = workflow.read_state(root / member["folder"])
        assert [item["name"] for item in child["stages"]] == ["relax", "bands"]
        assert child["stages"][1]["parameters"]["functional"] == "HSE06"
        volumes.append(Structure.from_file(root / member["folder"] / "01_relax/inputs/POSCAR").volume)
    assert volumes[0] < volumes[1]
    monkeypatch.setattr(sys, "argv", ["dft-agent", "status", str(root)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["batch_id"] == state["batch_id"]


def test_changed_original_refuses_before_prepare(setup, monkeypatch):
    source, root, config = setup
    mock_response(monkeypatch, response())
    plan = task_agent.draft_task("Relax then SOC bands", source, agent.ModelSettings("codex"))
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="structure differs"):
        task_agent.prepare_task(source, root, config, plan)
    assert not root.exists()


def test_source_guidance_reaches_task_model(setup, monkeypatch):
    source, _, _ = setup
    payloads = []
    def request(payload, *args, **kwargs):
        payloads.append(json.loads(payload))
        return json.dumps(response([stage("bands", functional="HSE06")]))
    monkeypatch.setattr(agent, "_request_structured", request)
    plan = task_agent.draft_task("HSE06 bands", source, agent.ModelSettings("codex"))
    assert payloads[0]["scientific_context"]["knowledge"]
    assert plan["scientific_report"]["action_checks"]
    assert any(item["id"] == "hybrid" for item in plan["scientific_report"]["evidence"])


def test_workflow_actions_are_not_scientific_stage_identifiers():
    from vasp_slurm_agent import agent, task_agent
    schema = task_agent._schema()['properties']['intent']
    agent._check_schema({'requested_tasks':['relax'], 'forbidden_tasks':['bands'], 'unsupported':[]}, schema)
    for phrase in ('draft inputs without submission', 'submission', 'magnetism'):
        with pytest.raises(agent.AgentError):
            agent._check_schema({'requested_tasks':['relax',phrase], 'forbidden_tasks':[], 'unsupported':[]}, schema)
