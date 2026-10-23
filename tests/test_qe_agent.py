import hashlib
import json
from importlib.resources import files

import pytest
from pymatgen.core import Structure

from vasp_slurm_agent import agent, qe_agent, qe_workflow
from vasp_slurm_agent.config import ClusterConfig


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(host="example.invalid", user="test", remote_root="/scratch/test",
        partition="cpu", engine="qe", qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})
    monkeypatch.setattr(qe_workflow, "SSHTransport", lambda *a: pytest.fail("Planning must stay local"))
    return source, tmp_path / "run", config


def response(tasks=("relax", "bands"), parameters=None, operations=None):
    return {"status": "ready", "summary": "Relax and calculate bands.", "questions": [], "notes": [],
        "intent": {"requested_tasks": list(tasks), "forbidden_tasks": [], "unsupported": []},
        "operations": operations or [], "stages": [{"task": task, "parameters_json": json.dumps(parameters or {}),
        "requirements": {key: None for key in ("spin", "soc", "functional", "hubbard_u", "cell_relax")}} for task in tasks]}


def mock(monkeypatch, value):
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(value))


def test_review_edit_and_prepare_no_cluster(setup, monkeypatch):
    source, run, config = setup
    mock(monkeypatch, response(operations=[{"type": "supercell", "matrix": [2, 1, 1]}]))
    plan = qe_agent.draft_task("Double x; relax then bands", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "ready", plan["questions"]
    state = qe_agent.prepare_task(source, run, config, plan)
    assert [s["name"] for s in state["stages"]] == ["relax", "scf", "bands"]
    assert len(Structure.from_file(run / "01_relax/inputs/structure.cif")) == 4
    assert len(Structure.from_file(source)) == 2
    assert not any(s["job_id"] for s in state["stages"])
    assert json.loads((run / "proposal.json").read_text())["execution"]["plan_sha256"] == state["plan_sha256"]


@pytest.mark.parametrize("parameters", [{"ENCUT": 400}, {"spin": "collinear"}, {"soc": True},
    {"images": ["/invented/file"]}, {"namelists": {"system": {"prefix": "bad"}}}])
def test_model_errors_are_questions(setup, monkeypatch, parameters):
    source, run, config = setup
    mock(monkeypatch, response(("scf",), parameters))
    plan = qe_agent.draft_task("Calculate", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "needs_input" and plan["questions"]
    with pytest.raises(ValueError, match="questions"):
        qe_agent.prepare_task(source, run, config, plan)


def test_missing_neb_endpoints_is_question(setup, monkeypatch):
    source, _, config = setup
    mock(monkeypatch, response(("neb",)))
    plan = qe_agent.draft_task("Compute a barrier", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "needs_input"
    assert "final" in plan["questions"][0]


def test_endpoint_change_during_preparation_cannot_publish_run(setup, monkeypatch, tmp_path):
    source, run, config = setup
    final = tmp_path / "final.cif"
    crystal = Structure.from_file(source)
    crystal.translate_sites([0], [0.01, 0, 0])
    crystal.to(filename=final)
    mock(monkeypatch, response(("neb",), {"num_of_images": 3}))
    plan = qe_agent.draft_task("NEB", source, agent.ModelSettings("codex"), config=config, images=[source, final])
    assert plan["status"] == "ready", plan["questions"]
    review = qe_agent._review
    def change_after_review(*args, **kwargs):
        result = review(*args, **kwargs)
        crystal.translate_sites([0], [0.01, 0, 0])
        crystal.to(filename=final)
        return result
    monkeypatch.setattr(qe_agent, "_review", change_after_review)
    with pytest.raises(ValueError, match="images changed"):
        qe_agent.prepare_task(source, run, config, plan, images=[source, final])
    assert not run.exists()


def test_legacy_native_hubbard_cannot_bypass_no_u(setup, monkeypatch):
    source, _, config = setup
    plan = response(("scf",), {"namelists": {"system": {"lda_plus_u": True, "hubbard_u(1)": 5.0}}})
    plan["stages"][0]["requirements"]["hubbard_u"] = False
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("No Hubbard U", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "needs_input" and "Hubbard card" in result["questions"][0]


def test_changed_structure_blocks_submission(setup, monkeypatch):
    source, run, config = setup
    mock(monkeypatch, response(("scf",)))
    plan = qe_agent.draft_task("SCF", source, agent.ModelSettings("codex"), config=config)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        qe_agent.prepare_task(source, run, config, plan)
    assert not run.exists()


def test_hidden_parameters_cannot_override_reviewed_json(setup, monkeypatch):
    source, run, config = setup
    mock(monkeypatch, response(("scf",)))
    plan = qe_agent.draft_task("SCF", source, agent.ModelSettings("codex"), config=config)
    plan["stages"][0]["parameters"] = {"ecutwfc": 1}
    with pytest.raises(ValueError, match="proposal changed"):
        qe_agent.prepare_task(source, run, config, plan)
    assert not run.exists()


def test_revision_uses_original_once(setup, monkeypatch):
    source, run, config = setup
    mock(monkeypatch, response(("scf",), operations=[{"type": "supercell", "matrix": [2, 1, 1]}]))
    plan = qe_agent.draft_task("Double x", source, agent.ModelSettings("codex"), config=config)
    seen = []
    def reply(payload, *a, **kw):
        seen.append(json.loads(payload))
        return json.dumps(response(("scf",), operations=[{"type": "supercell", "matrix": [3, 1, 1]}]))
    monkeypatch.setattr(agent, "_request_structured", reply)
    revised = qe_agent.draft_task(plan["goal"], source, agent.ModelSettings("codex"),
        history=plan["dialogue"] + [{"role": "user", "content": "Triple x instead"}], config=config)
    qe_agent.prepare_task(source, run, config, revised)
    assert len(Structure.from_file(run / "01_scf/inputs/structure.cif")) == 6
    assert seen[0]["history"][-1]["content"] == "Triple x instead"


@pytest.mark.parametrize("key,value", [("soc", True), ("spin", "collinear"), ("functional", "HSE06"), ("hubbard_u", True), ("cell_relax", True)])
def test_dropped_explicit_requirement_is_not_ready(setup, monkeypatch, key, value):
    source, _, config = setup
    plan = response(("scf",))
    plan["stages"][0]["requirements"][key] = value
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Use the specified method", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "needs_input" and result["questions"]


def test_forbidden_prerequisite_is_not_silently_inserted(setup, monkeypatch):
    source, _, config = setup
    plan = response(("bands",))
    plan["intent"]["forbidden_tasks"] = ["scf"]
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Bands without SCF", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "needs_input"
    assert "excluded" in result["questions"][0]


def test_forbidden_cell_relaxation_keeps_fixed_cell_relaxation(setup, monkeypatch):
    source, _, config = setup
    plan = response(("relax", "bands"), {"cell_relax": False})
    plan["intent"]["forbidden_tasks"] = ["vc-relax"]
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Relax ions at fixed cell", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "ready", result["questions"]


def test_vc_relax_cannot_hide_as_relax_when_cell_change_forbidden(setup, monkeypatch):
    source, _, config = setup
    plan = response(("relax",), {"cell_relax": True})
    plan["intent"]["forbidden_tasks"] = ["vc-relax"]
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Relax ions at fixed cell", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "needs_input"


def test_requested_comparison_cannot_drop_a_case(setup, monkeypatch):
    source, _, config = setup
    plan = response(("scf",))
    plan["intent"]["requested_tasks"] = ["scf", "scf"]
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Compare two cases", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "needs_input"


def test_unimplemented_pairing_cannot_be_replaced_with_dos(setup, monkeypatch):
    source, _, config = setup
    plan = response(("dos",))
    plan["intent"]["unsupported"] = ["unconventional_pairing"]
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("Find unconventional pairing", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "unsupported"


def test_complete_electron_phonon_plan_expands_all_dependencies(setup, monkeypatch):
    source, run, config = setup
    parameters = {"mesh": [4, 4, 4], "dense_mesh": [8, 8, 8], "qmesh": [2, 2, 2],
                  "phonon_dos_mesh": [8, 8, 8], "mu_star": 0.1, "electronic_type": "metal"}
    mock(monkeypatch, response(("electron_phonon",), parameters))
    plan = qe_agent.draft_task("Test the conventional electron-phonon workflow", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "ready", plan["questions"]
    state = qe_agent.prepare_task(source, run, config, plan)
    assert [stage["name"] for stage in state["stages"]] == ["eph_dense_scf", "eph_scf", "eph_ph", "eph_q2r", "eph_matdyn", "eph_lambda"]


def test_electron_phonon_missing_mu_star_is_a_question(setup, monkeypatch):
    source, _, config = setup
    parameters = {"mesh": [4, 4, 4], "dense_mesh": [8, 8, 8], "qmesh": [2, 2, 2], "phonon_dos_mesh": [8, 8, 8]}
    mock(monkeypatch, response(("electron_phonon",), parameters))
    plan = qe_agent.draft_task("Estimate Tc", source, agent.ModelSettings("codex"), config=config)
    assert plan["status"] == "needs_input" and any("mu_star" in question for question in plan["questions"])


def test_guidance_and_verified_cases_reach_planning_model(setup, monkeypatch, tmp_path):
    from vasp_slurm_agent import qe_guidance
    source, _, config = setup
    guidance = {"guidance": [{"id": "qe-soc", "checks": ["Require fully relativistic UPFs."]}],
                "experience": [{"id": "qe-verified-case", "engine": "qe"}], "limits": ["Cases are advisory."]}
    requested, payloads = [], []
    def build(structure_path, **kwargs):
        requested.append((structure_path, kwargs))
        return guidance
    def reply(payload, *a, **kw):
        payloads.append(json.loads(payload))
        return json.dumps(response(("scf",)))
    monkeypatch.setattr(qe_guidance, "build_context", build)
    monkeypatch.setattr(agent, "_request_structured", reply)
    plan = qe_agent.draft_task("SCF", source, agent.ModelSettings("codex"), config=config, runs_root=tmp_path)
    assert requested == [(source, {"runs_root": tmp_path})]
    assert payloads[0]["scientific_context"] == plan["scientific_report"] == guidance


def test_planner_sees_each_upf_functional_and_type(setup, monkeypatch):
    # Without the functional, the planner asked whether the UPFs suit PBE+U.
    from dataclasses import replace
    source, _, config = setup
    config = replace(config, qe_pseudopotentials={"Si": {"filename": "Si.UPF", "sha256": "a" * 64, "functional": "PBE",
                                                         "pseudo_type": "NC", "relativistic": "scalar"}})
    seen = {}

    def reply(payload, *args, **kwargs):
        seen.update(json.loads(payload))
        return json.dumps(response(("scf",)))

    monkeypatch.setattr(agent, "_request_structured", reply)
    qe_agent.draft_task("SCF of silicon", source, agent.ModelSettings("codex"), config=config)
    assert seen["pseudopotentials"]["Si"] == {"configured": True, "fully_relativistic": False, "functional": "PBE", "pseudo_type": "NC"}


def test_unsupported_plan_is_not_asked_to_confirm_a_sequence(setup, monkeypatch):
    source, _, config = setup
    plan = response(("bands",))
    plan.update(status="unsupported", stages=[], intent={"requested_tasks": ["bands"], "forbidden_tasks": [], "unsupported": ["GW"]})
    mock(monkeypatch, plan)
    result = qe_agent.draft_task("GW bands of silicon", source, agent.ModelSettings("codex"), config=config)
    assert result["status"] == "unsupported" and not any("sequence" in question for question in result["questions"])
