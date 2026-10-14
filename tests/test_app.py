"""Exercise user actions through Streamlit; all remote execution is stubbed."""

from importlib.resources import files
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Incar, Kpoints
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent import agent, task_agent
from test_cli_plan import model_response


MODEL_USAGE = {"input_tokens": 1234, "output_tokens": 56, "cached_input_tokens": 1024,
               "reasoning_tokens": None, "latency_seconds": 1.25}


def task_response(tasks=None, missing_moments=False):
    legacy = model_response(tasks, missing_moments)
    parameters = deepcopy(legacy["parameters"])
    parameters["hubbard_u"] = [{"element": element, **value}
                              for element, value in parameters["hubbard_u"].items() if value is not None]
    return {"status": legacy["status"], "summary": legacy["summary"], "operations": [],
            "stages": [{"task": task, "parameters": deepcopy(parameters),
                        "requirements": {key: legacy["intent"][key]
                                         for key in ("spin", "soc", "functional", "hubbard_u")}}
                       for task in legacy["tasks"]],
            "variants": [], "questions": legacy["questions"], "notes": legacy["notes"],
            "intent": {key: legacy["intent"][key]
                       for key in ("requested_tasks", "forbidden_tasks", "unsupported")}}


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    from vasp_slurm_agent import explanation

    config = ClusterConfig(
        host="cluster.example.invalid", user="researcher", remote_root="/scratch/test",
        vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu",
    )
    config_path = config.save(tmp_path / "cluster.json")
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(config_path))

    def no_remote(*args, **kwargs):
        pytest.fail("UI smoke tests must never contact an SSH server")

    monkeypatch.setattr(workflow, "SSHTransport", no_remote)
    worker = Mock(return_value=43210)
    monkeypatch.setattr(workflow, "start_worker", worker)

    def fixture_context(run_dir):
        state = workflow.read_state(run_dir)
        return {"status": state["status"], "context_sha256": hashlib.sha256(json.dumps(state).encode()).hexdigest(),
                "facts": {f"stage_{index}.accepted": {"value": bool((stage.get("result") or {}).get("success"))}
                          for index, stage in enumerate(state["stages"], 1)}, "goal": "", "limits": []}

    monkeypatch.setattr(explanation, "load_run_context", fixture_context)
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(app.text_input, "Run folder").set_value(str(tmp_path / "runs")).run()
    widget(app.radio, "Mode").set_value("Manual").run()
    assert not app.exception
    return app, worker, tmp_path / "runs"


def silicon_bytes():
    return files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()


def usage_caption(app):
    return next(item.value for item in app.caption if item.value.startswith("Model usage ·"))


def test_usage_display_does_not_render_provider_text_or_invalid_numbers():
    app = AppTest.from_string("""
from vasp_slurm_agent.app import _model_usage
_model_usage({'input_tokens': True, 'output_tokens': -1, 'cached_input_tokens': '/private/token',
              'reasoning_tokens': None, 'latency_seconds': float('nan'), 'model':'secret-key'})
""").run()
    assert not app.exception
    text = usage_caption(app)
    assert text.count("unknown") == 4
    assert "secret-key" not in text and "/private" not in text and "nan" not in text


@pytest.mark.parametrize("file_format", ["cif", "poscar"])
def test_structure_import_and_review_gate(workbench, file_format):
    app, worker, runs_root = workbench
    assert widget(app.button, "Prepare inputs").disabled
    if file_format == "cif":
        filename, payload = "Si.cif", silicon_bytes()
    else:
        structure = Structure.from_str(silicon_bytes().decode(), fmt="cif")
        filename, payload = "POSCAR", structure.to(fmt="poscar").encode()
    app.file_uploader[0].upload(filename, payload).run()
    assert not app.exception
    assert widget(app.metric, "Formula").value == "Si"
    assert not widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()

    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["status"] == "planned"
    assert state["stages"][0]["job_id"] is None
    assert (run_dir / "01_relax" / "inputs" / "POSCAR").is_file()
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()

    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    assert not widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    widget(app.button, "Submit calculation").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    assert any("43210" in message.value for message in app.success)


def test_invalid_structure_cannot_be_prepared(workbench):
    app, worker, runs_root = workbench
    app.file_uploader[0].upload("POSCAR", b"This is not a crystal structure").run()
    assert not app.exception
    assert any("Cannot read structure" in message.value for message in app.error)
    assert widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()


@pytest.mark.parametrize("electronic_type, expected_smearing, expected_sigma", [
    ("auto", 0, 0.05), ("metal", 1, 0.2),
])
def test_manual_automatic_numerics_reach_inputs(workbench, electronic_type, expected_smearing, expected_sigma):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    assert widget(app.text_input, "Gamma-centered k-point grid").value == ""
    assert widget(app.selectbox, "ISMEAR").value is None
    widget(app.selectbox, "Electronic type").select(electronic_type)
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    inputs = run_dir / "01_relax" / "inputs"
    metadata = json.loads((inputs / "metadata.json").read_text())
    incar = Incar.from_file(inputs / "INCAR")
    assert Kpoints.from_file(inputs / "KPOINTS").kpts == [(9, 9, 9)]
    assert metadata["numerical_choices"]["mesh_mode"] == "reciprocal_spacing"
    assert metadata["parameters"]["electronic_type"] == electronic_type
    assert incar["ISMEAR"] == expected_smearing and incar["SIGMA"] == expected_sigma
    assert incar["ISIF"] == 2
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_manual_explicit_numerics_override_automatic_choices(workbench):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Electronic type").select("metal")
    widget(app.number_input, "K-point spacing / Å⁻¹").set_value(0.1)
    widget(app.text_input, "Gamma-centered k-point grid").set_value("3, 4, 5")
    widget(app.selectbox, "ISMEAR").select(-1)
    widget(app.text_input, "SIGMA / eV (optional)").set_value("0.08")
    widget(app.checkbox, "Relax the cell as well as atomic positions").check()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    inputs = run_dir / "01_relax" / "inputs"
    metadata = json.loads((inputs / "metadata.json").read_text())
    incar = Incar.from_file(inputs / "INCAR")
    assert Kpoints.from_file(inputs / "KPOINTS").kpts == [(3, 4, 5)]
    assert metadata["numerical_choices"]["mesh_mode"] == "explicit"
    assert incar["ISMEAR"] == -1 and incar["SIGMA"] == 0.08 and incar["ISIF"] == 3
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "-0.1", "not a number"])
def test_manual_invalid_sigma_is_visible_and_does_not_submit(workbench, value):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.text_input, "SIGMA / eV (optional)").set_value(value)
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    assert any("Cannot prepare inputs" in error.value for error in app.error)
    assert not any(button.label == "Submit calculation" for button in app.button)
    worker.assert_not_called()


def test_cluster_editor_saves_resources_used_by_prepared_job(workbench):
    app, worker, _ = workbench
    widget(app.number_input, "MPI tasks").set_value(16)
    widget(app.number_input, "Nodes").set_value(2)
    widget(app.text_area, "Extra Slurm options").set_value(
        "--mem=32G\n#SBATCH --qos normal\n--ntasks-per-node=8\n--cpus-per-task=2")
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception
    path = Path(widget(app.text_input, "Configuration file").value)
    saved = ClusterConfig.load(path)
    assert saved.tasks == 16 and saved.nodes == 2
    assert saved.extra_sbatch == ["--mem=32G", "#SBATCH --qos normal", "--ntasks-per-node=8", "--cpus-per-task=2"]
    assert widget(app.number_input, "Nodes").value == 2
    assert "--mem=32G" in widget(app.text_area, "Extra Slurm options").value
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    frozen = ClusterConfig.load(run_dir / "config.json")
    assert frozen == saved
    script = workflow._script(frozen, state["stages"][0], None)
    assert "#SBATCH --nodes=2" in script and "#SBATCH --ntasks=16" in script
    assert "#SBATCH --qos=normal" in script and "#SBATCH --ntasks-per-node=8" in script
    assert script.index("#SBATCH --mem=32G") < script.index("set -e")
    worker.assert_not_called()


def test_cluster_editor_rejects_srun_option_without_replacing_saved_config(workbench):
    app, worker, _ = workbench
    path = Path(widget(app.text_input, "Configuration file").value)
    original = path.read_bytes()
    widget(app.text_area, "Extra Slurm options").set_value("--cpu-bind=cores")
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception
    assert any("with srun in the VASP command" in error.value for error in app.error)
    assert path.read_bytes() == original
    worker.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_completed_result_is_visible_without_resubmission(workbench, accepted):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state["status"] = "succeeded" if accepted else "needs_attention"
    stage = state["stages"][0]
    stage.update(status=state["status"], job_id="12345", result={
        "success": accepted, "reason": "Electronic convergence was not reached." if not accepted else "Accepted test fixture",
        "final_energy_ev": -10.123456, "final_max_force_ev_angstrom": 0.002,
    })
    (run_dir / "run.json").write_text(json.dumps(state))
    output = run_dir / stage["folder"] / "outputs"
    output.mkdir()
    # Rejected output must not appear as an accepted result.
    (output / "final_structure.cif").write_bytes(silicon_bytes())
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    fig.savefig(output / "result.png")
    plt.close(fig)
    app.run()
    assert not app.exception
    assert not any(button.label == "Submit calculation" for button in app.button)
    if accepted:
        assert widget(app.metric, "Final total energy (eV)").value == "-10.123456"
        assert widget(app.download_button, "Download structure (.cif)").proto.url
        assert any("Si" in item.value and "atoms" in item.value for item in app.markdown)
    else:
        assert any("Electronic convergence" in item.value for item in app.warning)
        assert not any(button.label == "Download structure (.cif)" for button in app.download_button)
    captions = [image.caption for block in app.get("image") for image in block.proto.imgs]
    assert bool(captions) is accepted
    widget(app.button, "Prepare download").click().run()
    assert not app.exception
    assert (run_dir / "results.zip").is_file()
    assert widget(app.download_button, "Download results (.zip)").proto.url
    worker.assert_not_called()


def test_installed_examples_are_readable_crystal_inputs():
    examples = files("vasp_slurm_agent").joinpath("examples")
    expected = {"Si", "C", "Ge", "Al", "Cu", "MgO", "NaCl", "SiC", "SrTiO3",
                "VSe2", "In2Se3", "V2Te2O", "CrSBr", "CrOCl", "CrCl3", "CrPS4", "GaFe2O4", "CrOCl-MoS2"}
    structures = {sample.name[:-4]: Structure.from_str(sample.read_text(), fmt="cif")
                  for sample in examples.iterdir() if sample.name.endswith(".cif")}
    assert structures.keys() == expected
    for structure in structures.values():
        assert structure.is_ordered and len(structure) > 0 and structure.volume > 0


def test_example_resources_match_repository_and_preserve_geometry():
    from vasp_slurm_agent.structures import preview_structure
    repository = Path(__file__).resolve().parents[1] / "examples"
    packaged = files("vasp_slurm_agent").joinpath("examples")
    records = json.loads(packaged.joinpath("sources.json").read_text())
    assert len(records["structures"]) == 9
    public_names = {path.name for path in repository.iterdir() if path.suffix in {".cif", ".vasp", ".json"}}
    package_names = {path.name for path in packaged.iterdir() if path.name.endswith((".cif", ".vasp", ".json"))}
    assert public_names == package_names
    for name in public_names:
        assert (repository / name).read_bytes() == packaged.joinpath(name).read_bytes()
    for entry in records["structures"]:
        poscar = preview_structure(repository / (entry["name"] + ".vasp"), [])["input"]
        cif = preview_structure(repository / (entry["name"] + ".cif"), [])["input"]
        assert poscar["number_of_sites"] == cif["number_of_sites"] == entry["atoms"]
        assert [site["element"] for site in poscar["sites"]] == [site["element"] for site in cif["sites"]]
        left, right = np.asarray(poscar["lattice_angstrom"]), np.asarray(cif["lattice_angstrom"])
        np.testing.assert_allclose(left @ left.T, right @ right.T, rtol=1e-9, atol=1e-8)
        delta = np.asarray([site["fractional_coordinates"] for site in poscar["sites"]]) - np.asarray([site["fractional_coordinates"] for site in cif["sites"]])
        np.testing.assert_allclose(delta - np.rint(delta), 0, atol=1e-9)
        for name, digest in entry["files"].items():
            assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == digest
        assert entry["source"].startswith(("cqr/", "cqr2/")) and len(entry["source_sha256"]) == 64
    assert "/Users/" not in json.dumps(records) and "local_path" not in json.dumps(records)


@pytest.mark.parametrize("name", ["Al.cif", "MgO.cif", "SiC.cif"])
def test_packaged_example_prepares_without_upload(workbench, name):
    app, worker, runs_root = workbench
    widget(app.radio, "Structure source").set_value("Example").run()
    widget(app.selectbox, "Example").select(name).run()
    assert not app.exception
    assert widget(app.download_button, "Download input").proto.url
    assert not runs_root.exists()
    worker.assert_not_called()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    assert state["formula"] == name.removesuffix(".cif")
    expected = files("vasp_slurm_agent").joinpath("examples", name).read_bytes()
    assert (run_dir / "source/input/structure.cif").read_bytes() == expected
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_switching_example_invalidates_existing_plan(workbench, monkeypatch):
    app, worker, _ = workbench
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(task_response()))
    widget(app.radio, "Structure source").set_value("Example").run()
    widget(app.selectbox, "Example").select("Si.cif").run()
    widget(app.radio, "Mode").set_value("Agent").run()
    widget(app.selectbox, "Provider").select("codex").run()
    widget(app.text_area, "Goal").set_value("Relax this structure").run()
    widget(app.button, "Plan").click().run()
    assert not app.exception
    assert "agent_proposal" in app.session_state
    widget(app.selectbox, "Example").select("MgO.cif").run()
    assert not app.exception
    assert "agent_proposal" not in app.session_state
    assert not any(button.label == "Prepare inputs" for button in app.button)
    worker.assert_not_called()


def test_complex_examples_use_poscar_without_duplicate_choices(workbench):
    app, worker, _ = workbench
    widget(app.radio, "Structure source").set_value("Example").run()
    examples = widget(app.selectbox, "Example")
    assert len(examples.options) == 19
    examples.select("CrOCl-MoS2.vasp").run()
    assert not app.exception
    assert widget(app.metric, "Atoms").value == "114"
    assert app.session_state["structure_original"]["number_of_sites"] == 114
    worker.assert_not_called()


def test_format_conversion_needs_no_model_or_cluster(workbench, monkeypatch):
    app, worker, runs_root = workbench
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: pytest.fail("Conversion must not call a model"))
    widget(app.text_input, "Configuration file").set_value(str(runs_root.parent / "missing.json")).run()
    widget(app.radio, "Structure source").set_value("Example").run()
    widget(app.selectbox, "Example").select("Si.cif").run()
    widget(app.selectbox, "Convert to").select("both").run()
    widget(app.button, "Convert").click().run()
    assert not app.exception
    assert widget(app.download_button, "Download CIF").proto.url
    assert widget(app.download_button, "Download POSCAR").proto.url
    assert "structure_active" not in app.session_state
    widget(app.button, "Use structure").click().run()
    assert not app.exception
    assert app.session_state["structure_active"]["output"]["number_of_sites"] == 2
    # Inputs can be prepared without a cluster; nothing is created until the button is used.
    assert not widget(app.button, "Prepare inputs").disabled
    assert any("without a cluster" in item.value for item in app.caption)
    assert not runs_root.exists()
    worker.assert_not_called()


def test_structure_revisions_use_original_and_clear_calculation_state(workbench, monkeypatch):
    from vasp_slurm_agent import structure_agent
    app, worker, _ = workbench
    calls = []
    def draft(goal, source, settings, history=None):
        data = Path(source).read_bytes()
        calls.append((data, history))
        repeats = 2 if len(calls) == 1 else 3
        return {"status": "ready", "summary": "Make a supercell.",
                "operations": [{"type": "supercell", "matrix": [repeats, 1, 1]}],
                "output_format": "both", "source_sha256": hashlib.sha256(data).hexdigest(),
                "model_usage": MODEL_USAGE,
                "questions": [], "notes": [], "dialogue": [*(history or []), {"role": "assistant", "content": "Supercell planned."}]}
    monkeypatch.setattr(structure_agent, "draft_structure", draft)
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.text_area, "Site moments (JSON)").set_value("[1,-1]").run()
    widget(app.text_input, "Spin axis").set_value("1 0 0").run()
    widget(app.text_area, "Structure goal").set_value("Make a 2 by 1 by 1 supercell").run()
    widget(app.button, "Plan structure").click().run()
    assert "input 1,234" in usage_caption(app) and "reasoning unknown" in usage_caption(app)
    for _ in range(3):
        app.run()
    assert len(calls) == 1
    widget(app.button, "Preview structure").click().run()
    assert not app.exception
    assert app.session_state["structure_preview"]["output"]["number_of_sites"] == 4
    assert "structure_active" not in app.session_state
    app.session_state["agent_proposal"] = {"old": "plan"}
    app.session_state["agent_history"] = [{"role": "user", "content": "Old goal"}]
    widget(app.button, "Use structure").click().run()
    assert not app.exception
    assert "agent_proposal" not in app.session_state and "agent_history" not in app.session_state
    assert widget(app.text_area, "Site moments (JSON)").value == "null"
    assert widget(app.text_input, "Spin axis").value == "0 0 1"
    widget(app.text_input, "Follow-up").set_value("Use 3 by 1 by 1 instead").run()
    assert widget(app.button, "Preview structure").disabled
    widget(app.button, "Plan structure").click().run()
    widget(app.button, "Preview structure").click().run()
    assert app.session_state["structure_preview"]["output"]["number_of_sites"] == 6
    widget(app.button, "Use structure").click().run()
    assert not app.exception
    assert all(data == silicon_bytes() for data, _ in calls)
    assert calls[-1][1][-1]["content"] == "Use 3 by 1 by 1 instead"
    active = app.session_state["structure_active"]
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    assert (run_dir / "source/input/structure.vasp").read_bytes() == active["files"]["POSCAR"]
    assert (run_dir / "structure_edit/source.cif").read_bytes() == silicon_bytes()
    record = json.loads((run_dir / "structure_edit/structure.json").read_text())
    assert record["output"]["number_of_sites"] == 6
    assert (run_dir / "structure_edit/POSCAR").read_bytes() == active["files"]["POSCAR"]
    widget(app.button, "Use original").click().run()
    assert not app.exception
    assert "structure_active" not in app.session_state
    assert widget(app.metric, "Atoms").value == "2"
    assert widget(app.text_area, "Site moments (JSON)").value == "null"
    worker.assert_not_called()


def test_calculation_usage_is_saved_without_model_calls_on_refresh(workbench, monkeypatch):
    app, worker, _ = workbench
    reply = task_response()
    reply.update(model_usage=MODEL_USAGE, source_sha256=hashlib.sha256(silicon_bytes()).hexdigest(),
                 dialogue=[{"role": "assistant", "content": "Relax the structure."}])
    planner = Mock(return_value=reply)
    monkeypatch.setattr(task_agent, "draft_task", planner)
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.radio, "Mode").set_value("Agent").run()
    widget(app.text_area, "Goal").set_value("Relax this structure.").run()
    widget(app.button, "Plan").click().run()
    assert not app.exception
    text = usage_caption(app)
    assert "input 1,234" in text and "cached 1,024" in text and "1.2 s" in text
    for _ in range(3):
        app.run()
    assert planner.call_count == 1
    assert app.session_state["agent_proposal"]["plan"]["model_usage"] == MODEL_USAGE
    worker.assert_not_called()


@pytest.mark.parametrize("kind", ["calculation", "structure", "explanation"])
def test_full_history_reaches_model_after_eight_rounds(workbench, monkeypatch, kind):
    history = []
    for index in range(8):
        history.extend([{"role": "user", "content": f"Keep constraint {index}."},
                        {"role": "assistant", "content": f"Constraint {index} retained."}])
    app, worker, _ = workbench
    if kind == "explanation":
        app, worker, run_dir, context, planner, _ = explanation_run(workbench, monkeypatch)
        (run_dir / "explanations.json").write_text(json.dumps({"context_sha256": context["context_sha256"],
                                                               "history": history, "exchanges": []}))
        app.run()
        widget(app.button, "Explain results").click().run()
        assert planner.call_args.kwargs["history"] == history
    elif kind == "calculation":
        def draft(*args, history=None, **kwargs):
            result = task_response()
            result.update(source_sha256=hashlib.sha256(silicon_bytes()).hexdigest(),
                          dialogue=history + [{"role": "assistant", "content": "Plan updated."}])
            return result
        planner = Mock(side_effect=draft)
        monkeypatch.setattr(task_agent, "draft_task", planner)
        open_agent(app)
        app.session_state["agent_history"] = history
        app.run()
        widget(app.text_input, "Change the plan").set_value("SCF only.").run()
        widget(app.button, "Plan").click().run()
        assert planner.call_args.kwargs["history"][:16] == history
        assert app.session_state["agent_history"][:16] == history
        assert len(app.session_state["agent_history"]) == 18
    else:
        from vasp_slurm_agent import structure_agent
        def draft(*args, history=None, **kwargs):
            return {"status": "needs_input", "summary": "Choose a site.", "questions": ["Which site?"],
                    "notes": [], "operations": [], "output_format": "both",
                    "dialogue": history + [{"role": "assistant", "content": "Which site?"}]}
        planner = Mock(side_effect=draft)
        monkeypatch.setattr(structure_agent, "draft_structure", planner)
        app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
        widget(app.text_area, "Structure goal").set_value("Replace one silicon atom.").run()
        app.session_state["structure_history"] = history
        app.run()
        widget(app.text_input, "Follow-up").set_value("Use Ge.").run()
        widget(app.button, "Plan structure").click().run()
        assert planner.call_args.kwargs["history"][:16] == history
        assert app.session_state["structure_history"][:16] == history
        assert len(app.session_state["structure_history"]) == 18
    assert not app.exception
    worker.assert_not_called()


def test_cluster_control_socket_is_saved(workbench):
    app, worker, runs_root = workbench
    socket = str(runs_root.parent / "master.sock")
    widget(app.text_input, "SSH control socket (optional)").set_value(socket)
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception
    assert ClusterConfig.load(runs_root.parent / "cluster.json").ssh_control_path == socket
    worker.assert_not_called()


def test_structure_editor_indices_follow_cif_rows(workbench):
    app, _, _ = workbench
    structure = Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["O", "Mg"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    text = str(CifWriter(structure, symprec=None, refine_struct=False))
    app.file_uploader[0].upload("ordered.cif", text.encode()).run()
    assert not app.exception
    sites = app.session_state["structure_original"]["sites"]
    assert [(site["index"], site["element"]) for site in sites] == [(0, "O"), (1, "Mg")]
    table = next(item.value for item in app.dataframe if {"index", "element"}.issubset(item.value.columns))
    assert table["index"].tolist() == [0, 1]
    assert table["element"].tolist() == ["O", "Mg"]
    widget(app.button, "Convert").click().run()
    widget(app.button, "Use structure").click().run()
    active = app.session_state["structure_active"]
    assert [site["element"] for site in active["output"]["sites"]] == ["O", "Mg"]


def test_paused_job_can_reconnect_from_the_result_panel(workbench):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True)
    (run_dir / "run.json").write_text(json.dumps(state))
    app.run()
    worker.assert_not_called()
    widget(app.button, "Reconnect").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    restored = workflow.read_state(run_dir)
    assert restored["stages"][0]["job_id"] == "12345"
    assert restored["status"] not in workflow.TERMINAL


@pytest.mark.parametrize("task", ["bands", "dos"])
@pytest.mark.parametrize("scf_fermi", [None, 5.25])
def test_spectral_results_use_scf_reference_not_ground_state_metrics(workbench, task, scf_fermi):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select(task)
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["task"] == task
    state.update(status="succeeded", current_stage=1)
    state["stages"][0].update(status="succeeded", job_id="12345", result={
        "success": True, "final_energy_ev": -10.0, "final_max_force_ev_angstrom": 0.002,
        "fermi_energy_ev": scf_fermi,
    })
    state["stages"][1].update(status="succeeded", job_id="12346", result={
        "success": True, "final_energy_ev": 123.0, "final_max_force_ev_angstrom": 99.0,
        "fermi_energy_ev": 8.88,
    })
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    app.run()
    assert not app.exception
    assert [item.value for item in app.metric if item.label == "Final total energy (eV)"] == ["-10.000000"]
    assert [item.value for item in app.metric if item.label == "Maximum atomic force (eV/Å)"] == ["0.002000"]
    references = [item.value for item in app.metric if item.label == "SCF Fermi energy (eV)"]
    assert references == (["5.250000"] if task == "bands" and scf_fermi is not None else [])
    if task == "dos":
        assert widget(app.metric, "DOS Fermi energy (eV)").value == "8.880000"
    assert any("Fixed-charge spectrum" in item.value for item in app.caption)
    assert state_path.read_bytes() == original
    worker.assert_not_called()


@pytest.mark.parametrize("scheduler_state", [None, "RUNNING", "COMPLETED", "CANCELLED", "FAILED"])
def test_paused_job_can_request_cancel_only_before_confirmed_scheduler_end(workbench, monkeypatch, scheduler_state):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True, scheduler_state=scheduler_state)
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    cancellation = Mock(return_value=state)
    monkeypatch.setattr(workflow, "cancel", cancellation)
    app.run()
    assert not app.exception
    if scheduler_state in {None, "RUNNING"}:
        widget(app.button, "Cancel calculation").click().run()
        assert not app.exception
        cancellation.assert_called_once_with(run_dir)
    else:
        assert not any(button.label == "Cancel calculation" for button in app.button)
        cancellation.assert_not_called()
    assert state_path.read_bytes() == original
    worker.assert_not_called()


def open_agent(app):
    widget(app.radio, "Mode").set_value("Agent").run()
    widget(app.selectbox, "Provider").select("codex").run()
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.text_area, "Goal").set_value("Relax, then SCF.").run()


def test_agent_proposal_requires_review_and_keeps_runtime_plan(workbench, monkeypatch):
    app, worker, runs_root = workbench
    provider = Mock(return_value=json.dumps(task_response()))
    monkeypatch.setattr(agent, "_request_structured", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not app.exception
    assert app.session_state["agent_proposal"]["plan"]["status"] == "ready"
    assert not runs_root.exists()
    worker.assert_not_called()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    proposal = json.loads((root / "proposal.json").read_text())
    frozen = json.loads((root / "plan.json").read_text())
    assert proposal["status"] == "ready" and frozen["stages"]
    assert proposal["scientific_report"]["evidence"]
    assert "status" not in frozen
    assert state["plan_sha256"] == hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    workflow._check_config(root, state)
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    worker.assert_not_called()
    widget(app.button, "Submit calculation").click().run()
    worker.assert_called_once_with(root)
    assert provider.call_count == 1


def test_agent_questions_block_preparation(workbench, monkeypatch):
    app, worker, runs_root = workbench
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(task_response(["scf"], missing_moments=True)))
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not app.exception
    proposal = app.session_state["agent_proposal"]["plan"]
    assert proposal["status"] == "needs_input"
    assert any("moment" in message.value for message in app.info)
    assert widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()


def test_agent_revision_failure_preserves_feedback_and_blocks_stale_prepare(workbench, monkeypatch):
    app, worker, runs_root = workbench
    provider = Mock(side_effect=[json.dumps(task_response()), agent.AgentError("Provider unavailable"),
                                 json.dumps(task_response(["scf"]))])
    monkeypatch.setattr(agent, "_request_structured", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    widget(app.text_input, "Change the plan").set_value("SCF only. Do not relax.").run()
    assert widget(app.button, "Prepare inputs").disabled
    widget(app.button, "Plan").click().run()
    assert not app.exception
    assert any("Provider unavailable" in message.value for message in app.error)
    assert not any(button.label == "Prepare inputs" for button in app.button)
    assert not runs_root.exists()
    feedback = {"role": "user", "content": "SCF only. Do not relax."}
    assert app.session_state["agent_history"][-1] == feedback
    app.run()
    assert widget(app.text_input, "Change the plan").value == feedback["content"]
    widget(app.button, "Plan").click().run()
    assert not app.exception
    retry_payload = json.loads(provider.call_args.args[0])
    assert retry_payload["history"].count(feedback) == 1
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    proposal = json.loads((root / "proposal.json").read_text())
    assert proposal["goal"] == "Relax, then SCF."
    assert proposal["dialogue"][1] == feedback
    assert len(proposal["dialogue"]) == 3
    assert proposal["tasks"] == ["scf"]
    assert workflow.read_state(root)["tasks"] == ["scf"]
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_agent_clarification_is_saved_with_accepted_plan(workbench, monkeypatch):
    app, worker, _ = workbench
    reply = task_response(["scf"], missing_moments=True)
    resolved = task_response(["scf"], missing_moments=True)
    resolved["stages"][0]["parameters"]["magmom"] = [1, -1]
    provider = Mock(side_effect=[json.dumps(reply), json.dumps(resolved)])
    monkeypatch.setattr(agent, "_request_structured", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert widget(app.button, "Prepare inputs").disabled
    widget(app.text_input, "Change the plan").set_value("Use site moments [1, -1].").run()
    widget(app.button, "Plan").click().run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    proposal = json.loads((root / "proposal.json").read_text())
    assert proposal["parameters"]["magmom"] == [1, -1]
    assert json.loads(proposal["dialogue"][0]["content"])["status"] == "needs_input"
    assert proposal["dialogue"][1]["content"] == "Use site moments [1, -1]."
    worker.assert_not_called()


@pytest.mark.parametrize("changed", ["goal", "structure"])
def test_agent_cannot_prepare_stale_proposal(workbench, monkeypatch, changed):
    app, worker, runs_root = workbench
    provider = Mock(return_value=json.dumps(task_response()))
    monkeypatch.setattr(agent, "_request_structured", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not widget(app.button, "Prepare inputs").disabled
    if changed == "goal":
        widget(app.text_area, "Goal").set_value("SCF only.").run()
    else:
        payload = files("vasp_slurm_agent").joinpath("examples", "Al.cif").read_bytes()
        app.file_uploader[0].upload("Al.cif", payload).run()
    assert not app.exception
    assert not any(button.label == "Prepare inputs" for button in app.button)
    assert provider.call_count == 1
    assert not runs_root.exists()
    worker.assert_not_called()


def test_agent_preparation_rechecks_source_hash(workbench, monkeypatch):
    app, worker, runs_root = workbench
    proposal = {"status": "ready", "summary": "SCF", "tasks": ["scf"], "parameters": {},
                "source_sha256": "0" * 64, "questions": []}
    monkeypatch.setattr(task_agent, "draft_task", Mock(return_value=proposal))
    open_agent(app)
    widget(app.button, "Plan").click().run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    assert any("structure changed" in message.value for message in app.error)
    assert not runs_root.exists()
    worker.assert_not_called()


@pytest.mark.parametrize("method", ["hybrid", "u", "soc"])
def test_manual_method_controls_reach_frozen_inputs(workbench, method):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select("scf")
    if method == "hybrid":
        widget(app.selectbox, "Functional").select("HSE06")
    elif method == "u":
        widget(app.selectbox, "Spin").select("collinear")
        widget(app.text_area, "Site moments (JSON)").set_value("[1, -1]")
        widget(app.text_area, "Hubbard U (JSON)").set_value('{"Si":{"l":1,"u":2,"j":0}}')
    else:
        widget(app.text_input, "SOC / noncollinear command").set_value("srun vasp_ncl")
        widget(app.button, "Save cluster settings").click().run()
        widget(app.selectbox, "Calculation").select("scf")
        widget(app.selectbox, "Spin").select("noncollinear")
        widget(app.checkbox, "SOC").check()
        widget(app.text_area, "Site moments (JSON)").set_value("[[0,0,1],[0,0,-1]]")
        widget(app.text_input, "Spin axis").set_value("1 0 0")
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    metadata = state["stages"][0]["metadata"]
    if method == "hybrid":
        assert metadata["method"]["functional"] == "HSE06"
    elif method == "u":
        assert metadata["method"]["hubbard_u"] == {"Si": {"l": 1, "u": 2., "j": 0.}}
        assert metadata["method"]["magmom"] == [1., -1.]
    else:
        assert metadata["requires_ncl"] and metadata["method"]["soc"]
        assert metadata["method"]["saxis"] == [1., 0., 0.]
    assert state["status"] == "planned"
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


@pytest.mark.parametrize("task", ["bands", "dos"])
def test_hybrid_spectrum_uses_own_fermi_reference(workbench, task):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select(task)
    widget(app.selectbox, "Functional").select("HSE06")
    widget(app.button, "Prepare inputs").click().run()
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    spectrum = state["stages"][0]
    spectrum.update(status="succeeded", job_id="12346", result={"success": True, "energy_reference_ev": 6.75,
                    "fermi_energy_ev": 8.88, "energy_reference_source": "hybrid_scf_mesh"})
    state["stages"].insert(0, {"name": "scf", "folder": "00_scf", "status": "succeeded", "job_id": "12345",
                              "result": {"success": True, "fermi_energy_ev": 1.25, "final_energy_ev": -10.}})
    state.update(status="succeeded", current_stage=1)
    (root / "run.json").write_text(json.dumps(state))
    app.run()
    assert not app.exception
    label = "DOS Fermi energy (eV)" if task == "dos" else "Fermi energy (eV)"
    assert [item.value for item in app.metric if item.label == label] == ["6.750000"]
    assert not any(item.label == "SCF Fermi energy (eV)" for item in app.metric)
    worker.assert_not_called()


def explanation_run(workbench, monkeypatch):
    from vasp_slurm_agent import explanation

    app, worker, _ = workbench
    context = {"context_sha256": "a" * 64, "goal": "Relax silicon, then inspect its structure.",
               "status": "succeeded", "outcome": "Relaxation completed.",
               "facts": {"energy": {"label": "Final energy", "value": -10.0, "unit": "eV", "source": "01_relax/outputs/result.json"}},
               "limits": [], "artifacts": [], "final_plan": {"tasks": ["relax"]}, "dialogue": []}
    monkeypatch.setattr(explanation, "load_run_context", lambda _: deepcopy(context))

    def answer(*args, **kwargs):
        return {"answer": "The requested relaxation completed.", "evidence": ["energy"],
                "limits": ["Material-specific accuracy is not established."], "next_steps": ["Download the structure."],
                "model_usage": MODEL_USAGE,
                "context_sha256": context["context_sha256"], "provenance": {"provider": "codex", "model": "test"}}

    explain = Mock(side_effect=answer)
    monkeypatch.setattr(explanation, "explain_run", explain)
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    return app, worker, root, context, explain, answer


def test_explanation_is_explicit_grounded_and_persists_across_sessions(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    original = (root / "run.json").read_bytes()
    assert any(context["goal"] == item.value for item in app.markdown)
    assert any("Relaxation completed." == item.value for item in app.markdown)
    explain.assert_not_called()
    app.run()
    explain.assert_not_called()
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert explain.call_args.kwargs == {"question": "Explain the results.", "history": []}
    assert any("energy · Final energy: -10.0 eV" in item.value for item in app.markdown)
    record = json.loads((root / "explanations.json").read_text())
    assert record["context_sha256"] == context["context_sha256"]
    assert len(record["exchanges"]) == 1
    assert "output 56" in usage_caption(app)
    assert record["exchanges"][0]["model_usage"] == MODEL_USAGE
    for _ in range(3):
        app.run()
    assert explain.call_count == 1
    widget(app.text_input, "Ask about this run").set_value("Where is the structure?").run()
    widget(app.button, "Ask").click().run()
    assert not app.exception
    assert explain.call_args.kwargs["history"] == record["history"]
    assert explain.call_args.kwargs["question"] == "Where is the structure?"
    assert len(json.loads((root / "explanations.json").read_text())["exchanges"]) == 2
    reopened = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(reopened.text_input, "Or enter a run folder").set_value(str(root)).run()
    widget(reopened.button, "Open run").click().run()
    assert not reopened.exception
    assert any("The requested relaxation completed." == item.value for item in reopened.markdown)
    assert explain.call_count == 2
    assert (root / "run.json").read_bytes() == original
    worker.assert_not_called()


def test_failed_model_usage_survives_refresh_without_changing_accepted_records(workbench, monkeypatch):
    app, worker, root, context, explain, answer = explanation_run(workbench, monkeypatch)
    widget(app.button, "Explain results").click().run()
    accepted = (root / "explanations.json").read_bytes()
    frozen_plan = (root / "plan.json").read_bytes()
    failed_usage = {**MODEL_USAGE, "input_tokens": 4321, "output_tokens": 7}
    explain.side_effect = agent.AgentError("Model returned an incomplete answer.", model_usage=failed_usage)
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert "input 4,321" in usage_caption(app)
    assert app.session_state[f"explanation_error_usage_{root}_{context['context_sha256']}"] == failed_usage
    for _ in range(3):
        app.run()
        assert "input 4,321" in usage_caption(app)
    assert explain.call_count == 2
    assert (root / "explanations.json").read_bytes() == accepted
    assert (root / "plan.json").read_bytes() == frozen_plan
    assert any("The requested relaxation completed." == item.value for item in app.markdown)
    explain.side_effect = answer
    widget(app.button, "Explain results").click().run()
    assert not any(item.value == "Last failed call" for item in app.caption)
    assert explain.call_count == 3
    worker.assert_not_called()


def test_changed_result_context_hides_old_answers_and_resets_model_history(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    widget(app.button, "Explain results").click().run()
    old_record = (root / "explanations.json").read_bytes()
    context.update(context_sha256="b" * 64, status="needs_attention", outcome="Output checks failed.")
    app.run()
    assert not app.exception
    assert any("Results changed" in item.value for item in app.info)
    assert not any("The requested relaxation completed." == item.value for item in app.markdown)
    assert (root / "explanations.json").read_bytes() == old_record
    assert explain.call_count == 1
    widget(app.text_input, "Ask about this run").set_value("Why did it fail?").run()
    widget(app.button, "Ask").click().run()
    assert explain.call_args.kwargs["history"] == []
    new_record = json.loads((root / "explanations.json").read_text())
    assert new_record["context_sha256"] == "b" * 64
    assert len(new_record["exchanges"]) == 1
    worker.assert_not_called()


def test_explanation_provider_failure_can_retry_without_submission(workbench, monkeypatch):
    app, worker, root, _, explain, answer = explanation_run(workbench, monkeypatch)
    explain.side_effect = agent.AgentError("Provider unavailable")
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert any("Provider unavailable" in item.value for item in app.error)
    assert not (root / "explanations.json").exists()
    app.run()
    assert explain.call_count == 1
    explain.side_effect = answer
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert (root / "explanations.json").is_file()
    assert explain.call_count == 2
    worker.assert_not_called()


def test_current_evidence_overrules_cached_success_in_results_panel(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    state = workflow.read_state(root)
    state["status"] = "succeeded"
    stage = state["stages"][0]
    stage.update(status="succeeded", job_id="12345", scheduler_state="COMPLETED",
                 result={"success": True, "final_energy_ev": -10.0})
    (root / "run.json").write_text(json.dumps(state))
    saved = (root / "run.json").read_bytes()
    output = root / stage["folder"] / "outputs"
    output.mkdir()
    (output / "final_structure.cif").write_bytes(silicon_bytes())
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    fig.savefig(output / "result.png")
    plt.close(fig)
    context.update(status="needs_attention", outcome="No accepted results.")
    context["facts"].update({"stage_1.accepted": {"value": False},
                             "stage_1.availability": {"value": "Current output checks rejected this result."}})
    app.run()
    assert not app.exception
    assert any("Needs attention" in item.value for item in app.markdown)
    assert any("Current output checks rejected" in item.value for item in app.warning)
    assert not any("Complete. Results are ready" in item.value for item in app.info)
    assert not any(item.label == "Final total energy (eV)" for item in app.metric)
    assert not any(item.label == "Download structure (.cif)" for item in app.download_button)
    assert not any("relax · result.png" in block.captions for block in app.get("image"))
    assert (root / "run.json").read_bytes() == saved
    explain.assert_not_called()
    worker.assert_not_called()


def test_explanation_does_not_save_answer_if_results_change_during_request(workbench, monkeypatch):
    app, worker, root, context, explain, answer = explanation_run(workbench, monkeypatch)

    def changed(*args, **kwargs):
        response = answer(*args, **kwargs)
        context["context_sha256"] = "b" * 64
        return response

    explain.side_effect = changed
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert any("Results changed" in item.value for item in app.error)
    assert not (root / "explanations.json").exists()
    assert not any("The requested relaxation completed." == item.value for item in app.markdown)
    worker.assert_not_called()


def test_repair_actions_require_explicit_planning_and_submission(workbench, monkeypatch):
    from vasp_slurm_agent import recovery

    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    parent = Path(app.session_state["active_run"])
    state = workflow.read_state(parent)
    state.update(status="needs_attention", last_error=None)
    state["stages"][0].update(status="needs_attention", scheduler_state="COMPLETED", job_id="123",
                               result={"success": False, "reason": "Electronic convergence was not reached."})
    (parent / "run.json").write_text(json.dumps(state))
    before = (parent / "run.json").read_bytes()
    draft = Mock(return_value={"status": "ready", "diagnosis": "Increase the iteration limit.",
        "model_usage": MODEL_USAGE,
        "questions": [], "attempt": 1, "max_attempts": 2,
        "changes": [{"scope": "parameters", "key": "nelm", "before": 120, "after": 240}]})
    monkeypatch.setattr(recovery, "draft_repair", draft)

    def prepare(original, destination, proposal):
        assert original == parent
        return workflow.prepare_run(parent / "source/input/structure.cif", destination,
                                    ClusterConfig.load(parent / "config.json"), "scf", {"nelm": 240})

    preparation = Mock(side_effect=prepare)
    monkeypatch.setattr(recovery, "prepare_repair", preparation)
    app.run()
    draft.assert_not_called()
    widget(app.button, "Plan repair").click().run()
    assert not app.exception and draft.call_count == 1
    assert "cached 1,024" in usage_caption(app)
    worker.assert_not_called()
    for _ in range(3):
        app.run()
    assert draft.call_count == 1
    widget(app.button, "Prepare repair").click().run()
    assert not app.exception and preparation.call_count == 1
    child = Path(app.session_state["active_run"])
    assert child != parent
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    widget(app.button, "Submit calculation").click().run()
    worker.assert_called_once_with(child)
    assert (parent / "run.json").read_bytes() == before
