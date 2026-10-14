"""Switch solvers without replacing their settings or submitting jobs."""

from importlib.resources import files
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import qe_setup, qe_workflow, settings, workflow
from vasp_slurm_agent.config import ClusterConfig


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def configuration(engine):
    shared = dict(host=f"{engine}.example.invalid", user=engine,
                  remote_root=f"/scratch/{engine}", partition=engine,
                  setup_commands=[f"module load {engine}"])
    if engine == "qe":
        return ClusterConfig(**shared, engine="qe", qe_bin_dir="/opt/qe/bin",
                             qe_pseudo_root="/scratch/upf", qe_pseudopotentials={"Si": "Si.UPF"})
    return ClusterConfig(**shared, vasp_command="srun vasp_std", potcar_root="/licensed/pbe",
                         compute_backend="gpu", tasks=2, gpus_per_node=2, cpus_per_task=8)


@pytest.fixture
def open_app(tmp_path, monkeypatch):
    settings.save_settings({"auto_resume": False, "check_updates": False})
    monkeypatch.setenv("DFT_AGENT_RUNS", str(tmp_path / "runs"))
    remote = Mock(side_effect=AssertionError("Solver switching must not contact the cluster"))
    worker = Mock(side_effect=AssertionError("Preparing inputs must not submit jobs"))
    for module in (workflow, qe_workflow, qe_setup):
        monkeypatch.setattr(module, "SSHTransport", remote)
    monkeypatch.setattr(workflow, "start_worker", worker)

    def open_at(path):
        monkeypatch.setenv("DFT_AGENT_CONFIG", str(path))
        app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
        assert not app.exception
        return app

    yield open_at
    remote.assert_not_called()
    worker.assert_not_called()


def test_saving_qe_keeps_vasp_gpu_settings_and_switches_back(tmp_path, open_app):
    base = configuration("vasp").save(tmp_path / "cluster.json")
    original = base.read_bytes()
    app = open_app(base)
    widget(app.radio, "Engine").set_value("qe").run()
    assert not app.exception
    assert not any(field.label == "VASP command" for field in app.text_input)
    assert "module load vasp" not in widget(app.text_area, "Environment commands").value
    for label, value in {"SSH host": "qe.example.invalid", "SSH user": "qe",
                         "Run folder on cluster": "/scratch/qe", "Partition": "cpu",
                         "QE bin folder": "/opt/qe/bin", "UPF folder on cluster": "/scratch/upf"}.items():
        widget(app.text_input, label).set_value(value)
    widget(app.text_area, "Environment commands").set_value("module load qe")
    widget(app.text_area, "UPF mapping").set_value('{"Si":"Si.UPF"}')
    widget(app.button, "Save settings").click().run()
    assert not app.exception and not app.error
    qe_path = tmp_path / "cluster.qe.json"
    saved_qe = ClusterConfig.load(qe_path)
    assert saved_qe.engine == "qe" and saved_qe.compute_backend == "cpu"
    assert saved_qe.qe_pseudopotentials == {"Si": "Si.UPF"}
    assert saved_qe.setup_commands == ["module load qe"]
    assert base.read_bytes() == original
    qe_bytes = qe_path.read_bytes()

    widget(app.radio, "Engine").set_value("vasp").run()
    assert not app.exception
    assert widget(app.selectbox, "Compute").value == "gpu"
    assert widget(app.number_input, "GPUs per node").value == 2
    assert widget(app.text_area, "Environment setup commands (one per line)").value == "module load vasp"
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception and not app.error
    assert base.read_bytes() == original and qe_path.read_bytes() == qe_bytes
    widget(app.radio, "Engine").set_value("qe").run()
    assert widget(app.text_input, "QE bin folder").value == "/opt/qe/bin"
    assert json.loads(widget(app.text_area, "UPF mapping").value) == {"Si": "Si.UPF"}


def test_changing_config_file_loads_its_solver_without_stale_fields(tmp_path, open_app):
    first = configuration("qe").save(tmp_path / "first.json")
    second = configuration("vasp").save(tmp_path / "second.json")
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    app = open_app(first)
    assert widget(app.radio, "Engine").value == "qe"
    widget(app.text_input, "QE bin folder").set_value("/unsaved/old-bin")
    widget(app.text_input, "Configuration file").set_value(str(second)).run()
    assert not app.exception
    assert widget(app.radio, "Engine").value == "vasp"
    assert widget(app.text_input, "SSH host").value == "vasp.example.invalid"
    widget(app.text_input, "Configuration file").set_value(str(first)).run()
    assert not app.exception
    assert widget(app.radio, "Engine").value == "qe"
    assert widget(app.text_input, "QE bin folder").value == "/opt/qe/bin"
    assert first.read_bytes() == first_bytes and second.read_bytes() == second_bytes


def test_first_save_can_be_qe_then_add_vasp_without_replacing_it(tmp_path, open_app):
    base = tmp_path / "cluster.json"
    app = open_app(base)
    widget(app.radio, "Engine").set_value("qe").run()
    for label, value in {"SSH host": "qe.example.invalid", "SSH user": "qe",
                         "Run folder on cluster": "/scratch/qe", "Partition": "cpu",
                         "UPF folder on cluster": "/scratch/upf"}.items():
        widget(app.text_input, label).set_value(value)
    widget(app.button, "Save settings").click().run()
    assert not app.exception and not app.error
    assert ClusterConfig.load(base).engine == "qe"
    qe_bytes = base.read_bytes()
    widget(app.radio, "Engine").set_value("vasp").run()
    assert not app.exception
    assert widget(app.text_input, "SSH host").value == ""
    assert widget(app.text_area, "Environment setup commands (one per line)").value == ""
    for label, value in {"SSH host": "vasp.example.invalid", "SSH user": "vasp",
                         "Remote run folder (absolute path)": "/scratch/vasp",
                         "Remote POTCAR folder": "/licensed/pbe", "Slurm partition": "cpu"}.items():
        widget(app.text_input, label).set_value(value)
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception and not app.error
    assert ClusterConfig.load(tmp_path / "cluster.vasp.json").engine == "vasp"
    assert base.read_bytes() == qe_bytes


@pytest.mark.parametrize("initial", ["vasp", "qe"])
def test_switching_engine_prepares_the_selected_inputs_and_keeps_existing_runs(tmp_path, open_app, initial):
    other = "qe" if initial == "vasp" else "vasp"
    base = configuration(initial).save(tmp_path / "cluster.json")
    configuration(other).save(tmp_path / f"cluster.{other}.json")
    app = open_app(base)
    widget(app.radio, "Mode").set_value("Manual").run()
    app.file_uploader[0].upload("Si.cif", files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()).run()
    runs = []
    for engine in (initial, other):
        widget(app.radio, "Engine").set_value(engine).run()
        assert not app.exception
        assert any(field.label == "Calculations, in order" for field in app.multiselect) is (engine == "qe")
        assert any(field.label == "ENCUT / eV" for field in app.number_input) is (engine == "vasp")
        widget(app.button, "Prepare inputs").click().run()
        assert not app.exception and not app.error
        run = Path(app.session_state["active_run"])
        state = workflow.read_state(run)
        assert state.get("engine", "vasp") == engine
        assert state["status"] == "planned"
        assert all(stage["job_id"] is None for stage in state["stages"])
        assert ClusterConfig.load(run / "config.json").engine == engine
        inputs = run / state["stages"][0]["folder"] / "inputs"
        assert (inputs / ("pw.in" if engine == "qe" else "INCAR")).is_file()
        assert not (inputs / ("INCAR" if engine == "qe" else "pw.in")).exists()
        assert widget(app.button, "Submit calculation").disabled
        runs.append((run, (run / "config.json").read_bytes(), (run / "run.json").read_bytes()))
    assert runs[0][0] != runs[1][0]
    for run, config_bytes, state_bytes in runs:
        assert (run / "config.json").read_bytes() == config_bytes
        assert (run / "run.json").read_bytes() == state_bytes
