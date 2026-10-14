"""Cluster-free preparation, sample runs, detection, presets and preferences in the app."""

from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import cli, credentials, discovery, settings, workflow
from vasp_slurm_agent.config import ClusterConfig


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def start(tmp_path, monkeypatch):
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "cluster.json"))
    monkeypatch.setattr(workflow, "SSHTransport", lambda *args, **kwargs: pytest.fail("UI tests never contact SSH"))
    worker = Mock(return_value=4321)
    monkeypatch.setattr(workflow, "start_worker", worker)
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
    widget(app.text_input, "Run folder").set_value(str(tmp_path / "runs")).run()
    assert not app.exception
    return app, worker


@pytest.fixture
def bench(tmp_path, monkeypatch):
    app, worker = start(tmp_path, monkeypatch)
    return app, worker, tmp_path


def save_cluster(app, **values):
    fields = {"SSH host": "cluster.example.invalid", "SSH user": "researcher", "Remote run folder (absolute path)": "/scratch/test",
              "Remote POTCAR folder": "/licensed/pbe", "Slurm partition": "cpu", **values}
    for label, value in fields.items():
        widget(app.text_input, label).set_value(value)
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception


def test_prepare_without_cluster_then_save_and_submit(bench):
    app, worker, tmp_path = bench
    widget(app.radio, "Mode").set_value("Manual").run()
    app.file_uploader[0].upload("Si.cif", files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()).run()
    assert not widget(app.button, "Prepare inputs").disabled
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    assert not (run_dir / "config.json").exists()
    assert (run_dir / "01_relax" / "inputs" / "INCAR").is_file()
    assert any("No cluster is attached" in item.value for item in app.info)
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    save_cluster(app)
    assert (tmp_path / "cluster.json").is_file()
    assert widget(app.button, "Submit calculation").disabled
    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    widget(app.button, "Submit calculation").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    state = workflow.read_state(run_dir)
    assert ClusterConfig.load(run_dir / "config.json").host == "cluster.example.invalid"
    assert state["remote_root"].startswith("/scratch/test/")
    assert state["history"][-1]["event"] == "Cluster settings attached."
    assert any("Monitoring started" in item.value for item in app.success)


def test_sample_runs_load_from_the_runs_tab(bench):
    app, worker, tmp_path = bench
    assert any("No runs yet" in item.value for item in app.info)
    widget(app.button, "Load sample runs").click().run()
    assert not app.exception
    names = sorted(path.name for path in (tmp_path / "runs").iterdir())
    assert names == ["sample-fe-seed-comparison", "sample-mgo-cell-relax", "sample-si-pbe-chain", "sample-srtio3-pbe-chain"]
    assert app.session_state["active_run"].endswith("sample-srtio3-pbe-chain")
    assert any("Sample run" in item.value for item in app.info)
    assert any(metric.label == "Final total energy (eV)" for metric in app.metric)
    choices = widget(app.selectbox, "Select a run")
    assert len(choices.options) == 4
    widget(app.button, "Load sample runs").click().run()
    assert any("already" in item.value for item in app.info)
    worker.assert_not_called()


def test_detect_fills_the_cluster_form(bench, monkeypatch):
    app, worker, tmp_path = bench
    report = {"facts": {"partitions": [{"name": "main", "default": True, "state": "up", "timelimit": "1-00:00:00", "nodes": "4"}],
                        "accounts": ["proj1"], "modules": ["VASP/6.4"], "executables": {}, "potcar_roots": ["/sw/potpaw_PBE"],
                        "scratch_dirs": ["/scratch/alice"], "slurm": ["sbatch", "squeue", "sacct", "scancel"]},
              "suggested": {"partition": "main", "account": "proj1", "potcar_root": "/sw/potpaw_PBE",
                            "remote_root": "/scratch/alice/dft-agent-runs", "setup_commands": ["module load VASP/6.4"],
                            "vasp_command": "srun vasp_std", "vasp_ncl_command": "srun vasp_ncl"},
              "notes": ["Partition main allows up to 1-00:00:00 per job."]}
    probes = []
    monkeypatch.setattr(discovery, "probe_cluster", lambda config, transport=None, timeout=90: probes.append(config) or report)
    widget(app.button, "Detect from cluster").click().run()
    assert any("SSH host and user first" in item.value for item in app.error)
    widget(app.text_input, "SSH host").set_value("cluster.example.invalid")
    widget(app.text_input, "SSH user").set_value("alice")
    widget(app.button, "Detect from cluster").click().run()
    assert not app.exception
    assert probes[0].host == "cluster.example.invalid" and probes[0].user == "alice"
    assert widget(app.text_input, "Slurm partition").value == "main"
    assert widget(app.text_input, "Remote POTCAR folder").value == "/sw/potpaw_PBE"
    assert widget(app.text_input, "SSH host").value == "cluster.example.invalid"
    assert "module load VASP/6.4" in widget(app.text_area, "Environment setup commands (one per line)").value
    assert any("Detected:" in item.value for item in app.success)
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception
    saved = ClusterConfig.load(tmp_path / "cluster.json")
    assert saved.partition == "main" and saved.account == "proj1" and saved.setup_commands == ["module load VASP/6.4"]
    worker.assert_not_called()


def test_preset_fills_known_values(bench):
    app, worker, tmp_path = bench
    presets = widget(app.selectbox, "Preset")
    dardel = next(index for index, item in enumerate(presets.options) if "Dardel" in str(item))
    presets.set_value(presets.options[dardel]).run()
    widget(app.button, "Use preset").click().run()
    assert not app.exception
    assert widget(app.text_input, "SSH host").value == "dardel.pdc.kth.se"
    assert widget(app.text_input, "Slurm partition").value == "main"
    assert widget(app.number_input, "MPI tasks").value == 128


def test_cluster_check_shows_a_checklist(tmp_path, monkeypatch):
    ClusterConfig(host="cluster.example.invalid", user="researcher", remote_root="/scratch/test", vasp_command="srun vasp_std",
                  potcar_root="/licensed/pbe", partition="cpu").save(tmp_path / "cluster.json")
    report = {"ok": False, "checks": {"sbatch": {"ok": True, "detail": "/usr/bin/sbatch"}, "vasp": {"ok": False, "detail": ""}}}
    monkeypatch.setattr(cli, "doctor", lambda config, transport=None: report)
    app, worker = start(tmp_path, monkeypatch)
    widget(app.button, "Check cluster connection").click().run()
    assert not app.exception
    assert any("Some checks failed" in item.value for item in app.warning)
    table = next(frame for frame in app.dataframe if "Suggested fix" in list(frame.value.columns))
    assert list(table.value["Result"]) == ["✓", "✗"]
    assert "VASP" in table.value["Suggested fix"][1]
    worker.assert_not_called()


def test_remember_key_uses_the_keychain(bench, monkeypatch):
    app, worker, tmp_path = bench
    store = {}

    class FakeKeyring:
        def get_password(self, service, name):
            return store.get(name)

        def set_password(self, service, name, value):
            store[name] = value

        def delete_password(self, service, name):
            store.pop(name, None)

    monkeypatch.setattr(credentials, "_keyring", lambda: FakeKeyring())
    widget(app.selectbox, "Provider").select("anthropic").run()
    widget(app.text_input, "API key").set_value("claude-test-key").run()
    assert store == {}
    widget(app.checkbox, "Remember on this computer").check().run()
    assert not app.exception
    assert store == {"anthropic": "claude-test-key"}
    assert settings.load_settings()["keychain"] is True
    reopened = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
    widget(reopened.selectbox, "Provider").select("anthropic").run()
    assert widget(reopened.text_input, "API key").value == "claude-test-key"
    assert widget(reopened.checkbox, "Remember on this computer").value is True
    widget(reopened.checkbox, "Remember on this computer").uncheck().run()
    assert store == {}


def test_preferences_are_saved(bench):
    app, worker, tmp_path = bench
    assert settings.load_settings()["check_updates"] is True
    widget(app.checkbox, "Check for new versions").uncheck().run()
    assert not app.exception
    assert settings.load_settings()["check_updates"] is False
    widget(app.text_input, "Notification webhook URL (optional)").set_value("https://hooks.example/x").run()
    assert settings.load_settings()["notify_webhook"] == "https://hooks.example/x"


def test_switching_config_path_loads_and_saves_the_selected_cluster(bench):
    app, worker, tmp_path = bench
    save_cluster(app, **{"SSH host": "first.invalid"})
    second = ClusterConfig(host="second.invalid", user="other", remote_root="/scratch/other",
                           vasp_command="mpprun vasp_std", potcar_root="/licensed/other", partition="other")
    second_path = second.save(tmp_path / "second.json")
    widget(app.text_input, "Configuration file").set_value(str(second_path)).run()
    assert widget(app.text_input, "SSH host").value == "second.invalid"
    assert widget(app.text_input, "VASP command").value == "mpprun vasp_std"
    widget(app.button, "Save cluster settings").click().run()
    assert not app.exception
    assert ClusterConfig.load(second_path) == second
    assert ClusterConfig.load(tmp_path / "cluster.json").host == "first.invalid"
    worker.assert_not_called()


def test_switching_config_path_discards_a_draft_from_the_old_file(bench):
    app, worker, tmp_path = bench
    widget(app.button, "Use preset").click().run()
    assert app.session_state["cluster_draft"]
    widget(app.text_input, "Configuration file").set_value(str(tmp_path / "new.json")).run()
    assert not app.exception
    assert widget(app.text_input, "SSH host").value == ""
    assert widget(app.text_area, "Environment setup commands (one per line)").value == ""
    assert "cluster_draft" not in app.session_state


def test_detect_preserves_preset_launcher_and_environment(bench, monkeypatch):
    app, worker, tmp_path = bench
    presets = widget(app.selectbox, "Preset")
    presets.set_value(next(item for item in presets.options if "Tetralith" in str(item))).run()
    widget(app.button, "Use preset").click().run()
    widget(app.text_input, "SSH user").set_value("alice")
    widget(app.text_area, "Environment setup commands (one per line)").set_value("source /site/modules.sh\nmodule load compiler/2026")
    facts = {"modules": ["VASP/6.4"], "executables": {}, "slurm": ["sbatch", "squeue", "sacct", "scancel"]}
    monkeypatch.setattr(discovery, "probe_cluster", lambda *args, **kwargs: {"facts": facts, **discovery.suggest(facts)})
    widget(app.button, "Detect from cluster").click().run()
    assert not app.exception
    assert widget(app.text_input, "VASP command").value == "mpprun vasp_std"
    assert widget(app.text_input, "SOC / noncollinear command").value == "mpprun vasp_ncl"
    assert widget(app.text_area, "Environment setup commands (one per line)").value.splitlines() == [
        "source /site/modules.sh", "module load compiler/2026", "module load VASP/6.4"]
    worker.assert_not_called()


def test_gpu_settings_save_reload_and_reach_job_inputs(bench):
    app, worker, tmp_path = bench
    widget(app.selectbox, "Compute").select("gpu").run()
    widget(app.number_input, "Nodes").set_value(2)
    widget(app.number_input, "GPUs per node").set_value(2)
    widget(app.number_input, "CPUs per task").set_value(8).run()
    assert widget(app.number_input, "MPI tasks").disabled
    assert widget(app.number_input, "MPI tasks").value == 4
    widget(app.text_area, "Environment setup commands (one per line)").set_value("module load VASP/6.5-GPU")
    save_cluster(app, **{"Slurm partition": "gpu", "Slurm account (optional)": "test-gpu"})
    saved = ClusterConfig.load(tmp_path / "cluster.json")
    assert saved.compute_backend == "gpu" and saved.gpus_per_node == 2 and saved.cpus_per_task == 8
    assert saved.tasks == 4 and saved.nodes == 2 and saved.account == "test-gpu"
    reopened = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
    assert widget(reopened.selectbox, "Compute").value == "gpu"
    assert widget(reopened.number_input, "GPUs per node").value == 2
    assert widget(reopened.number_input, "CPUs per task").value == 8
    widget(app.radio, "Mode").set_value("Manual").run()
    app.file_uploader[0].upload("Si.cif", files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    frozen = ClusterConfig.load(run_dir / "config.json")
    assert frozen == saved
    script = workflow._script(frozen, workflow.read_state(run_dir)["stages"][0], None)
    assert "#SBATCH --gpus-per-node=2" in script and "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --ntasks=4" in script
    worker.assert_not_called()


def test_gpu_toggle_retains_cpu_task_count(bench):
    app, worker, tmp_path = bench
    widget(app.number_input, "MPI tasks").set_value(32).run()
    widget(app.selectbox, "Compute").select("gpu").run()
    assert widget(app.number_input, "MPI tasks").value == 1
    widget(app.selectbox, "Compute").select("cpu").run()
    assert widget(app.number_input, "MPI tasks").value == 32
    save_cluster(app)
    saved = ClusterConfig.load(tmp_path / "cluster.json")
    assert saved.compute_backend == "cpu" and saved.tasks == 32
    worker.assert_not_called()


def test_gpu_detect_passes_selected_backend_and_lists_resources(bench, monkeypatch):
    app, worker, tmp_path = bench
    probes = []
    facts = {"partitions": [{"name": "gpu", "default": False, "state": "up", "timelimit": "1:00:00",
                              "nodes": "8", "gres": "gpu:gh200:1", "gpus_per_node": 1}],
             "modules": ["VASP/6.5-GPU"], "slurm": ["sbatch", "squeue", "sacct", "scancel"]}
    def probe(config, **kwargs):
        probes.append(config)
        return {"facts": facts, **discovery.suggest(facts, config.compute_backend)}
    monkeypatch.setattr(discovery, "probe_cluster", probe)
    widget(app.selectbox, "Compute").select("gpu").run()
    widget(app.text_input, "SSH host").set_value("cluster.example.invalid")
    widget(app.text_input, "SSH user").set_value("alice")
    widget(app.button, "Detect from cluster").click().run()
    assert not app.exception
    assert probes[0].compute_backend == "gpu"
    assert widget(app.selectbox, "Compute").value == "gpu"
    assert widget(app.text_input, "Slurm partition").value == "gpu"
    table = next(frame.value for frame in app.dataframe if "GPU resources" in frame.value.columns)
    assert table["GPU resources"].tolist() == ["gpu:gh200:1"]
    assert "VASP/6.5-GPU" in widget(app.text_area, "Environment setup commands (one per line)").value
    worker.assert_not_called()
