"""Interface text falls back to English and switches to Chinese in the app."""

from importlib.resources import files
import re

from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import i18n


def test_translation_fallback_and_placeholders():
    i18n.set_language("zh")
    assert i18n.t("Cluster setup") == "集群设置"
    assert i18n.t("Not a known string") == "Not a known string"
    i18n.set_language("nope")
    assert i18n.current_language() == "en"
    assert i18n.t("Cluster setup") == "Cluster setup"
    for key, value in i18n.ZH.items():
        assert set(re.findall(r"\{[a-z_]+\}", key)) == set(re.findall(r"\{[a-z_]+\}", value)), key
        assert value.strip()
        assert '"' not in value or key.startswith(("Optional", "Example")), key


def test_language_switch_translates_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "missing.json"))
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    assert [tab.label for tab in app.tabs] == ["Chat", "New calculation", "Runs", "Cluster setup", "Memory"]
    next(box for box in app.selectbox if box.label == "Language / 语言").select("zh").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["对话", "新建计算", "计算记录", "集群设置", "记忆"]
    assert any(button.label == "保存集群设置" for button in app.button)
    assert any(button.label == "新建对话" for button in app.button)
    assert any(box.label == "对话记录" for box in app.selectbox)
    reopened = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    assert [tab.label for tab in reopened.tabs][0] == "对话"
    next(box for box in reopened.selectbox if box.label == "Language / 语言").select("en").run()
    assert [tab.label for tab in reopened.tabs][0] == "Chat"


def test_language_switch_preserves_model_structure_and_manual_inputs(tmp_path, monkeypatch):
    from vasp_slurm_agent import credentials, workflow

    def widget(elements, label):
        return next(item for item in elements if item.label == label)

    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setattr(credentials, "stored_key", lambda provider: None)
    monkeypatch.setattr(credentials, "keyring_available", lambda: False)
    monkeypatch.setattr(workflow, "SSHTransport", lambda *args: (_ for _ in ()).throw(AssertionError("No SSH")))
    monkeypatch.setattr(workflow, "start_worker", lambda *args: (_ for _ in ()).throw(AssertionError("No submission")))
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
    widget(app.text_input, "Configuration file").set_value(str(tmp_path / "alternate.json")).run()
    widget(app.text_input, "Run folder").set_value(str(tmp_path / "chosen-runs")).run()
    widget(app.selectbox, "Provider").select("anthropic").run()
    widget(app.text_input, "Model name").set_value("test-model").run()
    widget(app.text_input, "API key").set_value("test-key").run()
    widget(app.text_input, "SSH host").set_value("unsaved.invalid").run()
    widget(app.text_input, "Remote run folder (absolute path)").set_value("/scratch/unsaved").run()
    widget(app.radio, "Mode").set_value("Manual").run()
    payload = files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()
    app.file_uploader[0].upload("Si.cif", payload).run()
    widget(app.selectbox, "Calculation").set_value("scf")
    widget(app.selectbox, "Spin").set_value("collinear")
    widget(app.text_area, "Site moments (JSON)").set_value("[1, 1]")
    widget(app.text_input, "Gamma-centered k-point grid").set_value("3 3 3")
    widget(app.number_input, "ENCUT / eV").set_value(600.0)
    widget(app.selectbox, "Language / 语言").select("zh").run()
    assert not app.exception
    for key, expected in {"config_path": str(tmp_path / "alternate.json"), "runs_root": str(tmp_path / "chosen-runs"),
                          "model_provider": "anthropic", "model_name_anthropic": "test-model", "model_api_key_anthropic": "test-key",
                          "calculation_mode": "Manual", "manual_task": "scf", "manual_spin": "collinear",
                          "manual_moments": "[1, 1]", "manual_grid": "3 3 3", "manual_encut": 600.0}.items():
        assert app.session_state[key] == expected, key
    assert app.file_uploader[0].value.getvalue() == payload
    assert widget(app.text_input, i18n.t("SSH host")).value == "unsaved.invalid"
    assert widget(app.text_input, i18n.t("Remote run folder (absolute path)")).value == "/scratch/unsaved"
    assert not (tmp_path / "alternate.json").exists()
    assert not (tmp_path / "chosen-runs").exists()
    assert not widget(app.button, "准备输入文件").disabled
    widget(app.selectbox, "Language / 语言").select("en").run()
    assert not app.exception
    assert widget(app.text_input, "Gamma-centered k-point grid").value == "3 3 3"
    assert widget(app.selectbox, "Calculation").value == "scf"


def test_language_switch_keeps_agent_goal(tmp_path, monkeypatch):
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "missing.json"))
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=30).run()
    app.text_area(key="calculation_goal").set_value("Relax silicon and calculate bands").run()
    app.selectbox(key="language_choice").select("zh").run()
    assert not app.exception
    assert app.text_area(key="calculation_goal").value == "Relax silicon and calculate bands"
