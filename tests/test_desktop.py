"""Desktop process lifecycle and bundled worker commands."""

import sys
from unittest.mock import Mock

import pytest

from vasp_slurm_agent import desktop, runtime


def test_frozen_worker_runs_cli_without_opening_a_window(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert runtime.cli_command("watch", "/a run") == [sys.executable, "--cli", "watch", "/a run"]
    from vasp_slurm_agent import cli
    invoke = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", invoke)
    monkeypatch.setattr(sys, "argv", ["DFT Agent", "--cli", "watch", "/a run"])
    assert desktop.main() == 0
    assert sys.argv == ["DFT Agent", "watch", "/a run"]
    invoke.assert_called_once()


def test_source_worker_command():
    assert runtime.cli_command("watch", "/a run")[1:] == ["-m", "vasp_slurm_agent.cli", "watch", "/a run"]


def test_backend_start_failure_does_not_wait_for_timeout():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="could not start"):
        desktop.wait_for_server(process, "http://127.0.0.1:1")


def test_close_stops_only_its_backend():
    process = Mock()
    process.poll.return_value = None
    desktop.stop_server(process)
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=8)
    process.kill.assert_not_called()


def test_askpass_keeps_password_out_of_arguments(monkeypatch, capsys):
    monkeypatch.setenv("_DFT_AGENT_ASKPASS_PASSWORD", "test-only-password")
    monkeypatch.setattr(sys, "argv", ["DFT Agent", "--askpass"])
    assert desktop.main() == 0
    assert capsys.readouterr().out == "test-only-password\n"


def test_backend_stays_on_loopback(monkeypatch):
    from streamlit.web import cli
    start = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", start)
    monkeypatch.setattr(sys, "argv", ["test"])
    assert desktop.serve(54321) == 0
    assert "--server.address=127.0.0.1" in sys.argv
    assert "--server.port=54321" in sys.argv
    assert "--browser.gatherUsageStats=false" in sys.argv
    assert "--global.developmentMode=false" in sys.argv


def test_window_check_requires_the_app_not_just_an_empty_browser(tmp_path):
    window = Mock()
    window.evaluate_js.side_effect = ["", "DFT Agent New calculation Structure source"]
    errors = []
    report = tmp_path / "window.json"
    desktop.check_window(window, report, errors)
    assert not errors
    assert report.is_file()
    window.destroy.assert_called_once()


def test_window_check_reports_render_errors(tmp_path):
    window = Mock()
    window.evaluate_js.side_effect = RuntimeError("webview failed")
    errors = []
    desktop.check_window(window, tmp_path / "window.json", errors)
    assert errors == ["webview failed"]
    window.destroy.assert_called_once()


def test_bundle_check_renders_plot_exports(tmp_path):
    desktop._check_plot_exports(tmp_path / "plots")
    assert {path.suffix for path in (tmp_path / "plots").iterdir()} == {".png", ".pdf", ".svg"}


def test_bundle_check_prepares_qe_with_packaged_catalogues_without_remote_calls(tmp_path, monkeypatch):
    import json
    from vasp_slurm_agent import qe_setup, qe_workflow

    remote = Mock(side_effect=AssertionError("Bundle checks must not contact a cluster"))
    monkeypatch.setattr(qe_setup, "SSHTransport", remote)
    monkeypatch.setattr(qe_workflow, "SSHTransport", remote)
    desktop._check_qe_bundle(tmp_path)
    state = json.loads((tmp_path / "qe-run/run.json").read_text())
    assert state["engine"] == "qe" and state["status"] == "planned"
    assert all(stage["job_id"] is None for stage in state["stages"])
    assert "Si.pbe-nc-sr-04-pd.upf" in (tmp_path / "qe-run/01_scf/inputs/pw.in").read_text()
    remote.assert_not_called()


def test_bundle_check_rejects_empty_plot_exports(tmp_path, monkeypatch):
    from vasp_slurm_agent import vasp

    def empty_exports(figure, output, name):
        for extension in ("png", "pdf", "svg"):
            (output / f"{name}.{extension}").write_bytes(b"")

    monkeypatch.setattr(vasp, "_save_figure", empty_exports)
    with pytest.raises(AssertionError):
        desktop._check_plot_exports(tmp_path / "plots")
