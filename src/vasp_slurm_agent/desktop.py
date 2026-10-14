"""Run the existing app in a native desktop window."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from . import __version__
from .paths import app_home


def desktop_command(*arguments):
    if getattr(sys, "frozen", False):
        return [sys.executable, *map(str, arguments)]
    return [sys.executable, "-m", "vasp_slurm_agent.desktop", *map(str, arguments)]


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def wait_for_server(process, url, timeout=60):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The app could not start. See desktop.log in the settings folder.")
        try:
            with opener.open(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError("The app took too long to start. See desktop.log in the settings folder.")


def stop_server(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def serve(port):
    from streamlit.web import cli

    os.environ["DFT_AGENT_DESKTOP"] = "1"
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")),
                "--server.address=127.0.0.1", f"--server.port={port}",
                "--server.headless=true", "--browser.gatherUsageStats=false",
                "--client.toolbarMode=minimal", "--server.fileWatcherType=none",
                "--global.developmentMode=false"]
    return cli.main()


def check_window(window, output, errors):
    """Wait for the installed app to render inside its native webview."""
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            rendered = window.evaluate_js("document.body.innerText") or ""
            if "New calculation" in rendered and "Structure source" in rendered:
                if "Traceback" in rendered or "ModuleNotFoundError" in rendered:
                    raise RuntimeError("The desktop page contains an exception.")
                Path(output).write_text(json.dumps({"version": __version__, "platform": sys.platform,
                    "native_window": True, "app_rendered": True}, indent=2) + "\n", encoding="utf-8")
                return
            time.sleep(0.2)
        raise RuntimeError("The native window did not render the app.")
    except Exception as exc:
        errors.append(str(exc))
    finally:
        window.destroy()


def launch(window_report=None):
    import webview

    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    log_path = home / "desktop.log"
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    with log_path.open("ab") as log:
        process = subprocess.Popen(desktop_command("--serve", port), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, **flags)
        try:
            wait_for_server(process, url)
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.settings["ALLOW_FILE_URLS"] = False
            window = webview.create_window("DFT Agent", url, width=1200, height=850, min_size=(800, 600),
                                           text_select=True)
            errors = []
            checks = {"func": check_window, "args": (window, window_report, errors)} if window_report else {}
            webview.start(gui="edgechromium" if sys.platform == "win32" else None,
                          storage_path=str(home / "webview"), private_mode=True, **checks)
            if errors:
                log.write(("\n".join(errors) + "\n").encode("utf-8"))
                return 1
        except Exception as exc:
            message = f"DFT Agent could not open. {exc}\nLog: {log_path}"
            log.write((message + "\n").encode("utf-8"))
            log.flush()
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, message, "DFT Agent", 0x10)
            else:
                webview.create_window("DFT Agent", html="<h2>Cannot open DFT Agent</h2><p>" +
                                      html.escape(message) + "</p>", width=620, height=300)
                webview.start()
            return 1
        finally:
            stop_server(process)
    return 0


def _check_plot_exports(output):
    from matplotlib.figure import Figure
    from .vasp import _save_figure

    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(2, 2))
    figure.subplots().plot([0, 1], [0, 1])
    _save_figure(figure, output, "plot-check")
    contents = {extension: (output / f"plot-check.{extension}").read_bytes()
                for extension in ("png", "pdf", "svg")}
    assert all(len(data) > 100 for data in contents.values())
    assert contents["png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert contents["pdf"].startswith(b"%PDF-")
    assert b"<svg" in contents["svg"][:2048]


def _check_qe_bundle(root):
    from importlib.resources import files
    from .config import ClusterConfig
    from . import qe, qe_setup, qe_workflow

    for name in ("qe_pseudos.json", "qe_nc_pseudos.json", "qe_setup.py", "qe_epw.py", "qe_pairing.py"):
        assert files("vasp_slurm_agent").joinpath(name).is_file()
    assert qe_setup.plan_pseudopotentials(["Si"])["elements"]["Si"]
    nc = qe_setup.plan_pseudopotentials(["Si"], library="pseudodojo")
    assert nc["pseudo_type"] == "NC"
    source = root / "qe-Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(engine="qe", host="cluster.example.invalid", user="researcher",
                           remote_root="/scratch/check-qe", partition="cpu", qe_pseudo_root="/scratch/upf",
                           qe_pseudopotentials={"Si": nc["elements"]["Si"]["filename"]})
    plan = qe_setup.install_plan(config, include_epw=True)
    assert {"pw.x", "ph.x", "epw.x", "wannier90.x"} <= set(plan["programs"])
    run = root / "qe-run"
    state = qe_workflow.prepare_plan(source, run, config, ["scf"],
                                    {"ecutwfc": nc["ecutwfc_ry"], "ecutrho": nc["ecutrho_ry"], "mesh": [2, 2, 2]})
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    inputs = run / "01_scf" / "inputs"
    assert (inputs / "pw.in").is_file() and not (inputs / "INCAR").exists()
    metadata = json.loads((inputs / "qe-input.json").read_text())
    assert len(qe._expected_structure(inputs, metadata)) == 2


def check_bundle(output):
    """Exercise packaged resources and input preparation without network access."""
    import tempfile
    from importlib.resources import files
    from .config import ClusterConfig
    from .knowledge import load_catalog
    from .samples import install_samples
    from . import agent, workflow, batch, transport

    with tempfile.TemporaryDirectory(prefix="dft-agent-check-") as folder:
        root = Path(folder)
        source = root / "Si.cif"
        source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
        run = root / "run"
        workflow.prepare_plan(source, run, None, ["scf"], {"soc": True, "spin": "noncollinear",
                              "magmom": [[0, 0, 0], [0, 0, 0]], "saxis": [0, 0, 1]})
        config = ClusterConfig(host="cluster.example.invalid", user="researcher", remote_root="/scratch/check",
                               vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl",
                               potcar_root="/licensed/pbe", partition="cpu")
        state = workflow.attach_config(run, config)
        workflow._check_config(run, state)
        assert state["stages"][0]["metadata"]["requires_ncl"]
        assert (run / "01_scf/inputs/INCAR").is_file()
        installed = install_samples(root / "samples")
        assert len(installed) == 4
        assert load_catalog()
        import inspect
        assert inspect.getsource(transport._extract_checked_archive)
        for resource in ("app.py", "dispatch.py", "restart.py"):
            assert files("vasp_slurm_agent").joinpath(resource).is_file()
        for module in ("openai", "anthropic", "keyring", "webview"):
            __import__(module)
        if sys.platform == "win32":
            __import__("paramiko")
        _check_qe_bundle(root)
        _check_plot_exports(Path(output).resolve().parent / "plot-exports")
        from streamlit.testing.v1 import AppTest
        original = dict(os.environ)
        try:
            os.environ.update(DFT_AGENT_HOME=str(root / "settings"), DFT_AGENT_CONFIG=str(root / "cluster.json"),
                              DFT_AGENT_RUNS=str(root / "ui-runs"), DFT_AGENT_NO_UPDATE_CHECK="1")
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=60)
            assert not app.exception, str(app.exception)
            def widget(elements, label):
                return next(item for item in elements if item.label == label)
            widget(app.radio, "Structure source").set_value("Example").run()
            widget(app.radio, "Mode").set_value("Manual").run()
            widget(app.button, "Prepare inputs").click().run(timeout=60)
            assert not app.exception, str(app.exception)
            prepared = Path(app.session_state["active_run"])
            assert (prepared / "01_relax/inputs/INCAR").is_file()
            assert not (prepared / "config.json").exists()
            widget(app.button, "Load sample runs").click().run(timeout=60)
            assert not app.exception, str(app.exception)
            assert any(metric.label == "Final total energy (eV)" for metric in app.metric)
        finally:
            os.environ.clear()
            os.environ.update(original)
        result = {"version": __version__, "platform": sys.platform, "frozen": bool(getattr(sys, "frozen", False)),
                  "input_preparation": True, "cluster_attach": True, "samples": len(installed),
                  "resources": True, "provider_sdks": True, "first_run_ui": True,
                  "plot_exports": True, "qe_catalogues": True, "qe_input_preparation": True,
                  "qe_install_plan": True}
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main():
    if sys.argv[1:] == ["--askpass"]:
        sys.stdout.write(os.environ.get("_DFT_AGENT_ASKPASS_PASSWORD", "") + "\n")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from .cli import main as cli_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return cli_main()
    parser = argparse.ArgumentParser(description="DFT Agent desktop")
    parser.add_argument("--serve", type=int, metavar="PORT")
    parser.add_argument("--check", metavar="REPORT")
    parser.add_argument("--check-window", metavar="REPORT", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    if args.serve is not None:
        if not 1024 <= args.serve <= 65535:
            parser.error("Choose a port between 1024 and 65535.")
        return serve(args.serve)
    if args.check:
        return check_bundle(args.check)
    return launch(args.check_window)


if __name__ == "__main__":
    raise SystemExit(main())
