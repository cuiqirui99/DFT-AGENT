"""Run the existing app in a native desktop window."""

from __future__ import annotations

import argparse
import hashlib
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
            if "Traceback" in rendered or "ModuleNotFoundError" in rendered:
                raise RuntimeError("The desktop page contains an exception.")
            if "DFT Agent" in rendered and "Create your first project" in rendered:
                Path(output).write_text(json.dumps({"version": __version__, "platform": sys.platform,
                    "native_window": True, "app_rendered": True, "project_ui": True}, indent=2) + "\n", encoding="utf-8")
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


def _check_project_bundle(root):
    """Check first-run projects and the exact MCP subprocess used by the CLI agents."""
    from importlib.resources import files
    from streamlit.testing.v1 import AppTest
    from .workspace import Workspace
    from . import project_files
    from .runtime import cli_command

    for resource in ("agent_runner.py", "cli_tools.py", "mcp_server.py", "ssh_broker.py", "project_app.py",
                     "project_execution.py", "project_analysis.py", "project_input_artifacts.py"):
        assert files("vasp_slurm_agent").joinpath(resource).is_file()
    original = dict(os.environ)
    try:
        os.environ.update(DFT_AGENT_HOME=str(root / "project-settings"), DFT_AGENT_UI="project",
                          DFT_AGENT_NO_UPDATE_CHECK="1")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=60)
        assert not app.exception, str(app.exception)
        next(item for item in app.button if item.label == "Create your first project").click().run()
        next(item for item in app.text_input if item.label == "Name").set_value("Silicon").run()
        next(item for item in app.button if item.label == "Create project").click().run()
        assert not app.exception, str(app.exception)
        assert app.chat_input[0].disabled
        assert {"Connect a model", "Connect a cluster"} <= {item.label for item in app.button}
        workspace = Workspace()
        project = workspace.list_projects()[0]
        project_files.write_file(workspace, project["id"], "files/Si.cif",
                                 files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
        session = workspace.list_sessions(project["id"])[0]
        workspace.append_message(project["id"], session["id"], "user", "Keep this structure for later.")
        reopened = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=60)
        assert not reopened.exception, str(reopened.exception)
        assert "Keep this structure for later." in [item.value for item in reopened.markdown]
        env = dict(os.environ, DFT_AGENT_PROJECT=project["id"], DFT_AGENT_SESSION=session["id"])
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "project_overview", "arguments": {}}},
        ]
        result = subprocess.run(cli_command("mcp"), input="".join(json.dumps(item) + "\n" for item in requests),
                                capture_output=True, text=True, check=True, timeout=60, env=env)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        assert [item["id"] for item in responses] == [1, 2, 3]
        assert responses[0]["result"]["serverInfo"]["version"] == __version__
        tools = {item["name"] for item in responses[1]["result"]["tools"]}
        assert {"create_workflow", "plan_step", "explain_results", "qe_install_plan"} <= tools
        assert responses[2]["result"]["isError"] is False
        overview = json.loads(responses[2]["result"]["content"][0]["text"])
        assert overview["project"]["id"] == project["id"]
        assert [item["path"] for item in overview["files"]] == ["files/Si.cif"]
    finally:
        os.environ.clear()
        os.environ.update(original)


def _check_broker_bundle(root):
    """Start and stop the packaged SSH service through its normal keychain and CLI."""
    from . import credentials, ssh_broker
    from .runtime import cli_command

    home = root / "broker-settings"
    try:
        assert ssh_broker.start(home)["running"]
        result = subprocess.run(cli_command("ssh-broker", "status", "--home", home, "--connection", "missing"),
                                capture_output=True, text=True, check=True, timeout=30)
        assert json.loads(result.stdout) == {"connected": False}
        result = subprocess.run(cli_command("ssh-broker", "stop", "--home", home),
                                capture_output=True, text=True, check=True, timeout=30)
        assert json.loads(result.stdout)["stopped"]
        endpoint = home / "runtime" / "ssh-broker.json"
        deadline = time.monotonic() + 10
        while endpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not endpoint.exists(), "The local SSH service did not stop."
    finally:
        ssh_broker.stop(home)
        # This smoke check owns only the fresh workspace's capability.
        name = hashlib.sha256(str(home.resolve()).encode()).hexdigest()
        backend = credentials._keyring()
        if backend.get_password(ssh_broker._SERVICE, name) is not None:
            backend.delete_password(ssh_broker._SERVICE, name)


def _check_npm_cli_bundle(root):
    """Exercise external Node from the installed app without an account."""
    import shutil
    from .cli_tools import command

    if not shutil.which("node"):
        return "skipped"
    prefix = root / "CLI tools 中文"
    package = prefix / "node_modules" / "@anthropic-ai" / "claude-code"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"name": "@anthropic-ai/claude-code",
        "bin": {"claude": "cli.js"}}), encoding="utf-8")
    (package / "cli.js").write_text("const fs=require('fs');process.stdout.write(JSON.stringify({"
        "args:process.argv.slice(2),input:fs.readFileSync(0,'utf8')}));", encoding="utf-8")
    launcher = prefix / "claude.cmd"
    launcher.write_text("@echo off\r\nexit /b 9\r\n", encoding="utf-8")
    arguments = ['{"path":"C:\\\\Users\\\\研究者","items":[]}', "", '%PATH% & !x! | < > ^',
                 'first line\nsecond line', 'approval_policy="never"', "x" * 10000]
    prompt = '计算含 SOC 的能带。 Keep "quotes", %PATH% and C:\\Users\\研究者.'
    result = subprocess.run(command([str(launcher), *arguments], windows=True), input=prompt,
                            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
    assert json.loads(result.stdout) == {"args": arguments, "input": prompt}
    return "checked"


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
        if sys.platform in {"darwin", "win32"}:
            from .credentials import _keyring
            _keyring()  # Load and validate the native backend without reading credentials.
        if sys.platform == "win32":
            __import__("paramiko")
        _check_qe_bundle(root)
        _check_project_bundle(root)
        if sys.platform in {"darwin", "win32"}:
            _check_broker_bundle(root)
        npm_cli = _check_npm_cli_bundle(root) if sys.platform == "win32" else "not_applicable"
        _check_plot_exports(Path(output).resolve().parent / "plot-exports")
        from streamlit.testing.v1 import AppTest
        original = dict(os.environ)
        try:
            os.environ.update(DFT_AGENT_HOME=str(root / "settings"), DFT_AGENT_CONFIG=str(root / "cluster.json"),
                              DFT_AGENT_RUNS=str(root / "ui-runs"), DFT_AGENT_NO_UPDATE_CHECK="1", DFT_AGENT_UI="legacy")
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
                  "native_keychain": sys.platform in {"darwin", "win32"},
                  "ssh_broker_lifecycle": sys.platform in {"darwin", "win32"},
                  "windows_npm_cli": npm_cli,
                  "project_ui": True, "project_persistence": True, "mcp_subprocess": True,
                  "plot_exports": True, "qe_catalogues": True, "qe_input_preparation": True,
                  "qe_install_plan": True}
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main():
    if sys.argv[1:2] == ["--askpass"]:
        from .transport import askpass_main
        return askpass_main(sys.argv[1:])
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
