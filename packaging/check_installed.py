"""Check an installed desktop bundle in a new, isolated user workspace."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import ProxyHandler, build_opener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    executable = str(args.executable.resolve())
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(DFT_AGENT_HOME=str(root / "settings"), DFT_AGENT_RUNS=str(root / "runs"),
               DFT_AGENT_CONFIG=str(root / "settings" / "cluster.json"),
               DFT_AGENT_NO_UPDATE_CHECK="1", DFT_AGENT_NO_BROWSER="1", DFT_AGENT_UI="project")
    env.pop("VASP_AGENT_CONFIG", None)
    report = root / "bundle-check.json"
    subprocess.run([executable, "--check", str(report)], cwd=root, env=env, check=True, timeout=180)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    with (root / "server.log").open("wb") as log:
        process = subprocess.Popen([executable, "--serve", str(port)], cwd=root, env=env,
                                   stdout=log, stderr=log)
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Installed server exited; see server.log")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        if response.read().strip() == b"ok":
                            break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Installed server did not become ready")
            with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as response:
                assert response.status == 200
                assert b"<html" in response.read().lower()
            result = json.loads(report.read_text())
            assert result["frozen"] is True
            if os.name == "nt":
                assert result["windows_npm_cli"] == "checked", "Windows acceptance needs Node.js on PATH"
            result["installed_server"] = True
            report.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    subprocess.run([executable, "--check-window", str(root / "window-check.json")],
                   cwd=root, env=env, check=True, timeout=120)
    assert json.loads((root / "window-check.json").read_text())["app_rendered"] is True


if __name__ == "__main__":
    main()
