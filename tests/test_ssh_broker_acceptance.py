"""Real local SSH MFA and session reuse across a daemon and worker processes.

The SSH server uses real SSH and simulated Slurm replies. This verifies Windows
authentication/transport, not an external cluster or a scientific calculation.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import pytest

pytest.importorskip("paramiko")
from test_windows_transport import native_home, ssh_server  # noqa: F401


_BROKER_PROCESS = """
import json, sys
import paramiko
from vasp_slurm_agent import ssh_broker
class EmptyAgent:
    def get_keys(self): return []
    def close(self): pass
paramiko.client.Agent = EmptyAgent
data = json.loads(sys.stdin.readline())
capability = bytes.fromhex(data['capability'])
ssh_broker._capability = lambda home, create=False: capability
ssh_broker.BrokerServer(data['home'], capability, idle_timeout=120).serve()
"""

_WORKER_PROCESS = """
import json, platform, sys
from pathlib import Path
from vasp_slurm_agent import ssh_broker
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import SSHTransport, VerificationRequired
data = json.loads(sys.stdin.readline())
capability = bytes.fromhex(data['capability'])
ssh_broker._capability = lambda home, create=False: capability
platform.system = lambda: 'Windows'
try:
    with SSHTransport(ClusterConfig(**data['config'])) as transport:
        result = transport.run("python3 -c 'print(41)'")
        assert result.returncode == 0 and result.stdout.strip() == '41'
        if data.get('stage') == 'submit':
            result = transport.run('sbatch --parsable submit.sh')
            assert result.returncode == 0 and result.stdout.strip() == '12345'
        elif data.get('stage') == 'monitor':
            result = transport.run('squeue -h -j 12345 -o %T')
            assert result.returncode == 0 and result.stdout.strip() == 'COMPLETED'
        if data.get('input'):
            uploaded = transport.upload(Path(data['input']), '/cluster/worker-input.txt')
            downloaded = transport.download('/cluster/worker-input.txt', Path(data['output']))
            assert uploaded == downloaded
    print(json.dumps({'connected': True}))
except VerificationRequired:
    print(json.dumps({'connected': False, 'needs_sign_in': True}))
"""


def _child_environment(home):
    environment = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
    for key in ("SSH_AUTH_SOCK", "DFT_AGENT_SSH_PASSWORD", "_DFT_AGENT_WORKER_PASSWORD", "_DFT_AGENT_ASKPASS_PASSWORD",
                "_DFT_AGENT_ASKPASS_CODE"):
        environment.pop(key, None)
    return environment


def test_mfa_session_survives_independent_workers_and_expires_without_reusing_code(native_home, ssh_server, tmp_path, monkeypatch):
    from vasp_slurm_agent import ssh_broker
    from vasp_slurm_agent.config import ClusterConfig
    from vasp_slurm_agent.connections import ConnectionRegistry
    from vasp_slurm_agent.workspace import Workspace

    ssh_server.authentication.mfa = True
    workspace = Workspace(tmp_path / "workspace")
    registry = ConnectionRegistry(workspace)
    record = registry.create("hpc", "Local MFA acceptance", {
        "host": "127.0.0.1", "user": "test-user", "port": ssh_server.config.port,
        "auth_mode": "code", "scheduler": "slurm", "remote_base": "/cluster"})
    config = ClusterConfig(host="127.0.0.1", user="test-user", port=ssh_server.config.port,
                           remote_root="/cluster", partition="test", engine="qe", qe_pseudo_root="/cluster/upf",
                           connection_id=record["id"], connection_home=str(workspace.home), connect_timeout=3)
    capability = secrets.token_bytes(32)
    monkeypatch.setattr(ssh_broker, "_capability", lambda home, create=False: capability)
    environment = _child_environment(native_home)
    process = subprocess.Popen([sys.executable, "-u", "-c", _BROKER_PROCESS], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    process.stdin.write(json.dumps({"home": str(workspace.home), "capability": capability.hex()}) + "\n")
    process.stdin.flush()
    process.stdin.close()
    try:
        endpoint = workspace.home / "runtime" / "ssh-broker.json"
        deadline = time.monotonic() + 20
        while not endpoint.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert process.poll() is None, process.stderr.read()
        assert endpoint.is_file(), "Broker did not start"
        ssh_broker.sign_in(config, "test-password", "123456")
        assert ssh_broker.status(workspace.home, record["id"])["connected"]
        assert ssh_server.authentication.accepted == 1

        def worker(**extra):
            # Password and OTP are deliberately absent from both payload and environment.
            payload = {"capability": capability.hex(), "config": asdict(config), **extra}
            result = subprocess.run([sys.executable, "-u", "-c", _WORKER_PROCESS],
                                    input=json.dumps(payload) + "\n", text=True, capture_output=True,
                                    env=environment, timeout=30, check=False)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        assert worker(stage="submit") == {"connected": True}
        source, output = tmp_path / "input.txt", tmp_path / "download.txt"
        source.write_text("Separate worker file transfer.\n")
        assert worker(stage="monitor", input=str(source), output=str(output)) == {"connected": True}
        assert output.read_bytes() == source.read_bytes()
        assert ssh_server.authentication.accepted == 1
        assert ssh_server.authentication.password_attempts == 1
        assert ssh_server.authentication.code_attempts == 1
        assert len(ssh_server.transports) == 1
        assert "sbatch --parsable submit.sh" in ssh_server.calls
        assert "squeue -h -j 12345 -o %T" in ssh_server.calls

        for transport in ssh_server.transports:
            transport.close()
        assert worker() == {"connected": False, "needs_sign_in": True}
        assert ssh_server.authentication.accepted == 1
        assert ssh_server.authentication.code_attempts == 1
        assert not ssh_broker.status(workspace.home, record["id"])["connected"]
        for path in workspace.home.rglob("*"):
            if path.is_file():
                assert b"test-password" not in path.read_bytes(), str(path)
                assert b"123456" not in path.read_bytes(), str(path)
    finally:
        ssh_broker.stop(workspace.home)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    assert process.returncode == 0, process.stderr.read()
