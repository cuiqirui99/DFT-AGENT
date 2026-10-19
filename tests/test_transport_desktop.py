"""Desktop-specific SSH setup which does not require any SSH server."""
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import pytest

from vasp_slurm_agent import transport as transport_module
from vasp_slurm_agent.transport import SSHTransport


@pytest.mark.skipif(sys.platform == "win32", reason="Unix askpass wrapper; Windows authenticates through Paramiko")
def test_frozen_unix_askpass_uses_desktop_helper_mode(monkeypatch):
    monkeypatch.setattr(transport_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    executable = "/Applications/DFT Agent.app/Contents/MacOS/DFT Agent"
    monkeypatch.setattr(sys, "executable", executable)
    config = SimpleNamespace(host="unused.invalid", user="unused", port=22, connect_timeout=3)
    with SSHTransport(config, password="fake-test-password") as transport:
        helper = transport._askpass_path
        content = helper.read_text()
        # The prompt text is forwarded so passwords and login codes each answer only their own prompt.
        assert shlex.split(content.splitlines()[1]) == ["exec", executable, "--askpass", "$@"]
        assert "fake-test-password" not in content
    assert not Path(helper).exists()
