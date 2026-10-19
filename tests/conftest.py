"""Keep tests away from the user's real settings, keychain, network and browser."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_AGENT_HOME", str(tmp_path / "dft-agent-home"))
    monkeypatch.setenv("DFT_AGENT_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("DFT_AGENT_NO_BROWSER", "1")
    # Existing workbench tests keep exercising the preserved legacy entrypoint.
    # Project UI tests opt into the new default explicitly.
    monkeypatch.setenv("DFT_AGENT_UI", "legacy")
    for name in ("DFT_AGENT_CONFIG", "DFT_AGENT_RUNS"):
        monkeypatch.delenv(name, raising=False)
    from vasp_slurm_agent.i18n import set_language

    set_language("en")
    yield
    set_language("en")
