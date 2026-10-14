from pathlib import PureWindowsPath

from vasp_slurm_agent import qe_epw
from test_qe_epw import checkpoint_fixture


def test_native_checkpoint_manifest_uses_remote_paths_on_windows(tmp_path, monkeypatch):
    checkpoint_fixture(tmp_path)

    class WindowsRelativePath(type(tmp_path)):
        def relative_to(self, *args, **kwargs):
            return PureWindowsPath(super().relative_to(*args, **kwargs))

    monkeypatch.setattr(qe_epw, "Path", WindowsRelativePath)
    manifest = qe_epw.read_checkpoint(tmp_path)["sha256"]
    assert "scratch/dft_agent.ephmat/ephmat1" in manifest
    assert "scratch/dft_agent.ephmat/freq" in manifest
    assert all("\\" not in name for name in manifest)
