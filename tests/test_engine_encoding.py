"""Native outputs and plots keep UTF-8 text on a Windows legacy locale."""
import shutil
from pathlib import Path

import pytest

from vasp_slurm_agent import abinit, cp2k, elk


@pytest.mark.parametrize("adapter", [abinit, cp2k, elk], ids=["abinit", "cp2k", "elk"])
def test_native_bands_keep_unicode_under_legacy_locale(tmp_path, monkeypatch, adapter):
    source = Path(__file__).parent / "data" / "engines" / adapter.ENGINE / "03_bands"
    target = tmp_path / "计算 Γ" / adapter.ENGINE
    shutil.copytree(source, target)
    metadata = target / adapter.METADATA_FILE
    metadata.write_text(metadata.read_text(encoding="utf-8").replace("GAMMA", "Γ"), encoding="utf-8")
    if adapter is cp2k:
        bands = target / "dft_agent.bs"
        bands.write_text(bands.read_text(encoding="utf-8").replace("GAMMA", "Γ"), encoding="utf-8")

    original_open = Path.open

    def legacy_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", legacy_open)
    result = adapter.analyze_outputs(target, "bands")
    assert result["success"], result["reason"]
    svg = (target / "bands.svg").read_text(encoding="utf-8")
    assert "Γ" in svg
    assert "Γ".encode("utf-8").decode("cp1252") not in svg
