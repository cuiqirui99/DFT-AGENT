from matplotlib.figure import Figure
import pytest

from vasp_slurm_agent.qe import _plot
from vasp_slurm_agent.vasp import _save_figure


@pytest.mark.parametrize("engine", ["qe", "vasp"])
def test_same_data_produces_identical_downloads(tmp_path, engine):
    """Evidence rechecking must not change PDF dates or SVG object IDs."""
    outputs = []
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        if engine == "qe":
            files = _plot(folder, "energy", [1, 2], [-2, -3], "Step", "Energy (eV)", steps=True)
        else:
            figure = Figure()
            figure.subplots().plot([1, 2], [-2, -3])
            files = _save_figure(figure, folder, "energy")
        outputs.append({file: (folder / file).read_bytes() for file in files})
    assert outputs[0] == outputs[1]
    assert b"/CreationDate" not in outputs[0]["energy.pdf"]
    assert b"<dc:date>" not in outputs[0]["energy.svg"]
