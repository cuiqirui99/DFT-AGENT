from io import BytesIO
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent.qe_app import _manual_recipes, _uploads


def test_manual_recipes_neb_does_not_accept_model_paths():
    with pytest.raises(ValueError, match="upload field"):
        _manual_recipes([], '[{"task":"neb","parameters":{"images":["/tmp/file"]}}]', ["a", "b"])
    with pytest.raises(ValueError, match="final structure"):
        _manual_recipes(["neb"], "", [])
    value = _manual_recipes(["neb"], "", [Path("a"), Path("b")])
    assert value[0]["parameters"]["images"] == ["a", "b"]
    assert _manual_recipes(["relax", "bands"], "", []) == [{"task": "relax", "parameters": {}}, {"task": "bands", "parameters": {}}]


def test_uploaded_names_cannot_escape_temp():
    source = BytesIO(b"source")
    source.name = "../../original.cif"
    endpoint = BytesIO(b"endpoint")
    endpoint.name = "/var/evil.cif"
    with _uploads(source, [endpoint]) as (path, images):
        assert path.name == "structure.cif"
        assert images[0] == path
        assert images[1].name == "image-1.cif"
        assert images[1].read_bytes() == b"endpoint"
    assert not path.exists()


def test_config_editor_first_launch(tmp_path):
    script = f'''
from pathlib import Path
from vasp_slurm_agent.qe_app import config_editor
config_editor(Path({str(tmp_path / "cluster.json")!r}), None)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    labels = [field.label for field in app.text_input]
    assert "SSH host" in labels and "UPF folder on cluster" in labels
    assert "Save settings" in [button.label for button in app.button]


def test_manual_ui_prepares_without_submitting(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "Si.cif"
    if not source.exists():
        source = Path(__file__).parents[1] / "src" / "vasp_slurm_agent" / "examples" / "Si.cif"
    script = f'''
from io import BytesIO
from pathlib import Path
from pymatgen.core import Structure
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.qe_app import new_run
source = Path({str(source)!r})
upload = BytesIO(source.read_bytes()); upload.name = source.name
config = ClusterConfig(host="example.invalid",user="tester",remote_root="/scratch/test",partition="test",engine="qe",qe_pseudo_root="/scratch/pseudo",qe_pseudopotentials={{"Si":"Si.UPF"}})
new_run(upload, Structure.from_file(source), config, Path({str(tmp_path / "runs")!r}), None, "Manual")
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    button = next(button for button in app.button if button.label == "Prepare inputs")
    button.click().run()
    assert not app.exception
    assert not app.error
    states = list((tmp_path / "runs").glob("*/run.json"))
    assert len(states) == 1
    import json
    state = json.loads(states[0].read_text())
    assert state["status"] == "planned" and all(stage["job_id"] is None for stage in state["stages"])
