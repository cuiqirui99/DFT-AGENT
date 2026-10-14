import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_prepare_and_review_pairing_without_submitting(tmp_path):
    hr = tmp_path / "model_hr.dat"
    hr.write_text("Model\n1\n3\n1 1 1\n-1 0 0 1 1 -1 0\n0 0 0 1 1 0 0\n1 0 0 1 1 -1 0\n")
    parameters = dict(lattice_angstrom=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], orbital="s",
                      spin_convention="spin_degenerate_no_soc", u_ev=.5, mu_ev=-.5,
                      temperature_k=100, kmesh=[16, 1, 1], matsubara_points=100,
                      channels=["singlet", "triplet"], double_counting="None; model")
    script = f'''
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.qe_pairing_app import new_run
source = Path({str(hr)!r})
upload = BytesIO(source.read_bytes()); upload.name = source.name
config = ClusterConfig(engine='qe',host='example.invalid',user='tester',partition='cpu',remote_root='/scratch/test',qe_pseudo_root='/scratch/pseudo')
with patch('streamlit.file_uploader', return_value=upload):
    new_run(config, Path({str(tmp_path / 'runs')!r}), None)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    app.radio(key="pairing_mode").set_value("Manual").run()
    app.text_input(key="pairing_python").set_value("python3").run()
    app.text_area(key="pairing_manual").set_value(json.dumps(parameters)).run()
    app.button(key="pairing_manual_prepare").click().run()
    assert not app.exception and not app.error
    runs = list((tmp_path / "runs").glob("*/run.json"))
    assert len(runs) == 1
    state = json.loads(runs[0].read_text())
    assert state["workflow"] == "pairing" and state["status"] == "planned"
    assert state["stages"][0]["job_id"] is None
    panel = AppTest.from_string(f'''
from pathlib import Path
from vasp_slurm_agent.qe_pairing_app import run_panel
run_panel(Path({str(runs[0].parent)!r}), None)
''').run()
    assert not panel.exception
    assert next(x for x in panel.button if x.label == "Submit calculation").disabled
