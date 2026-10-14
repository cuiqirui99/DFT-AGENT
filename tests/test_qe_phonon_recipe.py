"""Phonon postprocessing must retain the physics of its source matrices."""
import pytest

from vasp_slurm_agent import qe_workflow


@pytest.mark.parametrize("task", ["q2r", "matdyn"])
@pytest.mark.parametrize("changes", [
    {"ecutwfc": 80}, {"spin": "collinear"}, {"mesh": [8, 8, 8]},
    {"qmesh": [4, 4, 4]}, {"namelists": {"system": {"ecutwfc": 80}}},
])
def test_postprocessing_cannot_relabel_parent_physics(task, changes):
    source = {"qmesh": [2, 2, 2], "mesh": [4, 4, 4], "ecutwfc": 40}
    tasks = ["ph", "q2r"] + (["matdyn"] if task == "matdyn" else [])
    recipes = [source] + ([{}] if task == "matdyn" else []) + [changes]
    with pytest.raises(ValueError, match="source phonon"):
        qe_workflow._sequence(tasks, {}, recipes)


def test_postprocessing_inherits_source_and_keeps_its_own_controls():
    source = {"qmesh": [2, 2, 2], "ecutwfc": 40,
              "namelists": {"electrons": {"conv_thr": 1e-10}}}
    stages = qe_workflow._sequence(["ph", "q2r", "matdyn"], {}, [source,
        {"asr": "crystal", "namelists": {"q2r": {"zasr": "crystal"}}},
        {"line_density": 80, "qpoints": [[0, 0, 0], [.5, 0, 0]],
         "namelists": {"matdyn": {"asr": "crystal"}}}])
    q2r, matdyn = stages[-2:]
    assert q2r["parameters"]["ecutwfc"] == matdyn["parameters"]["ecutwfc"] == 40
    assert matdyn["parameters"]["namelists"]["electrons"] == {"conv_thr": 1e-10}
    assert q2r["parameters"]["asr"] == "crystal"
    assert matdyn["parameters"]["line_density"] == 80
    assert matdyn["parameters"]["qpoints"] == [[0, 0, 0], [.5, 0, 0]]
    assert q2r["parameters"]["namelists"]["q2r"] == {"zasr": "crystal"}
    assert matdyn["parameters"]["namelists"]["matdyn"] == {"asr": "crystal"}
