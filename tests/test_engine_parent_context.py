import json
import shutil

from vasp_slurm_agent import conversation, engine_context
from vasp_slurm_agent.config import ClusterConfig
from test_conversation import SETTINGS, provider, route
from test_engine_adapters import _plan
from test_engine_workflow import DATA, _drive, _setup


def test_continuation_explanation_checks_parent_geometry(tmp_path, monkeypatch):
    parent, cluster, _, _ = _setup(tmp_path, "abinit")
    assert _drive(parent, cluster)["status"] == "succeeded"
    folder = conversation.open_run(parent, tmp_path)
    provider(monkeypatch, [("dft_conversation", route("continue", 0)), ("abinit_task", _plan(["scf"]))])
    config = ClusterConfig.load(parent / "config.json")
    conversation.send(folder, "SCF from the relaxed geometry.", SETTINGS, config=config)
    child = parent.parent / "child"
    conversation.prepare(folder, child, config)
    replay = tmp_path / "continuation-replay"
    shutil.copytree(DATA / "abinit" / "02_scf", replay / "abinit" / "01_scf")
    cluster.env["REPLAY_DATA"] = str(replay)
    assert _drive(child, cluster)["status"] == "succeeded"
    facts = engine_context.load_run_context(child)["facts"]
    assert facts["continuation.source_verified"]["value"] is True
    assert facts["continuation.geometry_unchanged"]["value"] is True
    assert facts["continuation.source_task"]["value"] == "relax"
    assert facts["continuation.source_run_id"]["value"] == engine_context.load_run_context(parent)["facts"]["run.native_id"]["value"]
    assert facts["continuation.electronic_state_reused"]["value"] is False
    provenance = json.loads((child / "parent.json").read_text())
    provenance["source_structure_sha256"] = "0" * 64
    (child / "parent.json").write_text(json.dumps(provenance))
    facts = engine_context.load_run_context(child)["facts"]
    assert facts["continuation.source_verified"]["value"] is False
    assert "continuation.geometry_unchanged" not in facts
