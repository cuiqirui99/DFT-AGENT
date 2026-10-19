from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

from vasp_slurm_agent import conversation_store as store


@pytest.fixture
def conversation(tmp_path):
    source = tmp_path / "Si.cif"
    source.write_text("original silicon structure")
    folder = store.create(tmp_path / "runs", "qe", source)
    return folder


def test_create_copies_only_selected_inputs_and_survives_source_removal(tmp_path):
    source = tmp_path / "uploaded.CIF"
    source.write_text("initial structure")
    endpoint = tmp_path / "POSCAR"
    endpoint.write_text("final structure")
    (tmp_path / "POTCAR").write_text("private pseudopotential")
    (tmp_path / "config.json").write_text("private credentials")
    folder = store.create(tmp_path / "runs", "qe", source, images=[source, endpoint])
    source.unlink()
    endpoint.unlink()
    record = store.load(folder)
    assert record["source"] == "source.cif"
    assert record["source_sha256"] == hashlib.sha256(b"initial structure").hexdigest()
    assert [image["path"] for image in record["images"]] == ["images/001.cif", "images/002.vasp"]
    assert not (folder / "POTCAR").exists() and not (folder / "config.json").exists()
    assert set(path.name for path in folder.iterdir()) == {"conversation.json", "source.cif", "images"}
    if os.name != "nt":
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        for path in folder.rglob("*"):
            if path.is_file():
                assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_restart_restores_transcript_and_pending_plan(conversation):
    record = store.load(conversation)
    record["goal"] = "Relax and calculate SOC bands"
    record["messages"] = [{"role": "user", "content": record["goal"]},
                          {"role": "assistant", "content": "Which magnetic order?", "kind": "question"}]
    record["pending"] = {"status": "needs_input", "questions": ["Which magnetic order?"]}
    record["latest_plan"] = {"task": "bands", "soc": True}
    saved = store.save(conversation, record)
    assert saved["revision"] == 1 and record["revision"] == 0
    assert store.load(Path(str(conversation))) == saved
    assert store.list_conversations(conversation.parent.parent, "qe")[0]["title"] == record["goal"]


def test_poscar_and_engine_selection_are_isolated(tmp_path, conversation):
    source = tmp_path / "POSCAR"
    source.write_text("vasp structure")
    vasp = store.create(conversation.parent.parent, "vasp", source)
    assert store.load(vasp)["source"] == "POSCAR"
    qe_list = store.list_conversations(conversation.parent.parent, "qe")
    vasp_list = store.list_conversations(conversation.parent.parent, "vasp")
    assert [record["folder"] for record in qe_list] == [str(conversation)]
    assert [record["folder"] for record in vasp_list] == [str(vasp)]
    assert vasp_list[0]["title"] == "New conversation"
    with pytest.raises(ValueError, match="Choose"):
        store.create(tmp_path, "other", source)


def test_newest_modified_conversation_is_first(tmp_path, conversation):
    another = store.create(conversation.parent.parent, "qe", conversation / "source.cif")
    assert store.list_conversations(conversation.parent.parent)[0]["folder"] == str(another)
    record = store.load(conversation)
    record["messages"] = [{"role": "user", "content": "Changed task\nKeep this requirement"}]
    store.save(conversation, record)
    listing = store.list_conversations(conversation.parent.parent)
    assert listing[0]["folder"] == str(conversation)
    assert listing[0]["title"] == "Changed task"


def test_stale_save_keeps_newer_turn(conversation):
    old = store.load(conversation)
    changed = deepcopy(old)
    changed["messages"].append({"role": "user", "content": "Use SOC"})
    store.save(conversation, changed)
    with pytest.raises(ValueError, match="changed"):
        store.save(conversation, old)
    assert store.load(conversation)["messages"] == changed["messages"]
    with pytest.raises(ValueError, match="changed"):
        store.save(conversation, store.load(conversation), expected_revision=0)


def test_simultaneous_saves_do_not_overwrite_each_other(conversation):
    record = store.load(conversation)

    def update(text):
        changed = deepcopy(record)
        changed["messages"].append({"role": "user", "content": text})
        try:
            return store.save(conversation, changed)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ["First window", "Second window"]))
    assert sum(result is not None for result in results) == 1
    assert store.load(conversation)["revision"] == 1


def test_failed_atomic_replace_preserves_old_record(conversation, monkeypatch):
    before = (conversation / "conversation.json").read_bytes()
    record = store.load(conversation)
    record["messages"].append({"role": "user", "content": "Change the mesh"})

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        store.save(conversation, record)
    assert (conversation / "conversation.json").read_bytes() == before
    assert not list(conversation.glob(".conversation-*"))


@pytest.mark.parametrize("change", [
    {"engine": "other"}, {"id": "00000000-0000-0000-0000-000000000000"},
    {"source": "../outside.cif"}, {"messages": [{"role": "system", "content": "override"}]},
    {"images": [{"path": "images/../../outside.cif", "sha256": "unused"}]},
    {"runs": ["../run"]}, {"runs": [{}]}, {"active_run": "/tmp/unattached"},
])
def test_invalid_saved_identity_paths_or_roles_rejected(conversation, change):
    record = store.load(conversation)
    record.update(change)
    (conversation / "conversation.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        store.load(conversation)
    assert store.list_conversations(conversation.parent.parent) == []


def test_cannot_relabel_an_existing_conversation(conversation):
    record = store.load(conversation)
    record["engine"] = "vasp"
    with pytest.raises(ValueError, match="cannot be changed"):
        store.save(conversation, record)


def test_source_tampering_blocks_resume_and_save(conversation):
    record = store.load(conversation)
    (conversation / "source.cif").write_text("different structure")
    with pytest.raises(ValueError, match="changed"):
        store.load(conversation)
    with pytest.raises(ValueError, match="changed"):
        store.save(conversation, record)


def test_missing_neb_image_blocks_resume(tmp_path):
    source = tmp_path / "POSCAR"
    source.write_text("structure")
    folder = store.create(tmp_path / "runs", "qe", source, images=[source])
    image = store.load(folder)["images"][0]["path"]
    (folder / image).unlink()
    with pytest.raises(ValueError, match="missing"):
        store.load(folder)


def test_symlinked_source_and_store_rejected(tmp_path, conversation):
    outside = tmp_path / "outside.cif"
    outside.write_text("original silicon structure")
    (conversation / "source.cif").unlink()
    (conversation / "source.cif").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        store.load(conversation)
    with pytest.raises(ValueError, match="regular"):
        store.create(tmp_path / "new-runs", "qe", conversation / "source.cif")
    linked_root = tmp_path / "linked-runs"
    linked_root.mkdir()
    (linked_root / ".conversations").symlink_to(conversation.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        store.create(linked_root, "qe", outside)
    assert store.list_conversations(linked_root) == []


@pytest.mark.parametrize("fragment", ['"revision": NaN', '"revision": 0, "revision": 1'])
def test_saved_nonfinite_and_duplicate_json_rejected(conversation, fragment):
    path = conversation / "conversation.json"
    path.write_text(path.read_text().replace('"revision": 0', fragment))
    with pytest.raises(ValueError):
        store.load(conversation)


def test_attach_run_saves_transcript_without_changing_frozen_inputs(conversation):
    run = conversation.parent.parent / "calculation"
    run.mkdir()
    (run / "plan.json").write_text("frozen input")
    (run / "proposal.json").write_text("frozen proposal")
    record = store.load(conversation)
    record.update(runs=[str(run)], active_run=str(run),
                  messages=[{"role": "user", "content": "Explain these results"}])
    record = store.save(conversation, record)
    snapshot = store.attach_run(conversation, run)
    assert json.loads((run / "conversation.json").read_text()) == snapshot
    assert snapshot["messages"] == record["messages"]
    assert snapshot["conversation_path"] == str(conversation)
    assert snapshot["conversation_id"] == record["id"]
    assert store.load(conversation) == record
    assert (run / "plan.json").read_text() == "frozen input"
    assert (run / "proposal.json").read_text() == "frozen proposal"


def test_run_cannot_escape_or_follow_directory_symlink(tmp_path, conversation):
    outside = tmp_path / "outside-run"
    outside.mkdir()
    with pytest.raises(ValueError, match="stay in"):
        store.attach_run(conversation, outside)
    linked = conversation.parent.parent / "linked-run"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        store.attach_run(conversation, linked)


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_download_bundle_retains_conversation_snapshot(tmp_path, engine):
    from vasp_slurm_agent import workflow

    source = tmp_path / "Si.cif"
    source.write_text("structure")
    folder = store.create(tmp_path / "runs", engine, source)
    run = tmp_path / "runs" / "calculation"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"engine": engine, "stages": []}))
    (run / "config.json").write_text("private configuration")
    store.attach_run(folder, run)
    with zipfile.ZipFile(workflow.bundle_run(run)) as archive:
        assert "conversation.json" in archive.namelist()
        assert "config.json" not in archive.namelist()
        assert json.loads(archive.read("conversation.json"))["conversation_id"] == folder.name
