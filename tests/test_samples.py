"""Bundled sample runs load as verified, placeholder-only results."""

import csv
import hashlib
import json
import re
import sys

from vasp_slurm_agent import cli
from vasp_slurm_agent.explanation import load_run_context
from vasp_slurm_agent.samples import install_samples, list_samples, sample_card
from vasp_slurm_agent.workflow import read_state

PERSONAL = re.compile(r"/Users/|/public1/home|/nobackup/proj|naiss|qiruic|kth\.se", re.IGNORECASE)


def test_samples_are_described_and_install_once(tmp_path):
    samples = list_samples()
    assert [item["id"] for item in samples] == ["srtio3-pbe-chain", "si-pbe-chain", "mgo-cell-relax", "fe-seed-comparison"]
    assert all(item["title"] and item["description"] for item in samples)
    installed = install_samples(tmp_path / "runs")
    assert [path.name for path in installed] == ["sample-srtio3-pbe-chain", "sample-si-pbe-chain", "sample-mgo-cell-relax", "sample-fe-seed-comparison"]
    assert install_samples(tmp_path / "runs") == []
    assert sample_card(installed[0])["title"].startswith("SrTiO3")
    assert sample_card(tmp_path) is None


def test_samples_verify_and_carry_no_personal_details(tmp_path):
    for path in install_samples(tmp_path / "runs"):
        state = read_state(path)
        assert state["status"] == "succeeded"
        assert state["remote_root"].startswith("/home/researcher/dft-agent-runs/")
        context = load_run_context(path)
        for index in range(1, len(state["stages"]) + 1):
            assert context["facts"][f"stage_{index}.accepted"]["value"] is True
        for file in path.rglob("*"):
            if file.is_file() and file.suffix != ".png":
                assert not PERSONAL.search(file.read_text(errors="replace")), file
        assert not (path / "results.zip").exists()
    assert any((tmp_path / "runs" / "sample-si-pbe-chain").rglob("bands.png"))


def test_sample_band_paths_follow_the_lattice_and_stay_smooth(tmp_path):
    # The 0.4.4 silicon sample showed monoclinic labels and 13 points; a sample must look like the textbook.
    from pymatgen.core import Structure
    from vasp_slurm_agent.numerical_defaults import BAND_PATH_MIN_SEGMENT_POINTS, high_symmetry_path

    checked = 0
    for path in install_samples(tmp_path / "runs"):
        for stage in read_state(path)["stages"]:
            if stage["name"] != "bands":
                continue
            band_path = stage["metadata"]["band_path"]
            assert min(end - start for start, end in band_path["segments"]) >= BAND_PATH_MIN_SEGMENT_POINTS, path.name
            structure = Structure.from_file(path / stage["folder"] / "inputs" / "POSCAR")
            assert [label for label in band_path["labels"] if label] == \
                [label for label in high_symmetry_path(structure, 60)["labels"] if label], path.name
            checked += 1
    assert checked >= 2


def test_cli_lists_and_copies_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dft-agent", "samples", "--list"])
    assert cli.main() == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    monkeypatch.setattr(sys, "argv", ["dft-agent", "samples", "--runs", str(tmp_path / "runs")])
    assert cli.main() == 0
    assert "sample-mgo-cell-relax" in capsys.readouterr().out
    assert cli.main() == 0
    assert "already" in capsys.readouterr().err


def test_srtio3_sample_keeps_dense_data_and_honest_sanitized_evidence(tmp_path):
    from pymatgen.core import Structure

    path, = install_samples(tmp_path / "runs", names=["srtio3-pbe-chain"])
    state = read_state(path)
    assert len(Structure.from_file(path / "source/input/SrTiO3.cif")) == 5
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands", "dos"]
    assert state["stages"][1]["metadata"]["parameters"]["mesh"] == [8, 8, 8]
    assert state["stages"][2]["result"]["band_gap"]["kpoint_count"] == 353
    assert not state["stages"][2]["result"]["scientific_accuracy_validated"]
    assert not any(file.name in {"POTCAR", "WAVECAR", "CHGCAR", "config.json", "worker.json", "results.zip"}
                   for file in path.rglob("*"))
    for stage in state["stages"]:
        output = path / stage["folder"] / "outputs"
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        for name, metadata in manifest.items():
            data = (output / name).read_bytes()
            assert len(data) == metadata["size"]
            assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            expected = stage["metadata"]["input_sha256"][name]
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
        assert (output / "vasprun.xml").is_file() and (output / "OUTCAR").is_file()
    with (path / "04_dos/outputs/dos.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2001
    for folder, plot in (("03_bands", "bands"), ("03_bands", "bands_dos"),
                         ("04_dos", "dos"), ("04_dos", "dos_elements")):
        for extension in ("png", "pdf", "svg"):
            assert (path / folder / "outputs" / f"{plot}.{extension}").is_file()
    provenance = json.loads((path / "archive_provenance.json").read_text())
    assert provenance["scientific_inputs_unchanged"] and provenance["original_run_unchanged"]
    for correction in provenance.get("derived_data_corrections", []):
        assert correction["raw_solver_files_unchanged"]
        for item in correction["files"]:
            assert hashlib.sha256((path / item["path"]).read_bytes()).hexdigest() == item["corrected_sha256"]
    for item in provenance["files"]:
        assert item["original_sha256"] != item["archive_sha256"]
        assert hashlib.sha256((path / item["path"]).read_bytes()).hexdigest() == item["archive_sha256"]
        assert not item["path"].endswith(("/POSCAR", "/INCAR", "/KPOINTS", "/input_hashes.sha256"))
    for file in path.rglob("*"):
        if file.is_file():
            content = file.read_bytes().lower()
            assert b"arrheniuscpu" not in content
            assert b"/software/sse2/" not in content


def test_srtio3_vaspkit_exports_match_receipts_and_archived_inputs(tmp_path):
    path, = install_samples(tmp_path / "runs", names=["srtio3-pbe-chain"])
    state = read_state(path)
    task_names = set()
    for stage in state["stages"]:
        postprocessing = stage.get("postprocessing")
        if not postprocessing:
            continue
        assert postprocessing["status"] == "complete"
        folder = path / postprocessing["folder"]
        receipt = json.loads((folder / "receipt.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["energy_reference"] == "provided_fermi"
        for task in receipt["tasks"]:
            task_names.add(task["name"])
            assert task["status"] == "complete" and task["version"] == "1.5.1"
            fermi_lines = (folder / task["folder"] / "FERMI_ENERGY.in").read_text().splitlines()
            assert fermi_lines[0] == "# Fermi energy reference (eV)"
            assert float(fermi_lines[1]) == receipt["fermi_energy_ev"]
            for group in ("outputs", "logs"):
                for name, expected in task[group].items():
                    data = (folder / task["folder"] / name).read_bytes()
                    assert len(data) == expected["size"]
                    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
            for name, expected in task["inputs"].items():
                data = (path / stage["folder"] / "outputs" / name).read_bytes()
                assert len(data) == expected["size"]
                assert hashlib.sha256(data).hexdigest() == expected["sha256"]
    assert task_names == {"bands", "projected_bands", "total_dos", "projected_dos"}
    provenance = json.loads((path / "archive_provenance.json").read_text())
    for export in provenance["postprocessing_exports"]:
        assert not export["input_hash_differences"]
        record = next(item for item in provenance["files"] if item["path"] == export["folder"] + "/receipt.json")
        assert record["original_sha256"] == export["source_receipt_sha256"]
        assert record["archive_sha256"] == export["archive_receipt_sha256"]
        assert hashlib.sha256((path / export["folder"] / "receipt.json").read_bytes()).hexdigest() == export["archive_receipt_sha256"]
