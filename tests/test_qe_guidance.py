import json
import shutil

from pymatgen.core import Lattice, Structure
import pytest

from vasp_slurm_agent import qe_context, qe_guidance, qe_workflow
from test_qe_context import accepted_qe


def test_official_qe_guidance_is_available_without_runs():
    context = qe_guidance.build_context()
    assert context["experience"] == []
    assert {item["id"] for item in context["guidance"]} == {
        "qe_inputs", "qe_spectra", "qe_phonons", "qe_neb", "qe_pairing", "qe_wannier_pairing"}
    assert all(item["checks"] and item["sources"] for item in context["guidance"])
    text = json.dumps(context)
    assert "INPUT_PW.html" in text and "INPUT_PH.html" in text and "INPUT_NEB.html" in text
    assert "docs.epw-code.org" in text and "VASP" not in text and "ENCUT" not in text


def test_verified_matching_structure_has_measured_facts_and_effective_method(accepted_qe):
    source = accepted_qe / "source/structure.cif"
    context = qe_guidance.build_context(source, runs_root=accepted_qe.parent)
    assert len(context["experience"]) == 1
    case = context["experience"][0]
    assert case["formula"] == "Si" and "source_structure" in case["match_criterion"]
    stage = case["stages"][0]
    assert stage["task"] == "scf"
    assert stage["method"]["namelists"]["system"]["input_dft"] == "PBE"
    assert stage["method"]["namelists"]["electrons"]["conv_thr"] > 0
    energy = next(fact for fact in stage["facts"] if fact["id"] == "stage_1.energy")
    assert energy["value"] == pytest.approx(-272.11386245988) and energy["unit"] == "eV"


def test_formula_equivalence_nonmatch_and_exclusion(accepted_qe):
    context = qe_guidance.build_context(formula="Si2", runs_root=accepted_qe.parent)
    assert len(context["experience"]) == 1
    assert "same_composition_only" in context["experience"][0]["match_criterion"]
    assert qe_guidance.build_context(formula="Ge", runs_root=accepted_qe.parent)["experience"] == []
    assert qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent, exclude_run=accepted_qe)["experience"] == []


def test_same_composition_different_geometry_is_not_a_structure_match(accepted_qe, tmp_path):
    source = tmp_path / "different.cif"
    Structure(Lattice.tetragonal(5.43, 10), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=source)
    assert qe_guidance.build_context(source, runs_root=accepted_qe.parent)["experience"] == []


@pytest.mark.parametrize("status", ["planned", "failed", "needs_attention"])
def test_incomplete_and_failed_runs_are_not_successful_cases(accepted_qe, status):
    state = qe_workflow.read_state(accepted_qe)
    state["status"] = status
    qe_workflow._write(accepted_qe / "run.json", state)
    assert qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent)["experience"] == []


def test_vasp_is_not_passed_to_qe_verifier(tmp_path, monkeypatch):
    run = tmp_path / "vasp"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"engine": "vasp", "status": "succeeded"}))
    monkeypatch.setattr(qe_context, "load_run_context", lambda *args: pytest.fail("Do not inspect VASP as QE"))
    assert qe_guidance.build_context(formula="Si", runs_root=tmp_path)["experience"] == []


@pytest.mark.parametrize("filename", ["01_scf/outputs/pw.out", "source/structure.cif"])
def test_changed_evidence_is_rejected(accepted_qe, filename):
    path = accepted_qe / filename
    path.write_bytes(path.read_bytes() + b"\nchanged")
    assert qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent)["experience"] == []


def test_private_metadata_never_enters_context_and_source_is_unchanged(accepted_qe):
    state = qe_workflow.read_state(accepted_qe)
    qe_workflow._write(accepted_qe / "proposal.json", {"goal": "private.customer.secret", "dialogue": [
        {"role": "user", "content": "/private/customer private-token"}],
        "execution": {"plan_sha256": state["plan_sha256"]}})
    before = {str(path): qe_context._sha(path) for path in accepted_qe.rglob("*") if path.is_file()}
    context = qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent)
    assert len(context["experience"]) == 1
    text = json.dumps(context)
    for private in ["private.customer", "/private", "private-token", "/scratch/jobs", "/pseudo", "test.invalid", str(accepted_qe), "Si.UPF"]:
        assert private not in text
    assert before == {str(path): qe_context._sha(path) for path in accepted_qe.rglob("*") if path.is_file()}


def test_bounded_cases_and_duplicate_evidence(accepted_qe, monkeypatch):
    for index in range(4):
        shutil.copytree(accepted_qe, accepted_qe.parent / f"copy-{index}")
    context = qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent, limit=1)
    assert len(context["experience"]) == 1
    assert len(json.dumps(context["experience"][0], ensure_ascii=False).encode()) <= qe_guidance.MAX_CASE_BYTES
    assert len(qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent)["experience"]) == 1
    monkeypatch.setattr(qe_context, "load_run_context", lambda *args: pytest.fail("Zero limit must skip retrieval"))
    assert qe_guidance.build_context(formula="Si", runs_root=accepted_qe.parent, limit=0)["experience"] == []


def test_directory_scan_and_verification_count_are_bounded(tmp_path, monkeypatch):
    for index in range(8):
        (tmp_path / f"case-{index}").mkdir()
    monkeypatch.setattr(qe_guidance, "MAX_SCAN_ENTRIES", 5)
    monkeypatch.setattr(qe_guidance, "MAX_RUNS", 2)
    inspected = []
    def inspect(root, *args):
        inspected.append(root)
        return None
    monkeypatch.setattr(qe_guidance, "_case", inspect)
    assert qe_guidance.build_context(formula="Si", runs_root=tmp_path)["experience"] == []
    assert len(inspected) == 2
