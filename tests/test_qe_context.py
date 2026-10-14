"""Result dialogue facts are tied to retained QE solver evidence."""
import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent import qe_workflow, qe_context
from test_qe import xml_output
from test_qe_workflow import FakeTransport, complete


@pytest.fixture
def accepted_qe(tmp_path):
    crystal = tmp_path / "Si.cif"
    Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=crystal)
    config = ClusterConfig(engine="qe", host="test.invalid", user="test", partition="cpu",
                           remote_root="/scratch/jobs", qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})
    root = tmp_path / "run"
    qe_workflow.prepare_plan(crystal, root, config, ["scf"])
    transport = FakeTransport()
    qe_workflow.advance(root, transport)
    complete(root, transport)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    xml_output(fixture, crystal)
    for name in ("data-file-schema.xml", "pw.out"):
        transport.files[name] = (fixture / name).read_bytes()
    transport.files["checkpoint.json"] = json.dumps({
        "scratch/dft_agent.save/data-file-schema.xml": {"size": len(transport.files["data-file-schema.xml"]), "sha256": qe_context._sha(fixture / "data-file-schema.xml")},
        "scratch/dft_agent.save/charge-density.dat": {"size": 4, "sha256": "0" * 64},
        "scratch/dft_agent.save/wfc1.dat": {"size": 4, "sha256": "0" * 64},
    }).encode()
    state = qe_workflow.advance(root, transport)
    assert state["status"] == "succeeded", state
    return root


def test_reparse_accepts_measured_facts_without_changing_files(accepted_qe):
    root = accepted_qe
    before = {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    context = qe_context.load_run_context(root)
    assert context["status"] == "succeeded", context
    assert context["facts"]["stage_1.accepted"]["value"]
    assert context["facts"]["stage_1.energy"]["value"] == pytest.approx(-272.11386245988)
    assert context["facts"]["stage_1.band_gap"]["value"] > 0
    assert any(a["id"] == "stage_1.CONTCAR" and a["status"] == "verified" for a in context["artifacts"])
    assert before == {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    assert qe_context.load_run_context(root)["context_sha256"] == context["context_sha256"]


@pytest.mark.parametrize("file", ["pw.out", "data-file-schema.xml", "execution.json", "runtime.json", "pseudo_manifest.json", "pw.in"])
def test_changed_raw_evidence_revokes_facts(accepted_qe, file):
    path = accepted_qe / "01_scf" / "outputs" / file
    path.write_bytes(path.read_bytes() + b"\nchanged")
    context = qe_context.load_run_context(accepted_qe)
    assert context["status"] == "needs_attention"
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert "stage_1.energy" not in context["facts"]


def test_cached_number_edit_cannot_become_a_fact(accepted_qe):
    root = accepted_qe
    state = json.loads((root / "run.json").read_text())
    result = state["stages"][0]["result"]
    result["final_energy_ev"] = -9999
    (root / "run.json").write_text(json.dumps(state))
    (root / "01_scf/outputs/result.json").write_text(json.dumps(result))
    context = qe_context.load_run_context(root)
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert "numerical" in context["facts"]["stage_1.availability"]["value"]


def test_verified_failure_reaches_model_without_cached_claims(accepted_qe):
    from vasp_slurm_agent import qe
    root = accepted_qe
    output = root / "01_scf/outputs"
    xml = output / "data-file-schema.xml"
    xml.write_text(xml.read_text().replace("<scf_conv><convergence_achieved>true", "<scf_conv><convergence_achieved>false"))
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    manifest[xml.name].update(sha256=qe_context._sha(xml), size=xml.stat().st_size)
    (output / "artifact_manifest.json").write_text(json.dumps(manifest))
    state = json.loads((root / "run.json").read_text())
    old = state["stages"][0]["result"]
    result = qe.analyze_outputs(output, "scf", output / "structure.cif")
    for key in ("input_identity_verified", "execution_verified", "pseudopotentials_verified", "scheduler_state"):
        result[key] = old[key]
    result["reason"] = "Untrusted cached diagnosis"
    state["status"] = "needs_attention"
    state["stages"][0].update(status="needs_attention", result=result)
    (root / "run.json").write_text(json.dumps(state))
    (output / "result.json").write_text(json.dumps(result))
    context = qe_context.load_run_context(root)
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert "stage_1.energy" not in context["facts"]
    assert "Electronic convergence" in context["facts"]["stage_1.failure_reason"]["value"]
    assert "Untrusted cached diagnosis" not in json.dumps(context)
    xml.write_bytes(xml.read_bytes() + b"\nchanged")
    assert "stage_1.failure_reason" not in qe_context.load_run_context(root)["facts"]


def test_changed_plan_and_source_revoke_acceptance(accepted_qe):
    (accepted_qe / "source" / "structure.cif").write_text("changed")
    context = qe_context.load_run_context(accepted_qe)
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert context["final_plan"] == {}


def test_private_locations_never_enter_model_facts(accepted_qe):
    context = qe_context.load_run_context(accepted_qe)
    text = json.dumps(context)
    assert "/scratch/jobs" not in text
    assert "/pseudo" not in text
    assert "test.invalid" not in text


def test_neb_endpoint_scope_reaches_result_dialogue_without_altering_evidence(tmp_path):
    crystal = tmp_path / "Si.cif"
    Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=crystal)
    config = ClusterConfig(engine="qe", host="test.invalid", user="test", partition="cpu",
                           remote_root="/scratch/jobs", qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})
    root = tmp_path / "run"
    qe_workflow.prepare_plan(crystal, root, config, ["neb"],
                             {"images": [str(crystal), str(crystal)], "num_of_images": 3, "path_thr": .08})
    transport = FakeTransport()
    qe_workflow.advance(root, transport)
    complete(root, transport)
    transport.files["neb.out"] = b"image energy (eV) error (eV/A) frozen\n1 -10 2.01 T\n2 -9.5 .06 F\n3 -10 1.98 T\nneb: convergence achieved\nJOB DONE.\n"
    assert qe_workflow.advance(root, transport)["status"] == "succeeded"
    before = {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    context = qe_context.load_run_context(root)
    facts = context["facts"]
    assert facts["stage_1.path_converged"]["value"] is True
    assert facts["stage_1.ionic_converged"]["value"] is False
    assert facts["stage_1.endpoint_relaxed"]["value"] is False
    assert facts["stage_1.endpoint_forces"]["value"] == [2.01, 1.98]
    assert facts["stage_1.barrier_scope"]["value"] == "unrelaxed_endpoints"
    assert any("unrelaxed endpoints" in warning for warning in context["limits"])
    assert before == {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}


def test_magnetic_hubbard_parameters_retain_physics_without_private_text():
    values = {"spin": "collinear", "site_labels": ["Fe1", "Fe2"],
              "starting_magnetization": {"Fe1": 0.5, "Fe2": -0.5, "/private/secret": 0.1},
              "hubbard": [{"label": "Fe1", "manifold": "3d", "U": 4.0, "J0": 0.8, "token": "secret-token"},
                          {"label": "Fe2", "manifold": "3d", "U": 4.0},
                          {"label": "/private", "manifold": "3d", "U": 4.0}],
              "hubbard_projector": "ortho-atomic", "images": ["/private/endpoint.cif"], "api_key": "secret-token",
              "namelists": {"system": {"nspin": 2, "starting_magnetization(1)": 0.5,
                                          "starting_magnetization(2)": -0.5, "angle1(1)": 90,
                                          "Hubbard_U(1)": 4.0, "input_dft": "HSE", "pseudo_dir": "/private/pseudo", "token": "secret-token"},
                            "electrons": {"conv_thr": 1e-10, "startingwfc": "/private/state"},
                            "control": {"prefix": "secret-token", "outdir": "/private"}}}
    result = qe_context._parameters(values)
    assert result["site_labels"] == ["Fe1", "Fe2"]
    assert result["starting_magnetization"] == {"Fe1": 0.5, "Fe2": -0.5}
    assert result["hubbard"] == [{"label": "Fe1", "manifold": "3d", "U": 4.0, "J0": 0.8},
                                 {"label": "Fe2", "manifold": "3d", "U": 4.0}]
    assert result["hubbard_projector"] == "ortho-atomic"
    assert result["namelists"]["system"]["hubbard_u(1)"] == 4.0
    assert result["namelists"]["system"]["input_dft"] == "HSE"
    assert result["namelists"]["electrons"] == {"conv_thr": 1e-10}
    serialized = json.dumps(result)
    assert "/private" not in serialized and "secret-token" not in serialized


def test_effective_frozen_input_defaults_are_visible_for_comparison(accepted_qe):
    context = qe_context.load_run_context(accepted_qe)
    native = context["facts"]["stage_1.parameter.namelists"]
    assert native["value"]["system"]["input_dft"] == "PBE"
    assert native["value"]["system"]["ecutwfc"] > 0
    assert native["value"]["electrons"]["conv_thr"] > 0
    assert native["source"] == "01_scf/outputs/qe-input.json"
    assert context["facts"]["stage_1.parameter.site_labels"]["value"] == ["Si", "Si"]
    assert len(context["facts"]["stage_1.parameter.mesh"]["value"]) == 3


def test_unsafe_and_nonfinite_physics_values_are_not_sent_to_model():
    cleaned = qe_context._parameters({"site_labels": ["/private"], "starting_magnetization": {"Fe": float("nan")},
        "hubbard": [{"label": "Fe", "manifold": "3d", "U": float("inf")}],
        "namelists": {"system": {"starting_magnetization(1)": float("nan"), "input_dft": "api-secret", "angle1(/private)": 90}}})
    assert "site_labels" not in cleaned
    assert cleaned["starting_magnetization"] == {} and cleaned["hubbard"] == []
    assert "namelists" not in cleaned


@pytest.fixture
def magnetic_pair(tmp_path):
    def build(second_parameters=None, second_energy=-9.99, same_seed=False, second_total=0, second_site_moment=-1.1):
        crystal = tmp_path / "Si.cif"
        Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=crystal)
        config = ClusterConfig(engine="qe", host="test.invalid", user="test", partition="cpu",
            remote_root="/scratch/jobs", qe_pseudo_root="/pseudo", qe_pseudopotentials={"Si": "Si.UPF"})
        root = tmp_path / "pair"
        seeds = [{"starting_magnetization": {"SiA": .5, "SiB": value}}
                 for value in [.5, .5 if same_seed else -.5]]
        seeds[1].update(second_parameters or {})
        qe_workflow.prepare_plan(crystal, root, config, ["scf", "scf"],
            {"spin": "collinear", "site_labels": ["SiA", "SiB"]}, stage_parameters=seeds)
        transport = FakeTransport()
        for index, energy in enumerate([-10, second_energy]):
            qe_workflow.advance(root, transport)
            complete(root, transport)
            fixture = tmp_path / f"magnetic-{index}"
            fixture.mkdir()
            xml_output(fixture, crystal)
            magnetic = f"""<magnetization><lsda>true</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit>
                <total>{2 if index == 0 else second_total}</total><absolute>2.4</absolute><Scalar_Site_Magnetic_Moments nat="2">
                <SiteMagnetization atom="1" species="Si">1.1</SiteMagnetization>
                <SiteMagnetization atom="2" species="Si">{1.1 if index == 0 else second_site_moment}</SiteMagnetization>
                </Scalar_Site_Magnetic_Moments></magnetization>"""
            xml = fixture / "data-file-schema.xml"
            xml.write_text(xml.read_text().replace("<etot>-10</etot>", f"<etot>{energy}</etot>")
                           .replace("</output>", magnetic + "</output>"))
            exx = "est. exchange err (dexx) = 2.0D-9 Ry\nEXX self-consistency reached\n" if seeds[index].get("functional") == "PBE0" else ""
            (fixture / "pw.out").write_text(f"! total energy = {energy * 2} Ry\n{exx}JOB DONE.\n")
            for name in ("data-file-schema.xml", "pw.out"):
                transport.files[name] = (fixture / name).read_bytes()
            state = qe_workflow.advance(root, transport)
            assert state["stages"][index]["status"] == "succeeded", state
        assert state["status"] == "succeeded"
        return root
    return build


def test_verified_hybrid_outer_loop_facts_reach_result_dialogue(magnetic_pair):
    root = magnetic_pair(second_parameters={"functional": "PBE0"})
    before = {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    facts = qe_context.load_run_context(root)["facts"]
    assert facts["stage_2.accepted"]["value"] is True
    assert facts["stage_2.exx_converged"]["value"] is True
    error = facts["stage_2.exx_exchange_error"]
    assert error["value"] == pytest.approx(2e-9)
    assert error["unit"] == "Ry"
    assert error["source"] == "02_scf/outputs/pw.out"
    assert "stage_1.exx_converged" not in facts
    assert "stage_1.exx_exchange_error" not in facts
    assert before == {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("key,value", [("converged_exx", False), ("exx_exchange_error_ry", 0)])
def test_cached_hybrid_convergence_edits_revoke_facts(magnetic_pair, key, value):
    root = magnetic_pair(second_parameters={"functional": "PBE0"})
    state = qe_workflow.read_state(root)
    state["stages"][1]["result"][key] = value
    qe_workflow._write(root / "run.json", state)
    qe_workflow._write(root / "02_scf/outputs/result.json", state["stages"][1]["result"])
    facts = qe_context.load_run_context(root)["facts"]
    assert facts["stage_2.accepted"]["value"] is False
    assert "numerical" in facts["stage_2.availability"]["value"]
    assert "stage_2.exx_converged" not in facts
    assert "stage_2.exx_exchange_error" not in facts


def test_verified_magnetic_comparison_uses_reparsed_energy_and_final_moments(magnetic_pair):
    root = magnetic_pair()
    before = {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    context = qe_context.load_run_context(root)
    facts = context["facts"]
    assert facts["comparison.status"]["value"] == "available"
    assert facts["comparison.group_1.lowest_tested_stage"]["value"] == "stage_1"
    assert facts["comparison.group_1.stage_1.relative_energy"]["value"] == 0
    delta = facts["comparison.group_1.stage_2.relative_energy"]
    assert delta["value"] == pytest.approx(.01 * qe_context.qe.HA_EV / 2 * 1000)
    assert delta["unit"] == "meV/atom"
    assert facts["stage_2.total_magnetization"]["value"] == 0
    assert facts["stage_2.site_moment.2"]["value"] == -1.1
    assert "does not establish" in facts["comparison.group_1.scope"]["value"]
    assert before == {str(p): qe_context._sha(p) for p in root.rglob("*") if p.is_file()}
    assert qe_context.load_run_context(root)["context_sha256"] == context["context_sha256"]


def test_equal_energies_do_not_choose_an_arbitrary_winner(magnetic_pair):
    facts = qe_context.load_run_context(magnetic_pair(second_energy=-10))["facts"]
    assert facts["comparison.group_1.lowest_tested_stages"]["value"] == ["stage_1", "stage_2"]
    assert "comparison.group_1.lowest_tested_stage" not in facts


def test_final_magnetic_state_can_differ_from_the_initial_seed(magnetic_pair):
    facts = qe_context.load_run_context(magnetic_pair(second_total=2, second_site_moment=1.1))["facts"]
    assert facts["comparison.status"]["value"] == "available"
    assert facts["stage_2.parameter.starting_magnetization"]["value"]["SiB"] == -.5
    assert facts["stage_2.total_magnetization"]["value"] == 2
    assert facts["stage_2.site_moment.2"]["value"] == 1.1
    assert not any("AFM" in str(item["value"]) for key, item in facts.items() if key.startswith("comparison."))


@pytest.mark.parametrize("parameters", [{"functional": "PBE0"}, {"ecutwfc": 80}, {"mesh": [3, 3, 3]}])
def test_different_settings_are_not_ranked(magnetic_pair, parameters):
    facts = qe_context.load_run_context(magnetic_pair(second_parameters=parameters))["facts"]
    assert facts["stage_1.accepted"]["value"] and facts["stage_2.accepted"]["value"]
    assert facts["comparison.status"]["value"] == "unavailable"
    assert not any("relative_energy" in key or "lowest_tested" in key for key in facts)


def test_repeated_identical_seeds_are_not_a_magnetic_order_search(magnetic_pair):
    facts = qe_context.load_run_context(magnetic_pair(same_seed=True))["facts"]
    assert facts["comparison.status"]["value"] == "unavailable"


@pytest.mark.parametrize("name", ["pw.in", "data-file-schema.xml", "pseudo_manifest.json"])
def test_changed_comparison_evidence_revokes_ranking(magnetic_pair, name):
    root = magnetic_pair()
    artifact = root / "02_scf/outputs" / name
    artifact.write_bytes(artifact.read_bytes() + b"\nchanged")
    facts = qe_context.load_run_context(root)["facts"]
    assert facts["comparison.status"]["value"] == "unavailable"
    assert not any("relative_energy" in key or "lowest_tested" in key for key in facts)


def test_cached_energy_tampering_cannot_change_the_magnetic_ranking(magnetic_pair):
    root = magnetic_pair()
    original = qe_context.load_run_context(root)
    state = qe_workflow.read_state(root)
    state["stages"][1]["result"]["final_energy_ev_per_atom"] = -9999
    qe_workflow._write(root / "run.json", state)
    qe_workflow._write(root / "02_scf/outputs/result.json", state["stages"][1]["result"])
    context = qe_context.load_run_context(root)
    assert context["context_sha256"] != original["context_sha256"]
    assert not context["facts"]["stage_2.accepted"]["value"]
    assert context["facts"]["comparison.status"]["value"] == "unavailable"
    assert not any("relative_energy" in key or "lowest_tested" in key for key in context["facts"])


def test_signature_retains_geometry_method_constraints_and_all_nonseed_bytes():
    original = b"&SYSTEM\n nspin = 2,\n starting_magnetization(1) = 0.5,\n ecutwfc = 60,\n lspinorb = .false.,\n occupations = 'smearing',\n/\nK_POINTS automatic\n5 5 5 0 0 0\nATOMIC_POSITIONS crystal\nFe 0 0 0\nHUBBARD (ortho-atomic)\nU Fe-3d 4\n"
    result = {"spin_mode": "collinear", "number_of_atoms": 2, "final_energy_ev_per_atom": -1}
    signature = lambda text, pseudo="a": qe_context._scf_comparison_signature(text, pseudo, result)[0]
    assert signature(original) == signature(original.replace(b"0.5,", b"-0.5,"))
    for before, after in ((b"nspin = 2", b"nspin = 1"), (b"60,", b"80,"),
                          (b".false.", b".true."), (b"'smearing'", b"'fixed'"),
                          (b"5 5 5", b"7 7 7"), (b"Fe 0 0 0", b"Fe 0.1 0 0"),
                          (b"U Fe-3d 4", b"U Fe-3d 5"),
                          (b" nspin = 2,", b" nspin = 2,\n tot_magnetization = 0,")):
        assert signature(original) != signature(original.replace(before, after))
    assert signature(original) != signature(original, "b")
    # A second assignment on the same line must never disappear with the seed.
    assert signature(original) != signature(original.replace(b"0.5,", b"-0.5, tot_magnetization = 0,"))


def test_seed_parsing_is_confined_to_system_and_preserves_other_text():
    original = b"&SYSTEM\n starting_magnetization(1) = .5,\n/\n&ELECTRONS\n starting_magnetization(1) = .5,\n/\n! starting_magnetization(1) = .5,\n"
    result = {"spin_mode": "collinear", "number_of_atoms": 2, "final_energy_ev_per_atom": -1}
    first = qe_context._scf_comparison_signature(original, "a", result)
    assert first[1] == ((1, .5),)
    outside = original.replace(b"&ELECTRONS\n starting_magnetization(1) = .5", b"&ELECTRONS\n starting_magnetization(1) = -.5")
    comment = original.replace(b"! starting_magnetization(1) = .5", b"! starting_magnetization(1) = -.5")
    assert qe_context._scf_comparison_signature(outside, "a", result)[0] != first[0]
    assert qe_context._scf_comparison_signature(comment, "a", result)[0] != first[0]


def test_reordering_identical_seed_assignments_is_not_a_new_magnetic_trial():
    first = b"&SYSTEM\n starting_magnetization(1) = .5,\n starting_magnetization(2) = -.5,\n/\n"
    reordered = b"&SYSTEM\n starting_magnetization(2) = -.5,\n starting_magnetization(1) = .5,\n/\n"
    result = {"spin_mode": "collinear", "number_of_atoms": 2, "final_energy_ev_per_atom": -1}
    assert qe_context._scf_comparison_signature(first, "a", result) == qe_context._scf_comparison_signature(reordered, "a", result)
