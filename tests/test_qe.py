"""QE adapter contracts. Synthetic fixtures test parsing, not scientific accuracy."""

import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent.qe import BOHR_ANGSTROM, HA_EV, RY_EV, analyze_outputs, namelist, prepare_inputs, validate_upf


@pytest.fixture
def crystal(tmp_path):
    path = tmp_path / "si.cif"
    Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]).to(filename=path)
    return path


def prepare(crystal, folder, task="scf", **parameters):
    return prepare_inputs(crystal, folder, task, parameters, {"Si": "Si.UPF"})


@pytest.mark.parametrize("parameters,message", [
    ({"functional": "HSE06"}, "hybrid-functional phonons"),
    ({"namelists": {"system": {"input_dft": "PBE0"}}}, "hybrid-functional phonons"),
    ({"hubbard": [{"label": "Si", "manifold": "3p", "U": 2}]}, "atomic projectors"),
    ({"hubbard_projector": "atomic", "hubbard": [
        {"label": "Si", "manifold": "3p", "U": 2},
        {"label": "Si", "manifold": "3s", "U": 1}]}, "one Hubbard channel"),
])
def test_phonon_rejects_native_unsupported_methods(crystal, tmp_path, parameters, message):
    with pytest.raises(ValueError, match=message):
        prepare(crystal, tmp_path / "ph", task="phonon", **parameters)
    assert not (tmp_path / "ph").exists()


def test_atomic_u_phonons_and_supported_u_scf_remain_available(crystal, tmp_path):
    hubbard = [{"label": "Si", "manifold": "3p", "U": 2, "J0": 0.1}]
    prepare(crystal, tmp_path / "ph", task="phonon", hubbard=hubbard, hubbard_projector="atomic")
    prepare(crystal, tmp_path / "scf", hubbard=hubbard)
    with pytest.raises(ValueError, match="J0.*noncollinear"):
        prepare(crystal, tmp_path / "nc", hubbard=hubbard, spin="noncollinear")
    prepare(crystal, tmp_path / "nc-u", hubbard=[{**hubbard[0], "J0": 0}], spin="noncollinear")


def test_pseudo_projector_requires_explicit_force_and_stress_exclusion(crystal, tmp_path):
    options = {"hubbard": [{"label": "Si", "manifold": "3p", "U": 2}], "hubbard_projector": "pseudo"}
    with pytest.raises(ValueError, match="forces or stress"):
        prepare(crystal, tmp_path / "bad", **options)
    options["namelists"] = {"control": {"tprnfor": False, "tstress": False}}
    prepare(crystal, tmp_path / "fixed", **options)
    with pytest.raises(ValueError, match="forces or stress"):
        prepare(crystal, tmp_path / "relax", task="relax", **options)


@pytest.mark.parametrize("pseudo_type", ["US", "PAW"])
def test_hybrid_restricts_force_and_noncollinear_pp_combinations(crystal, tmp_path, pseudo_type):
    potentials = {"Si": {"filename": "Si.UPF", "functional": "PBE", "pseudo_type": pseudo_type}}
    metadata = prepare_inputs(crystal, tmp_path / "scf", "scf", {"functional": "HSE06"}, potentials)
    assert metadata["namelists"]["control"]["tprnfor"] is False
    assert metadata["namelists"]["control"]["tstress"] is False
    with pytest.raises(ValueError, match="hybrid forces.*norm-conserving"):
        prepare_inputs(crystal, tmp_path / "forces", "scf", {"functional": "HSE06", "namelists": {"control": {"tprnfor": True}}}, potentials)
    for task, options in (("relax", {}), ("scf", {"spin": "noncollinear"})):
        with pytest.raises(ValueError, match="norm-conserving"):
            prepare_inputs(crystal, tmp_path / task, task, {"functional": "HSE06", **options}, potentials)
    potentials["Si"]["pseudo_type"] = "NC"
    prepare_inputs(crystal, tmp_path / "nc-relax", "relax", {"functional": "HSE06"}, potentials)


@pytest.mark.parametrize("task", ["scf", "relax"])
@pytest.mark.parametrize("spin", [
    {"soc": True}, {"spin": "noncollinear"},
    {"namelists": {"system": {"input_dft": "HSE", "noncolin": True, "lspinorb": True}}},
])
def test_noncollinear_hybrid_fixed_cell_defaults_to_no_stress(crystal, tmp_path, task, spin):
    parameters = {"functional": "HSE06", **spin}
    before = json.dumps(parameters, sort_keys=True)
    potentials = {"Si": {"filename": "Si.UPF", "pseudo_type": "NC", "fully_relativistic": True, "functional": "PBE"}}
    metadata = prepare_inputs(crystal, tmp_path / "fixed", task, parameters, potentials)
    assert metadata["namelists"]["control"]["tstress"] is False
    assert metadata["namelists"]["control"]["tprnfor"] is True
    assert metadata["namelists"]["system"]["input_dft"] == "HSE"
    assert metadata["namelists"]["system"]["noncolin"] is True
    assert "tstress = .false." in (tmp_path / "fixed" / "pw.in").read_text()
    assert json.dumps(parameters, sort_keys=True) == before


@pytest.mark.parametrize("task,parameters", [
    ("scf", {"namelists": {"control": {"tstress": True}}}),
    ("relax", {"cell_relax": True}),
    ("vc-relax", {"namelists": {"control": {"tstress": False}}}),
])
def test_noncollinear_hybrid_rejects_required_or_explicit_stress(crystal, tmp_path, task, parameters):
    potentials = {"Si": {"filename": "Si.UPF", "pseudo_type": "NC", "fully_relativistic": True}}
    with pytest.raises(ValueError, match="stress or cell relaxation"):
        prepare_inputs(crystal, tmp_path / "invalid", task, {"functional": "HSE06", "soc": True, **parameters}, potentials)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("parameters", [{"functional": "HSE06"}, {"soc": True}])
def test_stress_remains_enabled_for_supported_methods(crystal, tmp_path, parameters):
    potentials = {"Si": {"filename": "Si.UPF", "pseudo_type": "NC", "fully_relativistic": True}}
    metadata = prepare_inputs(crystal, tmp_path / "supported", "scf", parameters, potentials)
    assert metadata["namelists"]["control"]["tstress"] is True


def test_input_dft_cannot_silently_mix_lda_and_pbe_potentials(crystal, tmp_path):
    potentials = {"Si": {"filename": "Si.UPF", "functional": "SLA PW PBX PBC"}}
    with pytest.raises(ValueError, match="compatible LDA UPFs"):
        prepare_inputs(crystal, tmp_path / "lda-pbe", "scf", {"functional": "LDA"}, potentials)
    potentials["Si"]["functional"] = "SLA PZ NOGX NOGC"
    with pytest.raises(ValueError, match="compatible PBE UPFs"):
        prepare_inputs(crystal, tmp_path / "pbe-lda", "scf", {}, potentials)
    prepare_inputs(crystal, tmp_path / "lda", "scf", {"functional": "LDA"}, potentials)


@pytest.mark.parametrize("functional", ["B3LYP-V1R", "X3LYP", "GAUPBE", "XC-000I-000I-000L-000L-000L-000L"])
def test_native_functional_cannot_bypass_method_checks(crystal, tmp_path, functional):
    with pytest.raises(ValueError, match="input_dft choices need separate compatibility checks"):
        prepare(crystal, tmp_path / "unchecked", namelists={"system": {"input_dft": functional}})
    assert not (tmp_path / "unchecked").exists()


@pytest.mark.parametrize("pseudo_type,parameters", [
    ("US", {}), ("PAW", {}),
    ("NC", {"hubbard": [{"label": "Si", "manifold": "3p", "U": 2}]}),
])
def test_fully_relativistic_potential_requires_compatible_soc_method(crystal, tmp_path, pseudo_type, parameters):
    potentials = {"Si": {"filename": "Si.UPF", "pseudo_type": pseudo_type, "fully_relativistic": True}}
    with pytest.raises(ValueError, match="requires SOC.*no-SOC calculation"):
        prepare_inputs(crystal, tmp_path / "bad", "scf", parameters, potentials)
    assert not (tmp_path / "bad").exists()
    metadata = prepare_inputs(crystal, tmp_path / "soc", "scf", {**parameters, "soc": True}, potentials)
    assert metadata["namelists"]["system"]["lspinorb"] is True


def test_no_soc_remains_available_with_scalar_uspp_or_full_nc(crystal, tmp_path):
    for kind, fully in [("US", False), ("NC", True)]:
        metadata = prepare_inputs(crystal, tmp_path / kind, "scf", {},
            {"Si": {"filename": "Si.UPF", "pseudo_type": kind, "fully_relativistic": fully}})
        assert metadata["namelists"]["system"].get("lspinorb", False) is False


def test_magnetic_phonon_native_restrictions():
    from vasp_slurm_agent.qe import validate_compatibility
    potentials = {"Fe": {"pseudo_type": "PAW", "functional": "PBE"}}
    recipe = {"spin": "noncollinear", "starting_magnetization": {"Fe": 0.5}}
    with pytest.raises(ValueError, match="magnetic phonons.*PAW"):
        validate_compatibility("phonon", recipe, potentials, verified=True)
    validate_compatibility("phonon", {**recipe, "starting_magnetization": {"Fe": 0}}, potentials, verified=True)
    validate_compatibility("phonon", {**recipe, "spin": "collinear"}, potentials, verified=True)
    with pytest.raises(ValueError, match="constrained magnetization"):
        validate_compatibility("phonon", {"namelists": {"system": {"constrained_magnetization": "atomic"}}})


def xml_output(folder, crystal, calculation="scf", scf=True, ionic=True, energies=None, occupations=None, change_atom=False):
    structure = Structure.from_file(crystal)
    matrix = structure.lattice.matrix / BOHR_ANGSTROM
    xyz = structure.cart_coords / BOHR_ANGSTROM
    cell = "<cell>" + "".join(f"<a{i + 1}>{' '.join(map(str, v))}</a{i + 1}>" for i, v in enumerate(matrix)) + "</cell>"
    atoms = "<atomic_positions>" + "".join(f'<atom name="{"Ge" if change_atom else site.specie.symbol}">{" ".join(map(str, v))}</atom>' for site, v in zip(structure, xyz)) + "</atomic_positions>"
    atom_xml = f'<atomic_structure nat="2">{cell}{atoms}</atomic_structure>'
    energies = energies or [[-0.2, -0.1, 0.05], [-0.15, -0.11, 0.06]]
    occupations = occupations or [[1, 1, 0], [1, 1, 0]]
    bands = '<band_structure><fermi_energy>0</fermi_energy><nks>2</nks><nbnd>3</nbnd>'
    for energy, occ in zip(energies, occupations):
        bands += '<ks_energies><k_point>0 0 0</k_point><eigenvalues>' + ' '.join(map(str, energy)) + '</eigenvalues><occupations>' + ' '.join(map(str, occ)) + '</occupations></ks_energies>'
    bands += '</band_structure>'
    raw = f'''<qes:espresso xmlns:qes="urn:qe"><general_info><creator VERSION="7.4.1"/></general_info>
      <input><control_variables><calculation>{calculation}</calculation></control_variables>{atom_xml}</input>
      <output>{atom_xml}<convergence_info><scf_conv><convergence_achieved>{str(scf).lower()}</convergence_achieved></scf_conv><opt_conv><convergence_achieved>{str(ionic).lower()}</convergence_achieved></opt_conv></convergence_info>
      <total_energy><etot>-10</etot></total_energy><forces>0.001 0 0 -0.001 0 0</forces>{bands}</output><exit_status>0</exit_status></qes:espresso>'''
    (folder / "data-file-schema.xml").write_text(raw)
    (folder / "pw.out").write_text("!    total energy = -20.0 Ry\nEnd of band structure calculation\nJOB DONE.\n")


def test_poscar_orientation_survives_cif_display_roundtrip(tmp_path):
    from vasp_slurm_agent.qe import _same_structure
    source = tmp_path / "POSCAR"
    structure = Structure([[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
                          ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]])
    structure.to(filename=source, fmt="poscar")
    folder = tmp_path / "stage"
    prepare(source, folder)
    xml_output(folder, source)
    assert not _same_structure(structure, Structure.from_file(folder / "structure.cif"))
    result = analyze_outputs(folder, "scf", folder / "structure.cif")
    assert result["success"], result["reason"]
    # Rotating only the XML is still rejected, even though the physical cells match.
    xml_output(folder, folder / "structure.cif")
    result = analyze_outputs(folder, "scf", folder / "structure.cif")
    assert not result["success"]
    assert "submitted structure" in result["reason"]


def test_nested_qe_xml_artifact_uses_portable_path(crystal, tmp_path):
    folder = tmp_path / "stage"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    nested = folder / "scratch/dft_agent.save/data-file-schema.xml"
    nested.parent.mkdir(parents=True)
    (folder / "data-file-schema.xml").rename(nested)
    result = analyze_outputs(folder, "scf", folder / "structure.cif")
    assert result["success"], result["reason"]
    assert "scratch/dft_agent.save/data-file-schema.xml" in result["artifacts"]
    assert all("\\" not in path for path in result["artifacts"])


@pytest.mark.parametrize("change", ["metadata", "input", "species"])
def test_native_structure_must_match_frozen_pw_cards(crystal, tmp_path, change):
    import hashlib
    folder = tmp_path / "stage"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    metadata = json.loads((folder / "qe-input.json").read_text())
    if change == "metadata":
        metadata["native_structure"]["lattice_angstrom"][0][0] += .1
    elif change == "species":
        metadata["species"]["Si"] = "Ge"
    else:
        source = folder / "pw.in"
        source.write_text(source.read_text().replace("CELL_PARAMETERS angstrom", "CELL_PARAMETERS bohr"))
        metadata["input_sha256"]["pw.in"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (folder / "qe-input.json").write_text(json.dumps(metadata))
    result = analyze_outputs(folder, "scf")
    assert not result["success"] and "Native PW" in result["reason"]


def test_legacy_metadata_uses_supplied_structure(crystal, tmp_path):
    folder = tmp_path / "stage"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    metadata = json.loads((folder / "qe-input.json").read_text())
    metadata.pop("native_structure")
    (folder / "qe-input.json").write_text(json.dumps(metadata))
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"], result["reason"]


@pytest.mark.parametrize("events,accepted", [
    ("", False),
    ("Calculation (EXX) stopped after iteration # 10\n", False),
    ("EXX self-consistency reached\nEXX: now go back to refine exchange calculation\n", False),
    ("EXX self-consistency reached\nEXX convergence NOT achieved after 10 iterations: stopping\n", False),
    ("EXX: now go back to refine exchange calculation\nest. exchange err (dexx) = 2.0D-9 Ry\nEXX self-consistency reached\n", True),
])
def test_hybrid_requires_final_outer_loop_convergence(crystal, tmp_path, events, accepted):
    folder = tmp_path / "hse"
    prepare(crystal, folder, functional="HSE06")
    xml_output(folder, crystal, scf=True)
    (folder / "pw.out").write_text("convergence has been achieved in 10 iterations\n" + events + "JOB DONE.\n")
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"] is accepted
    if "NOT achieved" in events:
        assert result["converged_electronic"] is not True
        assert "incomplete convergence" in result["reason"]
        return
    assert result["converged_exx"] is accepted
    assert result["converged_electronic"] is accepted
    if accepted:
        assert result["exx_exchange_error_ry"] == pytest.approx(2e-9)
    else:
        assert "EXX outer-loop convergence" in result["reason"]


@pytest.mark.parametrize("pseudo_type,has_forces", [("US", False), ("PAW", False), ("NC", True)])
def test_hybrid_results_use_verified_upf_type_for_forces(crystal, tmp_path, pseudo_type, has_forces):
    folder = tmp_path / "hybrid"
    prepare_inputs(crystal, folder, "scf", {"functional": "HSE06"},
        {"Si": {"filename": "Si.UPF", "pseudo_type": "NC"}})
    # The runtime header receipt takes precedence over planned metadata.
    (folder / "pseudo_manifest.json").write_text(json.dumps({"Si": {"pseudo_type": pseudo_type}}))
    xml_output(folder, crystal)
    (folder / "pw.out").write_text("EXX self-consistency reached\nJOB DONE.\n")
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"]
    assert result["final_energy_ev"] == pytest.approx(-10 * HA_EV)
    assert ("final_max_force_ev_angstrom" in result) is has_forces
    assert ("forces.csv" in result["artifacts"]) is has_forces
    if not has_forces:
        assert any("Forces were omitted" in warning for warning in result["warnings"])


def test_disabled_forces_and_stress_do_not_expose_xml_placeholders(crystal, tmp_path):
    folder = tmp_path / "scf"
    prepare(crystal, folder, namelists={"control": {"tprnfor": False, "tstress": False}})
    xml_output(folder, crystal)
    path = folder / "data-file-schema.xml"
    path.write_text(path.read_text().replace("</output>", "<stress>0 0 0 0 0 0 0 0 0</stress></output>"))
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"]
    assert "final_max_force_ev_angstrom" not in result
    assert "pressure_gpa" not in result
    assert "forces.csv" not in result["artifacts"]
    assert "stress.csv" not in result["artifacts"]


@pytest.mark.parametrize("functional", ["HSE06", "PBE"])
def test_relaxation_energy_history_counts_completed_ionic_steps(crystal, tmp_path, functional):
    folder = tmp_path / "relax"
    prepare(crystal, folder, task="relax", functional=functional)
    xml_output(folder, crystal, calculation="relax")
    lines = []
    for total in (-19.5, -20):
        if functional == "HSE06":
            lines += [f"! total energy = {total + .2} Ry", f"! total energy = {total + .1} Ry",
                      f"!! total energy = {total} Ry", "EXX self-consistency reached"]
        else:
            lines.append(f"! total energy = {total} Ry")
    (folder / "pw.out").write_text("\n".join([*lines, "JOB DONE.", ""]))
    result = analyze_outputs(folder, "relax", crystal)
    assert result["success"], result["reason"]
    assert result["ionic_steps_count"] == 2
    history = np.loadtxt(folder / "relax_energy.csv", delimiter=",", skiprows=1)
    assert history[:, 0] == pytest.approx([1, 2])
    assert history[:, 1] == pytest.approx(np.asarray([-19.5, -20]) * RY_EV)
    assert history[-1, 1] == pytest.approx(result["final_energy_ev"])


def test_pw_inputs_and_density(crystal, tmp_path):
    folder = tmp_path / "stage"
    meta = prepare(crystal, folder)
    text = (folder / "pw.in").read_text()
    assert "ecutwfc = 60.0" in text
    assert "CELL_PARAMETERS angstrom" in text
    assert "Si.UPF" in text and "POTCAR" not in text
    assert meta["mesh"] == [5, 5, 5]
    assert meta["executable"] == "pw.x"
    assert not meta["requires_restart"]
    bigger = Structure.from_file(crystal)
    bigger.make_supercell([2, 2, 2])
    bigger.to(filename=tmp_path / "large.cif")
    assert prepare(tmp_path / "large.cif", tmp_path / "large")["mesh"] == [3, 3, 3]


@pytest.mark.parametrize("native", [
    {"lda_plus_u": True, "hubbard_u(1)": 5.0},
    {"LDA_PLUS_U_KIND": 1},
    {"hubbard_j0(1)": 0.8},
])
def test_native_hubbard_cannot_bypass_typed_card(crystal, tmp_path, native):
    with pytest.raises(ValueError, match="Hubbard card"):
        prepare(crystal, tmp_path / "legacy", namelists={"SYSTEM": native})


def test_hubbard_card_remains_available(crystal, tmp_path):
    folder = tmp_path / "card"
    prepare(crystal, folder, hubbard=[{"label": "Si", "manifold": "3p", "U": 2.0, "J0": 0.1}])
    text = (folder / "pw.in").read_text()
    assert "HUBBARD (ortho-atomic)" in text and "U Si-3p 2" in text and "J0 Si-3p 0.1" in text


def test_vacuum_mesh_and_override(tmp_path):
    source = tmp_path / "layer.cif"
    Structure(Lattice.orthorhombic(3, 3, 22), ["Si"], [[0, 0, 0.5]]).to(filename=source)
    assert prepare(source, tmp_path / "auto")["mesh"][2] == 1
    assert prepare(source, tmp_path / "explicit", mesh=[2, 2, 2])["mesh"] == [2, 2, 2]


@pytest.mark.parametrize("value", ["PBE'\n/\nATOMIC_POSITIONS", "../../elsewhere", "$(rm)", "a;b", float("nan"), [1, 2]])
def test_namelist_rejects_injection(value):
    with pytest.raises(ValueError):
        namelist("system", {"input_dft": value})


@pytest.mark.parametrize("key", ["prefix", "outdir", "nat", "filband", "wfcdir"])
def test_native_cannot_redirect_workflow(crystal, tmp_path, key):
    with pytest.raises(ValueError, match="managed"):
        prepare(crystal, tmp_path / "bad", namelists={"system": {key: "other"}})


@pytest.mark.parametrize("mapping", [{}, {"Si": "../Si.UPF"}, {"Si": "Si.UPF;pwd"}, {"Si": {"filename": "Ge.UPF", "element": "Ge"}}])
def test_upf_mapping_checks(crystal, tmp_path, mapping):
    with pytest.raises(ValueError):
        prepare_inputs(crystal, tmp_path / "stage", "scf", {}, mapping)


def test_soc_requires_verified_potential(crystal, tmp_path):
    with pytest.raises(ValueError, match="fully relativistic"):
        prepare(crystal, tmp_path / "bad", soc=True)
    prepare_inputs(crystal, tmp_path / "soc", "scf", {"soc": True}, {"Si": {"filename": "Si.rel.UPF", "element": "Si", "relativistic": "full"}})
    text = (tmp_path / "soc" / "pw.in").read_text()
    assert "noncolin = .true." in text and "lspinorb = .true." in text


def test_upf_header(crystal, tmp_path):
    path = tmp_path / "Si.UPF"
    path.write_text('<UPF><PP_HEADER element="Si" relativistic="full" has_so="T" wfc_cutoff="45" rho_cutoff="360"/></UPF>')
    metadata = validate_upf(path, "Si", require_soc=True)
    assert metadata["fully_relativistic"] and metadata["ecutrho"] == 360
    assert len(metadata["sha256"]) == 64
    with pytest.raises(ValueError, match="mismatch"):
        validate_upf(path, "Ge")


def test_species_magnetism_hubbard(tmp_path):
    path = tmp_path / "fe.cif"
    Structure(Lattice.cubic(4), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(filename=path)
    prepare_inputs(path, tmp_path / "afm", "scf", {"spin": "collinear", "site_labels": ["Fe1", "Fe2"], "starting_magnetization": {"Fe1": 0.5, "Fe2": -0.5}, "hubbard": [{"label": "Fe1", "manifold": "3d", "U": 4}, {"label": "Fe2", "manifold": "3d", "U": 4}]}, {"Fe": "Fe.UPF"})
    text = (tmp_path / "afm" / "pw.in").read_text()
    assert "nspin = 2" in text
    assert "starting_magnetization(2) = -0.5" in text
    assert "U Fe2-3d 4" in text
    with pytest.raises(ValueError, match="starting_magnetization"):
        prepare_inputs(path, tmp_path / "bad", "scf", {"spin": "collinear"}, {"Fe": "Fe.UPF"})


def test_hybrid_scf_and_unsupported_nscf(crystal, tmp_path):
    prepare(crystal, tmp_path / "hse", functional="HSE06")
    text = (tmp_path / "hse" / "pw.in").read_text()
    assert "input_dft = 'HSE'" in text and "screening_parameter = 0.106" in text
    with pytest.raises(ValueError, match="Hybrid"):
        prepare(crystal, tmp_path / "bands", task="bands", functional="HSE06")


@pytest.mark.parametrize("task,stem", [("dos", "dos"), ("pdos", "projwfc"), ("ph", "ph"), ("q2r", "q2r"), ("matdyn", "matdyn")])
def test_postprocessor_inputs(crystal, tmp_path, task, stem):
    meta = prepare_inputs(crystal, tmp_path / task, task, {"qpoints": [[0, 0, 0], [0.5, 0, 0]]}, {})
    assert meta["input_file"] == stem + ".in"
    assert meta["requires_restart"]
    assert (tmp_path / task / (stem + ".in")).stat().st_size > 30


def test_qmesh_and_partial_phonon_block(crystal, tmp_path):
    meta = prepare_inputs(crystal, tmp_path / "ph", "ph", {"qmesh": [2, 2, 2]}, {})
    assert meta["phonon_grid"] == [2, 2, 2]
    assert "nq1 = 2" in (tmp_path / "ph" / "ph.in").read_text()
    with pytest.raises(ValueError, match="Partial"):
        prepare_inputs(crystal, tmp_path / "bad", "ph", {"namelists": {"inputph": {"last_irr": 1}}}, {})


def test_bands_explicit_path(crystal, tmp_path):
    meta = prepare(crystal, tmp_path / "bands", task="bands", kpoints=[[0, 0, 0], [0.5, 0, 0]])
    assert meta["band_path"]["distance_inv_angstrom"][-1] == pytest.approx(np.pi / 5.43)
    assert "K_POINTS crystal\n2" in (tmp_path / "bands" / "pw.in").read_text()


def test_neb_inputs_and_endpoint_checks(crystal, tmp_path):
    final = Structure.from_file(crystal)
    final.translate_sites([1], [0.05, 0, 0])
    final.to(filename=tmp_path / "final.cif")
    meta = prepare(crystal, tmp_path / "neb", task="neb", images=[str(crystal), str(tmp_path / "final.cif")])
    text = (tmp_path / "neb" / "neb.in").read_text()
    assert text.count("FIRST_IMAGE") == 1 and text.count("LAST_IMAGE") == 1
    assert meta["neb_num_images"] == 5
    final.scale_lattice(final.volume * 1.2)
    final.to(filename=tmp_path / "wrong.cif")
    with pytest.raises(ValueError, match="share the cell"):
        prepare(crystal, tmp_path / "bad", task="neb", images=[str(crystal), str(tmp_path / "wrong.cif")])


def test_parse_scf_units_gap_and_structure(crystal, tmp_path):
    folder = tmp_path / "scf"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"], result
    assert result["energy_ev"] == pytest.approx(-10 * HA_EV)
    assert result["final_max_force_ev_angstrom"] == pytest.approx(0.001 * HA_EV / BOHR_ANGSTROM)
    assert result["band_gap_ev"] == pytest.approx(0.15 * HA_EV)
    assert "final.cif" in result["artifacts"] and (folder / "CONTCAR").is_file()
    assert result["scientific_accuracy_validated"] is False


def test_magnetization_and_stress_from_xml(crystal, tmp_path):
    from vasp_slurm_agent.qe import HA_BOHR3_GPA
    folder = tmp_path / "magnetic"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    path = folder / "data-file-schema.xml"
    magnetic = """<magnetization><lsda>true</lsda><noncolin>false</noncolin><spinorbit>false</spinorbit>
    <total>0.2</total><absolute>3.5</absolute><Scalar_Site_Magnetic_Moments nat="2">
    <SiteMagnetization atom="1" species="Si">1.4</SiteMagnetization>
    <SiteMagnetization atom="2" species="Si">-1.2</SiteMagnetization>
    </Scalar_Site_Magnetic_Moments></magnetization>
    <stress>0.0001 0 0 0 0.0002 0 0 0 0.0003</stress>"""
    path.write_text(path.read_text().replace("</output>", magnetic + "</output>"))
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"], result
    assert result["total_magnetization_mu_b"] == 0.2
    assert result["absolute_magnetization_mu_b"] == 3.5
    assert result["site_magnetization"][1]["moment_mu_b"] == -1.2
    assert result["pressure_gpa"] == pytest.approx(0.0002 * HA_BOHR3_GPA)
    assert "magnetization.csv" in result["artifacts"] and "stress.csv" in result["artifacts"]
    path.write_text(path.read_text().replace("<total>0.2</total>", "<total_vec>0.3 0.4 0</total_vec>"))
    result = analyze_outputs(folder, "scf", crystal)
    assert result["total_magnetization_mu_b"] == 0.5
    assert result["total_magnetization_vector_mu_b"] == [0.3, 0.4, 0]


def test_incomplete_magnetic_output_rejected(crystal, tmp_path):
    folder = tmp_path / "magnetic"
    prepare(crystal, folder)
    xml_output(folder, crystal)
    path = folder / "data-file-schema.xml"
    path.write_text(path.read_text().replace("</output>", "<magnetization><total_vec>0.2 0.3</total_vec></magnetization></output>"))
    result = analyze_outputs(folder, "scf", crystal)
    assert not result["success"] and "three components" in result["reason"]


def test_requested_magnetism_needs_magnetic_evidence(crystal, tmp_path):
    folder = tmp_path / "magnetic"
    prepare(crystal, folder, spin="collinear", starting_magnetization={"Si": 0.1})
    xml_output(folder, crystal)
    result = analyze_outputs(folder, "scf", crystal)
    assert not result["success"] and "no magnetization" in result["reason"]
    path = folder / "data-file-schema.xml"
    path.write_text(path.read_text().replace("</output>", "<magnetization><lsda>false</lsda><total>0</total></magnetization></output>"))
    result = analyze_outputs(folder, "scf", crystal)
    assert not result["success"] and "spin treatment" in result["reason"]


@pytest.mark.parametrize("case,reason", [("scf", "Electronic"), ("ionic", "Ionic"), ("mismatch", "structure"), ("truncated", "no element"), ("done_only", "missing")])
def test_reject_false_success(crystal, tmp_path, case, reason):
    folder = tmp_path / "relax"
    prepare(crystal, folder, task="relax")
    xml_output(folder, crystal, "relax", scf=case != "scf", ionic=case != "ionic", change_atom=case == "mismatch")
    if case == "truncated":
        (folder / "data-file-schema.xml").write_text("<espresso><output>")
    if case == "done_only":
        (folder / "data-file-schema.xml").unlink()
    result = analyze_outputs(folder, "relax", crystal)
    assert not result["success"] and reason.lower() in result["reason"].lower(), result


def test_nscf_not_scf_convergence(crystal, tmp_path):
    folder = tmp_path / "nscf"
    prepare(crystal, folder, task="nscf")
    xml_output(folder, crystal, "nscf", scf=False)
    result = analyze_outputs(folder, "nscf", crystal)
    assert result["success"], result
    assert result["final_energy_ev"] is None
    assert "final_energy_ev_per_atom" not in result and "final_max_force_ev_angstrom" not in result
    assert "forces.csv" not in result["artifacts"] and "stress.csv" not in result["artifacts"]
    (folder / "pw.out").write_text("End of band structure calculation\nc_bands: 2 eigenvalues not converged\nJOB DONE.\n")
    result = analyze_outputs(folder, "nscf", crystal)
    assert not result["success"] and "eigenvalues" in result["reason"]
    assert not (folder / "final.cif").exists()


def test_nscf_end_marker_does_not_clear_final_eigenvalue_failure(crystal, tmp_path):
    folder = tmp_path / "nscf"
    prepare(crystal, folder, task="nscf")
    xml_output(folder, crystal, "nscf", scf=True)
    # QE 7.4.1 non_scf.f90 prints the end marker and sets conv_elec true
    # even when diag_bands exhausted its retries at individual k points.
    (folder / "pw.out").write_text("Computing kpt #: 92 of 679\nc_bands: 1 eigenvalues not converged\n"
                                    "Computing kpt #: 679 of 679\nethr = 1.25E-10\nEnd of band structure calculation\nJOB DONE.\n")
    result = analyze_outputs(folder, "nscf", crystal)
    assert result["success"] is False and "eigenvalues" in result["reason"]


def test_partial_occupations_not_false_gap(crystal, tmp_path):
    folder = tmp_path / "scf"
    prepare(crystal, folder)
    xml_output(folder, crystal, occupations=[[1, 0.6, 0.4], [1, 0.7, 0.3]])
    result = analyze_outputs(folder, "scf", crystal)
    assert result["success"] and result["band_gap"]["metallic_or_smeared"]


def test_dos_requires_numeric_output(crystal, tmp_path):
    folder = tmp_path / "dos"
    prepare_inputs(crystal, folder, "dos", {}, {})
    (folder / "dos.out").write_text("JOB DONE.\n")
    assert not analyze_outputs(folder, "dos")["success"]
    (folder / "dos.dat").write_text("# E DOS integrated\n-1 0.1 1\n0 0.2 2\n1 0.1 3\n")
    result = analyze_outputs(folder, "dos")
    assert result["success"] and "dos.csv" in result["artifacts"]


def test_phonon_modes_and_missing_grid_matrix(crystal, tmp_path):
    folder = tmp_path / "ph"
    prepare_inputs(crystal, folder, "phonon", {"qmesh": [2, 2, 2]}, {})
    (folder / "dft_agent.dyn0").write_text("2 2 2\n2\n0 0 0\n0.5 0 0\n")
    (folder / "dft_agent.dyn1").write_text("Dynamical matrix file\n")
    (folder / "ph.out").write_text("Convergence has been achieved\n" + "".join(f"freq ({i}) = 1.0 [THz] = {i * 10}.0 [cm-1]\n" for i in range(1, 7)) * 2 + "JOB DONE.\n")
    assert not analyze_outputs(folder, "phonon")["success"]
    (folder / "dft_agent.dyn2").write_text("Dynamical matrix file\n")
    assert analyze_outputs(folder, "phonon")["success"]


def test_matdyn_data(crystal, tmp_path):
    folder = tmp_path / "matdyn"
    prepare_inputs(crystal, folder, "matdyn", {"qpoints": [[0, 0, 0], [0.5, 0, 0]]}, {})
    (folder / "matdyn.out").write_text("JOB DONE.\n")
    (folder / "phonon.freq").write_text("&plot nbnd=6, nks=2 /\n0 0 0\n0 0 0 100 100 100\n0.5 0 0\n10 20 30 110 120 130\n")
    result = analyze_outputs(folder, "matdyn")
    assert result["success"] and result["qpoint_count"] == 2, result
    assert (folder / "phonon_bands.csv").is_file()


def test_matdyn_fixed_width_weights_and_multiline_modes(tmp_path):
    crystal = tmp_path / "fcc.cif"
    Structure(Lattice([[0, 2, 2], [2, 0, 2], [2, 2, 0]]), ['Al'] * 3,
              [[0, 0, 0], [.25, .25, .25], [.5, .5, .5]]).to(filename=crystal)
    folder = tmp_path / "matdyn"
    requested = [[0, 0, 0], [.125, .25, 0]]
    prepare_inputs(crystal, folder, "matdyn", {"qpoints": requested}, {})
    structure = Structure.from_file(folder / "structure.cif")
    points = np.array(requested) @ structure.lattice.reciprocal_lattice.matrix * np.linalg.norm(structure.lattice.matrix[0]) / (2 * np.pi)
    rows = ['&plot nbnd=9, nks=2 /']
    for point, weight in zip(points, ['  0.000000', '**********']):
        rows.append(' ' * 10 + ''.join(f'{number:10.6f}' for number in point) + weight)
        rows.extend([''.join(f'{number:10.4f}' for number in [-1234.5678, -1234.5678, 0, 10, 20, 30]),
                     ''.join(f'{number:10.4f}' for number in [40, 50, 60])])
    (folder / 'matdyn.out').write_text('JOB DONE.\n')
    (folder / 'phonon.freq').write_text('\n'.join(rows) + '\n')
    result = analyze_outputs(folder, 'matdyn')
    assert result['success'], result
    assert result['qpoint_count'] == 2
    assert result['min_frequency_cm1'] == -1234.5678
    assert result['imaginary_mode_count'] == 4
    csv = np.loadtxt(folder / 'phonon_bands.csv', delimiter=',', skiprows=1)
    assert np.allclose(csv[:, 1:4], points, atol=1e-6)
    assert 'qx_2pi_alat' in (folder / 'phonon_bands.csv').read_text().splitlines()[0]
    assert 'mode_9_cm1' in (folder / 'phonon_bands.csv').read_text().splitlines()[0]


@pytest.mark.parametrize('problem', ['frequency_overflow', 'nan_frequency', 'missing_mode',
                                    'coordinate_overflow', 'wrong_path', 'extra_row', 'wrong_mode_count'])
def test_matdyn_rejects_damaged_frequencies_or_coordinates(crystal, tmp_path, problem):
    folder = tmp_path / 'matdyn'
    prepare_inputs(crystal, folder, 'matdyn', {'qpoints': [[0, 0, 0], [.5, 0, 0]]}, {})
    (folder / 'matdyn.out').write_text('JOB DONE.\n')
    rows = ['&plot nbnd=6, nks=2 /', ' ' * 10 + ''.join(f'{value:10.6f}' for value in [0, 0, 0]) + '**********',
            ''.join(f'{value:10.4f}' for value in [0, 0, 0, 100, 100, 100]),
            ' ' * 10 + ''.join(f'{value:10.6f}' for value in [.5, 0, 0]) + '  0.000000',
            ''.join(f'{value:10.4f}' for value in [10, 20, 30, 110, 120, 130])]
    if problem == 'frequency_overflow': rows[2] = '**********' + rows[2][10:]
    elif problem == 'nan_frequency': rows[2] = '       NaN' + rows[2][10:]
    elif problem == 'missing_mode': rows[2] = rows[2][10:]
    elif problem == 'coordinate_overflow': rows[1] = rows[1][:20] + '**********' + rows[1][30:]
    elif problem == 'wrong_path': rows[3] = rows[3].replace('0.500000', '0.400000')
    elif problem == 'extra_row': rows.append('0 0 0')
    else: rows[0] = rows[0].replace('nbnd=6', 'nbnd=3')
    (folder / 'phonon.freq').write_text('\n'.join(rows) + '\n')
    result = analyze_outputs(folder, 'matdyn')
    assert not result['success'], result


def test_neb_convergence_and_barrier(crystal, tmp_path):
    folder = tmp_path / "neb"
    prepare(crystal, folder, task="neb", images=[str(crystal), str(crystal)], num_of_images=3)
    table = "image energy (eV) error (eV/A) frozen\n1 -10.0 0.01 T\n2 -9.5 0.02 F\n3 -9.8 0.01 T\n"
    (folder / "neb.out").write_text(table + "neb: convergence achieved in 4 iterations\nJOB DONE.\n")
    result = analyze_outputs(folder, "neb")
    assert result["success"], result
    assert result["path_converged"] and result["converged_ionic"]
    assert result["endpoint_relaxed"]
    assert result["endpoint_forces_ev_angstrom"] == [0.01, 0.01]
    assert result["barrier_scope"] == "relaxed_endpoints"
    assert result["barrier_forward_ev"] == pytest.approx(0.5)
    assert result["barrier_reverse_ev"] == pytest.approx(0.3)
    (folder / "neb.out").write_text(table.replace("0.02 F", "0.2 F") + "neb: convergence achieved in 4 iterations\nJOB DONE.\n")
    assert not analyze_outputs(folder, "neb")["success"]


def test_neb_unrelaxed_endpoints_do_not_claim_full_ionic_convergence(crystal, tmp_path):
    folder = tmp_path / "neb"
    prepare(crystal, folder, task="neb", images=[str(crystal), str(crystal)], num_of_images=3, path_thr=0.08)
    table = "image energy (eV) error (eV/A) frozen\n1 -10.0 2.01 T\n2 -9.5 0.06 F\n3 -9.8 1.98 T\n"
    (folder / "neb.out").write_text(table + "neb: convergence achieved in 4 iterations\nJOB DONE.\n")
    result = analyze_outputs(folder, "neb")
    assert result["success"] and result["path_converged"], result
    assert result["converged_ionic"] is False
    assert result["endpoint_relaxed"] is False
    assert result["endpoint_forces_ev_angstrom"] == [2.01, 1.98]
    assert result["endpoint_force_threshold_ev_angstrom"] == 0.08
    assert result["barrier_forward_ev"] == pytest.approx(0.5)
    assert result["barrier_scope"] == "unrelaxed_endpoints"
    assert any("unrelaxed endpoints" in warning for warning in result["warnings"])


def test_neb_endpoint_threshold_requires_input_evidence(tmp_path):
    (tmp_path / "neb.out").write_text("image energy (eV) error (eV/A) frozen\n1 -10 0.01 T\n2 -9.5 0.02 F\n3 -10 0.01 T\nneb: convergence achieved\nJOB DONE.\n")
    result = analyze_outputs(tmp_path, "neb")
    assert result["success"] and result["path_converged"]
    assert result["endpoint_relaxed"] is None and result["converged_ionic"] is None
    assert result["barrier_scope"] == "unverified_endpoints"


def test_neb_optimizes_endpoints_when_requested(crystal, tmp_path):
    folder = tmp_path / "neb"
    prepare(crystal, folder, task="neb", images=[str(crystal), str(crystal)], num_of_images=3,
            namelists={"path": {"first_last_opt": True, "ci_scheme": "auto"}})
    assert "first_last_opt = .true." in (folder / "neb.in").read_text()
    table = "image energy (eV) error (eV/A) frozen\n1 -10 0.01 F\n2 -9.5 0.02 F\n3 -10 0.01 F\n"
    (folder / "neb.out").write_text(table + "neb: convergence achieved\nJOB DONE.\n")
    result = analyze_outputs(folder, "neb")
    assert result["success"] and result["endpoint_relaxed"] and result["converged_ionic"]


def test_unknown_parameters_rejected(crystal, tmp_path):
    with pytest.raises(ValueError, match="Unsupported QE parameters: encut"):
        prepare(crystal, tmp_path / "bad", encut=520)
    with pytest.raises(ValueError, match="requires collinear"):
        prepare(crystal, tmp_path / "magnetic", starting_magnetization={"Si": 0.5})


def test_charge_density_cube(crystal, tmp_path):
    folder = tmp_path / "pp"
    metadata = prepare_inputs(crystal, folder, "pp", {}, {})
    assert metadata["executable"] == "pp.x"
    assert "plot_num = 0" in (folder / "pp.in").read_text()
    (folder / "pp.out").write_text("JOB DONE.\n")
    assert not analyze_outputs(folder, "pp")["success"]
    cube = "Density\nSynthetic test\n2 0 0 0\n2 1 0 0\n2 0 1 0\n2 0 0 1\n14 4 0 0 0\n14 4 1 1 1\n1 1 1 1 1 1 1 1\n"
    (folder / "charge_density.cube").write_text(cube)
    result = analyze_outputs(folder, "pp")
    assert result["success"] and result["integrated_pseudocharge_electrons"] == 8
    (folder / "charge_density.cube").write_text(cube.rsplit("1 1 1 1 1 1 1 1", 1)[0] + "1 1\n")
    assert not analyze_outputs(folder, "pp")["success"]
