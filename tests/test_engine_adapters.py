"""ABINIT, CP2K and Elk adapters: inputs, validation and checks on recorded outputs.

The recorded outputs in tests/data/engines come from real runs of ABINIT
10.0.3, CP2K 2026.2 and Elk 10.2.4 on a slightly displaced silicon cell, made
on the Arrhenius cluster; user names, hosts and paths are placeholders.
"""
import json
import shutil
from pathlib import Path

import pytest

from vasp_slurm_agent import abinit, agent, cp2k, elk, engine_agent, engine_setup
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.engine_workflow import _command_for

DATA = Path(__file__).parent / "data" / "engines"
ADAPTERS = {"abinit": abinit, "cp2k": cp2k, "elk": elk}
STAGES = ("01_relax", "02_scf", "03_bands", "04_dos")


def recorded(tmp_path, engine, folder):
    target = tmp_path / f"{engine}-{folder}"
    shutil.copytree(DATA / engine / folder, target)
    return target


@pytest.mark.parametrize("engine", ADAPTERS)
@pytest.mark.parametrize("task", ["relax", "scf", "bands", "dos"])
def test_inputs_pin_library_files_and_freeze_structure(tmp_path, engine, task):
    module = ADAPTERS[engine]
    metadata = module.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "inputs", task, {"mesh": [2, 2, 2]})
    assert metadata["engine"] == engine and metadata["canonical_task"] == task
    for name in metadata["input_files"]:
        assert (tmp_path / "inputs" / name).is_file()
    assert metadata["library_files"] and all(len(entry["sha256"]) == 64 for entry in metadata["library_files"].values())
    assert metadata["native_structure"]["elements"] == ["Si", "Si"]
    text = (tmp_path / "inputs" / metadata["input_file"]).read_text()
    if engine == "abinit":
        assert 'pseudos "Si.pbe_sr.psp8"' in text and "chksymtnons 0" in text
        assert ("iscf -2" in text) == (task in {"bands", "dos"})
    elif engine == "cp2k":
        assert "ABC " in text and "SCHEME MACDONALD 2 2 2 0.25 0.25 0.25" in text and "SYMMETRY OFF" in text or task == "dos"
        assert ("WFN_RESTART_FILE_NAME" in text) == (task == "bands")
    else:
        assert "'Si.in'" in text and ".false." in text
    assert module.restart_mapping(task, metadata) or task not in {"bands"}


@pytest.mark.parametrize("engine, parameters, message", [
    ("abinit", {"functional": "LDA"}, "PBE or PBEsol"),
    ("abinit", {"functional": "PBEsol", "soc": True}, "PBEsol"),
    ("abinit", {"hubbard": [{"element": "Si", "l": 1, "u_ev": 2.0}]}, "PAW"),
    ("abinit", {"spin": "collinear", "soc": True}, "noncollinear"),
    ("cp2k", {"soc": True}, "Spin-orbit"),
    ("cp2k", {"spin": "noncollinear"}, "noncollinear"),
    ("cp2k", {"hubbard": [{"element": "Si", "l": 1, "u_ev": 2.0}], "mesh": [2, 2, 2]}, "Γ-point"),
    ("cp2k", {"functional": "LDA", "basis_family": "uzh_dzvp"}, "LDA"),
    ("elk", {"hubbard": [{"element": "Si", "l": 1, "u_ev": 2.0}]}, "spin"),
    ("elk", {"functional": "HSE06"}, "Hybrid"),
    ("abinit", {"ecutwfc": 40}, "Unsupported"),
    ("elk", {"spin": "collinear", "site_moments": [1.0]}, "one starting moment"),
])
def test_unsupported_methods_are_explained(tmp_path, engine, parameters, message):
    with pytest.raises(ValueError, match=message):
        ADAPTERS[engine].prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "inputs", "scf", parameters)


def test_cp2k_uses_time_reversal_for_odd_and_shifted_grids(tmp_path):
    # Both routes bypass the broken K290 point-group reduction in CP2K 2026.1.
    for mesh, scheme in (([3, 3, 3], "MONKHORST-PACK 3 3 3"), ([4, 4, 3], "MACDONALD 4 4 3 0.125 0.125 0")):
        metadata = cp2k.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / str(mesh[0]), "scf", {"mesh": mesh})
        lines = [line.strip() for line in (tmp_path / str(mesh[0]) / metadata["input_file"]).read_text().splitlines()]
        assert lines[lines.index("&KPOINTS") + 1:lines.index("&END KPOINTS")] == ["SCHEME " + scheme, "SYMMETRY OFF", "FULL_GRID OFF"]


def test_abinit_monolayer_insulator_dos_uses_gaussian_broadening(tmp_path):
    # Tetrahedra need a 3D mesh; a monolayer's DOS used to be refused for an insulator.
    from pymatgen.core import Lattice, Structure

    source = tmp_path / "MoS2.cif"
    Structure(Lattice.hexagonal(3.18, 20.0), ["Mo", "S", "S"],
              [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.578], [2 / 3, 1 / 3, 0.422]]).to(filename=str(source))
    metadata = abinit.prepare_inputs(source, tmp_path / "dos", "dos", {"mesh": [6, 6, 1], "electronic_type": "insulator"})
    lines = (tmp_path / "dos" / metadata["input_file"]).read_text().splitlines()
    assert "occopt 7" in lines and "prtdos 1" in lines and any(line.startswith("tsmear") for line in lines)
    assert metadata["dos_method"] == "gaussian_smearing" and "Gaussian" in " ".join(metadata["notes"])


def test_abinit_vacuum_cells_use_a_vacuum_preconditioner(tmp_path):
    # With the metallic default (diemac 1e6) layers kept a potential residual of 1-20 and an
    # MoS2 relaxation stopped on noisy forces; diemac 2 converged it in three steps on Arrhenius.
    from pymatgen.core import Lattice, Structure

    source = tmp_path / "MoS2.cif"
    Structure(Lattice.hexagonal(3.18, 20.0), ["Mo", "S", "S"],
              [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.578], [2 / 3, 1 / 3, 0.422]]).to(filename=str(source))
    for task in ("relax", "scf", "bands"):
        metadata = abinit.prepare_inputs(source, tmp_path / task, task, {"mesh": [6, 6, 1]})
        assert "diemac 2" in (tmp_path / task / metadata["input_file"]).read_text().splitlines()
    abinit.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "bulk", "scf", {"mesh": [2, 2, 2]})
    assert "diemac" not in (tmp_path / "bulk" / "run.abi").read_text()


def test_abinit_spinor_band_counts_are_even(tmp_path):
    # m_inwffil.F90 stops a spinor run with an odd band count (Bi2Se3 with SOC asked for 117).
    for task in ("scf", "bands", "dos"):
        for extra in (None, 7, 8):
            parameters = {"soc": True, "mesh": [2, 2, 2], **({} if extra is None else {"extra_bands": extra})}
            metadata = abinit.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / f"{task}{extra}", task, parameters)
            assert metadata["bands"] % 2 == 0 and f"nband {metadata['bands']}" in (tmp_path / f"{task}{extra}" / "run.abi").read_text()


def test_abinit_error_documents_are_reported(tmp_path):
    folder = recorded(tmp_path, "abinit", "01_relax")
    output = folder / "run.abo"
    output.write_text(output.read_text().replace("Calculation completed.", "") + "\n--- !Ionmov22Error\nsrc_file: m_pred_bfgs.F90\n"
                      "message: |\n    Lbfgs routine failed. Returned value: 0\n    Restart your calculation from last step or try a different ionmov\n...\n")
    reason = abinit.analyze_outputs(folder, "relax")["reason"]
    assert 'Ionmov22Error: "Lbfgs routine failed. Returned value: 0 Restart your calculation' in reason
    # Another rank's error reaches only the abort file, which is now collected.
    folder = recorded(tmp_path, "abinit", "03_bands")
    output = folder / "run.abo"
    output.write_text(output.read_text().replace("Calculation completed.", ""))
    (folder / "__ABI_MPIABORTFILE__").write_text("\n--- !BUG\nsrc_file: m_inwffil.F90\nmpi_rank: 8\nmessage: |\n"
                                                 "    When nspinor2 is 2, nbd2 must be even, while it is nbd2=117\n...\n")
    assert 'BUG: "When nspinor2 is 2, nbd2 must be even' in abinit.analyze_outputs(folder, "bands")["reason"]
    assert "__ABI_MPIABORTFILE__" in abinit.OUTPUT_FILES


def test_magnetic_inputs_follow_requested_moments(tmp_path):
    parameters = {"spin": "collinear", "site_moments": [2.0, -2.0], "mesh": [2, 2, 2]}
    text = (tmp_path / "a").mkdir() or abinit.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "a", "scf", parameters)
    assert "nsppol 2" in (tmp_path / "a" / "run.abi").read_text()
    cp2k.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "c", "scf", {**parameters, "mesh": [1, 1, 1]})
    deck = (tmp_path / "c" / "cp2k.inp").read_text()
    assert "UKS TRUE" in deck and "MULTIPLICITY 1" in deck and "MAGNETIZATION -2" in deck and "&KIND Si_2" in deck
    elk.prepare_inputs(DATA / "Si-perturbed.cif", tmp_path / "e", "scf", parameters)
    deck = (tmp_path / "e" / "elk.in").read_text()
    assert "spinpol" in deck and "reducebf" in deck and "-0.01" in deck and "0.01" in deck


@pytest.mark.parametrize("engine", ADAPTERS)
@pytest.mark.parametrize("folder", STAGES)
def test_recorded_outputs_are_accepted(tmp_path, engine, folder):
    task = folder.split("_", 1)[1]
    result = ADAPTERS[engine].analyze_outputs(recorded(tmp_path, engine, folder), task)
    assert result["success"], result.get("reason")
    assert result["converged_electronic"] is not False
    if task in {"relax", "scf"}:
        assert -16000 < result["final_energy_ev"] < -200
        assert "final.cif" in result["artifacts"] and "CONTCAR" in result["artifacts"]
    if task == "relax":
        assert result["converged_ionic"] is True
    if task in {"bands", "dos"}:
        gap = result["band_gap"]
        assert gap["status"] == "gap" and 0.3 < gap["gap_ev"] < 1.2
        assert f"{task}.png" in result["artifacts"]


def test_spin_polarized_abinit_path_keeps_the_scf_fermi_level(tmp_path):
    # A non-self-consistent path prints no Fermi line of its own; ResultsGS carries the SCF density's value.
    # Without it a spin-polarized path (no electron count per channel) had no gap and an absolute energy axis.
    folder = recorded(tmp_path, "abinit", "03_bands")
    metadata = json.loads((folder / "abinit-input.json").read_text())
    (folder / "abinit-input.json").write_text(json.dumps({**metadata, "nsppol": 2}))
    eig = folder / "dft_agento_EIG"
    text = eig.read_text()
    eig.write_text(text.replace("k points:", "k points, SPIN UP:", 1) + text.replace("k points:", "k points, SPIN DOWN:", 1))
    result = abinit.analyze_outputs(folder, "bands")
    assert result["success"], result.get("reason")
    assert result["fermi_energy_ev"] == pytest.approx(1.81922413e-01 * abinit.HA_EV)
    assert result["band_gap"]["status"] == "gap" and 0.3 < result["band_gap"]["gap_ev"] < 1.2
    # An insulator's plot starts at its valence band maximum; its Fermi level may sit anywhere in the gap.
    assert result["energy_reference"] == "valence_band_maximum"
    assert result["plot_reference_ev"] == pytest.approx(result["band_gap"]["vbm_ev"])


def test_cp2k_path_gap_follows_the_fermi_level_not_the_band_file_occupations(tmp_path):
    # For a ferromagnet CP2K filled 39 and 31 bands for 38 and 32 electrons per spin and split a
    # degenerate pair; those occupations made CrI3 look metallic on its path.
    folder = recorded(tmp_path, "cp2k", "03_bands")
    expected = cp2k.analyze_outputs(folder, "bands")["band_gap"]
    band_file = folder / "dft_agent.bs"
    lines = band_file.read_text().splitlines(keepends=True)
    homo = next(i for i, line in enumerate(lines) if line.split()[:1] == ["4"])
    lines[homo] = lines[homo].replace("2.00000000", "0.00000000")
    lines[homo + 1] = lines[homo + 1].replace("0.00000000", "2.00000000")
    band_file.write_text("".join(lines))
    gap = cp2k.analyze_outputs(folder, "bands")["band_gap"]
    assert gap["status"] == "gap" and gap["gap_ev"] == pytest.approx(expected["gap_ev"])


def _cp2k_histogram(bins, columns):
    """A CP2K DOS file on CP2K's 0.0005 a.u. grid; bins maps energies to (density, occupation, ...)."""
    lines = ["# DOS at iteration step i = 0", "# E(Fermi) =     0.200000 a.u. =     5.442277 eV",
             "#  Energy[a.u.]" + "".join(f"  {name}" for name in columns)]
    for step in range(-600, 801):
        energy = round(step * 0.0005, 4)
        values = bins.get(energy, (0.0,) * len(columns))
        lines.append(f"{energy:15.8f}" + "".join(f"{value:20.8f}" for value in values))
    return "\n".join(lines) + "\n"


def test_cp2k_dos_reads_both_spins_and_finds_metals_by_occupation(tmp_path):
    # CP2K 2026 writes both spins in one file, of which only the alpha column was read. On a sparse
    # mesh a metal's Fermi level falls among empty bins, so Al and Co had shown 0.12 and 0.05 eV gaps.
    spins = _cp2k_histogram({-0.1: (0.5, 5.0, 0.5, 5.0), 0.35: (0.5, 0.0, 0.5, 0.0)},
                            ["Alpha_Density", "Occupation", "Beta_Density", "Occupation"])
    alpha, beta = cp2k._histograms(spins)
    assert alpha.shape[1] == beta.shape[1] == 3 and alpha[:, 1].sum() == beta[:, 1].sum() == 1.0
    (tmp_path / "insulator.dos").write_text(spins)
    gap = cp2k._histogram_gap(tmp_path, ["insulator.dos"])
    assert gap["status"] == "gap" and gap["gap_ev"] == pytest.approx(0.45 * cp2k.HA_EV)
    # A populated bin below the Fermi level holding half its states: the Fermi level cuts a band.
    (tmp_path / "metal.dos").write_text(_cp2k_histogram({-0.1: (0.4, 8.0), 0.15: (0.2, 2.0), 0.35: (0.4, 0.0)},
                                                        ["Density", "Occupation"]))
    assert cp2k._histogram_gap(tmp_path, ["metal.dos"])["status"] == "metallic_or_partially_occupied"


def test_abinit_dos_keeps_the_scf_fermi_level(tmp_path):
    folder = recorded(tmp_path, "abinit", "04_dos")
    fermie = float([line for line in (folder / "run.abo").read_text().splitlines() if line.startswith("fermie")][-1].split()[-1])
    result = abinit.analyze_outputs(folder, "dos")
    assert result["fermi_energy_ev"] == pytest.approx(fermie * abinit.HA_EV)
    assert result["band_gap"]["status"] == "gap"


def test_elk_spin_orbit_dos_without_a_moment_is_one_total(tmp_path):
    # Elk needs spinpol for spin-orbit coupling and writes two spin halves even without a moment;
    # they were drawn as spin up and down of a non-magnetic crystal.
    import csv

    folder = recorded(tmp_path, "elk", "04_dos")
    single = elk.analyze_outputs(folder, "dos")
    total = [float(row[1]) for row in list(csv.reader((folder / "dos.csv").open()))[1:]]
    tdos = folder / "TDOS.OUT"
    halves = "".join(f"{line.split()[0]} {float(line.split()[1]) / 2}\n" if line.split() else line + "\n"
                     for line in tdos.read_text().splitlines())
    tdos.write_text(halves + "\n" + halves)
    result = elk.analyze_outputs(folder, "dos")
    rows = list(csv.reader((folder / "dos.csv").open()))
    assert result["success"] and single["success"] and rows[0] == ["energy_ev", "dos"]
    assert [float(row[1]) for row in rows[1:]] == pytest.approx(total)


def test_elk_lithium_uses_the_finer_radial_mesh(tmp_path):
    # The valence Li 1s density lost 0.028 of 12 electrons on lradstp 4 and LiF PBE never converged.
    from pymatgen.core import Lattice, Structure

    for formula, species in (("LiF", ["Li", "F"]), ("NaCl", ["Na", "Cl"])):
        path = tmp_path / f"{formula}.cif"
        Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.03), species, [[0, 0, 0], [0.5, 0.5, 0.5]]).get_primitive_structure().to(filename=str(path))
        elk.prepare_inputs(path, tmp_path / formula, "scf", {"mesh": [4, 4, 4]})
        assert ("lradstp\n  2\n" in (tmp_path / formula / "elk.in").read_text()) == (formula == "LiF")


def test_cp2k_2026_1_dos_without_header_fermi_level_uses_the_run_output(tmp_path):
    # CP2K 2026.1 writes no E(Fermi) line in the DOS histogram; the gap must not depend on it.
    folder = recorded(tmp_path, "cp2k", "04_dos")
    expected = cp2k.analyze_outputs(folder, "dos")["band_gap"]
    histogram = folder / "dft_agent-1.dos"
    histogram.write_text("".join(line for line in histogram.read_text().splitlines(keepends=True)
                                 if not line.startswith(("# E(Fermi)", "# E(HOCO)"))))
    result = cp2k.analyze_outputs(folder, "dos")
    assert result["success"] and result["band_gap"]["status"] == "gap"
    assert result["band_gap"]["gap_ev"] == pytest.approx(expected["gap_ev"], abs=1e-6)


@pytest.mark.parametrize("engine, name, old, new", [
    ("abinit", "run.abo", "Calculation completed.", "Calculation interrupted."),
    ("cp2k", "cp2k.out", "PROGRAM ENDED AT", "PROGRAM KILLED AT"),
    ("elk", "elk.log", "Elk code stopped", "Elk code killed"),
])
def test_unfinished_runs_are_rejected(tmp_path, engine, name, old, new):
    folder = recorded(tmp_path, engine, "02_scf")
    path = folder / name
    path.write_text(path.read_text().replace(old, new))
    result = ADAPTERS[engine].analyze_outputs(folder, "scf")
    assert not result["success"] and "finish" in result["reason"]


def test_unconverged_electronic_and_ionic_steps_are_rejected(tmp_path):
    folder = recorded(tmp_path, "abinit", "02_scf")
    (folder / "run.abo").write_text((folder / "run.abo").read_text() + "\n nstep=   40 was not enough SCF cycles to converge;\n")
    assert "electronic" in abinit.analyze_outputs(folder, "scf")["reason"]
    folder = recorded(tmp_path, "cp2k", "01_relax")
    (folder / "cp2k.out").write_text((folder / "cp2k.out").read_text().replace("GEOMETRY OPTIMIZATION COMPLETED", "OPTIMIZATION STOPPED"))
    assert "geometry" in cp2k.analyze_outputs(folder, "relax")["reason"].lower()
    folder = recorded(tmp_path, "elk", "02_scf")
    (folder / "elk.log").write_text((folder / "elk.log").read_text() + "\nWarning(gndstate): failed to reach self-consistency after  200 loops\n")
    assert "self-consistent" in elk.analyze_outputs(folder, "scf")["reason"]


def test_a_different_structure_is_detected(tmp_path):
    folder = recorded(tmp_path, "abinit", "02_scf")
    metadata = json.loads((folder / "abinit-input.json").read_text())
    metadata["native_structure"]["fractional_positions"][1][0] += 0.01
    (folder / "abinit-input.json").write_text(json.dumps(metadata))
    result = abinit.analyze_outputs(folder, "scf")
    assert not result["success"] and "structure" in result["reason"]


def test_launch_commands_keep_the_launcher_and_append_stage_arguments():
    config = ClusterConfig(engine="cp2k", host="h", user="u", remote_root="/r", partition="cpu", tasks=16,
                           engine_command="/opt/cp2k/bin/mpiexec -n {tasks} /opt/cp2k/bin/cp2k.psmp", engine_library_root="/lib")
    assert _command_for(config, {"engine": "cp2k", "arguments": ["-i", "cp2k.inp", "-o", "cp2k.out"]}) == \
        "/opt/cp2k/bin/mpiexec -n 16 /opt/cp2k/bin/cp2k.psmp -i cp2k.inp -o cp2k.out"
    for command in ("srun abinit; rm -rf /", "mpiexec -n 4 cp2k.psmp -i x.inp", "srun pw.x", "srun elk --x"):
        engine = "cp2k" if "cp2k" in command else "elk" if "elk" in command else "abinit"
        bad = ClusterConfig(engine=engine, host="h", user="u", remote_root="/r", partition="cpu", engine_command=command,
                            engine_library_root="/lib")
        with pytest.raises(ValueError):
            _command_for(bad, {"engine": engine, "arguments": []})


def test_config_keeps_older_engines_unchanged_and_validates_new_ones(tmp_path):
    qe = ClusterConfig(engine="qe", host="h", user="u", remote_root="/r", partition="cpu", qe_pseudo_root="/p")
    assert not {"engine_command", "engine_library_root", "engine_library"} & set(qe.to_dict())
    with pytest.raises(ValueError, match="launch command"):
        ClusterConfig(engine="elk", host="h", user="u", remote_root="/r", partition="cpu", engine_library_root="/lib")
    with pytest.raises(ValueError, match="engine_library_root"):
        ClusterConfig(engine="elk", host="h", user="u", remote_root="/r", partition="cpu", engine_command="elk")
    with pytest.raises(ValueError, match="belong"):
        ClusterConfig(engine="qe", host="h", user="u", remote_root="/r", partition="cpu", qe_pseudo_root="/p", engine_command="elk")
    config = ClusterConfig(engine="abinit", host="h", user="u", remote_root="/r", partition="cpu", engine_command="srun abinit",
                           engine_library_root="/lib", engine_library={"Si.pbe_sr.psp8": {"sha256": "a" * 64}})
    assert ClusterConfig.load(config.save(tmp_path / "c.json")) == config


@pytest.mark.parametrize("engine, platform, glibc, expected", [
    ("abinit", "linux-64", "2.34", "2.28"), ("abinit", "linux-64", "2.17", "2.17"),
    ("cp2k", "linux-64", "2.34", "2.28"), ("cp2k", "linux-aarch64", "2.39", "2.38"), ("elk", "linux-aarch64", "2.31", "2.17"),
])
def test_package_sets_follow_architecture_and_glibc(engine, platform, glibc, expected):
    assert engine_setup.choose_variant(engine, platform, glibc)["min_glibc"] == expected


@pytest.mark.parametrize("engine, platform, glibc, message", [
    ("cp2k", "linux-64", "2.17", "glibc 2.28"), ("cp2k", "linux-aarch64", "2.34", "glibc 2.38"),
    ("elk", "linux-ppc64le", "2.28", "architecture"), ("abinit", "linux-64", None, "glibc"),
])
def test_unsupported_platforms_point_to_cluster_modules(engine, platform, glibc, message):
    with pytest.raises(ValueError, match=message):
        engine_setup.choose_variant(engine, platform, glibc)


def test_install_plan_is_deterministic_and_pinned():
    from types import SimpleNamespace
    profile = SimpleNamespace(remote_root="/home/u/dft", partition="main", account="naiss", scheduler="slurm", extra_pbs=[])
    first = engine_setup.install_plan(profile, "elk", {"platform": "linux-64", "glibc": "2.28"})
    assert first == engine_setup.install_plan(profile, "elk", {"platform": "linux-64", "glibc": "2.28"})
    assert all(item["url"].startswith("https://") and len(item["sha256"]) == 64 for item in first["downloads"])
    assert {item["role"] for item in first["downloads"]} == {"micromamba", "package", "species_archive"}
    assert first["library_root"].endswith("share/elk/species") and first["command"].endswith("/bin/elk")
    script = engine_setup._job_script(first)
    assert "#SBATCH --account=naiss" in script and "#SBATCH --cpus-per-task=4" in script and "python3 install.py" in script
    cp2k_plan = engine_setup.install_plan(profile, "cp2k", {"platform": "linux-64", "glibc": "2.28"})
    assert "-n {tasks}" in cp2k_plan["command"] and "export HYDRA_BOOTSTRAP=fork" in cp2k_plan["setup_commands"]
    # CP2K data files are pinned on their own, in the same folder site modules use.
    assert cp2k_plan["version"] == "2026.1" and "/software/cp2k-2026.1-" in cp2k_plan["prefix"]
    assert cp2k_plan["library_root"] == "/home/u/dft/libraries/cp2k-2026.2"
    assert {item["name"] for item in cp2k_plan["downloads"] if item["role"] == "library"} == {
        "BASIS_MOLOPT", "GTH_POTENTIALS", "BASIS_MOLOPT_UZH", "POTENTIAL_UZH"}
    arm = engine_setup.install_plan(profile, "cp2k", {"platform": "linux-aarch64", "glibc": "2.39"})
    assert arm["version"] == "2026.2" and "/software/cp2k-2026.2-" in arm["prefix"]


def test_library_plans_name_pinned_files():
    assert [item["name"] for item in engine_setup.plan_library("abinit", ["Si", "O"], soc=True)] == ["Si.pbe_fr.psp8", "O.pbe_fr.psp8"]
    assert {item["name"] for item in engine_setup.plan_library("cp2k", [])} == {"BASIS_MOLOPT", "GTH_POTENTIALS", "BASIS_MOLOPT_UZH", "POTENTIAL_UZH"}
    assert [item["name"] for item in engine_setup.plan_library("elk", ["Fe"])] == ["Fe.in"]
    with pytest.raises(ValueError, match="Ni"):
        engine_setup.plan_library("abinit", ["Ni"], soc=True)


FAKE_MAMBA = """#!{python}
import os, pathlib, shutil, sys
prefix = pathlib.Path(sys.argv[sys.argv.index("-p") + 1])
(prefix / "bin").mkdir(parents=True)
for name in ("cp2k.psmp", "mpiexec"):
    shutil.copy({fake!r} + "/" + name, prefix / "bin" / name)
cache = pathlib.Path(os.environ["CONDA_PKGS_DIRS"]) / "cp2k-2026.1"
cache.mkdir(parents=True)
(cache / "extracted").write_text("unpacked package")
"""
FAKE_CP2K = {
    "ok": "#!/bin/sh\necho ' PROGRAM ENDED AT  2026-10-22' > cp2k.out\n",
    "glibc": "#!/bin/sh\necho \"cp2k.psmp: /lib64/libc.so.6: version \\`GLIBC_2.38' not found (required by libxs.so.1)\" >&2\nexit 1\n",
}


@pytest.mark.parametrize("mode", ["ok", "glibc"])
def test_installer_places_pinned_library_and_explains_glibc_failures(tmp_path, mode):
    import hashlib, subprocess, sys, tarfile
    fake, work = tmp_path / "fake", tmp_path / "work"
    for folder in (fake, work / "pkgs"):
        folder.mkdir(parents=True)
    (fake / "cp2k.psmp").write_text(FAKE_CP2K[mode])
    (fake / "mpiexec").write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    (fake / "micromamba").write_text(FAKE_MAMBA.format(python=Path(sys.executable).resolve(), fake=str(fake)))
    for name in ("cp2k.psmp", "mpiexec", "micromamba"):
        (fake / name).chmod(0o755)
    with tarfile.open(work / "pkgs" / "micromamba.tar.bz2", "w:bz2") as tar:
        tar.add(fake / "micromamba", arcname="bin/micromamba")
    (work / "pkgs" / "cp2k-2026.1.conda").write_bytes(b"package")
    for name in engine_setup.plan_library("cp2k", []):
        (work / "pkgs" / name["name"]).write_text(f"pinned {name['name']}\n")
    roles = {"micromamba.tar.bz2": "micromamba", "cp2k-2026.1.conda": "package"}
    downloads = [{"name": path.name, "role": roles.get(path.name, "library"), "size": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted((work / "pkgs").iterdir())]
    plan = {"id": "p1", "engine": "cp2k", "version": "2026.1", "program": "cp2k.psmp", "prefix": str(tmp_path / "prefix"),
            "library_root": str(tmp_path / "libraries" / "cp2k"), "mpi": "mpich", "build_cpus": 2, "extras": {}, "downloads": downloads}
    (work / "plan.json").write_text(json.dumps(plan))
    (work / "install.py").write_text(engine_setup._INSTALL_PY)
    metadata = engine_setup._smoke_inputs("cp2k", work / "smoke")
    (work / "smoke" / "metadata.json").write_text(json.dumps(metadata))
    done = subprocess.run([sys.executable, "install.py"], cwd=work, capture_output=True, text=True, timeout=120)
    status = json.loads((work / "install-status.json").read_text())
    library = tmp_path / "libraries" / "cp2k"
    assert sorted(path.name for path in library.iterdir()) == sorted(item["name"] for item in engine_setup.plan_library("cp2k", []))
    assert (work / "smoke" / "library" / "BASIS_MOLOPT").read_text() == "pinned BASIS_MOLOPT\n"
    # The extracted package cache never outlives the environment creation.
    assert not (work / "cache").exists()
    if mode == "ok":
        assert done.returncode == 0 and status["status"] == "installed" and status["library_root"] == str(library)
        # Package archives go once the test passed; a repeated job still finds the installation.
        assert sorted(path.name for path in (work / "pkgs").iterdir()) == sorted(
            [*(item["name"] for item in engine_setup.plan_library("cp2k", [])), "micromamba.tar.bz2"])
        again = subprocess.run([sys.executable, "install.py"], cwd=work, capture_output=True, text=True, timeout=120)
        assert again.returncode == 0 and json.loads((work / "install-status.json").read_text())["status"] == "installed"
        return
    assert done.returncode == 1 and status["status"] == "failed" and (work / "pkgs" / "cp2k-2026.1.conda").is_file()
    assert "cp2k.psmp needs glibc 2.38 but the compute node has" in status["error"] and "module" in status["error"]
    # A later attempt with the same packages reuses the environment instead of refusing the folder.
    shutil.copy(fake / "mpiexec", tmp_path / "prefix" / "bin" / "mpiexec")
    (tmp_path / "prefix" / "bin" / "cp2k.psmp").write_text(FAKE_CP2K["ok"])
    (fake / "micromamba").unlink()
    (work / "plan.json").write_text(json.dumps({**plan, "id": "p2"}))
    again = subprocess.run([sys.executable, "install.py"], cwd=work, capture_output=True, text=True, timeout=120)
    status = json.loads((work / "install-status.json").read_text())
    assert again.returncode == 0 and status["status"] == "installed" and status["created"] is False
    assert json.loads((tmp_path / "prefix" / ".dft-agent-install.json").read_text())["plan_id"] == "p2"
    assert not (work / "pkgs" / "cp2k-2026.1.conda").exists()


def _plan(tasks, parameters=None, engine_requirements=None):
    return {"status": "ready", "summary": "Relax and calculate bands.", "questions": [], "notes": [],
            "intent": {"requested_tasks": list(tasks), "forbidden_tasks": [], "unsupported": []}, "operations": [],
            "stages": [{"task": task, "parameters_json": json.dumps(parameters or {}),
                        "requirements": {**{key: None for key in ("spin", "soc", "functional", "hubbard_u", "cell_relax")},
                                         **(engine_requirements or {})}} for task in tasks]}


@pytest.mark.parametrize("engine", ["abinit", "cp2k", "elk"])
def test_planner_reviews_and_prepares_without_a_cluster(tmp_path, monkeypatch, engine):
    source = tmp_path / "Si.cif"
    shutil.copyfile(DATA / "Si-perturbed.cif", source)
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(_plan(["relax", "bands"], {"mesh": [2, 2, 2]})))
    plan = engine_agent.draft_task("Relax, then bands", source, agent.ModelSettings("codex"), engine=engine)
    assert plan["status"] == "ready" and plan["kind"] == f"{engine}_task", plan["questions"]
    state = engine_agent.prepare_task(source, tmp_path / "run", None, plan)
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands"]
    assert json.loads((tmp_path / "run" / "proposal.json").read_text())["execution"]["plan_sha256"] == state["plan_sha256"]


def test_unsupported_engine_plan_is_not_asked_to_confirm_a_sequence(tmp_path, monkeypatch):
    # CP2K has no SOC here; the reply named other programs but also asked to "confirm the calculation sequence".
    source = tmp_path / "Si.cif"
    shutil.copyfile(DATA / "Si-perturbed.cif", source)
    plan = _plan(["bands"])
    plan.update(status="unsupported", stages=[], intent={"requested_tasks": ["bands"], "forbidden_tasks": [], "unsupported": ["SOC"]})
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(plan))
    result = engine_agent.draft_task("SOC bands", source, agent.ModelSettings("codex"), engine="cp2k")
    assert result["status"] == "unsupported" and not any("sequence" in question for question in result["questions"])


def test_planner_turns_unsupported_requests_into_questions(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    shutil.copyfile(DATA / "Si-perturbed.cif", source)
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(_plan(["scf"], {"soc": True})))
    plan = engine_agent.draft_task("SOC SCF", source, agent.ModelSettings("codex"), engine="cp2k")
    assert plan["status"] == "needs_input" and any("Spin-orbit" in question for question in plan["questions"])
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **kw: json.dumps(
        _plan(["scf"], {"spin": "collinear"}, {"spin": "none"})))
    plan = engine_agent.draft_task("Nonmagnetic SCF", source, agent.ModelSettings("codex"), engine="elk")
    assert plan["status"] == "needs_input" and any("spin" in question for question in plan["questions"])
