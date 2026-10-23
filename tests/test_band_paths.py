"""Band paths keep the lattice's labels after relaxation and never collapse to straight lines.

A relaxed cell carries residual distortions of 1e-3 to 1e-2 Å. With seekpath's default tolerance a
relaxed silicon cell got a monoclinic path (C, Y_2, M_2, ...), and a sample computed with
line_density 1 drew its bands as straight lines between 13 points.
"""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import qe, vasp
from vasp_slurm_agent.numerical_defaults import BAND_PATH_MIN_SEGMENT_POINTS, high_symmetry_path

CRYSTALS = {
    "diamond Si": lambda: Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]]),
    "bcc Fe": lambda: Structure.from_spacegroup("Im-3m", Lattice.cubic(2.866), ["Fe"], [[0, 0, 0]]),
    "hcp Mg": lambda: Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.209, 5.211), ["Mg"], [[1 / 3, 2 / 3, 0.25]]),
    "perovskite SrTiO3": lambda: Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
                                                           [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]),
    "rutile TiO2": lambda: Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
                                                     [[0, 0, 0], [0.3053, 0.3053, 0]]),
    "wurtzite GaN": lambda: Structure.from_spacegroup("P6_3mc", Lattice.hexagonal(3.189, 5.185), ["Ga", "N"],
                                                      [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.377]]),
}


def _primitive(name):
    return CRYSTALS[name]().get_primitive_structure()


def _relaxed_like(structure, seed):
    """Up to 3 mÅ of atomic noise and 0.05 % strain, as a relaxation from a displaced start leaves."""
    rng = np.random.default_rng(seed)
    strain = np.eye(3) + rng.uniform(-5e-4, 5e-4, (3, 3))
    lattice = Lattice(structure.lattice.matrix @ strain)
    cartesian = structure.cart_coords + rng.uniform(-0.003, 0.003, structure.cart_coords.shape)
    return Structure(lattice, structure.species, cartesian, coords_are_cartesian=True)


def _vertices(path):
    return [label for label in path["labels"] if label]


@pytest.mark.parametrize("name", list(CRYSTALS))
def test_residual_distortion_keeps_the_lattice_path(name):
    ideal = _primitive(name)
    expected = _vertices(high_symmetry_path(ideal, 40))
    for seed in range(10):
        assert _vertices(high_symmetry_path(_relaxed_like(ideal, seed), 40)) == expected


@pytest.mark.parametrize("distortion, expected", [
    ("tetragonal", ["GAMMA", "X", "M", "GAMMA", "Z", "R", "A", "Z", "X", "R", "M", "A"]),
    ("polar", ["GAMMA", "X", "M", "GAMMA", "Z", "R", "A", "Z", "X", "R", "M", "A"]),
    ("rhombohedral", ["GAMMA", "T", "H_2", "H_0", "L", "GAMMA", "S_0", "S_2", "F", "GAMMA"]),
])
def test_real_distortions_keep_their_lower_symmetry(distortion, expected):
    cubic = _primitive("perovskite SrTiO3")
    if distortion == "tetragonal":
        structure = Structure(Lattice.tetragonal(3.905, 3.905 * 1.01), cubic.species, cubic.frac_coords)
    elif distortion == "polar":
        structure = cubic.copy()
        structure.translate_sites([1], [0, 0, 0.05], frac_coords=False)
    else:
        structure = Structure(Lattice.from_parameters(3.84, 3.84, 3.84, 59.5, 59.5, 59.5), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    assert _vertices(high_symmetry_path(structure, 20)) == expected


@pytest.mark.parametrize("name", list(CRYSTALS))
@pytest.mark.parametrize("density", [1, 4, 60])
def test_every_segment_is_sampled_and_joined_like_seekpath(name, density):
    path = high_symmetry_path(_primitive(name), density)
    points, labels, distance = np.asarray(path["reciprocal_fractional"]), path["labels"], np.asarray(path["distance_inv_angstrom"])
    assert len(points) == len(labels) == len(distance) and np.all(np.diff(distance) >= -1e-12)
    for index, (start, end) in enumerate(path["segments"]):
        assert end - start >= BAND_PATH_MIN_SEGMENT_POINTS
        assert labels[start] and labels[end - 1] and not any(labels[start + 1:end - 1])
        if index:
            previous = path["segments"][index - 1][1]
            # Either the segments share their joining point, or the path jumps without adding distance.
            assert start == previous - 1 or (start == previous and distance[start] == pytest.approx(distance[start - 1]))


def test_large_cells_still_get_smooth_segments():
    # A 3x3x3 supercell shrinks the zone; a fixed line density alone would leave 2-4 points per segment.
    supercell = _primitive("diamond Si") * (3, 3, 3)
    path = high_symmetry_path(supercell, 60)
    assert path["is_supercell"] and min(end - start for start, end in path["segments"]) >= BAND_PATH_MIN_SEGMENT_POINTS


def test_vasp_and_qe_use_the_shared_path(tmp_path):
    from pymatgen.io.vasp import Poscar

    structure = _relaxed_like(_primitive("diamond Si"), 7)
    Poscar(structure).write_file(tmp_path / "POSCAR")
    expected = high_symmetry_path(structure, 2)
    prepared = vasp.prepare_inputs(tmp_path / "POSCAR", tmp_path / "bands", "bands", {"line_density": 2})["band_path"]
    assert prepared["labels"] == expected["labels"] and prepared["segments"] == expected["segments"]
    np.testing.assert_allclose(prepared["reciprocal_fractional"], expected["reciprocal_fractional"], atol=1e-12)
    assert qe._path(structure, {"line_density": 2})["labels"] == expected["labels"]


@pytest.mark.parametrize("engine", ["vasp", "qe"])
def test_one_step_relaxation_plot_shows_a_whole_step(tmp_path, monkeypatch, engine):
    # A relaxation that converges at once used to get axis ticks 0.945, 0.960, ... (or, for QE, no visible point).
    from types import SimpleNamespace
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: (saved.append(self), original(self, *args, **kwargs))[1])
    if engine == "vasp":
        vasp._export_plots(SimpleNamespace(ionic_steps=[{"e_0_energy": -10.8447, "e_fr_energy": -10.8447}]), tmp_path, "relax")
    else:
        qe._plot(tmp_path, "relax_energy", [1], [-10.8447], "Ionic step", "Energy (eV)", steps=True)
    ax = saved[0].axes[0]
    ticks = [tick for tick in ax.get_xticks() if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]]
    assert ticks == [1.0]
    assert any(line.get_marker() == "o" and len(line.get_xdata()) == 1 for line in ax.get_lines())


@pytest.mark.parametrize("name, structure, expected", [
    ("MoS2 monolayer", lambda: Structure(Lattice.hexagonal(3.19, 20.0), ["Mo", "S", "S"],
                                         [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.578], [2 / 3, 1 / 3, 0.422]]),
     ["GAMMA", "M", "K", "GAMMA"]),
    ("graphene", lambda: Structure(Lattice.hexagonal(2.46, 20.0), ["C", "C"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]]),
     ["GAMMA", "M", "K", "GAMMA"]),
    ("square monolayer", lambda: Structure(Lattice.tetragonal(3.0, 20.0), ["Fe", "Se", "Se"],
                                           [[0, 0, 0.5], [0.5, 0.5, 0.57], [0.5, 0.5, 0.43]]),
     ["GAMMA", "X", "M", "GAMMA"]),
    ("carbon chain", lambda: Structure(Lattice.tetragonal(20.0, 2.5), ["C"], [[0.5, 0.5, 0]]), ["GAMMA", "Z"]),
])
def test_low_dimensional_cells_skip_the_flat_out_of_plane_path(name, structure, expected):
    # Across a vacuum, Γ–A–L–H segments carry no dispersion and drew flat bands.
    path = high_symmetry_path(structure(), 40)
    assert _vertices(path) == expected and path["vacuum_axes"]
    assert min(end - start for start, end in path["segments"]) >= BAND_PATH_MIN_SEGMENT_POINTS


def _nio():
    a = 4.177
    lattice = Lattice(np.array([[0.5, 0.5, 1.0], [0.5, 1.0, 0.5], [1.0, 0.5, 0.5]]) * a)
    return Structure(lattice, ["Ni", "Ni", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])


def test_antiferromagnet_follows_its_magnetic_cell(tmp_path):
    # AFM-II NiO: structurally a doubled fcc cell (folded bands); magnetically rhombohedral.
    from pymatgen.io.vasp import Poscar
    from vasp_slurm_agent.engine_common import band_path

    nio = _nio()
    assert high_symmetry_path(nio, 20)["is_supercell"]
    afm = band_path(nio, {"spin": "collinear", "site_moments": [2.0, -2.0, 0.0, 0.0], "line_density": 20})
    assert not afm["is_supercell"] and "T" in _vertices(afm) and "X" not in _vertices(afm)
    labels = qe._path(nio, {"site_labels": ["Ni1", "Ni2", "O", "O"], "line_density": 20})
    assert _vertices(labels) == _vertices(afm)
    Poscar(nio).write_file(tmp_path / "POSCAR")
    prepared = vasp.prepare_inputs(tmp_path / "POSCAR", tmp_path / "bands", "bands",
                                   {"line_density": 20, "spin": "collinear", "magmom": [2.0, -2.0, 0.0, 0.0]})["band_path"]
    assert _vertices(prepared) == _vertices(afm)
    ferro = band_path(_primitive("bcc Fe"), {"spin": "collinear", "initial_moments": {"Fe": 3.0}, "line_density": 20})
    assert _vertices(ferro) == _vertices(high_symmetry_path(_primitive("bcc Fe"), 20, time_reversal=False))


def test_spin_orbit_path_keeps_time_reversal_without_a_moment():
    # Kramers pairs keep E(k) = E(-k) under SOC; GaAs and MoS2 SOC plots repeated their path as X', K', ...
    from vasp_slurm_agent.engine_common import band_path
    from vasp_slurm_agent.numerical_defaults import path_time_reversal

    gaas = Structure.from_spacegroup("F-43m", Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]).get_primitive_structure()
    primed = lambda path: [label for label in path["labels"] if "'" in label]
    assert not primed(band_path(gaas, {"soc": True}))
    assert not primed(qe._path(gaas, {"soc": True, "spin": "noncollinear"}))
    # A collinear magnet keeps it in each spin channel; a moment acting on spinors breaks it.
    assert not primed(band_path(gaas, {"spin": "collinear", "site_moments": [1.0, 0.0]}))
    assert primed(band_path(gaas, {"spin": "noncollinear", "soc": True, "site_moments": [[0, 0, 1.0], [0, 0, 0]]}))
    assert path_time_reversal(None, True) and path_time_reversal([0.0, 0.0], True)
    assert path_time_reversal([2.0], False) and not path_time_reversal([[0, 0, 2.0]], True)


def test_band_figures_read_like_the_vasp_ones(tmp_path):
    from vasp_slurm_agent import spectra

    # Joined jump labels and seekpath subscripts, never "H_2|H_0" or a lone second label.
    ticks = spectra.path_ticks([0.0, 1.0, 1.0, 2.0], ["GAMMA", "H_2", "H_0", "L"])
    assert [spectra.tick_label(label) for label in ticks.values()] == ["Γ", "H$_{2}$|H$_{0}$", "L"]
    # The window stops just below the highest computed band, so no band is missing from the figure.
    energies = np.zeros((2, 5, 3))
    energies[..., 0], energies[..., 1], energies[..., 2] = -3.0, 1.0, 4.5
    assert spectra.band_window(energies) == (-8.0, 4.4)
    energies[..., 2] = 20.0
    assert spectra.band_window(energies) == (-8.0, 8.0)
    # An insulator's Fermi level may sit anywhere in its gap; its figure starts at the valence band maximum.
    assert spectra.band_reference({"status": "gap", "vbm_ev": 5.0}, 7.9) == (5.0, "valence_band_maximum")
    assert spectra.band_reference({"status": "metallic_or_partially_occupied", "vbm_ev": 5.0}, 7.9) == (7.9, "fermi_level")
    names = spectra.plot_bands(tmp_path, "bands", np.linspace(0, 2, 5), energies, ["GAMMA", "", "", "", "X"], [[0, 5]])
    assert names == ["bands.png", "bands.pdf", "bands.svg"]
    svg = (tmp_path / "bands.svg").read_text(encoding="utf-8")
    assert "Spin ↑" in svg and "Spin ↓" in svg


def test_dos_window_stops_where_the_computed_states_end():
    from vasp_slurm_agent import spectra

    energies = np.linspace(-10, 10, 401)
    assert spectra.dos_window(energies, np.where(energies < 3.5, 1.0, 0.0))[1] == pytest.approx(3.5, abs=0.06)
    assert spectra.dos_window(energies, np.ones((401, 2))) == (-8.0, 8.0)
