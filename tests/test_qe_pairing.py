import json

import numpy as np
import pytest

from vasp_slurm_agent import qe_pairing as pairing


def _hr(tmp_path, rows=None, weights="1 1 1"):
    path = tmp_path / "model_hr.dat"
    rows = rows or ["0 0 0 1 1 0 0", "1 0 0 1 1 -1 0", "-1 0 0 1 1 -1 0"]
    path.write_text("Test\n1\n3\n" + weights + "\n" + "\n".join(rows) + "\n")
    return path


def _parameters():
    return {"lattice_angstrom": np.eye(3).tolist(), "orbital": "s", "spin_convention": "spin_degenerate_no_soc",
            "u_ev": 1., "mu_ev": 0., "temperature_k": 100., "kmesh": [8, 1, 1],
            "matsubara_points": 100, "channels": ["singlet", "triplet"],
            "double_counting": "Explicit Hubbard model; no DFT baseline."}


def test_wannier_wigner_seitz_degeneracy(tmp_path):
    hopping = pairing.read_hoppings(_hr(tmp_path, weights="1 2 2"))
    assert hopping[(1, 0, 0)] == -0.5
    assert hopping[(-1, 0, 0)] == -0.5


@pytest.mark.parametrize("rows", [
    ["0 0 0 1 1 0 0", "1 0 0 1 1 -1 0", "-1 0 0 1 1 -2 0"],
    ["0 0 0 1 1 0 0", "1 0 0 1 1 -1 0", "1 0 0 1 1 -1 0"],
    ["0 0 0 1 1 0 0", "1 0 0 1 1 -1 1", "-1 0 0 1 1 -1 -1"],
    ["0 0 0 1 1 0 0", "1 0 0 1 1 nan 0", "-1 0 0 1 1 -1 0"],
])
def test_invalid_or_time_reversal_breaking_hamiltonian(tmp_path, rows):
    with pytest.raises(ValueError):
        pairing.read_hoppings(_hr(tmp_path, rows))


def test_multiorbital_not_silently_projected(tmp_path):
    path = _hr(tmp_path)
    path.write_text(path.read_text().replace("Test\n1\n", "Test\n2\n"))
    with pytest.raises(ValueError, match="one spin-degenerate"):
        pairing.read_hoppings(path)


def test_singlet_vertex_has_charge_screening_sign_and_factor():
    chi, u = 0.2, 1.0
    interaction = pairing.pairing_vertex([chi], u, "singlet")[0] / 2
    assert interaction == pytest.approx(u + 1.5 * u*u*chi/(1-u*chi) - 0.5*u*u*chi/(1+u*chi))
    small_u = 1e-5
    expansion = (pairing.pairing_vertex([chi], small_u, "singlet")[0] / 2 - small_u) / small_u**2
    assert expansion.real == pytest.approx(chi, rel=1e-5)


def test_triplet_bare_onsite_term_cancels():
    assert pairing.pairing_vertex([0], 2, "triplet")[0] == 0
    assert pairing.pairing_vertex([0.2], 1, "triplet")[0] == pytest.approx(-0.2/0.8 - 0.2/1.2)
    with pytest.raises(ValueError, match="pole"):
        pairing.pairing_vertex([0.5], 2, "singlet")


def test_static_lindhard_flat_band_and_zero_momentum():
    beta = 5
    assert pairing._static_lindhard(np.zeros(8), beta, (8, 1, 1)) == pytest.approx(np.full(8, beta/4))
    energy = -2*np.cos(2*np.pi*np.arange(8)/8)
    fermi = 1/(1+np.exp(beta*energy))
    result = pairing._static_lindhard(energy, beta, (8, 1, 1))
    assert result[0] == pytest.approx(np.mean(beta*fermi*(1-fermi)))
    assert result[1] == pytest.approx(result[-1])


@pytest.mark.parametrize("key,value", [
    ("spin_convention", "spinor_soc"), ("u_ev", 0), ("mu_ev", float("nan")),
    ("temperature_k", -1), ("kmesh", [1, 1, 1]), ("matsubara_points", 3),
    ("double_counting", ""), ("channels", ["arbitrary"]),
])
def test_model_choices_are_explicit(key, value):
    parameters = _parameters()
    parameters[key] = value
    with pytest.raises(ValueError):
        pairing.validate_parameters(parameters)


def test_preparation_preserves_inputs_without_material_claim(tmp_path):
    source = _hr(tmp_path)
    destination = tmp_path / "run"
    request = pairing.prepare_run(source, destination, _parameters(), origin="wannier_unvalidated")
    assert (destination / "hamiltonian_hr.dat").read_bytes() == source.read_bytes()
    assert json.loads((destination / "request.json").read_text()) == request
    assert request["origin"] == "wannier_unvalidated"
    assert request["backend"]["name"] == "numpy"
    with pytest.raises(FileExistsError):
        pairing.prepare_run(source, destination, _parameters(), origin="model")
    with pytest.raises(ValueError, match="separately validated"):
        pairing.prepare_run(source, tmp_path / "unfounded", _parameters(), origin="validated_material")


def test_frequency_padding_and_periodic_momentum_against_explicit_sum():
    mesh, nf = (4, 3, 1), 6
    rng = np.random.default_rng(11)
    vertex = rng.normal(size=(2*nf-1, *mesh))
    value = rng.normal(size=(nf, *mesh))
    expected = np.zeros_like(value)
    for f, *k in np.ndindex(value.shape):
        for g, *q in np.ndindex(value.shape):
            difference = tuple((a-b) % n for a, b, n in zip(k, q, mesh))
            expected[(f, *k)] += vertex[(f-g+nf-1, *difference)] * value[(g, *q)]
    actual = pairing._convolution(vertex, mesh, nf)(value).reshape(value.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=1e-13)


@pytest.mark.parametrize("sign", [1, -1])
def test_symmetry_basis_is_orthonormal_and_physical(sign):
    dimension, expand, compress, inverse = pairing._symmetry_basis(4, (4, 3, 1), sign)
    vector = np.random.default_rng(19).normal(size=dimension)
    full = expand(vector)
    np.testing.assert_allclose(compress(full), vector, atol=1e-14)
    assert np.linalg.norm(full) == pytest.approx(np.linalg.norm(vector))
    np.testing.assert_allclose(full[::-1], full)
    np.testing.assert_allclose(full[:, inverse], sign*full)


def test_dynamic_bubble_matches_complex_lindhard_expression():
    mesh, beta = (4, 2, 1), 3.0
    k = np.array(list(np.ndindex(*mesh)))
    energy = -2*np.cos(2*np.pi*k/np.array(mesh)).sum(axis=1)
    result = pairing._dynamic_lindhard(energy, beta, mesh, 8)
    fermi = 1/(1+np.exp(beta*energy))
    for q in [1, 3, 5]:
        shifted = np.array([np.ravel_multi_index(tuple((x+k[q]) % np.array(mesh)),mesh) for x in k])
        for m in [1, 4, 7]:
            expected = np.mean((fermi-fermi[shifted]) /
                               (energy[shifted]-energy-2j*np.pi*m/beta))
            assert result[m,q] == pytest.approx(expected, abs=1e-14)


def test_numpy_backend_matches_independent_dense_matrix_reference(tmp_path):
    source = tmp_path / "square_hr.dat"
    source.write_text("Square lattice t=1 eV\n1\n5\n1 1 1 1 1\n"
                      "0 0 0 1 1 0 0\n1 0 0 1 1 -1 0\n-1 0 0 1 1 -1 0\n"
                      "0 1 0 1 1 -1 0\n0 -1 0 1 1 -1 0\n")
    parameters = {**_parameters(), "temperature_k": 1/pairing.KB_EV_K,
                  "kmesh": [4,4,1], "matsubara_points": 32, "channels": ["singlet"]}
    destination = tmp_path / "run"
    pairing.prepare_run(source, destination, parameters, origin="model")
    result = pairing.solve_run(destination)
    assert result["backend"]["name"] == "numpy"
    assert result["channels"][0]["eigenvalue"] == pytest.approx(0.0014492556824397813, abs=1e-12)
    assert result["channels"][0]["relative_residual"] < 1e-8
    assert result["material_prediction_accepted"] is False
    assert (destination / "pairing.npz").is_file()


def test_changed_hamiltonian_cannot_use_frozen_request(tmp_path):
    source = _hr(tmp_path)
    destination = tmp_path / "run"
    pairing.prepare_run(source, destination, _parameters(), origin="model")
    (destination / "hamiltonian_hr.dat").write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Hamiltonian changed"):
        pairing.solve_run(destination)
