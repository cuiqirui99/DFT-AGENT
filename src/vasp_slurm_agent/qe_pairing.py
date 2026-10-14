"""Single-band Hubbard RPA pairing with NumPy/SciPy or optional TRIQS/TPRF.

This is a model solver. A Wannier file alone does not establish a material Tc.
The singlet vertex follows TPRF's theory equation ``singlet_gamma_2``; its
published Python helper has a different density-channel sign and is not used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


THEORY = "https://triqs.github.io/tprf/3.3.x/theory/eliashberg.html"
SUPPORTED_VERSION = "3.3.1"
KB_EV_K = 8.617333262145e-5


def read_hoppings(path):
    """Read a spin-independent one-orbital Wannier90 hr file, in eV."""
    lines = Path(path).read_text().splitlines()
    try:
        orbitals, count = int(lines[1]), int(lines[2])
        if orbitals != 1:
            raise ValueError("Pairing currently requires one spin-degenerate Wannier orbital.")
        if count < 1 or count > 100000:
            raise ValueError("Invalid hopping count.")
        end = 3 + math.ceil(count / 15)
        weights = [int(x) for line in lines[3:end] for x in line.split()]
        if len(weights) != count or min(weights) < 1 or len(lines[end:]) != count:
            raise ValueError("Incomplete Wannier hopping data.")
        hopping = {}
        for line, weight in zip(lines[end:], weights):
            parts = line.split()
            if len(parts) != 7 or [int(x) for x in parts[3:5]] != [1, 1]:
                raise ValueError("Invalid Wannier matrix element.")
            vector = tuple(int(x) for x in parts[:3])
            value = complex(float(parts[5]), float(parts[6])) / weight
            if vector in hopping or not np.isfinite(value):
                raise ValueError("Duplicate or nonfinite hopping.")
            hopping[vector] = value
    except (IndexError, OverflowError) as exc:
        raise ValueError("Incomplete Wannier hopping data.") from exc
    for vector, value in hopping.items():
        opposite = tuple(-x for x in vector)
        if opposite not in hopping or not np.isclose(value, hopping[opposite].conjugate(), atol=1e-9, rtol=1e-8):
            raise ValueError("The Wannier Hamiltonian is not Hermitian.")
        if abs(value.imag) > 1e-9:
            raise ValueError("This pairing route requires time-reversal-invariant real hoppings.")
    return {vector: float(value.real) for vector, value in hopping.items()}


def validate_parameters(parameters):
    """Require the interacting model and its numerical controls explicitly."""
    required = {"lattice_angstrom", "orbital", "spin_convention", "u_ev", "mu_ev",
                "temperature_k", "kmesh", "matsubara_points", "channels", "double_counting"}
    if set(parameters) != required:
        raise ValueError("Specify lattice, orbital, spin, U, chemical potential, temperature, grids, channels and double counting.")
    if parameters["spin_convention"] != "spin_degenerate_no_soc":
        raise ValueError("This route requires a paramagnetic, spin-degenerate Hamiltonian without SOC.")
    if not all(isinstance(parameters[key], str) and parameters[key].strip()
               for key in ("orbital", "double_counting")):
        raise ValueError("Name the orbital and the double-counting treatment.")
    lattice = np.asarray(parameters["lattice_angstrom"], dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-8:
        raise ValueError("Provide three independent lattice vectors in angstrom.")
    for key in ("u_ev", "mu_ev", "temperature_k"):
        if isinstance(parameters[key], bool) or not isinstance(parameters[key], (float, int)) or not math.isfinite(parameters[key]):
            raise ValueError(f"{key} must be finite.")
    if parameters["u_ev"] <= 0 or parameters["temperature_k"] <= 0:
        raise ValueError("Repulsive U and temperature must be positive.")
    mesh = parameters["kmesh"]
    if not isinstance(mesh, list) or len(mesh) != 3 or any(type(x) is not int or x < 1 for x in mesh):
        raise ValueError("Use three positive integer k-grid sizes.")
    if math.prod(mesh) < 4:
        raise ValueError("Use at least four k points.")
    nw = parameters["matsubara_points"]
    if type(nw) is not int or nw < 32:
        raise ValueError("Use at least 32 positive Matsubara frequencies.")
    channels = parameters["channels"]
    if not isinstance(channels, list) or not channels or len(set(channels)) != len(channels) or set(channels) - {"singlet", "triplet"}:
        raise ValueError("Choose singlet, triplet, or both.")
    return json.loads(json.dumps(parameters))


def pairing_vertex(chi0, u_ev, channel):
    """Return Γ = 2 V in TPRF's -Γ GG Δ / (2 β N) convention."""
    chi0 = np.asarray(chi0, dtype=complex)
    if not np.isfinite(chi0).all() or not math.isfinite(u_ev) or u_ev <= 0:
        raise ValueError("Invalid RPA susceptibility or U.")
    if channel not in {"singlet", "triplet"}:
        raise ValueError("Unknown pairing channel.")
    if np.any(np.abs(1 - u_ev * chi0) < 1e-8) or np.any(np.abs(1 + u_ev * chi0) < 1e-8):
        raise ValueError("The RPA susceptibility has a pole.")
    magnetic = chi0 / (1 - u_ev * chi0)
    density = chi0 / (1 + u_ev * chi0)
    if channel == "singlet":
        return 2 * u_ev + u_ev ** 2 * (3 * magnetic.conjugate() - density.conjugate())
    return -u_ev ** 2 * (magnetic.conjugate() + density.conjugate())


def prepare_run(hr_file, destination, parameters, *, origin, backend="numpy"):
    """Freeze explicit model inputs; never infer a material provenance claim."""
    parameters = validate_parameters(parameters)
    read_hoppings(hr_file)
    if origin not in {"model", "wannier_unvalidated"}:
        raise ValueError("A material calculation requires a separately validated QE/Wannier handoff.")
    if backend not in {"numpy", "tprf"}:
        raise ValueError("Choose the NumPy or TPRF pairing backend.")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    content = Path(hr_file).read_bytes()
    (root / "hamiltonian_hr.dat").write_bytes(content)
    request = {"method": "single_band_hubbard_rpa", "origin": origin,
               "parameters": parameters, "hamiltonian_sha256": hashlib.sha256(content).hexdigest(),
               "backend": {"name": backend, **({"triqs": SUPPORTED_VERSION, "triqs_tprf": SUPPORTED_VERSION}
                                               if backend == "tprf" else {})}, "source": THEORY}
    (root / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    return request


def _static_lindhard(energies, beta, mesh):
    """Independent Fermi-function sum, one spin species, on the same grid."""
    energy = np.asarray(energies).reshape(mesh)
    fermi = 0.5 * (1 - np.tanh(0.5 * beta * energy))
    result = np.empty(mesh)
    for q in np.ndindex(*mesh):
        shifted = np.roll(energy, tuple(-x for x in q), axis=(0, 1, 2))
        shifted_fermi = np.roll(fermi, tuple(-x for x in q), axis=(0, 1, 2))
        difference = shifted - energy
        ratio = beta * fermi * (1 - fermi)
        np.divide(fermi - shifted_fermi, difference, out=ratio, where=np.abs(difference) > 1e-10)
        result[q] = ratio.mean()
    return result.ravel()


def _solve_tprf_run(run_dir):
    """Run the optional backend and retain full gap and susceptibility arrays."""
    import triqs.version
    import triqs_tprf.version
    from triqs.gf import MeshImFreq
    from triqs.lattice.tight_binding import TBLattice
    from triqs_tprf.lattice import lattice_dyson_g0_wk, eliashberg_product_fft
    from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
    from triqs_tprf.eliashberg import solve_eliashberg, preprocess_gamma_for_fft, semi_random_initial_delta

    root = Path(run_dir)
    request = json.loads((root / "request.json").read_text())
    parameters = validate_parameters(request["parameters"])
    actual_hash = hashlib.sha256((root / "hamiltonian_hr.dat").read_bytes()).hexdigest()
    if actual_hash != request["hamiltonian_sha256"]:
        raise ValueError("The frozen Hamiltonian changed.")
    versions = {"triqs": triqs.version.version, "triqs_tprf": triqs_tprf.version.version}
    if any(value != SUPPORTED_VERSION for value in versions.values()):
        raise ValueError("Use the pinned TRIQS and TPRF 3.3.1 runtime.")
    hoppings = read_hoppings(root / "hamiltonian_hr.dat")
    lattice = TBLattice(units=parameters["lattice_angstrom"],
                        hoppings={key: np.array([[value]], complex) for key, value in hoppings.items()},
                        orbital_names=[parameters["orbital"]])
    mesh = tuple(parameters["kmesh"])
    for axis, size in enumerate(mesh):
        if size == 1 and any(vector[axis] and abs(value) > 1e-10 for vector, value in hoppings.items()):
            raise ValueError("Sample every dispersive direction with more than one k point.")
    energy = lattice.fourier(lattice.get_kmesh(mesh))
    beta = 1 / (KB_EV_K * parameters["temperature_k"])
    nw = parameters["matsubara_points"]
    green = lattice_dyson_g0_wk(mu=parameters["mu_ev"], e_k=energy,
                               mesh=MeshImFreq(beta=beta, S="Fermion", n_max=nw))
    bubble = imtime_bubble_chi0_wk(green, nw=nw)
    zero = bubble.mesh[0].to_data_index(0)
    chi0 = bubble.data[..., 0, 0, 0, 0]
    expected = _static_lindhard(energy.data[:, 0, 0].real - parameters["mu_ev"], beta, mesh)
    lindhard_error = float(np.max(np.abs(chi0[zero] - expected)))
    if lindhard_error > max(1e-6, 1e-4 * np.max(expected)):
        raise ValueError("The Matsubara bubble fails the independent static Lindhard check. Increase the frequency cutoff.")
    stoner = float(np.max(expected) * parameters["u_ev"])
    if stoner >= 1:
        raise ValueError("The assumed paramagnetic state is RPA-unstable. Do not interpret its pairing eigenvalues.")
    results, arrays = [], {"energy_ev": energy.data[:, 0, 0].real, "chi0": chi0,
                           "k_cartesian": np.array([list(point.value) for point in energy.mesh]),
                           "fermionic_frequency_ev": np.array([complex(point.value).imag for point in green.mesh[0]])}
    inverse = np.array([np.ravel_multi_index(tuple((-i) % n for i, n in zip(index, mesh)), mesh)
                        for index in np.ndindex(*mesh)])
    for channel in parameters["channels"]:
        parity = "even" if channel == "singlet" else "odd"
        sign = 1 if channel == "singlet" else -1

        def symmetrize(gap):
            projected = gap.copy()
            projected.data[:] = (gap.data + gap.data[::-1]) / 2
            projected.data[:] = (projected.data + sign * projected.data[:, inverse]) / 2
            return projected

        vertex = bubble.copy()
        vertex.data[..., 0, 0, 0, 0] = pairing_vertex(chi0, parameters["u_ev"], channel)
        values, gaps = solve_eliashberg(vertex, green, k=1, tol=1e-10, symmetrize_fct=symmetrize,
                                       initial_delta=symmetrize(semi_random_initial_delta(green, seed=1729)))
        gap = gaps[0]
        value = complex(values[0])
        dynamic, constant = preprocess_gamma_for_fft(vertex)
        applied = eliashberg_product_fft(dynamic, constant, green, gap)
        norm = float(np.linalg.norm(gap.data))
        unprojected_residual = float(np.linalg.norm(applied.data - value * gap.data) / norm)
        residual = float(np.linalg.norm(symmetrize(applied).data - value * gap.data) / norm)
        relative_residual = residual / max(abs(value), 1e-8)
        symmetry_error = float(max(np.linalg.norm(gap.data[::-1] - gap.data),
                                   np.linalg.norm(gap.data[:, inverse] - sign * gap.data)) / norm)
        if not np.isfinite([value.real, value.imag, residual, symmetry_error]).all() or abs(value.imag) > 1e-7 or relative_residual > 1e-6 or symmetry_error > 1e-7:
            raise ValueError(f"The {channel} eigenvector fails its residual or exchange-symmetry check: "
                             f"eigenvalue={value}, residual={residual:.6g}, symmetry={symmetry_error:.6g}.")
        arrays[f"gap_{channel}"] = gap.data[..., 0, 0]
        results.append({"channel": channel, "frequency_parity": "even", "momentum_parity": parity,
                        "eigenvalue": value.real, "residual": residual,
                        "relative_residual": relative_residual,
                        "unprojected_fft_residual": unprojected_residual, "symmetry_error": symmetry_error})
    np.savez_compressed(root / "pairing.npz", **arrays)
    result = {"method": request["method"], "origin": request["origin"], "backend": versions,
              "temperature_k": parameters["temperature_k"], "hamiltonian_sha256": actual_hash,
              "static_lindhard_max_error_per_ev": lindhard_error, "stoner_factor": stoner,
              "filling_per_cell": float(2 * np.mean(0.5 * (1 - np.tanh(0.5 * beta * (arrays["energy_ev"] - parameters["mu_ev"]))))),
              "channels": results, "numerical_checks_passed": True, "material_prediction_accepted": False,
              "limits": ["One spin-degenerate orbital; paramagnetic Hubbard RPA without SOC.",
                         "No self-energy feedback. U, filling and double counting define the model.",
                         "A temperature point is not Tc; grid, frequency and temperature convergence remain necessary.",
                         "A two-dimensional pairing scale is not a phase-ordering temperature."],
              "source": THEORY}
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _dynamic_lindhard(energies, beta, mesh, frequency_count):
    """Exact band sum at nonnegative bosonic Matsubara frequencies, per spin."""
    energy = np.asarray(energies).reshape(mesh)
    fermi = 0.5 * (1 - np.tanh(0.5 * beta * energy))
    omega = 2 * np.pi * np.arange(frequency_count) / beta
    result = np.empty((frequency_count, math.prod(mesh)))
    for index, q in enumerate(np.ndindex(*mesh)):
        difference = (np.roll(energy, tuple(-x for x in q), axis=(0, 1, 2)) - energy).ravel()
        numerator = (fermi - np.roll(fermi, tuple(-x for x in q), axis=(0, 1, 2))).ravel()
        static = (beta * fermi * (1 - fermi)).ravel()
        np.divide(numerator, difference, out=static, where=np.abs(difference) > 1e-10)
        result[0, index] = static.mean()
        result[1:, index] = np.mean(numerator * difference /
                                    (omega[1:, None] ** 2 + difference ** 2), axis=1)
    return result


def _convolution(interaction, mesh, frequency_count):
    """Linear frequency convolution and periodic momentum convolution."""
    from scipy.fft import fftn, ifftn, next_fast_len
    length = next_fast_len(3 * frequency_count - 2)
    shape = (length, *mesh)
    vertex = np.asarray(interaction).reshape(2 * frequency_count - 1, *mesh)
    transformed = fftn(vertex, s=shape)

    def apply(values):
        data = np.asarray(values).reshape(frequency_count, *mesh)
        product = ifftn(transformed * fftn(data, s=shape))
        return product[frequency_count - 1:2 * frequency_count - 1].reshape(frequency_count, -1).real

    return apply


def _symmetry_basis(nw, mesh, sign):
    """An orthonormal basis for even frequency and the chosen momentum parity."""
    count = math.prod(mesh)
    inverse = np.array([np.ravel_multi_index(tuple((-i) % n for i, n in zip(index, mesh)), mesh)
                        for index in np.ndindex(*mesh)])
    full, reduced, coefficient = [], [], []
    dimension = 0
    for frequency in range(nw, 2 * nw):
        for k in range(count):
            opposite = inverse[k]
            if k > opposite or (sign == -1 and k == opposite):
                continue
            members = [(frequency, k, 1), (2 * nw - 1 - frequency, k, 1)]
            if k != opposite:
                members += [(frequency, opposite, sign), (2 * nw - 1 - frequency, opposite, sign)]
            for f, point, phase in members:
                full.append(f * count + point)
                reduced.append(dimension)
                coefficient.append(phase / math.sqrt(len(members)))
            dimension += 1
    if dimension == 0:
        raise ValueError("This grid has no states in the requested pairing symmetry.")
    full, reduced, coefficient = np.array(full), np.array(reduced), np.array(coefficient)

    def expand(values):
        result = np.zeros(2 * nw * count)
        result[full] = np.asarray(values)[reduced] * coefficient
        return result.reshape(2 * nw, count)

    def compress(values):
        return np.bincount(reduced, weights=np.asarray(values).ravel()[full] * coefficient, minlength=dimension)

    return dimension, expand, compress, inverse


def _solve_numpy_run(run_dir):
    import scipy
    from scipy.sparse.linalg import LinearOperator, eigsh

    root = Path(run_dir)
    request = json.loads((root / "request.json").read_text())
    parameters = validate_parameters(request["parameters"])
    actual_hash = hashlib.sha256((root / "hamiltonian_hr.dat").read_bytes()).hexdigest()
    if actual_hash != request["hamiltonian_sha256"]:
        raise ValueError("The frozen Hamiltonian changed.")
    hoppings = read_hoppings(root / "hamiltonian_hr.dat")
    mesh = tuple(parameters["kmesh"])
    for axis, size in enumerate(mesh):
        if size == 1 and any(vector[axis] and abs(value) > 1e-10 for vector, value in hoppings.items()):
            raise ValueError("Sample every dispersive direction with more than one k point.")
    fractional = np.array(list(np.ndindex(*mesh))) / np.array(mesh)
    energy = sum(value * np.cos(2 * np.pi * fractional @ np.array(vector)) for vector, value in hoppings.items())
    xi = energy - parameters["mu_ev"]
    beta = 1 / (KB_EV_K * parameters["temperature_k"])
    nw = parameters["matsubara_points"]
    frequencies = (2 * np.arange(-nw, nw) + 1) * np.pi / beta
    static = _static_lindhard(xi, beta, mesh)
    stoner = float(np.max(static) * parameters["u_ev"])
    if stoner >= 1:
        raise ValueError("The assumed paramagnetic state is RPA-unstable. Do not interpret its pairing eigenvalues.")
    positive_chi = _dynamic_lindhard(xi, beta, mesh, 2 * nw)
    static_error = float(np.max(np.abs(positive_chi[0] - static)))
    chi0 = np.concatenate((positive_chi[:0:-1], positive_chi), axis=0)
    weight = 1 / (beta * math.prod(mesh) * (frequencies[:, None] ** 2 + xi[None, :] ** 2))
    square_root = np.sqrt(weight)
    arrays = {"energy_ev": energy, "chi0": chi0, "fermionic_frequency_ev": frequencies,
              "bosonic_frequency_ev": 2 * np.pi * np.arange(-2 * nw + 1, 2 * nw) / beta,
              "k_cartesian": fractional @ (2 * np.pi * np.linalg.inv(parameters["lattice_angstrom"]).T)}
    channels = []
    for channel in parameters["channels"]:
        sign = 1 if channel == "singlet" else -1
        convolution = _convolution(pairing_vertex(chi0, parameters["u_ev"], channel).real / 2, mesh, 2 * nw)
        dimension, expand, compress, inverse = _symmetry_basis(nw, mesh, sign)

        def apply_full(values):
            return -square_root * convolution(square_root * values)

        def apply_reduced(values):
            return compress(apply_full(expand(values)))

        if dimension == 1:
            value, vector = float(apply_reduced(np.ones(1))[0]), np.ones(1)
        else:
            operator = LinearOperator((dimension, dimension), matvec=apply_reduced, dtype=float)
            values, vectors = eigsh(operator, k=1, which="LA", tol=1e-10,
                                    v0=np.random.default_rng(1729).normal(size=dimension))
            value, vector = float(values[0]), vectors[:, 0]
        weighted = expand(vector)
        gap = weighted / square_root
        gap /= np.linalg.norm(gap)
        applied = -convolution(weight * gap)
        residual = float(np.linalg.norm(applied - value * gap))
        relative = residual / max(abs(value), 1e-8)
        symmetry_error = float(max(np.linalg.norm(gap[::-1] - gap),
                                   np.linalg.norm(gap[:, inverse] - sign * gap)))
        if not np.isfinite([value, residual, relative, symmetry_error]).all() or relative > 1e-6 or symmetry_error > 1e-7:
            raise ValueError("The pairing eigenvector fails its residual or exchange-symmetry check.")
        arrays[f"gap_{channel}"] = gap
        channels.append({"channel": channel, "frequency_parity": "even",
                         "momentum_parity": "even" if sign == 1 else "odd", "eigenvalue": value,
                         "residual": residual, "relative_residual": relative,
                         "unprojected_fft_residual": residual, "symmetry_error": symmetry_error})
    np.savez_compressed(root / "pairing.npz", **arrays)
    result = {"method": request["method"], "origin": request["origin"],
              "backend": {"name": "numpy", "numpy": np.__version__, "scipy": scipy.__version__},
              "temperature_k": parameters["temperature_k"], "hamiltonian_sha256": actual_hash,
              "static_lindhard_max_error_per_ev": static_error, "stoner_factor": stoner,
              "filling_per_cell": float(np.mean(1 - np.tanh(0.5 * beta * xi))),
              "channels": channels, "numerical_checks_passed": True, "material_prediction_accepted": False,
              "limits": ["One spin-degenerate orbital; paramagnetic Hubbard RPA without SOC.",
                         "No self-energy feedback. U, filling and double counting define the model.",
                         "Finite Matsubara sum; verify frequency and momentum convergence.",
                         "A temperature point is not Tc; a two-dimensional pairing scale is not a phase-ordering temperature."],
              "source": THEORY}
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def solve_run(run_dir):
    """Run the frozen backend; retain numerical evidence without a material claim."""
    request = json.loads((Path(run_dir) / "request.json").read_text())
    backend = request["backend"].get("name", "tprf")
    if backend == "numpy":
        return _solve_numpy_run(run_dir)
    if backend == "tprf":
        return _solve_tprf_run(run_dir)
    raise ValueError("Unknown pairing backend.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a prepared single-band RPA pairing model.")
    parser.add_argument("run_dir")
    print(json.dumps(solve_run(parser.parse_args().run_dir), indent=2))
