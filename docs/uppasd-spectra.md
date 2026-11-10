# Spin waves and correlations

Supply a spin Hamiltonian and a magnetic structure, then ask for a magnon spectrum or a finite-temperature spin-correlation calculation. Exchange values are never inferred from geometry.

For example: “Calculate the adiabatic magnon spectrum along these reciprocal points, and the DOS on a uniform mesh.” Use `magnon_spectrum` with `q_points` or `q_mesh`. Points are fractional reciprocal coordinates of the supplied magnetic cell. `dos_mesh` adds a separate uniform sampling for the DOS; `dos_sigma_mev` controls its Gaussian broadening. A plotted path alone is not a DOS.

For a labelled magnon or finite-temperature correlation path, use `q_path`, for example `[{"label":"Γ","q":[0,0,0]},{"label":"X","q":[0.5,0,0]}]`, and `q_points_per_segment` (default 41). Every intermediate point is sent to the native calculation. The plot displays the computed samples using reciprocal-space distances and the supplied labels. Labels and coordinates must describe the supplied magnetic cell, whose bands may be folded relative to a smaller primitive cell. Choose exactly one of `q_path`, `q_points` and `q_mesh`.

Adiabatic spectra use a collinear reference. An antiferromagnet needs its full magnetic unit cell and explicit opposite initial directions. The adapter checks transverse stability at the sampled points because native UppASD reports absolute frequencies. It also checks the returned energies against the linearized scalar Hamiltonian. Parallel fields and easy-axis anisotropy are supported for aligned spins. Antiparallel references with anisotropy and noncollinear references require a full dynamical-matrix method, not this native AMS route.

UppASD 6.0.2 AMS uses a different uniaxial-anisotropy factor from its spin-dynamics routine. The adapter records and applies that input conversion while retaining the original physical Hamiltonian. A disagreement between native frequencies and the Hamiltonian is rejected.

For example: “At 10 K, calculate S(q,ω) on a 16×1×1 mesh, with 2,048 samples separated by 10 integration steps.” Use `spin_correlations`, `temperature_k`, `q_mesh`, `correlation_samples`, `correlation_interval` and `timestep_fs`. `equilibration_steps` precedes measurement. `steps` equals the sample count times the sampling interval. Windows are `box`, `hann`, `hamming` or `blackman_harris`; `connected` controls subtraction of the mean.

The correlation run returns dynamic and static structure-factor tables, plots and native files. A uniform mesh also gives the q-averaged spectral weight. Energy spacing is h divided by the sampled duration; the Nyquist energy is h divided by twice the sampling interval. The files retain the native diagonal correlation components in arbitrary units. They are not an absolute neutron-scattering cross section.

Correlation maps display the sampled bins with a shared logarithmic color scale. New runs export focused and full-range plots. The colorbar arrow marks saturated intensities, and CSV files retain every sampled value. A multidimensional q mesh uses mesh-point indices.

Completed correlation runs also save the final spin configuration. Ask to continue from that state with a different temperature or field. Older runs require a saved native snapshot for this transfer.

Check timestep, simulation size, equilibration and sampling duration before interpreting finite-temperature peaks. Peaks can include elastic scattering. Finite-window widths and damping do not establish magnon lifetimes. Completing a run does not establish a converged dispersion, transition temperature or magnetic ground state.

The implementation follows the [UppASD input reference](https://uppasd.github.io/UppASD-manual/input/) and [magnon tutorial](https://uppasd.github.io/UppASD-tutorial/magnon/magnon/). Native controls use UppASD 6.0.2.
