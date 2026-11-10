# UppASD

UppASD simulates classical spins from a supplied spin Hamiltonian. DFT Agent can run spin dynamics, Monte Carlo, spin relaxation, sequential annealing, magnon spectra and spin correlations. It does not infer exchange interactions from a structure alone.

## Set up

Connect your cluster. After sign-in the app looks for an existing UppASD. When it is missing, **Connections → Software** offers **Install** for it, or you can tick **Install missing open-source software automatically**, which installs it when the app can read your storage quota. The app then fetches the pinned UppASD 6.0.2 source release, builds it in your own directory without administrator access and runs a small test. It checks the GNU compilers, CMake and BLAS/LAPACK first. If these are missing, load the required cluster modules in the UppASD installer under **Connections → Cluster → Advanced installation**. The installer accepts Linux x86_64 and ARM64 clusters; the build has been tested on Linux ARM64 only.

You can also add an existing executable under **Manual setup**, with a command such as `srun /path/to/sd`. The executable may be named `sd.f95`. Use one node and one task; set CPU threads through **CPUs per task**. Check the environment before submission.

[UppASD](https://github.com/UppASD/UppASD) is distributed under GPL-3.0. Its source and executable are installed separately from DFT Agent.

## Use in chat

Start with an accepted RSPt exchange result, or import a spin Hamiltonian with its source and units:

> Use the RSPt exchange result for an UppASD Monte Carlo scan at 10, 300 and 1500 K. Use a 6 × 6 × 6 supercell and show magnetization and energy.

> Relax this spin model at 0 K, then anneal it from 500 to 100 to 0 K. Keep each phase's final spins for the next phase.

Review the Hamiltonian, cell, boundaries, initial spins and sampling before submitting. Revise them in the same conversation. An independent temperature scan restarts each temperature from the chosen initial state; annealing carries the complete spin configuration between temperatures.

[Spin states and annealing](uppasd-spins.md) describes initial textures, convergence checks and checkpoint reuse. [Spin waves and correlations](uppasd-spectra.md) describes magnon bands, DOS and S(q,ω).

## Native inputs

For methods outside the guided tasks, `uppasd_native` accepts a complete input deck with its source, required outputs and chosen plots. This keeps native controls for DMI, general exchange interactions, spin torques, spin–lattice dynamics and temperature models. All required model files must be supplied. The Agent checks execution and requested output files; method-specific physical validation remains necessary.

A native tutorial is a starting point, not automatically a material model. For example, UppASD's bcc Fe three-temperature example explicitly uses truncated exchange, no spin–lattice coupling and some parameters chosen for demonstration.

## Results

Results include tables, PNG/PDF/SVG plots and a ZIP of the native output. Spin dynamics also reports time in femtoseconds. Guided spin energies are in meV per magnetic spin and magnetization is in μB per magnetic spin.

The last temperature's complete spin configuration is available as CSV, JSON and a native restart, with an arrow plot. Every temperature's native files remain in the archive. Older runs without a saved snapshot keep their measured results and show that the snapshot is unavailable.

The guided Hamiltonian uses fixed-length classical spins with scalar or tensor exchange, an optional field and uniaxial anisotropy. Scalar exchange uses `H = -sum_ij J_ij e_i · e_j`, with both directed bonds present. Positive J favours parallel spins. For `H_anisotropy = +sum_i K_i (e_i · axis_i)^2`, negative K gives an easy axis. Site order and geometry must match the structure.

Full tensors use `H = -sum_ij e_i^T J_ij e_j`, with `J_ji = transpose(J_ij)` for reverse bonds. They include the isotropic, symmetric anisotropic and DMI terms together. Supply `exchange_tensor` instead of `exchange`; adding both would count the isotropic term twice. Tensor models work with dynamics, Monte Carlo, relaxation, annealing and finite-temperature correlations. Harmonic magnon bands currently require scalar exchange.

For RSPt tensors, use the spin-model file returned by `combine_exchange_tensor`. The Agent rechecks the three source calculations before preparing an UppASD run. Other interactions, such as biquadratic exchange and dipolar terms, use native inputs.

A completed run does not establish statistical convergence. Transition temperatures need equilibration, longer sampling, several seeds and cell sizes. Antiferromagnets also need an appropriate staggered order parameter; a small net moment alone does not establish loss of order. The Agent does not fit a transition temperature from a short scan or identify a global ground state from one relaxed configuration.
