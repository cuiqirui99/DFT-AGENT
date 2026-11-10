# Spin states and annealing

UppASD can relax a supplied spin Hamiltonian at zero temperature or anneal it through an ordered temperature schedule. Atomic positions and moment magnitudes stay fixed. These operations change spin directions, not the crystal structure.

## Relax a spin state

> Relax this spin model at 0 K. Start with random directions, use damping 1 and a 1 fs time step, and run 20,000 steps.

`spin_relax` checks the perpendicular effective field, change in spin direction and change in energy over the last samples. The defaults are 0.001 T, 0.0001 in unit-spin direction and 0.00001 meV per spin. A run that ends before reaching these tolerances needs attention. The final state and raw data remain available.

Passing these checks establishes a stationary state. It does not establish the global minimum or stability against every perturbation. Compare several initial states when looking for the ground state.

## Anneal sequentially

> Anneal this model from 500 to 100 to 0 K, with 10,000 Monte Carlo sweeps at each temperature. Keep the final spins from each phase as the starting state for the next.

`spin_anneal` accepts an ordered `annealing_schedule` with a temperature and step count for each phase. It runs either Monte Carlo or spin dynamics, controlled by `annealing_method`. Every phase starts from the complete native checkpoint of the preceding phase. This differs from an independent temperature scan.

For Monte Carlo, steps are sweeps. For spin dynamics, steps are time steps. Completing the schedule does not by itself establish thermal equilibration or statistical convergence.

## Choose or reuse spins

Initial states can be uniform, random, within a cone, or specified for every magnetic site in the full supercell. Spin spirals must fit the periodic cell. A single domain wall requires an open boundary along its normal. Random states use the supplied seed.

These initial states also work for ordinary spin dynamics and Monte Carlo runs.

A saved `spin-state.json` can be reused in a later task. The Agent checks the cell, boundaries, magnetic-site order and moment magnitudes. It retains every supercell spin rather than reducing the state to one crystallographic cell. A new Hamiltonian with the same geometry and moments may be used deliberately.

## Results

- `final-spins.csv`: coordinates, moment magnitude and direction for every magnetic spin.
- `spin-state.json` and `final.restart`: the complete final state for further work.
- `spin-protocol.csv`: magnetization and energy through the temperature phases.
- `spin-convergence.csv`: the relaxation checks, when applicable.
- PNG, PDF and SVG plots, plus a ZIP of the native output from every phase.

The spin figure shows at most 2,000 arrows for readability; the state files and tables always contain every spin. Energy and magnetization summaries use the last recorded sample, which may precede the final checkpoint by one sampling interval.

Native acceptance includes analytic field alignment, refusal of an unfinished relaxation, sequential Monte Carlo and spin-dynamics anneals, checkpoint reuse, and a two-basis spin spiral with distinct moment magnitudes. These are numerical controls, not a material-specific accuracy benchmark. They ran as UppASD 6.0.2 Slurm jobs; see [0.5.0 checks](validation-0.5.0.md#rspt-and-uppasd).
