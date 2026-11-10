# RSPt

RSPt joins the project and chat workflow in version 0.5.0.
Use a cluster installation of RSPt; its source, program and manual are not bundled or downloaded by DFT Agent.

## Set up

1. Connect your cluster under **Connections**.
2. After sign-in the app looks for RSPt and checks the installation on the login node. If it is not found, enter the path to your `rspt` program under **Connections → Software**.
3. To set the launch command yourself, add **RSPt** under **Manual setup** with a command such as `srun /path/to/RSPt/bin/rspt`. Add any module commands in the environment settings.
4. Check the environment before submitting a calculation.

The RSPt installation must contain `rspt`, `symt`, `atom`, `cub` and `stick` in the same binary directory. The calculation host also needs Python 3 and `make`. No pseudopotential library is required. The app submits through Slurm or PBS, using the same connection and resource review as the other engines.

## Use in chat

Import or build a structure, then describe the calculation:

> Use RSPt to calculate the SCF energy and bands of this silicon structure.

> Use ferromagnetic order for this iron structure, include SOC along z, and calculate the bands and DOS.

You can revise the plan before submission:

> Change the spin axis to x. Keep the same k-point mesh and add DOS from −8 to 8 eV.

The supported tasks are fixed-geometry SCF, bands, total DOS and exchange with PBE, PBEsol or LDA. SCF and total DOS also support DFT+U; fixed-potential SPTF is a separate DMFT task. Magnetic moments share one spin axis; up/down seeds can describe ferromagnetic or specified antiferromagnetic arrangements. Seed magnitudes are not constrained moments. Bands combine both spin channels; DOS is the total density of states.

The Agent asks about missing magnetic arrangements. Relaxation, canted magnetic textures and projected spectra are not exposed by this adapter yet. The adapter does not support DFT+U band structures, DFT+U exchange or real-frequency DMFT spectra. A workflow can first relax the structure with another engine, then pass that structure to RSPt. Charge density and wavefunctions are not transferred between different engines.

## Results and continuation

Each calculation returns its native output. SCF includes energy, density residual, magnetic moment when present, and convergence history. Bands and DOS include CSV tables and PNG, PDF and SVG plots. Chat explanations use the checked results.

An SCF is accepted only when both its energy change and native `fsq` residual meet the reviewed tolerances. Band gaps refer to the sampled path; a path alone cannot establish the global gap. Convergence of the SCF does not establish convergence of the basis, mesh or magnetic configuration.

Bands, DOS and exchange in the same workflow reuse the accepted SCF checkpoint. Changing physical settings creates a matching SCF first. **Resume monitoring** reconnects to the same job; a new follow-up run currently starts a new SCF from the accepted structure rather than restarting an interrupted density iteration.

## Exchange and spin simulations

For a magnetic structure, ask:

> Calculate Fe d-orbital exchange with RSPt, using collinear ferromagnetic order and neighbours within 3.2 Å. Then use the result for an UppASD temperature scan.

Choose the magnetic sites, orbital manifold and neighbour radius before submission. Site indices are zero-based, matching the structure editor. The default projection is Löwdin; muffin-tin heads can be selected instead. The choice changes the local basis and can change the resulting exchange parameters.

Exchange uses a converged non-SOC collinear SCF potential and the same k-point grid expanded to the full Brillouin zone. Each selected magnetic site is evaluated as a central site. The output includes exchange versus distance, a CSV table, native input/output evidence and `spin-hamiltonian.json`. The handoff contains the original site mapping, lattice translations, measured projected moments and exchange in meV. Positive exchange favours parallel spins. Both directed bonds are included in the convention `H = -sum_ij J_ij e_i · e_j`.

The Agent exports an UppASD model only when every bond has a matching reverse bond. Converge the k-point grid, Matsubara sampling, orbital basis and neighbour range before drawing quantitative conclusions. Completing an exchange calculation or a short temperature scan does not establish a Curie temperature; see [UppASD](uppasd.md).

## SOC exchange

> Calculate the exchange tensor and DMI with RSPt, then use the result for UppASD spin dynamics.

The Agent plans matching SOC calculations with magnetization along Cartesian x, y and z. Each supplies its transverse exchange components. After all three finish, `combine_exchange_tensor` checks the geometry, settings, magnetic states and reverse bonds, then saves the full tensor and an UppASD spin model. The output includes a CSV table, figures and the source records.

Choose the projector explicitly. Löwdin projection can make SOC exchange much more expensive than muffin-tin heads. The two choices define different local bases and should be compared as separate calculations.

The three-axis construction assumes a common bilinear spin model. Its consistency checks detect incompatible results; k-point, basis and neighbour-range convergence still need testing. The method follows the native global-coordinate output and the [RSPt exchange extractor](https://github.com/aibergman/rspt-extractor).

## Magnetic anisotropy

Run matched SOC calculations with different spin axes, then ask:

> Compare the completed RSPt runs for magnetization along [001] and [111]. Use [001] as the reference and plot the energy difference.

The comparison checks the accepted SCF results, fixed geometry, magnetic starting state, numerical settings and program fingerprints. It returns energy differences per cell and per atom, a CSV table and PNG, PDF and SVG plots. Different meshes or methods must be compared in separate matched pairs. Historical runs without an executable fingerprint need to be rerun.

These are fixed-axis magnetocrystalline energy differences, without dipolar shape anisotropy. Check no-SOC rotations and converge the differences with k points, basis, smearing and SCF tolerances before interpreting an easy axis. The reported SCF and print-resolution indicator is not an error bound. DMFT-stage energies are not supported by this comparison.

## DMFT with SPTF

The adapter includes a fixed-potential SPTF route for moderately correlated metals. It first converges DFT at the requested temperature, then solves the local d-shell problem while holding the DFT charge density fixed. This is not charge-self-consistent DMFT, and SPTF is not suitable for a Mott insulator or arbitrary strong correlations.

For example:

> Use RSPt SPTF for the Fe d shell at 580 K, with U = 2 eV, J = 0.6 eV, SPTF double counting and muffin-tin-head projection. Start from ferromagnetic PBE and show the self-energy and convergence.

These example parameters define a workflow test, not universal values for iron. Choose U, J, projection and temperature for the material and model being studied. The current route uses the complete d shell, SPTF mode 1, the solver's SPTF double-counting prescription and no SOC. Both nonmagnetic and collinear starting states are accepted.

The Agent checks the native electron-count residual, self-energy residual, frequency mesh, temperature and diagonal causality. Results include Matsubara self-energy and convergence tables, plots and the native output archive. A converged loop does not establish convergence of the k mesh, Matsubara mesh or orbital basis. Matsubara self-energy is not a real-frequency spectrum; this route does not automatically infer spectral peaks or a Mott gap.

The method follows the [RSPt SPTF tutorial](https://github.com/RSPt-tutorials/day2-DFT_DMFT/tree/master/Fe). Other impurity solvers, including Hubbard-I, ED and CT-QMC, are not exposed by this route.

## DFT+U

DFT+U is available for fixed-geometry SCF and total DOS. For example:

> Use PBE+U for NiO with U = 8 eV and J = 1 eV on Ni d orbitals, FLL double counting and muffin-tin-head projection. Keep the specified antiferromagnetic order.

Choose U, J, the correlated orbitals, double counting (FLL or AMF) and projection explicitly. These are model choices, not values the Agent infers from the element. The native calculation first converges the DFT density, then the static Hartree–Fock self-energy and updated charge density.

Results report both the outer SCF and inner self-energy convergence. The displayed total energy already includes the native Hubbard correction; do not add the correction a second time. Correlated bands and exchange are not connected yet.

For the DOS, specify an energy window and broadening.

> Calculate the DFT+U DOS from −10 to 10 eV with 401 points and 0.1 eV broadening. Then repeat with 0.2 eV broadening using the same converged state.

The Green-function calculation reads the accepted static self-energy and keeps it, the potential and the charge density fixed. Energies are relative to the correlated chemical potential. The Agent checks these files and returns the native spectrum, CSV and plots. Broadening is the Lorentzian half-width η used to evaluate the Green function.

## Program documentation

- [RSPt at Uppsala University](https://www.uu.se/en/department/physics-and-astronomy/research/materials-theory/code-development)
- [NAISS installation and module information](https://hpc.pages.naiss.se/user-documentation/support-docs/applications/rspt/general/)

These describe the native program, whose capabilities extend beyond the tasks currently exposed by this adapter.
