# Molecular dynamics

Open a project, choose a structure, and describe the temperature and duration:

> Run molecular dynamics on this structure at 500 K for 2 ps. Use QE and a 0.5 fs time step.

For a tested classical route:

> Equilibrate this silicon crystal at 300 K, then run NVE to check energy conservation. Choose a tested potential and prepare the inputs for review.

You can revise the plan before submitting it. Building a structure or asking for an MD plan does not start a calculation.

## Programs

VASP, Quantum ESPRESSO, CP2K, SIESTA and ABINIT support fixed-cell NVT and NVE dynamics. VASP and QE also preserve supported fixed atoms, vacuum directions and magnetic seeds. The other adapters reject those properties when they cannot preserve them.

LAMMPS supports classical dynamics with a listed interatomic potential: equilibration, melt–quench, stability checks, diffusion and two-phase melting protocols. It accepts fully periodic atomic systems. NPT is available in the LAMMPS protocols. Molecular force fields and machine-learned potential drivers are not integrated in this release.

The agent checks the potential's elements, composition, crystal type, temperature and intended use. Eight potentials have passed short crystalline tests at 300 K on two clusters, covering Si, SiC, quartz, Cu, Ni, equiatomic Cu–Ni, Fe and Al. Automatic selection stays within those tested conditions. Other conditions remain unvalidated; the agent can suggest a supported DFT route. Automatic DFT route advice is limited to 200 atoms; larger structures can still be built and edited.

## Results

The run view separates completion from numerical checks. It shows temperature, energy, pressure and density when available, with RDF and MSD plots where the trajectory supports them.

- Use **Trajectory** to inspect saved frames in the 3D viewer.
- Use **MD data** to download thermodynamic, RDF and MSD tables and the retained trajectory as extended XYZ, including cells and times.
- Raw engine files remain in the run downloads. Long trajectories may be thinned during collection; the viewer uses a further bounded preview.
- Ask follow-up questions using the same conversation. The agent uses recorded results and their checks; short or unsuitable trajectories do not receive a diffusion coefficient.

Bulk RDF and density estimates are omitted for cells with vacuum. Diffusion is omitted for changing cells or irregular sampling. Missing results remain missing.

## Continuing

A later relaxation or electronic calculation can use an accepted final structure. A new MD run from that structure usually initializes new velocities; it is not an exact restart.

VASP, CP2K and SIESTA can pass verified final positions and velocities into NVE dynamics. ABINIT supports this from NVE output with adjacent saved frames. Missing, changed or mismatched checkpoints stop preparation; velocities are never silently regenerated for a continuation. Thermostat state is not carried into NVE.

QE continues its native position-Verlet state at the same time step. Its verified restart files must remain on the original cluster. The charge density provides an SCF starting guess; every new step is solved again.

LAMMPS retains state between stages in one job. For a stopped run, ask the agent to list its MD checkpoints and continue from one. It prepares a separate task for review and keeps the old run. Native restart preserves velocities and supported thermostat or barostat state, and requires the same executable and MPI rank count. To extend a completed trajectory, plan a new MD stage.

The same operation is available from the [bundled CLI](install.md#command-line): `dft-agent restart-md SOURCE --list`, then `dft-agent restart-md SOURCE DEST --checkpoint NAME`. Neither command submits a calculation.

## Validation

Classical MD passed 60 runs and 20 timestep comparisons on Parallel Supercomputing and Arrhenius: force relaxation, 5 ps NVT followed by 5 ps NVE, and 20 ps isotropic NPT at zero pressure. Checks include independent static energies, energy conservation, temperature, composition and trajectory completion. The bundled acceptance records state the tested conditions. They do not establish accuracy for melting, glasses, diffusion or untested materials and temperatures.

Separate real calculations checked continuation in QE, CP2K, SIESTA, ABINIT and LAMMPS. Short continuity tests check state transfer, not thermodynamic equilibration. Local tests also check parsing, analysis, downloads and project conversations. Users still need convergence and sampling checks for their intended calculation.
