# ABINIT, CP2K and Elk

DFT Agent can install, prepare and run three more open-source codes on your cluster, alongside VASP and Quantum ESPRESSO:

| Code | Version | Method | Library |
|---|---|---|---|
| [ABINIT](https://www.abinit.org/) | 10.0.3 | Plane waves, norm-conserving pseudopotentials | [PseudoDojo 0.4](https://www.pseudo-dojo.org/) psp8: PBE, PBE fully relativistic, PBEsol |
| [CP2K](https://www.cp2k.org/) | 2026.1 (managed, x86-64); 2026.2 (managed, Arm) | Gaussian and plane waves (GPW, Quickstep) | MOLOPT basis sets with GTH pseudopotentials from the CP2K 2026.2 data |
| [Elk](https://elk.sourceforge.io/) | 10.2.4 | All-electron full-potential LAPW | Species files from the Elk 10.2.4 release |

These codes are open source. Like QE, each is chosen per workflow step, and a later step can continue from a relaxed structure computed by any of the five codes.

## Calculations

Each code runs relaxation (atomic positions, optionally the cell), SCF, band structures along seekpath paths, and densities of states. The executor adds the SCF that bands and DOS need and passes its checkpoint to them.

| | ABINIT | CP2K | Elk |
|---|---|---|---|
| Functionals | PBE, PBEsol | PBE, LDA (PADE) | PBE, PBEsol, LDA |
| Collinear spin | Yes | Yes (UKS) | Yes |
| Noncollinear spin | Yes | No | Yes |
| Spin–orbit coupling | Yes (PBE) | No | Yes |
| DFT+U | No (needs PAW) | Γ-point cells only | Yes, with spin |
| Band gap reported from | Every stage | Bands and DOS stages | Every stage |

Each code also runs what it is best known for, described below. Hybrid functionals, NEB and GW are not offered for these codes yet. The planner says so instead of substituting another method, and names the DFT Agent programs that offer it.

### ABINIT phonons

The `phonons` task runs density-functional perturbation theory on a Γ-centred `qmesh` (default: half the k mesh), one dataset per irreducible q point, after a ground state converged to a potential residual of 1e-18. A nonmagnetic insulator without vacuum also gets the derivative with respect to k and the electric-field response at Γ. The same job then merges the derivative databases with mrgddb and runs anaddb, which interpolates the frequencies along the band path and onto `phonon_dos_mesh` for the DOS. Results are in cm⁻¹: the dispersion beside the DOS, the frequencies at Γ, the highest and lowest frequency, and a warning when frequencies below −10 cm⁻¹ show an unstable structure or too coarse meshes. For an insulator, anaddb also reports the electronic and relaxed-ion dielectric constants, each atom's Born charge, and how far the charges missed neutrality before it was imposed; the dipole term gives the LO–TO splitting, and Γ on the path is approached along the path. Meshes are Γ-centred. On an 8×8×8 mesh through Γ, where GaAs's PBE gap is 0.5 eV, its charges missed neutrality by 0.27 e and ε∞ came out as 29.7, about twice what QE gave on a half-step shifted mesh (13.6); the result warns about such a miss. ABINIT stopped on that shifted mesh because a symmetry of the fcc cell maps it outside itself, so `kshift` is QE-only. Spin-orbit and noncollinear phonons are not set up. On Arrhenius, silicon's optical mode came out at 514 cm⁻¹ (517 measured). Aluminium's longitudinal branch needed a 16×16×16 mesh with 0.27 eV broadening (highest frequency 315 cm⁻¹, as in QE): on 12×12×12 it came out 7 % higher.

### ABINIT molecular dynamics

The `md` task takes the same keys as CP2K's below and reports the same results: `md.csv`, `rdf.csv`, `msd.csv`, their plots, and `final.cif` and `CONTCAR` for the next stage. It starts from Maxwell–Boltzmann velocities at `md_temperature_k` without drift, and later stages continue from its last frame, so `["relax", "md"]` and the quench `["md", "relax", "scf", "bands"]` both work.

NVE runs velocity Verlet (`ionmov 24`). NVT runs a Nosé–Hoover chain of three thermostats (`ionmov 13`), with masses from `md_thermostat_fs` by Martyna, Tuckerman and Klein. ABINIT 10.0.3 has no CSVR. Of its other thermostats, `ionmov 12` holds the kinetic energy fixed, which is not canonical, and `ionmov 8` and `9` move the atoms with the force of the step before. With their coupling removed, 8 Si atoms at 1 fs gained 1.4 eV/atom of total energy in 200 fs and kept heating, where `ionmov 13` and `24` held it. Coupled at 100 fs on the same crystal, they averaged 496 and 481 K over 200–400 fs against a 300 K target, where the chain averaged 283 K. They also stop at once from a crystal at rest, whose forces already pass their force test.

Temperatures count 3N−3 degrees of freedom, because the velocities have no drift and keep none. The chain counts 3N, so its target is lowered to match: uncorrected, 8 atoms would sit 14 % too hot. `ionmov 24` files a velocity estimated with the previous force, which read Si8 at 304, 310 and 319 K at 1, 2 and 3 fs, so NVE takes Verlet's own velocity from the positions and forces instead; a run thinned to 2000 frames cannot, and keeps the filed one. The chain's velocities agree with central differences of the positions within 0.1 %, one step behind the frames, which does not affect averages.

**What was measured.** 8 Si atoms at 5.431 Å on a Γ-centred 2×2×2 mesh, 14 Ha, 8 ranks, about 0.7 s per step. NVE from 600 K, after the first 20 % of each run:

| Time step | Steps | Mean T, second half | Spread of E_pot + E_kin | Drift |
|---|---|---|---|---|
| 1 fs | 1000 | 302.4 K | 0.016 meV/atom | 0.019 meV/atom/ps |
| 2 fs | 500 | 302.3 K | 0.065 meV/atom | 0.015 meV/atom/ps |
| 3 fs | 334 | 302.1 K | 0.148 meV/atom | −0.049 meV/atom/ps |

The spread grows with the square of the time step, as Verlet's should. The crystal shares its energy evenly: at 1 fs the potential energy above the static crystal and the kinetic energy are 33.7 and 34.2 meV/atom, against 33.9 for 300 K. NVT at 300 K with a 100 fs constant: 281 ± 87 K over the second half of 2 ps at 2 fs, 274 ± 82 K over that of 1.5 ps at 1 fs, where a canonical spread is 93 K. The temperature takes about 400 fs to climb back from half its value and then swings around the target with a period near 800 fs, so a short window can read low: over 6 ps at 10 Ha and 2 fs, the mean after the first 500 fs was 295 ± 7 K, and 305 ± 4 K with a 30 fs constant, while `md_thermostat_fs` 10 reaches the target within 50 fs (304 ± 91 K over the second half of 300 fs). The first g(r) peak is at 2.325 Å in 0.05 Å bins, against a 2.352 Å bond. The spectrum of the positions has peaks near 136, 426 and 477 cm⁻¹ and ends near 560 cm⁻¹, as silicon's phonons at X and Γ do. A relaxation of the final frame came back within 0.02 meV of the static crystal's energy, and an SCF on the last frame of a stopped run agreed with that frame's energy within 1e-7 eV.

**Small cells.** A Γ-only mesh is not adequate for 8 Si atoms: the crystal is not a minimum there, and it melted from a 300 K start, falling 0.8 eV/atom below the crystal under NVE and heating to 7300 K. The result reports this, but the mesh must be chosen first.

**Files.** ABINIT 10.0.3 writes `_HIST.nc` as NetCDF-3, which scipy reads, whatever its NetCDF library could do. A run beyond 4000 steps keeps about 2000 frames with `nctime`. The output cannot be thinned: `run.abo` grows by about 2 KB per step plus 100 bytes per atom after turning off every per-step table ABINIT allows, and it is the record of each step's SCF, which is checked. A step whose SCF did not converge refuses the run. `_NOLOG` keeps the log empty when several ranks run; one rank writes its errors only there, so that run keeps it. `--timelimit` stops the run cleanly before the job's limit, and the frames so far are kept.

**Ranks.** ABINIT shares only k points by default, so ranks beyond nkpt × nspins idle. Then `md` adds `paral_kgb 1` and `autoparal 1`, which pick how to share bands and FFTs at run time, sometimes leaving a few ranks unused (28 of 32 for 14 k points). This needs `istwfk *1`, because real Γ wavefunctions segfault in that split, and at least two ranks, because on one process 10.0.3 gets the total energy wrong by 4.2 Ha for Si8 with the right forces; the app never asks for that and refuses such an output. From 2 to 32 ranks the energies agreed with the k-point route within 6e-7 Ha, with and without spin and spin–orbit coupling. At Γ, 8 ranks are about twice as fast as one for 8 atoms, and more add little; 64 atoms took 219 s for 8 steps on 8 ranks and 81 s on 28 of 32, with the same energy to 1e-8 Ha.

### Elk optics and real-time TDDFT

The `optics` task computes the dielectric function on a dense mesh (`optics_mesh`, default twice the SCF mesh) without local fields, with `optics_max_ev`, `optics_points`, `optics_broadening_ev`, `optics_components` and an optional `scissor_ev`. With `tddft_kernel` (`rpa`, `alda` or `bootstrap`) Elk adds linear-response TDDFT with local fields up to `response_cutoff`; the bootstrap kernel gives LiF its exciton below the gap, which the spectrum without local fields lacks. Reported are each spectrum's strongest absorption and, for a gapped crystal, Re ε at zero frequency. The optics step uses Gaussian occupations: at the 0.1 eV broadening, Fermi–Dirac tails had half-filled the bands next to silicon's small PBE gap.

The `tddft` task follows the electrons under a Gaussian laser pulse (`pulse_photon_ev`, `pulse_field_v_per_angstrom`, `pulse_polarization`, `pulse_fwhm_fs`, `pulse_peak_fs`, `simulation_time_fs`, `time_step_as`) and reports the field, the current, the absorbed energy and a magnet's spin moment over time. Without spin–orbit coupling the total spin moment is conserved, so laser demagnetization needs `soc`. A run with a zero field is the check Elk's authors ask for: its moment must stay constant, otherwise the time step is too long. Before time zero Elk reruns one ground-state iteration under a static field; that iteration's files are not reported as the ground state.

Lithium compounds use `lradstp 2`, as in Elk's own LiF examples: Elk keeps lithium's 1s shell in the valence, and on the default coarse radial mesh its density lost 0.2 % of the charge and PBE self-consistency did not converge.

### CP2K large cells and molecular dynamics

An insulator at mesh [1, 1, 1] runs with orbital transformation instead of diagonalization, which scales to large supercells: on Arrhenius a 512-atom silicon cell converged in 2.5 minutes on 128 cores and a 1000-atom cell in 4 minutes on 256. The `md` task runs Born–Oppenheimer molecular dynamics, NVT with a CSVR thermostat or NVE (`md_ensemble`, `md_temperature_k`, `md_steps`, `md_timestep_fs`, `md_thermostat_fs`). It reports the temperature and its spread, the drift of the conserved energy, the final frame and g(r) with each pair's first peak, averaged over the second half of the run. When that half is still settling, with a trend in the potential energy beyond its noise or, under a thermostat, a mean temperature further from the target than both 10 % and one standard deviation, the result says so. A built structure releases energy at first: a packed water box started from random orientations released 10 eV in its first 25 fs and was still 100 K too hot after 500 fs. Relax a crystal or supercell first (`["relax", "md"]`). A liquid's flat energy surface keeps a relaxation from finishing (200 steps for 64 water molecules did not reach 0.05 eV/Å), so run two `md` stages instead: a few hundred steps with a strong thermostat (`md_thermostat_fs` 10), then the production run, which continues from the first one's final frame.

ABINIT cells with a vacuum gap, such as monolayers, use `diemac 2`: with ABINIT's metallic default, the potential along the vacuum barely converged, and graphene, MoS2 and CrI3 kept a large residual while the energy test passed. Spinor runs keep an even band count, which ABINIT requires. A spin-polarized insulator uses a 0.03 eV Gaussian smearing, as in QE.

Parameters use one vocabulary across the three codes: `functional`, `mesh` or `kspacing`, `spin`, `soc`, `initial_moments` or `site_moments` (μB), `electronic_type`, `smearing_ev`, `cell_relax`, `force_tolerance_ev_angstrom`, `max_ionic_steps`, `max_scf_steps`, `line_density`, `kpoints`, `extra_bands`, `dos_mesh` and `hubbard` (element, l, `u_ev`, `j_ev`). `energy_tolerance_ev` applies to ABINIT and Elk. CP2K uses `eps_scf`, its SCF residual tolerance, instead. `extra_bands` counts additional orbitals per spin; Elk converts this to its per-atom `nempty` setting. Code-specific keys are `ecut_ha` for ABINIT; `cutoff_ry`, `rel_cutoff_ry`, `basis_family`, `eps_scf`, `added_mos` and `dos_broadening_ev` for CP2K; and `rgkmax`, `gmaxvr` and `nempty` for Elk. Defaults follow each library's own recommendations, for example the PseudoDojo "normal" cutoff, and a Γ-centred mesh with 0.25 Å⁻¹ spacing.

With the selected mixing method, [CP2K checks the density residual against `EPS_SCF`](https://github.com/cp2k/cp2k/blob/v2026.1/src/qs_scf_loop_utils.F#L471-L539), so an energy threshold in eV cannot replace it. The [Elk manual](https://elk.sourceforge.io/elk.pdf#page=32) defines `nempty` per atom and spin and permits fractional values; `extra_bands=3` therefore becomes `nempty=1.5` for a two-atom cell.

## Install on your cluster

Open **Connections → Cluster** after the connection test. Under **Install open-source codes**, choose a code, check the partition and account, and click **Plan the installation**. The app reads the login node's CPU architecture and GNU C library version and picks a pinned conda-forge package set:

| Code | linux-64 (x86-64) | linux-aarch64 (Arm) | Size |
|---|---|---|---|
| ABINIT | glibc 2.17 or newer | glibc 2.17 or newer | about 190 MB |
| CP2K | glibc 2.28 or newer (CP2K 2026.1, MPICH) | glibc 2.38 or newer (CP2K 2026.2, MPICH) | about 750 MB |
| Elk | glibc 2.17 or newer | glibc 2.17 or newer | about 100 MB |

These limits come from scanning every program and library file in each package set, not from the package metadata alone: the C library symbol versions each file uses, and whether every shared library the program loads is in the set. conda-forge's CP2K 2026.2 packages for x86-64 declare glibc 2.17 or 2.28, but their libxs and libxsmm libraries use glibc 2.38 symbols. x86-64 clusters therefore get CP2K 2026.1, with its direct dependencies, such as DBCSR and libxc, pinned to the versions it was built and tested with.

Review the plan, tick the allocation box and click **Start the installation**. Then:

1. The login node downloads every package and checks its SHA-256. Packages it cannot reach are downloaded by the app and uploaded.
2. One short scheduler job creates the environment offline with micromamba from the pinned list.
3. The job places the pinned CP2K data files or ABINIT silicon pseudopotential in their library folder, or extracts the Elk species, checking every hash.
4. It runs a small silicon SCF. The app downloads that result and checks it with the same parser used for project runs. If a library needs a newer C library than the compute node has, the check names both versions.
5. **Check the installation** follows the job. When the test passes, the software environment is created.

If the installation fails, **Plan again** creates a new attempt with its own job. It reuses an environment that an earlier attempt already created from the same packages.

Nothing is compiled and no administrator rights are needed. After a passed test, the downloaded package archives and the unpacked package cache are removed, so only the environment stays. Programs live under `<work folder>/software/`, ABINIT pseudopotentials under `<work folder>/pseudos/abinit` and CP2K data files under `<work folder>/libraries/cp2k-2026.2`. The CP2K data files are pinned separately from the program, so managed and cluster-provided CP2K use the same files.

A code that the cluster already provides is detected when you detect software, from the PATH or from `module avail`, and gets its own environment. Its pinned library then goes to `<work folder>/libraries/`.

Use the [bundled CLI](install.md#command-line) with a saved `cluster.json`:

```bash
dft-agent engine detect --config cluster.json --engine cp2k
dft-agent engine install-plan --config cluster.json --engine cp2k
dft-agent engine install --config cluster.json --engine cp2k
dft-agent engine install-status --config cluster.json --engine cp2k
dft-agent engine library --config abinit.json --elements Si O
dft-agent engine doctor --config abinit.json
```

## Libraries

ABINIT pseudopotentials are installed per element: the step card offers **Install library files** when a structure contains elements the environment lacks. You can also ask the agent to install them. Files come from the PseudoDojo GitHub repositories at fixed commits and are checked against `abinit_pseudos.json`. The fully relativistic PBE set has no Ni, Pd, Pt or Lu files, so ABINIT spin–orbit runs for those elements are refused.

CP2K basis sets and potentials are named per element in `cp2k_basis.json`. The default family is DZVP-MOLOPT-SR-GTH with GTH-PBE or GTH-PADE potentials. `basis_family` can also select `uzh_dzvp` or `uzh_tzv2p` with the UZH potentials. Elk species come from `elk_species.json`.

Before every job, the cluster hashes the library files the inputs use and checks a short header: valence charge for psp8, the named basis and potential for CP2K, and the element for Elk species. The hashes must match the pinned catalog or the environment's recorded installation. The job then runs on copies in its own folder.

## Official manuals and examples

For Elk, ABINIT, CP2K and Quantum ESPRESSO, the planner and the agent can look things up in the program's own documentation for the version an environment runs:

- **Elk:** every input block and task described in the manual, which Elk keeps as LaTeX inside its source, and every `examples/*/elk.in`. They come from the release archive: 10.2.4 for the managed installation and 11.0.2 for newer modules; other versions use the nearest older one. The 11.0.2 example inputs are identical to those installed on Arrhenius.
- **ABINIT:** all input variables of abinit and anaddb with their types, defaults and units, every tutorial and topic page, the guides to abinit, anaddb, mrgddb and response functions, and the inputs of the ground-state and response-function tutorials. ABINIT keeps its variable database as Python code; DFT Agent reads it as data and never runs it. The files come from the release commits of ABINIT's GitHub mirror, each pinned by its SHA-256: 10.0.3 for the managed installation and 10.6.7 for the Arrhenius module. The planner shows a variable only when the request names it, and marks examples that use what DFT Agent's ABINIT does not offer, such as PAW or GW.
- **CP2K:** the complete keyword reference, written by the installed program itself with `cp2k --xml` on the login node (a few seconds), plus 44 documentation pages and 42 example inputs (benchmarks, regression tests and manual examples) from CP2K's v2026.2 release.
- **Quantum ESPRESSO:** the input descriptions of pw.x, ph.x, q2r.x, matdyn.x, dynmat.x, dos.x, projwfc.x, bands.x, pp.x and neb.x, read from QE's own `INPUT_*.def` files with their units, defaults and options; the README and script of every official PW, PHonon, PP and NEB example, each input the script writes with the README step that explains it; and the EPW example inputs. 7.4.1 for the managed build and 7.6 for the Arrhenius module, from their release tags. QE plans may set native namelist options, so the planner gets the pw.x entry of a variable the request names or that matches it best, and examples that need what DFT Agent's QE does not run, such as molecular dynamics, are marked.
- **DFT Agent's own cluster checks**, such as the finer radial mesh lithium needs in Elk, the dense k mesh GaAs needs for its dielectric response in ABINIT or the equilibration of liquid water in CP2K.

Nothing is bundled, because the manuals and examples are GPL. DFT Agent ships only their sources and SHA-256 hashes in `references.json`. It downloads them on first use, checks them and keeps them in `~/.dft-agent/references`. `packaging/pin_references.py` pins the files of a new ABINIT or QE release. The CP2K keyword reference needs the cluster: click **Get manuals and examples** under the cluster's software, or ask the agent to build it.

In **Step by step**, the planner gets up to four short excerpts that match the request, for example Elk's LiF bootstrap example when you ask for LiF's excitons, and the plan lists them under `references`. Examples demonstrate features with modest settings: the planner uses them for what a setting means and for typical values, not as converged settings, and the app's checks still apply. In **Automatic** mode the agent searches and reads them itself with `search_reference` and `read_reference`. The search runs on your computer and maps common Chinese terms to the manuals' English; excerpts go only to your own model, as part of the plan request.

## Checks on results

When a code stops, the step quotes its own error: ABINIT's error documents, including the abort file that another MPI rank writes, and QE's routine and message. Every stage is accepted only when the code finished normally, the electronic loop converged and, for relaxations, the ionic or cell loop converged. The code must also have used the prepared structure; ABINIT may symmetrize it slightly, so its check allows 1e-4. A program that writes a complete output but then exits with an error, for example a crash while shutting down, is not accepted either: the step quotes the crash line from the job's error log and suggests another build. Uploaded inputs, library files, execution receipts and collected outputs are checked by SHA-256, as for QE. Explanations re-parse the retained outputs and never trust a saved number on its own.

Gaps are statements about sampled k-points only. A band path keeps the Fermi level of the SCF it continues, and a band touching that level, such as graphene's Dirac cone, gives a zero gap. CP2K gives its gap from the band path or the DOS-mesh eigenvalue histogram, with the bin width as resolution; along a path it uses the Fermi level, because its band file mislabels the occupations of magnets. All-electron Elk total energies are not comparable with pseudopotential codes.

## Known limits

- Managed CP2K and ABINIT builds use the MPI shipped in their environment (`mpiexec -n {tasks}`) and run within one node. For several nodes, use the cluster's own build.
- ABINIT inputs other than `md` use k-point and spin parallelism only. Processes beyond the number of irreducible k-points times spins stay idle, so ask for at most that many tasks. ABINIT reports `nkpt` in its log and warns when there are more processes.
- Elk from conda-forge uses OpenMP only: set **Tasks** to 1 and **CPUs per task** to the cores to use.
- A local macOS CP2K build stalled in its closing DBCSR statistics report. The managed Linux build and the cluster module on Arrhenius did not. A stall would show in the installation's silicon test, which has a 20-minute limit.

See [0.4.5 checks](validation-0.4.5.md) for what has been verified.
