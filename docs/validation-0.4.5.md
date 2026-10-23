# 0.4.5 checks

These checks cover the ABINIT, CP2K and Elk integration. They show that inputs, parsers, receipts and the managed installer work as designed. They do not establish numerical convergence for any material.

## Real programs on a workstation

Machine: Apple Silicon Mac (macOS 26). Programs: ABINIT 10.0.3 from conda-forge (osx-64, under Rosetta); CP2K 2026.2 and Elk 10.2.4 built from the release sources against conda-forge libraries. The local CP2K build skips CP2K's closing DBCSR statistics report, which stalls with that library combination. Results and timings are not affected.

Silicon (two-atom PBE cell, experimental lattice 5.43 Å, small test settings):

| Check | ABINIT | CP2K | Elk |
|---|---|---|---|
| Gap along the band path (eV) | 0.57 | 0.62 | 0.55 |
| Relaxed lattice constant, cell relaxation (Å) | 5.470 | 5.475 | 5.488 |
| Spin–orbit SCF gap, 4×4×4 grid (eV) | 0.67 | not offered | 0.67 |

Ferromagnetic bcc iron (PBE, 6×6×6 grid): ABINIT 2.221 μB, CP2K 2.222 μB, Elk 2.222 μB.

CP2K's stress needed two input choices, both now fixed. Cells are written as lengths and angles to preserve the intended cell geometry. Even meshes are shifted to be Γ-centred. Without these, ideal silicon showed spurious forces and an anisotropic stress.

Elk jobs now set the stack sizes from Elk's own launcher (`OMP_STACKSIZE=512M`, a 512 MB process stack). Without them, an Elk cell relaxation crashed with a segmentation fault inside the job.

## Workflow end to end

A stand-in Slurm on the same machine ran the real programs through `engine_workflow`: freezing, upload, the stage runtime, submission, polling, collection, receipts and analysis. For each code, a relaxation followed by bands and DOS expanded to relax → SCF → bands → DOS. All four stages were accepted, with input identity, execution and library receipts verified. Explanations and structure continuation then read the verified results.

`tests/test_engine_workflow.py` replays recorded outputs together with refusal cases: a changed or missing library file, a changed prepared input, a failed program and a crash after a complete output. The recordings now come from Arrhenius (managed ABINIT 10.0.3 and Elk 10.2.4, the CP2K 2026.2 module) with the current band paths; user names, hosts and paths in them are replaced by placeholders.

## Installer

- The installation job script ran on 50 real conda-forge packages (the osx-64 ABINIT set). It verified each file, created the environment offline with micromamba from a hash-pinned explicit list, placed the pseudopotential and passed the silicon test with two MPI ranks. A second run reused the environment.
- The app-side test check parsed that silicon result with the project parser.
- The login-node stager downloaded real linux-64 packages over HTTPS with matching hashes and refused an entry with a wrong hash.

## Real cluster: Arrhenius

Arrhenius (NAISS) runs Slurm on 256-core CPU nodes with an EL9 system and glibc 2.34. The app reached it through the user's own SSH session; everything else went through the app's code: software discovery, installation plans and jobs, library installation and the workflow.

Each code ran a slightly displaced two-atom silicon cell (PBE): relaxation, then SCF, bands and DOS. ABINIT and Elk relaxed the cell as well, CP2K only the positions. Meshes were 6×6×6 for ABINIT and Elk, 4×4×4 for CP2K, with denser DOS meshes; ABINIT and CP2K used 4 MPI ranks, Elk 8 OpenMP threads.

| Check | ABINIT | CP2K | Elk |
|---|---|---|---|
| Cluster module | 10.6.7: crashes while exiting after a complete output; not accepted | 2026.2: all four stages accepted | 11.0.2: all four stages accepted |
| Managed installation and silicon test | 10.0.3: passed | 2026.1: passed after the fixes below | 10.2.4: passed |
| Managed run | All four stages accepted | All four stages accepted | All four stages accepted |
| Relaxed lattice constant (Å) | 5.471 | not relaxed | 5.473 managed, 5.469 module |
| Gap along the band path (eV) | 0.604 | 0.617, both builds | 0.598 managed, 0.601 module |
| Gap on the DOS mesh (eV) | 0.642 | 0.776, both builds | 0.636 managed, 0.638 module |

The managed CP2K 2026.1 and the CP2K 2026.2 module agree on the total energy to 1e-7 eV. Elk 10.2.4 and 11.0.2 total energies differ by 2.1 eV for the same cell and species files, so compare Elk energies only within one build.

These pipelines ran before the band-path fix below, so their bands followed the monoclinic path that the old symmetry tolerance found for the relaxed cells; the gaps compare the codes on that same path.

A managed ABINIT SCF with 256 MPI ranks on one full node was also accepted. ABINIT used 68 of them, one per irreducible k-point, and warned that the others idled.

These problems appeared only on the cluster. Each is fixed, and all except the environment reuse in item 8 were checked again there:

1. conda-forge's CP2K 2026.2 packages need glibc 2.38: their libxs and libxsmm use glibc 2.38 symbols although the metadata declares 2.17 or 2.28. x86-64 clusters now get CP2K 2026.1, and every package set has been checked with `packaging/check_engine_packages.py`, which reads the glibc versions and shared libraries of all ELF files.
2. The CP2K 2026.1 build links `libdbcsr.so.2.9`, but its dependency range admits DBCSR 2.10. The check now follows every shared library the program loads, and DBCSR is pinned to 2.9.1.
3. With libxc 7.1, CP2K 2026.1 stopped while reading functional names. Its direct dependencies are now pinned to versions no newer than the CP2K build.
4. CP2K 2026.1's spatial k-point reduction failed for both shifted even meshes and an unshifted 5×5×5 mesh. The latter crashed in `kpsym.F` with a negative Fortran unit. All meshes now use `SYMMETRY OFF` and `FULL_GRID OFF`, retaining k/−k pairing and the requested integration mesh. The repaired 5×5×5 relaxation and subsequent SCF, bands and DOS passed on the managed build. In the earlier even-mesh comparison, the CP2K 2026.2 module gave the same energy, pressure and gaps with the old and new input, within 3e-7 eV.
5. CP2K 2026.1 writes no Fermi level in its DOS file. The DOS gap now uses the Fermi energy from the same run's main output, and both builds give 0.776 eV.
6. conda-forge's Elk links a pthreads OpenBLAS, which started its own threads inside each Elk thread: an SCF took 16 s instead of 3 s, and the first managed relaxation hit its 30-minute limit. Elk jobs now set `OPENBLAS_NUM_THREADS=1`. The relaxation then needed about ten minutes for the same 66 steps as the module. For the Elk module, which uses FlexiBLAS, the setting made no difference: the same SCF took 5.0 and 5.4 s without it and 4.7 and 5.4 s with it on one node.
7. The ABINIT module wrote a complete output and then aborted with "double free or corruption" in a library destructor. Such results are still refused; the explanation now quotes the crash and suggests another build.
8. A failed installation could not be retried, because the same plan returned the old job. **Plan again** now creates a new attempt, as used on Arrhenius. A new attempt reuses an environment an earlier attempt created; a local test covers this, because the first CP2K environment there predated the change.
9. Two processes sharing a workspace could mark a running job as needing attention when SQLite removed its `-wal` or `-shm` file mid-check. This is fixed. Resuming that run polled the same job and finished it.
10. Each CP2K installation attempt left about 3.7 GB in its work folder: the package archives and micromamba's unpacked cache. The cache is now removed after the environment is created, and the archives after the test passes. A fresh Elk installation then left 9.5 MB there.

## Fresh managed installations: 22 October 2026

New app-managed installations of ABINIT 10.0.3, CP2K 2026.1 and Elk 10.2.4 passed their installation tests on Arrhenius. The following calculations then ran through the app's preparation, Slurm submission, collection and result verification. Every listed stage finished with exit code zero and accepted input, execution and library receipts.

| Program | Calculation | Accepted stages | Result |
|---|---|---|---|
| ABINIT | Displaced Si: positions-only relaxation → SCF → bands → DOS; 4×4×4 mesh and 6×6×6 DOS mesh | 6, including the SCF prerequisites | Sampled band-path gap 0.74 eV |
| CP2K | Displaced Si: positions-only relaxation → SCF → bands → DOS; 5×5×5 mesh and 6×6×6 DOS mesh | 4 | Final maximum force 0.00011 eV/Å; band-path gap 0.63 eV; DOS-mesh gap 0.78 eV |
| Elk | fcc Al: SCF → bands; 8×8×8 mesh | 2 | Metallic, zero gap in both stages |

The CP2K rerun preserved the failed relaxation's inputs except for switching spatial k-point symmetry off. The original failed run remains available as evidence. Elk's Al run exposed valid Methfessel–Paxton integration weights outside the usual occupation range; these are now supported without weakening checks for physical occupation probabilities. The metal test also checks that a band crossing between sampled points is not mistaken for a small gap.

These are execution and parser checks, not accuracy benchmarks. The band paths, DOS meshes and basis settings are finite; differing sampled gaps do not establish numerical convergence or agreement with experiment. Raw outputs, structures, tabulated results and PNG plots passed artifact verification. PDF and SVG files from these earlier runs retain their original metadata; deterministic export is covered separately for newly generated files.

## Band paths and the silicon sample

The 0.4.4 demo video showed the bundled silicon sample with monoclinic path labels (C, Y_2, M_2, ...) and bands drawn as straight lines. Two causes, both fixed for VASP, QE, ABINIT, CP2K and Elk:

- Band paths used seekpath's default 1e-5 Å symmetry tolerance. A cell relaxed from a displaced start keeps a few mÅ of distortion, so silicon was classed as monoclinic. With synthetic relaxation noise (3 mÅ, 0.05 % strain) on six lattices (diamond, bcc, hcp, cubic perovskite, rutile, wurtzite), 0.01 Å recovered 31 of 60 cells and 0.02 Å all 60, while a 1 % tetragonal strain, a 0.05 Å polar shift and a 59.5° rhombohedral cell kept their lower-symmetry paths. 0.05 Å turned the tetragonal cell cubic. Paths now use 0.02 Å.
- The sample had been computed with `line_density: 1`, 13 points in all. Every segment now gets at least 10 points.

`tests/test_band_paths.py` checks both rules on these lattices, and `tests/test_samples.py` checks every bundled band path against them. The replay fixtures were re-recorded on Arrhenius with the new paths: 56 instead of 15 points at the test density. The gap along the denser path is 0.47–0.53 eV instead of 0.60–0.66 eV at those coarse test settings, because the sparse path missed the conduction-band minimum on Γ–X.

The silicon sample was recomputed on Arrhenius with VASP 5.4.4: a PBE cell relaxation from 5.431 to 5.468 Å in three steps, a 16×16×16 mesh, the 302-point path Γ–X–U|K–Γ–L–W–X and a tetrahedron DOS. The sampled gap is 0.61 eV on the path and 0.62 eV on the mesh, and the direct gap at Γ is 2.56 eV. The cluster's central VASP 6 modules stopped at start-up because they look for a personal licence key file (`VASP_LICENSE_FILE`, the run folder or `~/.vasp/`), which this account has not set up; its own VASP 6.5.1 build runs normally. The sample therefore used the central VASP 5.4.4 module, like the SrTiO3 sample. DFT Agent now names this cause when VASP stops at start-up.

Relaxation plots also showed fractional step numbers (0.945, 0.960, ...) when a relaxation converged after one step, and QE plots drew nothing for a single step. Both now show whole steps with marked points.

## Materials and each program's strengths: Arrhenius, 22 October 2026

All of these ran through DFT Agent's own preparation, submission, collection and checks on Arrhenius: QE 7.6, the CP2K 2026.2 and Elk 11.0.2 modules, the managed ABINIT 10.0.3, and VASP 5.4.4. Settings are moderate, not converged; the comparisons show that the codes agree when run the same way, not that a number is final.

### The same materials in five codes

PBE with each code's default library: SSSP for QE (PseudoDojo FR with spin–orbit coupling), PseudoDojo for ABINIT, MOLOPT for CP2K, Elk's species and VASP's PBE.54 PAW. Gaps in eV, sampled; band-path gaps where computed.

| | QE | ABINIT | Elk | CP2K | VASP |
|---|---|---|---|---|---|
| GaAs + SOC | 0.411 | 0.411 | 0.405 | not offered | 0.413 |
| MoS2 monolayer + SOC | 1.617 | 1.617 | 1.642 | not offered | 1.604 |
| CrI3 monolayer, ferromagnetic | 0.988 | 0.978 | 1.008 | 0.994 | 0.910 |
| CrI3 total moment (μB) | 6 | 6 | 6 | 6 | 6.000 |
| NiO, U = 6 eV on Ni d, antiferromagnetic | 3.11 | not offered | 3.18 | Γ-point cells only | 3.26 |
| NiO + U, Ni moment (μB) | 1.68 | | 1.72 | | 1.73 |
| bcc Fe, Fe moment (μB) | 2.22 | 2.32 | 2.15 | 2.28 | 2.22 |
| Si, after a PBE cell relaxation | | | | | 0.610 |

### What each program is best at

**ABINIT phonons** by density-functional perturbation theory, 4×4×4 q mesh, interpolated by anaddb (cm⁻¹):

| | Computed | Measured |
|---|---|---|
| Si, Γ optical | 514.4 | 517 |
| Si, X: TA, LA, TO | 140, 410, 462 | 150, 410, 463 |
| Si, L: TA, LA, LO, TO | 107, 374, 414, 490 | 114, 378, 420, 490 |
| GaAs, Γ: TO, LO | 268.3, 277.9 | 268, 292 |
| GaAs, X: TA, LA | 70, 220 | 79, 227 |
| GaAs, L: TA, LA, LO, TO | 56, 203, 239, 262 | 62, 209, 238, 261 |
| Al, highest frequency, 16×16×16 mesh | 315 | 323 |

Silicon's electronic dielectric constant came out as 14.0 and its Born charges zero. GaAs's came out as 29.7, with Born charges ±2.27 that missed neutrality by 0.27 e before it was imposed, so its LO–TO splitting is too small: on that Γ-centred 8×8×8 mesh the direct PBE gap of 0.5 eV sits on a k point. The result says so. On a 12×12×12 mesh, aluminium's highest frequency was 7 % higher than on 16×16×16.

**QE phonons and the frame of the cell.** pw.x found 2 of GaAs's 24 symmetry operations in a cubic primitive cell as stored in a CIF: an 8×8×8 SCF used 150 k points, phonons 24 q points, and an HSE06 SCF on a 6×6×6 mesh (68 k points) did not finish in 2 hours. In the rotated frame pw.x found all 24: 29 and 16 k points, 8 q points. That HSE06 run then managed 21 outer exchange iterations in 2 hours instead of 4, but still did not finish: its exchange error levelled off near 1e-7 Ry and never met the 1e-8 Ry threshold, although the gap had settled at 1.379 eV once the error was below 1e-6 Ry. Hybrid SCF runs now default to QE's own 1e-6 Ry. Silicon phonons in that frame agree with ABINIT to 1 cm⁻¹ (Γ 514.5, X 140/410/462, L 107/374/414/490) and so does ε∞ (14.024 against 14.021). GaAs with SSSP's Ga and As UPFs did not: 118 cm⁻¹ at Γ before the sum rule and transverse branches about 50 % above experiment (122 cm⁻¹ at X), at a density cutoff of 560 or 1120 Ry and on shifted or denser meshes. With PseudoDojo norm-conserving UPFs it gave 50 cm⁻¹ and branches within 11 cm⁻¹ of ABINIT. On the half-step shifted 8×8×8 mesh QE gave ε∞ = 13.6 and neutral Born charges with either library.

**Elk optics.** Silicon without local fields: strongest absorption 3.72 eV and Re ε(0) = 13.48, after a Fermi–Dirac artefact (14.67) was removed. LiF with a 5.2 eV scissor: the bootstrap TDDFT kernel gives an exciton at 13.76 eV, below the 14.19 eV gap, with Im ε = 48; the spectrum without local fields has none and peaks at 16.6 eV.

**Elk real-time TDDFT on fcc Ni**, 6×6×6 mesh, 2.4 as steps:

| | Field | Time | Absorbed (eV/atom) | Moment change |
|---|---|---|---|---|
| Collinear, 1.55 eV pulse | 1.0 V/Å | 6 fs | 1.99 | +0.005 % |
| Spin–orbit, no field | 0 | 15 fs | 0.0 | 5e-8 % |
| Spin–orbit, 1.55 eV pulse, 3 fs FWHM | 2.0 V/Å | 15 fs | 15.4 | −13.8 % |

Without spin–orbit coupling the total spin is conserved; with it, the moment falls during the pulse and keeps falling after it, as in Elk's own nickel example.

**CP2K at scale.** With orbital transformation at Γ, a 512-atom silicon cell converged in 2.5 minutes on 128 cores and a 1000-atom cell in 4.2 minutes on 256; their energies per atom differ by 2 meV. Liquid water (64 molecules, 300 K, 0.5 fs steps, 128 cores, about 1.4 s per step): started from a packed box with random orientations, the first run released 10 eV in 25 fs and was still at 397 K over its second half, which the result now reports; a relaxation of the box did not reach 0.05 eV/Å in 200 steps. Its O–H, H–H and O–O first peaks were 0.975, 1.575 and 2.875 Å. Two MD stages then ran from the same box: 200 fs under a 10 fs thermostat held 312 ± 10 K, and the following picosecond, started from that stage's final frame, held 294 ± 15 K with a conserved-energy drift of 0.07 meV/atom/ps and the O–O peak at 2.725 Å. Its potential energy was still falling by 9 meV/atom over the second half, which the result reports: a packed box needs several picoseconds before its averages are equilibrium values.

**VASP.** Besides the comparisons above, VASP's HSE06 path needed three fixes, listed below.

### Problems found and fixed in this round

1. Elk judged its laser runs by the static-field iteration it writes before time zero, which rewrites INFO.OUT, EFERMI.OUT and EIGVAL.OUT. Only the log is used now, and those files are not reported as the ground state.
2. Elk's lithium species keeps 1s in the valence; on the default radial mesh LiF lost 0.2 % of its charge and PBE did not converge. Lithium compounds use `lradstp 2`, as Elk's LiF examples do.
3. Elk's TDDFT spectra leave out zero frequency; the parser had required both spectra on one grid.
4. The Elk TDDFT moment panel drew 1e-4 % noise as a signal; it now shows the relative change with at least a 1 % range.
5. ABINIT, CP2K and Elk warnings were not shown in the app or given to the planner.
6. QE 7.6's ph.x stopped with a segmentation fault in its electron–phonon step with PAW UPFs for aluminium, also on a small mesh, while norm-conserving UPFs ran through. Electron–phonon plans now require norm-conserving UPFs.
7. www.pseudo-dojo.org did not answer from the laptop or from Arrhenius. The install now names the host and where a verified copy of the archive can go.
8. pw.x cells are written in the rotated frame described above.
9. The VASP hybrid warm start used Python 3.10's `newline` argument and stopped on Arrhenius's Python 3.9. Every script that runs on the cluster is now checked for Python 3.6.
10. Its PBE seed copied the hybrid's `ISYM = 3`, which VASP 5.4.4 accepts only with exact exchange; the seed now uses `ISYM = 2`.
11. HSE06 runs set `HFRCUT = -1`. VASP 5.4.4 then truncated the bare Coulomb kernel instead of screening it: silicon came out at 1.97 eV with 4.0 eV at Γ, PBE0's values, 0.75 eV above QE's HSE06 on the same mesh, and GaAs at 2.05 eV. Only PBE0 truncates the kernel now; HSE06 then gave silicon 1.31 eV with 3.34 eV at Γ, and GaAs 1.43 eV, against QE's 1.22 and 1.38 eV on the same meshes.
12. The CP2K water run showed that averages over a run still settling were presented as equilibrium values, and that a liquid's relaxation does not finish. MD results now check the second half for continued settling; passing this check does not establish equilibrium. Later stages continue from the final MD frame.
13. ABINIT's mrgddb 10.0.3 read its own output name as an input when given on the command line; the list now goes through standard input.

## Official manuals and examples for the planners: 22 October 2026

**What the references hold.** Built on the laptop from the pinned sources:

| | Source | Entries | Index |
|---|---|---|---|
| Elk 10.2.4 | release archive (2.9 MB) | 133 input blocks, 93 tasks, 9 manual sections, 136 examples | 0.40 MB |
| Elk 11.0.2 | release archive | 143 examples, the same as installed on Arrhenius | 0.44 MB |
| CP2K 2026.2 keywords | `cp2k.psmp --xml` on the Arrhenius login node, a few seconds | 1779 sections and 6762 keywords, each defined once under its most central path (the XML repeats them as 9983 sections and 67 516 keywords) | 4.0 MB |
| CP2K documents | 86 files at the v2026.2 tag (0.7 MB) | 278 page sections, 42 example inputs | 0.79 MB |
| ABINIT 10.0.3 | 308 files of the release commit on GitHub (3.3 MB) | 1026 variables of abinit and anaddb, 778 sections of tutorials, topics and guides, 150 tutorial inputs | 3.7 MB |
| ABINIT 10.6.7 | 339 files (4.4 MB), the version of the Arrhenius module | 1146 variables, 971 sections, 173 tutorial inputs | 5.0 MB |
| QE 7.4.1 | 183 files of the release tag (1.0 MB) | 532 variables and 33 namelists and cards of ten programs, 78 example descriptions, 344 example inputs | 1.1 MB |
| QE 7.6 | 220 files (1.1 MB), the version of the Arrhenius module | 543 variables, 33 namelists and cards, 83 example descriptions, 373 example inputs | 1.2 MB |

Loading the CP2K index takes 0.3 s and a search well under 10 ms.

**Planning with and without them.** The same 20 requests went twice through the Elk and CP2K planners with the Codex CLI, each time with the references and without them: 80 plans. Each request has fixed checks, for example a bootstrap kernel and a 4–6.5 eV scissor for LiF's exciton, or an equilibration stage, at most 0.5 fs steps and 2 ps of production for liquid water.

| | With references | Without |
|---|---|---|
| Checks passed | 87 of 92 | 84 of 92 |
| Plans meeting every check | 37 of 40 | 34 of 40 |
| Plans the input checks refused | 0 | 1 |
| Extra questions to the user | 1 | 1 |
| Median planning time | 12.2 s | 10.2 s |
| Input tokens per plan | 18 200 | 17 300 |

- **LiF exciton (Elk):** with references, both plans used LDA with the 5.2 eV scissor of Elk's LiF bootstrap example and said so in their notes. Without, both used PBE without a scissor, which puts the exciton about 5 eV too low.
- **Laser demagnetization of Ni (Elk):** without references, one plan combined collinear spin with spin–orbit coupling, which the input check refused. With them, both plans used noncollinear spin and the zero-field reference run.
- **bcc iron with PBEsol (Elk):** without references, one plan asked whether a one-atom cell should be ferromagnetic. With them, both started a ferromagnet at 3 μB and cited Elk's iron example.
- **ZnO relaxation and bands (CP2K):** with references, one plan asked whether to relax the lattice too, prompted by CP2K's page on GEO_OPT and CELL_OPT. Without, both relaxed the cell.
- **Vacancy in a 63-atom silicon cell (CP2K):** all four plans used the geometry-based k mesh. The check had expected Γ only, so this request counts against both variants.

The other 15 requests, among them GW, phonons and the Fermi surface in Elk, and hybrids, NPT and D3 in CP2K, gave the same correct plan either way; each unsupported request was declined with the program that offers it.

**ABINIT.** Ten requests went twice through the ABINIT planner with each variant: 40 plans. With references 19 of 20 plans met every check, without 18 of 20. Eight requests gave the same correct plan either way: GaAs phonons with a dense mesh for the dielectric response, aluminium phonons with broadening, GaAs bands with spin–orbit coupling and PBE, 40 Ry converted to `ecut_ha` 20, a PBEsol cell relaxation followed by bands and DOS, and GW, elastic constants and DFT+U declined with a note that ABINIT itself offers the first two and Elk, QE or VASP the third. ABINIT's planner prompt already carries most of what the cluster checks found, so the references changed little here. Asked for GaAs bands "with nspinor 2", every plan asked whether spin–orbit coupling was meant, which is right: two-component spinors alone do not imply it. Asked for aluminium's DOS "with occopt 7 and tsmear 0.01 Ha", three of four plans declined occopt 7 as unsupported, although DFT Agent's `auto` occupations are exactly that; the one that accepted it had read ABINIT's occopt entry. The prompt now states how `electronic_type` maps to occopt, and all six repeated plans, with or without references, used Gaussian smearing of 0.272 eV.

**Quantum ESPRESSO.** Ten requests went twice through the QE planner with each variant, and the whole set ran twice: 80 plans. With references 37 of 40 plans met every check, without 33 of 40. Asked for GaAs's HSE06 gap, all four plans with references proposed an HSE06 SCF and read the gap on its mesh, after DFT Agent's GaAs HSE06 check; without them, two of four declined the request because hybrid band paths are not offered. Requests in QE's own terms were mapped correctly either way: degauss 0.02 Ry with Marzari–Vanderbilt smearing, nosym, 30 Ha as `ecutwfc` 60 Ry, conv_thr with mixing_beta, and the cg diagonalization, each in its namelist. Asked for GaAs's phonons, dielectric constant and Born charges, seven of eight plans asked for the q mesh and whether ε∞ or the static constant was meant; all used the half-step shifted mesh. Molecular dynamics was declined, NiO with DFT+U and the projected DOS of SrTiO3 were planned correctly in every case.

QE plans may set native namelist options, which QE rejects only once the job runs. These are now checked against the input description of the environment's QE version when its index is built: a name QE does not document in that namelist is a question before submission, with the namelist it belongs to, for example mixing_beta under SYSTEM instead of ELECTRONS. q2r.x's la2f is read by QE but missing from its description, so names the official examples use count as known. No plan above was stopped by the check.

**VASP HSE06 band paths.** VASP 5.4.4 spreads the bands of each k point over all MPI ranks unless KPAR is set, and rounds NBANDS up to their number: silicon got 64 bands on 64 ranks. Its HSE06 SCF took 22 minutes, and the band stage, which carries the 216 weighted mesh points beside 202 path points, did not finish one electronic step in two hours. With KPAR = 16 the same SCF took 34 seconds with the same gaps (1.307 eV, 3.344 eV at Γ). A band stage with KPAR = 16 took about 2.4 minutes per electronic step on the full mesh, and 43 seconds when only the 16 irreducible mesh points were weighted; the latter finished in 15 minutes with a 1.159 eV indirect gap along the path. Hybrid runs now set KPAR from the job's tasks and the k points; the warm start no longer compares KPAR with vasprun.xml, which in VASP 5.4.4 does not record it. DFT Agent's own silicon run then passed end to end on VASP 5.4.4: the band stage (KPAR = 64, 8 bands) converged in 12 electronic steps in 19 minutes, with a 1.159 eV indirect gap along the path and 3.345 eV at Γ, and all 11 pairs of equivalent k points agreed exactly. Hybrid band paths had been refused on VASP 5 because it ignores `LFOCKACE` (it has no adaptively compressed exchange); they are accepted now. With only 8 bands, the highest one drew false kinks where uncomputed bands cross it, so VASP band plots now stop just below that band, as the other codes' plots do. GaAs followed: its band stage (409 k points, KPAR = 64, 13 bands) converged in 14 electronic steps in 72 minutes on 64 ranks, with a 1.437 eV direct gap at Γ along the path against 1.434 eV on the SCF mesh, and all 9 pairs of equivalent k points agreed exactly. Its plot first stopped at the lowest point of the highest band, the triply degenerate conduction band at Γ, which then traced the frame there; the window now ends 0.1 eV lower.

**Aluminium electron–phonon coupling.** matdyn's interpolated α²F had small negative weights below 6.8 meV for the narrowest of ten broadenings, 0.64 % of its λ; the other nine had none. The stage had been refused for any negative weight. A broadening is now kept when its negative weights stay within 1 % of λ and reported in the result, and excluded beyond that; the stage is refused only when every broadening is excluded. λ came out between 0.42 and 0.45 with ω_log near 290 K (measured λ ≈ 0.43).

**Aluminium and niobium superconductivity through EPW 6.1.** Arrhenius's QE 7.6 module ships EPW 6.1, whose Tc summary uses other labels than EPW 5.9; DFT Agent had reported the finished niobium run as not finishing. Both chains now pass, from the dense SCF through DFPT, λ, the isotropic gap and the linearized Tc scan, and each spectrum was continued once with a wider temperature range (μ* = 0.1, second broadening):

| | λ (lambda.x, 10 broadenings) | ω_log | Simplified Allen–Dynes Tc | Eliashberg Tc | Gap at lowest T | Measured Tc |
|---|---|---|---|---|---|---|
| Al | 0.44–0.46 | 290–297 K | 2.0–2.4 K | 2.5–3.0 K | 0.42 meV at 0.25 K | 1.18 K |
| Nb | 1.08–1.54 | 147–152 K | 11.6–17.7 K | 18–22 K | 3.80 meV at 1 K | 9.25 K |

Aluminium's λ matches the measured 0.43; its Tc is high for μ* = 0.1, as usual for this choice. Niobium's λ depends strongly on the broadening and is above the 1.0–1.2 that converged calculations give, so these meshes overestimate its coupling. Both are functional checks of the chain, not predictions. Aluminium's three acoustic modes at Γ came out of ph.x at −2.5 cm⁻¹ with λ = 0 and reached lambda.x at zero frequency, as recorded in the stage. The first scans, 0.25–2 K and 1–14 K, did not reach an eigenvalue of one; the results said so instead of reporting a Tc.

**Disk use of QE stages.** Each QE stage copied the whole checkpoint of the stage it continues. After ph.x that included its scratch: aluminium's electron–phonon run used 225 GB, of which q2r, matdyn and lambda.x held three 53 GB copies, niobium's 417 GB, and a Bi2Se3 spin–orbit DOS stage a 49 GB copy of nscf wavefunctions. These runs pushed the group's project storage over its quota. q2r, matdyn and lambda.x now take only the XML they need, and band, nscf and DOS stages the charge density without wavefunctions. Silicon bands, DOS and phonons ran through on Arrhenius that way: the DOS stage held 4 scratch files and no wavefunctions, q2r and matdyn one file each.

**QE GaAs HSE06 with conv_thr 1e-6.** The rerun in the rotated frame finished in 1 h 38 min on 32 ranks with a 1.379 eV gap. Its exchange error fell to 1.6e-6 Ry in 5 outer iterations and then by about 10 % per iteration, so 15 of the 20 iterations only polished a gap that had already settled. Splitting its 16 k points into 16 pools instead of the 2 pw.x chose made it slower: the first exchange iteration had not finished after 12 minutes, against about 5 minutes per iteration with 2 pools, since every pool needs the orbitals of all others. QE hybrid runs keep pw.x's own choice.

## Release review: 23 October 2026

A new CP2K check on Arrhenius ran three MD steps on a two-atom silicon cell, followed by SCF. Both stages passed input, execution and library verification. The exported geometry retained the fractional coordinates in CP2K's rotated cell frame, and the SCF input exactly matched the final MD frame rather than the starting structure. The workflow now also discards an older SCF checkpoint after MD moves the atoms.

Ten retained native outputs were independently reparsed with the reviewed code: ABINIT phonons for Si, GaAs and Al; CP2K SCF for 512- and 1000-atom silicon cells and both water MD stages; Elk optics for Si and LiF; and Elk's Ni time evolution. All ten passed, and their Slurm records show successful exits. Regression checks also cover incomplete phonon tables, Γ-only DDB collection, magnetic optical components, and mismatched spin-response timestamps. These checks verify execution and interpretation of the retained outputs, not converged material properties.

## Not yet verified

- Runs across several nodes with the managed MPI builds.
- PBS clusters with these codes.
- Managed installations on Arm; no Arm cluster with a scheduler was available.

## Automated tests

`tests/test_engine_adapters.py`, `tests/test_engine_workflow.py` and `tests/test_engine_integration.py` cover input generation, method limits, recorded outputs, the workflow, the 0.4.4 → 0.4.5 workspace migration, discovery, installation records, libraries and the agent tools. `tests/test_engine_science_regressions.py` checks metallic crossings, finite-temperature occupations, extra-band and convergence settings, and CP2K's unchanged odd integration meshes. `tests/test_band_paths.py` covers band-path symmetry, sampling and relaxation plots for all codes, and `packaging/check_engine_packages.py` re-checks the pinned package sets.
