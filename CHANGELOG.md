# Changelog

## 0.4.5 — Unreleased

- Add ABINIT 10, CP2K 2026 and Elk 10 for relaxation, SCF, bands and DOS, with spin, spin–orbit coupling and DFT+U where each code supports them. Plan them in chat and combine them with VASP and QE steps.
- Install these codes from **Connections** without compiling: the app picks pinned conda-forge packages for the cluster's architecture and glibc, checks every hash, and verifies the installation with a silicon calculation. Each package set is checked for the glibc symbols and shared libraries its programs actually load, a failed installation can be planned again without rebuilding its environment, and installer downloads are removed once the test passes.
- Detect site installations of ABINIT, CP2K and Elk. Install pinned PseudoDojo pseudopotentials, CP2K basis sets and Elk species per element.
- Check library files, inputs, checkpoints and outputs by hash before results are accepted or explained. A program that writes a complete output but crashes while exiting is not accepted; the step quotes the crash from the job's error log.
- Upgrade older workspaces automatically to record the new codes.
- Fix a race in which two app processes sharing a workspace could mark a running job as needing attention.
- Keep a relaxed crystal's band path: paths for VASP, QE and the new codes use a 0.02 Å symmetry tolerance, and every path segment gets at least 10 k-points. A relaxed silicon cell had shown monoclinic labels, and the bundled silicon sample drew its bands as straight lines.
- Recompute the silicon sample with VASP: a PBE cell relaxation, a 16×16×16 mesh, the Γ–X–U|K–Γ–L–W–X path at 60 points per Å⁻¹ and a tetrahedron DOS.
- Draw relaxation energies on whole ionic steps, also when a relaxation converges after one step; QE relaxation plots now mark each step.
- Say when VASP stops because it finds no licence file, instead of reporting a missing vasprun.xml.
- Draw QE, ABINIT, CP2K and Elk band structures and DOS like the VASP ones: an 8 eV window on either side that stops where the computed bands end, spin up and down with a legend, joined labels at path jumps (A|L, which QE had shown as L) and subscripts (H₂ instead of H_2). Insulators start at their valence band maximum. Earlier plots included every semicore band down to −120 eV.
- Give monolayer band paths only their in-plane segments, antiferromagnets the path of their magnetic cell, and spin–orbit runs without a magnetic moment the time-reversal path: GaAs and MoS2 had repeated theirs with primed points.
- Add empty bands to QE band, NSCF and DOS stages, and a few to insulator SCF runs, which now report a gap. QE band plots and DOS of insulators had shown no conduction bands. The count uses each UPF's valence, now recorded at installation and listed for the bundled libraries.
- Converge QE band and NSCF eigenvalues to 1e-6 Ry, about 1e-5 eV. pw.x's default of about 1e-11 Ry stalled or stopped the diagonalization for GaAs, Fe and CrI3.
- Add spin–orbit coupling to QE with the fully relativistic PseudoDojo UPFs (`--library pseudodojo-fr`), pinned per file.
- Run spin-polarized QE insulators with a narrow Gaussian smearing; pw.x stops on fixed occupations with two spins unless the total moment is fixed.
- Compute QE DOS with optimized tetrahedra on three-dimensional meshes and a 0.14 eV Gaussian on layers; dos.x had kept an insulator's unbroadened occupations, which gave a spiky DOS. Run dos.x on one MPI rank, which on many ranks stopped or hung for spin-polarized iron.
- Read both spins of CP2K 2026 DOS files, which hold them in one file and of which only spin up had been read, and recognise metals there by their fractional occupations: Al and Co had shown gaps of 0.12 and 0.05 eV. Show Elk's spin–orbit DOS of a non-magnetic crystal as one total.
- Converge ABINIT cells with a vacuum gap: graphene, MoS2 and CrI3 kept a large potential residual, and an MoS2 relaxation stopped on the resulting noisy forces. Keep ABINIT spinor band counts even (a Bi2Se3 spin–orbit band run stopped) and give ABINIT band paths the Fermi level of their SCF, so spin-polarized paths report a gap.
- Quote the program's own error when ABINIT or QE stops, including ABINIT's abort file from other MPI ranks, and offer CG diagonalization when QE's Davidson stops on a singular overlap matrix.
- Report zero gap when a band touches the Fermi level, as at graphene's Dirac point (Elk had reported 0.13 eV), and read CP2K path gaps from the Fermi level: CP2K's band file mislabels the occupations of magnets, which made CrI3 look metallic.
- Planners now assign an obvious ferromagnetic order, or an antiferromagnetic order of two equal sites, themselves; see each UPF's functional and type; and name the programs that offer a method the chosen one lacks.
- Add Elk optics: the dielectric function on a dense mesh, and with a TDDFT kernel (RPA, ALDA or bootstrap) the response with local fields, which shows excitons such as LiF's. Add real-time TDDFT under a laser pulse, reporting the field, current, absorbed energy and a magnet's spin moment over time; a run with zero field checks that the moment stays constant.
- Run large insulating CP2K cells at Γ with orbital transformation (a 1000-atom silicon cell converged in 4 minutes on 256 cores), and Born–Oppenheimer molecular dynamics (NVT with a CSVR thermostat, or NVE) that reports temperature, energy drift, the final frame and g(r), and checks the second half for continued settling. Passing that check does not establish equilibrium. Later MD and electronic stages use the final MD frame, with fresh SCF calculations after the atoms move.
- Preserve CP2K trajectory coordinates in the program's cell frame. Keep the DDB from Γ-only ABINIT phonons, reject incomplete phonon tables, and check Elk spin-response timestamps. Magnetic Elk optics includes all three diagonal components by default.
- Add ABINIT phonons by density-functional perturbation theory. One job runs the perturbations on a q mesh, merges them with mrgddb and interpolates the dispersion and DOS with anaddb; nonmagnetic insulators also get Born charges, the electronic dielectric constant and the LO–TO splitting. QE and ABINIT phonon dispersions are drawn like band structures.
- Default QE hybrid SCF runs to `conv_thr` = 1e-6 Ry, which also ends the outer exact-exchange loop: with 1e-8 Ry, GaAs's exchange error levelled off near 1e-7 Ry and the run hit its time limit with the gap long settled.
- Give pw.x every cell rotated so its symmetry axes lie along x, y and z, where pw.x looks for them. A cubic primitive cell from a CIF had shown pw.x 2 of GaAs's 24 symmetry operations, so an 8×8×8 SCF used 150 k points instead of 29, and phonons 24 q points instead of 8.
- Report how far QE and ABINIT phonons miss the acoustic sum rule and charge neutrality before these are imposed, and warn when that makes a result unreliable; QE phonon stages now also warn about imaginary branches. With SSSP's Ga and As UPFs, QE 7.6 gave GaAs 118 cm⁻¹ at Γ and wrong transverse branches at every density cutoff and k mesh tried, while PseudoDojo norm-conserving UPFs gave branches within 11 cm⁻¹ of ABINIT.
- Add `kshift` to QE, which moves the k mesh by half a step: on an 8×8×8 mesh through Γ, QE and ABINIT gave GaAs an electronic dielectric constant of 27–30, QE on the shifted mesh 13.6.
- Require norm-conserving UPFs for QE electron–phonon coupling: QE 7.6's ph.x stopped with a segmentation fault in that step with PAW UPFs for aluminium, while PseudoDojo norm-conserving ones ran through.
- Fix Elk details found on the cluster: lithium compounds use the finer radial mesh of Elk's own LiF examples (LiF had not converged with PBE), a TDDFT spectrum that leaves out zero frequency is read, and a laser run is no longer judged by the static-field iteration Elk writes before time zero.
- Run VASP HSE06 with its own screened exchange. With `HFRCUT = -1`, VASP 5.4.4 truncated the bare Coulomb kernel instead, and silicon came out with PBE0's gaps (1.97 eV, 4.0 eV at Γ); it now gives 1.31 and 3.34 eV. Only PBE0 keeps the truncation.
- Split VASP hybrid-functional runs into k-point groups (KPAR) that fit the job's tasks and k points. Without it, VASP 5.4.4 spread each k point over all ranks and rounded the band count up to their number: a silicon HSE06 SCF on 64 ranks took 22 minutes instead of 34 seconds, and its band path did not finish in two hours. A run whose task count no longer divides KPAR asks to be prepared again. Hybrid band paths now also run on VASP 5, which has no adaptively compressed exchange and ignores `LFOCKACE`; they had been refused there. VASP band plots now stop 0.1 eV below the highest computed band, as the other codes' plots do: with fewer bands, the highest one had drawn false kinks where uncomputed bands cross it, and its lowest point had traced the frame.
- Keep QE electron–phonon results whose interpolated α²F has a few negative weights at the lowest frequencies, as matdyn's Fourier interpolation leaves for narrow broadenings: a broadening is kept and reported when they stay within 1 % of its λ, and excluded beyond that. Aluminium had been refused for 0.64 % of λ in one of ten broadenings. Acoustic modes at Γ that ph.x leaves a few cm⁻¹ imaginary with zero coupling now reach lambda.x at zero frequency, recorded in the stage, instead of stopping the run; lambda.x would have turned them into NaN spectra. Electron–phonon runs prepared before the rotated pw.x frame continue to EPW.
- Copy into each QE stage only the files it reads from the run it continues: q2r, matdyn and lambda.x take its XML, and band, nscf and DOS stages its charge density without wavefunctions. ph.x's scratch had been copied into every later stage of an electron–phonon run, 53 GB each for aluminium, and a DOS stage copied 49 GB of Bi2Se3 nscf wavefunctions that dos.x does not read.
- Keep stages accepted by an earlier version usable for explanations and continuations: a field that a newer analysis adds, such as the phonon sum-rule check, no longer counts as an edited number. Every saved number must still match the fresh analysis.
- Accept lambda.x tables in which a broadening's coupling is too weak for the chosen μ* to give a McMillan Tc; lambda.x prints asterisks there, which had stopped a coarse aluminium check. That broadening is reported without a Tc.
- Pad remote commands of about 8 KB, which OpenSSH's connection sharing on macOS cannot send (`mm_send_fd: sendmsg(2): Message too long`). Commands of 8084–8124 characters failed every time, and an electron–phonon run's output collection on Arrhenius had hit that range.
- Read isotropic EPW gap and Tc runs from EPW 6.0 and 6.1 (QE 7.5 and 7.6) as well as 5.9. EPW 6 renamed its Allen–Dynes estimate the "McMillan expression" and prints it with fewer digits, so niobium's finished run on Arrhenius had been reported as not finishing. Anisotropic runs are still read for EPW 5.9 only.
- Keep the scripts DFT Agent runs on the cluster compatible with Python 3.6. The VASP hybrid warm start used a Python 3.10 feature and stopped on clusters with Python 3.9. Its PBE seed now uses `ISYM = 2` when the hybrid uses 3, which reduces the k-points the same way: VASP 5.4.4 stops a PBE run with `ISYM = 3`.
- Give the QE, Elk, ABINIT and CP2K planners the programs' own manuals and examples for the version an environment runs: QE's input descriptions (`INPUT_*.def`) of the programs DFT Agent runs with its official PW, PHonon, PP, NEB and EPW examples, Elk's input-block and task reference and its example inputs from the pinned release, ABINIT's input variables (abinit and anaddb), tutorials, topic pages and tutorial inputs from the pinned release, CP2K's keyword reference from the installed program (`cp2k --xml`) with its documentation pages and example inputs from a pinned release, and DFT Agent's own cluster checks. They are downloaded on first use and searched locally. The planner gets a few matching excerpts and lists them in the plan; in Automatic mode the agent searches and reads them. Nothing is bundled, as the texts are GPL. Native QE namelist options in a plan are checked against the input description of the environment's QE version, so a misspelt or misplaced name, such as mixing_beta under SYSTEM, is a question before submission instead of a failed job. The ABINIT planner now also knows which `occopt` each `electronic_type` writes, so a request in ABINIT's own terms, such as occopt 7 with tsmear 0.01 Ha, maps to its settings instead of being declined.
- Say which host failed and where a verified copy can go when a pseudopotential download fails, instead of showing a traceback.
- Show the warnings of ABINIT, CP2K and Elk steps with their results, and pass them to the planner.
- Simplify the interface: setup steps use short forms instead of chat bubbles, the project header is one row, the modes are named Automatic and Step by step, Settings groups the project, archive, storage and the classic workbench, and more labels are translated.

## 0.4.4 — 2026-10-19

- Organize files, conversations and calculations in separate projects.
- Guide model and cluster setup, with Codex CLI, Claude Code CLI and API connections.
- Use CLI agent tools to plan, prepare, revise and analyze work in a project. Review and submit jobs in the app.
- Connect VASP, QE, Wannier90 and Python steps through checked file transfers.
- Detect QE build tools, review installation settings and follow the build from Connections.
- Preserve calculation attempts, QE installation progress and earlier runs when reconnecting or repairing.
- Reuse Windows password/key + verification-code sessions across background jobs; ask for fresh sign-in after disconnection.
- Update the desktop interface and installation checks for macOS and Windows.

## 0.4.3 — 2026-10-14

- Add Quantum ESPRESSO installation, public UPFs and a VASP/QE switch.
- Plan and run QE calculations through conversation, including phonons, NEB and electron–phonon coupling.
- Add Wannier/EPW spectra, isotropic Eliashberg gaps and Tc brackets, and anisotropic gaps.
- Add a bounded single-band Hubbard RPA pairing workflow with explicit model inputs.
- Continue from accepted stages, preserve frozen inputs during repair, and verify collected files before analysis.
- Include raw outputs, numerical tables and execution receipts in downloads. See [QE validation](validation/qe-0.4.3/README.md) for tested scope and limits.

## 0.4.2 — 2026-10-10

- Add desktop packaging for Apple Silicon Macs (`.dmg`) and Windows x64 (`.exe`), including Python and model SDKs. Installation and platform acceptance are tracked separately from publication.
- Prepare and review inputs without a cluster; attach cluster settings at submission.
- Load four completed sample runs, including SrTiO3 with a 353-point band path.
- Show periodic crystals with separate lattice axes and element labels.
- Use denser band paths and export plots as PNG, PDF and SVG.
- Correct VASP XML ionic energies in relaxation plots and CSV files; refresh older plots from saved outputs.
- Add CSV downloads beside plots and show raw files directly.
- Return projected DOS, forces, stress, convergence tables and selected cluster files.
- Detect partitions, accounts, VASP modules and POTCAR folders from the cluster, with presets for known clusters.
- Remember API keys in the system keychain on request; `dft-agent key` manages them.
- Open the browser automatically and hide the Streamlit toolbar.
- Move settings to `~/.dft-agent/` and runs to `~/dft-agent-runs/`; earlier locations keep working.
- Show runs as a table, cluster checks as a checklist with fixes, and clearer errors.
- Add screenshots, a Chinese README, an install guide and a Windows guide.
- Notice new releases in the app.
- Interactive 3D structure viewer with a static fallback.
- Resume monitoring at startup and notify when a run finishes.
- Switch the interface between English and Chinese.
- Connect from native Windows with bundled SSH/SFTP support; WSL remains optional.
- Preserve cluster configurations and unfinished input fields when switching configuration files or interface language.
- Preserve launch commands and environment prerequisites during cluster detection, verify keychain deletion, and send Discord-compatible notifications.
- Document desktop and source installation.

## 0.4.1 — 2026-10-09

- Generate k meshes from cell size and detect vacuum directions.
- Choose smearing, symmetry and PAW defaults with explicit overrides.
- Start hybrid calculations from a checked PBE WAVECAR.
- Support multiple nodes and extra Slurm resource options.
- Use structured Claude output and specific provider errors.
- Preserve prepared calculations when upgrading.

## 0.4.0 — 2026-10-09

- Plan structure edits and calculations in one conversation.
- Choose methods by stage and compare magnetic orders, U values and strains.
- Extract sampled band gaps and continue from accepted structures.
- Connect scientific checks to planning and preserve stage settings during repair.

## 0.3.3 — 2026-10-09

- Add Claude, Qwen, Grok, GLM and DeepSeek providers.
- Keep each provider's credentials separate.
- Include setup instructions and token counts for the new providers.

## 0.3.2 — 2026-10-09

- Add model setup, API key and token billing instructions.
- Walk through a first plan without a cluster.
- Show provider help in Model and disable unused Codex fields.

## 0.3.1 — 2026-10-09

- Include checked guidance, calculation cases and repair examples.
- Search memory and import selected local records.
- Keep rejected and unverified lessons out of recommendations.

## 0.3.0 — 2026-10-09

- Restore scientific guidance and relevant run history in planning.
- Review failed runs and prepare bounded repairs.
- Edit structures through conversation and convert CIF/POSCAR.
- Add nine structures from earlier cluster calculations.
- Keep model context compact and record reported token use.

## 0.2.1 — 2026-10-07

- Explain results and answer follow-up questions.
- Save plan revisions and retain them after a failed model request.

## 0.2.0 — 2026-10-07

- Plan and revise calculations with a language model.
- Add magnetism, SOC, DFT+U, HSE06 and PBE0.
- Chain calculations and compare magnetic seeds.

## 0.1.3 — 2026-10-07

- Renamed to DFT Agent.

## 0.1.2 — 2026-10-07

- Shorter text.

## 0.1.1 — 2026-10-07

- English UI and docs.

## 0.1.0 — 2026-10-07

- Initial release: relax, SCF, bands and DOS.
- Job recovery, structure preview and result downloads.

[Validation](docs/validation.md)
