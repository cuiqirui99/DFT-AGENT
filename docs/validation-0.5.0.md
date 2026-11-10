# 0.5.0 checks

The 0.5.0 app was built on an Apple Silicon Mac from a frozen snapshot of the
development tree with the same dependency versions as the 0.4.9 and 0.4.10
builds (PyInstaller 6.22, Python 3.13).

## Automatic cluster setup on a real cluster

Paracloud (CentOS 7, glibc 2.17, Slurm 23.11 with PBS-style wrapper commands,
Environment Modules 4.1 with about 600 modules) was set up six times during
development on 6 November 2026. The user signed in once in the app, with a
password and a login code. Later runs were started with Check again or after
the app was rebuilt; run 4 was started from a terminal. Problems a run showed
were fixed before the next one, except the run 4 problem below, which was
found only after run 6.

- Run 1: the module listing did not answer within 40 s, so the model judged the
  cluster from an empty module list. It identified Slurm, found the user's VASP
  and Wannier90, and reported most other programs as not found.
- Runs 2, 3, 5 and 6 also asked the connected model (Claude Code CLI) for its
  judgement, ran its module loads and version commands on the login node and
  submitted short test jobs.
- Run 4 was driven from a terminal, not from the app. Its model call timed out,
  it fell back to the built-in rules and was stopped. It had rewritten the
  Quantum ESPRESSO environment that run 3 had verified with `mpirun`, putting
  `srun` back; this is why run 6's first Quantum ESPRESSO test failed. A check
  without the model now keeps what a judgement set up.
- Run 5 stopped while preparing the LAMMPS test: the packaged app lacked ASE's
  LAMMPS data writer. All of ASE except its own test suite is now included, and
  a test that cannot be prepared is marked as not tested instead of stopping
  the run.

The earlier runs also led to these fixes: a module listing that needs more than
40 s (now 90 s, plus the module files under `MODULEPATH`), the PMI failure under
`srun` (a launcher per program), CP2K's older energy line and the minimum CP2K
version 2023.1.

Last run (run 6, 18 minutes):

- Scheduler: Slurm, judged from the `--version` lines and the Perl wrapper
  scripts next to `sbatch`, and confirmed by the first finished test job. The
  model chose the user's partition and account and the `srun` launcher.
- Verified by test jobs: the user's own VASP 6.2.1 with the Intel 17 modules;
  Quantum ESPRESSO 7.4.1 (`qe/7.4.1-oneAPI.2022.1` after `module purge`);
  ABINIT 9.4.2; CP2K 2024.1 (`cp2k/2024.1-para`, chosen because the site's
  CP2K 2025.1 build stops with an illegal instruction on the AMD compute nodes);
  SIESTA 5.4.2 (an earlier app installation found through its receipt); LAMMPS
  2Aug2023 (`lammps/intel20/lammps-2Aug2023`).
- Checked on the login node: a site Python 3.9.6 module with numpy, and the
  user's Wannier90 3.1.0.
- Quantum ESPRESSO and LAMMPS failed their first test under `srun`, where their
  Intel MPI stops with PMI errors. The model read the failed jobs' output and
  switched both programs to `mpirun`, and the retest passed.
- Offered for installation: Elk (the `elk/4.8.0` module provides no `elk`
  program) and UppASD (not installed). Automatic installation was off, so
  nothing was installed. RSPt needs the user's own installation.
- The verified load commands, programs, launchers and versions are kept as the
  cluster's experience and shown to the model at the next check.

Run 6 used a packaged development build, made at 12:13 CET from the same setup
code as the first 0.5.0 build, not the final release build. The release build
adds the changes listed at the end of this page.

## RSPt and UppASD

Functional checks on 3 November 2026 ran on a private single-node Slurm 23.11.4
cluster in an isolated container on an ARM64 machine (2 CPUs, 6 GiB, real Slurm
accounting), with RSPt and UppASD installations mounted read-only. They used
the development code of that day, not the packaged app.

- RSPt: a silicon SCF, bands and DOS sequence ran and was collected. After the
  connection was dropped and reopened, the app resumed the same job. Queued and
  running jobs were cancelled, and a deliberately unconverged SCF was rejected
  before the bands step.
- RSPt to UppASD: in a real Codex conversation, a three-step Fe project computed
  RSPt exchange, passed it unchanged to UppASD spin dynamics and a Monte Carlo
  run, and revised each step on request.
- UppASD 6.0.2 was built by the app's installer from the pinned source (GNU
  Fortran, CMake, BLAS/LAPACK) on Linux ARM64 and passed its smoke test; Linux
  x86_64 and PBS were not exercised. Twelve runs of UppASD's own examples and
  tests (DMI, tensor exchange, a random alloy, GNEB, spin–lattice dynamics, a
  three-temperature model, spin-transfer torque, dipolar interaction)
  completed, and magnon spectra of short spin chains matched the analytic
  dispersions within 0.00001 meV.
- UppASD spin-state controls ran as UppASD 6.0.2 Slurm jobs on the same
  cluster: relaxation in a field to the analytic alignment, refusal of an
  unfinished relaxation, sequential Monte Carlo and spin-dynamics anneals,
  reuse of a saved spin state, and a spin spiral on a two-site basis. Each
  gave the expected result.
- Fe magnetic anisotropy: eleven RSPt SCF jobs completed. Increasing the k mesh
  from 6³ to 8³ changed the [111]−[001] energy difference from −8.37 to
  +3.19 µeV/atom, so the easy axis stays unresolved.

These are functional and sensitivity checks, not converged materials results.

## Test suite

Full suite on the final frozen source on 10 November 2026: 6,778 passed,
8 skipped, 0 failed (pytest, Python 3.13, 14 min 42 s). The npm launcher test
ran with Node.js 22 on PATH. The skips are four Windows PowerShell tests and
four test modules that need paramiko, which only the Windows build includes:
the SSH broker, its acceptance test, Unix SSH onboarding and the Windows
transport. All 34 tests in those modules passed separately on macOS with
Paramiko 4.0.0 against the same frozen source.

## Packaged app

- Ad-hoc signature verified; the bundle's self-check (`--check`) passed for
  input preparation (including RSPt exchange, DFT+U and DMFT stages and a
  LAMMPS input), cluster attachment, samples, resources, provider SDKs,
  first-run and project interfaces, native keychain, SSH broker lifecycle,
  model setup, CLI setup script, MCP subprocess, plot exports and the QE
  catalogues, inputs and installation plan.
- The DMG was mounted, installed and compared file by file with the build; the
  installed app passed the same checks, served its pages and rendered its
  window. The app does not ship the DFT Agent Python source tree. It includes
  the runtime helpers needed to prepare and run jobs.
- The installed app was opened through macOS Launch Services in an empty test
  workspace. Its native window showed version 0.5.0, detected the signed-in
  Claude Code CLI, and reported a successful connection after one test request.

## Fixes after the first 0.5.0 build

The first 0.5.0 build was not published. Checks after it found these problems,
which the release build fixes:

- RSPt exchange, DFT+U and DMFT stages failed to prepare in the packaged app,
  because their helper scripts were read from source files that the app does
  not ship. They now come from the app's bundled scripts, and the app's
  self-check prepares these stages.
- A bundled library (orjson) was built for macOS 15 and later, as in earlier
  Mac releases, so the app's pages could fail to load on macOS 14. The release
  build bundles orjson's build for macOS 11 and later, and the packaging check
  now fails when a bundled file needs a newer macOS than the app's minimum,
  14.0.
- A project whose folder was moved or deleted outside the app stopped the
  sidebar, so Connections and Settings could not be reached, as in releases
  since 0.4.4. The other projects, Connections and Settings now stay reachable.
- A check without the model, such as run 4, could rewrite an environment that a
  model's judgement had set up. Such a check now keeps it.

During the release review, model-suggested setup commands were changed to use
only locally checked program paths and arguments. Saved model suggestions are
checked again when setup resumes. An offline replay of the recorded run 6
probe and accepted settings preserved all six solver environments and both
login-node tools, including the `mpirun` launchers for Quantum ESPRESSO and
LAMMPS. This replay did not submit new jobs.

The release build also starts the automatic setup only within an hour of a
sign-in or a successful connection test, and names runs under
**Attach results** by step and start time. A cluster that the built-in rules
set up, because no model was connected or the model did not answer, is now
checked again with the model once, after a model connection test has succeeded
since that setup finished; installing a program or turning on automatic
installation afterwards does not repeat this check. Tests cover these changes;
no cluster run used them.

Not covered: the app on a macOS 14 machine, installation through the Install
button or by automatic installation on a real cluster, the automatic setup on
a PBS cluster, storage quota reading on GPFS and BeeGFS (checked against
sample output only), the one-rank copy check with a real mismatched MPI
launcher (checked with a simulated one), and UppASD builds on Linux x86_64.
This release contains no new Windows installer. No test establishes
compatibility with every cluster, engine configuration or material.
