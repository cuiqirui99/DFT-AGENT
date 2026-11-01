# v0.1.0 validation

For current checks, see [v0.3.0 validation](validation-0.3.0.md).
The previous release is covered in [v0.2.1 validation](validation-0.2.1.md).

The first release focused on getting from an input structure to a completed
VASP calculation and a structure you can download. Its validation finished on
2026-10-07 using VASP 6.2.1 and PBE.54 potentials on one Slurm cluster. This page
records those v0.1.0 tests, including the numerical limits and failed attempts.
The results show what worked in that environment; other materials and clusters
may need different settings.

## Completed checks

| Check | Observed result | Evidence |
|---|---|---|
| Eight materials, four tasks each | 32/32 workflows accepted; 48 successful solver stages | [Campaign](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/campaign.json) |
| Independent output parsing and file checks | All 48 stages checked; 720 file receipts matched actual hashes and sizes | [Evidence audit](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/reference/evidence-audit.v1.json) |
| Separately prepared SCF references | Si, Al and MgO passed the predefined energy and force comparisons | [Reference summary](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/reference/campaign-summary.v1.json) |
| Relaxation from displaced structures | Displaced Si and MgO moved internally, lowered energy and reached the predefined force threshold with the cell fixed | [Reference summary](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/reference/campaign-summary.v1.json) |
| Complete transfer | A 4 MiB SSH round trip retained identical bytes | [Transport](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/transport.json) |
| Monitor interruption | Restart resumed the same existing solver job | [Restart](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/restart.json) |
| Lost submission reply, reconnect and cancellation | A real scheduler job kept the same identity through process restarts and simulated connection failures; cancellation was confirmed | [Recovery](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/recovery.json) |
| Timeout and missing output | Actual Slurm TIMEOUT and a COMPLETED job without VASP output were both rejected, with no automatic resubmission | [Failure cases](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/failures.json) |
| Isolated wheel installation | The release wheel completed a real displaced Si relaxation, recovered its outputs and produced a result bundle | [Installed wheel](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/installed-wheel.json) |

The eight baseline structures are Si, C, Ge, Al, Cu, MgO, NaCl and SiC.
Each ran `relax`, `scf`, `bands` and `dos`; bands and DOS include a preceding
SCF stage. The separate reference campaign adds three manual baseline SCFs,
three sensitivity SCFs and two displaced relaxations using the same materials.
See the [fixed protocol](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/protocol.v1.json) and
[reference procedure](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/reference/README.md).

The v0.1.0 wheel's SHA-256 is
`dca668ad66b5cb9efd104a51e07dc89fd33b8bd4077ebfb679a1bc0599c048be`.
It was built from code commit `2f7026832b69f7b6a49e44615832328405a2d7ea`;
the v0.1.0 tag adds final documentation and evidence without changing its
Python modules or bundled example structures. It was installed outside the
source checkout with dependency checks passing. Its real Si optimization used
eight ionic steps and reduced the maximum force from 0.79756 to
0.00946 eV/angstrom; composition, fixed cell, coordinates and input identity
were checked. The installation test was automated.

Offline core, UI and reference checks recorded **138 passed, zero failed**.
The [CI workflow](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/.github/workflows/ci.yml) runs these checks on Python
3.11, 3.12 and 3.13 and installs a wheel outside the checkout. Real-cluster
acceptance is recorded separately above and is not run by public CI.

## Numerical sensitivity and failed attempts

The main campaign used 400 eV, a 4x4x4 mesh, EDIFF=1e-5 and a
0.03 eV/angstrom force threshold. New application inputs default to 520 eV.
These are starting settings; convergence depends on the material and property.
The manual baseline references matched the agent's energies and forces at the
reported output precision, checking input preparation and output parsing under
the same physical model.

Changing **both** cutoff to 520 eV and mesh to 6x6x6 produced these absolute
total-energy differences from the baseline:

| Material | Difference (eV/atom) |
|---|---:|
| Si | 0.082133 |
| Al | 0.361898 |
| MgO | 0.003059 |

These measurements show sensitivity to the settings. The coarse Al baseline
needs further convergence work before it can support small energy comparisons.
Because cutoff and mesh changed together, their effects cannot be separated
here. The symmetric SCF examples also have zero forces, so they do not test
force accuracy for distorted structures.

Initial C relaxation and Cu band calculations failed during diagonalization
with 64 MPI tasks. The attempts were kept in the record and followed by resource
adjustments to eight tasks, with unchanged declared physical inputs. VASP's
automatic band allocation also changed, so the tests do not isolate the cause.
The replacement runs passed. See the [C adjustment](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/resource-adjustments.v1.json),
[Cu adjustment](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/resource-adjustments.v2.json) and
[actual outcomes](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/resource-adjustment-outcomes.v1.json).
The 32/32 count includes these replacement runs. The resource changes were
made explicitly, not by the application's recovery logic.

These tests cover local, single-user, nonmagnetic PBE bulk workflows. Property
convergence, magnetism, SOC and other advanced methods need separate work.
Earlier platform test results are kept separate in the
[legacy scope note](legacy-scope.md).
