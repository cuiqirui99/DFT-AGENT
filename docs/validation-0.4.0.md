# v0.4.0 validation

Checked on 9 October 2026.

| Workflow | Result |
| --- | --- |
| Si supercell, Ge substitution, PBE relaxation, SOC bands | Three stages accepted |
| PBE relaxation followed by HSE06 bands | Two stages accepted |
| NiO magnetic orders, U values and strain | Four variants accepted; only matched magnetic cases ranked |
| Continue from an accepted relaxation to DOS | Two stages accepted; original run unchanged |
| Missing settings and plan revision | Real model asked for missing geometry and spin axis; clarification produced the requested plan |

All 11 new Slurm jobs completed and passed the retained-output checks. Five real
Codex calls covered planning, clarification, revision, batch variants and continuation.
Inputs and output hashes are in the [acceptance record](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/tasks-0.4.0.json).

Offline tests cover stage dependencies, site-order preservation, failed results,
changed inputs, source-bound continuation, batch isolation, repair settings,
provider adapters and the app. Release CI runs on Python 3.11, 3.12 and 3.13.

These are workflow checks at the recorded settings. The band gaps cover sampled
k points. No precision convergence or exhaustive magnetic search was performed.
Raw VASP outputs remain local; the archive includes inputs and result hashes.
