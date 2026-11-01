# v0.3.0 validation

Functional checks completed on 2026-10-09 using VASP 6.2.1 and 5.4.4 on two
Slurm clusters. These checks test the workflow, not material-specific accuracy.

| Check | Result |
|---|---|
| Offline tests | 488 passed; zero failures or skips |
| Cluster results | 71 accepted stages, each independently checked |
| Calculation dialogue | [11 real Codex calls](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/dialogue-0.3.0.json) checked revisions, methods and missing inputs |
| Structure editing | [14 turns, 19 checks and nine exports](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/structures-0.3.0.json) covered supercells, substitution, vacancies, slabs, coordinates and conversion |
| Model use | [Three paired cases](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/token-use-0.3.0.json) recorded actual usage and retained their checked constraints |
| Installation | [Two structure turns produced Si5Ge](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/installed-0.3.0.json), followed by relaxation, SCF, explanation and a checked bundle outside the checkout |

See the [cluster records](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/cluster-0.3.0.json) and
[feature matrix](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/features-0.3.0.json) for inputs, hashes,
individual outcomes and the distinction between offline and live tests.
Current package rechecks preserve the original calculation and model receipts.

## Methods and recovery

The campaign covered PBE, collinear and noncollinear magnetism, SOC, DFT+U,
HSE06 and PBE0, including mixed-method probes and NM/FM/AFM seed comparisons.
Relaxation, SCF, bands and DOS were checked where supported by the solver.
Input receipts, method settings, energies, spectra, forces and structure handoffs
were checked independently of the product parser.

Three reviewed repairs passed: NELM=1 to 120, an increased ionic-step limit,
and a Slurm time limit of one to two minutes. The timeout used a 75-second
startup delay. Repairs preserved their original runs, structures, k-points,
methods and thresholds. A separate sleep-job control checked lost submission
replies, reconnecting without duplicate jobs, process restart and cancellation.
Missing solver output was rejected without a repair submission.

## Limits and unsuccessful cases

All 17 packaged examples passed local import and preparation. Of nine added
materials tested on the clusters, seven SCF calculations passed. GaFe2O4 reached
NELM=160 without convergence. The 114-atom CrOCl-MoS2 example was stopped at
the recorded 30-minute validation budget; its outputs were retained. Neither
counts as an accepted scientific result.

VASP 5 cannot run this release's hybrid-band recipe, which requires VASP 6 and
LFOCKACE. Its Si PBE0+SOC band probe was rejected and the following DOS stage
was not submitted. A separate VASP 5 LDAUTYPE representation issue was fixed;
the original NiO PBE0+U+SOC job then passed rechecking with identical inputs and
solver outputs.

For VASP 5, independent XML and OUTCAR checks reconstructed the energy using
the final SCF correction. This resolved an audit-field mismatch without changing
the application's energies. Original and revised checks remain in the records.

Live model checks used Codex. Responses and compatible Chat Completions adapters
have offline tests. Earlier dialogue records state which later changes they
predate. Generated geometry still needs physical review and relaxation; these
tests do not establish every material, method combination or magnetic ground state.
