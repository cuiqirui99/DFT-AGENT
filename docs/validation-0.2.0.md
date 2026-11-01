# v0.2.0 validation

Tested on 2026-10-07 with VASP 6.2.1 on one Slurm cluster.
These are workflow checks, not material-specific convergence studies.

| Check | Result |
|---|---|
| Offline tests | 254 passed |
| Real model calls | Five requests and one follow-up passed |
| Fe magnetism | NM, FM and AFM seeds completed on the same cell |
| Fe SOC | Noncollinear SCF completed with `vasp_ncl` |
| NiO +U | AFM seed completed with explicit Ni U=5 eV, J=0 |
| Si HSE06 | SCF, bands and DOS completed |
| Si PBE0 | SCF completed |
| Si PBE chain | Relaxation → SCF → bands completed |
| Wheel candidates | Model → SOC relaxation → SCF → bands; final wheel: model → HSE06 SCF → bands → DOS |

The seven workflows contain 15 solver stages. A separate standard-library
XML/OUTCAR reader checked method tags, input hashes, energies, moments and
spectral outputs. Both chains retained the relaxed structure and Cartesian
lattice frame in later stages. The installed wheel also produced a results ZIP.

The live model tests used Codex CLI. They covered method selection, missing U/J,
unsupported requests and a follow-up that retained explicit exclusions.
OpenAI and compatible API adapters have offline tests; they were not tested
against live API services in this campaign.

## Limits and recovery

The Fe ranking compares three initial seeds, not all magnetic orders. The NiO
cell and U/J are test settings. Neither establishes a material's ground state.

The first HSE06 band attempt returned nonfinite forces. Recollection handled
those correctly, but plot inspection then found a 1.2721 eV disagreement at
repeated Γ points. Raising the minimum iteration count to 60 did not fix it.
That attempt is excluded from the accepted cases.

Hybrid bands now use Davidson iteration and reject equivalent-point energies
that disagree by more than 0.05 eV. The replacement passed this check.
Static hybrid-band forces are omitted when nonfinite; relaxation and SCF still
reject them. Failed attempts are retained in the records.

Hybrid zero-weight path convergence was not independently established.
Empty-state and full-path convergence need separate checks. These tests do not
replace cutoff, k-point or property-convergence studies.

## Records

- [Solver cases and hashes](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/methods-0.2.0.json)
- [Independent audit](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/methods-audit-0.2.0.json) and [reader](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/methods-0.2.0/audit.py)
- [Model checks](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/agent-0.2.0.json)
- [Hybrid band diagnosis](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/hybrid-orbitals-0.2.0.json)
- [Installed wheel](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/installed-wheel-0.2.0.json)
- [Test structures](https://github.com/cuiqirui99/DFT-AGENT/tree/v0.4.7/validation/methods-0.2.0)
- [Earlier PBE validation](validation.md)

Inputs are reproducible from the recorded structure and parameters. VASP and
POTCAR files require your own license. Cluster credentials and licensed files
are excluded from the repository.
