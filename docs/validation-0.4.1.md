# v0.4.1 validation

## Live calculations

VASP 6.2.1 on Slurm, using licensed PBE potentials. Inputs and output hashes are
recorded in [the results](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/input-defaults-0.4.1.json).

| Case | Settings | Result |
| --- | --- | --- |
| Si SCF | Automatic 9×9×9 mesh, 520 eV | −5.421644395 eV/atom; sampled gap 0.5845 eV |
| Al SCF | Automatic 11×11×11 mesh, ISMEAR=1, SIGMA=0.2 eV | Accepted; −3.75397105 eV/atom |
| Si HSE06 bands | Explicit 2×2×2 mesh plus path | PBE seed and WAVECAR restart verified |
| Si HSE06 + SOC | Explicit 1×1×1 mesh | PBE seed and spinor WAVECAR restart verified |

These are input-policy and workflow checks. They do not establish converged
material properties. The Al occupations from Methfessel–Paxton smearing fall
outside the gap parser's supported range, so no numerical gap is reported.
The hybrid meshes are deliberately coarse restart tests. Accepted jobs were
7465955, 7465956, 7465992 and 7465993. The first two hybrid attempts (7465969 and
7465971) stopped after converged seeds because of XML field representations;
the parser was corrected before fresh attempts. Those failed records remain
separate. Final validation checks actual band counts, k-point order, spin,
cutoff, file hashes and VASP's successful WAVECAR-read message.

The older Si smoke test used 350 eV and a 2×2×2 mesh. Retained outputs with the
same POTCAR hash and volume give −4.1446 eV/atom for that recipe, −5.3284 at
400 eV/4×4×4, and −5.4105 at 520 eV/6×6×6. Both cutoff and mesh changed;
these runs do not isolate their separate effects. Historical cases remain
available as workflow evidence and carry no claim of numerical convergence.

## Model and interface checks

A real Codex request planned a PBE calculation for a metal with automatic mesh
and smearing. A second request explained Si's measured sampled gap from retained
output facts. Claude's native structured-output requests were checked through
its actual SDK with simulated HTTP responses, including the minimum supported
SDK version. No live Claude API request was made: no API key was configured.

All 887 local tests passed, including the reference-tool contracts. The installed
wheel also prepared a hybrid job and revalidated all four accepted live runs.
Offline checks cover geometry-based grids, vacuum axes, explicit overrides,
smearing, symmetry, PAW labels, multi-node Slurm scripts, interface actions,
provider errors, failed seeds, incompatible restarts and preserved old inputs.
Multi-node execution and Dardel-specific policies were not tested on a live
cluster.

Nine independent input regenerations match v0.4.0 code byte-for-byte, including
PBE, magnetism, SOC and hybrids. Prepared old runs retain that input policy;
new runs use the updated defaults.
