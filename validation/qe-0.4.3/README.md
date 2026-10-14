# QE acceptance

Development version: `0.4.3.dev0`. **Release approval is pending.**

[report.json](report.json) records actual Paracloud CPU jobs, accepted and failed stages, pending work, model usage and file hashes. These checks use QE 7.4.1 and EPW 5.9. A completed scheduler job alone does not pass a test.

The exported inputs and selected native outputs are in `calculations`, `native` and `pairing`. [files.sha256.json](files.sha256.json) lists their checksums. Cluster settings, credentials, pseudopotential contents and large restart files stay private. Derived JSON reports omit private locations and retain the original report's checksum.

## Coverage

| Route | Evidence |
| --- | --- |
| Structures and spectra | Si relaxation, checkpoint reuse, bands, DOS, projections and charge density |
| Magnetism and DFT+U | Fe comparisons, final moments, NiO sublattices and explicit Hubbard settings |
| SOC and hybrid functionals | Fully relativistic Si calculations and separate PBE/HSE06 stages |
| Phonons and NEB | Al phonon dispersion; H₃ paths with separate endpoint checks |
| Electron–phonon coupling | Al DFPT and an accepted direct q-point integration; failed Fourier interpolation retained |
| Isotropic Eliashberg | Eight native EPW cases across two broadenings and two frequency cutoffs, checked against independent linear and nonlinear equations |
| Anisotropic Eliashberg | A complete pinned upstream MgB₂ regression; the stricter continuation and generic app workflow remain separate acceptance gates |
| Model pairing | Single-band Hubbard RPA checked against TRIQS/TPRF, an independent matrix, and Paracloud results |
| Installation | Managed Slurm compilation and an independent Si comparison with the site installation |

These are functional and software regression tests. Their meshes, cutoffs and broadenings do not establish converged material predictions or a reliable material transition temperature.

## Model workflow

Actual Codex calls covered planning, missing settings, revisions, magnetic comparisons, result explanations and bounded repair. Electron–phonon and pairing plans include explicit scientific choices. The pairing plan was executed through the shared Slurm workflow and its verified results were returned to the model.

The single-band benchmark uses an explicit model Hamiltonian. It does not validate a QE-derived material Hamiltonian, a multiorbital mechanism or a material Tc. Both pairing channels and the unstable-input rejection are retained. See [Superconductivity](../../docs/qe-superconductivity.md) for the method limits.

Only Codex was called in this campaign. Mock tests for other providers are not live API acceptance.

## Failures and remaining checks

The archive retains unconverged Si bands, an incompatible no-SOC pseudopotential attempt, the initial H₃ endpoint limitation, negative Al spectral weights, and failed EPW execution attempts. Subsequent runs have separate IDs; they do not replace those records.

Al grid refinement, the application EPW sequence, and the application MgB₂ sequence are tracked stage by stage in the report. Native EPW regression results do not automatically accept those workflows. QE GPU execution and new macOS/Windows installers require separate validation.
