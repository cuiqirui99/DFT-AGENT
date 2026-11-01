# Memory sources

The installed package includes 20 active records: eight scientific notes, seven
successful examples, three reviewed repairs and two failure cases. It also keeps
21 inactive legacy entries so their original status is visible.

`verified` means the stated evidence was checked within its recorded scope. It
does not make a numerical setting universal, prove a magnetic ground state or
establish material accuracy. A verified failure case has a failed calculation
outcome; its evidence, rather than its calculation, passed review.

## Scientific notes

These are short original summaries, checked against the official VASP pages on
9 October 2026. The VASP documentation corpus is not redistributed.

| Topic | Source |
|---|---|
| Energy comparisons | [ENCUT](https://vasp.at/wiki/index.php/ENCUT) |
| Magnetic seeds | [MAGMOM](https://vasp.at/wiki/index.php/MAGMOM) |
| DFT+U | [LDAUTYPE](https://vasp.at/wiki/index.php/LDAUTYPE) |
| SOC coordinates | [LSORBIT](https://vasp.at/wiki/index.php/LSORBIT) |
| Hybrid bands | [Hybrid band calculations](https://vasp.at/wiki/index.php/Band-structure_calculation_using_hybrid_functionals) |
| PBE spectra | [ICHARG](https://vasp.at/wiki/index.php/ICHARG) |
| Occupations | [ISMEAR](https://vasp.at/wiki/index.php/ISMEAR) |
| Vacuum and surfaces | [Electrostatic corrections](https://vasp.at/wiki/index.php/Electrostatic_corrections) |

## Cases and repairs

The cases come from the [public v0.3.0 acceptance record](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/validation/results/cluster-0.3.0.json).
They cover the Si PBE chain, Fe magnetism and SOC, NiO with U, Si HSE06 and PBE0
bands, and a NiO HSE06+U+SOC SCF probe. Method choices and limits accompany each
case. The U values and coarse meshes are validation inputs, not defaults for a
new material.

The three repair examples preserve both sides of the change:

| Failure | Reviewed change | Recorded result |
|---|---|---|
| Si stopped at NELM=1 | NELM=120 | Electronic convergence reached |
| Displaced Si stopped at NSW=1 above the force target | NSW=100 | Original force target reached |
| Si job timed out after a 75-second startup delay | One minute to two minutes | Same calculation completed |

Only the recorded limit changed. These examples do not widen the executor's
repair actions or remove review. A separate missing-output case preserves a
refusal without another submission. The VASP 5 hybrid-band failure explains why
this runner requires VASP 6 for that recipe.

Each example includes its original INCAR, KPOINTS and POSCAR, a result summary,
and hashes of the available original outputs. Eighteen stage records were
checked: distributed inputs match their executed copies byte for byte, and the
accepted XML and OUTCAR hashes match the public receipt. This was a local
integrity check, not a new calculation or a repeat scientific validation.

Raw solver outputs, CHGCAR, WAVECAR, POTCAR and cluster configuration are not in
the package. Hashes identify the retained outputs; they do not let an installation
reparse files that are not distributed. A fresh run needs licensed PAW datasets
and VASP. Upstream charge files must be generated for dependent PBE spectra.

## Legacy migration

The old lessons database was read without changes on 9 October 2026.

| Original status | All modules | Selected DFT and general entries |
|---|---:|---:|
| Candidate | 18 | 16 |
| Shadow verified | 11 | 2 |
| Rejected | 21 | 3 |
| Verified | 0 | 0 |
| **Total** | **50** | **21** |

The selected inventory contains all 20 DFT entries and one general lesson about
keeping solver energy references separate. The other 29 belong to other
scientific modules. The inventory preserves each selected key, status, version
and source-row hash. Its short descriptions were reviewed for public release;
private database rows were not copied. No legacy entry was promoted.

Candidate, shadow-verified and rejected entries remain browseable but are
excluded from advice. Rejected describes the historical tested claim, which must
not become a recommended parameter set. Old executor fixes, GW/BSE proposals,
and property campaigns have not been revalidated in the current runner.

The 68 legacy experience records and four failure records stay local. Their
referenced files and scientific claims were not audited as a public collection.
Users can select local records for import; importing does not verify or publish
them. Personal goals, run paths, cluster details and unpublished data are not
part of the bundled collection.

The installed `knowledge/catalog.json` is the retrieval catalogue. Its `evidence/`
directory holds the case inputs, receipts, source checks and migration inventory.
