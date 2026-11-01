# Scientific guidance

The planner retrieves short local notes from `guidance.py`. It uses the goal,
recent user replies and structure summary. No network access or private corpus
is needed. The notes are original summaries, with links to the VASP documentation
checked on 9 October 2026.

| Topic | Source |
|---|---|
| Cutoff and comparisons | [ENCUT](https://www.vasp.at/wiki/index.php/ENCUT) |
| Magnetic seeds and local moments | [MAGMOM](https://www.vasp.at/wiki/index.php/MAGMOM) |
| Dudarev U/J and charge reuse | [LDAUTYPE](https://www.vasp.at/wiki/index.php/LDAUTYPE) |
| SOC and spin coordinates | [LSORBIT](https://www.vasp.at/wiki/index.php/LSORBIT), [SAXIS](https://www.vasp.at/wiki/index.php/SAXIS) |
| Hybrid band convergence | [Hybrid bands](https://www.vasp.at/wiki/index.php/Band-structure_calculation_using_hybrid_functionals) |
| PBE charge reuse | [ICHARG](https://www.vasp.at/wiki/index.php/ICHARG) |
| Occupations | [ISMEAR](https://www.vasp.at/wiki/index.php/ISMEAR) |
| Vacuum and surfaces | [Electrostatic corrections](https://www.vasp.at/wiki/index.php/Electrostatic_corrections) |

This restores the earlier project's source-linked method guidance and plan review
without importing its runtime or data. Two old assumptions are corrected: hybrid
bands must not use fixed charge, and zero net moment can be normal for AFM.

Chemistry and vacuum detection are hints, not measurements or literature evidence
for a material. They never select U/J, magnetic order or extra tasks. Explicit user
choices take priority. Keyword matching selects notes; the model and plan validator
still handle intent. Review adds explanations and questions without changing inputs.
Missing inputs for a selected method must be supplied before preparation. Checks
record their source, applicable method and place in the workflow; a pending check
is not a validated result.

These notes do not establish a ground state or property convergence. They do not
establish charged-cell treatment or an exhaustive magnetic-order search.
