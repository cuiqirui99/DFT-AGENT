# VASP methods

These settings apply to VASP. Use them in a model plan, the **Method** form, or CLI `--parameters`.
Defaults are starting values, not convergence studies.

| Method | Settings |
|---|---|
| PBE | `"functional": "PBE"` (default) |
| Collinear spin | `"spin": "collinear"`, one `magmom` value per uploaded site |
| Noncollinear spin | `"spin": "noncollinear"`, one three-component `magmom` per site |
| SOC | `"soc": true`, `vasp_ncl_command` in the cluster config |
| DFT+U | `"hubbard_u": {"Ni": {"l": 2, "u": 5, "j": 0}}` |
| Hybrid | `"functional": "HSE06"` or `"PBE0"` |

Moments are in μB, in uploaded site order. Vectors use the SAXIS spinor basis;
`saxis` defaults to `[0, 0, 1]`. SOC with moments requires noncollinear spin.
SOC with `spin: none` uses a zero-moment seed; it does not constrain the final
magnetization. Noncollinear calculations also require `vasp_ncl`.
[VASP SOC guide](https://vasp.at/wiki/LSORBIT).

DFT+U uses Dudarev's `LDAUTYPE=2`, with U and J in eV. Supply both explicitly.
The Ni example above is illustrative. Element settings follow the actual
POSCAR/POTCAR order. [VASP DFT+U guide](https://vasp.at/wiki/LDAU).

HSE06 uses 25% exact exchange and screening of 0.2 Å⁻¹; PBE0 is unscreened.
New hybrid jobs first run a static PBE seed on the same structure, k points,
potentials and spin settings. The hybrid step reads its checked WAVECAR with
`ISTART=1` and `ICHARG=0`. A failed seed or an incompatible restart stops the job.
Seed evidence is retained; WAVECAR stays on the cluster. Previously prepared
jobs keep their original inputs. [VASP hybrid guide](https://vasp.at/wiki/LHFCALC).
Inspect final moments: the PBE seed can change the magnetic state, and MAGMOM
does not reset moments when restarting. [VASP MAGMOM](https://vasp.at/wiki/MAGMOM).
Hybrid runs split the k points into groups that fit the job's tasks (`KPAR`). Without it,
VASP 5.4.4 gave silicon 64 bands on 64 ranks and its HSE06 SCF took 22 minutes instead of 34 seconds.
Hybrid bands use adaptively compressed exchange ([LFOCKACE](https://vasp.at/wiki/LFOCKACE)) where
VASP 6 provides it; VASP 5 ignores the tag and applies the full exchange operator, which is slower
(a silicon HSE06 path took 18 minutes on 64 ranks of VASP 5.4.4).
They combine a weighted mesh and a zero-weight path and are
self-consistent and do not use `ICHARG=11`. A minimum of ten electronic steps
is enforced. Bands use Davidson iteration and reject equivalent k points whose
energies disagree by more than 0.05 eV. This is a consistency check; convergence
along the full path still needs checking.
[VASP hybrid bands guide](https://vasp.at/wiki/Band-structure_calculation_using_hybrid_functionals).

## Magnetic comparisons

Compare NM, FM and AFM with SCF on the same structure and cell. AFM alternates
initial moments; a shared supercell is made if needed. The default seed is
3 μB per magnetic site and can be changed in a plan.

Only successful, compatible runs enter the energy-per-atom ranking. Seeds can
converge to the same state. Inspect final moments; this is not a ground-state
search over all magnetic orders. PAW-sphere moments are not the full-cell moment.

## Plans

Plans can chain relaxation, SCF, bands and DOS with separate method settings
for each stage. PBE spectra reuse a matching SCF charge; hybrid spectra solve
self-consistently. A changed method starts a matching calculation.

The NM/FM/AFM preset comparison uses SCF. Explicit batches can instead use
supplied magnetic patterns, U/J values or strain points with the same stage
sequence. Keep the cell fixed during strain relaxation. Compare energies only
within a matching method and U/J choice. [Workflows](workflows.md).

The structure editor supports substitutions, vacancies and slabs. VASP workflows
do not yet include defect formation energies, phonons, NEB or MD.
