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

## Molecular dynamics

See [Molecular dynamics](molecular-dynamics.md) for supported programs, trajectory downloads and restart limits.

The `md` stage runs Born–Oppenheimer dynamics at a fixed cell, with the keys of the other programs: `md_ensemble` (`nvt` or `nve`), `md_temperature_k` (300), `md_steps` (1000), `md_timestep_fs` (0.5) and `md_thermostat_fs` (100). It may repeat. What follows it continues from its last frame: a relaxation to quench, or SCF, bands and DOS of that frame. Fixed atoms, vacuum directions and magnetic seeds are kept. Hybrids and charged cells are refused.

**Thermostat.** VASP 6.4.3 and newer have CSVR (`MDALGO = 5`, `CSVR_PERIOD` = `md_thermostat_fs`; 6.5.1's source uses it as a time constant in fs). The version is read from the environment's module or path. Older or unnamed versions use Langevin (`MDALGO = 3`) with `LANGEVIN_GAMMA` = 1/(2 `md_thermostat_fs`), because kinetic energy relaxes at twice the friction: the key then means the same under both. The friction slows atoms, so for a diffusion coefficient run NVT, then NVE. Nosé–Hoover (`MDALGO = 2`) is not used: on 8 Si atoms its temperature spread was 105 K against 93 K for a canonical ensemble, while Langevin gave 86 K against 87 K.

NVE is `MDALGO = 0`, `SMASS = -3`. It starts from velocities at `md_temperature_k`, so a crystal at rest settles near half of it; run NVT first. Two neighbouring settings looked fine and were wrong on VASP 5.4.4: `MDALGO = 2` with `SMASS = -3` is a thermostat of negative mass (8 Si atoms froze to 0 K within 200 steps), and `MDALGO = 5` is accepted and runs NVE. `MDALGO = 1` prints a kinetic energy half a step late, so its E wobbled by 82 meV at 2 fs.

**Inputs.** Forces along a trajectory need less than a final energy, so `md` starts from `PREC = Normal`, `ALGO = Fast`, `NELMIN = 4`, the POTCAR's ENMAX and a k-point spacing of 0.5 Å⁻¹, unless `encut`, `mesh` or `kspacing` are set. On a thermal frame of 8 Si atoms this stayed within 1.5 % of the RMS force of a 6×6×6 mesh at 520 eV, in a quarter of the time. `KPAR` follows the k-points (2.2 times faster on 32 ranks), and `LREAL = Auto` above 30 atoms (1.3 times faster for 64 Si atoms, with the same energy conservation).

**Checks.** Results come from `OSZICAR` and `XDATCAR`. `vasprun.xml` stays on the cluster, and an `OUTCAR` over 4 MB comes back as its two ends, with both whole files compressed there. A step whose SCF used all of `NELM` fails the stage: VASP moves on, and pymatgen's check reads only the last step. So does a run that did not reach its last step: one died with `ERROR in subspace rotation PSSYEVX` and exit code 0. `XDATCAR` keeps about 2000 frames (`NBLOCK`), and VASP rounds `NSW` down to a multiple of it. Step 1 is the starting frame, so N steps span N−1 time steps.

VASP has no time limit of its own. The job writes a `STOPCAR` with `LSTOP` shortly before its limit, VASP ends cleanly after the step under way, and the next stage continues from the last frame. A single step longer than the margin (5 minutes, or a tenth of the limit) still loses to the scheduler.

**What was measured** on Arrhenius, with 8 Si atoms at 5.468 Å, a 3×3×3 mesh, 32 ranks of VASP 5.4.4 and about 0.33 s per step. NVE from 600 K, 400 steps, mean temperature about 300 K:

| Time step | Spread of E | Drift |
|---|---|---|
| 0.5 fs | 0.035 meV/atom | 0.08 meV/atom/ps |
| 1 fs | 0.084 meV/atom | 0.07 meV/atom/ps |
| 2 fs | 0.38 meV/atom | −0.13 meV/atom/ps |
| 3 fs | 1.0 meV/atom | −0.37 meV/atom/ps |

NVT at 300 K with Langevin gave 314 K (second half of 2 ps), and 306 K over 6 ps. CSVR on VASP 6.5.1 gave 329 K over 3 ps, with fewer independent samples. The first g(r) peak was at 2.375 Å in 0.05 Å bins, against 2.368 Å in the crystal. The mean potential energy above the crystal was 0.31 eV, against 0.32 eV from equipartition at that temperature. The kinetic energy from `XDATCAR` positions agreed with `EK` within 0.3 %. In a 12-atom Al(111) slab with its lower layer fixed, the fixed atoms did not move under any thermostat, and the temperature counted the free atoms only.

These are checks of the workflow on small cells. They say nothing about sampling or size convergence of your system.

## Plans

Plans can chain relaxation, SCF, bands and DOS with separate method settings
for each stage. PBE spectra reuse a matching SCF charge; hybrid spectra solve
self-consistently. A changed method starts a matching calculation.

The NM/FM/AFM preset comparison uses SCF. Explicit batches can instead use
supplied magnetic patterns, U/J values or strain points with the same stage
sequence. Keep the cell fixed during strain relaxation. Compare energies only
within a matching method and U/J choice. [Workflows](workflows.md).

The structure editor supports substitutions, vacancies and slabs. VASP workflows
do not yet include defect formation energies, phonons or NEB.
