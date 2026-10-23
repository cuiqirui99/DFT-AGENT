# Input defaults

Defaults provide a starting calculation. They do not establish convergence.
Explicit mesh, smearing and POTCAR settings take precedence.

## K points

Without a `mesh`, DFT Agent uses `kspacing: 0.25` Å⁻¹ and writes an explicit
Gamma-centered mesh. Each division is `ceil(|G_i| / kspacing)`, where `G_i`
includes **2π**. Larger cells therefore need fewer divisions. This follows the
[VASP KSPACING convention](https://vasp.at/wiki/KSPACING); the generated KPOINTS
file records the actual mesh.

A geometric vacuum check uses the largest periodic gap between atomic planes.
An empty gap of at least 10 Å, spanning at least half the perpendicular cell
height, sets that automatic mesh direction to one. This is a geometric estimate;
check it for porous or unusual structures. It never changes an explicit mesh.

```json
{"kspacing": 0.2}
```

Or choose the divisions directly:

```json
{"mesh": [8, 8, 1]}
```

Band paths use `line_density: 60` (about 0.017 Å⁻¹ between points). The total
depends on the path length. Increase this value for a finer path. Every segment
gets at least 10 points, so short segments and large cells still give smooth
bands.

The path follows the lattice found with a 0.02 Å symmetry tolerance, so a relaxed
cell keeps its labels (silicon: Γ–X–U|K–Γ–L–W–X) despite residual distortions of a
few mÅ. Distortions of about 0.04 Å or more, such as a 1 % tetragonal strain, keep
their lower-symmetry path. This applies to VASP, QE, ABINIT, CP2K and Elk.

Across a vacuum gap only the in-plane segments are kept, so a monolayer gets
Γ–M–K–Γ. Atoms with different starting moments count as different atoms, so an
antiferromagnet follows its magnetic cell. Spin–orbit coupling alone keeps time
reversal, so its path is not repeated with primed points (X′, K′); only a magnetic
moment acting on spinor states, with spin–orbit coupling or noncollinear spin,
adds the reversed half.

## Band and DOS plots

Band plots show −8 to 8 eV around their reference, like the VASP plots. The upper
end stops 0.1 eV below the highest computed band, so no band is missing inside the
window and the highest one, which follows whichever uncomputed band crosses it, is not drawn. Spin-up bands are solid and spin-down bands dashed; Elk's band file mixes
both spins, so its bands share one style. Metals are drawn against the Fermi level.
For QE, ABINIT, CP2K and Elk, an insulator's bands start at its valence band
maximum, because its Fermi level can sit anywhere in the gap. A jump in the path
reads U|K, and seekpath's numbered points read H₂. DOS plots use the same window,
with spin down below zero. The CSV files keep every band and energy.

## Smearing

`electronic_type` describes what you already know. The program does not infer
metallicity from the element names. Omitted `ismear` and `sigma` use these rules:

| Input | Choice |
| --- | --- |
| `auto` or unspecified | Gaussian: ISMEAR=0, SIGMA=0.05 eV |
| `metal`, relaxation or SCF | Methfessel–Paxton: ISMEAR=1, SIGMA=0.2 eV |
| `insulator`, DOS with a 3D mesh | Tetrahedra: ISMEAR=-5 |
| Other cases, including band paths | Gaussian: ISMEAR=0, SIGMA=0.05 eV |

Automatic tetrahedra require at least two divisions in each direction and no
inferred vacuum axis. An explicit ISMEAR=-5 is rejected for a band path or a
one-point axis. Explicit smearing values are otherwise retained. Check SIGMA
and the entropy contribution for metallic forces.
[VASP smearing guidance](https://vasp.at/wiki/Smearing_technique).

## Symmetry and potentials

Regular PBE calculations use ISYM=2; regular hybrid calculations use ISYM=3.
Collinear MAGMOM values retain the requested magnetic symmetry. SOC and
noncollinear calculations use ISYM=-1. Ordinary band paths use ISYM=0; hybrid
paths use ISYM=-1. [ISYM](https://vasp.at/wiki/ISYM),
[MAGMOM](https://vasp.at/wiki/MAGMOM).

Default PAW labels follow a frozen pymatgen MPRelaxSet recommendation map
(2026.9.24), including Ti_pv, Fe_pv, Ni_pv and Ge_d. W uses W_sv, and f-shell
elements keep active f electrons unless you explicitly choose a fixed-valence
variant. These labels must exist in your licensed collection. Set
`potcar_symbols` in the cluster configuration to override them, for example
`{"Fe": "Fe", "Ni": "Ni"}`.
[pymatgen input sets](https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.sets.MPRelaxSet),
[VASP potential selection](https://vasp.at/wiki/Choosing_pseudopotentials).

Prepared metadata records the effective mesh, smearing, vacuum estimate and
potential selection. Compare energies only with matching calculation settings
and potentials; a completed electronic iteration is not a convergence study.
