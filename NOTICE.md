# Notices

DFT Agent 0.5.0 is distributed under the [DFT Agent Free Use License](LICENSE), copyright 2026 Qirui Cui. The application source tree is not included. Runtime helpers and generated job scripts remain available where execution requires them. Earlier MIT-licensed releases retain their original terms; see LICENSE-MIT-PREVIOUS.txt. Third-party components retain the licenses listed in THIRD-PARTY-NOTICES.

VASP and POTCAR are not included. Use your own licensed installation and follow its terms.

RSPt is supplied separately by the user. DFT Agent does not include or download its source, binaries or manual.

UppASD is a separate GPL-3.0 program. DFT Agent does not bundle its source or binaries.
When you install it from the app, your cluster downloads the pinned UppASD 6.0.2 source from [GitHub](https://github.com/UppASD/UppASD) and builds it in your own folder there; if the cluster cannot reach GitHub, the app downloads the same checked archive and uploads it. You can use an existing installation instead. Cite the code when using its results.

Dependencies have their own licenses and citation requirements.

The local structure editor bundles [WEAS](https://github.com/superstar54/weas) 0.2.10
and its JavaScript dependencies. Their licenses and sources are listed in
`licenses/structure-editor/THIRD_PARTY_NOTICES.txt`.

The MOF building-block example includes coordinates from [PORMAKE](https://github.com/Sangwon91/PORMAKE)
under its MIT license, which is in `licenses/pormake/LICENSE.txt`.

Charged-defect corrections use [pymatgen-analysis-defects](https://github.com/materialsproject/pymatgen-analysis-defects).
Cite [Freysoldt, Neugebauer and Van de Walle (2009)](https://doi.org/10.1103/PhysRevLett.102.016402) when using this correction method.

Structures found by formula come from [JARVIS-DFT](https://jarvis.nist.gov/) and the [Materials Project](https://materialsproject.org/) (CC BY 4.0). Each saved structure records its database and identifier; cite the database you use. Only the formula is sent when searching.

Common molecules and nanoparticle shapes use [ASE](https://wiki.fysik.dtu.dk/ase/) (LGPL-2.1-or-later), including its G2 set of molecular geometries. Molecules found by name come from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) as computed conformers; only the name is sent, and each saved molecule records its compound number.
Special quasirandom structures follow [Zunger et al. (1990)](https://doi.org/10.1103/PhysRevLett.65.353); cite it when using them.

ABINIT, CP2K and Elk are not included. When you install them from the app, your cluster downloads their conda-forge packages, which keep their own licences (ABINIT: GPL-3.0-only; Elk: GPL-3.0-or-later; CP2K: GPL-2.0-or-later). Cite the codes you use.

SIESTA is installed separately from conda-forge and retains its own GPL licence.
Its versioned reference manual is downloaded from the [SIESTA project](https://gitlab.com/siesta-project/siesta), not bundled with the Agent.

LAMMPS is not included and stays a separate program. When you install it from the app, your cluster downloads its conda-forge package (GPL-2.0-only). Cite [Thompson et al. (2022)](https://doi.org/10.1016/j.cpc.2021.108171) and the potential each run names.
The potential files have no blanket licence. They are downloaded at installation from the [LAMMPS](https://github.com/lammps/lammps) stable_29Aug2024 potentials folder or from the [NIST Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/), which asks users to verify and cite each one; `lammps_potentials.json` records each file's source, checksum and citation.

The library catalogs record names and checksums only. The files are downloaded from their sources at installation:
[PseudoDojo](https://www.pseudo-dojo.org/) pseudopotentials (CC BY 4.0; cite [van Setten et al. (2018)](https://doi.org/10.1016/j.cpc.2018.01.012) and [Hamann (2013)](https://doi.org/10.1103/PhysRevB.88.085117)),
CP2K basis sets and potentials from the [CP2K](https://github.com/cp2k/cp2k) 2026.2 data folder (GPL-2.0-or-later), and Elk species from the [Elk](https://elk.sourceforge.io/) 10.2.4 release (GPL-3.0-or-later).

VASPKIT is distributed separately under its [usage agreement](https://vaspkit.com/installation.html#usage-agreement).
Cite [Wang et al. (2021)](https://doi.org/10.1016/j.cpc.2021.108033) when using its results in publications.

Remove credentials and private cluster details before sharing results.
