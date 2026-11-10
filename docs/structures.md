# Structures

Connect a model and describe the material in the conversation. **Build** uses that same
conversation; there is no construction form to fill in. No cluster connection is needed.
You can upload a CIF or POSCAR, ask for a published structure, or start from a description.

If the material, composition or connections are unclear, the Agent asks a short question.
If you do not know where to start, ask for an example. It explains its starting choices,
saves the structure and shows it below the reply, with downloads.

Continue with instructions such as “make it wider”, “change the top layer” or “undo that
change”. Select an earlier structure to work on that version. Each edit saves a new copy.

To move a structure between projects, download **Structure (ZIP)** and import the
`.dft.zip` file through **Upload**. It keeps periodic directions, fixed atoms and
magnetic moments and recorded source references together with the coordinates; bare
CIF/POSCAR files cannot keep all of these settings. References describe the original
inputs, not validation of the edited structure.

## Find a published structure

Ask for a formula, phase or common name, such as MoS2 or graphene. **Search** is also available.
Crystals come from [JARVIS-DFT](https://jarvis.nist.gov/) and the
[Materials Project](https://materialsproject.org/). You can narrow a search by dimension or space group.
The Agent follows result pages within a fixed limit and reports unavailable sources or incomplete searches.
An energy ranking within those results is not proof of the most stable phase.

For named frameworks, ask to search the [Crystallography Open Database](https://www.crystallography.net/cod/).
The Agent checks the paper and composition before importing the CIF. A name in a paper title may
describe a precursor or another material. Incomplete or partially occupied geometries are rejected
on import, rather than silently filled in. Search terms are sent to the selected public database;
saved structures retain their source identifiers. Imported geometries still need review before use.

Named COFs can also come from [CURATED COFs](https://github.com/danieleongari/CURATED-COFs).
The saved source identifies the catalogue revision and paper. These are curated models;
a 2D framework may still contain stacked periodic layers. The Agent preserves the supplied
stacking and reports what the source does not establish about composition or protonation.

Molecules can be imported by name from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
Their original SDF, bond orders, formal charges, isotope fields and source stereo markers stay with
the structure. Geometry edits preserve this information as source history; they do not establish
the chemistry of the edited structure. If protonation or connection sites are unclear, the Agent asks.

## Build a crystal

Name the crystal you want. The Agent can look up a published structure or use a
specified prototype. A formula alone does not determine its phase. The following
parameters describe what the builder needs internally; you do not have to fill in a form.

| Crystal type | Elements, in order | Required parameters |
|---|---|---|
| fcc, bcc, diamond | One element | `a` |
| rocksalt, zincblende | A, B | `a` |
| wurtzite | A, B | `a`, `c`, and internal coordinate `u` with `0 < u < 0.5` |
| Cubic oxide perovskite | A, B, O | `a` |
| Honeycomb sheet (graphene, h-BN) | A, B | `a`, vacuum |
| 2H or 1T sheet (MoS2, VSe2) | M, X | `a`, thickness between the X planes, vacuum |
| Custom cell | One element per site | Three lattice vectors and each site's fractional coordinates |
| Space group | One element per unique site | Space group number (1–230), conventional lattice vectors and unique fractional coordinates |

The saved crystal appears in the conversation for inspection and further changes.
Space groups use pymatgen's default conventional setting, with hexagonal axes
for rhombohedral groups. Incompatible cells and equivalent duplicate sites are rejected.

An optional source note records where the geometry came from. Built structures
are starting models; confirm the phase and relax them before use.

## Examples

Choose **Example**, or use the files in [`examples`](https://github.com/cuiqirui99/DFT-AGENT/tree/v0.4.7/examples). When both
formats are available, the app uses POSCAR to preserve the lattice frame and site order.
**MOF-5** and **CTF-1 building blocks** save their fragments and a starting framework
assembled from them. Describe changes to the recipe in the conversation.

The simple examples are Al, C, Cu, Ge, MgO, NaCl, Si, SiC and
[SrTiO3](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/SrTiO3.cif). SrTiO3 is a five-atom cubic perovskite.
Its completed sample includes relaxation, 353 band k-points, DOS, plots and raw data.
The [MgB2 example](qe-superconductivity.md#mgb2-example) includes a complete QE/EPW recipe.
The following starting geometries come from earlier user calculations:

| Structure | Atoms | Files |
|---|---:|---|
| VSe2 | 3 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/VSe2.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/VSe2.cif) |
| In2Se3 | 5 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/In2Se3.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/In2Se3.cif) |
| V2Te2O | 5 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/V2Te2O.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/V2Te2O.cif) |
| CrSBr | 6 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrSBr.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrSBr.cif) |
| CrOCl | 6 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrOCl.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrOCl.cif) |
| CrCl3 | 16 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrCl3.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrCl3.cif) |
| CrPS4 | 24 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrPS4.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrPS4.cif) |
| GaFe2O4 | 42 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/GaFe2O4.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/GaFe2O4.cif) |
| CrOCl–MoS2 | 114 | [POSCAR](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrOCl-MoS2.vasp) · [CIF](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/CrOCl-MoS2.cif) |

These files preserve the source geometry without refinement. They do not establish
relaxation, convergence or a magnetic ground state. Choose the method, moments,
k-points and resources for the calculation you want to perform.
[Source and file checksums](https://github.com/cuiqirui99/DFT-AGENT/blob/v0.4.7/examples/sources.json) are included in the package.

The editor shows the input cell as a ball-and-stick model. Displayed bonds use
covalent distances; they do not establish chemical bond order.

## Edit a structure

Describe the change in the conversation. The selected structure is the target for
“this”; naming another structure overrides that target. If the target is ambiguous,
the Agent should ask before editing. Earlier versions stay available.

For a direct atom edit, click atoms in the 3D editor, then use **Replace**, **Remove** or **Move**.
Use **Add** to insert an atom at fractional coordinates. Preview the change,
then **Save structure** or **Discard preview**. The editor runs locally; its
scripts are bundled with the app and need no CDN connection.

Open **Cell and atom tools** for supercells, strain, lattice, vacuum, surfaces and constraints.
Under **Constraints**, choose atoms and check the cell directions that may move. Checking all three releases them.
Choose an operation, select **Preview changes**, then **Save structure**.
You can make supercells; replace, remove, move or add
atoms; apply strain; change lattice lengths and angles; adjust a slab's vacuum;
or cut a surface. Site indices start at zero in the current structure.

Added atoms need an element and fractional position. Strain and lattice changes
keep fractional positions fixed. The panel uses percent strain, including tensor
shear components; lattice angles are in degrees. Vacuum sets the final empty gap
normal to the other two lattice vectors and preserves the slab geometry. It
requires an existing, distinct vacuum gap; use **Surface** to cut a bulk crystal.
Surfaces need a Miller index, thickness and vacuum, with a choice of termination
when more than one is available.

You can also describe edits in the conversation: “Make a 2 × 2 × 1 supercell,”
“Replace site 0 with Ge,” or “Apply 1% strain along x.” A revision uses the
original source and replaces the earlier operation list, so changing the repeats
does not multiply an already enlarged cell. The Agent asks for missing choices.
Each saved edit keeps its source and edit record; existing calculations retain
their original inputs. Preview and export work locally without a model.

## Choose a starting structure

A formula does not uniquely identify a crystal. Choose a prototype with explicit
parameters, enter a complete cell, or supply an experimental CIF, a specific
database structure or a checked POSCAR. Keep its source and phase.
The model chooses edits; pymatgen builds the coordinates. Checks cover composition,
site order, cell validity, overlaps and file conversion. They do not establish
stability or a magnetic ground state.

Built, imported and edited structures can all feed the same [point-defect workflow](defects.md).
Choose the host, supercell, defect sites and charge states; this is not limited to
GaN. For a surface, specify orientation, thickness, vacuum and termination.
Review the geometry before relaxing it. The editor supports ordered periodic
structures with up to 4,096 atoms; large model requests may need smaller steps.
It does not generate arbitrary crystals from formulas.

## Build an alloy

Say “Build a 50/50 Fe–Co alloy with about 128 atoms.” The Agent finds or builds a
parent crystal, reports the starting lattice and saves the alloy. If the elements
or proportions are missing, it asks. “I do not know; give me a simple example” lets
it choose an example for you. Continue with “change Co to 25%” to save another version.

## Build complex structures

Describe the material and its important features in the conversation. Existing project
structures can supply layers, crystals or molecular fragments. The Agent checks the
available sites and matching cells, asks for choices that change what material is built,
and reports the saved geometry. Routine starting values are identified as choices.

| Structure | Choices |
|---|---|
| Interface or layered stack | Ordered sources, surface cuts for bulk crystals, gaps, vacuum, layer rotation and allowed strain |
| Twisted bilayer | Two sheets and a target angle; the Agent checks periodic cells and reports the actual angle and size |
| Nanoribbon | Sheet, edge type or direction, width, and optional passivation with H or groups such as OH or COOH |
| Magnetic junction | Left electrode, barrier, right electrode, layer rotation and P/AP starting moments |
| MOF or COF | Full node/linker geometries, connection sites, bonds, cell and periodic topology |
| Surface with molecules | A slab or sheet, or a bulk crystal with a Miller index. A molecule from the list or found by name, a site from the list, height, orientation, repeats for the coverage |
| Functional groups | One of 20 groups or a mixture, on exposed atoms such as edges, on a face, or on chosen atoms; coverage and spacing |
| Alloy / disordered crystal | A partially occupied CIF, or an ordered crystal with a composition for each kind of site, vacancies included. Supercell from a list; special quasirandom or random arrangement |
| Nanotube | A sheet and a chirality from a list with its diameter and size; further walls |
| Nanoparticle | A bulk crystal with a radius, facets, surface energies (Wulff) or a magic-number shape |
| Core-shell particle | One lattice with other elements in the shell, or two crystals; core radius and shell thickness |
| Grain boundary | A bulk crystal, a rotation axis and a boundary from the list of Σ values; periodic with two boundaries, or one with free surfaces |
| Dislocation | A bulk crystal and a slip system from a list, or your own line and Burgers vector; a cylinder with a fixed shell, or a periodic dipole |
| Domain wall | A bulk crystal; magnetic (Ising, Bloch or Néel) or polar, wall plane and length |
| Liquid / solution / amorphous | Molecules or atoms at random in a box, around a solute or above a surface, from counts, density or box size |

These are geometry, and what they report is measured from the result. No energy, charge state, relaxation or
chemistry is predicted, so check them before calculating:

- A molecule starts at about a covalent bond length, or at van der Waals contact when flat. Real bonds are often shorter: C-Pt starts at 2.09 Å against about 1.85 Å measured. Slab polarity and reconstructions are not handled.
- An SQS matches pair statistics only, for a finite cell. The search is not a proof of the best arrangement, and a few-site cell can stay far from random. Mixed valence is not worked out from a composition.
- Rolling a thick sheet strains it strongly: MoS2 (6,6) has about 30% plane strain, and the preview says so. Multi-wall tubes are not optimised against each other.
- A particle's polar facets are reported, not corrected. Without strain matching, a shell on a very different lattice makes a poor interface, and the preview says how many atoms stayed bonded.
- A grain boundary shifts only in the plane, with no normal translation or choice of termination. Removing atoms in a compound can change its composition; there is a warning, and charge is not analysed.
- Domain walls are 180° walls in the bulk lattice, without ferroelastic strain. The wall width is a starting value.
- A dislocation uses the isotropic elastic field, which means little within a bond of the core. A dipole with an edge component adds atoms between the cores. Kinks, jogs and separated partials are not built.

**A liquid, solution or amorphous solid needs molecular dynamics.** A packing is only a random start: 64 Si atoms
at 2.29 g/cm³ do not become amorphous silicon just by being packed. The saved record includes a suggested dynamics recipe;
building or previewing the structure does not run it. Temperatures and durations need review, and the supplied density
is an input, not an equilibrium result. For a slab with solvent, vacuum is a minimum gap that preserves the filled region.

A bulk crystal needs a surface cut before it can be a layer. A cut with several terminations
lists them with their top and bottom elements. If no cell matches at the chosen angles, the
message lists rotations that do match, such as 45° for MgO on Fe.

A twist allows 0.1% strain by default, since identical layers match without any; other
assemblies allow 3%. A preview warns above 1%, which is enough to change electronic structure.

Without an edge offset, a ribbon is cut midway through the widest gaps between atomic rows, where
the fewest bonds cross: zigzag and armchair edges on a honeycomb sheet. An offset selects another
termination, and is refused if it would cut through a row of atoms. Edge atoms are the ones that lost a bond.
Caps too close together are refused: a Klein edge carbon loses two bonds per period, and its
hydrogens would sit 0.57 Å from their periodic images.

**Building block** accepts XYZ fragments. The MOF-5 and CTF-1 examples supply
editable fragments and construction recipes. MOF-5 uses the two-node primitive cell of Fm-3m,
with neighbouring nodes rotated by 90°; one node per cell cannot keep the linkers planar.
A linker rotation of 0° is coplanar with the first node at that connection, however the
fragment was drawn. The pcu preset lists a node's connection atoms as +a, −a, +b, −b, +c, −c, sql
as +a, −a, +b, −b, and hcb lists three 120° apart. Custom framework graphs can use
multiple block types; the tool checks connectivity and periodic bond lengths. The framework
type is a label: a surprising one, such as a COF with metal atoms, gives a warning.
These are starting geometries, not relaxed or experimentally verified structures.

Construction supports cells up to 4,096 atoms, within the chosen search limits. Above index 128
the search considers only reference cells whose vectors all layers nearly share, so it is bounded and
not exhaustive. Matching reports the actual angle and strain; it rejects requests that have no
acceptable cell within those limits. A small twist angle can require many more
atoms than a routine DFT calculation can afford: graphene at 1.89° has 3,676 atoms.
Building the same assembly twice returns the structure already saved.

## Build and change structures in the conversation

With a model connected, describe what you want and keep going, as you would with a colleague:

> Put CO on Pt(111) at the fcc site.
>
> Turn the CO by 30° around the surface normal and lift it by 0.2 Å.
>
> Put a second water molecule next to the first.
>
> Go back to the previous one and make the slab one layer thicker.
>
> How far is the oxygen from the surface?

Each change saves a new structure and leaves earlier versions intact. The Agent asks
about missing material identities, compositions or connections. It can choose routine
starting dimensions and report them, or choose an example when you ask for one.
Nothing is submitted by building. Say what to change next, or select an earlier version.
“This” means the structure currently selected for the conversation.

You can combine edits: “Increase the layer gap, remove one atom from the top layer,
then add CO above that site.” The Agent keeps intermediate structures and checks the
final composition, cell and requested distances. Saved versions can be compared by
region; reordered atoms need an explicit correspondence. These checks verify the
geometry, not chemical stability.

When no named builder fits, such as a helical wire, a composition gradient or a corrugated sheet, the Agent writes a
short Python geometry script, runs it on your computer, reads any error and fixes it. The script is saved with the
structure and shown under its 3D view. It runs in a separate process with no access to your files, keys, network or
other programs, and what it returns goes through the same checks as any other structure. A script can also just measure
something and save nothing. As with every structure here, the result is a starting geometry to review and relax.

You can describe the geometry and the calculation in one message. For example, after selecting a graphene sheet:

> Use this sheet for both layers. Make a twisted bilayer at 7.340993°, with a
> 3.35 Å gap and 20 Å vacuum. Keep strain below 0.1% and angle error below 0.001°.
> Fix the bottom layer, then relax the atoms with QE and calculate the bands.

The Agent builds a new structure and uses its saved path in the calculation workflow. You can revise the
geometry or steps before preparing inputs. Review the inputs and submit through the calculation card.

You need not know every value. Name a material or molecule, and the Agent can look it up.
It asks when a missing choice changes the requested material or geometry, rather than
silently choosing an unrelated structure. Suggested dimensions are starting values,
not measured or calculated results.

VASP and QE preserve construction periodicity, fixed atoms and supported starting
magnetic moments, also through batch comparisons and strain series. Vacuum directions receive one
mesh point; the calculations still use periodic supercells. Coulomb isolation needs a method
appropriate to the system. Cell relaxation keeps vacuum fixed: QE relaxes in-plane only
(`cell_dofree` `2Dxy`, or `x` for a ribbon), and VASP, whose ISIF has no such option, refuses it.
A strain series gives zero strain along vacuum. NM/FM/AFM trial seeds keep periodicity and fixed
atoms, and do not replace seeds the structure already records. Other adapters refuse vacuum directions,
fixed atoms and magnetic seeds before any plan is written; a fully periodic structure without them, such as a
packing, runs in any of them. Junction construction does not add transmission or TMR calculations.

[Materials Project structure selection](https://docs.materialsproject.org/frequently-asked-questions)
and [pymatgen transformations](https://pymatgen.org/pymatgen.transformations.html)
describe the source and geometry tools behind this approach.
