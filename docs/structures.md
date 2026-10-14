# Structures

Choose **Example** in the app, or use the files in [`examples`](../examples).
The package includes all 19 choices. When both formats are available, the app
uses POSCAR to preserve the lattice frame and site order.

The simple examples are Al, C, Cu, Ge, MgO, NaCl, Si, SiC and
[SrTiO3](../examples/SrTiO3.cif). SrTiO3 is a five-atom cubic perovskite.
Its completed sample includes relaxation, 353 band k-points, DOS, plots and raw data.
The [MgB2 example](qe-superconductivity.md#mgb2-example) includes a complete QE/EPW recipe.
The following starting geometries come from earlier user calculations:

| Structure | Atoms | Source | Files |
|---|---:|---|---|
| VSe2 | 3 | `cqr/irvsp/POSCAR` | [POSCAR](../examples/VSe2.vasp) · [CIF](../examples/VSe2.cif) |
| In2Se3 | 5 | `cqr/quantumdimer/POSCAR` | [POSCAR](../examples/In2Se3.vasp) · [CIF](../examples/In2Se3.cif) |
| V2Te2O | 5 | `cqr/vteowannier/POSCAR` | [POSCAR](../examples/V2Te2O.vasp) · [CIF](../examples/V2Te2O.cif) |
| CrSBr | 6 | `cqr/Crsbrfinal/POSCAR` | [POSCAR](../examples/CrSBr.vasp) · [CIF](../examples/CrSBr.cif) |
| CrOCl | 6 | `cqr/crocl/POSCAR` | [POSCAR](../examples/CrOCl.vasp) · [CIF](../examples/CrOCl.cif) |
| CrCl3 | 16 | `cqr/bilayercrcl3/POSCAR` | [POSCAR](../examples/CrCl3.vasp) · [CIF](../examples/CrCl3.cif) |
| CrPS4 | 24 | `cqr2/crps4/POSCAR` | [POSCAR](../examples/CrPS4.vasp) · [CIF](../examples/CrPS4.cif) |
| GaFe2O4 | 42 | `cqr/fegao/POSCAR` | [POSCAR](../examples/GaFe2O4.vasp) · [CIF](../examples/GaFe2O4.cif) |
| CrOCl–MoS2 | 114 | `cqr/crocl_mos2/POSCAR` | [POSCAR](../examples/CrOCl-MoS2.vasp) · [CIF](../examples/CrOCl-MoS2.cif) |

These files preserve the source geometry without refinement. They do not establish
relaxation, convergence or a magnetic ground state. Choose the method, moments,
k-points and resources for the calculation you want to perform.
[Source and file checksums](../examples/sources.json) are included in the package.

The **Crystal** view repeats small cells for display. **Input cell** shows the
original cell; neither view changes the calculation inputs.

## Edit or convert

Open **Edit structure** and describe an edit, such as “Make a 2 × 2 × 1
supercell,” “Replace site 0 with Ge,” or “Remove site 3.” Use the displayed
zero-based site indices. Slabs also need a Miller index, thickness and vacuum;
the planner asks about missing settings and alternative terminations.

Review **Preview structure**, then click **Use structure**. Revisions are applied
once to the original input. The calculation uses the edited POSCAR; its original
input, exports and edit record are saved under `structure_edit/` in the run.

**Convert** exports CIF, POSCAR or both without a model. Editing and conversion
prepare geometry locally; use a VASP relaxation when needed.

## Choose a starting structure

A formula does not uniquely identify a crystal. Supply an experimental CIF, a
specific database structure, or a checked POSCAR, and keep its source and phase.
The model chooses edits; pymatgen builds the coordinates. Checks cover composition,
site order, cell validity, overlaps and file conversion. They do not establish
stability or a magnetic ground state.

For a new material, choose its parent structure and specify the substitution or
defect sites. For a surface, specify orientation, thickness, vacuum and termination.
Review the geometry before relaxing it. The editor supports ordered periodic
structures with up to 4,096 atoms; large model requests may need smaller steps.
It does not generate arbitrary crystals from formulas.

[Materials Project structure selection](https://docs.materialsproject.org/frequently-asked-questions)
and [pymatgen transformations](https://pymatgen.org/pymatgen.transformations.html)
describe the source and geometry tools behind this approach.
