# Point defects

Point-defect workflows were checked on two x86-64 Slurm clusters with a 32-atom GaN cell. A nitrogen vacancy gave +1/0 transition levels of 1.39 eV with QE 7.4.1, 1.38 eV with VASP 6.2.1 and 1.39 eV with QE 7.6, the last driven entirely from a conversation with a real model, including the Freysoldt correction and the plots. A magnesium substitution gave a 0/−1 level of 0.35 eV. In a 16-atom cell, a manganese substitution with DFT+U on manganese went from a conversation to a 0/−1 level of 1.41 eV, and a VASP vacancy went through a failed run, its bounded repair and the analysis. These checks cover the workflow, parsers and bookkeeping, not supercell or k-point convergence.

In a project, import a structure (or the bundled `GaN.cif` example) and describe the defect, supercell and charge states. For example:

> Use QE to study a nitrogen vacancy in this GaN structure. Make a 2×2×2 supercell, show me the nitrogen sites, and compare charges 0, +1, +2 and +3. Relax the atoms while keeping the cell fixed.

The Agent creates a matching bulk cell and a calculation for each charge. Choose the site from the expanded structure before creating the defect. Positive charge removes electrons. Vacancies, substitutions and explicitly positioned interstitials are supported. VASP defect workflows currently use PBE; QE also supports its validated LDA and hybrid SCF paths, subject to the selected pseudopotentials and task.

Review the inputs and resources, then submit. Existing calculations stay unchanged when you request a new attempt. A failed step can get a bounded repair, which raises an iteration or time limit and keeps its method and charge.

All steps of a comparison must use one method, so preparing a step refuses settings that the analysis could not compare with a step already prepared or run. The message names them, and the plan stays open for revision. With QE, cutoffs you leave open are the largest recommended over the host and the dopant, so install the dopant's pseudopotential before preparing any step. DFT+U on a dopant applies only to the defect cells: all its charge states must share it, and formation energies need the dopant's chemical potential from a calculation with the same U.

## Results

Ask the Agent to analyze the completed bulk and defect runs. It saves a report, energy and eigenvalue tables, and figures in the project. Data are available as CSV/JSON; figures as PNG/PDF/SVG. The native calculation outputs remain available with each run.

You can continue the conversation:

> Compare the charge states and show which inputs are still missing.

> Calculate the potential alignment of every charge, then plot the levels from −1 to 3 eV relative to the host valence-band maximum.

> Use orange and title the figure “GaN nitrogen vacancy”.

These plot changes run locally and do not submit another calculation. Figures appear in the conversation with download buttons. Other analyses can use the saved tables directly in a reviewed Python script with explicit inputs and outputs through the project's cluster workflow.

## Interpreting the numbers

An eigenvalue, a localized defect state and a charge-transition level are different quantities. An in-gap eigenvalue is a candidate; localization needs further evidence. Thermodynamic transitions also require ionic relaxation of the charge states. Frozen geometries give energy crossings at those geometries.

Partial occupations are marked. Reduce smearing and check spin and defect-band filling before using transition energies quantitatively.

Charged-cell corrections use matching electrostatic potentials and a sourced dielectric constant or tensor. QE exports the ionic plus Hartree potential; VASP uses `LVHAR`. The Agent applies the Freysoldt correction and exports its potential-alignment plots. Check the sampled plateaus and supercell-size dependence before using the result quantitatively. This treatment assumes a three-dimensional insulating host and localized defect charge.

Absolute formation energies additionally require elemental chemical potentials. These cancel when comparing charge states of the same defect. Missing references are listed in the report; the model does not supply invented values. A source hash identifies the supplied data, not its scientific validity.

PBE gaps, finite supercells, magnetic states and reservoir choices can affect the result. A completed workflow does not establish convergence or experimental agreement.

Method: [Freysoldt, Neugebauer and Van de Walle (2009)](https://doi.org/10.1103/PhysRevLett.102.016402), implemented through [pymatgen-analysis-defects](https://materialsproject.github.io/pymatgen-analysis-defects/content/freysoldt-correction/).
