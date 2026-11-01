# SIESTA

On 2026-10-31 the managed SIESTA 5.4.2
installation passed its silicon test on two x86-64 Slurm clusters. On one of them,
conversations with a real model then built silicon and bcc iron, prepared their
inputs and ran relaxation with a variable cell, SCF, bands and DOS, nonmagnetic
and collinear, up to an explanation of the results. The DOS was then run on
silicon, bcc iron, graphene and NaCl. These checks cover the workflow, inputs,
parsers and receipts, not basis-set or mesh convergence.

SIESTA uses numerical atomic orbitals and norm-conserving pseudopotentials.
The Agent prepares native FDF inputs and runs SIESTA through the same project,
conversation and cluster workflow as the other engines.

## Start

1. Connect your model and cluster.
2. In **Connections**, detect an existing SIESTA installation or review a managed installation plan.
3. Install the PBE pseudopotentials for the elements in your structure.
4. Build, import or upload a structure in your project, then describe the task.

The managed package supports Linux x86-64 clusters and runs on one CPU node.
An existing site installation can use the cluster's own MPI setup.

For example:

> Use SIESTA to relax this silicon structure, then calculate its bands and DOS.

Review the generated inputs and resources before submitting. You can revise the
plan in the conversation, follow the jobs, and start another calculation from an
accepted structure. Existing runs keep their original inputs.

## Calculations

The integrated tasks are relaxation, SCF, bands, DOS and fixed-cell NVT/NVE
[molecular dynamics](molecular-dynamics.md), using PBE with either
nonmagnetic or collinear spin. Starting moments can be set per element or per
site. The adapter rejects unsupported methods instead of silently substituting
them. SOC, noncollinear spin, DFT+U, hybrid functionals, TranSIESTA transport and
the paired charged-defect analysis are not part of this integration.

Band and DOS stages inherit the accepted density matrix as an initial guess
and reconverge on their requested sampling mesh. The orbital basis and real-space
grid require separate convergence checks: `MeshCutoff` is not a plane-wave
orbital cutoff. A completed job does not establish basis-set convergence.

The DOS integrates the eigenvalues over the full k mesh with linear tetrahedra.
Its resolution still depends on the mesh. Ask for a
broadening (`dos_broadening_ev`) to get Gaussian curves instead. A list of k points
that tetrahedra cannot integrate, such as a single point, falls back to a 0.1 eV
Gaussian, and the results say so.

## Results

The project retains the native inputs and outputs, energy and convergence
evidence, final structure, band/DOS data and plots. Ask the Agent to explain the
verified results or prepare further analysis. A plotted band gap describes the
sampled states; it is not an experimental-gap prediction.

The Agent can retrieve the versioned [SIESTA manual](https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html)
for scientific guidance. Official manual capabilities may extend beyond this
adapter. Pseudopotentials come from [PseudoDojo](https://www.pseudo-dojo.org/);
the installation catalog records their source and checksums.
