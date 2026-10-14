# Quantum ESPRESSO

DFT Agent can prepare and run Quantum ESPRESSO on a Slurm cluster. You need an SSH account and compute allocation. A VASP license is not required.

## Connect

Choose **Quantum ESPRESSO** at the top of the app, then open **Cluster setup**. Enter your SSH address, remote run folder, partition and account. Add the environment commands provided by your cluster. VASP and QE settings are saved separately; switching engines leaves existing runs unchanged.

If QE is already installed, enter its `bin` folder and launch command, such as `srun pw.x`. The check covers PW, DOS, projections, charge density, phonons and NEB executables, including missing shared libraries.

On Paracloud, the tested site environment is:

```bash
module purge
module load qe/7.4.1-oneAPI.2022.1
```

The site binaries are in `/public1/soft/qe/7.4.1/bin`. Use your own cluster account and folders.

Set **Launch command** to `mpirun -np 8 pw.x` and **MPI tasks** to `8`.
If you change the task count, change both values together. This launcher was
tested on Paracloud; `srun pw.x` failed there because of the site's PMI2 setup.

## Install

The managed installer builds QE 7.4.1 in your cluster storage. It downloads the official source and pinned dependencies, checks their hashes, then submits a Slurm build job. Compute nodes do not need Internet access. Existing installations are kept. Select **Include EPW** to also build EPW 5.9 and Wannier90.

In **Cluster setup**, save your connection and environment settings. Open **Install Quantum ESPRESSO**, choose **Build CPUs** and **Include EPW** if needed, review the allocation and click **Install QE**. Use **Check installation** to follow the same build. When it finishes, the app saves the executable folder. Run **Check cluster** before preparing a calculation.

Load your cluster's MPI Fortran compiler, C compiler and math libraries first. For the tested Paracloud environment:

```bash
module purge
module load mpi/oneAPI/2022.1 gcc/9.3.0-kd
```

With the settings saved in `cluster.json`:

```bash
dft-agent qe inspect --config cluster.json
dft-agent qe install-plan --config cluster.json --build-cpus 8
dft-agent qe install --config cluster.json
dft-agent qe install-status --config cluster.json
```

Review `qe-install-plan.json` before installation. Keep `qe-install-receipt.json`; it identifies the build and prevents duplicate submissions. A successful status returns `bin_dir`. Save that path in the QE settings and keep the same runtime modules loaded.

For EPW, add `--include-epw` to `install-plan` before running `install`. The core and EPW builds have both passed installation checks on Paracloud. A calculation with the installed binaries is recorded in [QE acceptance](../validation/qe-0.4.3/README.md).

The QE integration uses CPU and MPI. QE GPU execution is not supported yet.

## Pseudopotentials

Set a dedicated remote UPF folder, then install the elements needed by your structure:

In the app, upload the structure in **New calculation**, open **Pseudopotentials**, select a **Library** and click **Install public UPFs**. The app saves the mapping automatically. For the CLI:

```bash
dft-agent qe pseudos --config cluster.json --elements Si O
```

The default is **SSSP 1.3.0 PBE efficiency**. For a norm-conserving library, select **PseudoDojo (NC)** in the app or use:

```bash
dft-agent qe pseudos --config cluster.json --elements Al Mg B --library pseudodojo
```

PseudoDojo 0.4 PBE standard covers 72 elements. The agent checks each selected UPF's hash, element and type, saves its source and license, and adds the filenames and cutoff hints to your settings. Existing files are kept. Choosing another library updates the selected elements' settings; it does not change completed runs.

Both libraries are scalar relativistic. SOC needs fully relativistic UPFs; select a compatible library explicitly. Custom UPFs must match the elements and functional. Keep their source and recommended cutoffs with your settings.

Cutoff hints are starting values. PseudoDojo's Hartree hints are converted to Ry automatically. Converge the quantities needed for your study, particularly phonons, energy differences and metallic Fermi surfaces.

## Run

Use the same plan, review, prepare, submit and monitor sequence as other calculations. QE keeps its own inputs and scratch files. Stages reuse QE checkpoints rather than VASP files.

In **Model**, select a provider and enter your API key and model name. See [Model setup](models.md) and [Token use](token-use.md). In **New calculation**, upload a structure, describe your goal and choose **Plan**. You can revise the structure and calculation steps in the same task. Missing scientific settings trigger a question.

After a stage finishes, open **Continue this task**, enter **What next?** and click **Plan next step**. Review the response, then use **Prepare next step** to create a new run. Submit that run after checking its inputs. The model receives verified results and their settings; planning alone submits nothing.

The current integration targets relaxation, SCF, bands, DOS, projected DOS, charge density, magnetism, SOC, DFT+U, hybrid functionals, phonons and NEB. Acceptance is tracked per workflow; installing the full QE suite does not mean every QE package is available through the agent.

The QE integration accepts PBE, LDA, PBEsol, HSE06 and PBE0. It checks method and UPF compatibility before submission. Hybrid forces require norm-conserving UPFs.

For NEB, provide endpoints with the same cell, elements and atom order. For phonon dispersions, use a consistent SCF calculation and regular q-point grid before real-space interpolation.

In **Runs**, open the calculation and use **Raw data and files** for individual tables and outputs. **Prepare download → Download results (.zip)** collects the inputs, retained outputs, tables, plots and plan. The CLI equivalent is `dft-agent bundle RUN`. Read convergence checks before interpreting a result; Slurm's `COMPLETED` alone does not establish numerical convergence.

Large checkpoints remain on the cluster. List and retrieve selected files with `dft-agent files RUN STAGE` and `dft-agent files RUN STAGE --fetch FILE`. Collected files are checked against the accepted checkpoint hashes.

Electron–phonon calculations require explicit electronic and phonon grids and a Coulomb pseudopotential, μ*. See [Superconductivity](qe-superconductivity.md) for the methods and acceptance scope. Unsupported methods are reported explicitly.

## Coverage

The model plans with implemented tools and verified results. It cannot turn an unavailable solver into a working calculation by generating a plan. Track broader coverage against QE's [package documentation](https://www.quantum-espresso.org/documentation/) and [executable list](https://www.quantum-espresso.org/documentation/input-data-description/).

| Calculation | Current integration |
| --- | --- |
| Relaxation, SCF, bands, DOS, projections, charge density | Implemented; acceptance recorded per workflow |
| Magnetism, SOC, DFT+U, hybrid SCF | Implemented with method and UPF checks |
| Harmonic phonons and NEB | Implemented; includes full dispersion and endpoint checks |
| Electron–phonon coupling, EPW and superconductivity | See [methods and acceptance scope](qe-superconductivity.md) |
| Molecular dynamics, linear-response U, dielectric and optical response | Not integrated |
| TDDFPT, magnons, EELS, X-ray spectra and ballistic conductance | Not integrated |
| NMR/EPR, GW/BSE, Koopmans methods and thermal transport | Not integrated; some routes require external packages |

Each added route needs native input validation, dependencies, raw-output parsing, result checks and a real reference calculation. Package installation alone does not complete that work. The current acceptance record is in [QE acceptance](../validation/qe-0.4.3/README.md).

## Sources

- [QE 7.4.1 source](https://gitlab.com/QEF/q-e/-/releases/qe-7.4.1)
- [Build guide](https://www.quantum-espresso.org/Doc/user_guide/node11.html)
- [PW input reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html)
- [Phonon input reference](https://www.quantum-espresso.org/Doc/INPUT_PH.html)
- [NEB input reference](https://www.quantum-espresso.org/Doc/INPUT_NEB.html)
- [SSSP archive and licenses](https://archive.materialscloud.org/records/rcyfm-68h65)
- [PseudoDojo library and cutoff hints](https://www.pseudo-dojo.org/faq.html)
- [PseudoDojo license](https://github.com/abinit/pseudo_dojo/blob/4048962957711c04281ffe6c2bd2c96f0f7110aa/README.md#license)
