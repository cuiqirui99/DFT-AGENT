# Superconductivity

This is the implementation and acceptance scope for DFT Agent. The workflows below are not yet release-accepted. Each needs a saved test report and downloadable evidence before being marked available.

QE supplies the normal-state electronic structure and lattice response. A superconducting result also requires a pairing model. The agent must name that model, its inputs and its limits.

## Workflows

| Route | Calculation | Results | Acceptance |
| --- | --- | --- | --- |
| Conventional, isotropic | QE → DFPT/`lambda.x` or Wannier interpolation → EPW | α²F, λ, ωlog, Allen–Dynes estimate, Eliashberg gaps and a linearized Tc bracket | Native Al tests and independent kernel checks pass; application acceptance is in progress |
| Conventional, anisotropic | QE → Wannier90 → EPW | Momentum-dependent gaps at supplied temperatures | Native MgB₂ reference passes; stricter solver and fresh application acceptance are pending |
| Unconventional, model based | Supplied one-band `hr` → explicit Hubbard model → RPA pairing | Singlet/triplet eigenvalues, gap functions and susceptibility | Numerical benchmarks and an LLM → Slurm model run pass; no automatic material-model construction |

The anisotropic phonon route remains a conventional pairing calculation. It does not establish a spin-fluctuation mechanism or arbitrary unconventional pairing. EPW's [theory description](https://docs.epw-code.org/Theory.html#superconductivity) defines its electron–phonon and Coulomb treatment.

## First conventional route

Pin inputs to [QE 7.4.1 example03](https://gitlab.com/QEF/q-e/-/tree/qe-7.4.1/PHonon/examples/example03), then adapt the structure and pseudopotentials explicitly. Its sequence is:

1. Dense, unshifted electronic grid with `la2F=.true.`; retain the generated `a2Fsave` data.
2. SCF on the grid used for the phonon calculation.
3. `ph.x` with `electron_phonon='interpolated'`, a regular q grid and saved `fildvscf` data.
4. `q2r.x` and `matdyn.x` with `la2F=.true.` for interpolation and spectra.
5. `lambda.x` with explicit irreducible q points, weights, matching coupling files and a user-selected Coulomb pseudopotential μ*.

Electronic grids must contain the required k and k+q points. Preserve the structure, pseudopotentials, q ordering and checkpoint lineage across stages. Distinguish electronic smearing in Ry from the spectral broadening in THz. These requirements follow the [PH input reference](https://www.quantum-espresso.org/Doc/INPUT_PH.html#electron_phonon) and the pinned example.

The pinned [lambda source](https://gitlab.com/QEF/q-e/-/blob/qe-7.4.1/PHonon/PH/lambda.f90) uses

\[
T_c=\frac{\omega_{\log}}{1.2}
\exp\left[-\frac{1.04(1+\lambda)}{\lambda-\mu^*(1+0.62\lambda)}\right],
\]

with ωlog in kelvin. Label this as the simplified Allen–Dynes estimate; it omits strong-coupling correction factors. Keep μ* visible and provide a sensitivity range. Do not report a value when the denominator is nonpositive.

Before analysis, check q coordinates and weights against every coupling file. The upstream program does not enforce all these matches. Material imaginary modes, missing q points, unconverged DFPT, nonfinite integrals or a mismatch between integrated and summed λ must prevent a validated Tc result. A coarse functional test is not a converged material prediction.

## EPW route

QE 7.4.1 contains EPW 5.9. Select **Include EPW** when preparing an installation. The managed build includes the pinned Wannier90 dependency and builds core QE before EPW. A real Slurm installation passed checks for all 13 programs; it uses a separate prefix and preserves previous installations.

For isotropic calculations, use **Eliashberg gap** or **Eliashberg Tc** after an accepted `electron_phonon` or `wannier_electron_phonon` spectrum. Choose temperatures, μ*, the frequency cutoff and `broadening_index` explicitly. The index selects an electronic-broadening column for `lambda.x`, or a phonon-broadening column for Wannier/EPW; that source's electronic broadening stays fixed.

Open the completed run's **Continue this task**, describe the solver and these settings, then **Plan next step → Prepare next step**. Review and submit the new run. Scalar EPW reuses the frozen spectrum without rerunning SCF or phonons. Anisotropic EPW also requires the original remote matrix checkpoint.

Eight native Al cases passed on the test cluster, covering two broadenings, two cutoffs and both scalar solvers. Independent calculations checked linearized eigenvalues and nonlinear residuals. These are functional checks, not a converged prediction for Al.

Negative interpolated spectral weights remain a failed result. An explicit `eph_integration="q_sum"` integrates the original DFPT modes by a separate method. It does not erase or validate a failed Fourier interpolation. Preserve both outcomes and examine grid and broadening dependence.

Retain the dynamical matrices, potential derivatives, phonon recovery data and electronic wavefunctions. This Wannier route requires scalar, nonmagnetic NC pseudopotentials. **PseudoDojo (NC)** supplies public PBE files; use compatible NC files for LDA. Projections, energy windows, band counts and coarse/fine grids must be explicit. Validate Wannier bands in the Fermi window before using a material result.

Use [EPW's MgB₂ reference](https://docs.epw-code.org/tutorials/archived/MgB2.html) to check the gap distribution and its temperature dependence, including μ*, fine grids and Matsubara cutoff. Newer documentation may describe features absent from EPW 5.9. Experimental agreement is a separate check from reproducing the chosen method.

### MgB2 example

This coarse example follows the [QE 7.4.1 reference](https://github.com/QEF/q-e/tree/500de340b820e1cb8c05f2d8bb8fced102f377c1/test-suite/epw_super), with a stricter gap-solver tolerance. It tests the workflow at 15 K; it is not a converged material prediction or a Tc calculation. Local preparation passes; acceptance of the fresh application run is pending.

1. Install QE with **Include EPW**.
2. In your cluster terminal, enter the **UPF folder on cluster** saved in Cluster setup. Download these exact LDA NC files and verify their hashes:

```bash
qe_ref_url='https://raw.githubusercontent.com/QEF/q-e/500de340b820e1cb8c05f2d8bb8fced102f377c1/EPW/examples/mgb2/pp'
curl -fL "$qe_ref_url/Mg.pz-n-vbc.UPF" -o Mg.pz-n-vbc.UPF
curl -fL "$qe_ref_url/B.pz-vbc.UPF" -o B.pz-vbc.UPF
sha256sum -c <<'EOF'
5357982ec1f8139c2cde411073103f9b33d5482a45e2401bcf1b8aea228db076  Mg.pz-n-vbc.UPF
eb7aa71704ded328f1c78b2e16b630d14fc97d688b0b23b51a56d541c97e1f31  B.pz-vbc.UPF
EOF
```

3. Paste [qe-mgb2-pseudos.json](../examples/qe-mgb2-pseudos.json) into **Cluster setup → Advanced → UPF mapping**, then save. This LDA recipe requires the files above; the public library buttons install PBE files.
4. In **New calculation**, choose **Quantum ESPRESSO**, **Example → MgB2** and **Manual**. Select **Wannier electron–phonon coupling** and **Anisotropic gap**, in that order. Paste the complete [qe-mgb2-stages.json](../examples/qe-mgb2-stages.json) into **Advanced parameters → Stage recipes (JSON)**. Click **Prepare inputs**, review the five stages, then submit. The structure is also available as [MgB2.vasp](../examples/MgB2.vasp).

## Unconventional route

The implemented route solves a **single-band paramagnetic Hubbard RPA model**. It accepts a real, spin-independent Wannier90 `hr` Hamiltonian and compares even-frequency singlet and triplet pairing. Multiple orbitals, SOC and magnetic reference states are not supported.

Choose **Quantum ESPRESSO → Pairing model**, upload the Hamiltonian and enter an existing cluster Python with NumPy and SciPy. Click **Check Python**, then describe the model in **Goal**. This Python runs on the cluster, even when using the Windows desktop app. The agent asks for missing lattice vectors, orbital convention, U, chemical potential, temperature, k grid, Matsubara cutoff, channels and double-counting treatment. Review the plan, prepare inputs and submit. U and chemical potential must be supplied; the agent does not infer them from composition or a DOS peak.

From the CLI, use `plan model_hr.dat "your model settings" --engine qe --pairing --origin model`, then `prepare model_hr.dat RUN --plan plan.json --config cluster.json --cluster-python /path/to/python` and `watch RUN`. Prefix each command with `dft-agent`. Use `--origin wannier_unvalidated` for a Wannier calculation. `files RUN 01_pairing` lists verified local outputs; `bundle RUN` exports them.

The default NumPy/SciPy solver follows the [TPRF linearized Eliashberg equations](https://triqs.github.io/tprf/3.3.x/theory/eliashberg.html). It was checked against pinned TRIQS/TPRF 3.3.1 and an independent dense-matrix calculation. The singlet vertex uses the theory's negative density-ladder term explicitly, rather than the differently signed Python helper in that release. The optional TPRF backend is a reference; users do not need it for the app workflow.

Paracloud job **7494956** passed numerical comparisons, grid/frequency refinement and magnetic-instability rejection. Job **7495022** completed a real LLM plan through preparation, Slurm execution and verified collection. The fixed square-lattice model at 40×40×1 k points and 150 positive Matsubara frequencies gave leading singlet and triplet eigenvalues of 0.05518270834 and 0.00748273678. These values are below one at the tested temperature. They validate the model workflow, not superconductivity in a material.

Downloads retain the reviewed plan, inputs, gap and susceptibility arrays, native output and execution receipts. The app verifies file hashes, the runtime, scheduler completion, filling, RPA stability, eigenpair residuals and exchange symmetry before exposing numerical results to the model. Interrupted monitoring reconnects to the original job.

A supplied `hr` file is labeled either a model or an unvalidated Wannier calculation. A material prediction still requires a checked QE/Wannier interpolation, justified interactions and double counting, and numerical convergence. A transition estimate also requires a stable eigenvalue crossing of one over temperature. A magnetic instability is rejected; one temperature point is not Tc. For two-dimensional systems, a pairing scale alone does not establish a phase-ordering temperature.

## Saved evidence

Use **Prepare download → Download results (.zip)** to retain inputs, raw outputs, parsed tables and execution receipts. Large QE restart files stay on the cluster; retrieve selected files with `dft-agent files RUN STAGE --fetch FILE` before removing that remote folder. Restart only from a compatible accepted checkpoint. A failed result remains part of the record; it is not silently replaced by another method.
