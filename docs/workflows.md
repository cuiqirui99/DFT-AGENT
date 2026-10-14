# Workflows

These examples use VASP. See [Quantum ESPRESSO](qe.md) for QE tasks and continuations.

Upload a structure or choose an example. In **Agent** mode, write the full task
in **Goal**, then click **Plan**. Structure edits and calculations share one
conversation. A revision replaces the whole plan and starts from the original
upload, so supercells and substitutions are not applied twice.

## Edits and stages

For example:

> Make a 2×2×1 supercell, replace site 0 with Ge, relax with PBE, then calculate HSE06 bands.

Review the edited cell, site order and each stage's method. Supply missing site
choices, magnetic moments, spin axes or U/J values when asked. Each stage keeps
its own settings. A method change starts a matching calculation; it does not
reuse an incompatible fixed charge density.

Click **Prepare inputs**, inspect the files, then confirm **Submit calculation**.
Planning and preparation submit no jobs.

## Comparisons

Specify the points to compare: magnetic moment patterns, U/J values or axial
strains. The plan lists every variant before preparation. It supports up to 32
variants with the same stage sequence. Strain relaxation must keep the cell fixed.

In **Runs**, review the table and click **Submit batch** after checking the
variants. Open individual runs to see their inputs and results. **Download table**
saves the comparison. Only accepted stages supply results; incomplete runs are not ranked.
Energies at different U/J values are not a magnetic-state ranking.

## Continue a result

Open a finished run and expand **Continue**. Choose **Starting stage**, describe
the **Next calculation**, then click **Plan next calculation**. Existing settings
are inherited unless you request a change. Review the plan and click
**Prepare continuation**, then confirm submission of the new run.

The original run stays intact. The selected result is rechecked before reuse.
A displayed **Sampled band gap** describes the computed k points; it is not a
claim of full Brillouin-zone or numerical convergence.

## CLI

```bash
dft-agent task examples/Si.cif "Relax with PBE, then calculate HSE06 bands" \
  --provider codex --output task.json
dft-agent prepare examples/Si.cif runs/si --config cluster.json --plan task.json
dft-agent watch runs/si
```

Use `task --previous task.json` to revise a task. `watch` submits jobs; it also
handles prepared batches. `status`, `resume`, `cancel` and `bundle` accept either
a run or a batch directory. See `dft-agent continue --help` for stage continuation.
