# CLI examples

Use the [bundled CLI](install.md#command-line) from the macOS app. Download the
[example files](https://github.com/cuiqirui99/DFT-AGENT/tree/v0.4.7/examples) into an
`examples` folder, or replace their paths below with your own CIF or POSCAR.
More inputs are listed in [Structures](structures.md).
Create `cluster.json` with `dft-agent init --config cluster.json` and review
its VASP, POTCAR and Slurm settings. The PBE defaults are starting points.

## Prepare locally

Use a new directory for each run. `prepare` submits nothing.

**Plan with a model**

```bash
dft-agent task examples/Si.cif "Relax, then calculate PBE bands and DOS" \
  --provider codex --output proposal.json
dft-agent prepare examples/Si.cif runs/si-plan --config cluster.json \
  --plan proposal.json
```

For an API, use `--provider responses --model MODEL` and set `OPENAI_API_KEY`.
For a compatible service, use `--provider chat_completions --base-url URL`.
Inspect the proposal before preparing it. A plan is bound to its source file.

Revise it before preparation:

```bash
dft-agent task examples/Si.cif "Use a 6x6x6 mesh" \
  --previous proposal.json --provider codex --output revised.json
```

**Edit a structure**

```bash
dft-agent structure-plan examples/Si.cif "Make a 2x2x1 supercell" \
  --provider codex --output edit.json
dft-agent structure-plan examples/Si.cif "Use 3x2x1 instead" \
  --previous edit.json --provider codex --output revised-edit.json
dft-agent prepare-structure examples/Si.cif edited --plan revised-edit.json
```

Review the plan before applying it. Use `edited/POSCAR` for the calculation.
For conversion without a model: `dft-agent convert examples/Si.cif converted`.

**Si: HSE06 bands**

```bash
dft-agent prepare examples/Si.cif runs/si-hse --config cluster.json \
  --task bands --parameters '{"functional": "HSE06", "mesh": [2, 2, 2]}'
```

**Compare magnetic seeds**

```bash
dft-agent prepare Fe.cif runs/fe --config cluster.json \
  --task scf --magnetic-states NM FM AFM
```

Supply your own Fe structure. See [Methods](methods.md) for moments, SOC and U/J.

**Si: relax atoms and cell**

```bash
dft-agent prepare examples/Si.cif runs/si-relax --config cluster.json \
  --task relax --parameters '{"cell_relax": true}'
```

Omit `--parameters` to keep the cell fixed.

**Al: SCF**

```bash
dft-agent prepare examples/Al.cif runs/al-scf --config cluster.json --task scf
```

**MgO: SCF, then DOS**

```bash
dft-agent prepare examples/MgO.cif runs/mgo-dos --config cluster.json --task dos
```

SCF and DOS use the supplied structure without relaxing it.

## Submit

Review `config.json` and each stage's `inputs/INCAR`, `KPOINTS` and `POSCAR`.
To change settings, prepare a new run. Choose the matching command:

```bash
dft-agent watch runs/si-relax
dft-agent watch runs/al-scf
dft-agent watch runs/mgo-dos
```

**`watch` submits the calculation.** Add `--password` for a private prompt.

## Status, recovery and download

```bash
dft-agent status runs/si-relax
dft-agent explain runs/si-relax --provider codex
dft-agent resume runs/si-relax
dft-agent bundle runs/si-relax
```

`status` reads saved progress. `resume` reconnects to the same run; inspect
`needs_attention` errors first. Substitute the Al or MgO directory as needed.
`explain` reads saved results without submitting a job. Use `--question` to ask
about a specific result.

| Run | Results |
|---|---|
| Si | `runs/si-relax/01_relax/outputs/`: `final_structure.cif`, `relax_energy.csv`, `relax_energy.png`, `result.json` |
| Al | `runs/al-scf/01_scf/outputs/`: `result.json`, `final_structure.cif` |
| MgO | `runs/mgo-dos/02_dos/outputs/`: `dos.csv`, `dos.png`, `result.json`; SCF results in `01_scf/outputs/` |

`bundle` writes `results.zip` with inputs, retained outputs, plots and `run.json`.
Check the saved status before using results. POTCAR is excluded; CHGCAR and
WAVECAR stay remote.
