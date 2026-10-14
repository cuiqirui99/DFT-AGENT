# Changelog

## 0.4.3 — 2026-10-14

- Add Quantum ESPRESSO installation, public UPFs and a VASP/QE switch.
- Plan and run QE calculations through conversation, including phonons, NEB and electron–phonon coupling.
- Add Wannier/EPW spectra, isotropic Eliashberg gaps and Tc brackets, and anisotropic gaps.
- Add a bounded single-band Hubbard RPA pairing workflow with explicit model inputs.
- Continue from accepted stages, preserve frozen inputs during repair, and verify collected files before analysis.
- Include raw outputs, numerical tables and execution receipts in downloads. See [QE validation](validation/qe-0.4.3/README.md) for tested scope and limits.

## 0.4.2 — 2026-10-10

- Add desktop packaging for Apple Silicon Macs (`.dmg`) and Windows x64 (`.exe`), including Python and model SDKs. Installation and platform acceptance are tracked separately from publication.
- Prepare and review inputs without a cluster; attach cluster settings at submission.
- Load four completed sample runs, including SrTiO3 with a 353-point band path.
- Show periodic crystals with separate lattice axes and element labels.
- Use denser band paths and export plots as PNG, PDF and SVG.
- Correct VASP XML ionic energies in relaxation plots and CSV files; refresh older plots from saved outputs.
- Add CSV downloads beside plots and show raw files directly.
- Return projected DOS, forces, stress, convergence tables and selected cluster files.
- Detect partitions, accounts, VASP modules and POTCAR folders from the cluster, with presets for known clusters.
- Remember API keys in the system keychain on request; `dft-agent key` manages them.
- Open the browser automatically and hide the Streamlit toolbar.
- Move settings to `~/.dft-agent/` and runs to `~/dft-agent-runs/`; earlier locations keep working.
- Show runs as a table, cluster checks as a checklist with fixes, and clearer errors.
- Add screenshots, a Chinese README, an install guide and a Windows guide.
- Notice new releases in the app.
- Interactive 3D structure viewer with a static fallback.
- Resume monitoring at startup and notify when a run finishes.
- Switch the interface between English and Chinese.
- Connect from native Windows with bundled SSH/SFTP support; WSL remains optional.
- Preserve cluster configurations and unfinished input fields when switching configuration files or interface language.
- Preserve launch commands and environment prerequisites during cluster detection, verify keychain deletion, and send Discord-compatible notifications.
- Document desktop and source installation.

## 0.4.1 — 2026-10-09

- Generate k meshes from cell size and detect vacuum directions.
- Choose smearing, symmetry and PAW defaults with explicit overrides.
- Start hybrid calculations from a checked PBE WAVECAR.
- Support multiple nodes and extra Slurm resource options.
- Use structured Claude output and specific provider errors.
- Preserve prepared calculations when upgrading.

## 0.4.0 — 2026-10-09

- Plan structure edits and calculations in one conversation.
- Choose methods by stage and compare magnetic orders, U values and strains.
- Extract sampled band gaps and continue from accepted structures.
- Connect scientific checks to planning and preserve stage settings during repair.

## 0.3.3 — 2026-10-09

- Add Claude, Qwen, Grok, GLM and DeepSeek providers.
- Keep each provider's credentials separate.
- Include setup instructions and token counts for the new providers.

## 0.3.2 — 2026-10-09

- Add model setup, API key and token billing instructions.
- Walk through a first plan without a cluster.
- Show provider help in Model and disable unused Codex fields.

## 0.3.1 — 2026-10-09

- Include checked guidance, calculation cases and repair examples.
- Search memory and import selected local records.
- Keep rejected and unverified lessons out of recommendations.

## 0.3.0 — 2026-10-09

- Restore scientific guidance and relevant run history in planning.
- Review failed runs and prepare bounded repairs.
- Edit structures through conversation and convert CIF/POSCAR.
- Add nine structures from earlier cluster calculations.
- Keep model context compact and record reported token use.

## 0.2.1 — 2026-10-07

- Explain results and answer follow-up questions.
- Save plan revisions and retain them after a failed model request.

## 0.2.0 — 2026-10-07

- Plan and revise calculations with a language model.
- Add magnetism, SOC, DFT+U, HSE06 and PBE0.
- Chain calculations and compare magnetic seeds.

## 0.1.3 — 2026-10-07

- Renamed to DFT Agent.

## 0.1.2 — 2026-10-07

- Shorter text.

## 0.1.1 — 2026-10-07

- English UI and docs.

## 0.1.0 — 2026-10-07

- Initial release: relax, SCF, bands and DOS.
- Job recovery, structure preview and result downloads.

[Validation](docs/validation.md)
