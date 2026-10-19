"""DFT Agent app."""

from __future__ import annotations

from datetime import datetime
from importlib.resources import files
from io import BytesIO
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from pymatgen.core import Structure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vasp_slurm_agent import __version__
from vasp_slurm_agent.cli import doctor, doctor_rows
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.i18n import LANGUAGE_NAMES, set_language, t
from vasp_slurm_agent.paths import default_config_path, default_runs_root, migrate_legacy_config
from vasp_slurm_agent.settings import load_settings, save_settings
from vasp_slurm_agent.workflow import (
    SLURM_TERMINAL,
    attach_config,
    bundle_run,
    cancel,
    has_config,
    prepare_run,
    prepare_plan,
    read_state,
    resume,
    start_worker,
)


TERMINAL = {"succeeded", "needs_attention", "failed", "cancelled"}
STATUS_LABELS = {
    "planned": "Awaiting review",
    "submitting": "Submitting",
    "queued": "Queued",
    "running": "Running",
    "collecting": "Collecting results",
    "succeeded": "Completed",
    "needs_attention": "Needs attention",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
ACTIVE = {"submitting", "queued", "running", "collecting"}
VIEWER_SCRIPT = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
VIEWER_FALLBACK = "https://3Dmol.org/build/3Dmol-min.js"


def _status(status):
    return t(STATUS_LABELS.get(status, status))


def _prefs():
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = load_settings()
    return st.session_state["prefs"]


def _with_password(function, *args):
    """Pass the session password to an action and its worker."""
    from vasp_slurm_agent.transport import password_context
    with password_context(st.session_state.get("ssh_password", "")):
        return function(*args)


def _load_config(path: Path) -> ClusterConfig | None:
    if not path.is_file():
        return None
    try:
        return ClusterConfig.load(path)
    except Exception as exc:
        st.error(t("Cannot read cluster settings: {error}").format(error=exc))
        return None


def _preset_panel(config):
    from vasp_slurm_agent.discovery import load_presets, preset_values

    presets = load_presets()
    if not presets:
        return
    with st.expander(t("Start from a preset"), expanded=config is None and "cluster_draft" not in st.session_state):
        st.caption(t("Presets are starting points for known clusters. Detect from cluster then fills in what it finds."))
        choice = st.selectbox(t("Preset"), presets, format_func=lambda item: item["name"], key="cluster_preset")
        for note in choice.get("notes", []):
            st.caption("• " + note)
        if choice.get("docs_url"):
            st.caption(f"[{t('Cluster documentation')}]({choice['docs_url']})")
        if st.button(t("Use preset"), key="use_preset"):
            draft = dict(st.session_state.get("cluster_draft") or {})
            draft.update(preset_values(choice))
            st.session_state["cluster_draft"] = draft
            st.session_state["cluster_draft_version"] = st.session_state.get("cluster_draft_version", 0) + 1
            st.rerun()


def _detect(host, user, port, control_path, timeout, setup, compute_backend="cpu"):
    from vasp_slurm_agent.discovery import DiscoveryError, probe_cluster, probe_config

    probe = probe_config(host, user, int(port), connect_timeout=int(timeout), ssh_control_path=control_path,
                         setup_commands=setup, compute_backend=compute_backend)
    try:
        with st.spinner(t("Reading partitions, accounts, VASP and POTCAR folders. Nothing is submitted.")):
            return _with_password(probe_cluster, probe)
    except DiscoveryError as exc:
        raise ValueError(str(exc)) from exc


def _detection_report(report):
    facts = report.get("facts") or {}
    found = []
    if facts.get("partitions"):
        found.append(t("{count} partitions").format(count=len(facts["partitions"])))
    if facts.get("accounts"):
        found.append(t("{count} accounts").format(count=len(facts["accounts"])))
    if facts.get("modules"):
        found.append(t("{count} VASP modules").format(count=len(facts["modules"])))
    if facts.get("potcar_roots"):
        found.append(t("{count} POTCAR folders").format(count=len(facts["potcar_roots"])))
    st.success(t("Detected: {items}. Review the filled fields, then save.").format(items=", ".join(found) if found else t("no Slurm or VASP details")))
    for note in report.get("notes", []):
        st.caption("• " + note)
    with st.expander(t("Detected details")):
        if facts.get("partitions"):
            st.dataframe([{t("Partition"): item["name"] + (" *" if item["default"] else ""), t("State"): item["state"],
                           t("Time limit"): item["timelimit"], t("Nodes"): item["nodes"],
                           t("GPU resources"): item.get("gres", "")} for item in facts["partitions"]],
                         hide_index=True)
        for key, label in (("accounts", "Accounts"), ("modules", "VASP modules"), ("potcar_roots", "POTCAR folders"),
                           ("scratch_dirs", "Writable folders")):
            if facts.get(key):
                st.write(f"**{t(label)}**: " + ", ".join(facts[key]))


def _config_editor(path: Path, config: ClusterConfig | None) -> None:
    engine = st.session_state.get("cluster_engine", config.engine if config else "vasp")
    if engine == "qe":
        from vasp_slurm_agent.qe_app import config_editor
        config_editor(path, config)
        return
    st.subheader(t("Cluster setup"))
    st.caption(t("Settings are saved on this computer. Fill in your SSH host and user, then let Detect from cluster find the rest."))
    editor_path = str(path.expanduser().resolve())
    if st.session_state.get("cluster_editor_path") != editor_path:
        for name in ("cluster_draft", "cluster_detection"):
            st.session_state.pop(name, None)
        st.session_state["cluster_draft_version"] = st.session_state.get("cluster_draft_version", 0) + 1
        st.session_state["cluster_editor_path"] = editor_path
    _preset_panel(config)
    draft = st.session_state.get("cluster_draft") or {}
    version = st.session_state.get("cluster_draft_version", 0)

    def value(name: str, default):
        if name in draft:
            return draft[name]
        return getattr(config, name, default)

    def key(name):
        return f"cfg_{name}_{version}"

    # Keep edits in session state before an outside control (such as language) reruns.
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            host = st.text_input(t("SSH host"), value=value("host", ""), key=key("host"))
            user = st.text_input(t("SSH user"), value=value("user", ""), key=key("user"))
            port = st.number_input(t("SSH port"), 1, 65535, int(value("port", 22)), key=key("port"))
            control_path = st.text_input(t("SSH control socket (optional)"), value=value("ssh_control_path", ""), key=key("socket"))
            remote_root = st.text_input(t("Remote run folder (absolute path)"), value=value("remote_root", ""), key=key("remote_root"))
            potcar_root = st.text_input(t("Remote POTCAR folder"), value=value("potcar_root", ""), key=key("potcar_root"))
            vasp_command = st.text_input(t("VASP command"), value=value("vasp_command", "srun vasp_std"), key=key("vasp_command"))
            ncl_command = st.text_input(t("SOC / noncollinear command"), value=value("vasp_ncl_command", ""),
                                        placeholder="srun vasp_ncl", key=key("ncl"))
            vaspkit_executable = st.text_input(t("VASPKIT executable"), value=value("vaspkit_executable", "vaspkit"),
                                              help=t("Optional post-processing tool on the cluster."), key=key("vaspkit"))
        with right:
            compute_backend = st.selectbox(t("Compute"), ("cpu", "gpu"),
                                           index=("cpu", "gpu").index(value("compute_backend", "cpu")),
                                           format_func=str.upper, key=key("compute_backend"))
            partition = st.text_input(t("Slurm partition"), value=value("partition", ""), key=key("partition"))
            account = st.text_input(t("Slurm account (optional)"), value=value("account", ""), key=key("account"))
            nodes = st.number_input(t("Nodes"), 1, 4096, int(value("nodes", 1)), key=key("nodes"))
            gpus_per_node = int(value("gpus_per_node", 1))
            cpus_per_task = int(value("cpus_per_task", 1))
            if compute_backend == "gpu":
                gpus_per_node = st.number_input(t("GPUs per node"), 1, 64, gpus_per_node, key=key("gpus_per_node"))
                cpus_per_task = st.number_input(t("CPUs per task"), 1, 4096, cpus_per_task, key=key("cpus_per_task"))
                tasks = int(nodes) * int(gpus_per_node)
                st.number_input(t("MPI tasks"), value=tasks, disabled=True,
                                help=t("One MPI task per GPU."), key=key(f"gpu_tasks_{tasks}"))
                st.caption(t("Use a GPU-enabled VASP build."))
            else:
                cpu_tasks = st.session_state.get(key("cpu_tasks_saved"),
                                                value("tasks", 8) if value("compute_backend", "cpu") == "cpu" else 8)
                tasks = st.number_input(t("MPI tasks"), 1, 4096, int(cpu_tasks),
                                        help=t("Total across all nodes."), key=key("tasks"))
                st.session_state[key("cpu_tasks_saved")] = int(tasks)
            walltime = st.text_input(t("Time limit per stage"), value=value("walltime", "00:30:00"), key=key("walltime"))
            timeout = st.number_input(t("SSH connection timeout (s)"), 1, 120, int(value("connect_timeout", 15)), key=key("timeout"))
            sbatch = st.text_area(t("Extra Slurm options"), value="\n".join(value("extra_sbatch", [])),
                                  placeholder="--mem=16G\n--qos=normal", help=t("One resource option per line."), key=key("sbatch"))
            setup = st.text_area(t("Environment setup commands (one per line)"), value="\n".join(value("setup_commands", [])),
                                 placeholder="module load your-vasp-module", key=key("setup"))
        symbols = st.text_area(t("POTCAR element mapping (JSON)"), value=json.dumps(value("potcar_symbols", {}), ensure_ascii=False),
                               help=t('Optional. Leave {} to use the recommended PAW set, or map elements to your POTCAR folder names, e.g. {"Ti": "Ti_pv"}.'),
                               key=key("symbols"))
        buttons = st.columns(2)
        save = buttons[0].button(t("Save cluster settings"), type="primary")
        detect = buttons[1].button(t("Detect from cluster"),
                                               help=t("Signs in over SSH and fills partitions, accounts, VASP and POTCAR folders. Nothing is submitted."))
    if detect:
        if not host.strip() or not user.strip():
            st.error(t("Enter the SSH host and user first."))
        else:
            try:
                report = _detect(host.strip(), user.strip(), port, control_path.strip(), timeout,
                                 [line for line in setup.splitlines() if line.strip()], compute_backend)
            except Exception as exc:
                st.error(t("Cannot detect cluster settings: {error}").format(error=exc))
            else:
                current = {"host": host.strip(), "user": user.strip(), "port": int(port), "ssh_control_path": control_path.strip(),
                           "remote_root": remote_root.strip(), "potcar_root": potcar_root.strip(), "vasp_command": vasp_command.strip(),
                           "vasp_ncl_command": ncl_command.strip(), "partition": partition.strip(), "account": account.strip(),
                           "vaspkit_executable": vaspkit_executable.strip(),
                           "tasks": int(tasks), "nodes": int(nodes), "walltime": walltime.strip(), "connect_timeout": int(timeout),
                           "compute_backend": compute_backend, "gpus_per_node": int(gpus_per_node),
                           "cpus_per_task": int(cpus_per_task),
                           "extra_sbatch": [line for line in sbatch.splitlines() if line.strip()],
                           "setup_commands": [line for line in setup.splitlines() if line.strip()]}
                try:
                    current["potcar_symbols"] = json.loads(symbols) if symbols.strip() else {}
                except ValueError:
                    current["potcar_symbols"] = {}
                from vasp_slurm_agent.discovery import merge_suggestions
                st.session_state["cluster_draft"] = merge_suggestions(current, report["suggested"])
                st.session_state["cluster_draft_version"] = version + 1
                st.session_state["cluster_detection"] = report
                st.rerun()
    if st.session_state.get("cluster_detection"):
        _detection_report(st.session_state["cluster_detection"])
    if save:
        try:
            mapping = json.loads(symbols) if symbols.strip() else {}
            if not isinstance(mapping, dict):
                raise ValueError(t("POTCAR mapping must be a JSON object."))
            candidate = ClusterConfig(
                host=host.strip(), user=user.strip(), remote_root=remote_root.strip(),
                vasp_command=vasp_command.strip(), potcar_root=potcar_root.strip(),
                partition=partition.strip(), port=int(port), tasks=int(tasks), walltime=walltime.strip(),
                account=account.strip(), setup_commands=[line for line in setup.splitlines() if line.strip()],
                connect_timeout=int(timeout), potcar_symbols=mapping,
                vasp_ncl_command=ncl_command.strip(),
                ssh_control_path=control_path.strip(),
                nodes=int(nodes), extra_sbatch=[line for line in sbatch.splitlines() if line.strip()],
                compute_backend=compute_backend, gpus_per_node=int(gpus_per_node), cpus_per_task=int(cpus_per_task),
                vaspkit_executable=vaspkit_executable.strip(),
            )
            candidate.save(path)
        except Exception as exc:
            st.error(t("Cannot save settings: {error}").format(error=exc))
        else:
            for name in ("cluster_draft", "cluster_detection"):
                st.session_state.pop(name, None)
            st.success(t("Saved to {path}").format(path=path))
            st.rerun()
    if st.button(t("Check cluster connection"), disabled=config is None):
        try:
            with st.spinner(t("Checking SSH, Slurm and VASP. No jobs submitted.")):
                report = _with_password(doctor, config)
            if report.get("ok"):
                st.success(t("Cluster checks passed."))
            else:
                st.warning(t("Some checks failed. See below."))
            st.dataframe([{t("Check"): row["check"], t("Result"): "✓" if row["ok"] else "✗",
                           t("Detail"): row["detail"], t("Suggested fix"): row["hint"]} for row in doctor_rows(report)],
                         hide_index=True)
            with st.expander(t("Raw report")):
                st.json(report)
        except Exception as exc:
            st.error(t("Cluster checks failed: {error}").format(error=exc))


def _structure_plot(structure: Structure) -> None:
    from vasp_slurm_agent.structure_view import display_repeat, structure_figure

    fig = structure_figure(structure)
    try:
        st.pyplot(fig)
        repeats = " × ".join(str(value) for value in display_repeat(structure))
        st.caption(f"{repeats} view · {len(structure)} atoms in input")
    finally:
        plt.close(fig)


def viewer_html(poscar: str, height: int = 430) -> str:
    from vasp_slurm_agent.structure_view import viewer_html as render_viewer

    return render_viewer(
        Structure.from_str(poscar, fmt="poscar"), VIEWER_SCRIPT, VIEWER_FALLBACK,
        t("The interactive viewer needs internet access to load 3Dmol.js. Turn it off in Preferences to use the static plot."),
        height=height,
    )


def _structure_viewer(structure: Structure, height=430) -> None:
    from pymatgen.io.vasp import Poscar
    import streamlit.components.v1 as components

    components.html(viewer_html(Poscar(structure).get_str(), height=height), height=height + 12)
    st.caption(t("Drag to rotate, scroll to zoom. Repetition changes the view only."))


def _structure_view(structure: Structure, height=430) -> None:
    if _prefs().get("interactive_viewer", True):
        try:
            _structure_viewer(structure, height=height)
            return
        except Exception:
            pass
    _structure_plot(structure)


def _preview(upload) -> Structure | None:
    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    try:
        with tempfile.TemporaryDirectory(prefix="vasp-agent-preview-") as directory:
            source = Path(directory) / f"structure{suffix}"
            source.write_bytes(upload.getvalue())
            structure = Structure.from_file(source)
        cols = st.columns(3)
        cols[0].metric(t("Formula"), structure.composition.reduced_formula)
        cols[1].metric(t("Atoms"), len(structure))
        cols[2].metric(t("Volume (Å³)"), f"{structure.volume:.3f}")
        st.caption("a, b, c (Å): " + ", ".join(f"{v:.4f}" for v in structure.lattice.abc))
        _structure_view(structure, height=330)
        with st.expander(t("Elements and fractional coordinates")):
            st.dataframe([
                {t("Element"): str(site.species_string), "x": site.a, "y": site.b, "z": site.c}
                for site in structure
            ], hide_index=True)
        return structure
    except Exception as exc:
        st.error(t("Cannot read structure: {error}").format(error=exc))
        return None


def _stage_results(run_dir: Path, stage: dict, scf_fermi_energy: float | None = None) -> None:
    if (stage.get("metadata") or {}).get("engine") == "qe":
        from vasp_slurm_agent.qe_app import stage_results
        stage_results(run_dir, stage)
        return
    result = stage.get("result")
    if not result:
        return
    label = f"{stage['label']} · " if stage.get("label") else ""
    st.subheader(f"{label}{stage['name']} · " + t("Results"))
    if not result.get("success"):
        st.warning(result.get("reason", t("Result checks failed.")))
        if result.get("recovery_hint"):
            st.info(result["recovery_hint"])
        return
    if stage["name"] in {"relax", "scf"}:
        metrics = st.columns(3)
        energy = result.get("final_energy_ev")
        force = result.get("final_max_force_ev_angstrom")
        if energy is not None:
            metrics[0].metric(t("Final total energy (eV)"), f"{energy:.6f}")
        if force is not None:
            metrics[1].metric(t("Maximum atomic force (eV/Å)"), f"{force:.6f}")
        per_atom = result.get("final_energy_ev_per_atom")
        if per_atom is not None:
            metrics[2].metric(t("Energy per atom (eV)"), f"{per_atom:.6f}")
    elif stage["name"] in {"bands", "dos"}:
        mode = (stage.get("metadata") or {}).get("spectral_charge_mode", "fixed")
        if stage["name"] == "dos":
            fermi = result.get("energy_reference_ev", result.get("fermi_energy_ev"))
            if fermi is not None:
                st.metric(t("DOS Fermi energy (eV)"), f"{fermi:.6f}")
        elif mode == "self_consistent":
            fermi = result.get("energy_reference_ev", result.get("fermi_energy_ev"))
            if fermi is not None:
                st.metric(t("Fermi energy (eV)"), f"{fermi:.6f}")
        else:
            if result.get("energy_reference_source") == "preceding_scf":
                scf_fermi_energy = result.get("energy_reference_ev", scf_fermi_energy)
            if scf_fermi_energy is not None:
                st.metric(t("SCF Fermi energy (eV)"), f"{scf_fermi_energy:.6f}")
        st.caption(t("Self-consistent spectrum.") if mode == "self_consistent" else t("Fixed-charge spectrum."))
    if result.get("magnetization"):
        with st.expander(t("Magnetic moments")):
            st.json(result["magnetization"])
    gap = result.get("band_gap") or {}
    if gap.get("gap_ev") is not None:
        st.metric(t("Sampled band gap (eV)"), f"{gap['gap_ev']:.3f}")
    elif gap.get("partial_occupations"):
        st.caption(t("Partial occupations; no insulating gap assigned."))
    output = run_dir / stage["folder"] / "outputs"
    final_structure = output / "final_structure.cif"
    if final_structure.is_file():
        try:
            structure = Structure.from_file(final_structure)
            with st.expander(t("Final structure"), expanded=stage["name"] == "relax"):
                st.write(f"{structure.composition.reduced_formula} · {len(structure)} " + t("atoms") + " · "
                         + t("Cell volume") + f" {structure.volume:.3f} Å³")
                _structure_view(structure)
            st.download_button(t("Download structure (.cif)"), final_structure.read_bytes(),
                               file_name=f"{run_dir.name}-{stage['name']}.cif",
                               mime="chemical/x-cif", key=f"structure_{run_dir.name}_{stage['folder']}")
        except Exception as exc:
            st.error(t("Cannot display structure: {error}").format(error=exc))


def _safe_output_file(path: Path, root: Path) -> bool:
    return (path.is_file() and not path.is_symlink() and path.name != "POTCAR"
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
            and not any(parent.is_symlink() for parent in path.parents))


def _plot_data_files(run_dir: Path, selected: Path, available: list[Path]) -> list[tuple[str, Path]]:
    if selected.stem != "bands_dos":
        source = selected.with_suffix(".csv")
        return [("CSV", source)] if source in available else []

    metadata = selected.with_name("bands_dos_metadata.json")
    if metadata not in available or metadata.stat().st_size > 1024 * 1024:
        return []
    try:
        sources = json.loads(metadata.read_text()).get("source_sha256", {})
    except (OSError, ValueError, AttributeError):
        return []
    if not isinstance(sources, dict):
        return []
    from vasp_slurm_agent.vasp import _sha256

    datasets = []
    for name, label in (("bands.csv", "Bands CSV"), ("dos.csv", "DOS CSV"),
                        ("dos_elements.csv", "Element DOS CSV")):
        if name not in sources:
            continue
        candidates = [selected.with_name(name)] if name == "bands.csv" else sorted(run_dir.glob(f"*/outputs/{name}"))
        for source in candidates:
            if (_safe_output_file(source, run_dir) and source.stat().st_size <= 50 * 1024 * 1024
                    and _sha256(source) == sources[name]):
                datasets.append((label, source))
                break
    return datasets


def _stage_downloads(run_dir: Path, stage: dict) -> None:
    output = run_dir / stage["folder"] / "outputs"
    key = f"files_{run_dir}_{stage['folder']}"
    show_plots = bool((stage.get("result") or {}).get("success"))
    is_qe = (stage.get("metadata") or {}).get("engine") == "qe"
    if (not is_qe and show_plots and stage["name"] == "relax" and (run_dir / "run.json").is_file()
            and (output / "relax_energy.png").is_file()):
        from vasp_slurm_agent.postprocessing import refresh_energy_exports
        try:
            if refresh_energy_exports(run_dir, stage["folder"]):
                st.session_state.pop(f"bundle_{run_dir}", None)
        except (OSError, ValueError, KeyError, RuntimeError):
            show_plots = False
            st.warning(t("Could not refresh the energy plot. Raw files are available below."))
    available = sorted(path for path in output.rglob("*") if _safe_output_file(path, output))
    plots = [path for path in available if path.parent == output and path.suffix == ".png"] if show_plots else []
    if plots:
        labels = {"bands": "Bands", "bands_dos": "Bands and DOS", "dos": "Total DOS", "dos_elements": "Element DOS",
                  "dos_orbitals": "Orbital DOS", "relax_energy": "Relaxation"}
        selected = st.selectbox(t("Plot"), plots, format_func=lambda p: t(labels.get(p.stem, p.stem)), key=key + "_plot")
        st.image(str(selected))
        downloads = [(suffix[1:].upper(), selected.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg")
                     if selected.with_suffix(suffix) in available]
        downloads += _plot_data_files(run_dir, selected, available)
        for column, (label, path) in zip(st.columns(len(downloads)), downloads):
            too_large = path.stat().st_size > 50 * 1024 * 1024
            column.download_button(t(label), b"" if too_large else path.read_bytes(),
                                   file_name=f"{path.parent.parent.name}-{path.name}",
                                   mime="text/csv" if path.suffix == ".csv" else None,
                                   disabled=too_large,
                                   help=t("Included in Download results (.zip).") if too_large else None,
                                   key=key + path.name)
    if available:
        st.write(t("Raw data and files"))
        raw_order = {name: index for index, name in enumerate(("vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "EIGENVAL", "DOSCAR", "PROCAR"))}
        choices = sorted(available, key=lambda path: (raw_order.get(path.name, len(raw_order)) if path.parent == output else len(raw_order), path))
        chosen = st.selectbox(t("File"), choices, format_func=lambda p: p.relative_to(output).as_posix(), key=key + "_file")
        st.caption(f"{chosen.stat().st_size / 1024:.1f} KiB")
        if chosen.suffix == ".csv" and chosen.stat().st_size <= 50 * 1024 * 1024:
            import pandas as pd
            try:
                st.dataframe(pd.read_csv(chosen, nrows=200), hide_index=True)
            except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
                st.caption(t("Preview unavailable."))
        if chosen.stat().st_size <= 50 * 1024 * 1024:
            st.download_button(t("Download file"), chosen.read_bytes(), file_name=chosen.name, key=key + "_download")
        else:
            st.caption(t("Included in Download results (.zip)."))
    if not is_qe and has_config(run_dir) and stage.get("job_id"):
        with st.expander(t("More files and VASPKIT")):
            _postprocess_panel(run_dir, stage, key)


def _postprocess_panel(run_dir: Path, stage: dict, key: str) -> None:
    if not has_config(run_dir) or not stage.get("job_id"):
        return
    from vasp_slurm_agent.postprocessing import fetch_outputs, list_remote_outputs
    st.write(t("More files from the cluster"))
    if st.button(t("List files"), key=key + "_list"):
        try:
            st.session_state[key + "_remote"] = _with_password(list_remote_outputs, run_dir, stage["folder"])
        except Exception as exc:
            st.error(str(exc))
    remote = st.session_state.get(key + "_remote", [])
    if key + "_remote" in st.session_state and not remote:
        st.caption(t("No additional files found."))
    if remote:
        sizes = {row["name"]: row["bytes"] for row in remote}
        names = st.multiselect(t("Collect files"), list(sizes),
                               format_func=lambda name: f"{name} · {sizes[name] / (1024 * 1024):.1f} MiB", key=key + "_collect")
        if st.button(t("Fetch selected files"), key=key + "_fetch", disabled=not names):
            try:
                with st.spinner(t("Collecting files…")):
                    _with_password(fetch_outputs, run_dir, stage["folder"], names)
                st.session_state.pop(f"bundle_{run_dir}", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if stage["name"] not in {"bands", "dos"} or not (stage.get("result") or {}).get("success"):
        return
    st.write("VASPKIT")
    st.caption(t("Uses VASPKIT installed on your cluster. Raw results are kept."))
    if st.button(t("Run VASPKIT"), key=key + "_vaspkit"):
        try:
            from vasp_slurm_agent.postprocessing import postprocess_run
            with st.spinner(t("Processing results…")):
                report = _with_password(postprocess_run, run_dir, stage["folder"])
            st.session_state[key + "_postprocess"] = report
            st.session_state.pop(f"bundle_{run_dir}", None)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    report = st.session_state.get(key + "_postprocess") or stage.get("postprocessing")
    if report:
        st.write(t("Status") + ": " + report.get("status", "unknown"))
        if report.get("message"):
            st.caption(report["message"])


def _remember_key(provider, key_widget, key):
    """Opt-in storage of the API key in the system keychain."""
    from vasp_slurm_agent import credentials

    if provider == "codex":
        return
    remember_widget = f"model_remember_{provider}"
    stored = st.session_state.get(f"{key_widget}_stored")
    available = credentials.keyring_available()
    remember = st.checkbox(t("Remember on this computer"), key=remember_widget, value=bool(stored), disabled=not available,
                           help=t("Stores the key in the system keychain (macOS Keychain, Windows Credential Manager or Secret Service). Untick to remove it.")
                           if available else t("Install the keyring package to store keys in the system keychain."))
    if not available:
        return
    try:
        if remember and key and key != stored:
            credentials.save_key(provider, key)
            st.session_state[f"{key_widget}_stored"] = key
            st.caption(t("Saved in the system keychain."))
        elif not remember and stored:
            credentials.delete_key(provider)
            st.session_state.pop(f"{key_widget}_stored", None)
            st.caption(t("Removed from the system keychain."))
    except credentials.CredentialError as exc:
        st.caption(str(exc))


def _model_editor():
    from vasp_slurm_agent.agent import ModelSettings
    from vasp_slurm_agent.providers import PROVIDERS, credential_envs

    with st.expander(t("Model")):
        provider = st.selectbox(t("Provider"), list(PROVIDERS),
                               format_func=lambda value: PROVIDERS[value]["label"], key="model_provider")
        preset = PROVIDERS[provider]
        st.caption(f"[{t('Model setup')}](https://github.com/cuiqirui99/dft-agent/blob/main/docs/models.md) · " + t("Use your own account."))
        if provider == "codex":
            st.caption(t("Run `codex login` first. Usage follows that CLI account."))
        elif provider == "responses":
            st.caption(f"[{t('Get an API key')}](https://platform.openai.com/api-keys). " + t("API billing is separate from ChatGPT."))
        elif provider == "qwen":
            st.caption(t("Use the key and API URL from the same Model Studio region."))
        elif provider == "glm":
            st.caption(t("Use a Z.AI key, or a BigModel key with its matching API URL."))
        elif provider == "chat_completions":
            st.caption(t("Use your provider's key and base URL. Strict JSON-schema output is required."))
        else:
            st.caption(t("Use your {provider} API key.").format(provider=preset['label']))
        model = st.text_input(t("Model name"), key=f"model_name_{provider}",
                              value=os.environ.get("DFT_AGENT_MODEL", "")
                              if provider in {"responses", "chat_completions"} else "",
                              placeholder=preset.get("model_example") or "",
                              help=t("Leave blank for the Codex CLI default.") if provider == "codex"
                              else t("Enter a model ID available to your API account."))
        base_url = st.text_input(t("API URL") if provider in {"qwen", "chat_completions"} else t("API URL (optional)"),
                                 key=f"model_base_url_{provider}",
                                 value=os.environ.get("DFT_AGENT_BASE_URL", "")
                                 if provider in {"responses", "chat_completions"} else "",
                                 placeholder=preset.get("base_url") or "",
                                 disabled=provider == "codex",
                                 help=t("Copy your regional Model Studio base URL, ending in /compatible-mode/v1.")
                                 if provider == "qwen" else t("Enter the service's API base URL, not its chat website.")
                                 if provider == "chat_completions" else t("Leave blank for OpenAI.")
                                 if provider == "responses" else t("Leave blank to use the shown default."))
        key_widget = "model_api_key" if provider == "responses" else f"model_api_key_{provider}"
        if provider != "codex" and key_widget not in st.session_state:
            from vasp_slurm_agent import credentials
            stored = credentials.stored_key(provider)
            if stored:
                st.session_state[key_widget] = stored
                st.session_state[f"{key_widget}_stored"] = stored
        env_names = " or ".join(credential_envs(provider))
        key_help = t("Kept in memory unless you tick Remember. Clear removes only the session key.")
        if env_names:
            key_help += " " + t("Leave blank if {names} is set.").format(names=env_names)
        key = st.text_input(t("API key"), type="password", key=key_widget, disabled=provider == "codex",
                            help=key_help)
        st.button(t("Clear API key"), disabled=provider == "codex",
                  on_click=lambda: st.session_state.update({key_widget: ""}))
        _remember_key(provider, key_widget, key)
        st.caption(t("Tokens measure model use. No tokens are included with DFT Agent."))
    return ModelSettings(provider=provider, model=model.strip(),
                         base_url=None if provider == "codex" else base_url.strip() or None,
                         api_key=None if provider == "codex" else key or None)


def _model_usage(usage):
    if not isinstance(usage, dict):
        return
    fields = (("input_tokens", "input"), ("output_tokens", "output"),
              ("cached_input_tokens", "cached"), ("reasoning_tokens", "reasoning"))
    parts = []
    for key, label in fields:
        value = usage.get(key)
        count = f"{value:,}" if type(value) is int and value >= 0 else "unknown"
        parts.append(f"{label} {count}")
    latency = usage.get("latency_seconds")
    if type(latency) in {int, float} and math.isfinite(latency) and latency >= 0:
        parts.append(f"{latency:.1f} s")
    st.caption(t("Model usage") + " · " + " · ".join(parts))


def _failed_model_usage(usage):
    if isinstance(usage, dict):
        st.caption(t("Last failed call"))
        _model_usage(usage)


def _after_prepare(config):
    st.success(t("Inputs ready. Review below, then submit."))
    if config is None:
        st.info(t("No cluster is attached yet. Save Cluster setup, then submit from the run page."))


def _prepare_from_plan(upload, runs_root, config, plan):
    if plan.get("source_sha256") != hashlib.sha256(upload.getvalue()).hexdigest():
        raise ValueError(t("The structure changed. Create a new plan."))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    with tempfile.TemporaryDirectory(prefix="dft-agent-input-") as directory:
        source = Path(directory) / f"structure{suffix}"
        source.write_bytes(upload.getvalue())
        if plan.get("kind") == "task":
            from vasp_slurm_agent.task_agent import prepare_task
            prepare_task(source, run_dir, config, plan)
        else:
            prepare_plan(source, run_dir, config, tasks=plan["tasks"], parameters=plan["parameters"],
                         magnetic_states=plan.get("magnetic_states") or None, initial_moment=plan.get("initial_moment", 3.0))
    _save_structure_edit(upload, run_dir)
    if plan.get("kind") != "task":
        (run_dir / "proposal.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    st.session_state["active_run"] = str(run_dir)
    _after_prepare(config)


def _clear_calculation_plan():
    for key in ("agent_stamp", "agent_proposal", "agent_history", "agent_clear_revision", "plan_revision", "agent_error_usage"):
        st.session_state.pop(key, None)
    st.session_state["manual_moments"] = "null"
    st.session_state["manual_axis"] = "0 0 1"


def _save_structure_edit(upload, run_dir):
    for name, data in getattr(upload, "edit_files", {}).items():
        destination = run_dir / "structure_edit" / name
        if not destination.resolve().is_relative_to((run_dir / "structure_edit").resolve()):
            raise ValueError("Invalid structure record path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _edit_preview(upload, plan, origin):
    from vasp_slurm_agent.structures import apply_structure_plan

    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    with tempfile.TemporaryDirectory(prefix="dft-structure-preview-") as directory:
        source = Path(directory) / f"input{suffix}"
        source.write_bytes(upload.getvalue())
        output = Path(directory).resolve() / "output"
        result = apply_structure_plan(source, output, plan)
        paths = {key: Path(value) for key, value in result["files"].items()}
        names = {key: str(path.relative_to(output)) for key, path in paths.items()}
        data = {names[key]: path.read_bytes() for key, path in paths.items()}
    return {"plan": plan, "origin": origin, "input": result["input"], "output": result["output"],
            "warnings": result.get("warnings", []), "names": names, "files": data}


def _structure_editor(upload, settings):
    raw_hash = hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else None
    if st.session_state.get("structure_source_hash") != raw_hash:
        for key in ("structure_plan", "structure_history", "structure_preview", "structure_active", "structure_original",
                    "structure_goal", "structure_followup", "structure_goal_stamp", "structure_clear_followup", "structure_error_usage"):
            st.session_state.pop(key, None)
        st.session_state["structure_source_hash"] = raw_hash
        _clear_calculation_plan()
    if upload is None:
        return None
    active = st.session_state.get("structure_active")
    with st.expander(t("Edit structure")):
        st.caption(t("Edits are local. Review the preview before using it."))
        try:
            if "structure_original" not in st.session_state:
                from vasp_slurm_agent.structures import preview_structure

                suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
                with tempfile.TemporaryDirectory(prefix="dft-structure-source-") as directory:
                    source = Path(directory) / f"input{suffix}"
                    source.write_bytes(upload.getvalue())
                    st.session_state["structure_original"] = preview_structure(source, [])["input"]
            st.caption(t("Original site indices (starting at 0)"))
            st.dataframe(st.session_state["structure_original"]["sites"], hide_index=True)
        except Exception as exc:
            st.error(t("Cannot edit structure: {error}").format(error=exc))
            return upload
        goal = st.text_area(t("Structure goal"), key="structure_goal", max_chars=4000,
                            placeholder=t("Make a 2 × 2 × 1 supercell and save both formats."))
        goal_stamp = hashlib.sha256((raw_hash + goal).encode()).hexdigest()
        if st.session_state.get("structure_goal_stamp") != goal_stamp:
            for key in ("structure_plan", "structure_history", "structure_preview", "structure_followup", "structure_error_usage"):
                st.session_state.pop(key, None)
            st.session_state["structure_goal_stamp"] = goal_stamp
        if st.session_state.pop("structure_clear_followup", False):
            st.session_state["structure_followup"] = ""
        plan = st.session_state.get("structure_plan")
        history = st.session_state.get("structure_history", [])
        revision = st.text_input(t("Follow-up"), key="structure_followup") if plan or history else ""
        if st.button(t("Plan structure"), disabled=not goal.strip()):
            from vasp_slurm_agent.structure_agent import draft_structure

            history = list(history)
            pending = {"role": "user", "content": revision.strip()}
            if pending["content"] and (not history or history[-1] != pending):
                history.append(pending)
            st.session_state["structure_history"] = history
            st.session_state.pop("structure_plan", None)
            st.session_state.pop("structure_preview", None)
            try:
                suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
                with tempfile.TemporaryDirectory(prefix="dft-structure-plan-") as directory:
                    source = Path(directory) / f"input{suffix}"
                    source.write_bytes(upload.getvalue())
                    with st.spinner(t("Planning structure…")):
                        plan = draft_structure(goal.strip(), source, settings, history=history)
                st.session_state["structure_plan"] = plan
                st.session_state.pop("structure_error_usage", None)
                st.session_state["structure_history"] = plan.get("dialogue", history)
                st.session_state["structure_clear_followup"] = True
                st.rerun()
            except Exception as exc:
                plan = None
                st.session_state["structure_error_usage"] = getattr(exc, "model_usage", None)
                st.error(t("Cannot plan structure: {error}").format(error=exc))
        _failed_model_usage(st.session_state.get("structure_error_usage"))
        if plan:
            st.write(plan["summary"])
            _model_usage(plan.get("model_usage"))
            for question in plan.get("questions", []):
                st.info(question)
            for note in plan.get("notes", []):
                st.caption(note)
            st.json(plan["operations"])
            st.caption(t("Output: ") + plan["output_format"].upper())
            if revision.strip():
                st.info(t("Update the structure plan before previewing it."))
            if st.button(t("Preview structure"), disabled=plan["status"] != "ready" or bool(revision.strip())):
                try:
                    st.session_state["structure_preview"] = _edit_preview(upload, plan, "model")
                except Exception as exc:
                    st.session_state.pop("structure_preview", None)
                    st.error(t("Cannot preview structure: {error}").format(error=exc))
        output_format = st.selectbox(t("Convert to"), ["cif", "poscar", "both"], format_func=str.upper)
        if st.button(t("Convert")):
            conversion = {"status": "ready", "source_sha256": raw_hash,
                          "operations": active["plan"]["operations"] if active else [], "output_format": output_format}
            try:
                st.session_state["structure_preview"] = _edit_preview(upload, conversion, "conversion")
            except Exception as exc:
                st.session_state.pop("structure_preview", None)
                st.error(t("Cannot convert structure: {error}").format(error=exc))
        preview = st.session_state.get("structure_preview")
        if preview:
            before, after = preview["input"], preview["output"]
            st.write(f"{before['formula']} · {before['number_of_sites']} {t('atoms')} → {after['formula']} · {after['number_of_sites']} {t('atoms')}")
            st.caption(t("Cell volume") + f": {abs(np.linalg.det(before['lattice_angstrom'])):.3f} → {abs(np.linalg.det(after['lattice_angstrom'])):.3f} Å³")
            for warning in preview["warnings"]:
                st.warning(warning)
            st.dataframe(after["sites"], hide_index=True)
            for kind, label in (("cif", "Download CIF"), ("poscar", "Download POSCAR")):
                if preview["plan"]["output_format"] in {kind, "both"}:
                    name = preview["names"][kind]
                    st.download_button(t(label), preview["files"][name], file_name=Path(name).name,
                                       mime="chemical/x-cif" if kind == "cif" else "text/plain")
            stale = preview["plan"].get("source_sha256") != raw_hash or (preview["origin"] == "model" and bool(revision.strip()))
            if st.button(t("Use structure"), disabled=stale):
                st.session_state["structure_active"] = preview
                _clear_calculation_plan()
                st.rerun()
        if active and st.button(t("Use original")):
            st.session_state.pop("structure_active", None)
            _clear_calculation_plan()
            st.rerun()
    if active:
        st.caption(t("Using edited structure. Site moments were cleared; check the new site order."))
        result = BytesIO(active["files"][active["names"]["poscar"]])
        result.name = "POSCAR"
        result.edit_files = active["files"]
        return result
    return upload


def _science_details(report):
    if not report:
        return
    with st.expander(t("Scientific guidance")):
        for item in report.get("method_decisions", []):
            st.write(item["decision"])
        for item in report.get("risks", []):
            st.caption(item)
        for item in report.get("questions", []):
            st.info(item)
        for item in report.get("validation_plan", []):
            st.write("• " + item)
        sources = {}
        for item in report.get("evidence", []):
            for source in item.get("sources", []):
                sources[source["url"]] = item["title"]
        for url, title in sources.items():
            st.markdown(f"[{title}]({url})")
        if report.get("experience"):
            st.caption(t("Matched past runs: {count}. Suggestions only; settings are unchanged.").format(count=len(report['experience'])))
        for item in report.get("knowledge", []):
            st.write(f"**{item['title']}** — {item['summary']}")
            for condition in item.get("conditions", []):
                st.caption(condition)
            for limit in item.get("limitations", []):
                st.caption(limit)
            st.caption(t("Reference: {id}. See Memory for evidence.").format(id=item['id']))


def _memory_panel(runs_root):
    from vasp_slurm_agent.knowledge import load_catalog, preview_import, import_selected

    st.subheader(t("Memory"))
    st.caption(t("Built-in guidance and cases. Only verified entries supply advice; settings still need review."))
    with st.expander(t("Import local records")):
        source = st.text_input(t("Local file or run folder"), key="memory_source",
                               help=t("Choose an old records.jsonl, platform.db, or a saved DFT Agent run."))
        stamp = (source.strip(), str(runs_root))
        if st.session_state.get("memory_import_stamp") != stamp:
            st.session_state["memory_import_stamp"] = stamp
            st.session_state.pop("memory_preview", None)
            st.session_state.pop("memory_selection", None)
        st.caption(t("Only selected records are imported. Legacy lessons stay local and need review. Saved runs are rechecked before use."))
        if st.button(t("Preview import"), disabled=not source.strip()):
            st.session_state.pop("memory_preview", None)
            st.session_state.pop("memory_selection", None)
            try:
                st.session_state["memory_preview"] = preview_import(Path(source.strip()).expanduser())
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
        preview = st.session_state.get("memory_preview")
        if preview is not None:
            if preview:
                choices = {item["id"]: item for item in preview}
                selected = st.multiselect(t("Records"), list(choices), key="memory_selection",
                                          format_func=lambda key: f"{choices[key]['title']} · {choices[key]['status']}")
                if st.button(t("Import selected"), disabled=not selected):
                    try:
                        result = import_selected(Path(source.strip()).expanduser(), selected, runs_root)
                        st.success(t("Imported {imported}; already saved {skipped}.").format(imported=result['imported'], skipped=result['skipped']))
                    except (ValueError, OSError, RuntimeError) as exc:
                        st.error(str(exc))
            else:
                st.info(t("No importable records found."))
    try:
        records = load_catalog(runs_root)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        return
    search = st.text_input(t("Search memory"), placeholder="SOC, convergence, NiO…").strip().lower()
    status = st.selectbox(t("Status"), ["All", "verified", "candidate", "shadow_verified", "rejected", "deprecated"])
    matches = [item for item in records
               if (status == "All" or item["status"] == status)
               and (not search or search in " ".join([item["title"], item["summary"],
                    *item.get("topics", []), *item.get("formulas", []), *item.get("methods", [])]).lower())]
    if not matches:
        st.info(t("No matching records."))
        return
    st.dataframe([{t("Title"): item["title"], t("Status"): t("Rechecked on use") if item.get("import_type") == "run" else item["status"], t("Source"): item["origin"],
                   t("Evidence"): t("Checked") if item.get("evidence_valid") else t("Needs review")}
                  for item in matches], hide_index=True)
    item = matches[st.selectbox(t("View record"), range(len(matches)),
                               format_func=lambda index: matches[index]["title"])]
    st.write(item["summary"])
    if item.get("import_type") == "run" and item.get("evidence_valid"):
        st.caption(t("The original inputs and outputs are rechecked for each plan."))
    elif item["status"] != "verified" or not item.get("evidence_valid"):
        st.info(t("This record is not used as verified advice."))
    for text in item.get("conditions", []):
        st.write("• " + text)
    for text in item.get("limitations", []):
        st.caption(text)
    for source in item.get("sources", []):
        st.markdown(f"[{source['title']}]({source['url']})")
    with st.expander(t("Evidence")):
        st.json({"id": item["id"], "status": item["status"], "outcome": item["outcome"],
                 "evidence": item.get("evidence", [])})


def _agent_plan(upload, structure, config, runs_root, settings):
    from vasp_slurm_agent.task_agent import draft_task

    goal = st.text_area(t("Goal"), max_chars=4000, key="calculation_goal",
                        placeholder=t("What would you like to calculate?"))
    stamp = hashlib.sha256((upload.getvalue() if upload else b"") + goal.encode()).hexdigest()
    if st.session_state.get("agent_stamp") != stamp:
        for key in ("agent_proposal", "agent_history", "agent_clear_revision", "plan_revision", "agent_error_usage"):
            st.session_state.pop(key, None)
        st.session_state["agent_stamp"] = stamp
    saved = st.session_state.get("agent_proposal")
    if st.session_state.pop("agent_clear_revision", False):
        st.session_state["plan_revision"] = ""
    revision = st.text_input(t("Change the plan"), key="plan_revision") if saved or st.session_state.get("agent_history") else ""
    if st.button(t("Plan"), type="primary", disabled=structure is None or not goal.strip()):
        history = list(st.session_state.get("agent_history", []))
        pending = {"role": "user", "content": revision.strip()}
        if pending["content"] and (not history or history[-1] != pending):
            history.append(pending)
        st.session_state["agent_history"] = history
        st.session_state.pop("agent_proposal", None)
        saved = None
        try:
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="dft-agent-plan-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                with st.spinner(t("Planning…")):
                    plan = draft_task(goal.strip(), source, settings, history=history, runs_root=runs_root)
            history = plan.get("dialogue", history + [{"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}])
            saved = {"stamp": stamp, "plan": plan, "goal": plan.get("goal", goal.strip())}
            st.session_state["agent_proposal"] = saved
            st.session_state.pop("agent_error_usage", None)
            st.session_state["agent_history"] = history
            st.session_state["agent_clear_revision"] = True
            revision = ""
            st.rerun()
        except Exception as exc:
            st.session_state["agent_error_usage"] = getattr(exc, "model_usage", None)
            st.error(str(exc))
    _failed_model_usage(st.session_state.get("agent_error_usage"))
    if not saved:
        return
    plan = saved["plan"]
    st.write(plan["summary"])
    _model_usage(plan.get("model_usage"))
    for question in plan.get("questions", []):
        st.info(question)
    if plan.get("status") == "unsupported":
        st.warning(t("This plan cannot run yet."))
    _task_preview(plan)
    _science_details(plan.get("scientific_report"))
    with st.expander(t("Plan details")):
        st.json({key: plan[key] for key in ("operations", "stages", "variants", "tasks", "parameters", "notes") if key in plan})
    if revision.strip():
        st.info(t("Update the plan before preparing inputs."))
    if st.button(t("Prepare inputs"), disabled=plan.get("status") != "ready" or bool(revision.strip())):
        try:
            _prepare_from_plan(upload, runs_root, config, {**plan, "goal": saved["goal"],
                              "dialogue": st.session_state.get("agent_history", [])})
        except Exception as exc:
            st.error(t("Cannot prepare inputs: {error}").format(error=exc))
    if config is None:
        st.caption(t("Inputs can be prepared and reviewed without a cluster. Save Cluster setup before submitting."))


def _task_preview(plan):
    preview = plan.get("preview")
    if preview and plan.get("operations"):
        result = preview["output"]
        st.write(f"{result['formula']} · {result['number_of_sites']} " + t("atoms"))
        st.dataframe(plan["operations"], hide_index=True)
    if plan.get("stages"):
        rows = []
        for variant in plan.get("variants") or [{"label": "", "stages": plan["stages"]}]:
            for stage in variant["stages"]:
                parameters = stage["parameters"]
                rows.append({t("Variant"): variant["label"], t("Stage"): stage["task"],
                             t("Method"): parameters["functional"], "SOC": parameters["soc"],
                             t("Spin"): parameters["spin"], t("Strain"): str(variant.get("strain") or "—")})
        st.dataframe(rows, hide_index=True)


def _new_run(config: ClusterConfig | None, runs_root: Path, model_settings=None) -> None:
    if st.session_state.get("cluster_engine", config.engine if config else "vasp") == "qe":
        kind = st.radio("Input", ["Structure", "Pairing model"], horizontal=True, key="qe_input_kind")
        if kind == "Pairing model":
            from vasp_slurm_agent.qe_pairing_app import new_run
            new_run(config, runs_root, model_settings)
            return
    with st.container(key="calculation_workspace"):
        structure_column, task_column = st.columns([1, 1.15], gap="medium")
        with structure_column, st.container(border=True, key="structure_panel"):
            st.subheader(t("Structure"))
            source = st.radio(t("Structure source"), ["Upload", "Example"], horizontal=True, format_func=t, key="structure_source")
            if source == "Upload":
                upload = st.file_uploader(t("Upload a structure"), key="structure_upload",
                                          help=t("CIF or POSCAR; no extension needed for POSCAR."))
            else:
                examples = files("vasp_slurm_agent").joinpath("examples")
                choices = {}
                for item in sorted(examples.iterdir(), key=lambda item: item.name):
                    if item.name.endswith((".cif", ".vasp")):
                        choices[Path(item.name).stem] = item.name
                name = st.selectbox(t("Example"), [choices[label] for label in sorted(choices)],
                                    format_func=lambda value: Path(value).stem, key="structure_example")
                upload = BytesIO(examples.joinpath(name).read_bytes())
                upload.name = name
            preview = st.container()
            upload = _structure_editor(upload, model_settings)
            with preview:
                structure = _preview(upload) if upload is not None else None
            if source == "Example":
                st.download_button(t("Download input"), examples.joinpath(name).read_bytes(), file_name=name,
                                   mime="chemical/x-cif" if name.endswith(".cif") else "text/plain")
        with task_column, st.container(border=True, key="task_panel"):
            st.subheader(t("Calculation"))
            _calculation_controls(upload, structure, config, runs_root, model_settings)


def _calculation_controls(upload, structure, config, runs_root, model_settings):
    mode = st.radio(t("Mode"), ["Agent", "Manual"], horizontal=True, format_func=t, key="calculation_mode")
    if st.session_state.get("cluster_engine", config.engine if config else "vasp") == "qe":
        from vasp_slurm_agent.qe_app import new_run
        new_run(upload, structure, config, runs_root, model_settings, mode)
        return
    if mode == "Agent":
        _agent_plan(upload, structure, config, runs_root, model_settings)
        return
    with st.container(border=True):
        task = st.selectbox(t("Calculation"), ["relax", "scf", "bands", "dos"], key="manual_task",
                            format_func=lambda key: t({"relax": "Structure relaxation", "scf": "Self-consistent calculation (SCF)",
                                                       "bands": "Band structure", "dos": "Density of states (DOS)"}[key]))
        st.caption(t("Starting values. Check convergence for your material."))
        left, right = st.columns(2)
        with left:
            encut = st.number_input("ENCUT / eV", 100.0, 2000.0, 520.0, 10.0, key="manual_encut")
            ediff = st.number_input("EDIFF / eV", min_value=1e-10, max_value=1e-2, value=1e-5,
                                    format="%.1e", key="manual_ediff")
            electronic_type = st.selectbox(t("Electronic type"), ["auto", "metal", "insulator"], key="manual_electronic_type")
            ismear = st.selectbox("ISMEAR", [None, -5, -1, 0, 1, 2], format_func=lambda v: t("Automatic") if v is None else str(v), key="manual_ismear")
            sigma_text = st.text_input(t("SIGMA / eV (optional)"), "", placeholder=t("Automatic"), key="manual_sigma")
        with right:
            kspacing = st.number_input(t("K-point spacing / Å⁻¹"), 0.01, 10.0, 0.25, 0.01,
                                       help=t("Reciprocal spacing includes 2π. An explicit grid overrides it."), key="manual_kspacing")
            k_grid = st.text_input(t("Gamma-centered k-point grid"), "", placeholder=t("Automatic"), help=t("Optional override: three positive integers."), key="manual_grid")
            nsw = st.number_input(t("Maximum relaxation steps (NSW)"), 1, 1000, 100, key="manual_nsw")
            ediffg = st.number_input(t("Force convergence threshold (eV/Å)"), 0.0001, 1.0, 0.03, 0.005, format="%.4f", key="manual_ediffg")
            cell_relax = st.checkbox(t("Relax the cell as well as atomic positions"), value=False, key="manual_cell_relax")
            st.caption(t("Unchecked: relax atoms at fixed cell."))
        with st.expander(t("Method")):
            functional = st.selectbox(t("Functional"), ["PBE", "HSE06", "PBE0"], key="manual_functional")
            spin = st.selectbox(t("Spin"), ["none", "collinear", "noncollinear"], key="manual_spin")
            soc = st.checkbox("SOC", key="manual_soc")
            moments = st.text_area(t("Site moments (JSON)"), value="null", key="manual_moments", help=t("One value per input site, or three components for noncollinear spins."))
            axis = st.text_input(t("Spin axis"), "0 0 1", key="manual_axis")
            hubbard = st.text_area(t("Hubbard U (JSON)"), value="{}", help=t('Example: {"Ni": {"l": 2, "u": 5, "j": 0}}'), key="manual_hubbard")
            comparisons = st.multiselect(t("Compare magnetic states"), ["NM", "FM", "AFM"], help=t("SCF seeds on the same cell."), key="manual_comparisons")
        prepared = st.button(t("Prepare inputs"), type="primary",
                                         disabled=structure is None)
    if config is None:
        st.caption(t("Inputs can be prepared and reviewed without a cluster. Save Cluster setup before submitting."))
    if prepared:
        try:
            grid = [int(part) for part in k_grid.replace(",", " ").split()] if k_grid.strip() else None
            if grid is not None and (len(grid) != 3 or any(part < 1 for part in grid)):
                raise ValueError(t("Use three positive integers for the k-point grid."))
            parameters = dict(encut=encut, ediff=ediff, ediffg=-ediffg, nsw=int(nsw),
                              mesh=grid, kspacing=kspacing, electronic_type=electronic_type,
                              ismear=ismear, sigma=float(sigma_text) if sigma_text.strip() else None, cell_relax=cell_relax,
                              functional=functional, spin=spin, soc=soc, magmom=json.loads(moments),
                              saxis=[float(v) for v in axis.split()], hubbard_u=json.loads(hubbard))
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = runs_root / run_id
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="vasp-agent-input-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                prepare_plan(source, run_dir, config, tasks=[task], parameters=parameters, magnetic_states=comparisons or None)
            _save_structure_edit(upload, run_dir)
            st.session_state["active_run"] = str(run_dir)
            _after_prepare(config)
        except Exception as exc:
            st.error(t("Cannot prepare inputs: {error}").format(error=exc))


def _result_dialogue(run_dir, settings, context):
    from vasp_slurm_agent.explanation import explain_run, load_run_context

    if context.get("goal"):
        st.subheader(t("Saved goal"))
        st.write(context["goal"])
    st.subheader(t("Outcome"))
    st.write(context.get("outcome") or t("Run status: {status}.").format(status=_status(context['status'])))
    record_path = run_dir / "explanations.json"
    record = {}
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text())
            if not isinstance(record, dict):
                raise ValueError("Invalid dialogue record.")
        except (OSError, ValueError) as exc:
            st.warning(t("Cannot read saved explanations: {error}").format(error=exc))
            record = {}
    if record and record.get("context_sha256") != context["context_sha256"]:
        st.info(t("Results changed. Ask again for an updated explanation."))
        record = {}
    question = st.text_input(t("Ask about this run"), key=f"result_question_{run_dir}", max_chars=4000)
    actions = st.columns(2)
    explain = actions[0].button(t("Explain results"), key=f"explain_{run_dir}")
    ask = actions[1].button(t("Ask"), disabled=not question.strip(), key=f"ask_{run_dir}")
    error_usage_key = f"explanation_error_usage_{run_dir}_{context['context_sha256']}"
    if explain or ask:
        prompt = question.strip() if ask else "Explain the results."
        history = record.get("history", [])
        try:
            with st.spinner(t("Reading results…")):
                response = explain_run(run_dir, settings, question=prompt, history=history)
            if response["context_sha256"] != context["context_sha256"] or response["context_sha256"] != load_run_context(run_dir)["context_sha256"]:
                record = {}
                raise ValueError(t("Results changed. Ask again for an updated explanation."))
            exchange = {"question": prompt, **response}
            reply = {key: response[key] for key in ("answer", "evidence", "limits", "next_steps")}
            record = {"context_sha256": response["context_sha256"],
                      "history": history + [{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)}],
                      "exchanges": record.get("exchanges", []) + [exchange]}
            with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, prefix=".explanations-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            try:
                os.replace(temporary, record_path)
            finally:
                temporary.unlink(missing_ok=True)
            st.session_state.pop(error_usage_key, None)
        except Exception as exc:
            st.session_state[error_usage_key] = getattr(exc, "model_usage", None)
            st.error(t("Cannot explain results: {error}").format(error=exc))
    _failed_model_usage(st.session_state.get(error_usage_key))
    for exchange in record.get("exchanges", []):
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            _model_usage(exchange.get("model_usage"))
            with st.expander(t("Evidence")):
                for fact_id in exchange.get("evidence", []):
                    fact = context["facts"].get(fact_id)
                    if fact:
                        unit = f" {fact['unit']}" if fact.get("unit") else ""
                        st.write(f"{fact_id} · {fact['label']}: {fact['value']}{unit}")
                        st.caption(fact["source"])
            for limit in exchange.get("limits", []):
                st.caption(limit)
            for step in exchange.get("next_steps", []):
                st.write(step)


def _repair_panel(run_dir, settings):
    from vasp_slurm_agent.recovery import draft_repair, prepare_repair

    key = f"repair_{run_dir}"
    with st.expander(t("Repair")):
        if st.button(t("Plan repair"), key=f"plan_repair_{run_dir}"):
            st.session_state.pop(key, None)
            try:
                with st.spinner(t("Checking the failed run…")):
                    st.session_state[key] = draft_repair(run_dir, settings)
                st.session_state.pop(key + "_error_usage", None)
            except Exception as exc:
                st.session_state[key + "_error_usage"] = getattr(exc, "model_usage", None)
                st.error(str(exc))
        _failed_model_usage(st.session_state.get(key + "_error_usage"))
        proposal = st.session_state.get(key)
        if not proposal:
            return
        st.write(proposal["diagnosis"])
        _model_usage(proposal.get("model_usage"))
        for question in proposal.get("questions", []):
            st.info(question)
        if proposal.get("changes"):
            st.dataframe(proposal["changes"], hide_index=True)
            st.caption(t("Repair {attempt} of {maximum}.").format(attempt=proposal['attempt'], maximum=proposal['max_attempts']))
        if st.button(t("Prepare repair"), key=f"prepare_repair_{run_dir}", disabled=proposal["status"] != "ready"):
            try:
                name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-repair-" + uuid.uuid4().hex[:8]
                child = run_dir.parent / name
                prepare_repair(run_dir, child, proposal)
                st.session_state["active_run"] = str(child)
                st.session_state.pop(key, None)
                st.rerun()
            except Exception as exc:
                st.error(t("Cannot prepare repair: {error}").format(error=exc))


def _sample_notice(run_dir):
    from vasp_slurm_agent.samples import sample_card

    card = sample_card(run_dir)
    if card:
        st.info(t("Sample run") + f": {card.get('title', '')}. " + t("Completed on a real cluster and bundled with DFT Agent; cluster details are placeholders."))


@st.fragment(run_every="5s")
def _run_panel(run_dir: Path, model_settings=None, config=None, *, read_only=False) -> None:
    from vasp_slurm_agent.explanation import load_run_context

    try:
        state = read_state(run_dir)
        if state.get("workflow") == "pairing":
            if read_only:
                st.json(state)
                return
            from vasp_slurm_agent.qe_pairing_app import run_panel
            run_panel(run_dir, model_settings)
            return
        context = load_run_context(run_dir)
    except Exception as exc:
        st.error(t("Cannot read run: {error}").format(error=exc))
        return
    status = context["status"]
    stages = []
    for index, original in enumerate(state.get("stages", []), 1):
        stage = dict(original)
        if (stage.get("result") or {}).get("success") and not context["facts"].get(f"stage_{index}.accepted", {}).get("value"):
            message = context["facts"].get(f"stage_{index}.availability", {}).get("value", "Saved results could not be verified.")
            stage = {**stage, "status": "needs_attention", "result": {"success": False, "reason": message}}
        elif (stage.get("result") or {}).get("success"):
            facts = context["facts"]
            gap = {"gap_ev": facts.get(f"stage_{index}.band_gap", {}).get("value"),
                   "scope": facts.get(f"stage_{index}.band_gap_scope", {}).get("value"),
                   "partial_occupations": facts.get(f"stage_{index}.band_gap_status", {}).get("value") == "metallic_or_partially_occupied"}
            stage["result"] = {**stage["result"], "band_gap": gap}
            if state.get("engine") == "qe":
                for field, suffix in (("endpoint_relaxed", "endpoint_relaxed"), ("barrier_scope", "barrier_scope"),
                                      ("converged_ionic", "ionic_converged"), ("path_converged", "path_converged")):
                    item = facts.get(f"stage_{index}.{suffix}")
                    if item is not None:
                        stage["result"][field] = item["value"]
        stages.append(stage)
    awaiting_submission = status == "planned" and not any(stage.get("job_id") for stage in stages)
    attached = has_config(run_dir)
    st.subheader(f"{state.get('formula', '')} · {state.get('task', '')}")
    st.caption(str(run_dir))
    if not read_only and st.button(t("Open chat"), key=f"open_chat_{run_dir}"):
        from vasp_slurm_agent.conversation_app import open_run_chat
        open_run_chat(run_dir)
    _sample_notice(run_dir)
    status_label = t("Preparing next stage") if status == "planned" and not awaiting_submission else _status(status)
    st.write(t("Status") + f": **{status_label}**")
    if state.get("comparison") and all((stage.get("result") or {}).get("success") for stage in stages):
        with st.expander(t("Magnetic comparison"), expanded=True):
            st.json(state["comparison"])
    if state.get("last_error"):
        st.warning(str(state["last_error"]))
    st.dataframe([
        {t("Stage"): stage.get("label", stage.get("name", "")), t("Status"): _status(stage.get("status")),
         t("Slurm job"): stage.get("job_id") or "—"}
        for stage in stages
    ], hide_index=True)
    with st.expander(t("Review settings"), expanded=awaiting_submission):
        st.json(state.get("parameters", {}))
        if attached:
            try:
                run_config = ClusterConfig.load(run_dir / "config.json")
                st.json(run_config.to_dict())
            except Exception as exc:
                st.error(t("Cannot read run settings: {error}").format(error=exc))
        else:
            st.info(t("No cluster is attached to this run yet. Submitting attaches the saved Cluster setup."))
    with st.expander(t("Stage details and run history")):
        st.json({"stages": stages, "history": state.get("history", [])})
    is_qe = state.get("engine") == "qe"
    with st.expander("Generated inputs" if is_qe else t("Generated INCAR and KPOINTS")):
        for stage in stages:
            folder = stage.get("folder")
            if not folder:
                continue
            st.write(stage.get("name", folder))
            for name in ([stage.get("metadata", {}).get("input_file", "pw.in")] if is_qe else ("INCAR", "KPOINTS")):
                source = run_dir / folder / "inputs" / name
                if source.is_file():
                    st.caption(name)
                    st.code(source.read_text(), language="text")
    scf_fermi_energy = None
    for stage in stages:
        if stage.get("name") == "scf" and (stage.get("result") or {}).get("success"):
            scf_fermi_energy = stage["result"].get("fermi_energy_ev")
        _stage_results(run_dir, stage, scf_fermi_energy)
        folder = stage.get("folder")
        if folder and (run_dir / folder / "outputs").is_dir():
            _stage_downloads(run_dir, stage)

    if read_only:
        if st.button(t("Prepare download"), key=f"project_bundle_{run_dir}"):
            try:
                st.session_state[f"project_bundle_path_{run_dir}"] = str(bundle_run(run_dir))
            except Exception as exc:
                st.error(str(exc))
        archive = st.session_state.get(f"project_bundle_path_{run_dir}")
        if archive and Path(archive).is_file() and Path(archive).stat().st_size <= 100 * 1024 * 1024:
            st.download_button(t("Download results (.zip)"), Path(archive).read_bytes(), file_name=Path(archive).name,
                               mime="application/zip", key=f"project_download_{run_dir}")
        return

    actions = st.columns(3)
    if awaiting_submission:
        reviewed = st.checkbox(t("Structure, settings and resources reviewed."), key=f"review_{run_dir.name}")
        if not attached and config is None:
            st.info(t("Save Cluster setup to submit this run."))
        if actions[0].button(t("Submit calculation"), type="primary", disabled=not reviewed or (not attached and config is None),
                             key=f"submit_{run_dir.name}"):
            try:
                if not attached:
                    attach_config(run_dir, config)
                pid = _with_password(start_worker, run_dir)
                st.success(t("Monitoring started (PID {pid}). You can close this page.").format(pid=pid))
            except Exception as exc:
                st.error(t("Cannot start monitoring: {error}").format(error=exc))
    elif status not in TERMINAL:
        if actions[0].button(t("Resume monitoring"), key=f"watch_{run_dir.name}",
                             help=t("Resume monitoring existing jobs after a restart or interruption.")):
            try:
                st.success(t("Monitoring started (PID {pid}).").format(pid=_with_password(start_worker, run_dir)))
            except Exception as exc:
                st.error(t("Cannot resume monitoring: {error}").format(error=exc))
    elif status in {"needs_attention", "failed"}:
        if actions[0].button(t("Reconnect"), key=f"reconnect_{run_dir.name}",
                             help=t("Collect output from the original job. Settings and convergence checks stay unchanged.")):
            try:
                restored = _with_password(resume, run_dir)
                if restored.get("status") not in TERMINAL:
                    st.success(t("Monitoring started (PID {pid}).").format(pid=_with_password(start_worker, run_dir)))
                else:
                    st.warning(restored.get("last_error") or t("Check the run history."))
            except Exception as exc:
                st.error(t("Cannot reconnect: {error}").format(error=exc))
    current_stage = stages[state.get("current_stage", 0)] if stages else {}
    has_job = bool(current_stage.get("job_id"))
    can_cancel = status not in {"succeeded", "cancelled"} and (
        (has_job and current_stage.get("scheduler_state") not in SLURM_TERMINAL)
        or (not has_job and status not in TERMINAL)
    )
    if can_cancel:
        if actions[1].button(t("Cancel calculation"), key=f"cancel_{run_dir.name}"):
            try:
                _with_password(cancel, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(t("Cancellation unconfirmed: {error}").format(error=exc))
    bundle_key = f"bundle_{run_dir}"
    if actions[2].button(t("Prepare download"), key=f"bundle_btn_{run_dir.name}"):
        try:
            st.session_state[bundle_key] = str(bundle_run(run_dir))
        except Exception as exc:
            st.error(t("Cannot prepare download: {error}").format(error=exc))
    if bundle_key in st.session_state:
        archive = Path(st.session_state[bundle_key])
        if archive.is_file():
            if archive.stat().st_size <= 100 * 1024 * 1024:
                st.download_button(t("Download results (.zip)"), archive.read_bytes(), file_name=archive.name,
                                   mime="application/zip", key=f"download_{run_dir.name}")
            else:
                st.caption(t("Archive saved to {path}").format(path=archive))
                if st.button(t("Open results folder"), key=f"open_results_{run_dir.name}"):
                    import platform
                    import subprocess
                    try:
                        if os.name == "nt":
                            os.startfile(str(run_dir))
                        else:
                            subprocess.Popen(["open" if platform.system() == "Darwin" else "xdg-open", str(run_dir)])
                    except OSError as exc:
                        st.error(str(exc))
    if status == "succeeded":
        st.info(t("Complete. Results are ready to download."))
    elif status == "needs_attention":
        st.info(t("Check the error before reconnecting. Settings are unchanged."))
    if not is_qe and status in {"needs_attention", "failed"}:
        _repair_panel(run_dir, model_settings)
    _result_dialogue(run_dir, model_settings, context)
    if not is_qe:
        _continue_panel(run_dir, model_settings, context)
    else:
        from vasp_slurm_agent.qe_app import followup_panel
        followup_panel(run_dir, model_settings, context)


def _continue_panel(run_dir, settings, context):
    from vasp_slurm_agent.continuation import inspect_continuation, prepare_task_continuation
    from vasp_slurm_agent.task_agent import draft_task

    state = read_state(run_dir)
    choices = [index for index in range(len(state["stages"]))
               if context["facts"].get(f"stage_{index + 1}.accepted", {}).get("value")]
    if not choices:
        return
    key = f"continue_{run_dir}"
    with st.expander(t("Continue")):
        index = st.selectbox(t("Starting stage"), choices, format_func=lambda i: state["stages"][i]["folder"], key=key + "_stage")
        goal = st.text_area(t("Next calculation"), key=key + "_goal")
        stamp = (index, goal, context["context_sha256"])
        saved = st.session_state.get(key)
        if saved and saved["stamp"] != stamp:
            st.session_state.pop(key, None)
            saved = None
        revision = st.text_input(t("Change the plan"), key=key + "_revision") if saved else ""
        if st.button(t("Plan next calculation"), disabled=not goal.strip(), key=key + "_plan"):
            try:
                source = inspect_continuation(run_dir, index)
                history = list(saved["plan"].get("dialogue", [])) if saved else []
                if revision.strip():
                    history.append({"role": "user", "content": revision.strip()})
                with st.spinner(t("Planning…")):
                    plan = draft_task(goal, source["structure_path"], settings, history=history,
                                      runs_root=run_dir.parent, previous_parameters=source["parameters"])
                saved = {"stamp": stamp, "source": source, "plan": plan, "revision": revision}
                st.session_state[key] = saved
            except Exception as exc:
                st.error(str(exc))
        if not saved:
            return
        plan = saved["plan"]
        st.write(plan["summary"])
        for question in plan["questions"]:
            st.info(question)
        _model_usage(plan.get("model_usage"))
        _task_preview(plan)
        _science_details(plan.get("scientific_report"))
        with st.expander(t("Plan details")):
            st.json({"operations": plan["operations"], "stages": plan["stages"], "variants": plan["variants"]})
        if st.button(t("Prepare continuation"), key=key + "_prepare", disabled=plan["status"] != "ready" or revision != saved.get("revision", "")):
            try:
                source = inspect_continuation(run_dir, index)
                if source["context_sha256"] != saved["source"]["context_sha256"]:
                    raise ValueError(t("The source results changed. Plan again."))
                destination = run_dir.parent / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
                prepare_task_continuation(run_dir, index, destination, ClusterConfig.load(run_dir / "config.json"),
                                          plan, saved["source"]["context_sha256"])
                st.session_state["active_run"] = str(destination)
                st.rerun()
            except Exception as exc:
                st.error(t("Cannot prepare continuation: {error}").format(error=exc))


def _batch_panel(batch_dir, settings, config=None, *, read_only=False):
    from vasp_slurm_agent.batch import (attach_batch_config, read_batch, summarize_batch, start_batch_worker,
                                        resume_batch, cancel_batch, bundle_batch)

    try:
        state = read_batch(batch_dir)
        summary = summarize_batch(batch_dir)
    except Exception as exc:
        st.error(t("Cannot read batch: {error}").format(error=exc))
        return
    attached = (batch_dir / "config.json").is_file()
    st.subheader(t("Comparison"))
    st.caption(str(batch_dir))
    rows = summary["rows"]
    st.dataframe([{key: row.get(key) for key in ("label", "status", "functional", "hubbard_u", "strain", "energy_ev_per_atom", "band_gap_ev", "relative_energy_mev_per_atom")}
                  for row in rows], hide_index=True)
    st.download_button(t("Download table"), (batch_dir / "summary.csv").read_bytes(), file_name="comparison.csv", mime="text/csv")
    with st.expander(t("Review settings"), expanded=state["status"] == "planned"):
        st.json(json.loads((batch_dir / "batch-plan.json").read_text()))
        if not attached:
            st.info(t("No cluster is attached to this batch yet. Submitting attaches the saved Cluster setup."))
    untouched = all(read_state(batch_dir / item["folder"])["status"] == "planned"
                    and not any(stage.get("job_id") for stage in read_state(batch_dir / item["folder"])["stages"])
                    for item in state["runs"])
    if read_only:
        if st.button(t("Prepare download"), key=f"project_batch_bundle_{batch_dir}"):
            bundle_batch(batch_dir)
        if (batch_dir / "results.zip").is_file():
            st.download_button(t("Download results (.zip)"), (batch_dir / "results.zip").read_bytes(), file_name="results.zip")
        selected = st.selectbox(t("Calculation"), state["runs"], format_func=lambda item: item["label"], key=f"project_batch_child_{batch_dir}")
        _run_panel(batch_dir / selected["folder"], settings, config, read_only=True)
        return
    if untouched and not state.get("cancel_requested"):
        reviewed = st.checkbox(t("Structures, settings and resources reviewed."), key=f"batch_review_{batch_dir}")
        if not attached and config is None:
            st.info(t("Save Cluster setup to submit this batch."))
        if st.button(t("Submit batch"), disabled=not reviewed or (not attached and config is None), key=f"batch_submit_{batch_dir}"):
            try:
                if not attached:
                    attach_batch_config(batch_dir, config)
                _with_password(start_batch_worker, batch_dir)
                st.success(t("Batch monitoring started."))
            except Exception as exc:
                st.error(t("Cannot start monitoring: {error}").format(error=exc))
    elif state["status"] not in TERMINAL:
        if st.button(t("Resume monitoring"), key=f"batch_watch_{batch_dir}"):
            _with_password(start_batch_worker, batch_dir)
            st.success(t("Batch monitoring started."))
    elif state["status"] in {"needs_attention", "failed"}:
        if st.button(t("Reconnect"), key=f"batch_reconnect_{batch_dir}"):
            restored = _with_password(resume_batch, batch_dir)
            if restored["status"] not in TERMINAL:
                _with_password(start_batch_worker, batch_dir)
            st.rerun()
    if state["status"] not in {"succeeded", "cancelled"}:
        if st.button(t("Cancel batch"), key=f"batch_cancel_{batch_dir}"):
            _with_password(cancel_batch, batch_dir)
            st.rerun()
    if st.button(t("Prepare download"), key=f"batch_bundle_{batch_dir}"):
        bundle_batch(batch_dir)
    if (batch_dir / "results.zip").is_file():
        st.download_button(t("Download results (.zip)"), (batch_dir / "results.zip").read_bytes(), file_name="results.zip")
    selected = st.selectbox(t("Calculation"), state["runs"], format_func=lambda item: item["label"], key=f"batch_child_{batch_dir}")
    _run_panel(batch_dir / selected["folder"], settings, config)


def run_rows(runs_root: Path) -> list[dict]:
    """One row per saved run or batch, newest first."""
    from vasp_slurm_agent.samples import sample_card

    rows = []
    if not runs_root.is_dir():
        return rows
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        try:
            if (path / "batch.json").is_file():
                state = json.loads((path / "batch.json").read_text())
                formula, task = t("batch"), t("comparison")
            elif (path / "run.json").is_file():
                state = read_state(path)
                formula, task = state.get("formula", ""), state.get("task", "")
            else:
                continue
        except (OSError, ValueError):
            continue
        stamp = str(state.get("updated_at") or state.get("created_at") or "")
        rows.append({"path": str(path), "folder": path.name, "formula": formula, "task": task,
                     "status": _status(state.get("status", "")), "updated": stamp[:16].replace("T", " "),
                     "sample": sample_card(path) is not None})
    rows.sort(key=lambda row: (row["updated"], row["folder"]), reverse=True)
    return rows


def _runs_tab(runs_root):
    from vasp_slurm_agent.samples import install_samples, list_samples

    rows = run_rows(runs_root)
    if rows:
        st.dataframe([{t("Material"): row["formula"], t("Task"): row["task"], t("Status"): row["status"],
                       t("Updated"): row["updated"], t("Folder"): row["folder"] + (" · " + t("sample") if row["sample"] else "")}
                      for row in rows], hide_index=True)
    else:
        st.info(t("No runs yet. Load the sample runs to explore completed results without a cluster, or prepare a new calculation."))
    labels = {row["path"]: f"{row['formula']} · {row['task']} · {row['status']} · {row['updated']}" for row in rows}
    selected = st.selectbox(t("Select a run"), list(labels), index=None, format_func=labels.get,
                            placeholder=t("Choose a run"))
    if st.button(t("View run"), disabled=selected is None):
        st.session_state["active_run"] = selected
    manual = st.text_input(t("Or enter a run folder"))
    if st.button(t("Open run"), disabled=not manual.strip()):
        st.session_state["active_run"] = str(Path(manual).expanduser())
    with st.expander(t("Sample runs"), expanded=not rows):
        st.caption(t("Explore completed calculations without a cluster."))
        for sample in list_samples():
            st.write(f"**{sample.get('title', sample['id'])}** — {sample.get('description', '')}")
        if st.button(t("Load sample runs")):
            try:
                installed = install_samples(runs_root)
            except OSError as exc:
                st.error(t("Cannot copy the sample runs: {error}").format(error=exc))
            else:
                if installed:
                    st.session_state["active_run"] = str(installed[0])
                    st.rerun()
                st.info(t("The sample runs are already in the run folder."))


def _worker_alive(run_dir):
    from vasp_slurm_agent.workflow import _acquire, _release

    try:
        with (run_dir / ".worker.lock").open("a") as lock:
            _acquire(lock, blocking=False)
            try:
                return False
            finally:
                _release(lock)
    except BlockingIOError:
        return True
    except OSError:
        return False


def _auto_resume(runs_root):
    """Restart monitoring of submitted runs once per session. Nothing new is submitted."""
    if st.session_state.get("auto_resume_done") or not runs_root.is_dir():
        return
    st.session_state["auto_resume_done"] = True
    resumed = []
    for path in sorted(runs_root.iterdir()):
        try:
            if (path / "batch.json").is_file() and (path / "config.json").is_file():
                from vasp_slurm_agent.batch import read_batch, start_batch_worker
                if read_batch(path).get("status") == "running" and not _worker_alive(path):
                    _with_password(start_batch_worker, path)
                    resumed.append(path.name)
            elif (path / "run.json").is_file() and has_config(path):
                state = read_state(path)
                if state.get("status") in ACTIVE and not _worker_alive(path):
                    _with_password(start_worker, path)
                    resumed.append(path.name)
        except Exception:
            continue
    if resumed:
        st.session_state["auto_resumed"] = resumed


def _preferences(prefs):
    with st.expander(t("Preferences")):
        values = {
            "auto_resume": st.checkbox(t("Resume monitoring at startup"), value=prefs["auto_resume"], key="pref_auto_resume",
                                       help=t("Restarts monitoring of submitted runs when the app opens. Nothing new is submitted.")),
            "notifications": st.checkbox(t("Desktop notification when a run finishes"), value=prefs["notifications"], key="pref_notifications"),
            "notify_webhook": st.text_input(t("Notification webhook URL (optional)"), value=prefs["notify_webhook"], key="pref_webhook",
                                            help=t("A JSON message is posted when a run finishes, for Slack, Discord, WeChat Work or similar.")),
            "interactive_viewer": st.checkbox(t("Interactive 3D structure viewer"), value=prefs["interactive_viewer"], key="pref_viewer",
                                              help=t("Loads 3Dmol.js from the internet. Untick for a static plot.")),
            "check_updates": st.checkbox(t("Check for new versions"), value=prefs["check_updates"], key="pref_updates",
                                         help=t("Reads the GitHub release list once a day. No usage data is sent.")),
        }
        if any(values[key] != prefs[key] for key in values):
            save_settings(values)
            st.session_state["prefs"] = load_settings()


def _language_selector(prefs):
    options = list(LANGUAGE_NAMES)
    current = st.session_state.get("language", prefs.get("language", "en"))
    if current not in options:
        current = "en"
    set_language(current)
    choice = st.selectbox("Language / 语言", options, index=options.index(current), format_func=LANGUAGE_NAMES.get,
                          key="language_choice")
    if choice != current:
        st.session_state["language"] = choice
        save_settings({"language": choice})
        st.session_state["prefs"] = load_settings()
        set_language(choice)
        st.rerun()


def _update_notice(prefs):
    from vasp_slurm_agent import updates

    if not prefs.get("check_updates"):
        return
    updates.start_background_check()
    info = updates.available_update()
    if info:
        st.caption(t("Version {version} is available.").format(version=info['latest']) + f" [{t('Release notes')}]({info['url']})")


def legacy_main() -> None:
    st.set_page_config(page_title="DFT Agent", page_icon="⚛", layout="wide")
    from vasp_slurm_agent.ui_style import apply_style
    apply_style()
    try:
        migrate_legacy_config()
    except OSError:
        pass
    prefs = _prefs()
    if requested_root := st.session_state.pop("requested_chat_runs_root", None):
        st.session_state["runs_root"] = requested_root
    with st.sidebar:
        if st.query_params.get("view") == "legacy" and st.button("Back to projects"):
            del st.query_params["view"]
            st.rerun()
        st.html('<div class="dfa-eyebrow">DFT Agent</div><div class="dfa-workspace">Workspace</div>')
        set_language(st.session_state.get("language", prefs.get("language", "en")))
        model_settings = _model_editor()
        local_settings = st.expander(t("Local settings"))
        with local_settings:
            _language_selector(prefs)
            base_path = Path(st.text_input(t("Configuration file"), value=str(default_config_path()), key="config_path")).expanduser()
            runs_root = Path(st.text_input(t("Run folder"), value=str(default_runs_root()), key="runs_root")).expanduser()
            st.text_input(t("SSH password (optional)"), type="password", key="ssh_password",
                          help=t("SSH keys are preferred. Passwords stay in memory and are never saved."))
            st.button(t("Clear password"), on_click=lambda: st.session_state.update(ssh_password=""))
        _preferences(prefs)
        if st.session_state.get("auto_resumed"):
            st.caption(t("Monitoring resumed for: ") + ", ".join(st.session_state["auto_resumed"]))
    base_config = _load_config(base_path)
    base_identity = str(base_path.resolve())
    if st.session_state.get("engine_base_path") != base_identity:
        st.session_state["engine_base_path"] = base_identity
        st.session_state["cluster_engine"] = base_config.engine if base_config else "vasp"
    if requested_engine := st.session_state.pop("requested_chat_engine", None):
        st.session_state["cluster_engine"] = requested_engine
    title, version = st.columns([5, 1], vertical_alignment="center")
    with title:
        st.title("DFT Agent")
    with version:
        st.html(f'<span class="dfa-version">{__version__}</span>')
    engine = st.radio("Engine", ("vasp", "qe"), horizontal=True,
                      format_func=lambda value: "Quantum ESPRESSO" if value == "qe" else "VASP",
                      key="cluster_engine", label_visibility="collapsed")
    config_path = base_path
    if base_config and base_config.engine != engine:
        config_path = base_path.with_name(f"{base_path.stem}.{engine}{base_path.suffix}")
    config = base_config if config_path == base_path else _load_config(config_path)
    profile_conflict = config is not None and config.engine != engine
    if profile_conflict:
        st.error(t("The configuration file belongs to another engine. Choose a different file in Local settings."))
        config = None
    profile_identity = (str(config_path.resolve()), engine)
    if st.session_state.get("active_engine_profile") != profile_identity:
        st.session_state["active_engine_profile"] = profile_identity
        st.session_state["qe_editor_version"] = st.session_state.get("qe_editor_version", 0) + 1
    with st.sidebar:
        st.html('<div class="dfa-section">Cluster</div>')
        st.caption(t("Settings saved") if config else t("Set up a cluster to submit calculations."))
    if config_path != base_path:
        with local_settings:
            st.caption(t("Active configuration"))
            st.code(str(config_path), language="text")
    st.session_state["qe_config_path"] = str(config_path)
    _update_notice(prefs)
    if prefs.get("auto_resume"):
        _auto_resume(runs_root)
    if requested_tab := st.session_state.pop("requested_workspace_tab", None):
        st.session_state["workspace_tab"] = requested_tab
    chat_tab, task_tab, history_tab, config_tab, memory_tab = st.tabs(
        [t("Chat"), t("New calculation"), t("Runs"), t("Cluster setup"), t("Memory")],
        key="workspace_tab", on_change="rerun")
    with config_tab:
        if not profile_conflict:
            _config_editor(config_path, config)
    with task_tab:
        _new_run(config, runs_root, model_settings)
    with memory_tab:
        _memory_panel(runs_root)
    with history_tab:
        _runs_tab(runs_root)
    with chat_tab:
        from vasp_slurm_agent.conversation_app import panel as chat_panel
        chat_panel(runs_root, engine, model_settings, config)
    if st.session_state.get("active_run"):
        st.divider()
        active = Path(st.session_state["active_run"])
        if (active / "batch.json").is_file():
            _batch_panel(active, model_settings, config)
        else:
            _run_panel(active, model_settings, config)


def main() -> None:
    if st.query_params.get("view") == "legacy" or os.environ.get("DFT_AGENT_UI") == "legacy":
        legacy_main()
    else:
        from vasp_slurm_agent.project_app import main as project_main
        project_main()


if __name__ == "__main__":
    main()
