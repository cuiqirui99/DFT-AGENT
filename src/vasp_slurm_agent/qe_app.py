"""Quantum ESPRESSO setup, planning and results in the desktop app."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import streamlit as st

from .config import ClusterConfig
from . import qe_agent, qe_setup, qe_workflow


LABELS = {"relax": "Relax atoms", "vc-relax": "Relax atoms and cell", "scf": "SCF", "nscf": "NSCF", "electron_phonon": "Electron–phonon coupling", "bands": "Band structure", "dos": "Density of states", "pdos": "Projected DOS", "phonon": "Phonons at one q point", "phonon_bands": "Phonon dispersion", "neb": "Reaction path (NEB)", "pp": "Charge density"}
LABELS.update(epw_iso="Eliashberg gap", epw_tc="Eliashberg Tc")
LABELS.update(wannier_electron_phonon="Wannier electron–phonon coupling", epw_aniso="Anisotropic gap")


def _remote(function, *args, **kwargs):
    from .transport import password_context
    with password_context(st.session_state.get("ssh_password", "")):
        return function(*args, **kwargs)


def _save_config(config, path=None):
    path = path or st.session_state.get("qe_config_path")
    if not path:
        raise ValueError("Save Cluster setup before installing pseudopotentials.")
    config.save(path)
    st.session_state["qe_editor_version"] = st.session_state.get("qe_editor_version", 0) + 1


def config_editor(path, config):
    """Save cluster settings; install and probe only on explicit button presses."""
    path = Path(path).expanduser()
    st.session_state["qe_config_path"] = str(path)
    version = st.session_state.get("qe_editor_version", 0)
    key = f"qe_config_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}_{version}"
    value = lambda name, fallback: getattr(config, name, fallback) if config else fallback
    st.subheader("Quantum ESPRESSO setup")
    st.caption("Connect to your Slurm cluster, then use its QE installation or install one below.")
    with st.form(key):
        left, right = st.columns(2)
        host = left.text_input("SSH host", value("host", ""))
        user = right.text_input("SSH user", value("user", ""))
        remote_root = st.text_input("Run folder on cluster", value("remote_root", ""), placeholder="/home/your-user/dft-agent-runs")
        partition = left.text_input("Partition", value("partition", ""))
        account = right.text_input("Account", value("account", ""))
        bin_dir = st.text_input("QE bin folder", value("qe_bin_dir", ""), help="Leave empty when pw.x is already on the cluster PATH.")
        pseudo_default = value("qe_pseudo_root", "") or (value("remote_root", "").rstrip("/") + "/pseudos" if value("remote_root", "") else "")
        pseudo_root = st.text_input("UPF folder on cluster", pseudo_default)
        with st.expander("Advanced"):
            command = st.text_input("Launch command", value("qe_command", "srun pw.x"))
            tasks = st.number_input("MPI tasks", min_value=1, max_value=4096, value=int(value("tasks", 8)))
            nodes = st.number_input("Nodes", min_value=1, max_value=4096, value=int(value("nodes", 1)))
            cpus = st.number_input("CPUs per task", min_value=1, max_value=256, value=int(value("cpus_per_task", 1)))
            walltime = st.text_input("Time limit", value("walltime", "00:30:00"))
            setup = st.text_area("Environment commands", "\n".join(value("setup_commands", [])), placeholder="module load your-compiler your-mpi")
            extra = st.text_area("Extra Slurm options", "\n".join(value("extra_sbatch", [])), placeholder="--mem=8G")
            port = st.number_input("SSH port", min_value=1, max_value=65535, value=int(value("port", 22)))
            socket = st.text_input("SSH control socket", value("ssh_control_path", ""), help="Optional: reuse an SSH connection authenticated with two-factor login.")
            mapping = st.text_area("UPF mapping", json.dumps(value("qe_pseudopotentials", {}), indent=2), help="Element to UPF filename. The public library fills this automatically.")
        saved = st.form_submit_button("Save settings", type="primary")
    if saved:
        try:
            fields = dict(engine="qe", host=host.strip(), user=user.strip(), remote_root=remote_root.strip(), partition=partition.strip(), account=account.strip(), qe_bin_dir=bin_dir.strip(), qe_pseudo_root=pseudo_root.strip(), qe_command=command.strip(), tasks=int(tasks), nodes=int(nodes), cpus_per_task=int(cpus), walltime=walltime.strip(), setup_commands=[x for x in setup.splitlines() if x.strip()], extra_sbatch=[x for x in extra.splitlines() if x.strip()], port=int(port), ssh_control_path=socket.strip(), qe_pseudopotentials=json.loads(mapping), compute_backend="cpu")
            updated = replace(config, **fields) if config else ClusterConfig(**fields)
            _save_config(updated, path)
            st.success("Settings saved.")
            st.rerun()
        except (ValueError, OSError, TypeError) as exc:
            st.error(str(exc))
    if not config or config.engine != "qe":
        st.info("Save your connection settings to continue.")
        return
    if st.button("Check cluster", key=key + "_check"):
        try:
            with st.spinner("Checking…"):
                report = _remote(qe_setup.doctor, config)
            st.dataframe([{"Check": name, "Ready": item["ok"], "Details": item["detail"]} for name, item in report["checks"].items()], hide_index=True)
        except Exception as exc:
            st.error(str(exc))
    receipt_path = path.with_name(path.stem + ".qe-install.json")
    with st.expander("Install Quantum ESPRESSO"):
        st.caption("Builds the CPU/MPI version in your cluster account. No administrator access is needed.")
        jobs = st.number_input("Build CPUs", 1, 64, min(config.tasks, 8), key=key + "_build_cpus")
        include_epw = st.checkbox("Include EPW", key=key + "_epw")
        try:
            plan = qe_setup.install_plan(config, build_jobs=int(jobs), include_epw=include_epw)
            st.write(f"QE {plan['version']} · {plan['build_jobs']} CPUs · {plan['walltime']}")
            st.code(plan["prefix"], language="text")
            reviewed = st.checkbox("Use this account and allocation to build QE.", key=key + "_install_review")
            if st.button("Install QE", disabled=not reviewed, key=key + "_install"):
                with st.spinner("Preparing the build…"):
                    receipt = _remote(qe_setup.start_install, config, plan)
                receipt_path.write_text(json.dumps({**receipt, "plan": plan}, indent=2) + "\n")
                receipt_path.chmod(0o600)
                st.success("Build submitted. Check its progress below.")
        except Exception as exc:
            st.error(str(exc))
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text())
                st.write("Build: " + str(receipt.get("job_id", receipt.get("status", "unknown"))))
                if st.button("Check installation", key=key + "_install_check"):
                    status = _remote(qe_setup.poll_install, config, receipt)
                    st.write(status.get("status", "unknown"))
                    if status.get("error"):
                        st.error(status["error"])
                    if status.get("status") == "installed":
                        updated = replace(config, qe_bin_dir=receipt["plan"]["bin_dir"])
                        _save_config(updated, path)
                        st.success("QE installed. Its executable folder has been saved.")
                    with st.expander("Build details"):
                        st.json(status)
            except Exception as exc:
                st.error(str(exc))


@contextmanager
def _uploads(upload, endpoints):
    with tempfile.TemporaryDirectory(prefix="dft-qe-input-") as directory:
        folder = Path(directory)
        source = folder / ("structure.cif" if upload.name.lower().endswith(".cif") else "structure.vasp")
        source.write_bytes(upload.getvalue())
        images = [source] if endpoints else []
        for i, image in enumerate(endpoints):
            path = folder / (f"image-{i + 1}.cif" if image.name.lower().endswith(".cif") else f"image-{i + 1}.vasp")
            path.write_bytes(image.getvalue())
            images.append(path)
        yield source, images


def _manual_recipes(tasks, text, images):
    recipes = json.loads(text) if text.strip() else [{"task": task, "parameters": {}} for task in tasks]
    if not isinstance(recipes, list) or not recipes or len(recipes) > 20:
        raise ValueError("Use a list of 1–20 calculation stages.")
    for recipe in recipes:
        if not isinstance(recipe, dict) or set(recipe) != {"task", "parameters"} or recipe["task"] not in LABELS or not isinstance(recipe["parameters"], dict):
            raise ValueError('Each stage needs a supported "task" and a "parameters" object.')
        if recipe["task"] == "neb":
            if len(images) < 2:
                raise ValueError("Upload a final structure for NEB.")
            if "images" in recipe["parameters"]:
                raise ValueError("Supply NEB images with the upload field.")
            recipe["parameters"]["images"] = [str(x) for x in images]
    return recipes


def _prepare_notice(run_dir):
    st.session_state["active_run"] = str(run_dir)
    st.success("Inputs ready. Review the run, then submit.")


def _run_dir(runs_root):
    return Path(runs_root).expanduser() / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-qe-" + uuid.uuid4().hex[:6])


def new_run(upload, structure, config, runs_root, settings, mode):
    if upload is None or structure is None:
        st.info("Choose a structure first.")
        return
    if config is None or config.engine != "qe":
        st.info("Save Quantum ESPRESSO settings in Cluster setup first.")
        return
    elements = list(dict.fromkeys(site.specie.symbol for site in structure))
    missing = [element for element in elements if element not in config.qe_pseudopotentials]
    with st.expander("Pseudopotentials", expanded=bool(missing)):
        st.write(" · ".join(elements))
        if missing:
            st.info("Missing UPFs: " + ", ".join(missing))
        else:
            st.caption("Using the UPFs in your cluster settings.")
        library = st.selectbox("Library", ["SSSP", "PseudoDojo (NC)", "PseudoDojo FR (NC, spin-orbit)"], key="qe_pseudo_library")
        st.caption("Choose NC for EPW. All libraries use PBE; SOC needs the fully relativistic PseudoDojo FR set.")
        if st.button("Install public UPFs", key="qe_install_pseudos"):
            try:
                with st.spinner("Installing pseudopotentials…"):
                    installed = _remote(qe_setup.ensure_pseudos, config, elements,
                                        library={"PseudoDojo (NC)": "pseudodojo", "PseudoDojo FR (NC, spin-orbit)": "pseudodojo-fr"}.get(library, "sssp"))
                mappings = dict(config.qe_pseudopotentials)
                for element, item in installed["elements"].items():
                    mappings[element] = {"filename": item["filename"], "element": element, "ecutwfc": item["cutoff_wfc"], "ecutrho": item["cutoff_rho"], "sha256": item["sha256"], **item.get("header", {})}
                _save_config(replace(config, qe_pseudopotentials=mappings))
                st.success("UPFs installed and saved.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    endpoints = st.file_uploader("NEB images (optional)", accept_multiple_files=True, key="qe_images", help="The main structure is the first image. Add intermediate images and the final structure in order.") or []
    stamp = hashlib.sha256(upload.getvalue() + b"".join(image.getvalue() for image in endpoints) + json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    if st.session_state.get("qe_structure_stamp") != stamp:
        for key in ("qe_proposal", "qe_history", "qe_revision"):
            st.session_state.pop(key, None)
        st.session_state["qe_structure_stamp"] = stamp
    if mode == "Manual":
        tasks = st.multiselect("Calculations, in order", list(LABELS), default=["relax", "bands"], format_func=LABELS.get, key="qe_manual_tasks")
        st.caption("SCF and NSCF prerequisites are added automatically.")
        with st.expander("Advanced parameters"):
            text = st.text_area("Stage recipes (JSON)", "", height=180, placeholder='[{"task":"relax","parameters":{"ecutwfc":60}},{"task":"bands","parameters":{"nbnd":16}}]', key="qe_manual_recipe")
            st.caption("QE cutoffs use Ry. Native options belong in namelists.")
            st.caption("[EPW example](https://github.com/cuiqirui99/DFT-AGENT/blob/HEAD/docs/qe-superconductivity.md#mgb2-example)")
        if st.button("Prepare inputs", disabled=bool(missing) or not tasks and not text.strip(), type="primary", key="qe_manual_prepare"):
            try:
                with _uploads(upload, endpoints) as (source, images):
                    recipes = _manual_recipes(tasks, text, images)
                    run_dir = _run_dir(runs_root)
                    qe_workflow.prepare_plan(source, run_dir, config, [x["task"] for x in recipes], stage_parameters=[x["parameters"] for x in recipes])
                _prepare_notice(run_dir)
            except Exception as exc:
                st.error(str(exc))
        return
    goal = st.text_area("Goal", max_chars=20000, key="qe_goal", placeholder="What would you like to calculate?")
    if st.session_state.get("qe_goal_value") != goal:
        for key in ("qe_proposal", "qe_history", "qe_revision"):
            st.session_state.pop(key, None)
        st.session_state["qe_goal_value"] = goal
    if st.session_state.pop("qe_revision_reset", False):
        st.session_state["qe_revision"] = ""
    plan = st.session_state.get("qe_proposal")
    revision = st.text_input("Change the plan", key="qe_revision") if plan else ""
    if st.button("Plan", disabled=not goal.strip() or settings is None or bool(missing), type="primary", key="qe_plan"):
        try:
            history = list(st.session_state.get("qe_history", []))
            if revision.strip():
                history.append({"role": "user", "content": revision.strip()})
            with _uploads(upload, endpoints) as (source, images), st.spinner("Planning…"):
                plan = qe_agent.draft_task(goal.strip(), source, settings, history=history, config=config, images=images, runs_root=runs_root)
            st.session_state["qe_proposal"] = plan
            st.session_state["qe_history"] = plan["dialogue"]
            st.session_state["qe_revision_reset"] = True
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if not plan:
        return
    from .app import _model_usage
    _model_usage(plan.get("model_usage"))
    st.write(plan["summary"])
    for question in plan.get("questions", []):
        st.info(question)
    for index, stage in enumerate(plan.get("stages", []), 1):
        st.write(f"{index}. {LABELS.get(stage['task'], stage['task'])}")
    with st.expander("Plan details"):
        st.json({key: plan[key] for key in ("operations", "stages", "notes", "references") if key in plan})
    if st.button("Prepare inputs", disabled=plan.get("status") != "ready" or bool(plan.get("questions")) or bool(revision.strip()), key="qe_agent_prepare"):
        try:
            with _uploads(upload, endpoints) as (source, images):
                run_dir = _run_dir(runs_root)
                qe_agent.prepare_task(source, run_dir, config, plan, images=images)
            _prepare_notice(run_dir)
        except Exception as exc:
            st.error(str(exc))


def stage_results(run_dir, stage):
    """Show checked numerical results and downloadable tables/plots."""
    result = stage.get("result") or {}
    if not result:
        return
    folder = Path(run_dir) / stage["folder"] / "outputs"
    if not folder.resolve().is_relative_to(Path(run_dir).resolve()):
        st.error("Invalid result folder.")
        return
    with st.expander(LABELS.get(stage.get("name"), stage.get("name", "Results")), expanded=True):
        if not result.get("success"):
            st.warning(result.get("reason", "Results need attention."))
            return
        for warning in result.get("warnings", []):
            st.warning(warning)
        if result.get("warning"):
            st.warning(result["warning"])
        if result.get("endpoint_relaxed") is False and not any("endpoint" in warning.lower() for warning in result.get("warnings", [])):
            st.warning("The endpoints are not relaxed. Treat this barrier as provisional.")
        values = [("Energy", result.get("final_energy_ev"), "eV"), ("Band gap", result.get("band_gap_ev"), "eV"), ("Total moment", result.get("total_magnetization_mu_b"), "μB"), ("Forward barrier", result.get("barrier_forward_ev"), "eV"), ("Lowest frequency", result.get("min_frequency_cm1"), "cm⁻¹")]
        for label, value, unit in values:
            if value is not None:
                st.metric(label, f"{value:.6g} {unit}")
        gap = result.get("band_gap") or {}
        if gap.get("metallic_or_smeared"):
            st.caption("Partially occupied or overlapping sampled states. Check smearing before assigning a gap.")
        if result.get("site_magnetization"):
            st.dataframe(result["site_magnetization"], hide_index=True)
        if result.get("electron_phonon"):
            fields = ("degauss_ry", "lambda", "omega_log_kelvin", "mu_star", "tc_estimate_kelvin")
            st.dataframe([{key: row[key] for key in fields if key in row} for row in result["electron_phonon"]], hide_index=True)
            st.caption("Tc is a phonon-mediated estimate for the selected μ*. Check mesh and broadening convergence.")
        if result.get("temperature_results"):
            fields = ("temperature_kelvin", "eigenvalue", "gap_iw0_mev", "gap_min_iw0_mev", "gap_max_iw0_mev", "renormalization_iw0", "residual")
            st.dataframe([{key: row[key] for key in fields if key in row} for row in result["temperature_results"]], hide_index=True)
            bracket = result.get("tc_bracket_kelvin")
            if bracket:
                st.metric("Tc bracket", f"{bracket[0]:g}–{bracket[1]:g} K")
            st.caption("Anisotropic Eliashberg model. Check interpolation, grids and frequency cutoff." if stage["name"] == "epw_aniso"
                       else "Isotropic Eliashberg model. Check the spectrum and frequency cutoff.")
        final = folder / "final.cif"
        if final.is_file():
            from pymatgen.core import Structure
            from .app import _structure_view
            with st.expander("Final structure"):
                _structure_view(Structure.from_file(final))
            st.download_button("Download structure", final.read_bytes(), file_name="final.cif", key=f"qe_structure_{run_dir}_{stage['folder']}")


def followup_panel(run_dir, settings, context):
    from . import qe_reasoning
    from .app import _model_usage

    key = "qe_next_" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    with st.expander("Continue this task"):
        question = st.text_area("What next?", key=key + "_question", max_chars=20000)
        if st.button("Plan next step", disabled=settings is None or not question.strip(), key=key + "_plan"):
            try:
                with st.spinner("Reviewing results…"):
                    decision = qe_reasoning.draft_next_action(question.strip(), run_dir, settings,
                        history=st.session_state.get(key + "_history", []))
                st.session_state[key + "_decision"] = decision
                st.session_state[key + "_history"] = decision["dialogue"]
                st.session_state[key + "_request"] = question
            except Exception as exc:
                st.error(str(exc))
        decision = st.session_state.get(key + "_decision")
        if not decision:
            return
        st.write(decision["summary"])
        _model_usage(decision.get("model_usage"))
        for item in decision.get("questions", []):
            st.info(item)
        with st.expander("Details"):
            st.json({name: decision.get(name) for name in ("action", "continuation", "repair_action", "evidence", "limits")})
        if decision.get("action") in {"continue", "repair"}:
            changed = (decision.get("context_sha256") != context["context_sha256"]
                       or st.session_state.get(key + "_request") != question)
            if changed:
                st.info("The request or results changed. Plan again.")
            if st.button("Prepare next step", disabled=changed or decision.get("status") != "ready", key=key + "_prepare"):
                try:
                    target = _run_dir(Path(run_dir).parent)
                    qe_reasoning.prepare_next_action(run_dir, target, decision)
                    _prepare_notice(target)
                except Exception as exc:
                    st.error(str(exc))
