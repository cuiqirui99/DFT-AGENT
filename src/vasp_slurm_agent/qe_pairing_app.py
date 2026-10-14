"""Pairing-model controls using the shared model and cluster connection."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import tempfile

import streamlit as st

from . import qe_pairing, qe_pairing_agent, qe_pairing_workflow
from .qe_app import _prepare_notice, _remote, _run_dir


def new_run(config, runs_root, settings):
    from .app import _model_usage

    st.subheader("Pairing model")
    st.caption("One spin-degenerate orbital · Hubbard RPA · no SOC · 1 CPU")
    upload = st.file_uploader("Wannier Hamiltonian", type=["dat"], key="pairing_hr")
    origin = st.selectbox("Source", ["model", "wannier_unvalidated"],
                         format_func=lambda x: {"model": "Model Hamiltonian", "wannier_unvalidated": "Wannier calculation"}[x],
                         key="pairing_origin")
    runtime = st.text_input("Cluster Python", placeholder="/path/to/python", key="pairing_python",
                           help="Use an existing Python environment with NumPy and SciPy.")
    st.caption("A Wannier file alone does not validate a material model.")
    if config is None or config.engine != "qe":
        st.info("Save Quantum ESPRESSO cluster settings first.")
        return
    config = replace(config, compute_backend="cpu", nodes=1, tasks=1, cpus_per_task=1)
    if st.button("Check Python", disabled=not runtime.strip(), key="pairing_doctor"):
        try:
            with st.spinner("Checking…"):
                report = _remote(qe_pairing_workflow.doctor, config, runtime.strip())
            st.json(report)
        except Exception as exc:
            st.error(str(exc))
    mode = st.radio("Mode", ["Agent", "Manual"], horizontal=True, key="pairing_mode")
    if upload is None:
        return
    content = upload.getvalue()
    fingerprint = hashlib.sha256(content).hexdigest() + ":" + origin
    with tempfile.TemporaryDirectory(prefix="dft-pairing-ui-") as temporary:
        hr = Path(temporary) / "hamiltonian_hr.dat"
        hr.write_bytes(content)
        try:
            qe_pairing.read_hoppings(hr)
        except (ValueError, IndexError, OSError) as exc:
            st.error(str(exc))
            return
        if mode == "Agent":
            goal = st.text_area("Goal", key="pairing_goal")
            if st.button("Plan", disabled=not goal.strip(), key="pairing_plan_btn"):
                try:
                    previous = st.session_state.get("pairing_plan", {})
                    history = previous.get("dialogue", []) if st.session_state.get("pairing_source") == fingerprint else []
                    with st.spinner("Planning…"):
                        plan = qe_pairing_agent.draft_task(goal, hr, settings, origin=origin, history=history)
                    st.session_state.update(pairing_plan=plan, pairing_source=fingerprint)
                except Exception as exc:
                    st.error(str(exc))
            plan = st.session_state.get("pairing_plan")
            if not plan:
                return
            st.write(plan["summary"])
            _model_usage(plan.get("model_usage"))
            for question in plan["questions"]:
                st.info(question)
            with st.expander("Settings", expanded=True):
                st.json(plan["parameters"])
            for limit in plan.get("limits", []):
                st.caption(limit)
            changed = st.session_state.get("pairing_source") != fingerprint or plan.get("goal") != goal
            if changed:
                st.info("The input or request changed. Plan again.")
            if st.button("Prepare inputs", key="pairing_prepare", disabled=changed or plan["status"] != "ready" or not runtime.strip()):
                try:
                    target = _run_dir(runs_root)
                    qe_pairing_agent.prepare_task(hr, target, config, plan, python_executable=runtime.strip())
                    _prepare_notice(target)
                except Exception as exc:
                    st.error(str(exc))
        else:
            recipe = st.text_area("Model settings (JSON)", "{}", key="pairing_manual")
            if st.button("Prepare inputs", disabled=not runtime.strip(), key="pairing_manual_prepare"):
                try:
                    target = _run_dir(runs_root)
                    qe_pairing_workflow.prepare_run(hr, target, config, json.loads(recipe),
                        python_executable=runtime.strip(), origin=origin)
                    _prepare_notice(target)
                except Exception as exc:
                    st.error(str(exc))


def run_panel(run_dir, settings):
    from .app import _result_dialogue
    from .workflow import read_state, start_worker, TERMINAL

    state = read_state(run_dir)
    context = qe_pairing_workflow.load_run_context(run_dir)
    status = context["status"]
    st.subheader("Pairing model")
    st.write("Status: **" + status.replace("_", " ") + "**")
    st.caption("One-band Hubbard RPA. This result is not a material Tc.")
    with st.expander("Settings"):
        st.json(state["stages"][0]["parameters"])
    if state.get("last_error"):
        st.error(state["last_error"])
    if context["facts"].get("stage_1.accepted", {}).get("value"):
        facts = context["facts"]
        rows = [{"Channel": channel, "Eigenvalue": facts[f"stage_1.{channel}.eigenvalue"]["value"],
                 "Residual": facts[f"stage_1.{channel}.relative_residual"]["value"]}
                for channel in ("singlet", "triplet") if f"stage_1.{channel}.eigenvalue" in facts]
        st.dataframe(rows, hide_index=True)
        st.write("Temperature (K):", facts["stage_1.temperature_k"]["value"])
    controls = st.columns(3)
    has_job = any(stage.get("job_id") for stage in state["stages"])
    if status == "planned" and not has_job:
        reviewed = st.checkbox("Model, settings and resources reviewed.", key=f"pairing_review_{run_dir}")
        if controls[0].button("Submit calculation", disabled=not reviewed, key=f"pairing_submit_{run_dir}"):
            try:
                _remote(start_worker, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    elif status not in TERMINAL:
        if controls[0].button("Resume monitoring", key=f"pairing_watch_{run_dir}"):
            try:
                _remote(start_worker, run_dir)
            except Exception as exc:
                st.error(str(exc))
    elif status in {"needs_attention", "failed"}:
        if controls[0].button("Reconnect", key=f"pairing_reconnect_{run_dir}"):
            try:
                restored = _remote(qe_pairing_workflow.resume, run_dir)
                if restored["status"] not in TERMINAL:
                    _remote(start_worker, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if status not in {"succeeded", "cancelled", "failed"}:
        if controls[1].button("Cancel calculation", key=f"pairing_cancel_{run_dir}"):
            try:
                _remote(qe_pairing_workflow.cancel, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if controls[2].button("Prepare download", key=f"pairing_bundle_{run_dir}"):
        try:
            st.session_state[f"pairing_download_{run_dir}"] = str(qe_pairing_workflow.bundle_run(run_dir))
        except Exception as exc:
            st.error(str(exc))
    bundle = st.session_state.get(f"pairing_download_{run_dir}")
    if bundle and Path(bundle).is_file():
        st.download_button("Download results (.zip)", Path(bundle).read_bytes(), file_name=Path(bundle).name,
                           mime="application/zip", key=f"pairing_zip_{run_dir}")
    _result_dialogue(run_dir, settings, context)
