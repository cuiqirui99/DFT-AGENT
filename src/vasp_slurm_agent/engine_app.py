"""Checked results for ABINIT, CP2K and Elk stages in the run panel."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from .i18n import t

LABELS = {"relax": "Relaxation", "scf": "SCF", "bands": "Band structure", "dos": "Density of states",
          "optics": "Optical spectrum", "tddft": "Laser-pulse response", "md": "Molecular dynamics", "phonons": "Phonons"}
PROGRAMS = {"abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk"}


def stage_results(run_dir, stage):
    """Show checked numerical results; plots and raw files follow in the downloads panel."""
    result = stage.get("result") or {}
    if not result:
        return
    folder = Path(run_dir) / stage["folder"] / "outputs"
    if not folder.resolve().is_relative_to(Path(run_dir).resolve()):
        st.error(t("Invalid result folder."))
        return
    program = PROGRAMS.get((stage.get("metadata") or {}).get("engine"), "")
    with st.expander(f"{program} · {t(LABELS.get(stage.get('name'), stage.get('name', 'Results')))}", expanded=True):
        if not result.get("success"):
            st.warning(result.get("reason", t("Results need attention.")))
            return
        for warning in result.get("warnings", []):
            st.warning(warning)
        for note in (stage.get("metadata") or {}).get("notes", []):
            st.caption(note)
        columns = st.columns(3)
        values = [(t("Total energy"), result.get("final_energy_ev"), "eV"),
                  (t("Fermi energy"), result.get("fermi_energy_ev"), "eV"),
                  (t("Band gap"), result.get("band_gap_ev"), "eV"),
                  (t("Total moment"), result.get("total_magnetization_mu_b"), "μB"),
                  (t("Maximum force"), result.get("final_max_force_ev_angstrom"), "eV/Å"),
                  (t("Pressure"), result.get("pressure_gpa"), "GPa")]
        shown = [item for item in values if item[1] is not None]
        for index, (label, value, unit) in enumerate(shown):
            columns[index % 3].metric(label, f"{value:.6g} {unit}")
        gap = result.get("band_gap") or {}
        if gap.get("metallic_or_smeared"):
            st.caption(t("Partially occupied or overlapping sampled states. Check smearing before assigning a gap."))
        elif gap.get("available"):
            st.caption(t("Gap from sampled k points only; not a converged full-zone gap."))
        spectra = (result.get("optics") or {}).get("spectra") or {}
        if spectra:
            st.dataframe([{t("Spectrum"): label, t("Strongest absorption (eV)"): entry.get("absorption_peak_ev"),
                           t("Dielectric constant"): entry.get("dielectric_constant")} for label, entry in spectra.items()],
                         hide_index=True)
        evolution = result.get("time_evolution") or {}
        if evolution:
            items = [(t("Absorbed energy"), evolution.get("absorbed_energy_ev"), "eV"),
                     (t("Peak field"), evolution.get("peak_field_v_per_angstrom"), "V/Å"),
                     (t("Moment change"), evolution.get("moment_change_percent"), "%")]
            shown = [item for item in items if item[1] is not None]
            for column, (label, value, unit) in zip(st.columns(max(1, len(shown))), shown):
                column.metric(label, f"{value:.4g} {unit}")
        phonons = result.get("phonons") or {}
        if phonons:
            items = [(t("Highest frequency"), phonons.get("max_frequency_cm1"), "cm⁻¹"),
                     (t("Lowest frequency"), phonons.get("min_frequency_cm1"), "cm⁻¹"),
                     (t("Dielectric constant"), phonons.get("dielectric_constant"), "")]
            shown = [item for item in items if item[1] is not None]
            for column, (label, value, unit) in zip(st.columns(len(shown)), shown):
                column.metric(label, f"{value:.4g} {unit}".strip())
            if phonons.get("born_charges"):
                st.dataframe([{t("Atom"): item["atom"], t("Element"): item["element"], t("Born charge"): round(item["born_charge"], 3)}
                              for item in phonons["born_charges"]], hide_index=True)
        dynamics = result.get("molecular_dynamics") or {}
        if dynamics:
            items = [(t("Mean temperature"), dynamics.get("mean_temperature_k"), "K"),
                     (t("Simulated time"), dynamics.get("simulated_time_fs"), "fs"),
                     (t("Energy drift"), dynamics.get("conserved_drift_mev_per_atom_ps"), "meV/atom/ps")]
            for column, (label, value, unit) in zip(st.columns(3), items):
                column.metric(label, f"{value:.4g} {unit}")
        if result.get("site_magnetization"):
            st.dataframe(result["site_magnetization"], hide_index=True)
            if result.get("site_magnetization_scope"):
                st.caption(result["site_magnetization_scope"])
        final = folder / "final.cif"
        if final.is_file() and stage.get("name") == "relax":
            from pymatgen.core import Structure
            from .app import _structure_view
            with st.expander(t("Final structure")):
                _structure_view(Structure.from_file(final))
            st.download_button(t("Download structure (.cif)"), final.read_bytes(), file_name="final.cif",
                               key=f"engine_structure_{run_dir}_{stage['folder']}")
