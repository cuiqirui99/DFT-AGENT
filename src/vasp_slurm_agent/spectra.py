"""Band-structure and DOS figures for QE, ABINIT, CP2K and Elk, drawn like the VASP ones.

Energies arrive relative to their reference (the Fermi level or the valence band maximum);
the CSV files written next to the figures keep every band and every energy.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

WINDOW_EV = (-8.0, 8.0)
SPINS = (("#2366a8", "-", "Spin ↑"), ("#c26a24", "--", "Spin ↓"))
REFERENCES = {"fermi_level": r"$E - E_\mathrm{F}$ (eV)", "valence_band_maximum": r"$E - E_\mathrm{VBM}$ (eV)"}


def tick_label(label):
    """GAMMA as Γ and seekpath subscripts (H_0) as subscripts, for each side of a jump such as X|R."""
    return "|".join("Γ" if part == "GAMMA" else re.sub(r"_([0-9]+)", r"$_{\1}$", part) for part in label.split("|"))


def path_ticks(x, labels):
    """Tick positions and labels; two vertices at one position, a jump in the path, read U|K."""
    ticks = {}
    for position, label in zip(x, labels or []):
        if label:
            previous = ticks.get(float(position))
            ticks[float(position)] = label if not previous or previous == label else f"{previous}|{label}"
    return ticks


def _axes(size):
    from matplotlib.figure import Figure

    figure = Figure(figsize=size)
    ax = figure.subplots()
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=11, colors="#253247")
    for spine in ax.spines.values():
        spine.set_color("#8792a2")
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    return figure, ax


def _save(figure, folder, stem):
    from matplotlib import rc_context

    figure.tight_layout(pad=1.2)
    names = []
    with rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "dft-agent"}):
        for extension in ("png", "pdf", "svg"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None} if extension == "svg" else None
            figure.savefig(Path(folder) / f"{stem}.{extension}", dpi=240, facecolor="white", bbox_inches="tight", metadata=metadata)
            names.append(f"{stem}.{extension}")
    return names


def band_reference(gap, fermi):
    """An insulator's Fermi level may sit anywhere in its gap, so its bands start at the valence
    band maximum; other bands are drawn against the Fermi level. Returns (energy in eV, kind)."""
    if gap.get("status") == "gap" and gap.get("vbm_ev") is not None:
        return float(gap["vbm_ev"]), "valence_band_maximum"
    if fermi is not None:
        return float(fermi), "fermi_level"
    return float(gap.get("vbm_ev") or 0.0), "valence_band_maximum"


def band_window(energies):
    """The plotted energy range: the VASP window, cut 0.1 eV below the highest computed band.
    Bands above it were not computed, and that band follows whichever uncomputed band crosses
    it; the margin keeps its lowest point from tracing the frame (GaAs HSE06 at Γ)."""
    top = float(np.min(np.asarray(energies, dtype=float)[..., -1])) - 0.1
    return WINDOW_EV[0], min(WINDOW_EV[1], max(top, 1.0))


def dos_window(energies, densities):
    """The VASP window, cut where the computed states end: above them the DOS is zero only
    because no bands were computed there."""
    values = np.asarray(densities, dtype=float).reshape(len(energies), -1)
    filled = np.asarray(energies)[values.max(axis=1) > 1e-3 * values.max()] if values.max() > 0 else np.asarray(energies)
    return WINDOW_EV[0], min(WINDOW_EV[1], max(float(filled.max()), 1.0))


def plot_bands(folder, stem, x, energies, labels, segments, reference="fermi_level"):
    """Band figure from energies shaped (spin, k, band) or (k, band); returns the file names."""
    values = np.asarray(energies, dtype=float)
    values = values[None] if values.ndim == 2 else values
    x = np.asarray(x, dtype=float)
    figure, ax = _axes((8.0, 5.6))
    handles = []
    for spin in range(values.shape[0]):
        color, style, _ = SPINS[spin]
        for start, stop in segments or [[0, len(x)]]:
            lines = ax.plot(x[start:stop], values[spin, start:stop], color=color, linestyle=style, linewidth=1.15, alpha=0.95)
        handles.append(lines[0])
    ticks = path_ticks(x, labels)
    ax.set_xticks(list(ticks), [tick_label(label) for label in ticks.values()])
    for position in ticks:
        ax.axvline(position, color="0.85", linewidth=0.6, zorder=0)
    if len(handles) > 1:
        ax.legend(handles, [SPINS[0][2], SPINS[1][2]], frameon=False, fontsize=10)
    ax.set(xlabel="", ylabel=REFERENCES[reference], xlim=(float(x[0]), float(x[-1])), ylim=band_window(values))
    ax.axhline(0, color="#687588", linewidth=0.85, linestyle="--", zorder=0)
    return _save(figure, folder, stem)


def plot_phonons(folder, stem, x, frequencies, labels, segments, dos=None):
    """Phonon dispersion in cm⁻¹ along the band path, with the phonon DOS beside it when given as
    (frequencies, states). Imaginary modes, written as negative frequencies, stay visible below zero."""
    from matplotlib.figure import Figure

    values = np.asarray(frequencies, dtype=float)
    x = np.asarray(x, dtype=float)
    figure = Figure(figsize=(8.6 if dos is not None else 7.2, 5.4))
    if dos is not None:
        ax, side = figure.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.05})
    else:
        ax, side = figure.subplots(), None
    for axis in (ax, side):
        if axis is None:
            continue
        axis.set_facecolor("white")
        axis.tick_params(direction="out", length=4, width=0.8, labelsize=11, colors="#253247")
        for spine in axis.spines.values():
            spine.set_color("#8792a2")
            spine.set_linewidth(0.8)
    for start, stop in segments or [[0, len(x)]]:
        ax.plot(x[start:stop], values[start:stop], color=SPINS[0][0], linewidth=1.2)
    ticks = path_ticks(x, labels)
    ax.set_xticks(list(ticks), [tick_label(label) for label in ticks.values()])
    for position in ticks:
        ax.axvline(position, color="0.85", linewidth=0.6, zorder=0)
    low, high = float(values.min()), float(values.max())
    bottom = 1.08 * low if low < -1.0 else 0.0
    ax.set(xlim=(float(x[0]), float(x[-1])), ylim=(bottom, 1.06 * high if high > 0 else 1.0))
    ax.set_ylabel("Frequency (cm$^{-1}$)", fontsize=12)
    if low < -1.0:
        ax.axhline(0, color="#687588", linewidth=0.8, linestyle="--", zorder=0)
    if side is not None:
        energy, states = (np.asarray(item, dtype=float) for item in dos)
        side.plot(states, energy, color=SPINS[0][0], linewidth=1.2)
        side.fill_betweenx(energy, states, color=SPINS[0][0], alpha=0.12, linewidth=0)
        side.set_xlim(0, 1.08 * float(states.max()) if states.max() > 0 else 1.0)
        side.set_xlabel("DOS", fontsize=12)
        side.set_xticks([])
    return _save(figure, folder, stem)


def plot_dielectric(folder, stem, energies, curves):
    """Imaginary (top) and real (bottom) dielectric function against photon energy in eV.
    curves maps a label to (real, imaginary) arrays on the same energies."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 6.4))
    top, bottom = figure.subplots(2, 1, sharex=True)
    palette = ("#2366a8", "#c26a24", "#238878", "#8651a2", "#bb456b", "#788238")
    for index, (label, (real, imaginary)) in enumerate(curves.items()):
        style = {"color": palette[index % len(palette)], "linewidth": 1.4, "linestyle": "--" if "TDDFT" in label else "-"}
        top.plot(energies, imaginary, label=label, **style)
        bottom.plot(energies, real, **style)
    for ax, label in ((top, r"Im $\varepsilon$"), (bottom, r"Re $\varepsilon$")):
        ax.set_facecolor("white")
        ax.tick_params(direction="out", length=4, width=0.8, labelsize=11, colors="#253247")
        for spine in ax.spines.values():
            spine.set_color("#8792a2")
            spine.set_linewidth(0.8)
        ax.set_ylabel(label, fontsize=12)
        ax.axhline(0, color="#687588", linewidth=0.6, zorder=0)
    bottom.set_xlabel("Photon energy (eV)", fontsize=12)
    bottom.set_xlim(float(energies[0]), float(energies[-1]))
    top.legend(frameon=False, fontsize=10)
    return _save(figure, folder, stem)


def plot_curves(folder, stem, x, curves, xlabel, ylabel):
    """Several labelled curves on one axis, such as radial distribution functions."""
    figure, ax = _axes((7.2, 4.8))
    palette = ("#2366a8", "#c26a24", "#238878", "#8651a2", "#bb456b", "#788238")
    for index, (label, values) in enumerate(curves.items()):
        ax.plot(x, values, color=palette[index % len(palette)], linewidth=1.5, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(float(x[0]), float(x[-1])))
    ax.axhline(1.0 if "g(r)" in ylabel else 0.0, color="#687588", linewidth=0.6, linestyle="--", zorder=0)
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=10)
    return _save(figure, folder, stem)


def plot_traces(folder, stem, time_fs, panels, spans=None, guides=None):
    """Stacked panels against time in fs; panels maps an axis label to a list of (label, values).
    spans gives a panel its smallest y range, so numerical noise is not drawn as a signal;
    guides draws a dashed reference level, such as a thermostat target, on a panel."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 2.3 * len(panels) + 0.6))
    axes = figure.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    palette = ("#2366a8", "#c26a24", "#238878")
    for ax, (axis_label, series) in zip(axes, panels.items()):
        for index, (label, values) in enumerate(series):
            ax.plot(time_fs, values, color=palette[index % len(palette)], linewidth=1.3, label=label)
        ax.set_facecolor("white")
        ax.tick_params(direction="out", length=4, width=0.8, labelsize=10, colors="#253247")
        for spine in ax.spines.values():
            spine.set_color("#8792a2")
            spine.set_linewidth(0.8)
        ax.set_ylabel(axis_label, fontsize=11)
        if (guides or {}).get(axis_label) is not None:
            ax.axhline(guides[axis_label], color="#687588", linewidth=0.8, linestyle="--", zorder=0)
        low, high = ax.get_ylim()
        span = (spans or {}).get(axis_label, 0.0)
        if high - low < span:
            ax.set_ylim((low + high - span) / 2, (low + high + span) / 2)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=9, ncol=len(series))
    axes[-1].set_xlabel("Time (fs)", fontsize=12)
    axes[-1].set_xlim(float(time_fs[0]), float(time_fs[-1]))
    return _save(figure, folder, stem)


def plot_dos(folder, stem, energies, densities, reference="fermi_level"):
    """DOS figure from densities shaped (energy, spin) in states/eV; spin down is drawn below zero."""
    energies = np.asarray(energies, dtype=float)
    values = np.asarray(densities, dtype=float)
    values = values[:, None] if values.ndim == 1 else values
    figure, ax = _axes((7.2, 4.8))
    for spin in range(values.shape[1]):
        color, _, label = SPINS[spin]
        signed = -values[:, spin] if spin else values[:, spin]
        ax.plot(energies, signed, color=color, linewidth=1.6, label=label if values.shape[1] > 1 else "Total")
        ax.fill_between(energies, signed, color=color, alpha=0.12, linewidth=0)
    window = dos_window(energies, values)
    ax.set(xlabel=REFERENCES[reference], ylabel="DOS (states/eV)", xlim=window)
    ax.legend(frameon=False, fontsize=11)
    visible = (energies >= window[0]) & (energies <= window[1])
    peak = float(values[visible].max()) if visible.any() else 0.0
    ax.set_ylim(-peak * 1.08 if values.shape[1] > 1 else 0.0, peak * 1.08 if peak > 0 else 1.0)
    ax.axvline(0, color="#687588", linewidth=0.8, linestyle="--")
    return _save(figure, folder, stem)
