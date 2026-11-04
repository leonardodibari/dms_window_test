"""Figures (PNG, 300 dpi). All positions shown 1-based; all three models per figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from dmswin_analysis import DatasetResult

# Categorical slots (fixed order, never cycled) + secondary encoding.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_MARKERS = ["o", "s", "^"]
MODEL_LINES = ["-", "--", "-."]
# Ordinal ramp for representative windows (light -> dark), full fold in ink.
WINDOW_COLORS = ["#86b6ef", "#3987e5", "#184f95"]
INK = "#222222"
MUTED = "#9a9a9a"
DPI = 300

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#777777",
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)


def _style(k: int) -> dict[str, object]:
    return {
        "color": MODEL_COLORS[k % 3],
        "marker": MODEL_MARKERS[k % 3],
        "linestyle": MODEL_LINES[k % 3],
        "linewidth": 2,
        "markersize": 5,
    }


def condition_label(res: DatasetResult) -> str:
    """Short dataset / condition string for titles."""
    ds = res.dataset
    c = ds.conditions
    parts = [ds.dataset_id, ds.reagent or "?"]
    if c.temperature_c is not None:
        parts.append(f"{c.temperature_c:g} °C")
    if c.mg_mm is not None:
        parts.append(f"{c.mg_mm:g} mM Mg²⁺")
    if c.buffer:
        parts.append(c.buffer)
    r0, r1 = res.region
    parts.append(f"R={r0 + 1}–{r1 + 1}")
    parts.append(f"{res.anchoring}")
    return " | ".join(parts)


def _offset_axis(ax: Axes, res: DatasetResult) -> None:
    """Add a top axis in source numbering when it is a constant offset."""
    sp = res.dataset.source_positions
    assert sp is not None
    off = int(sp[0]) - 1
    if np.array_equal(sp, np.arange(1, res.dataset.n + 1) + off) and off != 0:

        def to_src(x: npt.ArrayLike) -> npt.ArrayLike:
            return np.asarray(x) + off

        def to_pos(x: npt.ArrayLike) -> npt.ArrayLike:
            return np.asarray(x) - off

        top = ax.secondary_xaxis("top", functions=(to_src, to_pos))
        top.set_xlabel(f"Source numbering (= position + {off})", fontsize=8)


def _save(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.text(0.01, -0.02, caption, fontsize=7, color="#555555", ha="left", va="top")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _rows(res: DatasetResult, name: str) -> dict[str, np.ndarray]:
    rows = sorted(res.models[name].rows, key=lambda r: r["W"])
    return {
        k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in {"model", "dataset_id"}
    }


def fig_rho_vs_window(res: DatasetResult, path: Path) -> None:
    """Figure 1: rho_W vs W with CI band and rho_N reference; one panel per model."""
    names = list(res.models)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for k, (ax, name) in enumerate(zip(axes, names, strict=True)):
        d = _rows(res, name)
        st = _style(k)
        ax.fill_between(d["W"], d["rho_W_ci_lo"], d["rho_W_ci_hi"], color=st["color"], alpha=0.18, lw=0)
        ax.plot(d["W"], d["rho_W"], label="ρ_W (95% CI band)", **st)
        ax.axhline(d["rho_N"][0], color=INK, ls="--", lw=1.2, label=f"ρ_N = {d['rho_N'][0]:.3f}")
        ax.set_title(name)
        ax.set_xlabel("Window size W (nt)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Spearman ρ(u, reactivity) over E")
    fig.suptitle(f"Agreement with probing data vs window size\n{condition_label(res)}", fontsize=10)
    fig.tight_layout()
    _save(
        fig,
        path,
        f"Shaded: {int(res.bootstrap['ci_level'] * 100)}% paired block-bootstrap CI "
        f"(B={res.bootstrap['n_boot']}, block={res.bootstrap['block_length_nt']} nt). |E|={len(res.e_idx)}. "
        "Dashed: each model's full-fold ρ_N.",
    )


def fig_delta_rho(res: DatasetResult, path: Path) -> None:
    """Figure 2: delta_rho(W) with CI and epsilon; one panel per model."""
    names = list(res.models)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for k, (ax, name) in enumerate(zip(axes, names, strict=True)):
        d = _rows(res, name)
        st = _style(k)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhline(res.epsilon, color=MUTED, ls="--", lw=1.2, label=f"ε = {res.epsilon:g}")
        ax.fill_between(
            d["W"], d["delta_rho_ci_lo"], d["delta_rho_ci_hi"], color=st["color"], alpha=0.18, lw=0
        )
        sw = res.suggestions[name]["W_first"]
        ax.plot(d["W"], d["delta_rho"], label=f"Δρ (suggested W = {sw})", **st)
        ax.set_title(name)
        ax.set_xlabel("Window size W (nt)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Δρ(W) = ρ_N − ρ_W")
    fig.suptitle(f"Cost of truncating the folding context\n{condition_label(res)}", fontsize=10)
    fig.tight_layout()
    _save(
        fig,
        path,
        "Shaded: 95% paired block-bootstrap CI of Δρ. Suggested W: smallest W with Δρ ≤ ε (heuristic). "
        "Δρ < 0: the truncated fold agrees better with the data than the full fold.",
    )


def fig_convergence(res: DatasetResult, path: Path) -> None:
    """Figure 3: model-internal convergence (two panels, no dual axis)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for k, name in enumerate(res.models):
        d = _rows(res, name)
        st = _style(k)
        a1.plot(d["W"], d["conv_spearman"], label=name, **st)
        a2.plot(d["W"], d["max_abs_du"], label=name, **st)
    a1.set_xlabel("Window size W (nt)")
    a1.set_ylabel("Spearman(u^(W), u^(N)) over E")
    a1.set_title("Rank agreement with full fold")
    a2.set_xlabel("Window size W (nt)")
    a2.set_ylabel("max over E of |u^(W) − u^(N)|")
    a2.set_title("Largest change in unpaired probability")
    a2.set_ylim(-0.02, 1.02)
    a1.legend(fontsize=8)
    fig.suptitle(f"Model-internal convergence (no experimental data)\n{condition_label(res)}", fontsize=10)
    fig.tight_layout()
    _save(fig, path, "Each model compared to its own full-length fold.")


def representative_windows(res: DatasetResult, k: int = 3) -> list[int]:
    """Smallest W, a middle W, and the largest W < N."""
    ws = sorted(w for w in res.windows if w != res.dataset.n)
    if len(ws) <= k:
        return ws
    return sorted({ws[0], ws[len(ws) // 2], ws[-1]})


def fig_profile(res: DatasetResult, path: Path) -> None:
    """Figure 4: reactivity + u profiles (full fold and representative windows)."""
    ds = res.dataset
    n = ds.n
    x = np.arange(1, n + 1)
    names = list(res.models)
    reps = representative_windows(res)
    fig, axes = plt.subplots(
        len(names) + 1,
        1,
        figsize=(11, 2.1 * (len(names) + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2] + [1] * len(names)},
    )
    r0, r1 = res.region
    ev = np.array([c in res.eval_nts for c in ds.sequence])
    ax = axes[0]
    react = np.nan_to_num(ds.reactivity, nan=0.0)
    if np.any(~ev):
        ax.bar(x[~ev], react[~ev], color="#d0d0d0", width=0.85, label="other nucleotides")
    ax.bar(x[ev], react[ev], color=INK, width=0.85, label=f"evaluation nts ({''.join(sorted(res.eval_nts))})")
    nanpos = x[~np.isfinite(ds.reactivity)]
    if nanpos.size:
        ax.plot(nanpos, np.zeros_like(nanpos), "x", color="#c0392b", ms=3, label="no data (excluded)")
    ax.set_ylabel(f"{ds.reagent or ''} reactivity")
    ax.set_title(f"Per-position profile\n{condition_label(res)}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    _offset_axis(ax, res)
    for k, name in enumerate(names):
        ax = axes[k + 1]
        mr = res.models[name]
        ax.plot(x, mr.u_full, color=INK, lw=1.0, ls="--", zorder=5, label=f"full fold (N={n})")
        for j, w in enumerate(reps):
            a, b = res.windows[w]
            ax.plot(x, mr.u_windows[w], color=WINDOW_COLORS[j % 3], lw=1.3, label=f"W={w} ({a + 1}–{b + 1})")
            for edge in (a + 0.5, b + 1.5):
                ax.axvline(edge, color=WINDOW_COLORS[j % 3], lw=0.8, ls=":")
        ax.set_ylabel(f"{name}\nu (unpaired prob.)")
        ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    for ax in axes:
        ax.axvspan(r0 + 0.5, r1 + 1.5, color="#f2c94c", alpha=0.18, lw=0)
    axes[-1].set_xlabel("Position (1-based)")
    fig.tight_layout()
    _save(
        fig,
        path,
        "Yellow band: region of interest R. Dotted verticals: window bounds. "
        "u is shown only inside each window.",
    )


def fig_heatmap(res: DatasetResult, path: Path) -> None:
    """Figure 5: |u^(W) - u^(N)| heatmap, one panel per model."""
    names = list(res.models)
    ws = sorted(res.windows)
    n = res.dataset.n
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.4 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("Blues").with_extremes(bad="#f4f4f4")
    r0, r1 = res.region
    im = None
    for ax, name in zip(axes, names, strict=True):
        mr = res.models[name]
        mat = np.stack([np.abs(mr.u_windows[w] - mr.u_full) for w in ws])
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=(0.5, n + 0.5, -0.5, len(ws) - 0.5),
            interpolation="nearest",
        )
        step = max(1, len(ws) // 8)
        ax.set_yticks(range(0, len(ws), step), [str(w) for w in ws[::step]])
        ax.set_ylabel(f"{name}\nW (nt)")
        ax.axvline(r0 + 0.5, color="#c0392b", lw=0.8)
        ax.axvline(r1 + 1.5, color="#c0392b", lw=0.8)
        ax.grid(False)
    axes[0].set_title(f"|u_i^(W) − u_i^(N)| along the sequence\n{condition_label(res)}")
    axes[-1].set_xlabel("Position (1-based)")
    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="|Δu|")
    _save(fig, path, "Grey: outside the window (not folded). Red lines: bounds of R.")


def fig_scatter(res: DatasetResult, path: Path) -> None:
    """Figure 6: u (full fold) vs reactivity over E."""
    names = list(res.models)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    e = res.e_idx
    r = res.dataset.reactivity[e]
    for k, (ax, name) in enumerate(zip(axes, names, strict=True)):
        u = res.models[name].u_full[e]
        row = next(x for x in res.models[name].rows if x["W"] == res.dataset.n)
        ax.scatter(u, r, s=22, color=MODEL_COLORS[k], marker=MODEL_MARKERS[k], edgecolor="white", lw=0.6)
        ax.set_title(name)
        ax.set_xlabel("u (full fold)")
        ax.text(
            0.03,
            0.95,
            f"ρ_N = {row['rho_N']:.3f}\n95% CI [{row['rho_N_ci_lo']:.2f}, {row['rho_N_ci_hi']:.2f}]",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_xlim(-0.05, 1.05)
    axes[0].set_ylabel(f"{res.dataset.reagent or ''} reactivity")
    fig.suptitle(
        f"Unpaired probability vs reactivity over E (|E|={len(e)})\n{condition_label(res)}", fontsize=10
    )
    fig.tight_layout()
    _save(fig, path, "Spearman ρ with average ranks for ties.")


FIGURES = {
    "fig_rho_vs_window.png": fig_rho_vs_window,
    "fig_delta_rho_vs_window.png": fig_delta_rho,
    "fig_convergence_vs_window.png": fig_convergence,
    "fig_profile.png": fig_profile,
    "fig_du_heatmap.png": fig_heatmap,
    "fig_u_vs_reactivity.png": fig_scatter,
}


def make_all(res: DatasetResult, outdir: Path, prefix: str = "") -> list[str]:
    """Write all figures; return file names."""
    names = []
    for fname, fn in FIGURES.items():
        out = outdir / f"{prefix}{fname}"
        fn(res, out)
        names.append(out.name)
    return names


# --------------------------------------------------------------------------- #
# Overview figures across datasets
# --------------------------------------------------------------------------- #
# Categorical slots for datasets (fixed order), plus a marker each.
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DATASET_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _grid(n_panels: int, ncols: int = 3) -> tuple[plt.Figure, np.ndarray]:
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def _w_conv(rows: list[dict[str, float]], tol: float = 0.1) -> float:
    srt = sorted(rows, key=lambda r: r["W"])
    for k, r in enumerate(srt):
        if all(q["max_abs_du"] <= tol for q in srt[k:]):
            return float(r["W"])
    return float(srt[-1]["W"])


def _short(res: DatasetResult) -> str:
    ds = res.dataset
    return (
        f"{ds.dataset_id}\n{ds.metadata.get('name', '')[:38]} (N={ds.n}, {ds.reagent}, |E|={len(res.e_idx)})"
    )


def overview_delta_rho(results: list[DatasetResult], path: Path) -> None:
    """Delta rho vs W, one panel per dataset, all models."""
    fig, axes = _grid(len(results))
    for ax, res in zip(axes.flat, results, strict=False):
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhline(res.epsilon, color=MUTED, ls="--", lw=1)
        for k, name in enumerate(res.models):
            d = _rows(res, name)
            st = _style(k)
            ax.fill_between(
                d["W"], d["delta_rho_ci_lo"], d["delta_rho_ci_hi"], color=st["color"], alpha=0.10, lw=0
            )
            ax.plot(d["W"], d["delta_rho"], label=name, **{**st, "markersize": 3.5, "linewidth": 1.6})
        ax.set_title(_short(res), fontsize=8)
        ax.set_xlabel("Window size W (nt)")
        ax.set_ylabel("Δρ = ρ_N − ρ_W")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Cost of truncating the folding context, all datasets (5'-anchored windows)", fontsize=11)
    fig.tight_layout()
    _save(
        fig,
        path,
        f"Bands: 95% paired block-bootstrap CI. Dashed grey: ε = {results[0].epsilon:g}. "
        "Δρ > 0: the truncated fold agrees less with the probing data than the full fold.",
    )


def overview_convergence(results: list[DatasetResult], path: Path) -> None:
    """max |delta u| over E vs W, one panel per dataset, all models."""
    fig, axes = _grid(len(results))
    for ax, res in zip(axes.flat, results, strict=False):
        ax.axhline(0.1, color=MUTED, ls="--", lw=1)
        for k, name in enumerate(res.models):
            d = _rows(res, name)
            ax.plot(d["W"], d["max_abs_du"], label=name, **{**_style(k), "markersize": 3.5, "linewidth": 1.6})
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(_short(res), fontsize=8)
        ax.set_xlabel("Window size W (nt)")
        ax.set_ylabel("max over E of |u^(W) − u^(N)|")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Model-internal convergence to the full-length fold, all datasets", fontsize=11)
    fig.tight_layout()
    _save(fig, path, "Dashed grey: 0.1, the tolerance used for W_conv. No experimental data involved.")


def overview_relative(results: list[DatasetResult], path: Path) -> None:
    """Delta rho and convergence vs W/N; one column per model, one line per dataset."""
    names = list(results[0].models)
    fig, axes = plt.subplots(
        2, len(names), figsize=(4.4 * len(names), 7), sharex=True, sharey="row", squeeze=False
    )
    for j, name in enumerate(names):
        for k, res in enumerate(results):
            d = _rows(res, name)
            rel = d["W"] / res.dataset.n
            st = {
                "color": DATASET_COLORS[k % 6],
                "marker": DATASET_MARKERS[k % 6],
                "markersize": 3.5,
                "lw": 1.5,
            }
            axes[0, j].plot(rel, d["delta_rho"], label=res.dataset.dataset_id, **st)
            axes[1, j].plot(rel, d["max_abs_du"], label=res.dataset.dataset_id, **st)
        axes[0, j].axhline(0, color=INK, lw=0.8)
        axes[0, j].axhline(results[0].epsilon, color=MUTED, ls="--", lw=1)
        axes[1, j].axhline(0.1, color=MUTED, ls="--", lw=1)
        axes[0, j].set_title(name)
        axes[1, j].set_xlabel("Relative window size W / N")
    axes[0, 0].set_ylabel("Δρ = ρ_N − ρ_W")
    axes[1, 0].set_ylabel("max over E of |u^(W) − u^(N)|")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Truncation cost and convergence vs relative window size, all datasets", fontsize=11)
    fig.tight_layout()
    _save(
        fig,
        path,
        "Top: cost against the probing data (CIs omitted for readability; see the per-dataset figure). "
        "Bottom: distance to the model's own full fold. Dashed: ε (top), 0.1 (bottom).",
    )


def overview_summary(results: list[DatasetResult], path: Path) -> None:
    """(a) rho_N per dataset and model with CI; (b) window needed per dataset and model."""
    names = list(results[0].models)
    ids = [f"{r.dataset.dataset_id}\nN={r.dataset.n}, {r.dataset.reagent}" for r in results]
    y = np.arange(len(results))[::-1].astype(float)
    off = np.linspace(-0.22, 0.22, len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(results) + 2.2), sharey=True)
    for k, name in enumerate(names):
        st = _style(k)
        rho, lo, hi, wc, ws = [], [], [], [], []
        for res in results:
            rows = sorted(res.models[name].rows, key=lambda r: r["W"])
            full = rows[-1]
            rho.append(full["rho_N"])
            lo.append(full["rho_N_ci_lo"])
            hi.append(full["rho_N_ci_hi"])
            wc.append(_w_conv(rows) / res.dataset.n)
            stable = res.suggestions[name]["W_stable"]
            ws.append((stable if stable is not None else res.dataset.n) / res.dataset.n)
        yy = y + off[k]
        a1.errorbar(
            rho,
            yy,
            xerr=[np.array(rho) - lo, np.array(hi) - rho],
            fmt=str(st["marker"]),
            color=st["color"],
            ms=6,
            capsize=2,
            lw=1.2,
            label=name,
        )
        a2.plot(wc, yy, str(st["marker"]), color=st["color"], ms=7, label=f"{name}: W_conv / N")
        a2.plot(
            ws,
            yy,
            str(st["marker"]),
            mfc="white",
            mec=st["color"],
            mew=1.5,
            ms=7,
            label=f"{name}: W*_stable / N",
        )
    a1.axvline(0, color=INK, lw=0.8)
    a1.set_yticks(y, ids)
    a1.set_xlabel("ρ_N = Spearman(u full fold, reactivity) over E")
    a1.set_title("(a) Agreement of the full-length fold\nwith probing data")
    a1.legend(fontsize=7, loc="lower right")
    a2.set_xlim(0, 1.05)
    a2.set_xlabel("Window needed, as fraction of N")
    a2.set_title("(b) Window needed\n(filled: model convergence; open: data criterion)")
    a2.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(
        fig,
        path,
        "(a) Bars: 95% block-bootstrap CI. (b) W_conv: smallest W from which max |Δu| over E stays ≤ 0.1. "
        "W*_stable: smallest W from which Δρ ≤ ε for all larger W. 1.0 = the full length is needed.",
    )


OVERVIEW_FIGURES = {
    "overview_delta_rho.png": overview_delta_rho,
    "overview_convergence.png": overview_convergence,
    "overview_relative_window.png": overview_relative,
    "overview_summary.png": overview_summary,
}


def make_overview(results: list[DatasetResult], outdir: Path) -> list[str]:
    """Write the cross-dataset figures; return file names."""
    for fname, fn in OVERVIEW_FIGURES.items():
        fn(results, outdir / fname)
    return list(OVERVIEW_FIGURES)
