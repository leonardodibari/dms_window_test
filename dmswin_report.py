"""Render README.md (method + results) from the run results.

Dataset-specific prose comes only from the config's ``report_notes``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dmswin_analysis import DatasetResult


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ranges(idx: Sequence[int]) -> str:
    """Compress sorted integers into 'a-b, c' ranges."""
    idx = sorted(idx)
    if not idx:
        return "none"
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = i
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(out)


METHOD = r"""
## 2. Method

**Notation.** $s$ is the probed sequence (5'→3', RNA alphabet, T→U), length $N$,
positions $1..N$ (1-based in the text, tables and figures; code is 0-based and
converts at its boundaries). $r_i$ is the reactivity at position $i$ (NaN where there
are no data).

1. **Region of interest** $R=[r_\text{start}, r_\text{end}]$: the part of the RNA whose
   structure we care about.
2. **Windows.** For each size $W$ in a grid, a window $[a,b]$ with $a\le r_\text{start}$,
   $b\ge r_\text{end}$, $W=b-a+1$:
   * `five_prime`: $a=1$, $b=W$ (the mRNA use case: 5'UTR + Kozak + start of CDS);
   * `centered`: extend symmetrically around $R$ (extra nucleotide on the 3' side),
     shifted to stay inside $1..N$.
   The grid runs from the smallest $W$ that contains $R$ up to $N$ (always included).
   The actual $[a,b]$ used for each $W$ is listed in `summary.csv`.
3. **Folding.** The subsequence $s[a..b]$ is folded in isolation with each model.
   From its base-pair probability matrix $p_{ij}$,
   $$u_i^{(W)} = 1-\sum_j p_{ij}, \qquad i\in[a,b]$$
   is the probability that $i$ is unpaired. The reference $u_i^{(N)}$ comes from
   folding the full sequence with the same model and parameters.
4. **Evaluation set**
   $$E=\{\,i\in R : s_i\in\mathcal{N}_\text{eval},\ r_i \text{ finite}\,\},$$
   where $\mathcal{N}_\text{eval}$ depends on the reagent (DMS → {A, C}; SHAPE-type →
   {A, C, G, U}; overridable). Because $R$ is inside every window, $E$ is the same
   set for every $W$, every model and the full fold (checked at run time).
5. **Agreement with experiment.** Reactivity is high for *unpaired* nucleotides, so it
   is correlated with $u$ (not with the paired probability):
   $$\rho_W=\operatorname{Spearman}\big(u^{(W)}_E,\ r_E\big),\qquad \rho_N=\operatorname{Spearman}\big(u^{(N)}_E,\ r_E\big)$$
   (ties get average ranks, `scipy.stats.spearmanr`).
6. **Cost of truncation**, within each model:
   $$\Delta\rho(W)=\rho_N-\rho_W .$$
   Positive values mean the truncated fold agrees *less* with the data than the full fold.
7. **Model-internal convergence** (no experimental data):
   $\operatorname{Spearman}(u^{(W)}_E, u^{(N)}_E)$ and $\max_{i\in E}|u_i^{(W)}-u_i^{(N)}|$.
8. **Uncertainty: paired block bootstrap.** Neighbouring positions are correlated
   (helices), so positions are resampled in blocks of $L$ consecutive nucleotides of
   $R$ (random starts, circular inside $R$); the members of $E$ in the drawn blocks
   are concatenated until $|E|$ positions are collected. $\rho_N$, $\rho_W$ (all $W$,
   all models) are computed on the **same** resample, so the CI of $\Delta\rho$ accounts
   for their correlation. 95 % percentile intervals; RNG is a seeded
   `numpy.random.Generator`.
9. **Suggested window (heuristic).** For each model,
   $$W^*=\min\{W:\ \Delta\rho(W)\le\varepsilon\}.$$
   This is a heuristic decision rule, not a statistical test. Two companion values
   are reported: $W^*_\text{stable}$, the smallest $W$ from which *all* larger grid
   sizes satisfy the rule (robust to non-monotone curves), and $W^*_\text{CI}$, the
   smallest $W$ whose upper 95 % CI bound on $\Delta\rho$ is $\le\varepsilon$
   (conservative). Note that $\Delta\rho<0$ (the truncated fold agrees *better*
   with the data than the full fold) also satisfies the rule. That happens when the
   full-length fold predicts long-range pairs that the probing data do not support;
   in that case the rule says only that truncation does not hurt, and the
   model-internal convergence (step 7) shows how far the truncated fold actually is
   from the full one.
10. **Several RNAs/datasets** (when run): per-RNA $\Delta\rho$ is in each dataset's
    `summary.csv`, and a Wilcoxon signed-rank test across RNAs of $\Delta\rho(W)$
    (equivalently paired $\rho_N$ vs $\rho_W$) is written to `cross_dataset_wilcoxon.csv`
    for every $W$ shared by all datasets.
"""


def _models_section(res: DatasetResult, flags: list[str], bias: dict[str, Any] | None = None) -> str:
    lines = ["## 4. Models and parameters", ""]
    lines.append("| Model | Version | Parameters | Temperature | Salt | Mg²⁺ / tertiary |")
    lines.append("|---|---|---|---|---|---|")
    for name, mr in res.models.items():
        d = mr.description
        if name == "RNAfold":
            params = (
                f"{d['energy_params']}, dangles={d['dangles']}, partition function (`fc.pf()`, `fc.bpp()`)"
            )
            temp = f"{d['temperature_c']:g} °C"
            salt = (
                f"correction on, {d['salt_molar']} M"
                if d["salt_correction"]
                else f"standard {d['salt_molar']} M monovalent (correction off)"
            )
        elif name == "CONTRAfold":
            params = (
                f"`contrafold predict --posteriors {d['posterior_threshold']:g}`, {d['params']} parameters"
            )
            temp = "not representable"
            salt = "not representable"
        else:
            params = f"`mxfold2 predict --bpp`, {d['param']}, seed {d['seed']}"
            temp = "not representable"
            salt = "not representable"
        ver = d["version"]
        if name == "CONTRAfold" and d.get("conda_package_version"):
            ver = f"{ver} (bioconda package {d['conda_package_version']})"
        lines.append(f"| {name} | {ver} | {params} | {temp} | {salt} | not representable |")
    lines.append("")
    cf = res.models.get("CONTRAfold")
    if cf:
        thr = cf.description["posterior_threshold"]
        n = res.dataset.n
        lines.append(
            f"**CONTRAfold threshold bias.** CONTRAfold only writes pairs with posterior ≥ {thr:g}. "
            f"Omitted pairs are counted as 'unpaired', so $u_i$ is biased **upward** by at most "
            f"(number of omitted partners of $i$) × {thr:g} ≤ $(N-1)$ × {thr:g} = {(n - 1) * thr:.4f} "
            f"for the full fold (less for windows)."
        )
        if bias:
            meas = "; ".join(
                f"W = {w}: max |Δu| = {v['max_abs_du_all_positions']:.2g}, |Δρ| = {v['abs_rho_change']:.2g}"
                for w, v in bias["windows"].items()
            )
            lines[-1] += (
                f" **Measured on this dataset** by re-folding with threshold {bias['reference_threshold']:g}: {meas}."
            )
        lines.append("")
    lines.append(
        "**MXfold2 rounding.** MXfold2 writes $p_{ij}$ with 5 decimals; the resulting error on "
        "$u_i$ is ≤ $N\\times5\\cdot10^{-6}$ (≈0.001 here). Values of $u$ are clipped to [0, 1]; the "
        "maximal clipped amount per call is in `run_metadata.json`."
    )
    lines.append("")
    lines.append("**Condition mismatches flagged at run time:**")
    lines.append("")
    lines += [f"* {f}" for f in flags]
    lines.append("")
    return "\n".join(lines)


def _results_section(res: DatasetResult, prefix: str, figs: list[str]) -> str:
    names = list(res.models)
    ws = sorted(res.windows)
    out = ["## 5. Results", ""]
    out.append(
        f"|E| = {len(res.e_idx)} positions. ε = {res.epsilon:g}. "
        f"Bootstrap: B = {res.bootstrap['n_boot']}, block = {res.bootstrap['block_length_nt']} nt, "
        f"seed = {res.bootstrap['seed']}."
    )
    out.append("")
    out.append("### Summary table")
    out.append("")
    out.append(
        "$\\rho_W$ [95 % CI] and $\\Delta\\rho(W)$ [95 % CI] per model. "
        f"Full numbers: `{prefix}summary.csv` (long) and `{prefix}summary_wide.csv`."
    )
    out.append("")
    hdr = "| W | window (1-based) | " + " | ".join(f"{n} ρ_W | {n} Δρ" for n in names) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (2 + 2 * len(names)))
    for w in ws:
        a, b = res.windows[w]
        cells = [str(w) + (" (N)" if w == res.dataset.n else ""), f"{a + 1}–{b + 1}"]
        for n in names:
            r = next(x for x in res.models[n].rows if x["W"] == w)
            cells.append(f"{_f(r['rho_W'])} [{_f(r['rho_W_ci_lo'], 2)}, {_f(r['rho_W_ci_hi'], 2)}]")
            cells.append(
                f"{_f(r['delta_rho'])} [{_f(r['delta_rho_ci_lo'], 2)}, {_f(r['delta_rho_ci_hi'], 2)}]"
            )
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("### Suggested window per model (heuristic)")
    out.append("")
    out.append("| Model | ρ_N [95 % CI] | W* (first Δρ ≤ ε) | W*_stable | W*_CI (upper CI ≤ ε) |")
    out.append("|---|---|---|---|---|")
    for n in names:
        s = res.suggestions[n]
        r = next(x for x in res.models[n].rows if x["W"] == res.dataset.n)
        out.append(
            f"| {n} | {_f(r['rho_N'])} [{_f(r['rho_N_ci_lo'], 2)}, {_f(r['rho_N_ci_hi'], 2)}] | "
            f"{s['W_first']} | {s['W_stable']} | {s['W_ci']} |"
        )
    out.append("")
    captions = {
        "fig_rho_vs_window.png": "**Figure 1.** $\\rho_W$ vs $W$ (lines), each model's $\\rho_N$ (dotted), 95 % bootstrap CI (band).",
        "fig_delta_rho_vs_window.png": "**Figure 2.** $\\Delta\\rho(W)$ with 95 % CI; dashed line: ε.",
        "fig_convergence_vs_window.png": "**Figure 3.** Model-internal convergence: Spearman of $u^{(W)}$ to $u^{(N)}$ over $E$ (left) and max $|\\Delta u|$ over $E$ (right).",
        "fig_profile.png": "**Figure 4.** Reactivity (evaluation nucleotides dark, others grey, excluded positions ×) and $u$ for the full fold and three representative windows, one panel per model; yellow = $R$.",
        "fig_du_heatmap.png": "**Figure 5.** $|u_i^{(W)}-u_i^{(N)}|$; position on x, $W$ on y; grey = outside window.",
        "fig_u_vs_reactivity.png": "**Figure 6.** $u$ (full fold) vs reactivity over $E$, one panel per model.",
    }
    out.append("### Figures")
    out.append("")
    for f in figs:
        key = f[len(prefix) :]
        out.append(captions.get(key, ""))
        out.append("")
        out.append(f"![{key}]({f})")
        out.append("")
    out.append("### Comparison of the three models")
    out.append("")
    out += _auto_comparison(res)
    out.append("")
    return "\n".join(out)


def _auto_comparison(res: DatasetResult) -> list[str]:
    """Data-driven comparison sentences (no hand-written results)."""
    lines = []
    n = res.dataset.n
    full = {m: next(x for x in mr.rows if x["W"] == n) for m, mr in res.models.items()}
    order = sorted(full, key=lambda m: -full[m]["rho_N"])
    lines.append(
        "* **Absolute agreement (full fold):** "
        + ", ".join(
            f"{m} ρ_N = {_f(full[m]['rho_N'])} [{_f(full[m]['rho_N_ci_lo'], 2)}, {_f(full[m]['rho_N_ci_hi'], 2)}]"
            for m in order
        )
        + ". Overlapping CIs mean the ranking is not resolved by these data."
    )
    wmin = min(res.windows)
    small = {m: next(x for x in mr.rows if x["W"] == wmin) for m, mr in res.models.items()}
    lines.append(
        f"* **Smallest window (W = {wmin}):** "
        + ", ".join(
            f"{m} Δρ = {_f(small[m]['delta_rho'])} [{_f(small[m]['delta_rho_ci_lo'], 2)}, {_f(small[m]['delta_rho_ci_hi'], 2)}]"
            for m in res.models
        )
        + "."
    )
    lines.append(
        "* **Largest Δρ over the grid:** "
        + ", ".join(
            f"{m} {_f(max(r['delta_rho'] for r in mr.rows))} at W = {max(mr.rows, key=lambda r: r['delta_rho'])['W']}"
            for m, mr in res.models.items()
        )
        + "."
    )
    conv = []
    for m, mr in res.models.items():
        w_conv = next(
            (
                r["W"]
                for r in sorted(mr.rows, key=lambda r: r["W"])
                if all(q["max_abs_du"] <= 0.1 for q in mr.rows if q["W"] >= r["W"])
            ),
            None,
        )
        conv.append(f"{m} W = {w_conv}")
    lines.append(
        "* **Internal convergence** (smallest W from which max |Δu| over E stays ≤ 0.1): "
        + ", ".join(conv)
        + "."
    )
    for m, mr in res.models.items():
        neg = [r["W"] for r in mr.rows if r["delta_rho_ci_hi"] < 0]
        pos = [r["W"] for r in mr.rows if r["delta_rho_ci_lo"] > 0]
        if neg or pos:
            lines.append(
                f"* **{m}: Δρ CI excludes 0** "
                + (f"below 0 (truncated fold agrees *better* with data) at W = {neg}" if neg else "")
                + ("; " if neg and pos else "")
                + (f"above 0 (truncation costs agreement) at W = {pos}" if pos else "")
                + "."
            )
        else:
            lines.append(f"* **{m}:** the 95 % CI of Δρ includes 0 at every W.")
    lines.append(
        f"* **Suggested W (ε = {res.epsilon:g}):** "
        + ", ".join(
            f"{m} W* = {s['W_first']} (stable {s['W_stable']}, CI-based {s['W_ci']})"
            for m, s in res.suggestions.items()
        )
        + "."
    )
    any_pos = any(r["delta_rho_ci_lo"] > 0 for mr in res.models.values() for r in mr.rows)
    all_include_zero = all(
        r["delta_rho_ci_lo"] <= 0 <= r["delta_rho_ci_hi"] for mr in res.models.values() for r in mr.rows
    )
    lines.append("")
    lines.append("**Interpretation (generated from the numbers above).**")
    if not any_pos:
        lines.append(
            "For no model and no window is truncation shown to cost agreement with the data (no Δρ CI lies "
            "entirely above 0)."
            + (
                " Every CI also includes 0, so with these data truncated and full folds cannot be told apart "
                "on the evaluation set."
                if all_include_zero
                else ""
            )
            + " W* therefore equals the smallest window tested, which says only that the data do not penalise "
            "truncation; it is not evidence that small windows reproduce the full-length structure. The "
            "model-internal convergence (Figures 3 and 5) shows where the predicted structure inside R "
            "actually stops changing, and W*_CI / W*_stable give more conservative choices."
        )
    else:
        lines.append(
            "Truncation measurably reduces agreement with the data for some windows (Δρ CI above 0); "
            "W*, W*_stable and W*_CI should be read together with the convergence figures."
        )
    return lines


OVERVIEW_CAPTIONS = {
    "overview_summary.png": "**Overview A.** (a) Agreement of each model's full-length fold with the probing data, with 95 % CI. "
    "(b) Window needed as a fraction of the RNA length: filled = the model's own prediction in R has converged "
    "(max |Δu| ≤ 0.1), open = data criterion W*_stable (Δρ ≤ ε for all larger W).",
    "overview_delta_rho.png": "**Overview B.** Δρ(W) for every dataset (panels) and model (lines), with 95 % CI bands.",
    "overview_convergence.png": "**Overview C.** Model-internal convergence (max |Δu| over E) for every dataset and model.",
    "overview_relative_window.png": "**Overview D.** The same curves against relative window size W/N, one panel per model and "
    "one line per dataset, so RNAs of different lengths can be compared directly.",
}


def _overview_figures(figs: Sequence[str]) -> str:
    """Embed the cross-dataset figures."""
    if not figs:
        return ""
    out = ["### Overview figures", ""]
    order = [
        "overview_summary.png",
        "overview_delta_rho.png",
        "overview_convergence.png",
        "overview_relative_window.png",
    ]
    for f in [x for x in order if x in figs] + [x for x in figs if x not in order]:
        out += [OVERVIEW_CAPTIONS.get(f, ""), "", f"![{f}]({f})", ""]
    return "\n".join(out)


def _overview(results: Sequence[DatasetResult]) -> str:
    """Cross-dataset overview table (one row per dataset and model)."""
    out = ["## Overview across datasets", ""]
    out.append(
        "One row per dataset and model. ρ_N: full-fold agreement; Δρ(W_min): cost at the smallest window; "
        "'Δρ CI > 0 at': windows where truncation significantly lowers agreement; W_conv: smallest W from "
        "which max |Δu| over E stays ≤ 0.1. Details per dataset below."
    )
    out.append("")
    out.append(
        "| Dataset | N | reagent | R (1-based) | |E| | Model | ρ_N [95 % CI] | Δρ(W_min) [95 % CI] "
        "| Δρ CI > 0 at | W* | W*_stable | W*_CI | W_conv |"
    )
    out.append("|" + "---|" * 13)
    for res in results:
        ds = res.dataset
        wmin = min(res.windows)
        for m, mr in res.models.items():
            rows = sorted(mr.rows, key=lambda r: r["W"])
            full = rows[-1]
            small = rows[0]
            pos = [r["W"] for r in rows if r["delta_rho_ci_lo"] > 0]
            w_conv = next(
                (r["W"] for r in rows if all(q["max_abs_du"] <= 0.1 for q in rows if q["W"] >= r["W"])), None
            )
            s = res.suggestions[m]
            out.append(
                f"| {ds.dataset_id} | {ds.n} | {ds.reagent} | {res.region[0] + 1}–{res.region[1] + 1} | "
                f"{len(res.e_idx)} | {m} | {_f(full['rho_N'])} [{_f(full['rho_N_ci_lo'], 2)}, "
                f"{_f(full['rho_N_ci_hi'], 2)}] | {_f(small['delta_rho'])} [{_f(small['delta_rho_ci_lo'], 2)}, "
                f"{_f(small['delta_rho_ci_hi'], 2)}] (W={wmin}) | {pos if pos else 'none'} | {s['W_first']} | "
                f"{s['W_stable']} | {s['W_ci']} | {w_conv} |"
            )
    out.append("")
    return "\n".join(out)


GENERIC_LIMITATIONS = [
    "**Reagent coverage.** DMS mainly reports on the Watson–Crick face of A and C (N1-A, N3-C); G and U carry little "
    "signal under standard conditions, so they are excluded from $E$. Conclusions hold for A/C only.",
    "**Physical conditions.** RNAfold's Turner parameters assume standard salt (1 M NaCl); salt correction (if enabled) is "
    "monovalent only. None of the three models represents Mg²⁺ or Mg²⁺-dependent tertiary structure, which limits the "
    "absolute $\\rho$ any model can reach. Conclusions therefore rest on $\\Delta\\rho$ (each model vs its own full fold), "
    "not on absolute $\\rho$.",
    "**Trained models.** CONTRAfold and MXfold2 are trained on databases of known structures, have no explicit "
    "temperature or salt dependence, and may have seen this RNA or close homologues during training, which can "
    "inflate their absolute agreement.",
    "**Statistics.** $|E|$ is small, so CIs are wide; neighbouring positions are correlated, which the block bootstrap only "
    "partly accounts for (block length is a choice). The ε rule is a heuristic; the grid step sets the resolution of $W^*$. "
    "Spearman ρ uses only ranks, so it is insensitive to monotone rescaling of reactivity but also ignores the magnitude of $u$.",
    "**Single construct.** One RNA gives one estimate of $\\Delta\\rho(W)$; generalisation needs several RNAs "
    "(the across-RNA Wilcoxon test becomes available when several datasets are run).",
]


def render(
    results: Sequence[DatasetResult],
    contexts: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    cross: Sequence[dict[str, Any]],
) -> str:
    """Render the full README.md."""
    parts: list[str] = []
    ids = ", ".join(r.dataset.dataset_id for r in results)
    parts.append(f"# Probing-data validity test of folding models and folding-window size ({ids})")
    parts.append("")
    parts.append("*This file is generated by `run_test.py`; re-running the command below regenerates it.*")
    parts.append("")
    parts.append("## 1. Purpose")
    parts.append("")
    parts.append(
        "When an RNA is folded computationally, one often folds only a window around the region of interest "
        "(e.g. the 5' end of an mRNA) instead of the full molecule. This test answers two questions with "
        "chemical-probing data:\n\n"
        "1. How well does each folding model's per-nucleotide unpaired probability agree with the measured "
        "reactivity?\n"
        "2. How much does that agreement degrade when the folding context is truncated to a window of size $W$, "
        "and what is the smallest window that loses (almost) nothing?\n\n"
        "Three models are compared: ViennaRNA RNAfold (thermodynamic partition function), CONTRAfold "
        "(probabilistic, trained) and MXfold2 (deep learning with thermodynamic integration)."
    )
    parts.append(METHOD)
    if len(results) > 1:
        parts.append(_overview(results))
        parts.append(_overview_figures(meta.get("overview_figures", [])))
    for res, ctx in zip(results, contexts, strict=True):
        ds = res.dataset
        cfg = ctx["cfg"]
        notes = cfg.get("report_notes", {})
        sel = ctx["selection"]
        if len(results) > 1:
            parts.append(f"# Dataset {ds.dataset_id}")
            parts.append("")
        parts.append("## 3. Data")
        parts.append("")
        prov = ds.metadata.get("provenance", {})
        c = ds.conditions
        sp = ds.source_positions
        assert sp is not None
        r0, r1 = res.region
        parts.append(
            f"* **Source:** {ds.source_url} — file `{cfg['source']['path']}` downloaded from "
            f"{prov.get('url')} on {prov.get('download_date')} (SHA-256 `{prov.get('sha256')}`)."
        )
        if notes.get("construct"):
            parts.append(f"* **Construct:** {notes['construct']}")
        parts.append(
            f"* **Sequence:** N = {ds.n} nt, RDAT/source numbering {int(sp[0])}–{int(sp[-1])} "
            f"(1-based position = source − {int(sp[0]) - 1})."
        )
        parts.append(
            f"* **Row used:** {sel.get('row')} of {sel.get('n_constructs_in_file')} data rows, selected by "
            f"annotation match `{sel.get('selection')}` (row annotations: `{sel.get('row_annotations')}`). "
            "The selection fails loudly if it matches zero or several rows."
        )
        parts.append(
            f"* **Reagent:** {ds.reagent}. **Conditions (from file annotations):** "
            f"{c.temperature_c} °C, Mg²⁺ {c.mg_mm} mM, buffer {c.buffer}; all annotations: "
            f"`{', '.join(c.chemicals)}`."
        )
        parts.append(
            f"* **Evaluation nucleotides:** {''.join(sorted(res.eval_nts))} "
            f"({meta['datasets'][ds.dataset_id]['effective']['evaluation_nucleotides_source']})."
        )
        masked = sel.get("masked_indices_0based", [])
        parts.append(
            f"* **Excluded positions (set to NaN):** source {_ranges([int(sp[i]) for i in masked])} "
            f"(1-based {_ranges([i + 1 for i in masked])}). Reason: {sel.get('mask_reason')} "
            f"NaN already present in the source: {_ranges([int(sp[i]) for i in sel.get('nan_in_source', [])])}."
        )
        parts.append(
            f"* **Region of interest:** source {int(sp[r0])}–{int(sp[r1])} (1-based {r0 + 1}–{r1 + 1}), "
            f"anchoring `{res.anchoring}`. {notes.get('region', '')}"
        )
        parts.append(
            f"* **Evaluation set:** |E| = {len(res.e_idx)}; source positions "
            f"{_ranges([int(sp[i]) for i in res.e_idx])}."
        )
        parts.append(f"* **Window grid:** {', '.join(str(w) for w in sorted(res.windows))}.")
        for note in notes.get("data_caveats", []):
            parts.append(f"* {note}")
        parts.append("")
        parts.append(_models_section(res, ctx["flags"], ctx.get("bias")))
        parts.append(_results_section(res, ctx["prefix"], ctx["figs"]))
    if cross:
        parts.append("### Across datasets (Wilcoxon signed-rank on Δρ)")
        parts.append("")
        parts.append("| Model | W | n | median Δρ | p |")
        parts.append("|---|---|---|---|---|")
        for r in cross:
            parts.append(
                f"| {r['model']} | {r['W']} | {r['n_datasets']} | {_f(r['median_delta_rho'])} | {_f(r['wilcoxon_p'], 4)} |"
            )
        parts.append("")
    parts.append("## 6. Limitations")
    parts.append("")
    for lim in GENERIC_LIMITATIONS:
        parts.append(f"* {lim}")
    for ctx in contexts:
        tag = f"[{ctx['cfg'].get('dataset_id', '')}] " if len(contexts) > 1 else ""
        for lim in ctx["cfg"].get("report_notes", {}).get("limitations", []):
            parts.append(f"* {tag}{lim}")
    parts.append("")
    parts.append("## 7. Reproduction")
    parts.append("")
    tools = meta["tools"]
    cfgs = " ".join(meta["datasets"][r.dataset.dataset_id]["config_file"] for r in results)
    seeds = {meta["datasets"][r.dataset.dataset_id]["effective"]["bootstrap"]["seed"] for r in results}
    parts.append("From this folder:")
    parts.append("")
    parts.append("```bash")
    parts.append(f"{tools['python_executable']} run_test.py {cfgs} --out .")
    parts.append("```")
    parts.append("")
    parts.append("Tests and static checks:")
    parts.append("")
    parts.append("```bash")
    parts.append(f"{tools['python_executable']} -m pytest -q tests")
    parts.append(f"{tools['python_executable']} -m ruff check . && {tools['python_executable']} -m mypy .")
    parts.append("```")
    parts.append("")
    parts.append(
        f"* Environment: Python {tools['python'].split()[0]}, NumPy {tools['numpy']}, SciPy {tools['scipy']}, "
        f"Matplotlib {tools['matplotlib']}, ViennaRNA {tools['ViennaRNA']}, PyTorch {tools['torch']} (MXfold2 backend, CPU). "
        "Full `pip freeze`, model versions, binary paths and every parameter: `run_metadata.json`."
    )
    parts.append(
        f"* Bootstrap seed: {', '.join(str(s) for s in sorted(seeds))} (`numpy.random.default_rng`). "
        "MXfold2 runs with `--seed 0` on CPU."
    )
    parts.append(f"* Run: {meta['run_started_utc']} → {meta['run_finished_utc']} (UTC).")
    parts.append("")
    parts.append("### Files")
    parts.append("")
    parts.append("| File | Content |")
    parts.append("|---|---|")
    for f, d in [
        ("run_test.py", "CLI entry point"),
        ("dmswin_data.py", "Dataset dataclass, sequence/reagent handling, RDAT and CSV/TSV loaders"),
        ("dmswin_models.py", "RNAfold, CONTRAfold, MXfold2 wrappers (sequence → u, metadata)"),
        ("dmswin_analysis.py", "windows, evaluation set, Spearman, block bootstrap, suggested W"),
        ("dmswin_plots.py", "figures"),
        ("dmswin_report.py", "this README"),
        ("config_*.json", "dataset configs (all dataset-specific choices)"),
        ("*.rdat, *.provenance.json", "downloaded data + URL/date/SHA-256"),
        ("summary.csv", "one row per (model, W): window, ρ_W, ρ_N, Δρ with CIs, convergence metrics"),
        ("summary_wide.csv", "rows = W, columns per model"),
        (
            "per_position.csv",
            "one row per (model, W, position): reactivity, u_W, u_N, |Δu|, flags in_R / in_E / masked",
        ),
        ("run_metadata.json", "versions, parameters, windows, E, provenance, condition flags"),
        ("fig_*.png", "figures (300 dpi)"),
        ("tests/", "pytest suite"),
    ]:
        parts.append(f"| `{f}` | {d} |")
    parts.append("")
    return "\n".join(parts)
