"""Window construction, evaluation set, agreement metrics and block bootstrap.

Math is 1-based; this module is 0-based throughout (windows are inclusive
``(a, b)`` index pairs, the region of interest is an inclusive ``(r0, r1)``
pair). Conversion to 1-based happens only in tables, figures and the report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from dmswin_data import Dataset, FloatArray, IntArray
from dmswin_models import FoldingModel

Anchoring = Literal["centered", "five_prime"]


# --------------------------------------------------------------------------- #
# Windows and evaluation set
# --------------------------------------------------------------------------- #
def make_window(n: int, region: tuple[int, int], w: int, anchoring: Anchoring) -> tuple[int, int]:
    """Return the inclusive 0-based window ``(a, b)`` of size ``w``.

    Parameters
    ----------
    n
        Sequence length.
    region
        Inclusive 0-based region of interest ``(r0, r1)``.
    w
        Window size.
    anchoring
        ``"centered"``: extend symmetrically around the region (extra nt on
        the 3' side when odd), shifted to stay inside the sequence.
        ``"five_prime"``: ``a = 0``, ``b = w - 1``.

    Raises
    ------
    ValueError
        If ``w`` is outside ``[len(R), n]`` or the window does not contain R.
    """
    r0, r1 = region
    if not 0 <= r0 <= r1 < n:
        raise ValueError(f"Region {region} outside sequence of length {n}.")
    lr = r1 - r0 + 1
    if not lr <= w <= n:
        raise ValueError(f"Window size {w} outside [len(R)={lr}, N={n}].")
    if anchoring == "five_prime":
        a, b = 0, w - 1
    elif anchoring == "centered":
        extra = w - lr
        a = r0 - extra // 2
        b = a + w - 1
        if a < 0:
            a, b = 0, w - 1
        if b > n - 1:
            a, b = n - w, n - 1
    else:
        raise ValueError(f"Unknown anchoring {anchoring!r}.")
    if not (a <= r0 and b >= r1):
        raise ValueError(
            f"Region (1-based {r0 + 1}-{r1 + 1}) is not inside window W={w} "
            f"(1-based {a + 1}-{b + 1}) with anchoring {anchoring!r}."
        )
    return a, b


def min_window(n: int, region: tuple[int, int], anchoring: Anchoring) -> int:
    """Smallest window size that contains the region under the anchoring."""
    r0, r1 = region
    return r1 + 1 if anchoring == "five_prime" else r1 - r0 + 1


def default_grid(n: int, region: tuple[int, int], anchoring: Anchoring, step: int) -> list[int]:
    """``W_min``, then every multiple of ``step`` above it, plus ``n``.

    Aligning to multiples of ``step`` makes window sizes shared across
    datasets, so the across-RNA test can pair them.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    wmin = min_window(n, region, anchoring)
    first = -(-wmin // step) * step
    grid = [wmin, *range(first if first > wmin else first + step, n, step), n]
    return sorted(set(grid))


def evaluation_set(ds: Dataset, region: tuple[int, int], nts: frozenset[str]) -> IntArray:
    """``E = {i in R : s_i in nts, r_i finite}`` as sorted 0-based indices.

    Raises
    ------
    ValueError
        If E is empty (e.g. all reactivities NaN).
    """
    r0, r1 = region
    if not 0 <= r0 <= r1 < ds.n:
        raise ValueError(f"Region {region} outside sequence of length {ds.n}.")
    if not np.any(np.isfinite(ds.reactivity)):
        raise ValueError(f"[{ds.dataset_id}] all reactivities are NaN.")
    idx = [i for i in range(r0, r1 + 1) if ds.sequence[i] in nts and np.isfinite(ds.reactivity[i])]
    if len(idx) < 3:
        raise ValueError(
            f"[{ds.dataset_id}] evaluation set has {len(idx)} positions (< 3); "
            "check region, evaluation nucleotides and masking."
        )
    return np.asarray(idx, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def spearman(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Spearman rho with average ranks for ties (``scipy.stats.spearmanr``).

    Returns NaN if either input is constant.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("spearman needs two 1-D arrays of equal length.")
    if np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan")
    return float(stats.spearmanr(xa, ya).statistic)


def _spearman_rows(x: FloatArray, y: FloatArray) -> FloatArray:
    """Row-wise Spearman between ``x`` (..., m) and ``y`` (..., m), average ranks."""
    rx = stats.rankdata(x, axis=-1)
    ry = stats.rankdata(y, axis=-1)
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    num = (rx * ry).sum(axis=-1)
    den = np.sqrt((rx**2).sum(axis=-1) * (ry**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(den > 0, num / den, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Block bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_indices(
    e_idx: IntArray,
    region: tuple[int, int],
    block_length: int,
    n_boot: int,
    rng: np.random.Generator,
) -> IntArray:
    """Circular block bootstrap over nucleotide positions of the region.

    Blocks of ``block_length`` consecutive nucleotides are drawn with
    uniformly random starts in R (wrapping around inside R); the E positions
    they contain are concatenated until ``len(E)`` positions are collected.

    Returns
    -------
    ndarray of shape (n_boot, len(E))
        Indices into ``e_idx`` (i.e. 0..len(E)-1).
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1.")
    r0, r1 = region
    lr = r1 - r0 + 1
    m = len(e_idx)
    pos_to_k = {int(p) - r0: k for k, p in enumerate(e_idx)}
    # members[s] = E-indices in block starting at relative position s (circular)
    members = [
        [pos_to_k[(s + t) % lr] for t in range(min(block_length, lr)) if (s + t) % lr in pos_to_k]
        for s in range(lr)
    ]
    out = np.empty((n_boot, m), dtype=np.int64)
    for b in range(n_boot):
        buf: list[int] = []
        while len(buf) < m:
            buf.extend(members[int(rng.integers(lr))])
        out[b] = buf[:m]
    return out


def percentile_ci(x: FloatArray, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI ignoring NaN."""
    lo = (1 - level) / 2 * 100
    if not np.any(np.isfinite(x)):
        return float("nan"), float("nan")
    a, b = np.nanpercentile(x, [lo, 100 - lo])
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Main per-dataset analysis
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    """Results of one model on one dataset."""

    name: str
    description: dict[str, Any]
    u_full: FloatArray
    u_windows: dict[int, FloatArray]  # W -> length-N array, NaN outside window
    rows: list[dict[str, Any]] = field(default_factory=list)
    call_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DatasetResult:
    """Results of all models on one dataset."""

    dataset: Dataset
    region: tuple[int, int]
    anchoring: Anchoring
    windows: dict[int, tuple[int, int]]
    e_idx: IntArray
    eval_nts: frozenset[str]
    epsilon: float
    models: dict[str, ModelResult]
    bootstrap: dict[str, Any]
    suggestions: dict[str, dict[str, Any]]


def run_dataset(
    ds: Dataset,
    region: tuple[int, int],
    anchoring: Anchoring,
    grid: Sequence[int],
    eval_nts: frozenset[str],
    models: Mapping[str, FoldingModel],
    epsilon: float,
    *,
    n_boot: int = 2000,
    block_length: int = 10,
    seed: int = 0,
    ci_level: float = 0.95,
) -> DatasetResult:
    """Run the full window test on one dataset.

    Parameters
    ----------
    ds
        Dataset (reactivity NaN where there is no data).
    region
        Inclusive 0-based region of interest.
    anchoring
        Window anchoring.
    grid
        Window sizes; ``N`` is always added (full fold).
    eval_nts
        Evaluation nucleotides.
    models
        ``name -> FoldingModel``.
    epsilon
        Tolerance for the suggested window: smallest W with
        ``delta_rho(W) <= epsilon``.
    n_boot, block_length, seed, ci_level
        Paired block bootstrap settings.
    """
    n = ds.n
    ws = sorted(set(int(w) for w in grid) | {n})
    windows = {w: make_window(n, region, w, anchoring) for w in ws}  # validates R in window
    e_idx = evaluation_set(ds, region, eval_nts)
    r_e = ds.reactivity[e_idx]
    rng = np.random.default_rng(seed)
    boot = block_bootstrap_indices(e_idx, region, block_length, n_boot, rng)

    results: dict[str, ModelResult] = {}
    suggestions: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        u_full, md_full = model.u(ds.sequence)
        if u_full.shape != (n,) or np.any((u_full < 0) | (u_full > 1)):
            raise RuntimeError(f"{name}: invalid u for full sequence.")
        mr = ModelResult(name=name, description=model.describe(), u_full=u_full, u_windows={})
        mr.call_metadata[n] = md_full
        for w, (a, b) in windows.items():
            if w == n:
                uw_local = u_full.copy()
            else:
                uw_local, mdw = model.u(ds.sequence[a : b + 1])
                mr.call_metadata[w] = mdw
            full = np.full(n, np.nan)
            full[a : b + 1] = uw_local
            mr.u_windows[w] = full
        # E must be inside every window (guaranteed by make_window); assert anyway
        for w, uw in mr.u_windows.items():
            if not np.all(np.isfinite(uw[e_idx])):
                raise RuntimeError(f"{name}: E not inside window W={w}.")

        un_e = u_full[e_idx]
        rho_n = spearman(un_e, r_e)
        uw_mat = np.stack([mr.u_windows[w][e_idx] for w in ws])  # (nW, m)
        # bootstrap: (n_boot, m) resamples, same for every W and every model
        rb = r_e[boot]
        rho_n_b = _spearman_rows(un_e[boot], rb)
        for k, w in enumerate(ws):
            a, b = windows[w]
            uw_e = uw_mat[k]
            rho_w = spearman(uw_e, r_e)
            rho_w_b = _spearman_rows(uw_e[boot], rb)
            d_b = rho_n_b - rho_w_b
            conv_b = _spearman_rows(uw_e[boot], un_e[boot])
            du = np.abs(uw_e - un_e)
            mr.rows.append(
                {
                    "dataset_id": ds.dataset_id,
                    "model": name,
                    "W": w,
                    "a_1based": a + 1,
                    "b_1based": b + 1,
                    "a_source": int(ds.source_positions[a]),  # type: ignore[index]
                    "b_source": int(ds.source_positions[b]),  # type: ignore[index]
                    "n_E": len(e_idx),
                    "rho_W": rho_w,
                    "rho_W_ci_lo": percentile_ci(rho_w_b, ci_level)[0],
                    "rho_W_ci_hi": percentile_ci(rho_w_b, ci_level)[1],
                    "rho_N": rho_n,
                    "rho_N_ci_lo": percentile_ci(rho_n_b, ci_level)[0],
                    "rho_N_ci_hi": percentile_ci(rho_n_b, ci_level)[1],
                    "delta_rho": rho_n - rho_w,
                    "delta_rho_ci_lo": percentile_ci(d_b, ci_level)[0],
                    "delta_rho_ci_hi": percentile_ci(d_b, ci_level)[1],
                    "conv_spearman": spearman(uw_e, un_e),
                    "conv_spearman_ci_lo": percentile_ci(conv_b, ci_level)[0],
                    "conv_spearman_ci_hi": percentile_ci(conv_b, ci_level)[1],
                    "max_abs_du": float(du.max()),
                    "mean_abs_du": float(du.mean()),
                    "n_boot_valid": int(np.sum(np.isfinite(d_b))),
                }
            )
        suggestions[name] = suggest_window(mr.rows, epsilon)
        results[name] = mr

    return DatasetResult(
        dataset=ds,
        region=region,
        anchoring=anchoring,
        windows=windows,
        e_idx=e_idx,
        eval_nts=eval_nts,
        epsilon=epsilon,
        models=results,
        bootstrap={
            "n_boot": n_boot,
            "block_length_nt": block_length,
            "seed": seed,
            "ci_level": ci_level,
            "scheme": "paired circular block bootstrap over positions of R; E members kept",
        },
        suggestions=suggestions,
    )


def suggest_window(rows: Sequence[Mapping[str, Any]], epsilon: float) -> dict[str, Any]:
    """Heuristic window choice from the per-W rows of one model.

    Returns
    -------
    dict
        ``W_first``: smallest W with ``delta_rho(W) <= epsilon`` (the
        criterion of the test). ``W_stable``: smallest W such that every
        larger W in the grid also satisfies it. ``W_ci``: smallest W whose
        upper CI bound of ``delta_rho`` is ``<= epsilon`` (conservative).
    """
    srt = sorted(rows, key=lambda r: r["W"])
    ok = [bool(r["delta_rho"] <= epsilon) for r in srt]
    w_first = next((r["W"] for r, o in zip(srt, ok, strict=True) if o), None)
    w_stable = None
    for k in range(len(srt)):
        if all(ok[k:]):
            w_stable = srt[k]["W"]
            break
    w_ci = next((r["W"] for r in srt if r["delta_rho_ci_hi"] <= epsilon), None)
    return {"epsilon": epsilon, "W_first": w_first, "W_stable": w_stable, "W_ci": w_ci}


def cross_dataset_tests(results: Sequence[DatasetResult]) -> list[dict[str, Any]]:
    """Per model and per W shared by all datasets: Wilcoxon signed-rank on delta_rho.

    Only computed with at least two datasets; p-values with few RNAs are
    necessarily coarse.
    """
    if len(results) < 2:
        return []
    out: list[dict[str, Any]] = []
    model_names = set.intersection(*(set(r.models) for r in results))
    common_w = set.intersection(*(set(r.windows) for r in results))
    for name in sorted(model_names):
        for w in sorted(common_w):
            d = []
            for r in results:
                row = next(x for x in r.models[name].rows if x["W"] == w)
                d.append(row["delta_rho"])
            arr = np.asarray(d)
            if np.allclose(arr, 0):
                p = float("nan")
            else:
                p = float(stats.wilcoxon(arr, zero_method="wilcox").pvalue)
            out.append(
                {
                    "model": name,
                    "W": w,
                    "n_datasets": len(d),
                    "median_delta_rho": float(np.median(arr)),
                    "wilcoxon_p": p,
                }
            )
    return out
