"""Windows, evaluation set, Spearman, bootstrap, suggestion rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_model

import dmswin_analysis as an
import dmswin_data as dd
import dmswin_models as dm


def _dataset(n: int = 60, seed: int = 1) -> dd.Dataset:
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGU"), n))
    r = rng.random(n)
    r[-5:] = np.nan
    return dd.Dataset("toy", seq, r, reagent="DMS")


# ---- windows ---------------------------------------------------------------- #
def test_five_prime_window() -> None:
    assert an.make_window(100, (10, 30), 31, "five_prime") == (0, 30)
    assert an.make_window(100, (10, 30), 50, "five_prime") == (0, 49)


def test_centered_window_and_clipping() -> None:
    assert an.make_window(100, (40, 59), 30, "centered") == (35, 64)
    # near the 5' end: shifted to start at 0, size kept
    assert an.make_window(100, (2, 11), 40, "centered") == (0, 39)
    # near the 3' end
    assert an.make_window(100, (90, 99), 30, "centered") == (70, 99)


def test_region_outside_window_raises() -> None:
    with pytest.raises(ValueError, match="not inside window"):
        an.make_window(100, (10, 30), 25, "five_prime")
    with pytest.raises(ValueError, match="outside"):
        an.make_window(100, (10, 30), 10, "centered")  # W < len(R)
    with pytest.raises(ValueError, match="outside"):
        an.make_window(100, (10, 30), 101, "centered")  # W > N


def test_default_grid() -> None:
    g = an.default_grid(100, (10, 30), "five_prime", 20)
    assert g[0] == 31 and g[-1] == 100 and all(np.diff(g) > 0)
    assert g == [31, 40, 60, 80, 100]  # aligned to multiples of the step
    g = an.default_grid(100, (10, 30), "centered", 20)
    assert g[0] == 21
    assert an.default_grid(50, (0, 9), "centered", 10) == [10, 20, 30, 40, 50]


# ---- evaluation set --------------------------------------------------------- #
def test_evaluation_set_and_identical_across_models() -> None:
    ds = _dataset()
    e = an.evaluation_set(ds, (10, 40), frozenset("AC"))
    assert all(ds.sequence[i] in "AC" and 10 <= i <= 40 for i in e)
    assert np.all(np.isfinite(ds.reactivity[e]))
    hot = dm.RNAfoldModel(temperature_c=50.0)
    models = {"RNAfold": make_model("rnafold"), "RNAfold_50C": hot}
    res = an.run_dataset(
        ds, (10, 40), "centered", [31, 41, 51], frozenset("AC"), models, 0.05, n_boot=30, block_length=5
    )
    assert np.array_equal(res.e_idx, e)
    n_e = {r["n_E"] for mr in res.models.values() for r in mr.rows}
    assert n_e == {len(e)}
    for mr in res.models.values():
        for w, uw in mr.u_windows.items():
            assert np.all(np.isfinite(uw[e])), w


def test_nan_excluded_from_e() -> None:
    ds = _dataset()
    e = an.evaluation_set(ds, (0, ds.n - 1), frozenset("ACGU"))
    assert not set(e.tolist()) & set(range(ds.n - 5, ds.n))


def test_run_dataset_rejects_region_outside_window() -> None:
    ds = _dataset()
    with pytest.raises(ValueError, match="not inside window"):
        an.run_dataset(
            ds,
            (10, 40),
            "five_prime",
            [35],
            frozenset("AC"),
            {"RNAfold": make_model("rnafold")},
            0.05,
            n_boot=10,
        )


# ---- Spearman ---------------------------------------------------------------- #
def test_spearman_heavy_ties() -> None:
    # ranks x = [2, 2, 2, 4.5, 4.5], y = [1..5]
    # Pearson of ranks = 7.5 / sqrt(7.5 * 10) = sqrt(3)/2
    x = [1, 1, 1, 2, 2]
    y = [1, 2, 3, 4, 5]
    assert an.spearman(x, y) == pytest.approx(math.sqrt(3) / 2, abs=1e-12)
    rows = an._spearman_rows(np.array([x], float), np.array([y], float))
    assert rows[0] == pytest.approx(math.sqrt(3) / 2, abs=1e-12)


def test_spearman_constant_is_nan() -> None:
    assert math.isnan(an.spearman([1, 1, 1], [1, 2, 3]))


def test_spearman_rows_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=(20, 15)).astype(float)
    y = rng.random((20, 15))
    v = an._spearman_rows(x, y)
    for k in range(20):
        assert v[k] == pytest.approx(an.spearman(x[k], y[k]), abs=1e-12)


# ---- bootstrap ------------------------------------------------------------- #
def test_bootstrap_reproducible() -> None:
    e = np.array([10, 12, 13, 17, 20, 21, 25, 29, 30])
    b1 = an.block_bootstrap_indices(e, (10, 30), 4, 100, np.random.default_rng(7))
    b2 = an.block_bootstrap_indices(e, (10, 30), 4, 100, np.random.default_rng(7))
    b3 = an.block_bootstrap_indices(e, (10, 30), 4, 100, np.random.default_rng(8))
    assert b1.shape == (100, len(e))
    assert np.array_equal(b1, b2) and not np.array_equal(b1, b3)
    assert b1.min() >= 0 and b1.max() < len(e)


def test_bootstrap_ci_reproducible() -> None:
    ds = _dataset()
    m = {"RNAfold": make_model("rnafold")}
    r1 = an.run_dataset(ds, (10, 40), "centered", [31, 45], frozenset("AC"), m, 0.05, n_boot=100, seed=3)
    r2 = an.run_dataset(ds, (10, 40), "centered", [31, 45], frozenset("AC"), m, 0.05, n_boot=100, seed=3)
    keys = ["rho_W_ci_lo", "rho_W_ci_hi", "delta_rho_ci_lo", "delta_rho_ci_hi"]
    for a, b in zip(r1.models["RNAfold"].rows, r2.models["RNAfold"].rows, strict=True):
        assert all(a[k] == b[k] for k in keys)
    full = next(r for r in r1.models["RNAfold"].rows if r["W"] == ds.n)
    assert full["delta_rho"] == 0 and full["delta_rho_ci_lo"] == 0 == full["delta_rho_ci_hi"]


# ---- suggestion rule -------------------------------------------------------- #
def test_suggest_window() -> None:
    rows = [
        {"W": 10, "delta_rho": 0.01, "delta_rho_ci_hi": 0.2},
        {"W": 20, "delta_rho": 0.10, "delta_rho_ci_hi": 0.3},
        {"W": 30, "delta_rho": 0.00, "delta_rho_ci_hi": 0.01},
        {"W": 40, "delta_rho": 0.00, "delta_rho_ci_hi": 0.00},
    ]
    s = an.suggest_window(rows, 0.02)
    assert s == {"epsilon": 0.02, "W_first": 10, "W_stable": 30, "W_ci": 30}


def test_cross_dataset_wilcoxon() -> None:
    m = {"RNAfold": make_model("rnafold")}
    res = [
        an.run_dataset(_dataset(seed=s), (10, 40), "centered", [31], frozenset("AC"), m, 0.05, n_boot=20)
        for s in (1, 2, 3)
    ]
    out = an.cross_dataset_tests(res)
    assert {r["W"] for r in out} == {31, 60}
    assert an.cross_dataset_tests(res[:1]) == []
