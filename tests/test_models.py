"""Checks run for each of the three folding models."""

from __future__ import annotations

import numpy as np
import pytest

import dmswin_models as dm

HAIRPIN = "GGGGCGCAAAAGCGCCCC"  # 7-bp stem, 4-nt loop (positions 8-11, 1-based)
ARM, PARTNER = "GGAGGGAG", "CUCCCUCC"
LONG_RANGE = ARM + "A" * 30 + PARTNER + "A" * 20  # stem arms 1-8 and 39-46


def test_u_range_and_length(model: dm.FoldingModel) -> None:
    for seq in (HAIRPIN, LONG_RANGE, "acgUacguT"):
        u, md = model.u(seq)
        assert isinstance(u, np.ndarray)
        assert u.shape == (len(seq),)
        assert np.all((u >= 0) & (u <= 1))
        assert isinstance(md, dict)


def test_describe(model: dm.FoldingModel) -> None:
    d = model.describe()
    assert d["version"] and d["version"] != "unknown"
    assert "can_represent" in d


def test_hairpin_sign(model: dm.FoldingModel) -> None:
    u, _ = model.u(HAIRPIN)
    loop = u[7:11].mean()
    stem = np.r_[u[1:6], u[12:17]].mean()  # inner stem pairs
    assert loop > stem + 0.5


def test_truncation(model: dm.FoldingModel) -> None:
    u_full, _ = model.u(LONG_RANGE)
    # small window: 5' arm present, partner (39-46) outside -> arm becomes unpaired
    u_small, _ = model.u(LONG_RANGE[:20])
    assert np.max(np.abs(u_small[:8] - u_full[:8])) > 0.5
    # window holding both arms converges to the full fold
    u_both, _ = model.u(LONG_RANGE[:46])
    assert np.max(np.abs(u_both - u_full[:46])) < 0.1


def test_very_short_window(model: dm.FoldingModel) -> None:
    u, _ = model.u("ACG")
    assert u.shape == (3,) and np.allclose(u, 1.0, atol=1e-2)  # no stable pair possible


def test_invalid_sequence(model: dm.FoldingModel) -> None:
    with pytest.raises(ValueError, match="Invalid"):
        model.u("ACGX")


def test_build_model_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        dm.build_model("foo", {})


def test_contrafold_threshold_validation() -> None:
    with pytest.raises(ValueError):
        dm.ContrafoldModel(posterior_threshold=0)
