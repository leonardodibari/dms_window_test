"""Shared fixtures. The library modules live in the parent folder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dmswin_models as dm

CONTRAFOLD = os.environ.get("DMSWIN_CONTRAFOLD", "contrafold")
MXFOLD2 = os.environ.get("DMSWIN_MXFOLD2", "mxfold2")


def make_model(key: str) -> dm.FoldingModel:
    """Build a model with test defaults (executables overridable via env vars)."""
    if key == "rnafold":
        return dm.RNAfoldModel(temperature_c=37.0)
    if key == "contrafold":
        return dm.ContrafoldModel(executable=CONTRAFOLD, posterior_threshold=1e-5)
    return dm.MXfold2Model(executable=MXFOLD2)


@pytest.fixture(scope="session", params=["rnafold", "contrafold", "mxfold2"])
def model(request: pytest.FixtureRequest) -> dm.FoldingModel:
    """Each of the three folding models (cached)."""
    return dm.CachedModel(make_model(request.param))


TOY_RDAT = """VERSION\t0.34
NAME\ttoy
SEQUENCE\tggaaacgcgaaagcgcaaacc
OFFSET\t10
ANNOTATION\tmodifier:DMS\ttemperature:24C\tchemical:MgCl2:10mM\tchemical:HEPES:50mM(pH8.0)
SEQPOS\tA13\tA14\tA15\tC16\tG17\tC18
ANNOTATION_DATA:1\tmutation:WT
ANNOTATION_DATA:2\tmutation:A13U
ANNOTATION_DATA:3\tmutation:A14X
REACTIVITY:1\t0.5\t0.6\t0.7\t0.1\tNaN\t0.2
REACTIVITY:2\t0.4\t0.6\t0.7\t0.1\t0.0\t0.2
REACTIVITY:3\t0.4\t0.6\t0.7\t0.1\t0.0\t0.2
"""


@pytest.fixture
def toy_rdat(tmp_path: Path) -> Path:
    """A 21-nt toy RDAT with OFFSET 10 and three rows (WT + two mutants)."""
    p = tmp_path / "toy.rdat"
    p.write_text(TOY_RDAT)
    return p
