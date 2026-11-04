"""Loaders, index conversion, sequence validation and reagent mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dmswin_analysis as an
import dmswin_data as dd
import dmswin_models as dm


# ---- source numbering -> 0-based ------------------------------------------ #
def test_seqpos_to_index_basic() -> None:
    idx = dd.seqpos_to_index([11, 12, 31], offset=10, n=21)
    assert idx.tolist() == [0, 1, 20]


@pytest.mark.parametrize("seqpos", [[10], [32]])
def test_seqpos_out_of_range(seqpos: list[int]) -> None:
    with pytest.raises(ValueError, match="out of range"):
        dd.seqpos_to_index(seqpos, offset=10, n=21)


def test_seqpos_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicated"):
        dd.seqpos_to_index([12, 12], offset=10, n=21)


def test_rdat_mapping(toy_rdat: Path) -> None:
    with pytest.warns(UserWarning, match="could not be applied"):
        rows = dd.load_rdat(toy_rdat)
    assert len(rows) == 3
    wt = dd.select_dataset(rows, annotation={"mutation": "WT"})
    assert wt.sequence == "GGAAACGCGAAAGCGCAAACC"
    assert wt.n == 21
    # SEQPOS 13..18 -> indices 2..7; all others NaN
    r = wt.reactivity
    assert np.isnan(r[:2]).all() and np.isnan(r[8:]).all()
    assert r[2:5].tolist() == [0.5, 0.6, 0.7]
    assert np.isnan(r[6])  # NaN in file stays NaN, not 0
    assert wt.source_positions is not None and wt.source_positions[0] == 11
    assert wt.source_to_index(13) == 2
    assert wt.reagent == "DMS"
    assert wt.conditions.temperature_c == 24.0 and wt.conditions.mg_mm == 10.0
    mut = dd.select_dataset(rows, row=2)
    assert mut.sequence[2] == "U"  # A13U applied (index 13-10-1 = 2)
    assert dd.select_dataset(rows, row=3).metadata["sequence_notes"]


def test_rdat_letter_mismatch(tmp_path: Path) -> None:
    bad = tmp_path / "bad.rdat"
    bad.write_text("SEQUENCE\tGGAAAC\nOFFSET\t0\nSEQPOS\tC1\tG2\nREACTIVITY:1\t0.1\t0.2\n")
    with pytest.raises(ValueError, match="SEQPOS letter"):
        dd.load_rdat(bad)


def test_rdat_length_mismatch(tmp_path: Path) -> None:
    bad = tmp_path / "bad.rdat"
    bad.write_text("SEQUENCE\tGGAAAC\nOFFSET\t0\nSEQPOS\t1\t2\t3\nREACTIVITY:1\t0.1\t0.2\n")
    with pytest.raises(ValueError, match="SEQPOS entries"):
        dd.load_rdat(bad)


def test_select_ambiguous(toy_rdat: Path) -> None:
    with pytest.warns(UserWarning):
        rows = dd.load_rdat(toy_rdat)
    with pytest.raises(ValueError, match="matched 3"):
        dd.select_dataset(rows)
    with pytest.raises(ValueError, match="matched 0"):
        dd.select_dataset(rows, annotation={"mutation": "nope"})


# ---- CSV/TSV loader -------------------------------------------------------- #
def _write_table(path: Path, sep: str, constructs: dict[str, tuple[str, list[float]]]) -> None:
    lines = [sep.join(["construct", "position", "nucleotide", "reactivity"])]
    for c, (seq, vals) in constructs.items():
        for i, (nt, v) in enumerate(zip(seq, vals, strict=True), start=1):
            lines.append(sep.join([c, str(i), nt, "" if np.isnan(v) else str(v)]))
    path.write_text("\n".join(lines) + "\n")


def test_table_loader_runs_through_analysis(tmp_path: Path) -> None:
    seq = "GGGGCGCAAAAGCGCCCCAAAAAAAAACAACAAC"
    vals = [0.05] * 7 + [1.0, 0.9, 0.8, 0.7] + [0.05] * 7 + [0.6] * 10 + [np.nan] * 6
    p = tmp_path / "toy.tsv"
    _write_table(p, "\t", {"c1": (seq, vals), "c2": (seq.lower().replace("u", "t"), vals)})
    dsets = dd.load("tsv", p, reagent="DMS", temperature_c=37.0)
    assert len(dsets) == 2  # several constructs
    assert dsets[0].sequence == dsets[1].sequence
    ds = dsets[0]
    assert isinstance(ds, dd.Dataset) and np.isnan(ds.reactivity[-6:]).all()
    res = an.run_dataset(
        ds,
        region=(4, 14),
        anchoring="centered",
        grid=[11, 21],
        eval_nts=dd.evaluation_nucleotides(ds.reagent),
        models={"RNAfold": dm.RNAfoldModel()},
        epsilon=0.05,
        n_boot=50,
        block_length=3,
    )
    assert {r["W"] for r in res.models["RNAfold"].rows} == {11, 21, len(seq)}


def test_table_noncontiguous(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("position,nucleotide,reactivity\n1,G,0.1\n3,A,0.2\n")
    with pytest.raises(ValueError, match="contiguous"):
        dd.load("csv", p)


def test_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        dd.load("xlsx", tmp_path / "x")


# ---- reagent mapping ------------------------------------------------------- #
def test_reagent_mapping() -> None:
    assert dd.evaluation_nucleotides("DMS") == frozenset("AC")
    assert dd.evaluation_nucleotides("dms") == frozenset("AC")
    assert dd.evaluation_nucleotides("1M7") == frozenset("ACGU")
    assert dd.evaluation_nucleotides("SHAPE-MaP") == frozenset("ACGU")
    assert dd.evaluation_nucleotides("DMS", override="ACG") == frozenset("ACG")
    assert dd.evaluation_nucleotides(None, override=["a", "t"]) == frozenset("AU")
    with pytest.raises(ValueError, match="Unknown reagent"):
        dd.evaluation_nucleotides("kethoxal")


# ---- sequence edge cases --------------------------------------------------- #
def test_normalize_sequence() -> None:
    assert dd.normalize_sequence("acgt u\n") == "ACGUU"
    with pytest.raises(ValueError, match="Invalid"):
        dd.normalize_sequence("ACGN")
    with pytest.raises(ValueError, match="Empty"):
        dd.normalize_sequence("  ")


def test_dataset_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        dd.Dataset("x", "ACGU", np.zeros(3))


def test_all_nan_error() -> None:
    ds = dd.Dataset("x", "ACACACAC", np.full(8, np.nan), reagent="DMS")
    with pytest.raises(ValueError, match="all reactivities are NaN"):
        an.evaluation_set(ds, (0, 7), frozenset("AC"))


def test_mask_sets_nan() -> None:
    ds = dd.Dataset("x", "ACACACAC", np.ones(8), reagent="DMS").with_mask([6, 7])
    assert np.isnan(ds.reactivity[6:]).all() and ds.metadata["masked_indices"] == [6, 7]
