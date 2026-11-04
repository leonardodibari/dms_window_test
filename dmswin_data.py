"""Dataset objects, sequence normalisation, reagent mapping and file loaders.

Loaders turn a source file into a list of :class:`Dataset` objects (one per
construct / data row). They are registered in :data:`LOADERS` by format name,
so new formats can be added without touching the analysis code.

Index convention: everything stored in a :class:`Dataset` uses 0-based indices
into ``Dataset.sequence``. The original numbering of the source is kept in
``Dataset.source_positions`` (one integer per nucleotide) so that configs and
figures can refer to it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import re
import urllib.request
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

RNA_ALPHABET = frozenset("ACGU")

# Reagent -> nucleotides whose reactivity reports on pairing status.
# Keys are matched case-insensitively. Can be overridden from the config.
REAGENT_NUCLEOTIDES: dict[str, frozenset[str]] = {
    "dms": frozenset("AC"),
    "shape": frozenset("ACGU"),
    "1m7": frozenset("ACGU"),
    "1m6": frozenset("ACGU"),
    "nmia": frozenset("ACGU"),
    "nai": frozenset("ACGU"),
    "nai-n3": frozenset("ACGU"),
    "2a3": frozenset("ACGU"),
    "bzcn": frozenset("ACGU"),
}


# --------------------------------------------------------------------------- #
# Sequence handling
# --------------------------------------------------------------------------- #
def normalize_sequence(seq: str) -> str:
    """Return an upper-case RNA sequence (T -> U), validating the alphabet.

    Parameters
    ----------
    seq
        Nucleotide sequence, any case, DNA or RNA alphabet. Whitespace is
        removed.

    Returns
    -------
    str
        Sequence over ``{A, C, G, U}``.

    Raises
    ------
    ValueError
        If the sequence is empty or contains characters outside ACGTU.
    """
    s = re.sub(r"\s+", "", seq).upper().replace("T", "U")
    if not s:
        raise ValueError("Empty sequence.")
    bad = sorted(set(s) - RNA_ALPHABET)
    if bad:
        raise ValueError(f"Invalid characters in sequence: {bad} (allowed: A, C, G, U, T).")
    return s


def evaluation_nucleotides(
    reagent: str | None, override: Sequence[str] | str | None = None
) -> frozenset[str]:
    """Nucleotides that enter the evaluation set for a probing reagent.

    Parameters
    ----------
    reagent
        Probing reagent name (e.g. ``"DMS"``, ``"1M7"``).
    override
        Explicit nucleotide set (e.g. ``"AC"`` or ``["A", "C"]``). Takes
        precedence over the reagent mapping.

    Returns
    -------
    frozenset of str

    Raises
    ------
    ValueError
        If neither an override nor a known reagent is given.
    """
    if override is not None:
        nts = frozenset(normalize_sequence("".join(override)))
        return nts
    if reagent is None:
        raise ValueError("No reagent given and no evaluation_nucleotides override.")
    key = reagent.strip().lower()
    if key in REAGENT_NUCLEOTIDES:
        return REAGENT_NUCLEOTIDES[key]
    if "shape" in key:
        return REAGENT_NUCLEOTIDES["shape"]
    raise ValueError(f"Unknown reagent {reagent!r}; set 'evaluation_nucleotides' explicitly in the config.")


# --------------------------------------------------------------------------- #
# Dataset object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Conditions:
    """Experimental conditions parsed from the source (all optional)."""

    temperature_c: float | None = None
    buffer: str | None = None
    mg_mm: float | None = None
    chemicals: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "temperature_c": self.temperature_c,
            "buffer": self.buffer,
            "mg_mM": self.mg_mm,
            "chemicals": list(self.chemicals),
        }


@dataclass(frozen=True)
class Dataset:
    """One probed construct.

    Attributes
    ----------
    dataset_id
        Identifier used in output file names.
    sequence
        5'->3' RNA sequence (A, C, G, U).
    reactivity
        Array of length ``len(sequence)``; NaN where there is no data.
    reagent
        Probing reagent (e.g. ``"DMS"``).
    conditions
        Experimental conditions.
    source_url, download_date
        Provenance.
    source_positions
        Source numbering of each nucleotide (length ``len(sequence)``).
    metadata
        Free-form metadata (annotations, row index, ...).
    """

    dataset_id: str
    sequence: str
    reactivity: FloatArray
    reagent: str | None = None
    conditions: Conditions = field(default_factory=Conditions)
    source_url: str | None = None
    download_date: str | None = None
    source_positions: IntArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", normalize_sequence(self.sequence))
        r = np.asarray(self.reactivity, dtype=float)
        if r.ndim != 1 or r.shape[0] != len(self.sequence):
            raise ValueError(
                f"[{self.dataset_id}] reactivity length {r.shape} != sequence length {len(self.sequence)}."
            )
        object.__setattr__(self, "reactivity", r)
        if self.source_positions is None:
            pos = np.arange(1, len(self.sequence) + 1, dtype=np.int64)
        else:
            pos = np.asarray(self.source_positions, dtype=np.int64)
            if pos.shape != (len(self.sequence),):
                raise ValueError(f"[{self.dataset_id}] source_positions length mismatch.")
        object.__setattr__(self, "source_positions", pos)

    @property
    def n(self) -> int:
        """Sequence length."""
        return len(self.sequence)

    def source_to_index(self, source_pos: int) -> int:
        """Map a source-numbered position to a 0-based index into ``sequence``."""
        assert self.source_positions is not None
        hits = np.flatnonzero(self.source_positions == source_pos)
        if hits.size != 1:
            raise ValueError(
                f"[{self.dataset_id}] source position {source_pos} not found (or ambiguous); "
                f"valid range {int(self.source_positions.min())}..{int(self.source_positions.max())}."
            )
        return int(hits[0])

    def with_mask(self, indices: Sequence[int]) -> Dataset:
        """Return a copy with reactivity set to NaN at the given 0-based indices."""
        r = self.reactivity.copy()
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n):
            raise ValueError(f"[{self.dataset_id}] mask index out of range.")
        r[idx] = np.nan
        md = dict(self.metadata)
        md["masked_indices"] = sorted(set(md.get("masked_indices", [])) | {int(i) for i in idx})
        return Dataset(
            dataset_id=self.dataset_id,
            sequence=self.sequence,
            reactivity=r,
            reagent=self.reagent,
            conditions=self.conditions,
            source_url=self.source_url,
            download_date=self.download_date,
            source_positions=self.source_positions,
            metadata=md,
        )


# --------------------------------------------------------------------------- #
# Download with provenance
# --------------------------------------------------------------------------- #
def ensure_source(path: Path, url: str | None, redownload: bool = False) -> dict[str, Any]:
    """Make sure a public source file exists locally, downloading it if needed.

    A sidecar ``<file>.provenance.json`` stores URL, UTC download date and
    SHA-256. Only use this for public data.

    Returns
    -------
    dict
        Provenance record (url, download_date, sha256, path).
    """
    side = path.with_name(path.name + ".provenance.json")
    if redownload or not path.exists():
        if url is None:
            raise FileNotFoundError(f"{path} does not exist and no URL was given.")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data)
        rec = {
            "url": url,
            "download_date": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "path": path.name,
        }
        side.write_text(json.dumps(rec, indent=2))
        return rec
    if side.exists():
        rec = json.loads(side.read_text())
        rec["sha256_now"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(rec)
    return {
        "url": url,
        "download_date": None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": path.name,
    }


# --------------------------------------------------------------------------- #
# RDAT loader
# --------------------------------------------------------------------------- #
_MUT_RE = re.compile(r"^([ACGUT])(-?\d+)([ACGUT])$", re.IGNORECASE)
_CONC_RE = re.compile(r"^([\d.]+)\s*(mM|M|uM)", re.IGNORECASE)


def _parse_annotations(fields: Sequence[str]) -> dict[str, list[str]]:
    """Parse RDAT ``key:value`` annotation tokens into a multi-dict."""
    out: dict[str, list[str]] = {}
    for raw in fields:
        tok = raw.strip()
        if not tok:
            continue
        key, _, val = tok.partition(":")
        out.setdefault(key, []).append(val)
    return out


def _conc_mm(text: str) -> float | None:
    m = _CONC_RE.match(text)
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2).lower()
    return v * 1000.0 if unit == "m" else (v / 1000.0 if unit == "um" else v)


def conditions_from_annotations(ann: Mapping[str, list[str]]) -> Conditions:
    """Build :class:`Conditions` from RDAT annotations (``temperature``, ``chemical``)."""
    temp: float | None = None
    for t in ann.get("temperature", []):
        m = re.match(r"^\s*([-\d.]+)\s*C?", t, re.IGNORECASE)
        if m:
            temp = float(m.group(1))
    chems = tuple(ann.get("chemical", []))
    mg: float | None = None
    buffer: str | None = None
    for c in chems:
        name, _, conc = c.partition(":")
        if name.lower().startswith("mg"):
            mg = _conc_mm(conc)
        elif buffer is None:
            buffer = c
    return Conditions(temperature_c=temp, buffer=buffer, mg_mm=mg, chemicals=chems)


def _parse_seqpos(tokens: Sequence[str]) -> tuple[list[int], list[str | None]]:
    pos: list[int] = []
    nts: list[str | None] = []
    for t in tokens:
        m = re.match(r"^([A-Za-z]?)(-?\d+)$", t)
        if not m:
            raise ValueError(f"Cannot parse SEQPOS token {t!r}.")
        pos.append(int(m.group(2)))
        nts.append(m.group(1).upper().replace("T", "U") or None)
    return pos, nts


def seqpos_to_index(seqpos: Sequence[int], offset: int, n: int) -> IntArray:
    """Convert RDAT SEQPOS values to 0-based indices into the stored sequence.

    RDAT convention: the first nucleotide of ``SEQUENCE`` has number
    ``OFFSET + 1``, so ``index = SEQPOS - OFFSET - 1``.

    Raises
    ------
    ValueError
        If an index falls outside ``[0, n)`` or indices are duplicated.
    """
    idx = np.asarray(seqpos, dtype=np.int64) - int(offset) - 1
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(
            f"SEQPOS out of range for sequence of length {n} with OFFSET {offset}: "
            f"SEQPOS {min(seqpos)}..{max(seqpos)} -> indices {idx.min()}..{idx.max()}."
        )
    if len(set(idx.tolist())) != idx.size:
        raise ValueError("Duplicated SEQPOS values.")
    return idx


def _apply_mutation(seq: str, muts: Sequence[str], offset: int) -> tuple[str, list[str]]:
    """Apply ``X<pos>Y`` point mutations (source numbering). Returns notes for unparsed ones."""
    s = list(seq)
    notes: list[str] = []
    for mut in muts:
        if mut.upper() == "WT":
            continue
        m = _MUT_RE.match(mut)
        if not m:
            notes.append(f"mutation {mut!r} not applied (unparseable target base)")
            continue
        i = int(m.group(2)) - offset - 1
        wt = m.group(1).upper().replace("T", "U")
        if not 0 <= i < len(s) or s[i] != wt:
            raise ValueError(f"Mutation {mut} inconsistent with sequence/OFFSET.")
        s[i] = m.group(3).upper().replace("T", "U")
    return "".join(s), notes


def load_rdat(path: Path, **_opts: Any) -> list[Dataset]:
    """Load every data row of an RDAT file (format versions 0.2x-0.3x).

    Parameters
    ----------
    path
        RDAT file.

    Returns
    -------
    list of Dataset
        One per ``REACTIVITY:k`` row, with ids ``<NAME-or-stem>__row<k>``.
        Row annotations are in ``metadata["annotations"]``.
    """
    lines = Path(path).read_text().splitlines()
    glob: dict[str, str] = {}
    rows_ann: dict[int, list[str]] = {}
    rows_re: dict[int, list[str]] = {}
    rows_seq: dict[int, str] = {}
    ann_global: list[str] = []
    comments: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        key = parts[0].strip()
        if " " in key and len(parts) == 1:  # space-separated variant
            key, _, rest = line.partition(" ")
            parts = [key, *rest.split()]
        vals = parts[1:]
        kbase, _, knum = key.partition(":")
        if kbase == "ANNOTATION":
            ann_global += vals
        elif kbase == "COMMENT":
            comments.append(" ".join(vals))
        elif kbase == "ANNOTATION_DATA" and knum:
            rows_ann[int(knum)] = vals
        elif kbase == "REACTIVITY" and knum:
            rows_re[int(knum)] = [v for v in " ".join(vals).split() if v]
        elif kbase == "SEQUENCE" and knum:
            rows_seq[int(knum)] = "".join(vals)
        elif not knum:
            glob[kbase] = "\t".join(vals)
    if "SEQUENCE" not in glob and not rows_seq:
        raise ValueError(f"{path}: no SEQUENCE.")
    if not rows_re:
        raise ValueError(f"{path}: no REACTIVITY rows.")
    offset = int(glob.get("OFFSET", "0").strip() or 0)
    base_seq = normalize_sequence(glob["SEQUENCE"]) if "SEQUENCE" in glob else ""
    if "SEQPOS" in glob:
        seqpos, seqpos_nt = _parse_seqpos(glob["SEQPOS"].split())
    else:
        seqpos, seqpos_nt = [], []
    gann = _parse_annotations(ann_global)
    name = glob.get("NAME", Path(path).stem).strip()
    out: list[Dataset] = []
    unapplied: list[int] = []
    for k in sorted(rows_re):
        rann = _parse_annotations(rows_ann.get(k, []))
        merged: dict[str, list[str]] = {kk: list(v) for kk, v in gann.items()}
        for kk, v in rann.items():
            merged[kk] = list(v)  # row-level overrides global
        seq = normalize_sequence(rows_seq[k]) if k in rows_seq else base_seq
        notes: list[str] = []
        if "sequence" in rann:
            seq = normalize_sequence(rann["sequence"][0])
        elif "mutation" in rann:
            seq, notes = _apply_mutation(seq, rann["mutation"], offset)
        n = len(seq)
        values = [float(v) for v in rows_re[k]]
        sp = seqpos if seqpos else list(range(offset + 1, offset + 1 + len(values)))
        if len(sp) != len(values):
            raise ValueError(f"{path} row {k}: {len(values)} reactivities but {len(sp)} SEQPOS entries.")
        idx = seqpos_to_index(sp, offset, n)
        for j, nt in zip(idx, seqpos_nt, strict=False):
            if nt is not None and k not in rows_seq and "mutation" not in rann and seq[j] != nt:
                raise ValueError(f"{path} row {k}: SEQPOS letter {nt} at index {j} != sequence {seq[j]}.")
        r = np.full(n, np.nan)
        r[idx] = values
        reagent_list = merged.get("modifier", [])
        if notes:
            unapplied.append(k)
        out.append(
            Dataset(
                dataset_id=f"{Path(path).stem}__row{k}",
                sequence=seq,
                reactivity=r,
                reagent=reagent_list[0] if reagent_list else None,
                conditions=conditions_from_annotations(merged),
                source_positions=np.arange(offset + 1, offset + 1 + n, dtype=np.int64),
                metadata={
                    "format": "rdat",
                    "rdat_version": glob.get("VERSION", "").strip(),
                    "name": name,
                    "row": k,
                    "offset": offset,
                    "annotations": merged,
                    "row_annotations": rann,
                    "comments": comments,
                    "structure": glob.get("STRUCTURE", "").strip() or None,
                    "sequence_notes": notes,
                },
            )
        )
    if unapplied:
        warnings.warn(
            f"{path}: {len(unapplied)} rows have mutations that could not be applied "
            "(e.g. 'X' = unspecified base); their sequence is the reference sequence and "
            "metadata['sequence_notes'] says so. Do not analyse those rows as-is.",
            stacklevel=2,
        )
    return out


# --------------------------------------------------------------------------- #
# CSV / TSV loader
# --------------------------------------------------------------------------- #
def load_table(path: Path, **opts: Any) -> list[Dataset]:
    """Load a long-format CSV/TSV with one row per nucleotide.

    Required columns: ``position`` (1-based, or source numbering if
    ``position_offset`` is given), ``nucleotide``, ``reactivity`` (empty or
    ``nan`` for missing). Optional column ``construct`` splits the file into
    several datasets. Reagent and conditions come from ``opts``
    (``reagent``, ``temperature_c``, ``mg_mM``, ``buffer``).

    Positions must be contiguous ``1..N`` (after subtracting
    ``position_offset``); otherwise a ``ValueError`` is raised.
    """
    path = Path(path)
    delim = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    offset = int(opts.get("position_offset", 0))
    groups: dict[str, list[tuple[int, str, float]]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        need = {"position", "nucleotide", "reactivity"}
        if reader.fieldnames is None or not need <= set(reader.fieldnames):
            raise ValueError(f"{path}: columns {sorted(need)} required, got {reader.fieldnames}.")
        for row in reader:
            c = (row.get("construct") or path.stem).strip()
            val = row["reactivity"].strip()
            groups.setdefault(c, []).append(
                (int(row["position"]), row["nucleotide"].strip(), float(val) if val else np.nan)
            )
    cond = Conditions(
        temperature_c=opts.get("temperature_c"),
        buffer=opts.get("buffer"),
        mg_mm=opts.get("mg_mM"),
    )
    out: list[Dataset] = []
    for c, recs in groups.items():
        recs.sort()
        pos = [p for p, _, _ in recs]
        idx = np.asarray(pos) - offset - 1
        if idx.tolist() != list(range(len(recs))):
            raise ValueError(f"{path} construct {c}: positions are not contiguous 1..N.")
        out.append(
            Dataset(
                dataset_id=c,
                sequence="".join(nt for _, nt, _ in recs),
                reactivity=np.array([v for _, _, v in recs]),
                reagent=opts.get("reagent"),
                conditions=cond,
                source_positions=np.asarray(pos, dtype=np.int64),
                metadata={"format": "table", "construct": c, "annotations": {}},
            )
        )
    return out


Loader = Callable[..., list[Dataset]]
LOADERS: dict[str, Loader] = {"rdat": load_rdat, "csv": load_table, "tsv": load_table}


def load(fmt: str, path: Path, **opts: Any) -> list[Dataset]:
    """Dispatch to a registered loader."""
    try:
        loader = LOADERS[fmt.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown format {fmt!r}; known: {sorted(LOADERS)}") from exc
    return loader(path, **opts)


def select_dataset(
    datasets: Sequence[Dataset],
    annotation: Mapping[str, str] | None = None,
    row: int | None = None,
    construct: str | None = None,
) -> Dataset:
    """Select exactly one dataset by annotation match, row number, or construct id.

    Raises
    ------
    ValueError
        If zero or several datasets match.
    """
    cands = list(datasets)
    if row is not None:
        cands = [d for d in cands if d.metadata.get("row") == row]
    if construct is not None:
        cands = [d for d in cands if d.metadata.get("construct") == construct]
    if annotation:

        def ok(d: Dataset) -> bool:
            ra = d.metadata.get("row_annotations", d.metadata.get("annotations", {}))
            return all(v in ra.get(k, []) for k, v in annotation.items())

        cands = [d for d in cands if ok(d)]
    if len(cands) != 1:
        raise ValueError(
            f"Selection (annotation={annotation}, row={row}, construct={construct}) matched "
            f"{len(cands)} datasets; exactly one required."
        )
    return cands[0]
