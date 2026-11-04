"""Folding models with a common interface: ``sequence -> (u, metadata)``.

``u_i = 1 - sum_j p_ij`` is the probability that nucleotide ``i`` is unpaired
under the model's base-pair probability (partition function / posterior)
matrix. All three models return a NumPy array of length ``len(sequence)``
with values clipped to ``[0, 1]`` (clipping only removes rounding noise; the
amount clipped is recorded in the metadata).

CLI flags / API calls were checked against the installed versions:

* ViennaRNA 2.7.2 Python bindings: ``RNA.md()`` (attributes ``temperature``,
  ``salt``), ``RNA.params_load_RNA_Turner2004()`` etc., ``RNA.fold_compound``,
  ``fc.pf()``, ``fc.bpp()`` (1-based, upper triangle).
* CONTRAfold 2.0x: ``contrafold predict --posteriors CUTOFF OUTFILE INFILE``;
  output lines ``i base j:p j:p ...`` (1-based, only ``j > i``).
* MXfold2 0.1.2: ``mxfold2 predict --bpp DIR [--param P] [--seed S] FASTA``;
  writes ``DIR/<header>.bpp``, an ``(N+1) x (N+1)`` symmetric matrix with
  a padding row/column 0, printed with 5 decimals.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from dmswin_data import FloatArray, normalize_sequence


class FoldingModel(Protocol):
    """Interface implemented by every folding model."""

    name: str

    def u(self, sequence: str) -> tuple[FloatArray, dict[str, Any]]:
        """Return unpaired probabilities (length ``len(sequence)``) and metadata."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return model version and parameters (JSON-serialisable)."""
        ...


def _u_from_pairs(n: int, pairs: Mapping[tuple[int, int], float]) -> tuple[FloatArray, float]:
    """Compute ``u`` from 0-based ``(i, j)`` pair probabilities (each pair once).

    Returns ``u`` clipped to [0, 1] and the maximum amount clipped.
    """
    paired = np.zeros(n)
    for (i, j), p in pairs.items():
        paired[i] += p
        paired[j] += p
    raw = 1.0 - paired
    u = np.clip(raw, 0.0, 1.0)
    return u, float(np.max(np.abs(raw - u))) if n else 0.0


def _resolve_exe(exe: str) -> str:
    """Resolve an executable: absolute path, PATH lookup, or next to ``sys.executable``."""
    p = Path(exe)
    if p.is_absolute() and p.exists():
        return str(p)
    found = shutil.which(exe)
    if found:
        return os.path.abspath(found)
    sib = Path(os.path.abspath(sys.executable)).parent / exe
    if sib.exists():
        return str(sib)
    raise FileNotFoundError(f"Executable {exe!r} not found.")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# ViennaRNA RNAfold
# --------------------------------------------------------------------------- #
class RNAfoldModel:
    """ViennaRNA partition function (Python bindings of RNAfold -p).

    Parameters
    ----------
    temperature_c
        Folding temperature in Celsius.
    energy_params
        One of ``turner2004`` (default), ``turner1999``, ``andronescu2007``,
        ``langdon2018``, or a path to a ``.par`` file.
    salt_molar
        Monovalent salt concentration (M) for ViennaRNA's salt correction.
        ``None`` (default) leaves the library default (standard 1.021 M,
        i.e. no correction).
    dangles
        Dangle model (ViennaRNA default 2).
    """

    name = "RNAfold"

    _PARAM_LOADERS: ClassVar[dict[str, str]] = {
        "turner2004": "params_load_RNA_Turner2004",
        "turner1999": "params_load_RNA_Turner1999",
        "andronescu2007": "params_load_RNA_Andronescu2007",
        "langdon2018": "params_load_RNA_Langdon2018",
    }

    def __init__(
        self,
        temperature_c: float = 37.0,
        energy_params: str = "turner2004",
        salt_molar: float | None = None,
        dangles: int = 2,
    ) -> None:
        import RNA

        self._RNA = RNA
        self.temperature_c = float(temperature_c)
        self.energy_params = energy_params
        self.salt_molar = salt_molar
        self.dangles = int(dangles)
        md = RNA.md()
        if salt_molar is not None and not hasattr(md, "salt"):
            raise RuntimeError("Installed ViennaRNA does not support salt correction.")
        self._default_salt = float(md.salt) if hasattr(md, "salt") else None

    def _load_params(self) -> None:
        key = self.energy_params.lower()
        if key in self._PARAM_LOADERS:
            ok = getattr(self._RNA, self._PARAM_LOADERS[key])()
        else:
            ok = self._RNA.params_load(self.energy_params)
        if not ok:
            raise RuntimeError(f"Could not load energy parameters {self.energy_params!r}.")

    def u(self, sequence: str) -> tuple[FloatArray, dict[str, Any]]:
        """Fold ``sequence`` and return ``u`` and per-call metadata."""
        seq = normalize_sequence(sequence)
        RNA = self._RNA
        self._load_params()  # parameter loading is global in ViennaRNA: reload each call
        md = RNA.md()
        md.temperature = self.temperature_c
        md.dangles = self.dangles
        if self.salt_molar is not None:
            md.salt = float(self.salt_molar)
        fc = RNA.fold_compound(seq, md)
        fc.exp_params_rescale(fc.mfe()[1])
        _, ens = fc.pf()
        bpp = np.asarray(fc.bpp(), dtype=float)
        n = len(seq)
        up = bpp[1 : n + 1, 1 : n + 1]
        up = np.triu(up, 1)
        paired = up.sum(axis=0) + up.sum(axis=1)
        raw = 1.0 - paired
        u = np.clip(raw, 0.0, 1.0)
        return u, {"ensemble_energy": float(ens), "max_clip": float(np.max(np.abs(raw - u)))}

    def describe(self) -> dict[str, Any]:
        """Version and parameters."""
        return {
            "model": self.name,
            "implementation": "ViennaRNA Python bindings (RNA.fold_compound.pf / bpp)",
            "version": str(self._RNA.__version__),
            "temperature_c": self.temperature_c,
            "energy_params": self.energy_params,
            "salt_correction": self.salt_molar is not None,
            "salt_molar": self.salt_molar if self.salt_molar is not None else self._default_salt,
            "dangles": self.dangles,
            "can_represent": {
                "temperature": True,
                "monovalent_salt": self._default_salt is not None,
                "mg2+": False,
                "tertiary_structure": False,
            },
        }


# --------------------------------------------------------------------------- #
# CONTRAfold
# --------------------------------------------------------------------------- #
class ContrafoldModel:
    """CONTRAfold posterior base-pair probabilities.

    Parameters
    ----------
    executable
        Path or name of the ``contrafold`` binary.
    posterior_threshold
        Cutoff passed to ``--posteriors``. Pairs with probability below the
        cutoff are not written, so ``u`` is biased upward by at most
        ``(number of omitted partners) * threshold <= (N - 1) * threshold``
        per position.
    params
        Optional ``--params`` file (default: built-in parameters).
    """

    name = "CONTRAfold"

    def __init__(
        self,
        executable: str = "contrafold",
        posterior_threshold: float = 1e-5,
        params: str | None = None,
    ) -> None:
        self.exe = _resolve_exe(executable)
        if not 0 < posterior_threshold < 1:
            raise ValueError("posterior_threshold must be in (0, 1).")
        self.threshold = float(posterior_threshold)
        self.params = params
        proc = subprocess.run([self.exe, "predict", "--version"], capture_output=True, text=True, check=False)
        self._version = (proc.stdout + proc.stderr).strip().splitlines()[0]
        self._conda_version = self._conda_meta_version()

    def _conda_meta_version(self) -> str | None:
        real = Path(self.exe).resolve()
        for prefix in real.parents:
            meta = prefix / "conda-meta"
            if meta.is_dir():
                for f in meta.glob("contrafold-*.json"):
                    return str(json.loads(f.read_text()).get("version"))
                return None
        return None

    def u(self, sequence: str) -> tuple[FloatArray, dict[str, Any]]:
        """Fold ``sequence`` and return ``u`` and per-call metadata."""
        seq = normalize_sequence(sequence)
        n = len(seq)
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.txt"
            out = Path(tmp) / "post.txt"
            inp.write_text(seq + "\n")
            cmd = [self.exe, "predict", "--posteriors", repr(self.threshold), str(out), str(inp)]
            if self.params:
                cmd[2:2] = ["--params", self.params]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"CONTRAfold failed: {proc.stderr.strip()}")
            pairs: dict[tuple[int, int], float] = {}
            for line in out.read_text().splitlines():
                tok = line.split()
                if not tok:
                    continue
                i = int(tok[0]) - 1
                for t in tok[2:]:
                    js, ps = t.split(":")
                    j = int(js) - 1
                    pairs[(min(i, j), max(i, j))] = float(ps)
        if pairs and max(max(k) for k in pairs) >= n:
            raise RuntimeError("CONTRAfold output index out of range.")
        u, clip = _u_from_pairs(n, pairs)
        return u, {
            "n_pairs_reported": len(pairs),
            "max_clip": clip,
            "max_threshold_bias_bound": (n - 1) * self.threshold,
        }

    def describe(self) -> dict[str, Any]:
        """Version and parameters."""
        return {
            "model": self.name,
            "executable": self.exe,
            "binary_sha256": _sha256(self.exe),
            "version": self._version,
            "conda_package_version": self._conda_version,
            "posterior_threshold": self.threshold,
            "params": self.params or "built-in default",
            "temperature_c": None,
            "can_represent": {
                "temperature": False,
                "monovalent_salt": False,
                "mg2+": False,
                "tertiary_structure": False,
            },
        }


# --------------------------------------------------------------------------- #
# MXfold2
# --------------------------------------------------------------------------- #
class MXfold2Model:
    """MXfold2 base-pair probability matrix (``mxfold2 predict --bpp``).

    Parameters
    ----------
    executable
        Path or name of the ``mxfold2`` script.
    param
        Optional ``--param`` file. ``None`` uses the default bundled
        TrainSetAB model (MixC).
    seed
        ``--seed`` passed to MXfold2.
    """

    name = "MXfold2"

    def __init__(self, executable: str = "mxfold2", param: str | None = None, seed: int = 0):
        self.exe = _resolve_exe(executable)
        self.param = param
        self.seed = int(seed)
        self._version = self._pkg_version()

    def _pkg_version(self) -> str:
        # The mxfold2 script lives in a venv/conda bin; ask that interpreter.
        py = Path(self.exe).parent / "python"
        interp = str(py) if py.exists() else sys.executable
        proc = subprocess.run(
            [interp, "-c", "import importlib.metadata as m; print(m.version('mxfold2'))"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() or "unknown"

    def u(self, sequence: str) -> tuple[FloatArray, dict[str, Any]]:
        """Fold ``sequence`` and return ``u`` and per-call metadata."""
        seq = normalize_sequence(sequence)
        n = len(seq)
        with tempfile.TemporaryDirectory() as tmp:
            fa = Path(tmp) / "in.fa"
            fa.write_text(f">query\n{seq}\n")
            cmd = [self.exe, "predict", "--seed", str(self.seed), "--bpp", tmp]
            if self.param:
                cmd += ["--param", self.param]
            cmd.append(str(fa))
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = Path(tmp) / "query.bpp"
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"MXfold2 failed: {proc.stderr.strip()[-2000:]}")
            mat = np.loadtxt(out, ndmin=2)
        if mat.shape != (n + 1, n + 1):
            raise RuntimeError(f"MXfold2 bpp shape {mat.shape}, expected {(n + 1, n + 1)}.")
        p = mat[1:, 1:]
        raw = 1.0 - p.sum(axis=1)
        u = np.clip(raw, 0.0, 1.0)
        return u, {
            "max_clip": float(np.max(np.abs(raw - u))),
            "rounding_bound": n * 5e-6,
            "max_asymmetry": float(np.max(np.abs(p - p.T))),
        }

    def describe(self) -> dict[str, Any]:
        """Version and parameters."""
        return {
            "model": self.name,
            "executable": self.exe,
            "version": self._version,
            "param": self.param or "bundled default (TrainSetAB.conf, model MixC)",
            "seed": self.seed,
            "temperature_c": None,
            "output_precision": "5 decimals per p_ij",
            "can_represent": {
                "temperature": False,
                "monovalent_salt": False,
                "mg2+": False,
                "tertiary_structure": False,
            },
        }


MODEL_CLASSES: dict[str, type] = {
    "rnafold": RNAfoldModel,
    "contrafold": ContrafoldModel,
    "mxfold2": MXfold2Model,
}


def build_model(key: str, options: Mapping[str, Any]) -> FoldingModel:
    """Instantiate a model from its config key (``rnafold``, ``contrafold``, ``mxfold2``)."""
    try:
        cls = MODEL_CLASSES[key.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {key!r}; known: {sorted(MODEL_CLASSES)}") from exc
    model: FoldingModel = cls(**options)
    return model


class CachedModel:
    """Wrap a model with an in-memory cache keyed by sequence."""

    def __init__(self, model: FoldingModel) -> None:
        self.model = model
        self.name = model.name
        self._cache: dict[str, tuple[FloatArray, dict[str, Any]]] = {}

    def u(self, sequence: str) -> tuple[FloatArray, dict[str, Any]]:
        """Cached :meth:`FoldingModel.u`."""
        seq = normalize_sequence(sequence)
        if seq not in self._cache:
            u, md = self.model.u(seq)
            if u.shape != (len(seq),):
                raise RuntimeError(f"{self.name}: u has shape {u.shape}, expected ({len(seq)},).")
            self._cache[seq] = (u, md)
        u, md = self._cache[seq]
        return u.copy(), dict(md)

    def describe(self) -> dict[str, Any]:
        """Delegate."""
        return self.model.describe()
