"""CLI: DMS/SHAPE-based validity test of folding models and folding-window size.

Usage
-----
    python run_test.py config_TRP4P6_DMS_0007.json --out . --seed 0

Every dataset-specific choice comes from the JSON config(s); CLI flags
override config values for all datasets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import dmswin_analysis as an
import dmswin_data as dd
import dmswin_models as dm
import dmswin_plots as dp
import dmswin_report as rep

REQUIRED_MODELS = ("rnafold", "contrafold", "mxfold2")


def _to_index(ds: dd.Dataset, pos: int, numbering: str) -> int:
    if numbering == "source":
        return ds.source_to_index(int(pos))
    if numbering == "one_based":
        if not 1 <= int(pos) <= ds.n:
            raise ValueError(f"Position {pos} outside 1..{ds.n}.")
        return int(pos) - 1
    raise ValueError(f"Unknown numbering {numbering!r} (use 'source' or 'one_based').")


def prepare_dataset(
    cfg: dict[str, Any], cfg_dir: Path, redownload: bool
) -> tuple[dd.Dataset, dict[str, Any]]:
    """Load, select and mask the dataset described by a config.

    Returns the dataset and a provenance/selection record.
    """
    src = cfg["source"]
    path = (cfg_dir / src["path"]).resolve()
    prov = dd.ensure_source(path, src.get("url"), redownload=redownload)
    opts = dict(src.get("loader_options", {}))
    # auxiliary public files (e.g. the reference FASTA of a bigWig track):
    # downloaded with provenance and passed to the loader as ``option=path``
    extra_prov = {}
    for aux in src.get("extra_sources", []):
        apath = (cfg_dir / aux["path"]).resolve()
        extra_prov[aux["option"]] = dd.ensure_source(apath, aux.get("url"), redownload=redownload)
        opts[aux["option"]] = str(apath)
    if extra_prov:
        prov = {**prov, "extra_sources": extra_prov}
    all_ds = dd.load(src["format"], path, **opts)
    sel = cfg.get("select", {})
    ds = dd.select_dataset(
        all_ds, annotation=sel.get("annotation"), row=sel.get("row"), construct=sel.get("construct")
    )
    numbering = cfg.get("numbering", "one_based")
    mask_idx: list[int] = []
    for lo, hi in cfg.get("mask", []):
        a, b = _to_index(ds, lo, numbering), _to_index(ds, hi, numbering)
        mask_idx += list(range(a, b + 1))
    nan_before = [int(i) for i in np.flatnonzero(~np.isfinite(ds.reactivity))]
    ds = ds.with_mask(mask_idx)
    ds = dd.Dataset(
        dataset_id=cfg.get("dataset_id", ds.dataset_id),
        sequence=ds.sequence,
        reactivity=ds.reactivity,
        reagent=cfg.get("reagent", ds.reagent),
        conditions=ds.conditions,
        source_url=src.get("landing_page", src.get("url")),
        download_date=prov.get("download_date"),
        source_positions=ds.source_positions,
        metadata={**ds.metadata, "provenance": prov, "n_constructs_in_file": len(all_ds)},
    )
    # cross-check conditions written in the config against the file
    cond_cfg = cfg.get("conditions")
    if cond_cfg:
        c = ds.conditions
        for key, val in (("temperature_c", c.temperature_c), ("mg_mM", c.mg_mm)):
            if key in cond_cfg and val is not None and float(cond_cfg[key]) != float(val):
                raise ValueError(f"Config condition {key}={cond_cfg[key]} != file annotation {val}.")
    record = {
        "selection": sel,
        "row": ds.metadata.get("row"),
        "row_annotations": ds.metadata.get("row_annotations"),
        "n_constructs_in_file": len(all_ds),
        "nan_in_source": nan_before,
        "masked_indices_0based": sorted(set(mask_idx)),
        "mask_config": cfg.get("mask", []),
        "mask_reason": cfg.get("mask_reason"),
        "numbering": numbering,
    }
    return ds, record


def build_models(
    cfg: dict[str, Any], ds: dd.Dataset, only: list[str]
) -> tuple[dict[str, dm.FoldingModel], list[str]]:
    """Instantiate the configured models; set RNAfold temperature from the data."""
    mcfg = cfg.get("models", {})
    missing = [m for m in REQUIRED_MODELS if m not in mcfg]
    if missing:
        raise ValueError(f"Config lacks required models: {missing}")
    flags: list[str] = []
    t_cfg = cfg.get("temperature_c", "from_dataset")
    if t_cfg == "from_dataset":
        temp = ds.conditions.temperature_c
        if temp is None:
            temp = 37.0
            flags.append("No temperature in the data annotations; RNAfold uses 37 °C (default).")
    else:
        temp = float(t_cfg)
    models: dict[str, dm.FoldingModel] = {}
    for key in only:
        opts = dict(mcfg[key])
        if key == "rnafold":
            opts.setdefault("temperature_c", temp)
        try:
            model = dm.build_model(key, opts)
        except Exception as exc:  # stop: every model is required
            raise SystemExit(f"ERROR: model {key} could not be initialised: {exc}") from exc
        models[model.name] = dm.CachedModel(model)
    c = ds.conditions
    if c.mg_mm:
        flags.append(
            f"{c.mg_mm:g} mM Mg²⁺ in the experiment: no model represents Mg²⁺ or tertiary structure."
        )
    if c.temperature_c is not None:
        flags.append(
            f"Experiment at {c.temperature_c:g} °C: RNAfold folds at {temp:g} °C; "
            "CONTRAfold and MXfold2 have no temperature parameter."
        )
    if c.buffer:
        salt = mcfg["rnafold"].get("salt_molar")
        flags.append(
            f"Buffer {c.buffer}: RNAfold "
            + (
                f"uses salt correction at {salt:g} M monovalent"
                if salt
                else "uses standard 1.021 M monovalent salt (correction off)"
            )
            + "; CONTRAfold/MXfold2 have no salt dependence."
        )
    return models, flags


def contrafold_threshold_check(cfg: dict[str, Any], ds: dd.Dataset, res: an.DatasetResult) -> dict[str, Any]:
    """Measure the posterior-threshold bias of CONTRAfold on this dataset.

    Re-folds the full sequence and the smallest window with a threshold 1e4
    times lower and reports the largest change in u and in rho.
    """
    opts = dict(cfg["models"]["contrafold"])
    thr = float(opts.get("posterior_threshold", 1e-5))
    ref_thr = thr * 1e-4
    ref = dm.ContrafoldModel(**{**opts, "posterior_threshold": ref_thr})
    out: dict[str, Any] = {"threshold": thr, "reference_threshold": ref_thr, "windows": {}}
    e = res.e_idx
    for w in (min(res.windows), ds.n):
        a, b = res.windows[w]
        u_ref_local, _ = ref.u(ds.sequence[a : b + 1])
        u_ref = np.full(ds.n, np.nan)
        u_ref[a : b + 1] = u_ref_local
        u_run = res.models["CONTRAfold"].u_windows[w]
        rho_run = an.spearman(u_run[e], ds.reactivity[e])
        rho_ref = an.spearman(u_ref[e], ds.reactivity[e])
        out["windows"][w] = {
            "max_abs_du_all_positions": float(np.nanmax(np.abs(u_run - u_ref))),
            "rho_at_threshold": rho_run,
            "rho_at_reference": rho_ref,
            "abs_rho_change": abs(rho_run - rho_ref),
        }
    return out


def tool_versions() -> dict[str, Any]:
    """Python package versions for the run metadata."""
    import matplotlib
    import scipy

    out: dict[str, Any] = {
        "python": sys.version,
        "python_executable": os.path.normpath(sys.executable),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import RNA

        out["ViennaRNA"] = RNA.__version__
    except ImportError:
        out["ViennaRNA"] = None
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError:
        out["torch"] = None
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows to CSV (no-op for empty lists)."""
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def per_position_rows(res: an.DatasetResult) -> list[dict[str, Any]]:
    """Long table: one row per (position, model, W)."""
    ds = res.dataset
    e = set(res.e_idx.tolist())
    r0, r1 = res.region
    masked = set(ds.metadata.get("masked_indices", []))
    rows = []
    for name, mr in res.models.items():
        for w in sorted(res.windows):
            a, b = res.windows[w]
            uw = mr.u_windows[w]
            for i in range(a, b + 1):
                rows.append(
                    {
                        "dataset_id": ds.dataset_id,
                        "model": name,
                        "W": w,
                        "position_1based": i + 1,
                        "source_position": int(ds.source_positions[i]),  # type: ignore[index]
                        "nucleotide": ds.sequence[i],
                        "reactivity": float(ds.reactivity[i]),
                        "masked": i in masked,
                        "in_R": r0 <= i <= r1,
                        "in_E": i in e,
                        "u_W": float(uw[i]),
                        "u_N": float(mr.u_full[i]),
                        "abs_du": float(abs(uw[i] - mr.u_full[i])),
                    }
                )
    return rows


def wide_rows(res: an.DatasetResult) -> list[dict[str, Any]]:
    """Rows = W; columns per model (rho_W, CI, delta_rho, CI)."""
    out = []
    for w in sorted(res.windows):
        a, b = res.windows[w]
        row: dict[str, Any] = {"W": w, "window_1based": f"{a + 1}-{b + 1}"}
        for name, mr in res.models.items():
            r = next(x for x in mr.rows if x["W"] == w)
            row[f"{name}_rho_W"] = r["rho_W"]
            row[f"{name}_rho_W_CI"] = f"[{r['rho_W_ci_lo']:.3f}, {r['rho_W_ci_hi']:.3f}]"
            row[f"{name}_delta_rho"] = r["delta_rho"]
            row[f"{name}_delta_rho_CI"] = f"[{r['delta_rho_ci_lo']:.3f}, {r['delta_rho_ci_hi']:.3f}]"
        out.append(row)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="+", type=Path, help="Dataset config JSON file(s).")
    p.add_argument("--out", type=Path, default=Path("."), help="Output directory (default: .).")
    p.add_argument("--seed", type=int, default=None, help="Bootstrap RNG seed (overrides config).")
    p.add_argument("--n-boot", type=int, default=None, help="Bootstrap resamples (overrides config).")
    p.add_argument("--block-length", type=int, default=None, help="Bootstrap block length in nt.")
    p.add_argument("--epsilon", type=float, default=None, help="Tolerance on delta_rho.")
    p.add_argument("--step", type=int, default=None, help="Window grid step (nt).")
    p.add_argument("--anchoring", choices=["centered", "five_prime"], default=None)
    p.add_argument("--eval-nts", default=None, help="Evaluation nucleotides, e.g. 'AC'.")
    p.add_argument("--redownload", action="store_true", help="Re-download public source files.")
    p.add_argument("--no-report", action="store_true", help="Skip README.md generation.")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> list[an.DatasetResult]:
    """Run all configs; write tables, figures, metadata and report."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    multi = len(args.configs) > 1
    results: list[an.DatasetResult] = []
    meta: dict[str, Any] = {
        "run_started_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "tools": tool_versions(),
        "datasets": {},
    }
    contexts = []
    for cfg_path in args.configs:
        cfg = json.loads(cfg_path.read_text())
        ds, rec = prepare_dataset(cfg, cfg_path.resolve().parent, args.redownload)
        numbering = cfg.get("numbering", "one_based")
        region = (_to_index(ds, cfg["region"][0], numbering), _to_index(ds, cfg["region"][1], numbering))
        anchoring = args.anchoring or cfg.get("anchoring", "centered")
        grid_cfg = cfg.get("window_grid", {"step": 10})
        if "sizes" in grid_cfg and args.step is None:
            grid = [int(w) for w in grid_cfg["sizes"]]
        else:
            grid = an.default_grid(ds.n, region, anchoring, args.step or int(grid_cfg.get("step", 10)))
        nts_override = args.eval_nts or cfg.get("evaluation_nucleotides")
        eval_nts = dd.evaluation_nucleotides(ds.reagent, nts_override)
        boot = cfg.get("bootstrap", {})
        n_boot = args.n_boot or int(boot.get("n_boot", 2000))
        block = args.block_length or int(boot.get("block_length", 10))
        seed = args.seed if args.seed is not None else int(boot.get("seed", 0))
        eps = args.epsilon if args.epsilon is not None else float(cfg.get("epsilon", 0.02))
        models, flags = build_models(cfg, ds, list(REQUIRED_MODELS))
        print(
            f"[{ds.dataset_id}] N={ds.n} R={region[0] + 1}-{region[1] + 1} (1-based) "
            f"anchoring={anchoring} W grid={grid} eval nts={''.join(sorted(eval_nts))}",
            flush=True,
        )
        res = an.run_dataset(
            ds, region, anchoring, grid, eval_nts, models, eps, n_boot=n_boot, block_length=block, seed=seed
        )
        results.append(res)
        bias = contrafold_threshold_check(cfg, ds, res)
        prefix = f"{ds.dataset_id}_" if multi else ""
        summary = [r for mr in res.models.values() for r in mr.rows]
        write_csv(out / f"{prefix}summary.csv", summary)
        write_csv(out / f"{prefix}summary_wide.csv", wide_rows(res))
        write_csv(out / f"{prefix}per_position.csv", per_position_rows(res))
        figs = dp.make_all(res, out, prefix)
        meta["datasets"][ds.dataset_id] = {
            "config_file": str(cfg_path),
            "config": cfg,
            "effective": {
                "region_1based": [region[0] + 1, region[1] + 1],
                "region_source": [int(ds.source_positions[region[0]]), int(ds.source_positions[region[1]])],  # type: ignore[index]
                "anchoring": anchoring,
                "window_grid": sorted(res.windows),
                "windows_1based": {w: [a + 1, b + 1] for w, (a, b) in res.windows.items()},
                "evaluation_nucleotides": sorted(eval_nts),
                "evaluation_nucleotides_source": "config/CLI override"
                if nts_override
                else f"reagent mapping ({ds.reagent})",
                "E_1based": [int(i) + 1 for i in res.e_idx],
                "epsilon": eps,
                "bootstrap": res.bootstrap,
            },
            "dataset": {
                "sequence": ds.sequence,
                "length": ds.n,
                "reagent": ds.reagent,
                "conditions": ds.conditions.as_dict(),
                "source_url": ds.source_url,
                "download_date": ds.download_date,
                "provenance": ds.metadata.get("provenance"),
                "rdat_name": ds.metadata.get("name"),
            },
            "selection_and_masking": rec,
            "models": {name: mr.description for name, mr in res.models.items()},
            "model_call_metadata": {
                name: {str(w): v for w, v in mr.call_metadata.items()} for name, mr in res.models.items()
            },
            "condition_flags": flags,
            "contrafold_threshold_check": bias,
            "suggested_windows": res.suggestions,
            "figures": figs,
        }
        contexts.append(
            {"cfg": cfg, "prefix": prefix, "flags": flags, "figs": figs, "selection": rec, "bias": bias}
        )
    cross = an.cross_dataset_tests(results)
    if cross:
        write_csv(out / "cross_dataset_wilcoxon.csv", cross)
    meta["cross_dataset_wilcoxon"] = cross
    meta["overview_figures"] = dp.make_overview(results, out) if multi else []
    meta["run_finished_utc"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    meta["pip_freeze"] = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if not args.no_report:
        (out / "README.md").write_text(rep.render(results, contexts, meta, cross))
    return results


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    warnings.simplefilter("default")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
