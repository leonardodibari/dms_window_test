# Folding-window validity test with chemical-probing data

**Question.** RNAs are often folded in a *window* (e.g. the first W nt of an mRNA)
instead of at full length. Does the window change the predicted structure of a region
of interest, and does it make the prediction agree less with experimental probing data?

**What we do.** For 8 public RNAs probed in vitro (DMS or SHAPE; RMDB and RASP v2) we fold
5′-anchored windows of growing size W with three models (ViennaRNA RNAfold, CONTRAfold,
MXfold2). For each W we compare the unpaired probability u in a region R with:

* the **probing data**: ρ(W) = Spearman(u(W), reactivity) and Δρ = ρ(N) − ρ(W), with block-bootstrap CIs and a Wilcoxon test across RNAs;
* the model's **own full-length prediction**: max |u(W) − u(N)| over R, giving the converged window W_conv.

**Main result.** No window size works for every RNA. The prediction in R converges to the
full-length one only at W ≥ 0.77 N, and the worst cost of truncation ranges from
Δρ ≈ 0 to ≈ 0.7 depending on the RNA. Fold the full length whenever possible.
Details: [docs/overview.md](docs/overview.md).

## Documentation

| Document | Content |
|---|---|
| [docs/overview.md](docs/overview.md) | Results across all datasets: summary table, overview figures, Wilcoxon test, limitations, run environment |
| [docs/method.md](docs/method.md) | The test step by step (windows, u, evaluation set E, ρ, Δρ, bootstrap, suggested windows) |
| [docs/datasets/TRP4P6_DMS_0007.md](docs/datasets/TRP4P6_DMS_0007.md) | P4-P6 domain, *Tetrahymena* group I intron (DMS, RMDB) |
| [docs/datasets/HOXA9_STD_0000.md](docs/datasets/HOXA9_STD_0000.md) | Hox A9 mRNA 5′ UTR (DMS, RMDB) |
| [docs/datasets/HOXA5_STD_0000.md](docs/datasets/HOXA5_STD_0000.md) | Hox A5 mRNA 5′ UTR (DMS, RMDB) |
| [docs/datasets/CSDE1_DMS_0000.md](docs/datasets/CSDE1_DMS_0000.md) | Csde1 mRNA fragment (DMS, RMDB) |
| [docs/datasets/HIV3PR_DMS_0001.md](docs/datasets/HIV3PR_DMS_0001.md) | HIV-1 3′ UTR (DMS, RMDB) |
| [docs/datasets/RNASEP_DMS_0000.md](docs/datasets/RNASEP_DMS_0000.md) | RNase P catalytic domain (DMS, RMDB) |
| [docs/datasets/ROTA_RFS11_DMS.md](docs/datasets/ROTA_RFS11_DMS.md) | Rotavirus A segment 11 (DMS, RASP v2) |
| [docs/datasets/YXJA_EQR220_BZCN.md](docs/datasets/YXJA_EQR220_BZCN.md) | *B. subtilis* yxjA riboswitch leader (BzCN SHAPE, RASP v2) |
| [slides/dms_window_test_slides.pdf](slides/dms_window_test_slides.pdf) | 6-slide summary (also as `.pptx`) |

Each dataset document gives the data provenance (URL, download date, SHA-256), every
choice made (region, masked positions, conditions), the per-window tables, six figures and
an automatic comparison of the models. All files in `docs/` are **generated** by
`run_test.py`; do not edit them by hand.

## Reproduce everything

Tested on Linux x86_64 with conda (miniforge). No GPU needed.

### 1. Clone

```bash
git clone https://github.com/leonardodibari/dms_window_test.git
cd dms_window_test
```

### 2. Create the environment (once, ~10 min)

```bash
bash setup_env.sh          # creates the conda env "dmswin"
conda activate dmswin
```

`setup_env.sh` builds the env from [environment.yml](environment.yml) (Python 3.12,
NumPy/SciPy/Matplotlib, ViennaRNA 2.7.2, CONTRAfold 2.02 from bioconda, PyTorch CPU) and
then compiles MXfold2 from the exact commit used here
(`mxfold/mxfold2@51b2136`). It installs MXfold2 with `--no-deps` on purpose: its metadata asks
for torch < 2, but it runs on the torch version used for the results. The script ends by
printing where the three models were found.

### 3. Run

```bash
bash reproduce.sh            # unit tests + all 8 datasets (~35 min on 2× Xeon Silver 4210R)
bash reproduce.sh --quick    # same, 200 instead of 2000 bootstrap resamples (CIs differ slightly)
bash reproduce.sh --slides   # also rebuild the slides (needs slides/requirements.txt + Carlito font)
```

[reproduce.sh](reproduce.sh) runs `pytest`, `ruff` and `mypy`, then

```bash
python run_test.py configs/config_<dataset>.json [...] --out .
```

which rewrites `results/` and `docs/`. Then check what changed with `git diff --stat`.
With the same environment, the tables in `results/` are expected to come out identical. Only the dates in
`results/run_metadata.json` change.

### Input data

The raw files are committed in [data/](data/), each with a `.provenance.json` sidecar
holding its public URL, download date and SHA-256. If a file is missing, `run_test.py`
downloads it again from that URL; `--redownload` forces it. Only public data are used
(RMDB, RASP v2). Nothing is sent to external services.

### Add a dataset

Copy a file in [configs/](configs/) and edit it. Every dataset-specific choice lives in the
config: source and URL, row selection, masked positions, region R, window grid,
conditions, model options and the free-text `report_notes` copied into the dataset
document. Then add its name to `DATASETS` in `reproduce.sh`. Supported formats are RDAT,
CSV/TSV, and bigWig + FASTA (see `dmswin_data.py`).

## Repository layout

```
README.md            this file
docs/                generated reports: overview.md, method.md, datasets/<id>.md
configs/             one JSON config per dataset
data/                downloaded public data + provenance sidecars
results/<id>/        summary.csv, summary_wide.csv, per_position.csv, fig_*.png
results/overview/    overview_*.png, cross_dataset_wilcoxon.csv
results/run_metadata.json   versions, pip freeze, binary SHA-256, every effective parameter
run_test.py          command-line entry point
dmswin_data.py       loaders (RDAT, CSV/TSV, bigWig+FASTA), Dataset class, download + provenance
dmswin_models.py     RNAfold / CONTRAfold / MXfold2 wrappers: sequence → u
dmswin_analysis.py   windows, evaluation set, Spearman, block bootstrap, suggested W
dmswin_plots.py      per-dataset and overview figures
dmswin_report.py     writes docs/
tests/               pytest suite
slides/              build_deck.py → dms_window_test_slides.pptx / .pdf
environment.yml, setup_env.sh, reproduce.sh
```

## Output files

| File | Content |
|---|---|
| `results/<id>/summary.csv` | one row per (model, W): window coordinates, ρ_W, ρ_N, Δρ with 95 % CIs, convergence metrics |
| `results/<id>/summary_wide.csv` | rows = W, columns per model |
| `results/<id>/per_position.csv` | one row per (model, W, position): reactivity, u_W, u_N, \|Δu\|, flags in_R / in_E / masked |
| `results/<id>/fig_*.png` | six figures per dataset, explained in its document |
| `results/overview/overview_*.png` | cross-dataset figures, explained in [docs/overview.md](docs/overview.md) |
| `results/overview/cross_dataset_wilcoxon.csv` | Wilcoxon signed-rank test of Δρ(W) across RNAs |
| `results/run_metadata.json` | full provenance of the run |
