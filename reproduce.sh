#!/usr/bin/env bash
# Regenerate every result and report of this repository.
#
# Usage (from anywhere, inside the "dmswin" environment, see setup_env.sh):
#     bash reproduce.sh            # tests + full run (~35 min on 2x Xeon Silver 4210R)
#     bash reproduce.sh --quick    # tests + fast run (200 bootstrap resamples; CIs differ slightly)
#     bash reproduce.sh --slides   # also rebuild slides/ (see slides/requirements.txt)
#
# What it writes (overwrites the committed versions):
#     results/<dataset>/  results/overview/  results/run_metadata.json  docs/
# Input data are read from data/ (committed). If a file in data/ is missing it is
# re-downloaded from the public URL stored in its config (RMDB / RASP v2).
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
SLIDES=0
for a in "$@"; do
    case "$a" in
        --quick) QUICK=1 ;;
        --slides) SLIDES=1 ;;
        *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done

echo "== 1/3  unit tests and static checks"
python -m pytest -q tests
ruff check .
mypy .

echo "== 2/3  window test on all datasets"
# Order matters only for the colours/markers of the overview figures.
DATASETS=(TRP4P6_DMS_0007 HOXA9_STD_0000 HOXA5_STD_0000 CSDE1_DMS_0000
          HIV3PR_DMS_0001 RNASEP_DMS_0000 ROTA_RFS11_DMS YXJA_EQR220_BZCN)
CONFIGS=()
for d in "${DATASETS[@]}"; do CONFIGS+=("configs/config_${d}.json"); done
EXTRA=()
[[ $QUICK == 1 ]] && EXTRA=(--n-boot 200)
python run_test.py "${CONFIGS[@]}" --out . "${EXTRA[@]}" 2>&1 | tee run.log

if [[ $SLIDES == 1 ]]; then
    echo "== 3/3  slides"
    python slides/build_deck.py
else
    echo "== 3/3  slides skipped (use --slides)"
fi

echo
echo "Done. Start reading at docs/overview.md; 'git diff --stat' shows what changed."
