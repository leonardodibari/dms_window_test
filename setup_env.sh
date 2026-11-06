#!/usr/bin/env bash
# Create the "dmswin" conda environment with all three folding models.
# Usage:  bash setup_env.sh [env-name-or-path]
#   default name: dmswin ; a value containing "/" is used as a prefix path.
set -euo pipefail
cd "$(dirname "$0")"
ENV="${1:-dmswin}"
if [[ "$ENV" == */* ]]; then LOC=(-p "$ENV"); else LOC=(-n "$ENV"); fi
CONDA="$(command -v mamba || command -v conda)"

"$CONDA" env create "${LOC[@]}" -f environment.yml
# MXfold2 is not on PyPI/conda: build it from the exact commit used for the results.
# --no-build-isolation: its build needs the torch installed above.
# --no-deps: its metadata pins torch<2 / numpy<2, but it runs fine on the versions above
#            (these are the versions that produced the committed results).
"$CONDA" run "${LOC[@]}" python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/mxfold/mxfold2.git@51b213676708bebd664f0c40873a46e09353e1ee"

echo
echo "Check that the three models are found:"
"$CONDA" run "${LOC[@]}" python -c "import RNA; print('ViennaRNA', RNA.__version__)"
"$CONDA" run "${LOC[@]}" bash -c 'command -v contrafold && command -v mxfold2'
echo "Done. Activate with:  conda activate ${ENV}"
