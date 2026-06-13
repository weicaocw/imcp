#!/bin/bash
#SBATCH --account=def-thocking_cpu
#SBATCH --job-name=imcp
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=imcp-%j.out

set -eo pipefail                    # stop immediately on any error (no polluting cascade)

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "ERROR: \$SLURM_TMPDIR is empty -> you are NOT inside a SLURM job."
    echo "Submit it instead:   sbatch run.sh      (or test interactively: salloc ...)"
    exit 1
fi

export PIP_REQUIRE_VIRTUALENV=true            # pip MUST be inside a venv; never --user
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # don't grab the whole node

REPO_DIR="$HOME/code/python/imcp"               # <-- path to the repo (must contain imcp/, tests/, the .py)

module load python/3.11.5

# 2) Throwaway virtualenv on fast node-local disk ($SLURM_TMPDIR) -- Alliance best practice.
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip

# 3) Install everything from the offline Alliance wheelhouse (never touches PyPI).
pip install --no-index numpy pandas matplotlib scikit-learn torch

# 4) Run. The script finds imcp/ and tests/ via __file__ and writes figures + results.csv
#    to $REPO_DIR/classifier_imcp_out/ . The numpy>=2.x trapz shim is built in.
python "$REPO_DIR/run_classifiers_imcp.py"

