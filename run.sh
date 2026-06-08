#!/bin/bash
# SLURM job script for the Digital Research Alliance of Canada (Rorqual / Calcul Québec).
#
#   SUBMIT WITH:   sbatch run.sh        <-- NOT  ./run.sh  (see guard below)
#   MONITOR WITH:  sq
#   OUTPUT:        imcp-<jobid>.out  (stdout + the AU(MCP)/AU(IMCP) table)
#
#SBATCH --account=def-XXXX          # <-- REQUIRED: your def-/rrg- allocation account
#SBATCH --job-name=imcp
#SBATCH --time=00:20:00             # tiny job; a few minutes is enough, 20 min is generous
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=imcp-%j.out
# No GPU on purpose: the nets are tiny (2->50->3); CPU is faster here than GPU overhead.

set -eo pipefail                    # stop immediately on any error (no polluting cascade)

# Guard: $SLURM_TMPDIR only exists inside a SLURM allocation. If it is empty you ran this
# directly on the login node (./run.sh) instead of submitting it -- refuse, don't pollute.
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "ERROR: \$SLURM_TMPDIR is empty -> you are NOT inside a SLURM job."
    echo "Submit it instead:   sbatch run.sh      (or test interactively: salloc ...)"
    exit 1
fi

export PIP_REQUIRE_VIRTUALENV=true            # pip MUST be inside a venv; never --user
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # don't grab the whole node

REPO_DIR="$HOME/code/python/imcp"               # <-- path to the repo (must contain imcp/, tests/, the .py)

# 1) Software stack via Lmod. python/3.11.5 == cp311, matching the wheelhouse
#    (torch 2.12.0 cp311 x86-64-v3, scikit_learn 1.8.0 cp311).
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

# --- Re-running often? Build a persistent venv ONCE on the login node instead:
#       module load python/3.11.5
#       virtualenv --no-download ~/imcp-env && source ~/imcp-env/bin/activate
#       pip install --no-index --upgrade pip
#       pip install --no-index numpy pandas matplotlib scikit-learn torch
#     then replace steps 2-3 above with:  source ~/imcp-env/bin/activate
