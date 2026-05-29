#!/usr/bin/env bash
# colab_setup.sh - one-shot environment bootstrap for the CPM2 Colab notebook.
#
# The notebook (CPM2_colab.ipynb) already performs setup inline, cell by cell,
# with progress prints and graceful fallbacks. This script is the same logic in
# one place, for users who would rather run a single shell cell, or for a
# fresh VM outside Colab that already has conda/mamba on PATH.
#
# It assumes:
#   * conda OR mamba is on PATH (on Colab, install via the `condacolab` cell first)
#   * a CUDA GPU is available (Colab: Runtime -> Change runtime type -> T4 GPU)
#
# Usage:
#   export KEY_MODELLER="YOUR-MODELLER-LICENSE-KEY"   # free: salilab.org/modeller/registration.html
#   export REPO_URL="https://github.com/<you>/cpepmatch2.git"
#   bash colab_setup.sh
#
# WHY conda AND pip: Stage 1 (cPEPmatch) needs Modeller + vmd-python, which are
# conda-only (Modeller also needs a license). Stages 2/3 (Boltz-2, LigandMPNN)
# are pip-installable. A single shared env hosts all three.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/cpepmatch2}"
REPO_URL="${REPO_URL:?set REPO_URL to the cpepmatch2 clone URL}"
: "${KEY_MODELLER:?set KEY_MODELLER to your Modeller academic license key}"

CONDA="$(command -v mamba || command -v conda)"
echo "Using conda frontend: $CONDA"

# --- 1. Repo + vendored tools (lib/ is gitignored; fetch upstream) -----------
if [ ! -d "$REPO_DIR/src" ]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
LIB="$REPO_DIR/lib"
mkdir -p "$LIB"

if [ ! -f "$LIB/cPEPmatch/cpepmatch.py" ]; then
  git clone https://github.com/briandasantini/cPEPmatch.git "$LIB/cPEPmatch"
  git -C "$LIB/cPEPmatch" checkout e3c746999afa0715d819c42f550d8c4ccae48489 || true
fi
if [ ! -d "$LIB/Protein-Hunter" ]; then
  git clone https://github.com/yehlincho/Protein-Hunter.git "$LIB/Protein-Hunter"
fi
if [ ! -f "$LIB/Protein-Hunter/LigandMPNN/run.py" ]; then
  git clone https://github.com/dauparas/LigandMPNN.git "$LIB/Protein-Hunter/LigandMPNN"
fi

# --- 2. conda deps: Stage 1 (Modeller + vmd-python) --------------------------
# KEY_MODELLER is consumed automatically by the modeller conda package.
"$CONDA" install -y -c salilab -c conda-forge modeller vmd-python

# --- 3. pip deps: Stages 2/3 + light orchestration ---------------------------
pip install "boltz[cuda]"
pip install ligandmpnn || echo "ligandmpnn pip install failed; lib/ copy will be used"
pip install biopython gemmi pandas matplotlib plotly pyyaml MDAnalysis py3Dmol seaborn

# --- 4. LigandMPNN checkpoints ----------------------------------------------
MP="$LIB/Protein-Hunter/LigandMPNN/model_params"
if ! ls "$MP"/*.pt >/dev/null 2>&1; then
  bash "$LIB/Protein-Hunter/LigandMPNN/get_model_params.sh" "$MP"
fi

echo
echo "Setup complete. Boltz-2 weights (~8 GB) download on first prediction into ~/.boltz."
echo "In Python, add the package to sys.path:  sys.path.insert(0, '$REPO_DIR/src')"
