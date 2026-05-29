#!/usr/bin/env bash
#
# CPM2 bootstrap: create the four conda envs and do the editable install.
#
# Idempotent: envs that already exist are skipped, so re-running after a
# partial setup is safe. Run from the repo root:
#
#     bash bootstrap.sh
#
# The pipeline is split across four isolated conda envs (isolation is
# load-bearing: the runners shell out via `conda run -n <env>` and never
# import across envs):
#
#     cpm2          orchestrator (Snakemake DAG, helpers, notebook)
#     cpepmatch     Stage 1, backbone match  (Python 3.7, Modeller, vmd-python)
#     boltz         Stage 2, fold validation (Boltz-2)
#     proteinhunter Stage 3, sequence refine (Protein-Hunter + LigandMPNN)
#
# NOTE on vendored tools: the external tool repos under lib/ (Protein-Hunter,
# boltz, cPEPmatch) are gitignored and NOT shipped with this repo. After the
# envs exist you must obtain them and run their per-tool setup steps (Modeller
# license, LigandMPNN model weights, etc.). See lib/VERSIONS.md for sources,
# pinned commits, and setup commands.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${REPO_ROOT}/envs"

# Resolve a conda-like frontend. mamba is much faster if present.
if command -v mamba >/dev/null 2>&1; then
    CONDA="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA="conda"
else
    echo "ERROR: neither mamba nor conda found on PATH." >&2
    echo "Install Miniforge/Miniconda first: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi
echo ">>> Using '${CONDA}' as the conda frontend."

# Existing env names, one per line, for idempotent skip checks.
existing_envs() {
    conda env list | awk '!/^#/ {print $1}'
}

create_env() {
    local name="$1"
    local yml="${ENVS_DIR}/${name}.yml"
    if [[ ! -f "${yml}" ]]; then
        echo "ERROR: env spec not found: ${yml}" >&2
        exit 1
    fi
    if existing_envs | grep -qx "${name}"; then
        echo ">>> [skip] conda env '${name}' already exists."
    else
        echo ">>> [create] conda env '${name}' from ${yml} ..."
        "${CONDA}" env create -f "${yml}"
        echo ">>> [done]   conda env '${name}'."
    fi
}

echo "=== CPM2 bootstrap ==="
echo "repo root: ${REPO_ROOT}"
echo "Creating the four conda envs (this can take a while on first run)."
echo

# Env names match the `name:` field inside each envs/*.yml.
create_env cpm2
create_env cpepmatch
create_env boltz
create_env proteinhunter

echo
echo ">>> Editable-installing the cpm2 package into the 'cpm2' env ..."
# `conda run -n cpm2` avoids needing `conda activate` to work in this shell.
conda run -n cpm2 pip install -e "${REPO_ROOT}[dev]"
echo ">>> [done] cpm2 package installed (editable, with [dev] extras)."

echo
echo "=== bootstrap complete ==="
echo
echo "Next steps:"
echo "  1. Obtain the vendored tools under lib/ (gitignored). See lib/VERSIONS.md"
echo "     for sources, pinned commits, and per-tool setup (Modeller license,"
echo "     LigandMPNN model weights)."
echo "  2. Activate the orchestrator env:   conda activate cpm2"
echo "  3. Run a benchmark:                 scripts/run.py mdm2_p53_v1 --cores 8"
echo "     (or: cpm2 run mdm2_p53_v1 --cores 8)"
echo
echo "See QUICKSTART.md for the minimal end-to-end path."
