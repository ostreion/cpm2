# CPM2 on Google Colab

A lean, self-contained notebook to run the full CPM2 / CPNext cyclic-peptide
binder design pipeline (Stage 1 cPEPmatch -> Stage 2 Boltz-2 -> Stage 3
ProteinHunter) on a single free Colab T4 GPU, for a researcher who is not in
this field.

## Files

| File | What it is |
|---|---|
| `CPM2_colab.ipynb` | The notebook. Open it in Colab and run top to bottom. |
| `colab_setup.sh` | Same environment bootstrap as the notebook's setup cells, in one script (optional). |
| `README.md` | This file. |

## Quick start

1. Open `CPM2_colab.ipynb` in Google Colab.
2. `Runtime` -> `Change runtime type` -> **T4 GPU**.
3. Get a free Modeller academic license key (instant, by email):
   <https://salilab.org/modeller/registration.html>.
4. In the **Configuration** cell, set:
   * `MODELLER_LICENSE` to your key,
   * `repo_url` to wherever this repo is cloned (your fork / lab GitHub).
5. Run all cells. The first cell (`condacolab`) restarts the kernel once - that
   is expected; just continue with the cells below it.

The default run designs cyclic-peptide inhibitors of **MDM2/p53 (PDB 1YCR)**,
sized to fit the free tier (one target, up to 4 matches, 2 designs x 5 cycles
each). Expect roughly 1-2 hours of T4 time after the one-time ~15-25 min setup.

## How it works (and what is deliberately left out)

* **One shared Python environment.** Normally each stage runs in its own conda
  env. Here we use `condacolab` to get a Miniconda base in the Colab runtime,
  install Modeller + vmd-python (Stage 1's conda-only blockers) plus
  `boltz[cuda]` + `ligandmpnn` (Stages 2/3) into it, and run everything in that
  one env.
* **Reuses the repo's runners.** Boltz and ProteinHunter are invoked through
  the existing `cpm2.runners.*` functions with `conda_env=None`. cPEPmatch's
  runner lacks that switch, so its CLI is invoked directly and parsed with the
  runner's own `parse_match_list()`. `src/` is never modified.
* **Reuses the repo's analysis.** Results are written to the canonical
  `data/runs/<run_id>/output/summary.csv` layout and visualised with
  `cpm2.analysis` (`load_run`, `plot_topology`, `plot_distributions`,
  `plot_triage`, `plot_convergence`, `rank`).
* **Not included** (by design, to stay lean): MM-GBSA, Snakemake, MLflow, the
  headless/benchmark machinery.

## Caching

Mount Google Drive (a cell offers this) to cache the ~8 GB Boltz-2 weights and
the MSA so re-runs skip the slow downloads. Without Drive it still works as a
one-shot run; you just re-download on each fresh runtime.

## Cells you must verify on a real Colab runtime

This notebook was authored without a live Colab session. Cells whose behavior
could not be executed here are marked **[VERIFY ON COLAB]** in the notebook:
condacolab install + kernel restart, Drive mount, the conda/pip installs, and
the LigandMPNN weight download. Read those notes when you run it.
