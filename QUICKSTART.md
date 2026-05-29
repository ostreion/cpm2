# CPM2 quickstart

The minimal path to run the pipeline on your own machine. For benchmarking,
tracking, reproducibility, and development, see [docs/dev.md](docs/dev.md).

## What you need

- **conda** (or **mamba**, faster) on PATH. Get
  [Miniforge](https://github.com/conda-forge/miniforge) if you have neither.
- A machine with a **GPU** (Stage 2 Boltz-2 and Stage 3 Protein-Hunter need it).
- The **vendored tools** under `lib/` (Protein-Hunter, boltz, cPEPmatch). These
  are gitignored and not shipped with the clone. See
  [lib/VERSIONS.md](lib/VERSIONS.md) for sources, pinned commits, and per-tool
  setup (Modeller license, LigandMPNN model weights).

## Three commands

```bash
git clone <repo-url> cpm2 && cd cpm2
bash bootstrap.sh                      # creates 4 conda envs + editable install
conda activate cpm2 && scripts/run.py mdm2_p53_v1 --cores 8
```

`bootstrap.sh` is idempotent: it skips conda envs that already exist, so it is
safe to re-run. First-run env creation can take a while (each of the four envs
resolves and downloads its own stack).

## Picking a config

Each run is driven by one config file under `configs/`. Four benchmark targets
ship with the repo (pass the stem, not the path):

| Config         | Target / PPI                |
| -------------- | --------------------------- |
| `mdm2_p53_v1`  | MDM2 / p53 (1YCR)           |
| `14_3_3_v1`    | 14-3-3                      |
| `bclxl_v1`     | Bcl-xL                      |
| `cypA_v1`      | Cyclophilin A               |

(There are also `mdm2_p53_sens` and `mdm2_p53_phconv` parameter variants.)

List them any time:

```bash
cpm2 list-configs            # or: ls configs/
```

## Running

Either the canonical script or the thin `cpm2` CLI wrapper works (the CLI just
shells out to `scripts/run.py`):

```bash
scripts/run.py mdm2_p53_v1 --cores 8        # canonical
cpm2 run mdm2_p53_v1 --cores 8              # equivalent wrapper

scripts/run.py mdm2_p53_v1 --dry-run        # preview the DAG, run nothing
```

## Where output lands

Every run writes to a unique directory keyed by config, UTC timestamp, and git
SHA:

```
data/runs/<run_id>/
  manifest.json          # provenance: git SHA, config hash, env hashes, inputs
  intermediate/          # per-stage working files
  output/summary.csv     # <- the ranked results
  output/top_*.pdb       # top designed complexes
```

Start with `data/runs/<run_id>/output/summary.csv`.

## Resuming a partial run

```bash
scripts/run.py mdm2_p53_v1 --run-id <existing_run_id>
```

Re-enters the Snakemake DAG at the first missing target; finished stages are
not re-run.

---

For MLflow tracking, per-rule benchmarks, the reproducibility (commit-before-run)
discipline, the interactive notebook, and the legacy papermill driver, see
[docs/dev.md](docs/dev.md).
