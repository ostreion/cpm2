# CPM2 developer & benchmarking guide

Everything beyond the minimal run path. For first-time setup and a single run,
start with [QUICKSTART.md](../QUICKSTART.md).

## Entry points

- **Interactive:** `notebooks/CPM2_presentation.py` (marimo). Run with
  `marimo run notebooks/CPM2_presentation.py` in the `cpm2` env. Headless drivers expose `run_id` /
  `config_name` so headless drivers can override them.
- **Headless (canonical):** `scripts/run.py <config_name>` drives the Snakemake
  DAG, backed by the root `Snakefile`. This is the recommended way to run a
  benchmark.

### How `scripts/run.py` invokes Snakemake

It generates a `run_id`, then calls the `snakemake` that sits next to the
running interpreter (so PATH-less invocations such as cron still work). It does
**not** pass `--use-conda`: that would make Snakemake recreate the envs from
`envs/*.yml` under `.snakemake/conda/<hash>/`, duplicating the named envs that
already exist. Instead the per-rule helpers shell out via `conda run -n <env>`
themselves, and the `Snakefile`'s `conda:` directives stay as documentation.

## Per-run layout

Every run writes to `data/runs/<run_id>/` where
`run_id = <config>_<UTC-timestamp>_<git_sha8>`:

```
data/runs/<run_id>/
  manifest.json            # git SHA + dirty, config hash, env hashes, input PDB hash, hardware, seeds
  intermediate/0_import/   # processed.pdb, template.cif
  intermediate/1_cpepmatch/, 1_cpepmatch_renamed/
  intermediate/2_boltz/yaml_input/, predictions/
  intermediate/3_proteinhunter/<match>/
  output/summary.csv, alignments/, top_*.pdb
```

Per-target caches live at `data/cache/{msa,mmcif}/`, keyed by sequence/PDB hash
and shared across runs.

## Tracking & inspection

- **MLflow** writes a single run per pipeline run to `mlruns/` (file backend).
  As of Phase F2 this is automatic for every Snakemake invocation
  (`scripts/run.py`), not just notebook runs: the DAG starts with
  `init_run` -> `mlflow_start`, every stage rule has a `benchmark:` directive
  (Snakemake writes wall-time + max RSS per invocation), and `mlflow_finalize`
  reads the benchmark TSVs + `output/summary.csv` and logs final + per-match
  metrics. Inspect via:

  ```bash
  mlflow ui --backend-store-uri file://$(pwd)/mlruns
  ```

  Override the tracking backend with `MLFLOW_TRACKING_URI`. Lazy import: if
  mlflow can't be imported, the helpers no-op and the pipeline still produces
  all the same artifacts (manifest, archive, alignment grid). The notebook path
  also logs MLflow when run interactively; the calls are guarded against
  `CPM2_HEADLESS=1` so the Snakemake path doesn't double-log.
- **Per-rule benchmarks** land at `data/runs/<run_id>/benchmark/<rule>.tsv` (or
  `<rule>/<match>.tsv` for fan-out rules). MLflow scalars: `<rule>_seconds`,
  `<rule>_max_rss_mb`; per-match rules also get `<rule>_seconds_total`,
  `<rule>_seconds_mean`, `<rule>_invocations`.
- **Manifest** at `data/runs/<run_id>/manifest.json` is also copied into the
  archive.
- **Archive** at `archives/run_<ts>_<name>/` retains the lightweight legacy
  layout plus the per-run manifest.

## Reproducibility: commit before every benchmark run

Always `git commit` (or stash) all working-tree changes **before** invoking
`scripts/run.py` or any benchmark driver. The manifest + MLflow log the current
`git_sha` plus a `git.dirty` flag; if dirty, the recorded SHA does not fully
describe the code that produced the run and the result is not reproducible. The
quality notebook surfaces dirty runs in its provenance-flags callout: treat them
as untrusted. If a quick try-it-out run is unavoidable, commit a WIP snapshot
first.

## Recovery from a partial run

```bash
scripts/run.py <config> --run-id <existing_id>
```

Re-enters the Snakemake DAG at the first missing target; finished stages aren't
re-run.

## Environments

The pipeline is split across four isolated conda envs. Isolation is
load-bearing: the runners shell out via `conda run -n <env>` and never import
across envs.

| Env             | Stage / role                         | Spec                      |
| --------------- | ------------------------------------ | ------------------------- |
| `cpm2`          | orchestrator (DAG, helpers, notebook)| `envs/cpm2.yml`           |
| `cpepmatch`     | Stage 1, backbone match              | `envs/cpepmatch.yml`      |
| `boltz`         | Stage 2, fold validation (Boltz-2)   | `envs/boltz.yml`          |
| `proteinhunter` | Stage 3, sequence refine             | `envs/proteinhunter.yml`  |

`bootstrap.sh` creates all four idempotently and editable-installs the `cpm2`
package (`pip install -e ".[dev]"`) into the `cpm2` env. The `cpepmatch` env in
particular pins Python 3.7 and needs a Modeller license + vmd-python; see
[lib/VERSIONS.md](../lib/VERSIONS.md).

## Vendored tools (`lib/`)

External tool repos (Protein-Hunter, boltz, cPEPmatch) are vendored under
`lib/`, gitignored, never imported, only invoked as subprocesses. They are not
shipped with the clone. [lib/VERSIONS.md](../lib/VERSIONS.md) documents sources,
pinned commits, licenses, citations, and per-tool setup (Protein-Hunter
`setup.sh`, LigandMPNN model-weight download, cPEPmatch Modeller config).

## Convenience CLI

`bootstrap.sh` installs a thin `cpm2` console script (`src/cli.py`):

```bash
cpm2 list-configs                  # list configs/*.yaml stems
cpm2 run <config> [args...]        # forwards to scripts/run.py
```

It is a pure wrapper: `cpm2 run` shells out to `scripts/run.py` with the same
interpreter, and adds no pipeline logic.
