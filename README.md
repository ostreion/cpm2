# CPM2

Pipeline for designing cyclic-peptide PPI inhibitors. Three stages, each in its
own conda env: **cPEPmatch** (backbone match) -> **Boltz-2** (fold validate) ->
**ProteinHunter** (sequence refine).

```
cPEPmatch  ─►  Boltz-2  ─►  ProteinHunter
(backbone      (fold-validate  (sequence
 match)         complex)        refinement)
```

## Quickstart (3 commands)

```bash
git clone <repo-url> cpm2 && cd cpm2
bash bootstrap.sh                      # creates 4 conda envs + editable install
conda activate cpm2 && scripts/run.py mdm2_p53_v1 --cores 8
```

Results land in `data/runs/<run_id>/output/summary.csv`.

You need conda (or mamba) and a GPU, plus the vendored tools under `lib/`
(gitignored; see [lib/VERSIONS.md](lib/VERSIONS.md)). Full minimal path,
config list, and output layout: **[QUICKSTART.md](QUICKSTART.md)**.

## More

- **[QUICKSTART.md](QUICKSTART.md)** — minimal end-to-end run for an outside user.
- **[docs/dev.md](docs/dev.md)** — entry points, per-run layout, MLflow tracking,
  per-rule benchmarks, reproducibility discipline, environments, the CLI, and the
  legacy papermill driver.
