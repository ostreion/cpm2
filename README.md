# CPM2

Pipeline for designing cyclic-peptide PPI inhibitors. Three stages, each in its
own conda env: **cPEPmatch** (backbone match) -> **Boltz-2** (fold validate) ->
**ProteinHunter** (sequence refine).

```
cPEPmatch  ─►  Boltz-2  ─►  ProteinHunter
(backbone      (fold-validate  (sequence
 match)         complex)        refinement)
```

## Try it now on Colab

Run pipeline stages 1 to 3 (no MM-GBSA) on a free Google Colab GPU, no local install:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ostreion/cpm2/blob/main/notebooks/colab/CPM2_colab.ipynb)

You will need a free [Modeller license key](https://salilab.org/modeller/registration.html) (used by stage 1).

## Quickstart (3 commands)

```bash
git clone https://github.com/ostreion/cpm2.git && cd cpm2
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
  per-rule benchmarks, reproducibility discipline, environments, and the CLI.

## Notebooks and deliverables

- **`notebooks/CPM2_presentation.py`** — interactive [marimo](https://marimo.io)
  page for critically examining pipeline quality (triage plane and structure
  overlay, topology fidelity, MD / MM-GBSA physics checks). Run it with
  `marimo run notebooks/CPM2_presentation.py` in the `cpm2` env. Panels populate
  once you have runs under `data/runs/` and MD bundles.
- **`notebooks/slides/`** — a status-report slide deck
  (`CPM2_status_report.pptx`), regenerable with `python notebooks/slides/build_deck.py`.
- **`notebooks/colab/CPM2_colab.ipynb`** — a lean Google Colab notebook to try
  pipeline stages 1 to 3 (no MM-GBSA) on a free-tier GPU.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ostreion/cpm2/blob/main/notebooks/colab/CPM2_colab.ipynb)

## Citing and acknowledgements

This pipeline orchestrates three external tools; it does not redistribute them
(see [lib/VERSIONS.md](lib/VERSIONS.md) to obtain each). If you use CPM2, please
cite the upstream work:

- **cPEPmatch** (Santini and Zacharias, TU Munich) -
  [github.com/briandasantini/cPEPmatch](https://github.com/briandasantini/cPEPmatch).
  Santini and Zacharias, *J. Chem. Inf. Model.*, 2020.
- **Boltz-2** (MIT) -
  [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz).
- **Protein-Hunter** (Cho et al.) -
  [github.com/yehlincho/Protein-Hunter](https://github.com/yehlincho/Protein-Hunter),
  which builds on **LigandMPNN** (Dauparas et al., MIT) -
  [github.com/dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

Boltz-2 and LigandMPNN are MIT-licensed. cPEPmatch and Protein-Hunter do not
publish a license at the time of writing, so they are referenced and invoked,
never redistributed here; obtain and use them under terms agreed with their
authors.

## License

CPM2 (the code in this repository) is released under the [MIT License](LICENSE).
This covers only CPM2's own code, not the external tools above, which remain
under their respective terms.
