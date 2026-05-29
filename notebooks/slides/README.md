# CPM2 / CPNext status-report deck


> Note: the benchmark CSVs referenced below are internal lab data and are
> not included in this public release. The deck (`CPM2_status_report.pptx`)
> ships pre-rendered; `build_deck.py` is included for provenance.

Informal status report and hand-off briefing (Adrian -> Niklas, Martin). A
candid, critical evaluation of pipeline quality, making the case that the
designs are probably not yet competitive with natural binders, and flagging
exactly what is unclear / under-powered and what Slurm runs are queued.

## Build

From the repo root with the `cpm2` conda env active (it has pandas, numpy,
matplotlib; `python-pptx` is pip-installed into it):

```bash
python notebooks/slides/build_deck.py
```

Re-runnable: it reads the benchmark CSVs + `final_report.md`, regenerates the
two PNGs under `assets/`, and rewrites `CPM2_status_report.pptx` (10 slides,
16:9).

## Outputs

- `CPM2_status_report.pptx` - the deck (10 slides).
- `assets/fig_design_vs_natural.png` - grouped bar: best design vs natural
  partner dG_MMGBSA per target, honest +-5 kcal/mol bars.
- `assets/fig_specificity_heatmap.png` - negative-control ddG-vs-cognate matrix
  (the centrepiece), CypA-scrambled flagged.

## Data sources (every number is anchored to a file; none invented)

- `benchmarks/md_20260528_negcontrols/final_report.md` + `results_running.csv`
  - the negative-control spine and the per-target winners table.
- `benchmarks/md_20260526_overnight/results_running.csv` +
  `trajectory_metrics.csv` - cognate designs and natural-reference baselines.
- `benchmarks/cpepmatch_sensitivity_20260529/README.md` +
  `benchmarks/ph_convergence_20260529/README.md` - queued Slurm experiments
  (designed and submitted, no results yet; presented as pending).
- `data/runs/mdm2_p53_v1_20260521_051455_26bc2b5c/output/summary.csv` +
  `mmgbsa_shortlist.csv` - the triage-funnel counts (37 matches -> 37 folded
  poses -> 148 refined designs -> 3 shortlisted).

## Notes

- No em dashes anywhere (user style rule).
- Every quantitative slide carries an "Epistemic flag" footnote where the data
  is weak (n=1 MD, +-5 kcal/mol true SE, un-converged Bcl-xL ref, FF artefacts,
  GB-only scoring).
