#!/usr/bin/env python
"""Per-match Boltz predict (called by Snakemake rule boltz_predict).

CLI:
    boltz_predict_chunk.py <pipeline_root> <config_name> <run_root> <match>

Runs ``boltz predict`` on a single yaml directly so that the output dir
naming (``boltz_results_<match>/predictions/<match>/...``) matches what
Snakemake declares as the rule's outputs. Calling boltz on a tempdir of
symlinks instead would name the output ``boltz_results_<tempdir>/...``,
which Snakemake can't see.

Outputs (under <run_root>/intermediate/2_boltz/predictions/):
  boltz_results_<match>/predictions/<match>/<match>_model_0.cif
  boltz_results_<match>/predictions/<match>/confidence_<match>_model_0.json

Filename retains the legacy ``_chunk`` suffix for git-history continuity;
the script is per-match now.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "boltz_predict_chunk.py: usage: <pipeline_root> <config_name> "
            "<run_root> <match>"
        )
    pipeline_root = Path(sys.argv[1])
    config_name = sys.argv[2]
    run_root = Path(sys.argv[3])
    match_name = sys.argv[4]
    sys.path.insert(0, str(pipeline_root / "src"))

    from cpm2.config_loader import load_config
    from cpm2.runners.boltz_runner import BoltzPredictConfig, run_predict

    config = load_config(pipeline_root, config_name, run_root=run_root)
    intermediate = run_root / "intermediate"
    yaml_path = intermediate / "2_boltz" / "yaml_input" / f"{match_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing YAML for match {match_name}: {yaml_path}")
    out_dir = intermediate / "2_boltz" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_pdb = intermediate / "1_cpepmatch_renamed" / f"{match_name}.pdb"
    predict_config = BoltzPredictConfig.from_dict(config["boltz"])

    run_predict(
        input_yaml=yaml_path,
        output_dir=out_dir,
        conda_env="boltz",
        predict_config=predict_config,
        input_pdb=input_pdb if input_pdb.exists() else None,
        cp_chain="P",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
