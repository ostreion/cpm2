"""Tests for src/utils/archiver.py result-card mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cpm2.utils.archiver import archive_run


def _build_fake_run(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Build a synthetic data/runs/<id>/ tree + complex PDB + config dict."""
    pipeline_root = tmp_path / "pipeline"
    (pipeline_root / "data").mkdir(parents=True)
    run_root = pipeline_root / "data" / "runs" / "fake_run"
    intermediate = run_root / "intermediate"
    output = run_root / "output"

    # Stage 0
    (intermediate / "0_import").mkdir(parents=True)
    (intermediate / "0_import" / "processed.pdb").write_text("HEADER processed\nEND\n")
    (intermediate / "0_import" / "template.cif").write_text("data_template\n")

    # Stage 1
    (intermediate / "1_cpepmatch").mkdir(parents=True)
    (intermediate / "1_cpepmatch" / "match_list.txt").write_text("match_001\nmatch_002\n")
    (intermediate / "1_cpepmatch" / "match_001.pdb").write_text("ATOM\n")
    (intermediate / "1_cpepmatch_renamed").mkdir(parents=True)
    (intermediate / "1_cpepmatch_renamed" / "match_001.pdb").write_text("ATOM\n")

    # Stage 2 (boltz tree)
    boltz_pred = intermediate / "2_boltz" / "predictions" / "boltz_results_match_001" / "predictions" / "match_001"
    boltz_pred.mkdir(parents=True)
    (boltz_pred / "match_001_model_0.cif").write_text("data_pred\n")
    conf = boltz_pred / "confidence_match_001_model_0.json"
    conf.write_text(json.dumps({"iptm": 0.9, "ptm": 0.8}))

    # Stage 3
    ph_dir = intermediate / "3_proteinhunter" / "match_001"
    ph_dir.mkdir(parents=True)
    (ph_dir / "refine_results.json").write_text(json.dumps([{"name": "design_0"}]))
    # A heavy per-cycle PDB that should NOT be in the lightweight archive.
    (ph_dir / "cycle_0.pdb").write_text("ATOM heavy\n")

    # Output
    output.mkdir(parents=True)
    (output / "summary.csv").write_text("name,iptm,plddt\nmatch_001,0.9,0.85\n")
    (output / "top_1_match_001.pdb").write_text("ATOM top\n")
    alignments = output / "alignments"
    alignments.mkdir()
    (alignments / "match_001_best.png").write_bytes(b"\x89PNG\r\n")

    # Manifest (Phase A)
    (run_root / "manifest.json").write_text(json.dumps({"run_id": "fake_run"}))

    # Source config YAML for verbatim copy
    config_yaml = pipeline_root / "configs" / "fake.yaml"
    config_yaml.parent.mkdir()
    config_yaml.write_text("# fake yaml\n")

    # Complex PDB outside the run tree
    complex_pdb = pipeline_root / "data" / "raw" / "complex.pdb"
    complex_pdb.parent.mkdir(parents=True)
    complex_pdb.write_text("HEADER complex\nEND\n")

    config = {
        "complex_pdb": complex_pdb,
        "input_target_chain": "A",
        "input_ligand_chain": "C",
        "cpepmatch": {"motif_size": 4},
        "boltz": {"iptm_threshold": 0.7},
        "proteinhunter": {"num_designs": 2},
        "run": {"name": "test"},
        "_config_yaml_path": config_yaml,
    }
    return pipeline_root, run_root, config


def test_archive_result_card_contents(tmp_path: Path) -> None:
    pipeline_root, run_root, config = _build_fake_run(tmp_path)

    archive_path = archive_run(
        config,
        pipeline_root=pipeline_root,
        run_root=run_root,
        clean=False,
        full_export=False,
    )

    # Required top-level entries
    for rel in (
        "manifest.json",
        "config.yaml",
        "run_info.json",
        "inputs/complex.pdb",
        "inputs/processed.pdb",
        "inputs/template.cif",
        "results/summary.csv",
        "results/alignments/match_001_best.png",
        "metadata/match_list.txt",
        "metadata/boltz_confidence/confidence_match_001_model_0.json",
        "metadata/refine_results/match_001/refine_results.json",
    ):
        assert (archive_path / rel).exists(), f"missing in result card: {rel}"

    # At least one top design copied
    top_designs = list((archive_path / "results" / "top_designs").glob("top_*.pdb"))
    assert len(top_designs) >= 1

    # Lightweight: no intermediates/ tree
    assert not (archive_path / "intermediates").exists(), \
        "result-card archive should not include intermediates/"

    # And the heavy PH per-cycle PDB is not in the archive anywhere
    cycle_hits = list(archive_path.rglob("cycle_*.pdb"))
    assert cycle_hits == [], f"unexpected per-cycle PDBs in archive: {cycle_hits}"


def test_archive_full_export_includes_intermediates(tmp_path: Path) -> None:
    pipeline_root, run_root, config = _build_fake_run(tmp_path)
    archive_path = archive_run(
        config,
        pipeline_root=pipeline_root,
        run_root=run_root,
        clean=False,
        full_export=True,
    )
    # Full export DOES include the intermediates tree.
    assert (archive_path / "intermediates" / "1_cpepmatch_renamed" / "match_001.pdb").exists()
    assert (archive_path / "intermediates" / "3_proteinhunter" / "match_001" / "cycle_0.pdb").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
