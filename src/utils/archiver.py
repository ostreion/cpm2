"""
Archive utility for CPNext pipeline runs.

Provides functions to archive pipeline outputs, list past runs, and retrieve archive details.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def _detect_pipeline_root() -> Path:
    """Auto-detect PIPELINE root directory."""
    # Try common locations relative to this file
    this_file = Path(__file__).resolve()
    # This file is at PIPELINE/src/utils/archiver.py
    pipeline_root = this_file.parent.parent.parent
    if (pipeline_root / "data").exists():
        return pipeline_root
    raise RuntimeError("Could not auto-detect PIPELINE root. Please provide pipeline_root parameter.")


def _get_archives_dir(pipeline_root: Path) -> Path:
    """Get archives directory, creating if needed."""
    archives_dir = pipeline_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    return archives_dir


def _load_manifest(archives_dir: Path) -> dict:
    """Load manifest.json or return empty structure."""
    manifest_path = archives_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"version": "1.0", "runs": []}


def _save_manifest(archives_dir: Path, manifest: dict) -> None:
    """Save manifest.json."""
    manifest_path = archives_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _extract_results_summary(output_dir: Path) -> dict:
    """Extract summary statistics from output files."""
    summary = {
        "cpepmatch_matches": 0,
        "cpepmatch_filtered": 0,
        "boltz_predicted": 0,
        "boltz_validated": 0,
        "proteinhunter_designs": 0,
        "top_iptm": None,
        "top_plddt": None,
    }

    # Read summary.csv for design count and top scores
    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists():
        import csv
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            summary["proteinhunter_designs"] = len(rows)
            if rows:
                # Rows are sorted by iptm descending
                summary["top_iptm"] = float(rows[0].get("iptm", 0))
                summary["top_plddt"] = float(rows[0].get("plddt", 0))

    return summary


def _count_intermediate_files(intermediate_dir: Path) -> dict:
    """Count files in intermediate directories for summary stats."""
    counts = {
        "cpepmatch_matches": 0,
        "cpepmatch_filtered": 0,
        "boltz_predicted": 0,
        "boltz_validated": 0,
    }

    # Count cPEPmatch matches (original dir)
    cpepmatch_dir = intermediate_dir / "1_cpepmatch"
    if cpepmatch_dir.exists():
        counts["cpepmatch_matches"] = len(list(cpepmatch_dir.glob("match*.pdb")))

    # Count filtered matches (renamed dir)
    renamed_dir = intermediate_dir / "1_cpepmatch_renamed"
    if renamed_dir.exists():
        counts["cpepmatch_filtered"] = len(list(renamed_dir.glob("match*.pdb")))

    # Count Boltz predictions
    boltz_predictions = intermediate_dir / "2_boltz" / "predictions"
    if boltz_predictions.exists():
        # Find the results directory (boltz_results_*)
        for results_dir in boltz_predictions.glob("boltz_results_*"):
            pred_dir = results_dir / "predictions"
            if pred_dir.exists():
                counts["boltz_predicted"] = len(list(pred_dir.iterdir()))

    # Count validated (those that went to ProteinHunter)
    ph_dir = intermediate_dir / "3_proteinhunter"
    if ph_dir.exists():
        counts["boltz_validated"] = len([d for d in ph_dir.iterdir() if d.is_dir()])

    return counts


def archive_run(
    config: dict,
    run_name: Optional[str] = None,
    pipeline_root: Optional[Path] = None,
    clean: bool = True,
    full_export: bool = False,
    run_root: Optional[Path] = None,
) -> Path:
    """Archive current pipeline run as a curated "result card" (default).

    The default mode (``full_export=False``) writes a small, shareable artefact
    (a few MB) intended for long-term retention. The heavy reproducibility data
    (every match PDB, every Boltz mmCIF, every PH cycle) continues to live at
    ``data/runs/<run_id>/`` and is *not* duplicated into the archive. Use
    ``full_export=True`` only when you specifically want a self-contained
    snapshot that includes the full intermediate tree.

    Result-card contents (always written when source files exist):
        inputs/                      complex.pdb, processed.pdb, template.cif
        manifest.json                copied from <run_root>/manifest.json
        config.yaml                  verbatim source YAML
        run_info.json                serialized config + summary stats
        results/summary.csv
        results/refinement_metrics.png  (if present)
        results/top_designs/*.pdb
        results/alignments/             per-match alignment PNGs / grid
        metadata/match_list.txt
        metadata/boltz_confidence/      confidence JSONs per match
        metadata/refine_results/        refine_results.json per match

    Result-card mode does NOT copy the ``intermediates/`` tree.

    Args:
        config: The notebook's config dict containing:
            - config["complex_pdb"]: input PDB path
            - config["input_ligand_chain"]: ligand chain ID
            - config["input_target_chain"]: target chain ID
            - config["cpepmatch"], config["boltz"], config["proteinhunter"]: stage configs
            - (optional) config["run"]["name"]: human-friendly run name
            - (optional) config["_config_yaml_path"]: source YAML, copied verbatim
        run_name: Optional custom name for the archive (default: auto-generated from timestamp + PDB name)
        pipeline_root: Path to PIPELINE directory (default: auto-detect)
        clean: If True (default), delete intermediate and output folders after archiving
        full_export: If True, also copy the entire intermediate tree (every
            cPEPmatch match PDB, every Boltz mmCIF with PAE/PDE if present,
            every ProteinHunter cycle) into ``archive_dir/intermediates/``.
            Default False keeps the lightweight result-card archive.

    Returns:
        Path to created archive directory.
    """
    if pipeline_root is None:
        pipeline_root = _detect_pipeline_root()
    pipeline_root = Path(pipeline_root)

    # Setup paths. When run_root is given (Phase A+), the per-run layout is
    # <run_root>/{intermediate,output}/...; otherwise we fall back to the
    # legacy global <pipeline_root>/data/{intermediate,output}/.
    data_dir = pipeline_root / "data"
    if run_root is not None:
        run_root = Path(run_root)
        intermediate_dir = run_root / "intermediate"
        output_dir = run_root / "output"
    else:
        intermediate_dir = data_dir / "intermediate"
        output_dir = data_dir / "output"
    archives_dir = _get_archives_dir(pipeline_root)

    # Generate run ID and name
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    # Get input PDB name
    complex_pdb = Path(config["complex_pdb"])
    pdb_stem = complex_pdb.stem

    # Prefer explicit run_name arg, then config["run"]["name"], then PDB stem.
    if not run_name:
        run_name = (config.get("run") or {}).get("name")
    if run_name:
        run_id = f"run_{timestamp_str}_{run_name}"
    else:
        run_id = f"run_{timestamp_str}_{pdb_stem}"

    # Create archive directory
    archive_dir = archives_dir / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (archive_dir / "inputs").mkdir(exist_ok=True)
    (archive_dir / "results").mkdir(exist_ok=True)
    (archive_dir / "results" / "top_designs").mkdir(exist_ok=True)
    (archive_dir / "metadata").mkdir(exist_ok=True)
    (archive_dir / "metadata" / "boltz_confidence").mkdir(exist_ok=True)
    (archive_dir / "metadata" / "refine_results").mkdir(exist_ok=True)

    # Copy input PDB
    if complex_pdb.exists():
        shutil.copy(complex_pdb, archive_dir / "inputs" / complex_pdb.name)

    # Snapshot stage-0 import artifacts so retro-rendering and provenance survive
    # even after a later run overwrites the global data/intermediate/0_import dir.
    import_dir = intermediate_dir / "0_import"
    for fname in ("processed.pdb", "template.cif"):
        src = import_dir / fname
        if src.exists():
            shutil.copy(src, archive_dir / "inputs" / fname)

    # Copy per-run manifest.json (provenance: git SHA, config hash, env hashes,
    # input hash, hardware, seeds) into the archive root.
    if run_root is not None:
        manifest_src = run_root / "manifest.json"
        if manifest_src.exists():
            shutil.copy(manifest_src, archive_dir / "manifest.json")

    # Copy source config YAML verbatim, if the loader stashed its path.
    config_yaml_path = config.get("_config_yaml_path")
    if config_yaml_path:
        config_yaml_path = Path(config_yaml_path)
        if config_yaml_path.exists():
            shutil.copy(config_yaml_path, archive_dir / "config.yaml")

    # Copy output files
    for filename in ["summary.csv", "refinement_metrics.png", "session.pse"]:
        src = output_dir / filename
        if src.exists():
            shutil.copy(src, archive_dir / "results" / filename)

    # Copy top design PDBs (top_N_*.pdb pattern)
    for top_pdb in output_dir.glob("top_*.pdb"):
        shutil.copy(top_pdb, archive_dir / "results" / "top_designs" / top_pdb.name)

    # Copy Stage 3.5 alignment PNGs (per-match best PNGs and/or all_grid.png).
    alignments_src = output_dir / "alignments"
    if alignments_src.exists() and alignments_src.is_dir():
        shutil.copytree(
            alignments_src,
            archive_dir / "results" / "alignments",
            dirs_exist_ok=True,
        )

    # Copy cPEPmatch metadata
    match_list = intermediate_dir / "1_cpepmatch" / "match_list.txt"
    if match_list.exists():
        shutil.copy(match_list, archive_dir / "metadata" / "match_list.txt")

    # Copy Boltz confidence JSONs
    boltz_predictions = intermediate_dir / "2_boltz" / "predictions"
    if boltz_predictions.exists():
        for results_dir in boltz_predictions.glob("boltz_results_*"):
            pred_dir = results_dir / "predictions"
            if pred_dir.exists():
                for match_dir in pred_dir.iterdir():
                    if match_dir.is_dir():
                        for conf_json in match_dir.glob("confidence_*.json"):
                            shutil.copy(
                                conf_json,
                                archive_dir / "metadata" / "boltz_confidence" / conf_json.name
                            )

    # Copy ProteinHunter refine_results.json files
    ph_dir = intermediate_dir / "3_proteinhunter"
    if ph_dir.exists():
        for match_subdir in ph_dir.iterdir():
            if match_subdir.is_dir():
                refine_json = match_subdir / "refine_results.json"
                if refine_json.exists():
                    dest_dir = archive_dir / "metadata" / "refine_results" / match_subdir.name
                    dest_dir.mkdir(exist_ok=True)
                    shutil.copy(refine_json, dest_dir / "refine_results.json")

    # Full-export mode: preserve all intermediates (large but reproducible).
    if full_export:
        full_dir = archive_dir / "intermediates"
        full_dir.mkdir(exist_ok=True)
        # Stage 1: every cPEPmatch match PDB and the renamed copies.
        for sub in ("1_cpepmatch", "1_cpepmatch_renamed"):
            src = intermediate_dir / sub
            if src.exists():
                shutil.copytree(src, full_dir / sub, dirs_exist_ok=True)
        # Stage 2: full Boltz prediction tree (mmCIFs, confidence, PAE/PDE).
        boltz_root = intermediate_dir / "2_boltz"
        if boltz_root.exists():
            shutil.copytree(boltz_root, full_dir / "2_boltz", dirs_exist_ok=True)
        # Stage 3: full ProteinHunter per-cycle outputs.
        ph_src = intermediate_dir / "3_proteinhunter"
        if ph_src.exists():
            shutil.copytree(ph_src, full_dir / "3_proteinhunter", dirs_exist_ok=True)

    # Build run_info.json
    results_summary = _extract_results_summary(output_dir)
    intermediate_counts = _count_intermediate_files(intermediate_dir)
    results_summary.update(intermediate_counts)

    # Serialize config (convert Path objects to strings)
    def serialize_config(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: serialize_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_config(v) for v in obj]
        return obj

    run_info = {
        "id": run_id,
        "timestamp": timestamp.isoformat(),
        "run": serialize_config(config.get("run", {})),
        "config_yaml": (Path(config_yaml_path).name
                        if config_yaml_path and Path(config_yaml_path).exists()
                        else None),
        "full_export": full_export,
        "input": {
            "complex_pdb": complex_pdb.name,
            "ligand_chain": config.get("input_ligand_chain"),
            "target_chain": config.get("input_target_chain"),
        },
        "config": {
            "cpepmatch": serialize_config(config.get("cpepmatch", {})),
            "boltz": serialize_config(config.get("boltz", {})),
            "proteinhunter": serialize_config(config.get("proteinhunter", {})),
        },
        "results": results_summary,
    }

    (archive_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Update manifest
    manifest = _load_manifest(archives_dir)
    manifest["runs"].append({
        "id": run_id,
        "timestamp": timestamp.isoformat(),
        "run_name": (config.get("run") or {}).get("name"),
        "config_yaml": (Path(config_yaml_path).name
                        if config_yaml_path and Path(config_yaml_path).exists()
                        else None),
        "full_export": full_export,
        "input_pdb": complex_pdb.name,
        "archive_path": str(archive_dir.relative_to(pipeline_root)),
        "top_iptm": results_summary.get("top_iptm"),
        "top_plddt": results_summary.get("top_plddt"),
        "num_designs": results_summary.get("proteinhunter_designs", 0),
    })
    _save_manifest(archives_dir, manifest)

    print(f"Archived run '{run_id}' to {archive_dir}")

    # Clean up intermediate and output directories
    if clean:
        for d in [intermediate_dir, output_dir]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        scope = "run" if run_root is not None else "global"
        print(f"Cleaned {scope} intermediate and output directories.")

    return archive_dir


def list_archives(pipeline_root: Optional[Path] = None) -> list[dict]:
    """
    List all archived runs with summary info.

    Args:
        pipeline_root: Path to PIPELINE directory (default: auto-detect)

    Returns:
        List of run summary dicts.
    """
    if pipeline_root is None:
        pipeline_root = _detect_pipeline_root()
    pipeline_root = Path(pipeline_root)

    archives_dir = _get_archives_dir(pipeline_root)
    manifest = _load_manifest(archives_dir)

    runs = manifest.get("runs", [])

    if not runs:
        print("No archived runs found.")
        return []

    # Print table
    print(f"{'ID':<35} {'Input PDB':<15} {'Designs':>8} {'Top ipTM':>10} {'Top pLDDT':>10}")
    print("-" * 80)
    for run in runs:
        iptm = f"{run.get('top_iptm', 0):.3f}" if run.get('top_iptm') else "N/A"
        plddt = f"{run.get('top_plddt', 0):.3f}" if run.get('top_plddt') else "N/A"
        print(f"{run['id']:<35} {run.get('input_pdb', 'N/A'):<15} {run.get('num_designs', 0):>8} {iptm:>10} {plddt:>10}")

    return runs


def get_archive_info(run_id: str, pipeline_root: Optional[Path] = None) -> dict:
    """
    Get detailed info about a specific archived run.

    Args:
        run_id: The run ID (e.g., "run_20260127_143000_3sgq")
        pipeline_root: Path to PIPELINE directory (default: auto-detect)

    Returns:
        Full run_info.json contents, or empty dict if not found.
    """
    if pipeline_root is None:
        pipeline_root = _detect_pipeline_root()
    pipeline_root = Path(pipeline_root)

    archives_dir = _get_archives_dir(pipeline_root)
    run_info_path = archives_dir / run_id / "run_info.json"

    if not run_info_path.exists():
        print(f"Archive not found: {run_id}")
        return {}

    return json.loads(run_info_path.read_text())
