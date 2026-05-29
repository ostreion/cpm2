#!/usr/bin/env python
"""Finalize the MLflow run for this pipeline run (Snakemake rule mlflow_finalize).

Reattaches to the MLflow run via ``<run_root>/.mlflow_run_id``, logs final
metrics from ``output/summary.csv``, per-match metric series, benchmark
TSVs, and key artifacts (slim archive dir, summary.csv, alignment grid,
top PDBs). Sets run status to FAILED if any expected match is missing a
refine_results.json.

If the sentinel says ``disabled`` (mlflow not installed), this is a clean
no-op that just touches the output sentinel.

CLI: mlflow_finalize.py <pipeline_root> <config_name> <run_root> <out_sentinel>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _list_filtered_matches(run_root: Path) -> list[str]:
    renamed = run_root / "intermediate" / "1_cpepmatch_renamed"
    if not renamed.exists():
        return []
    return sorted(p.stem for p in renamed.glob("match*.pdb"))


def _count_failed_matches(run_root: Path) -> int:
    matches = _list_filtered_matches(run_root)
    ph = run_root / "intermediate" / "3_proteinhunter"
    failed = 0
    for m in matches:
        if not (ph / m / "refine_results.json").exists():
            failed += 1
    return failed


def main() -> int:
    pipeline_root, config_name, run_root, out_sentinel = (Path(p) for p in sys.argv[1:5])
    sys.path.insert(0, str(pipeline_root / "src"))

    out_sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel_in = run_root / ".mlflow_run_id"

    if not sentinel_in.exists():
        print("no .mlflow_run_id sentinel; nothing to finalize")
        out_sentinel.write_text("no-mlflow-sentinel\n")
        return 0

    mlflow_run_id = sentinel_in.read_text().strip()
    if mlflow_run_id == "disabled":
        print("MLflow disabled; finalize is a no-op")
        out_sentinel.write_text("disabled\n")
        return 0

    try:
        from cpm2.utils import mlflow_helpers
    except Exception as exc:  # noqa: BLE001
        print(f"mlflow_helpers import failed ({exc!r}); skipping finalize")
        out_sentinel.write_text(f"import-failed: {exc!r}\n")
        return 0

    if not getattr(mlflow_helpers, "_MLFLOW_OK", False):
        print("mlflow not installed; finalize is a no-op")
        out_sentinel.write_text("disabled\n")
        return 0

    ok = mlflow_helpers.attach_to_run(run_root, mlflow_run_id)
    if not ok:
        print("attach_to_run failed; skipping finalize")
        out_sentinel.write_text("attach-failed\n")
        return 0

    summary_csv = run_root / "output" / "summary.csv"
    if summary_csv.exists():
        mlflow_helpers.log_summary_csv(summary_csv)
        mlflow_helpers.log_per_match_metrics(summary_csv)

    benchmark_dir = run_root / "benchmark"
    mlflow_helpers.log_benchmark_tsvs(benchmark_dir)

    align_dir = run_root / "output" / "alignments"
    if align_dir.exists():
        mlflow_helpers.log_artifacts_dir(align_dir, "alignments")

    # Top PDBs (top_*.pdb under output/)
    output_dir = run_root / "output"
    if output_dir.exists():
        for pdb in output_dir.glob("top_*.pdb"):
            try:
                import mlflow as _mlflow  # type: ignore
                _mlflow.log_artifact(str(pdb))
            except Exception as exc:  # noqa: BLE001
                print(f"top pdb log failed: {exc!r}")
                break

    # Slim archive: locate via archives/<dir> matching run name suffix.
    archives_root = pipeline_root / "archives"
    if archives_root.exists():
        candidates = sorted(archives_root.glob(f"*{run_root.name}*"))
        if candidates:
            mlflow_helpers.log_artifacts_dir(candidates[-1], "archive")

    failed = _count_failed_matches(run_root)
    status = "FAILED" if failed > 0 else "FINISHED"
    if failed > 0:
        try:
            import mlflow as _mlflow  # type: ignore
            _mlflow.set_tag("partial_run", "true")
            _mlflow.log_metric("failed_matches_PH", float(failed))
        except Exception:  # noqa: BLE001
            pass

    mlflow_helpers.finalize(status, out_sentinel)

    # Make sure the sentinel exists even if finalize() didn't write it.
    if not out_sentinel.exists():
        payload = {"status": status, "failed_matches": failed,
                   "mlflow_run_id": mlflow_run_id}
        out_sentinel.write_text(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
