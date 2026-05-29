#!/usr/bin/env python
"""Start the MLflow run for this pipeline run (Snakemake rule mlflow_start).

Persists MLflow's own run_id to ``<run_root>/.mlflow_run_id`` so downstream
rules can reattach. If MLflow isn't installed (or fails to import), writes
the literal ``disabled`` to the sentinel and exits 0 cleanly so the rest of
the DAG keeps running.

CLI: mlflow_start.py <pipeline_root> <config_name> <run_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    pipeline_root, config_name, run_root = (Path(p) for p in sys.argv[1:4])
    sys.path.insert(0, str(pipeline_root / "src"))
    from cpm2.config_loader import load_config

    sentinel = run_root / ".mlflow_run_id"
    sentinel.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cpm2.utils import mlflow_helpers
    except Exception as exc:  # noqa: BLE001
        print(f"mlflow_helpers import failed ({exc!r}); writing disabled sentinel")
        sentinel.write_text("disabled\n")
        return 0

    if not getattr(mlflow_helpers, "_MLFLOW_OK", False):
        print("mlflow not installed; writing disabled sentinel")
        sentinel.write_text("disabled\n")
        return 0

    config = load_config(pipeline_root, str(config_name), run_root=run_root)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    active = mlflow_helpers.start_run(run_root, config, run_root.name, manifest=manifest)
    if active is None:
        print("start_run returned None; writing disabled sentinel")
        sentinel.write_text("disabled\n")
        return 0

    mlflow_run_id = active.info.run_id
    sentinel.write_text(mlflow_run_id + "\n")
    print(f"MLflow run started: {mlflow_run_id}")

    # End the active run in this process; downstream finalize will reattach
    # via the sentinel. MLflow runs are reattached, not kept alive across
    # rule boundaries.
    try:
        import mlflow as _mlflow  # type: ignore
        _mlflow.end_run()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
