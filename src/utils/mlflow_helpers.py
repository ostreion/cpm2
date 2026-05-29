"""Lazy MLflow integration helpers for the CPM2 pipeline.

A single MLflow run per pipeline run, named after RUN_ID. Logs flattened
config params, per-stage wall time + peak VRAM, final summary metrics, and
key artefacts (manifest.json, summary.csv, alignment grid).

MLflow is optional. If the import fails, all helpers no-op with a single
warning so the notebook still runs in the (interactive) cpm2 env even when
mlflow isn't installed there yet.

Tracking URI defaults to ``file://<pipeline_root>/mlruns``. Override via the
``MLFLOW_TRACKING_URI`` env var. Experiment name: ``cpm2``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

try:
    import mlflow as _mlflow  # type: ignore
    _MLFLOW_OK = True
except Exception as _exc:  # noqa: BLE001
    _mlflow = None
    _MLFLOW_OK = False
    _IMPORT_ERR = _exc


def _warn_disabled(reason: str) -> None:
    warnings.warn(f"MLflow disabled: {reason}", stacklevel=2)


def _flatten(prefix: str, obj: Any, out: dict) -> None:
    """Flatten a (possibly nested) config dict into dot-key params."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):  # internal keys like _config_yaml_path
                continue
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        # Skip lists (mlflow params are scalars). Stringify short ones.
        s = json.dumps(obj, default=str)
        if len(s) <= 250:
            out[prefix] = s
    elif isinstance(obj, Path):
        out[prefix] = str(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        out[prefix] = obj


def _set_tracking_uri(pipeline_root: Path) -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    uri = f"file://{Path(pipeline_root).resolve()}/mlruns"
    _mlflow.set_tracking_uri(uri)


def start_run(run_root: Path, config: dict, run_id: str, manifest: dict | None = None):
    """Start an MLflow run for this pipeline run.

    Returns the active run object on success, or ``None`` if MLflow is unavailable.
    """
    if not _MLFLOW_OK:
        _warn_disabled(f"import failed: {_IMPORT_ERR!r}")
        return None
    pipeline_root = Path(manifest.get("pipeline_root")) if manifest and manifest.get("pipeline_root") else Path(run_root).parent.parent.parent
    _set_tracking_uri(pipeline_root)
    _mlflow.set_experiment("cpm2")

    active = _mlflow.start_run(run_name=run_id)

    params: dict = {}
    _flatten("", {k: v for k, v in config.items() if not k.startswith("_")}, params)
    # MLflow caps each param to 500 chars; strip overlong values.
    safe_params = {k: (v if not isinstance(v, str) or len(v) <= 500 else v[:497] + "...")
                   for k, v in params.items()}
    try:
        _mlflow.log_params(safe_params)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_params failed: {exc!r}", stacklevel=2)

    if manifest:
        manifest_path = Path(run_root) / "manifest.json"
        if manifest_path.exists():
            _mlflow.log_artifact(str(manifest_path))
        git = manifest.get("git") or {}
        tags = {
            "git.sha": git.get("sha") or "",
            "git.branch": git.get("branch") or "",
            "git.dirty": str(git.get("dirty")),
            "config.name": Path(manifest.get("config_yaml_path") or "").stem,
            "config.hash": manifest.get("config_hash") or "",
            "run_id": run_id,
        }
        try:
            _mlflow.set_tags({k: v for k, v in tags.items() if v})
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"MLflow set_tags failed: {exc!r}", stacklevel=2)
    return active


def log_stage_timing(stage_name: str, wall_time_seconds: float,
                     peak_vram_mb: float | None = None) -> None:
    if not _MLFLOW_OK:
        return
    try:
        _mlflow.log_metric(f"{stage_name}_seconds", float(wall_time_seconds))
        if peak_vram_mb is not None:
            _mlflow.log_metric(f"{stage_name}_peak_vram_mb", float(peak_vram_mb))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_stage_timing failed: {exc!r}", stacklevel=2)


def log_final_metrics(summary_csv: Path) -> None:
    """Read a sorted summary CSV and log post-ProteinHunter scalars.

    Names carry a ``_PH`` suffix since ``summary.csv`` aggregates each
    ProteinHunter refinement's final-cycle Boltz scores.
    """
    if not _MLFLOW_OK:
        return
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return
    try:
        import csv
        with open(summary_csv) as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        _mlflow.log_metric("num_designs_PH", float(n))
        if rows:
            try:
                top_iptm = float(rows[0].get("iptm", 0) or 0)
                top_plddt = float(rows[0].get("plddt", 0) or 0)
                _mlflow.log_metric("top_iptm_PH", top_iptm)
                _mlflow.log_metric("top_plddt_PH", top_plddt)
            except (TypeError, ValueError):
                pass
            match_names = set()
            for r in rows:
                name = r.get("name") or ""
                # name pattern: <match>_model_0_designN -> strip suffix
                stem = name.rsplit("_model_0_design", 1)[0] if "_model_0_design" in name else name
                if stem:
                    match_names.add(stem)
            _mlflow.log_metric("num_matches_PH", float(len(match_names)))
        _mlflow.log_artifact(str(summary_csv))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_final_metrics failed: {exc!r}", stacklevel=2)


def log_artifacts_dir(dir_path: Path, artifact_subdir: str) -> None:
    if not _MLFLOW_OK:
        return
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return
    try:
        _mlflow.log_artifacts(str(dir_path), artifact_path=artifact_subdir)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_artifacts_dir failed: {exc!r}", stacklevel=2)


def end_run(status: str = "FINISHED") -> None:
    if not _MLFLOW_OK:
        return
    try:
        _mlflow.end_run(status=status)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Phase F2 helpers: reattach + finalize from a separate Snakemake rule.
#
# The Snakemake DAG runs each rule in its own subprocess, so the MLflow run
# started by ``mlflow_start`` is closed there. The ``mlflow_finalize`` rule
# uses ``attach_to_run`` to reopen it via the persisted run_id sentinel.
# ---------------------------------------------------------------------------

def attach_to_run(run_root: Path, mlflow_run_id: str) -> bool:
    """Reopen an existing MLflow run by id. Returns True on success."""
    if not _MLFLOW_OK:
        return False
    try:
        pipeline_root = Path(run_root).parent.parent.parent
        _set_tracking_uri(pipeline_root)
        _mlflow.set_experiment("cpm2")
        _mlflow.start_run(run_id=mlflow_run_id)
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow attach_to_run failed: {exc!r}", stacklevel=2)
        return False


def log_summary_csv(summary_csv: Path) -> None:
    """Backwards-compatible alias for ``log_final_metrics``."""
    log_final_metrics(summary_csv)


def log_per_match_metrics(summary_csv: Path) -> None:
    """Log per-match metric series with step=match_index from summary.csv.

    Each row in summary.csv becomes a step on the ``match_iptm_PH`` /
    ``match_plddt_PH`` / ``match_target_ca_rmsd_PH`` series. The ``_PH``
    suffix marks these as post-ProteinHunter outputs (Boltz scores on the
    final refined complex). Index reflects sorted CSV order
    (collect_designs sorts by ipTM desc).
    """
    if not _MLFLOW_OK:
        return
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return
    try:
        import csv
        with open(summary_csv) as f:
            rows = list(csv.DictReader(f))
        rmsd_values: list[float] = []
        for i, r in enumerate(rows):
            for key, label in (("iptm", "match_iptm_PH"),
                               ("plddt", "match_plddt_PH"),
                               ("target_ca_rmsd", "match_target_ca_rmsd_PH")):
                v = r.get(key)
                if v is None or v == "":
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                _mlflow.log_metric(label, fv, step=i)
                if key == "target_ca_rmsd":
                    rmsd_values.append(fv)
        if rmsd_values:
            _mlflow.log_metric("mean_target_ca_rmsd_PH",
                               sum(rmsd_values) / len(rmsd_values))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_per_match_metrics failed: {exc!r}", stacklevel=2)


def log_benchmark_tsvs(benchmark_dir: Path) -> None:
    """Read each Snakemake-emitted benchmark TSV and log as scalar metrics.

    Snakemake writes a single-row TSV per rule invocation with columns
    ``s, h:m:s, max_rss, max_vms, max_uss, max_pss, io_in, io_out,
    mean_load, cpu_time``. We log ``<rule>_seconds`` from ``s`` and
    ``<rule>_max_rss_mb`` from ``max_rss`` (already MB per Snakemake).
    For per-match rules (e.g., boltz_predict), the per-match TSVs are
    aggregated to mean / max scalars and the raw TSVs are uploaded as
    artifacts for full transparency.
    """
    if not _MLFLOW_OK:
        return
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        return
    try:
        import csv
        # Per-rule aggregation of seconds and max_rss across (possibly
        # multiple) per-match TSVs sitting in benchmark/<rule>/<match>.tsv
        # in addition to flat benchmark/<rule>.tsv files.
        per_rule_seconds: dict[str, list[float]] = {}
        per_rule_rss: dict[str, list[float]] = {}

        def _read_one(tsv_path: Path, rule_name: str) -> None:
            try:
                with open(tsv_path) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    rows = list(reader)
                if not rows:
                    return
                row = rows[0]
                s = row.get("s") or ""
                if s:
                    try:
                        per_rule_seconds.setdefault(rule_name, []).append(float(s))
                    except (TypeError, ValueError):
                        pass
                rss = row.get("max_rss") or ""
                if rss:
                    try:
                        per_rule_rss.setdefault(rule_name, []).append(float(rss))
                    except (TypeError, ValueError):
                        pass
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"benchmark tsv {tsv_path} failed: {exc!r}", stacklevel=2)

        for tsv in sorted(benchmark_dir.glob("*.tsv")):
            _read_one(tsv, tsv.stem)
        for sub in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            for tsv in sorted(sub.glob("*.tsv")):
                _read_one(tsv, sub.name)

        for rule_name, secs in per_rule_seconds.items():
            try:
                if len(secs) == 1:
                    _mlflow.log_metric(f"{rule_name}_seconds", secs[0])
                else:
                    _mlflow.log_metric(f"{rule_name}_seconds_total", sum(secs))
                    _mlflow.log_metric(f"{rule_name}_seconds_mean",
                                       sum(secs) / len(secs))
                    _mlflow.log_metric(f"{rule_name}_invocations", float(len(secs)))
            except Exception:  # noqa: BLE001
                pass
        for rule_name, rss in per_rule_rss.items():
            try:
                _mlflow.log_metric(f"{rule_name}_max_rss_mb", max(rss))
            except Exception:  # noqa: BLE001
                pass
        try:
            _mlflow.log_artifacts(str(benchmark_dir), artifact_path="benchmark")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"benchmark artifact upload failed: {exc!r}", stacklevel=2)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow log_benchmark_tsvs failed: {exc!r}", stacklevel=2)


def finalize(status: str, sentinel_out_path: Path | None = None) -> None:
    """End the active MLflow run with ``status`` and touch ``sentinel_out_path``."""
    end_run(status=status)
    if sentinel_out_path is not None:
        try:
            sentinel_out_path = Path(sentinel_out_path)
            sentinel_out_path.parent.mkdir(parents=True, exist_ok=True)
            sentinel_out_path.write_text(f"{status}\n")
        except OSError as exc:
            warnings.warn(f"finalize sentinel write failed: {exc!r}", stacklevel=2)
