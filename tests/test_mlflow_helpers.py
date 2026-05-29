"""Tests for src/utils/mlflow_helpers.py.

The point of these tests is to verify the lazy/no-op contract: pipeline
runs must not crash if MLflow ever fails to import or be available, even
though Phase F2 makes mlflow a first-class runtime dep.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _reload_helpers_with_mlflow_disabled(monkeypatch):
    """Force a fresh import of mlflow_helpers with mlflow unavailable."""
    # Drop any pre-imported mlflow + helpers from sys.modules.
    for mod in list(sys.modules):
        if mod == "mlflow" or mod.startswith("mlflow."):
            sys.modules.pop(mod, None)
        if mod == "cpm2.utils.mlflow_helpers":
            sys.modules.pop(mod, None)

    # Stub mlflow as missing by raising ImportError on import.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    helpers = importlib.import_module("cpm2.utils.mlflow_helpers")
    return helpers


def test_helpers_noop_when_mlflow_missing(monkeypatch, tmp_path: Path) -> None:
    helpers = _reload_helpers_with_mlflow_disabled(monkeypatch)
    assert helpers._MLFLOW_OK is False

    # All public helpers should run without raising and without side effects
    # other than a possible warning.
    assert helpers.start_run(tmp_path, {"x": 1}, "rid", manifest=None) is None
    helpers.log_stage_timing("foo", 1.5)
    helpers.log_final_metrics(tmp_path / "missing.csv")
    helpers.log_artifacts_dir(tmp_path, "sub")
    helpers.end_run("FINISHED")
    assert helpers.attach_to_run(tmp_path, "anyid") is False
    helpers.log_summary_csv(tmp_path / "missing.csv")
    helpers.log_per_match_metrics(tmp_path / "missing.csv")
    helpers.log_benchmark_tsvs(tmp_path)

    sentinel = tmp_path / ".mlflow_finalized"
    helpers.finalize("FAILED", sentinel)
    # finalize still writes the sentinel even when mlflow is disabled,
    # so the Snakemake rule's output is satisfied.
    assert sentinel.exists()
    assert sentinel.read_text().strip() == "FAILED"


def test_finalize_writes_sentinel_with_mlflow_present(tmp_path: Path) -> None:
    # When mlflow IS importable, finalize() should still touch the sentinel.
    # Use a fresh import (no monkeypatching) so the real mlflow is used.
    for mod in list(sys.modules):
        if mod == "cpm2.utils.mlflow_helpers":
            sys.modules.pop(mod, None)
    helpers = importlib.import_module("cpm2.utils.mlflow_helpers")

    sentinel = tmp_path / ".mlflow_finalized"
    # No active run -> end_run is a no-op, but the sentinel must still be touched.
    helpers.finalize("FINISHED", sentinel)
    assert sentinel.exists()
    assert sentinel.read_text().strip() == "FINISHED"
