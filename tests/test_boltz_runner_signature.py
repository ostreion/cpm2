"""Signature smoke tests for src/runners/boltz_runner.py.

Phase E shipped two parameter changes that callers depend on:
  * ``run_batch`` defaults ``batch_size=10`` (was ``None``).
  * ``run_predict`` / ``run_batch`` accept ``conda_env=None`` to skip the
    inner ``conda run`` prefix when the caller has already activated the
    target env (e.g. Snakemake's ``conda:`` directive).

These tests inspect the signatures only — they do not invoke Boltz.
"""

from __future__ import annotations

import inspect

from cpm2.runners.boltz_runner import run_batch, run_predict
from cpm2.runners.proteinhunter import run_refine


def test_run_batch_default_batch_size_is_10() -> None:
    sig = inspect.signature(run_batch)
    assert sig.parameters["batch_size"].default == 10


def test_run_batch_conda_env_default_is_boltz() -> None:
    sig = inspect.signature(run_batch)
    assert sig.parameters["conda_env"].default == "boltz"


def test_run_predict_conda_env_default_is_boltz() -> None:
    sig = inspect.signature(run_predict)
    assert sig.parameters["conda_env"].default == "boltz"


def test_proteinhunter_run_refine_conda_env_default() -> None:
    sig = inspect.signature(run_refine)
    assert sig.parameters["conda_env"].default == "proteinhunter"
