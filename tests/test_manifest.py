"""Smoke tests for src/utils/manifest.py."""

from __future__ import annotations

import json
from pathlib import Path

from cpm2.utils.manifest import compute_pdb_sha256, write_run_manifest


def _fake_config(tmp_path: Path) -> dict:
    pdb = tmp_path / "fake.pdb"
    pdb.write_text("HEADER fake\nEND\n")
    return {
        "complex_pdb": pdb,
        "input_target_chain": "A",
        "input_ligand_chain": "C",
        "cpepmatch": {"motif_size": 4},
        "boltz": {"iptm_threshold": 0.7, "seed": 42},
        "proteinhunter": {"num_designs": 2, "seed": 7},
    }


def test_write_run_manifest_basic(tmp_path: Path) -> None:
    run_root = tmp_path / "data" / "runs" / "test_run"
    config = _fake_config(tmp_path)

    out = write_run_manifest(run_root, config, "test_run")
    assert out == run_root / "manifest.json"
    assert out.exists()

    data = json.loads(out.read_text())
    assert data["run_id"] == "test_run"
    assert isinstance(data["timestamp_iso"], str) and data["timestamp_iso"].endswith("Z")
    assert isinstance(data["git"], dict)
    for key in ("sha", "sha_short", "dirty", "branch"):
        assert key in data["git"]
    assert isinstance(data["config_hash"], str) and len(data["config_hash"]) == 64
    assert data["input"]["target_chain"] == "A"
    assert data["input"]["ligand_chain"] == "C"
    assert isinstance(data["input"]["complex_pdb_sha256"], str)
    assert len(data["input"]["complex_pdb_sha256"]) == 64
    assert isinstance(data["envs"], dict)
    assert isinstance(data["hardware"], dict)
    assert "platform" in data["hardware"]
    assert "python_version" in data["hardware"]
    assert data["seeds"] == {"boltz_seed": 42, "proteinhunter_seed": 7}
    assert isinstance(data["pipeline_root"], str)


def test_write_run_manifest_idempotent(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    config = _fake_config(tmp_path)
    p1 = write_run_manifest(run_root, config, "id1")
    first = p1.read_text()
    p2 = write_run_manifest(run_root, config, "id1")
    assert p1 == p2
    # File still parseable; content may differ only in timestamp.
    second = json.loads(p2.read_text())
    assert second["run_id"] == "id1"
    # Confirm we can re-parse the first too.
    json.loads(first)


def test_compute_pdb_sha256(tmp_path: Path) -> None:
    p = tmp_path / "x.pdb"
    p.write_bytes(b"ATOM\n")
    h = compute_pdb_sha256(p)
    assert isinstance(h, str) and len(h) == 64
