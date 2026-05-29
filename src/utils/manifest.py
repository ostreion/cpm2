"""Run provenance manifest.

Writes a JSON manifest at ``<run_root>/manifest.json`` capturing everything
needed to reproduce a pipeline run after the fact: git SHA + dirty flag,
config hash, env file hashes, input PDB SHA-256, hardware info, and seeds.

Designed to be best-effort: failures in subprocess calls (git, nvidia-smi)
or missing files degrade gracefully to ``null`` fields rather than raising,
so the manifest is always written even on detached HEAD or offline hosts.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def compute_pdb_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's contents.

    Public helper for callers wanting to record the input PDB hash.
    """
    h = _sha256_file(Path(path))
    if h is None:
        raise FileNotFoundError(path)
    return h


def _serialise_for_hash(obj: Any) -> Any:
    """Make a config dict deterministically JSON-serialisable for hashing."""
    if isinstance(obj, dict):
        return {k: _serialise_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise_for_hash(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _config_hash(config: dict) -> str:
    """SHA-256 of the resolved config, sorted keys, excluding _config_yaml_path."""
    cfg = {k: v for k, v in config.items() if k != "_config_yaml_path"}
    payload = json.dumps(_serialise_for_hash(cfg), sort_keys=True).encode()
    return _sha256_bytes(payload)


def _git_info(pipeline_root: Path) -> dict:
    info = {"sha": None, "sha_short": None, "dirty": None, "branch": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pipeline_root), capture_output=True, text=True, check=False,
        ).stdout.strip() or None
        info["sha"] = sha
        info["sha_short"] = sha[:8] if sha else None
    except (FileNotFoundError, OSError):
        pass
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(pipeline_root), capture_output=True, text=True, check=False,
        )
        info["dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (FileNotFoundError, OSError):
        pass
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(pipeline_root), capture_output=True, text=True, check=False,
        ).stdout.strip() or None
        info["branch"] = branch
    except (FileNotFoundError, OSError):
        pass
    return info


def _gpu_info() -> str | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def _env_hashes(pipeline_root: Path) -> dict:
    envs_dir = pipeline_root / "envs"
    out: dict = {}
    for name in ("cpepmatch", "boltz", "proteinhunter", "cpm2", "pym"):
        for ext in ("yml", "yaml"):
            p = envs_dir / f"{name}.{ext}"
            if p.exists():
                h = _sha256_file(p)
                if h is not None:
                    out[name] = h
                break
    return out


def write_run_manifest(run_root: Path, config: dict, run_id: str) -> Path:
    """Write ``<run_root>/manifest.json`` and return its path.

    Idempotent: overwrites on each call. Best-effort on subprocess failures.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    pipeline_root = None
    yaml_path = config.get("_config_yaml_path")
    if yaml_path:
        # YAML lives at <pipeline_root>/configs/<name>.yaml
        yp = Path(yaml_path)
        if yp.parent.name == "configs":
            pipeline_root = yp.parent.parent
    if pipeline_root is None:
        # Fall back: run_root is <pipeline_root>/data/runs/<id>
        try:
            pipeline_root = run_root.parent.parent.parent
        except IndexError:
            pipeline_root = run_root

    complex_pdb = config.get("complex_pdb")
    complex_pdb_str = str(complex_pdb) if complex_pdb else None
    complex_pdb_hash = None
    if complex_pdb and Path(complex_pdb).exists():
        complex_pdb_hash = _sha256_file(Path(complex_pdb))

    manifest = {
        "run_id": run_id,
        "timestamp_iso": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git": _git_info(pipeline_root),
        "config_yaml_path": str(yaml_path) if yaml_path else None,
        "config_hash": _config_hash(config),
        "input": {
            "complex_pdb": complex_pdb_str,
            "complex_pdb_sha256": complex_pdb_hash,
            "target_chain": config.get("input_target_chain"),
            "ligand_chain": config.get("input_ligand_chain"),
        },
        "envs": _env_hashes(pipeline_root),
        "hardware": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": _gpu_info(),
        },
        "seeds": {
            "boltz_seed": (config.get("boltz") or {}).get("seed"),
            "proteinhunter_seed": (config.get("proteinhunter") or {}).get("seed"),
        },
        "cpepmatch_db_path": str(pipeline_root / "lib" / "cPEPmatch" / "database"),
        "pipeline_root": str(Path(pipeline_root).resolve()),
    }

    out_path = run_root / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path
