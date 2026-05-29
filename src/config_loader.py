"""YAML-based config loader for CPNext pipeline runs.

Usage from a notebook:

    from cpm2.config_loader import load_config
    config = load_config(PIPELINE_ROOT, "mdm2_p53_v1")

The loader:
  1. Builds the default config via `make_default_config(pipeline_root)`.
  2. Reads `configs/{name}.yaml` (or an absolute path) from disk.
  3. Deep-merges YAML values over the defaults.
  4. Resolves `complex_pdb` relative to PIPELINE_ROOT if not absolute.
  5. Validates and returns the dict.

The path of the source YAML is stashed at config["_config_yaml_path"] so the
archiver can copy it verbatim into the run archive.

The optional top-level `run:` block (with `name`, `notes`) is preserved at
config["run"] for the archiver to use as run metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from cpm2.default_config import make_default_config, validate_config


def resolve_msa_cache_path(pipeline_root: Path, target_seq: str) -> Path:
    """Return the shared MSA cache CSV path keyed on the target sequence.

    The CSV lives at ``<pipeline_root>/data/cache/msa/<sha8>.csv``; the sibling
    ``.npz`` (Boltz pre-pairing artefact) lives at the same stem with a
    ``.npz`` suffix. Caller is responsible for creating the parent dir.
    """
    seq_sha8 = hashlib.sha256(target_seq.encode()).hexdigest()[:8]
    return Path(pipeline_root) / "data" / "cache" / "msa" / f"{seq_sha8}.csv"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Returns a new dict.

    Lists and scalars in `override` replace the value in `base`. Dicts merge
    key-by-key. None values in `override` are kept (explicit override).
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_yaml_path(pipeline_root: Path, name_or_path: str) -> Path:
    """Resolve a config name to a concrete YAML path.

    Accepts:
      - bare name: "mdm2_p53_v1" -> PIPELINE_ROOT/configs/mdm2_p53_v1.yaml
      - relative path: "configs/foo.yaml" -> PIPELINE_ROOT/configs/foo.yaml
      - absolute path: passed through.
    """
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    if p.suffix in {".yaml", ".yml"}:
        return (pipeline_root / p).resolve()
    return (pipeline_root / "configs" / f"{name_or_path}.yaml").resolve()


def load_config(
    pipeline_root: Path,
    name_or_path: str,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Load a CPNext config: defaults + YAML overrides.

    Args:
        pipeline_root: Path to the cpepmatch2 pipeline root directory.
        name_or_path: Either a bare name like "mdm2_p53_v1" (resolved against
            PIPELINE_ROOT/configs/), a relative path, or an absolute path.
        run_root: Optional per-run root. Forwarded to ``make_default_config``
            so intermediate path defaults are scoped to this run.

    Returns:
        Merged + validated config dict.

    Note:
        ``proteinhunter.target_msa_path`` is left unresolved here unless the
        YAML sets it explicitly. The notebook is expected to call
        ``resolve_msa_cache_path`` once the target sequence is known
        (post-stage-0) and write it back into the config.
    """
    pipeline_root = Path(pipeline_root)
    yaml_path = _resolve_yaml_path(pipeline_root, name_or_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text()) or {}
    config = make_default_config(pipeline_root, run_root=run_root)

    # Pull the optional run-metadata block out before merging stage configs.
    run_meta = raw.pop("run", None)

    # Resolve complex_pdb relative to PIPELINE_ROOT if it's a relative string.
    if "complex_pdb" in raw:
        cp = Path(raw["complex_pdb"])
        if not cp.is_absolute():
            cp = (pipeline_root / cp).resolve()
        raw["complex_pdb"] = cp

    # Resolve proteinhunter.target_msa_path the same way (relative -> PIPELINE_ROOT).
    ph_raw = raw.get("proteinhunter") or {}
    if "target_msa_path" in ph_raw and ph_raw["target_msa_path"]:
        mp = Path(ph_raw["target_msa_path"])
        if not mp.is_absolute():
            ph_raw["target_msa_path"] = str((pipeline_root / mp).resolve())

    config = _deep_merge(config, raw)

    if run_meta is not None:
        config["run"] = run_meta
    config["_config_yaml_path"] = yaml_path

    validate_config(config)

    name = (run_meta or {}).get("name", yaml_path.stem)
    print(f"Loaded config '{name}' from {yaml_path}")
    print(f"  complex_pdb     = {config['complex_pdb']}")
    print(f"  target chain    = {config['input_target_chain']}")
    print(f"  ligand chain    = {config['input_ligand_chain']}")
    return config
