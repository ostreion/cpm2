#!/usr/bin/env python
"""Pre-stage-0 manifest writer (called by Snakemake rule init_run).

The manifest used to be a side-effect of stage0_import. Phase F2 factors it
out so MLflow can log the manifest as an artifact at run start, before any
heavy stages run.

CLI: init_run.py <pipeline_root> <config_name> <run_root>
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    pipeline_root, config_name, run_root = (Path(p) for p in sys.argv[1:4])
    sys.path.insert(0, str(pipeline_root / "src"))
    from cpm2.config_loader import load_config
    from cpm2.utils.manifest import write_run_manifest

    config = load_config(pipeline_root, str(config_name), run_root=run_root)
    out = write_run_manifest(run_root, config, run_root.name)
    print(f"manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
