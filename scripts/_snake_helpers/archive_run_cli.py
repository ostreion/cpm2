#!/usr/bin/env python
"""Snakemake wrapper around archive_run for `rule archive`.

CLI: archive_run_cli.py <pipeline_root> <config_name> <run_root> <sentinel>
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    pipeline_root = Path(sys.argv[1])
    config_name = sys.argv[2]
    run_root = Path(sys.argv[3])
    sentinel = Path(sys.argv[4])
    sys.path.insert(0, str(pipeline_root / "src"))

    from cpm2.config_loader import load_config
    from cpm2.utils.archiver import archive_run

    config = load_config(pipeline_root, config_name, run_root=run_root)
    archive_run(config, clean=False, run_root=run_root, pipeline_root=pipeline_root)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
