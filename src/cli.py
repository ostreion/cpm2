"""Thin convenience CLI for CPM2.

A wrapper, not a reimplementation: `cpm2 run` shells out to the canonical
headless entry point `scripts/run.py` (which drives the Snakemake DAG), and
`cpm2 list-configs` lists the available run configs under configs/. No
pipeline logic lives here.

    cpm2 list-configs
    cpm2 run mdm2_p53_v1 --cores 8
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# src/cli.py -> repo root is two levels up (src/ is the package dir).
REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "scripts" / "run.py"
CONFIGS_DIR = REPO_ROOT / "configs"


def _list_configs() -> int:
    configs = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    if not configs:
        print(f"No configs found under {CONFIGS_DIR}", file=sys.stderr)
        return 1
    print("Available configs (pass the stem to `cpm2 run`):")
    for name in configs:
        print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 1

    command, rest = argv[0], argv[1:]
    if command == "list-configs":
        return _list_configs()
    if command == "run":
        # Forward everything after `run` straight to scripts/run.py, using the
        # same interpreter so we stay in the active cpm2 env.
        return subprocess.run([sys.executable, str(RUN_PY), *rest]).returncode

    print(f"Unknown command: {command!r}. Use 'run' or 'list-configs'.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
