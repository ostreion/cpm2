#!/usr/bin/env python
"""Render Stage 3.5 alignment PNGs + grid for an in-progress run.

CLI:
    render_alignments.py <pipeline_root> <config_name> <run_root> [--mode all|best]

Reads design metadata from <run_root>/intermediate/3_proteinhunter/<match>/refine_results.json
and writes:
    <run_root>/output/alignments/<match>_design_<n>.png    (one per design, mode=all)
    <run_root>/output/alignments/<match>_best.png          (one per match, mode=best)
    <run_root>/output/alignments/all_grid.png              (rows=matches, cols=designs)

Each PNG superimposes:
  - the experimental input target (template.cif, light-pink cartoon),
  - the Boltz-refolded + ProteinHunter-refined target (cartoon, blue→red by
    per-residue CA displacement vs template),
  - the refined cyclic peptide binder (yellow sticks).

Renders shell out to alignment_render.py inside the `pym` micromamba env
(pymol2 lives there). Mode "all" is the Snakefile default so the user gets
the full grid by default at the end of every run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _render_one(render_script: Path, refined_pdb: Path, template: Path,
                out_png: Path, title: str, subtitle: str, rmsd_cap: float) -> bool:
    cmd = [
        "micromamba", "run", "-n", "pym", "python", str(render_script),
        str(refined_pdb), str(template), "T", "P", str(out_png),
        "--rmsd-cap", str(rmsd_cap),
        "--title", title, "--subtitle", subtitle,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  FAIL {out_png.name}: {res.stderr.strip()[-300:]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_root", type=Path)
    ap.add_argument("config_name")
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--mode", choices=("best", "all"), default="all",
                    help="best = one PNG per match; all = one PNG per design + grid")
    ap.add_argument("--rmsd-cap", type=float, default=3.0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.pipeline_root / "src"))
    from cpm2.config_loader import load_config
    from cpm2.utils.alignment_render import make_grid

    config = load_config(args.pipeline_root, args.config_name, run_root=args.run_root)
    template = args.run_root / "intermediate" / "0_import" / "template.cif"
    if not template.exists():
        print(f"ERROR: template not found at {template}", file=sys.stderr)
        return 1

    ph_root = args.run_root / "intermediate" / "3_proteinhunter"
    out_dir = args.run_root / "output" / "alignments"
    out_dir.mkdir(parents=True, exist_ok=True)

    render_script = args.pipeline_root / "src" / "utils" / "alignment_render.py"

    # Subtitle: announce the template-force mode so the user can interpret the
    # displacement field at a glance.
    force_on = config["proteinhunter"].get("template_force", False)
    force_thr = config["proteinhunter"].get("template_force_threshold", None)
    mode_str = (
        f"forced @ {force_thr:.1f} Å" if force_on and force_thr is not None
        else "soft template"
    )

    by_match: dict[str, list[dict]] = defaultdict(list)
    for rj in sorted(ph_root.glob("*/refine_results.json")):
        try:
            designs = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if not designs:
            continue
        match = rj.parent.name
        for d in designs:
            by_match[match].append(d)

    if not by_match:
        print("WARN: no designs found; writing empty grid sentinel")
        (out_dir / ".empty").touch()
        (out_dir / "all_grid.png").touch()
        return 0

    rendered: dict[str, dict[int, Path]] = defaultdict(dict)
    if args.mode == "best":
        for match, designs in sorted(by_match.items()):
            best = max(designs, key=lambda x: x["iptm"])
            out_png = out_dir / f"{match}_best.png"
            title = f"{match}  -  design {best['design_num']}"
            subtitle = (f"ipTM={best['iptm']:.3f}  pLDDT={best['plddt']:.2f}  "
                        f"{mode_str}")
            print(f"Rendering {match}...")
            if _render_one(render_script, Path(best["output_pdb"]),
                           template, out_png, title, subtitle, args.rmsd_cap):
                rendered[match][best["design_num"]] = out_png
    else:  # all
        for match, designs in sorted(by_match.items()):
            for d in sorted(designs, key=lambda x: x["design_num"]):
                out_png = out_dir / f"{match}_design_{d['design_num']}.png"
                title = f"{match}  -  design {d['design_num']}"
                subtitle = (f"ipTM={d['iptm']:.3f}  pLDDT={d['plddt']:.2f}  "
                            f"{mode_str}")
                print(f"Rendering {match} design {d['design_num']}...")
                if _render_one(render_script, Path(d["output_pdb"]),
                               template, out_png, title, subtitle, args.rmsd_cap):
                    rendered[match][d["design_num"]] = out_png

    # Build the master grid: rows sorted by best ipTM, cols by design_num
    match_best = {m: max(by_match[m], key=lambda x: x["iptm"])["iptm"] for m in rendered}
    sorted_matches = sorted(rendered, key=lambda m: -match_best[m])
    ncols = max((max(v) for v in rendered.values()), default=0) + 1
    rows: list[list[Path | None]] = []
    for m in sorted_matches:
        rows.append([rendered[m].get(c) for c in range(ncols)])

    grid_png = out_dir / "all_grid.png"
    if rows and any(any(r) for r in rows):
        make_grid(rows, grid_png, label_rows=sorted_matches,
                  title=f"{args.config_name}  -  rows: matches by ipTM, cols: designs ({mode_str})")
        print(f"Grid written: {grid_png}")
    else:
        grid_png.touch()
        print("No PNGs to grid; touched empty sentinel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
