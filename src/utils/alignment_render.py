#!/usr/bin/env python
"""Render a superposition of the refined target vs template, coloured by per-residue
CA displacement, into a single PNG.

Must run in the `pym` env (pymol2). Call via subprocess from the notebook:

    micromamba run -n pym python src/utils/alignment_render.py \\
        <refined_pdb> <template_cif> <target_chain> <cp_chain> <out_png> \\
        [--rmsd-cap 3.0] [--width 900] [--height 700]

Inside the image: template target as transparent grey cartoon; refined target as
cartoon coloured blue→red by per-residue CA displacement (cap in Å configurable);
refined CP as yellow sticks+cartoon. Structures are superposed via Kabsch on the
target CAs so displacement colours correspond exactly to the rendered overlay.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Legend / compose palette. Must match PyMOL's `blue_red` spectrum used below.
# blue_red is a 2-stop linear RGB interpolation: blue → purple → red.
# Chosen because it has no white (reserved for background), no green (which
# PyMOL's `cbmr` would introduce — `c` is carbon-green, not cyan), and no
# yellow (reserved for the CP binder). Monotonic in hue.
# ---------------------------------------------------------------------------
_PALETTE_STOPS = [
    (0.00, (0, 0, 255)),      # blue
    (1.00, (255, 0, 0)),      # red
]
_TEMPLATE_COLOUR = (255, 182, 193)   # PyMOL "lightpink"
_CP_COLOUR = (255, 255, 0)           # yellow


def _interp_palette(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_PALETTE_STOPS, _PALETTE_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (
                int(c0[0] + (c1[0] - c0[0]) * f),
                int(c0[1] + (c1[1] - c0[1]) * f),
                int(c0[2] + (c1[2] - c0[2]) * f),
            )
    return _PALETTE_STOPS[-1][1]


def _orient_pocket_facing(cmd, target_sel: str, cp_sel: str) -> None:
    """Rotate the camera so the CP sits between the viewer and the target.

    Strategy: put the CP→target axis along the screen's into-the-page direction
    (screen +Z points toward the viewer, so CP-to-target should point along -Z,
    i.e. the target-to-CP axis is screen +Z). The CP ends up nearest the camera
    with the pocket opening facing the viewer.
    """
    t_com = np.asarray(cmd.centerofmass(target_sel), dtype=np.float64)
    c_com = np.asarray(cmd.centerofmass(cp_sel), dtype=np.float64)
    axis = c_com - t_com
    n = float(np.linalg.norm(axis))
    if n < 1e-3:
        cmd.orient(f"({target_sel}) or ({cp_sel})")
        return
    axis /= n

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(axis @ world_up)) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(world_up, axis)
    right /= np.linalg.norm(right)
    up = np.cross(axis, right)

    # PyMOL's view matrix (first 9 floats of cmd.get_view()) is column-major
    # and encodes world→screen rotation. Rows of the row-major R are the screen
    # axes in world coordinates: screen_X = right, screen_Y = up, screen_Z = axis.
    R = np.vstack([right, up, axis])
    view = list(cmd.get_view())
    view[0:9] = [float(R[i, j]) for j in range(3) for i in range(3)]
    cmd.set_view(view)


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _compose_legend(raw_png: Path, out_png: Path, title: str,
                    subtitle: Optional[str], rmsd_cap: float) -> None:
    """Overlay a title bar + colour bar + swatch legend onto the raw render."""
    from PIL import Image, ImageDraw

    raw = Image.open(str(raw_png)).convert("RGB")
    W, H = raw.size
    title_h = 40 if subtitle else 28
    legend_h = 92

    img = Image.new("RGB", (W, H + title_h + legend_h), "white")
    draw = ImageDraw.Draw(img)

    font_title = _load_font(16, bold=True)
    font_sub = _load_font(12)
    font_small = _load_font(11)

    # Title bar
    draw.text((12, 6), title, fill="black", font=font_title)
    if subtitle:
        draw.text((12, 24), subtitle, fill=(80, 80, 80), font=font_sub)

    # Paste structure
    img.paste(raw, (0, title_h))

    # Legend
    y0 = H + title_h + 10
    bar_x0, bar_x1 = 16, 300
    bar_y0, bar_y1 = y0 + 22, y0 + 38
    for x in range(bar_x0, bar_x1):
        t = (x - bar_x0) / max(bar_x1 - bar_x0 - 1, 1)
        draw.line([(x, bar_y0), (x, bar_y1)], fill=_interp_palette(t))
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline="black")

    draw.text((bar_x0, bar_y0 - 14),
              "Refined target — per-residue CA displacement vs template",
              fill="black", font=font_small)
    draw.text((bar_x0, bar_y1 + 4), "0 Å", fill="black", font=font_small)
    cap_lbl = f"≥ {rmsd_cap:.1f} Å"
    tw = int(draw.textlength(cap_lbl, font=font_small))
    draw.text((bar_x1 - tw, bar_y1 + 4), cap_lbl, fill="black", font=font_small)

    sw_x = 340
    sw_size = 16
    items = [
        (_TEMPLATE_COLOUR, "Template — input target (reference)"),
        (_CP_COLOUR,       "Refined CP binder (sticks)"),
    ]
    for i, (colour, label) in enumerate(items):
        yy = y0 + 16 + i * 24
        draw.rectangle([sw_x, yy, sw_x + sw_size, yy + sw_size],
                       fill=colour, outline="black")
        draw.text((sw_x + sw_size + 8, yy + 1), label,
                  fill="black", font=font_small)

    img.save(str(out_png))


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that src @ R.T + t is the least-squares alignment onto dst."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - src_c @ R.T
    return R, t


def render_alignment(
    refined_pdb: str | Path,
    template_cif: str | Path,
    target_chain: str,
    cp_chain: str,
    out_png: str | Path,
    rmsd_cap: float = 3.0,
    width: int = 900,
    height: int = 700,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Path:
    import pymol2

    refined_pdb = str(refined_pdb)
    template_cif = str(template_cif)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Render raw 3D scene to a sibling "_raw" file; compose final with legend on top.
    raw_png = out_png.with_name(out_png.stem + "_raw.png")

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(template_cif, "template")
        cmd.load(refined_pdb, "refined")
        cmd.remove("resn HOH")
        cmd.remove("hydrogen")

        # Extract CA coords in atom order for both target chains.
        tmpl_xyz, tmpl_resi = [], []
        cmd.iterate_state(
            1,
            f"template and chain {target_chain} and name CA",
            "xyz.append((x, y, z)); resi_list.append(resi)",
            space={"xyz": tmpl_xyz, "resi_list": tmpl_resi},
        )
        ref_xyz, ref_resi = [], []
        cmd.iterate_state(
            1,
            f"refined and chain {target_chain} and name CA",
            "xyz.append((x, y, z)); resi_list.append(resi)",
            space={"xyz": ref_xyz, "resi_list": ref_resi},
        )

        if not tmpl_xyz or not ref_xyz:
            raise RuntimeError(
                f"No CA atoms found on chain {target_chain} "
                f"(template={len(tmpl_xyz)}, refined={len(ref_xyz)})"
            )

        # Pair CAs by residue number, not by atom-list index. The template may
        # contain gaps or extra termini that the refined target lacks, so
        # positional pairing silently misaligns past the first gap.
        tmpl_by_resi = dict(zip(tmpl_resi, tmpl_xyz))
        ref_by_resi = dict(zip(ref_resi, ref_xyz))
        def _resi_sort_key(r: str) -> int:
            try:
                return int(r)
            except ValueError:
                return 10**9
        common = sorted(
            (r for r in ref_by_resi if r in tmpl_by_resi),
            key=_resi_sort_key,
        )
        if not common:
            raise RuntimeError(
                f"No common residue numbers on chain {target_chain} "
                f"(template={len(tmpl_resi)}, refined={len(ref_resi)})"
            )
        T = np.asarray([tmpl_by_resi[r] for r in common], dtype=np.float64)
        R_coords = np.asarray([ref_by_resi[r] for r in common], dtype=np.float64)
        print(f"  alignment: {len(common)}/{len(ref_resi)} refined CAs matched "
              f"to template (template has {len(tmpl_resi)})")

        # Kabsch alignment of refined onto template, then per-residue CA displacement.
        R_mat, t_vec = _kabsch(R_coords, T)
        aligned = R_coords @ R_mat.T + t_vec
        disp = np.linalg.norm(aligned - T, axis=1)
        disp_by_resi = dict(zip(common, disp.tolist()))

        # Apply the Kabsch transform to the whole refined object so the rendered
        # overlay exactly matches the displacement colours we just computed.
        M = np.eye(4)
        M[:3, :3] = R_mat
        M[:3, 3] = t_vec
        cmd.transform_selection("refined", M.flatten().tolist())

        # Write per-residue displacement into B-factor column for spectrum colouring.
        cmd.alter(f"refined and chain {target_chain}", "b = 0.0")
        cmd.alter(
            f"refined and chain {target_chain}",
            "b = disp.get(resi, 0.0)",
            space={"disp": disp_by_resi},
        )
        cmd.rebuild()

        # Styling.
        cmd.hide("everything")
        cmd.show("cartoon", "template")
        cmd.show("cartoon", f"refined and chain {target_chain}")
        cmd.show("cartoon", f"refined and chain {cp_chain}")
        cmd.show("sticks", f"refined and chain {cp_chain}")

        # Template: light pink so it can never be confused with the gradient
        # (which has no white, no pink). Slight transparency keeps it subordinate.
        cmd.color("lightpink", "template")
        cmd.set("cartoon_transparency", 0.3, "template")
        # blue_red = blue → purple → red. No white (background), no green
        # (PyMOL's cbmr would introduce it — `c` is carbon-green, not cyan),
        # no yellow (reserved for the CP binder). Monotonic hue gradient.
        cmd.spectrum(
            "b",
            "blue_red",
            f"refined and chain {target_chain}",
            minimum=0.0,
            maximum=rmsd_cap,
        )
        cmd.color("yellow", f"refined and chain {cp_chain}")

        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.set("antialias", 2)
        cmd.set("cartoon_fancy_helices", 1)
        cmd.set("ray_shadows", 0)

        _orient_pocket_facing(
            cmd,
            target_sel=f"refined and chain {target_chain}",
            cp_sel=f"refined and chain {cp_chain}",
        )
        cmd.zoom(
            f"refined and (chain {target_chain} or chain {cp_chain})",
            buffer=5.0,
        )

        cmd.ray(int(width), int(height))
        cmd.png(str(raw_png), dpi=150)

    if title is None:
        title = out_png.stem
    _compose_legend(raw_png, out_png, title=title, subtitle=subtitle, rmsd_cap=rmsd_cap)
    try:
        raw_png.unlink()
    except OSError:
        pass

    print(f"rendered: {out_png}  (max displacement {disp.max():.2f} Å, mean {disp.mean():.2f} Å)")
    return out_png


def make_grid(
    rows: list[list[Optional[str | Path]]],
    out_png: str | Path,
    label_rows: Optional[list[str]] = None,
    tile_size: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
) -> Path:
    """Compose a 2D grid of alignment PNGs into a single image.

    `rows` is a list of rows, each row a list of image paths (or None for a
    blank cell). Rows are padded to the same length. Each tile is resized to
    `tile_size` (default: size of the first non-None tile).
    """
    from PIL import Image, ImageDraw

    flat = [p for row in rows for p in row if p]
    if not flat:
        raise ValueError("make_grid called with no images")

    if tile_size is None:
        with Image.open(str(flat[0])) as im:
            tile_size = im.size  # (w, h)
    tw, th = tile_size

    ncols = max(len(r) for r in rows)
    nrows = len(rows)

    pad = 6
    label_w = 110 if label_rows else 0
    title_h = 32 if title else 0
    W = label_w + ncols * (tw + pad) - pad
    H = title_h + nrows * (th + pad) - pad

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    if title:
        draw.text((10, 6), title, fill="black", font=_load_font(16, bold=True))

    font_lbl = _load_font(12, bold=True)
    for r, row in enumerate(rows):
        y = title_h + r * (th + pad)
        if label_rows and r < len(label_rows):
            draw.text((4, y + th // 2 - 8), label_rows[r], fill="black", font=font_lbl)
        for c, path in enumerate(row):
            if not path:
                continue
            x = label_w + c * (tw + pad)
            with Image.open(str(path)) as tile:
                if tile.size != (tw, th):
                    tile = tile.resize((tw, th))
                img.paste(tile, (x, y))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_png))
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refined_pdb")
    ap.add_argument("template_cif")
    ap.add_argument("target_chain")
    ap.add_argument("cp_chain")
    ap.add_argument("out_png")
    ap.add_argument("--rmsd-cap", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()
    render_alignment(
        args.refined_pdb,
        args.template_cif,
        args.target_chain,
        args.cp_chain,
        args.out_png,
        rmsd_cap=args.rmsd_cap,
        width=args.width,
        height=args.height,
        title=args.title,
        subtitle=args.subtitle,
    )


if __name__ == "__main__":
    main()
