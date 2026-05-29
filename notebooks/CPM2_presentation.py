import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CPM2: a critical examination of pipeline quality

    One unified, interactive page. We open a single run, examine its
    designs on the triage plane (overlaying their structures and, where
    we have MD on disk, their trajectories), then ask the only question
    that matters: **do the scores survive contact with physics?**

    The page is organised in three movements.

    * **A. Does the pipeline produce good-looking designs?** Topology
      fidelity, score and RMSD distributions, the triage plane with a
      live structure overlay, the refinement-convergence traces, and the
      ranked MM-GBSA shortlist.
    * **B. Do those scores survive contact with physics?** Designed
      binders vs their natural partners, negative-control specificity,
      the ipTM-vs-MM-GBSA agreement, and the MD trajectory stability of
      the per-target winners.
    * **C. What we cannot yet claim.** Pending experiments and an honest
      epistemics ledger, then a verdict.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    try:
        _start = Path(__file__).resolve().parent
    except NameError:
        _start = Path.cwd()

    repo = _start
    while repo != repo.parent and not (repo / "pyproject.toml").is_file():
        repo = repo.parent
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))

    import analysis as A
    import critique as C
    import quality as Q

    return A, C, Q, repo


@app.cell(hide_code=True)
def _():
    import altair as alt
    import pandas as pd

    return alt, pd


@app.cell(hide_code=True)
def _(A, mo):
    drilldown_runs = A.list_runs()
    _ids = list(drilldown_runs["run_id"]) if not drilldown_runs.empty else []
    run_picker = mo.ui.dropdown(
        options=_ids, value=_ids[0] if _ids else None,
        label="Run", searchable=True,
    )
    return drilldown_runs, run_picker


@app.cell(hide_code=True)
def _(A, run_picker):
    rd = A.load_run(run_picker.value) if run_picker.value else None
    return (rd,)


@app.cell(hide_code=True)
def _(mo, rd):
    # CLEAN run-info pane (replaces the dev provenance MISSING/DIRTY wall).
    # Human-readable: run id, target, n designs / n matches, ipTM range,
    # topology counts, formatted timestamp. A small red flag appears ONLY
    # if the run is git-dirty; otherwise no provenance noise.
    from datetime import datetime as _dt
    from pathlib import Path as _P

    if rd is None:
        run_info_pane = mo.md("")
    else:
        m = rd.manifest or {}
        g = m.get("git", {}) or {}
        inp = m.get("input", {}) or {}
        d = rd.designs

        # Human-readable timestamp.
        _ts_raw = m.get("timestamp_iso") or ""
        _ts_human = _ts_raw
        if _ts_raw:
            try:
                _parsed = _dt.fromisoformat(_ts_raw.replace("Z", "+00:00"))
                _ts_human = _parsed.strftime("%d %b %Y, %H:%M UTC")
            except ValueError:
                _ts_human = _ts_raw

        # Target from the input PDB name (e.g. 5m36.pdb).
        _pdb_name = (_P(inp.get("complex_pdb", "")).stem
                     if inp.get("complex_pdb") else "?")
        _target_chain = inp.get("target_chain") or "?"
        _ligand_chain = inp.get("ligand_chain") or "?"

        # ipTM range (min - median - max).
        _ip_min = float(d["iptm"].min())
        _ip_med = float(d["iptm"].median())
        _ip_max = float(d["iptm"].max())

        # Topology counts -> compact "cyclic x12, ..." string.
        _topo = d["topology"].value_counts()
        _topo_str = ", ".join(f"{k} x{v}" for k, v in _topo.items())

        # Topology-bug audit count.
        _status = d["topology_status"].value_counts().to_dict()
        _n_bug = int(_status.get("bug", 0))

        _dirty = bool(g.get("dirty"))
        _sha = g.get("sha_short") or (g.get("sha") or "")[:8] or "?"

        # Lean metric grid.
        def _stat(label, value):
            return (f"<div style='min-width:130px'>"
                    f"<div style='font-size:11px;color:#888;"
                    f"text-transform:uppercase;letter-spacing:.04em'>"
                    f"{label}</div>"
                    f"<div style='font-size:18px;font-weight:600'>"
                    f"{value}</div></div>")

        _cards = "".join([
            _stat("target", f"{_pdb_name}"),
            _stat("chains", f"{_target_chain} / {_ligand_chain}"),
            _stat("designs", f"{rd.n_designs}"),
            _stat("matches", f"{rd.n_matches}"),
            _stat("ipTM (min / med / max)",
                  f"{_ip_min:.2f} / {_ip_med:.2f} / {_ip_max:.2f}"),
        ])

        _md = (
            f"### `{rd.run_id}`\n\n"
            f"<div style='display:flex;flex-wrap:wrap;gap:22px;"
            f"margin:6px 0 10px 0'>{_cards}</div>\n\n"
            f"**Topology:** {_topo_str}.  "
            f"**Run finished:** {_ts_human}.  "
            f"**Code:** `{_sha}`"
            + (f" + **{_n_bug} topology-bug** design(s) (built unlike the "
               "cPEPmatch DB declaration -- see A.2 below)."
               if _n_bug else ".")
        )

        if _dirty:
            run_info_pane = mo.callout(
                mo.md(_md + "\n\n**Reproducibility flag:** this run is "
                      "**git-dirty** -- the recorded SHA does not fully "
                      "describe the code that produced it. Treat numbers "
                      "as untrusted."),
                kind="danger")
        else:
            run_info_pane = mo.callout(mo.md(_md), kind="neutral")
    return (run_info_pane,)


@app.cell(hide_code=True)
def _(mo, rd):
    # Triage-plane controls (separate cell -- marimo rule).
    if rd is None:
        topo_filter = mo.ui.multiselect(options=[], value=[],
                                        label="topologies shown")
        x_axis = mo.ui.dropdown(options=["iptm"], value="iptm", label="x")
        y_axis = mo.ui.dropdown(options=["iptm"], value="iptm", label="y")
    else:
        import analysis as _Aconst
        _present = [t for t in _Aconst.TOPOLOGY_ORDER
                    if t in set(rd.designs["topology"])]
        _axes = ["iptm", "plddt", "iplddt",
                 "peptide_drift_ca_rmsd", "cpepmatch_fit_rmsd",
                 "target_drift_ca_rmsd", "n_res"]
        topo_filter = mo.ui.multiselect(options=_present, value=_present,
                                        label="topologies shown")
        x_axis = mo.ui.dropdown(options=_axes,
                                value="peptide_drift_ca_rmsd",
                                label="x axis")
        y_axis = mo.ui.dropdown(options=_axes, value="iptm", label="y axis")
    mmgbsa_only_filter = mo.ui.switch(
        value=False, label="mmgbsa only")
    return mmgbsa_only_filter, topo_filter, x_axis, y_axis


@app.cell(hide_code=True)
def _(mdb, rd):
    # Set of design names whose AL_decomp.csv exists in the active MD bundle.
    # Empty when no bundle is loaded; cheap (stat() only, no parsing).
    # marimo mangles leading-underscore names per-cell; `from quality import
    # _name` doesn't get mangled consistently, so we go through
    # `import quality` + `getattr` instead.
    from pathlib import Path as _Path
    import quality as _quality
    _names: set[str] = set()
    if rd is not None and mdb is not None and not mdb.jobs.empty \
            and "job" in mdb.jobs.columns:
        _decomp_rel = getattr(
            _quality, "_DECOMP_REL",
            _Path("mmgbsa") / "AL_out" / "AL_decomp.csv")
        _job_dir_fn = getattr(_quality, "_md_job_dir_for_design", None)
        if _job_dir_fn is not None:
            for _n in rd.designs["name"].astype(str):
                _jd = _job_dir_fn(mdb, _n)
                if _jd is not None and (_jd / _decomp_rel).is_file():
                    _names.add(_n)
    mmgbsa_design_names = _names
    return (mmgbsa_design_names,)


@app.cell(hide_code=True)
def _(
    alt,
    mdb,
    mmgbsa_design_names,
    mmgbsa_only_filter,
    mo,
    rd,
    topo_filter,
    x_axis,
    y_axis,
):
    if rd is None:
        plane = mo.md("")
    else:
        _df = rd.designs[rd.designs["topology"].isin(topo_filter.value)].copy()
        if mmgbsa_only_filter.value:
            _df = _df[_df["name"].astype(str).isin(mmgbsa_design_names)].copy()
        if _df.empty:
            plane = mo.md("**No designs match the selected topologies.**")
        else:
            # Tag each design with whether its MD run has free_run/run.nc +
            # system_wb.prmtop on disk in the active bundle. Stars in the
            # triage plane = "you can play this trajectory in the structure
            # viewer below (toggle 'play MD trajectory')".
            _md_avail: set[str] = set()
            if mdb is not None and not mdb.jobs.empty:
                _j = mdb.jobs.copy()
                _j["_design"] = (_j["job"].astype(str).str.split("__")
                                 .str[-1])
                _have = _j[_j["free_run_nc"].notna()
                           & _j["prmtop"].notna()]
                _md_avail = set(_have["_design"].astype(str))
            _df["md_available"] = (_df["name"].astype(str)
                                   .isin(_md_avail))
            _x, _y = x_axis.value, y_axis.value
            _forced = _x == "target_drift_ca_rmsd" and rd.template_force
            _xl = _x + ("  [forcing-bounded]" if _forced else "")
            _brush = alt.selection_interval(name="brush")
            # Click selection: pick one or more designs for the structure
            # overlay viewer to the right of the plane. Shift-click extends.
            _pick = alt.selection_point(
                name="pick", fields=["name"], on="click",
                toggle="event.shiftKey", empty="none",
            )
            # 5-pointed star, unit-radius, centered at origin. Vega-lite
            # accepts SVG path strings as the `shape` value and scales them
            # by the `size` encoding (square pixels).
            _STAR = ("M0,-1L0.225,-0.309L0.951,-0.309L0.363,0.118"
                     "L0.588,0.809L0,0.382L-0.588,0.809L-0.363,0.118"
                     "L-0.951,-0.309L-0.225,-0.309Z")
            _c = (
                alt.Chart(_df)
                .add_params(_brush, _pick)
                # mark_point + shape encoding: mark_circle ignores `shape`.
                .mark_point(filled=True, stroke="white", strokeWidth=0.5)
                .encode(
                    # Auto-fit x to data extent with a small buffer, same
                    # as the y-axis treatment -- avoids forcing zero into
                    # the domain on metrics like peptide_drift_ca_rmsd that
                    # never hit zero in practice.
                    x=alt.X(f"{_x}:Q", title=_xl,
                            scale=alt.Scale(zero=False, nice=True,
                                            padding=10)),
                    # zero=False + nice=True auto-fits the y-domain to the
                    # data with a small buffer (avoids the huge empty area
                    # below an ipTM cluster sitting around 0.7-0.9).
                    y=alt.Y(f"{_y}:Q", title=_y,
                            scale=alt.Scale(zero=False, nice=True,
                                            padding=10)),
                    color=alt.Color("topology:N", title="topology"),
                    size=alt.Size("plddt:Q", title="pLDDT",
                                  scale=alt.Scale(range=[80, 600])),
                    shape=alt.Shape(
                        "md_available:N",
                        title="MD trajectory",
                        scale=alt.Scale(domain=[False, True],
                                        range=["circle", _STAR]),
                        legend=alt.Legend(
                            labelExpr=("datum.label == 'true' "
                                       "? 'MD on disk' : 'no MD'")),
                    ),
                    # Dim points outside the brush and unpicked points.
                    opacity=alt.condition(
                        _brush,
                        alt.condition(_pick, alt.value(1.0), alt.value(0.55)),
                        alt.value(0.18),
                    ),
                    tooltip=["name", "topology", "topology_status", "iptm",
                            "plddt", "iplddt", "cpepmatch_fit_rmsd",
                            "peptide_drift_ca_rmsd",
                            "target_drift_ca_rmsd", "n_res",
                            "md_available", "optimized_sequence"],
                )
                .properties(width="container", height=540,
                            title=f"{rd.run_id} -- {_y} vs {_x}  "
                                  "(brush to filter table; click to overlay; "
                                  "★ = MD trajectory on disk)")
                .interactive()
            )
            plane = mo.ui.altair_chart(_c)
    return (plane,)


@app.cell(hide_code=True)
def _(plane, rd):
    # Designs picked in the triage plane. `plane.selections` is the raw
    # vega selection-state dict, not a DataFrame: for our point selection
    # `pick = alt.selection_point(fields=["name"])`, it surfaces as
    # `{"pick": {"name": ["match...", ...], "vlPoint": ...}}` (empty when
    # nothing is clicked, since the selection is declared with empty="none").
    # When nothing is clicked, fall back to top-1 ipTM so the structure
    # viewer always has something to show. Cap at 6.
    _MAX_OVERLAY = 6
    _selected: list[str] = []
    _sels = getattr(plane, "selections", None)
    if isinstance(_sels, dict):
        _pick = _sels.get("pick")
        if isinstance(_pick, dict):
            _raw = _pick.get("name")
            if isinstance(_raw, (list, tuple)):
                _selected = [str(n) for n in _raw if n is not None][:_MAX_OVERLAY]
    if not _selected and rd is not None and not rd.designs.empty:
        _top = rd.designs.sort_values("iptm", ascending=False).head(1)
        _selected = _top["name"].astype(str).tolist()
    selected_design_names = _selected
    return (selected_design_names,)


@app.cell(hide_code=True)
def _(mo):
    struct_color_mode = mo.ui.radio(
        # "rmsf" and "contact-frequency" read precomputed parquet from
        # <jobdir>/analysis/per_residue.parquet (see scripts/backfill_md_analyses.py).
        # mmgbsa-hotspots uses AL_decomp.csv. chain-id is the no-extra-data default.
        options=["chain-id", "mmgbsa-hotspots", "rmsf",
                 "contact-frequency"],
        value="chain-id",
        label="overlay color mode",
    )
    # The binding partner is always included in the structures overlay
    # as a default-invisible row in the Structures legend (toggle visibility
    # there). Slider sets its render opacity when shown -- 0.3 makes the
    # reference cartoon translucent so designs read clearly on top.
    partner_opacity = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="partner opacity", show_value=True)
    # When on AND a single design is selected AND that design has an MD
    # job on disk (free_run/run.nc + system_wb.prmtop), the drill-down
    # structure viewer swaps from static overlay to a trajectory player
    # of that design's MD run. Otherwise it stays as the static overlay.
    show_md_trajectory = mo.ui.switch(
        value=False, label="play MD trajectory (single design only)")
    md_traj_drilldown_stride = mo.ui.slider(
        start=1, stop=200, step=1, value=1,
        label="MD stride", show_value=True)
    return (
        md_traj_drilldown_stride,
        partner_opacity,
        show_md_trajectory,
        struct_color_mode,
    )


@app.cell(hide_code=True)
def _(A, Q, mdb, pd, rd, selected_design_names, struct_color_mode):
    # Build residue_values payload for the Mol* widget when the user picks
    # one of the per-residue color modes (mmgbsa-hotspots / rmsf /
    # contact-frequency). Pulls from MD bundle artifacts produced by the
    # MMGBSA / md_analyses pipelines (cross-tab dep on `mdb`).
    from pathlib import Path as _Path
    residue_values_payload = None
    mmgbsa_warning = None
    _mode = struct_color_mode.value
    if _mode in ("mmgbsa-hotspots", "rmsf", "contact-frequency"):
        if mdb is None:
            mmgbsa_warning = ("No MD bundle loaded -- pick a bundle in the "
                              "MD section below, then return here.")
        elif not selected_design_names:
            mmgbsa_warning = "No designs selected."
        else:
            # The MD-side data is keyed by chain "A" (target) / "B" (peptide)
            # because amber prmtop has no native chain IDs and md_analyses
            # / mmgbsa_decomp_for_designs use the connectivity-split convention.
            # Mol* coloring queries on the design PDB's chain ids, which may
            # differ (e.g. T/P on bclxl). Detect target / ligand chain from
            # the first picked design and remap A->target, B->ligand below.
            _target_chain, _ligand_chain = "A", "B"
            if rd is not None:
                _byname = rd.designs.set_index("name")
                for _n in selected_design_names:
                    if _n not in _byname.index:
                        continue
                    _p = _Path(str(_byname.loc[_n, "_output_pdb"]))
                    if not _p.is_file():
                        continue
                    _ch = A._pdb_ca_by_chain(_p)
                    if not _ch:
                        continue
                    _ordered = sorted(_ch.items(), key=lambda kv: -len(kv[1]))
                    _target_chain = _ordered[0][0]
                    if len(_ordered) > 1:
                        _ligand_chain = _ordered[1][0]
                    break

            _chain_remap = {"A": _target_chain, "B": _ligand_chain}

            def _aggregate_parquet(_col: str):
                """Mean per (chain,resnum) across selected designs' parquet.

                The parquet uses global prmtop-1-indexed resnums (chain B
                continues after chain A: e.g. A 1..175, B 176..193). The
                design PDB Mol* loads uses per-chain 1-indexed resSeq
                (chain T 1..175, chain P 1..18). Remap by subtracting the
                chain's min resnum so the payload's chain-B residue 176
                becomes resnum 1 on chain P -- otherwise the overpaint
                query matches zero atoms on the peptide.
                """
                _agg2: dict[tuple[str, int], list[float]] = {}
                _hits, _misses = [], []
                _jb = mdb.jobs.copy()
                _jb["_design"] = (_jb["job"].astype(str).str.split("__")
                                  .str[-1])
                for _design in selected_design_names:
                    _row = _jb[_jb["_design"] == _design]
                    if _row.empty:
                        _misses.append(_design); continue
                    _jd = _Path(str(_row.iloc[0]["job_dir"]))
                    _pq = _jd / "analysis" / "per_residue.parquet"
                    if not _pq.is_file():
                        _misses.append(_design); continue
                    _df = pd.read_parquet(_pq)
                    # Per-chain offset: parquet global index -> per-chain
                    # 1-indexed PDB resSeq.
                    _offsets = (_df.groupby("chain")["resnum"].min()
                                - 1).to_dict()
                    for _r in _df.itertuples():
                        _ch = _chain_remap.get(_r.chain, _r.chain)
                        _rn = int(_r.resnum) - int(_offsets[_r.chain])
                        _agg2.setdefault((_ch, _rn), []).append(
                            float(getattr(_r, _col)))
                    _hits.append(_design)
                return _hits, _misses, _agg2

            if _mode == "mmgbsa-hotspots":
                _decomps = Q.mmgbsa_decomp_for_designs(
                    mdb, selected_design_names,
                    target_chain=_target_chain, ligand_chain=_ligand_chain)
                if not _decomps:
                    mmgbsa_warning = (
                        "None of the selected designs have an MMGBSA "
                        "`AL_decomp.csv` in the active bundle. Pick a "
                        "design whose MD job is done.")
                    _agg = {}
                else:
                    _agg = {}
                    for _d in _decomps.values():
                        for k, v in _d.items():
                            _agg.setdefault(k, []).append(v)
            elif _mode in ("rmsf", "contact-frequency"):
                _col = "rmsf_A" if _mode == "rmsf" else "contact_freq"
                _hits, _misses, _agg = _aggregate_parquet(_col)
                if not _hits:
                    mmgbsa_warning = (
                        f"None of the selected designs have "
                        "`analysis/per_residue.parquet`. Run "
                        "`scripts/backfill_md_analyses.py <bundle>` to "
                        "compute MD analyses for finished jobs.")
            else:
                _agg = {}

            if _agg:
                _vals = [[c, n, sum(vs) / len(vs)]
                         for (c, n), vs in _agg.items()]
                _nums = [v[2] for v in _vals]
                _data_min = float(min(_nums))
                _data_max = float(max(_nums))
                if _mode == "mmgbsa-hotspots":
                    # Diverging palette (negative = favourable). 75th
                    # percentile of |dG| clamp so mid-signal residues
                    # get colored and hot-spots clip to full red/blue.
                    _abs_sorted = sorted(abs(v) for v in _nums)
                    _qidx = max(0, int(round(0.75 * (len(_abs_sorted) - 1))))
                    _abs_cap = _abs_sorted[_qidx]
                    if _abs_cap <= 0:
                        _abs_cap = max(abs(v) for v in _nums) or 1.0
                    _vmin, _vmax = -float(_abs_cap), float(_abs_cap)
                    _ranked = sorted(_vals, key=lambda v: -abs(v[2]))
                    _label = ("MM-GBSA dG per residue (kcal/mol)  "
                              f"[{len(_agg) and len(selected_design_names)} "
                              "design(s) avg]")
                elif _mode == "rmsf":
                    # Non-negative -- sequential viridis. vmin=0,
                    # vmax = 95th-percentile so a few floppy residues
                    # don't desaturate the rest.
                    _sorted = sorted(_nums)
                    _qidx = max(0, int(round(0.95 * (len(_sorted) - 1))))
                    _vmin = 0.0
                    _vmax = float(_sorted[_qidx]) or 1.0
                    _ranked = sorted(_vals, key=lambda v: -v[2])
                    _label = ("Per-residue RMSF (Angstrom)  "
                              f"[{len(_hits)} design(s) avg]")
                else:  # contact-frequency
                    # Fraction in [0,1] -- sequential viridis 0->1.
                    _vmin, _vmax = 0.0, 1.0
                    _ranked = sorted(_vals, key=lambda v: -v[2])
                    _label = ("Per-residue contact frequency  "
                              f"[{len(_hits)} design(s) avg]")
                _hotspots = [
                    {"chain": str(c), "resnum": int(n)}
                    for c, n, _v in _ranked[:5]
                ]
                residue_values_payload = {
                    "values": _vals,
                    "vmin": float(_vmin),
                    "vmax": float(_vmax),
                    "data_vmin": _data_min,
                    "data_vmax": _data_max,
                    "hotspot_residues": _hotspots,
                    "label": _label,
                }
                _chains_seen = sorted({v[0] for v in _vals})
                _per_chain = {
                    c: sum(1 for v in _vals if v[0] == c)
                    for c in _chains_seen
                }
                print(f"[{_mode}] target_chain={_target_chain!r}"
                      f" ligand_chain={_ligand_chain!r}")
                print(f"[{_mode}] n_residues={len(_vals)} "
                      f"chains={_per_chain} "
                      f"vmin={_vmin:.3f} vmax={_vmax:.3f}")
    return mmgbsa_warning, residue_values_payload


@app.cell(hide_code=True)
def _(alt, mdb, mo, pd, selected_design_names):
    # MD traces line chart: peptide_rmsd_A + com_distance_A over time,
    # read from the per-job analysis/traces.parquet produced by
    # scripts/backfill_md_analyses.py. Active only when EXACTLY ONE
    # design is selected and has an MD job with the parquet on disk.
    from pathlib import Path as _Path
    md_traces_panel = None
    if mdb is None or mdb.jobs.empty:
        md_traces_panel = mo.md("")
    elif not selected_design_names:
        md_traces_panel = mo.md("")
    elif len(selected_design_names) != 1:
        md_traces_panel = mo.md(
            "*MD traces: select exactly one design to plot its "
            "peptide-drift + center-of-mass-distance trace.*")
    else:
        _design = selected_design_names[0]
        _jb = mdb.jobs.copy()
        _jb["_design"] = _jb["job"].astype(str).str.split("__").str[-1]
        _row = _jb[_jb["_design"] == _design]
        if _row.empty:
            md_traces_panel = mo.md(
                f"*No MD job for `{_design}` in bundle `{mdb.name}`.*")
        else:
            _jd = _Path(str(_row.iloc[0]["job_dir"]))
            _pq = _jd / "analysis" / "traces.parquet"
            if not _pq.is_file():
                md_traces_panel = mo.callout(
                    mo.md(f"*No `{_pq.relative_to(_jd.parent.parent)}` "
                          "for this job. Run "
                          "`scripts/backfill_md_analyses.py "
                          f"{_jd.parent.parent}` to compute traces.*"),
                    kind="info")
            else:
                _t = pd.read_parquet(_pq)
                # Long form so Altair can color-by-metric.
                _long = _t.melt(
                    id_vars=["frame", "time_ns"],
                    value_vars=["peptide_rmsd_A", "com_distance_A"],
                    var_name="metric", value_name="value")
                _long["metric"] = _long["metric"].map({
                    "peptide_rmsd_A": "peptide drift (RMSD vs frame 0, A)",
                    "com_distance_A": "peptide<->target CoM distance (A)",
                })
                _c = (
                    alt.Chart(_long)
                    .mark_line(strokeWidth=1.5)
                    .encode(
                        x=alt.X("time_ns:Q", title="time (ns)"),
                        y=alt.Y("value:Q", title="A",
                                scale=alt.Scale(zero=False, nice=True,
                                                padding=4)),
                        color=alt.Color("metric:N", title=None,
                                        legend=alt.Legend(orient="top")),
                        tooltip=["frame", "time_ns", "metric", "value"],
                    )
                    .properties(width="container", height=180,
                                title=f"MD traces -- {_design}")
                )
                md_traces_panel = mo.ui.altair_chart(_c)
    return (md_traces_panel,)


@app.cell(hide_code=True)
def _(
    A,
    md_traj_drilldown_stride,
    mdb,
    mmgbsa_warning,
    mo,
    partner_opacity,
    rd,
    residue_values_payload,
    selected_design_names,
    show_md_trajectory,
    struct_color_mode,
):
    from pathlib import Path as _Path

    # --- MD-trajectory short-circuit -----------------------------------
    # If the "play MD trajectory" switch is on AND exactly one design is
    # selected AND it has an MD job with .nc + .prmtop on disk in the
    # active bundle, swap the whole panel for a trajectory player of
    # that design's run. Misses fall through to the static-overlay path
    # with an explanatory callout.
    _md_traj_panel = None
    if show_md_trajectory.value:
        if rd is None or not selected_design_names:
            _md_traj_panel = mo.callout(
                mo.md("*Select a single design on the triage plane to "
                      "play its MD trajectory.*"), kind="info")
        elif len(selected_design_names) > 1:
            _md_traj_panel = mo.callout(
                mo.md(f"**MD trajectory requires a single design**; "
                      f"{len(selected_design_names)} are selected."),
                kind="warn")
        elif mdb is None or mdb.jobs.empty:
            _md_traj_panel = mo.callout(
                mo.md("**No MD bundle loaded.** Pick a bundle in the MD "
                      "section below, then return here."),
                kind="warn")
        else:
            _design = selected_design_names[0]
            _jobs = mdb.jobs
            _match = _jobs[_jobs["job"].astype(str).str.split("__")
                           .str[-1] == _design]
            if _match.empty:
                _md_traj_panel = mo.callout(
                    mo.md(f"**No MD job for `{_design}`** in bundle "
                          f"`{mdb.name}`."), kind="warn")
            else:
                _row = _match.iloc[0]
                _nc = _row.get("free_run_nc")
                _prm = _row.get("prmtop")
                if not _nc or not _prm:
                    _md_traj_panel = mo.callout(
                        mo.md(f"**MD job `{_row['job']}` has no "
                              f"`free_run/run.nc` + `system_wb.prmtop` "
                              "on disk yet.**"), kind="warn")
                else:
                    try:
                        import molstar_marimo as _mm
                        # amber_to_bytes splits chains by molecular
                        # connectivity in descending atom count, so
                        # chain A == target, chain B == cyclic peptide
                        # for CPM2 complexes. PyMOL-style default:
                        # target as surface, peptide as ball-and-stick.
                        _traj = _mm.MolstarViewer.from_amber(
                            nc_path=_nc, prmtop_path=_prm,
                            stride=md_traj_drilldown_stride.value,
                            strip="!:WAT,Na+,Cl-",
                            height=540,
                            representation="cartoon",
                            color_scheme="chain-id",
                            show_legend=True,
                            autoplay=True, fps=12, loop=True,
                            chain_representations={
                                "A": "spacefill",
                                "B": "ball-and-stick",
                            },
                        )
                        _md_traj_panel = mo.vstack([
                            mo.md(f"**MD trajectory:** `{_row['job']}`  "
                                  f"(stage `{_row.get('stage','?')}`, "
                                  f"status `{_row.get('status','?')}`)"),
                            _traj,
                        ])
                    except ImportError:
                        _md_traj_panel = mo.callout(
                            mo.md("**Mol* viewer not installed.**"),
                            kind="warn")
                    except Exception as _e:
                        _md_traj_panel = mo.callout(
                            mo.md(f"**MD trajectory load failed:** "
                                  f"`{type(_e).__name__}: {_e}`"),
                            kind="danger")

    if _md_traj_panel is not None:
        structure_viewer_panel = _md_traj_panel
    elif rd is None or not selected_design_names:
        structure_viewer_panel = mo.md(
            "*Click a point on the triage plane to overlay its structure.*")
    else:
        _byname = rd.designs.set_index("name")
        _paths: list[_Path] = []
        _missing: list[str] = []
        for _n in selected_design_names:
            if _n not in _byname.index:
                _missing.append(_n); continue
            _p = _Path(str(_byname.loc[_n, "_output_pdb"]))
            if _p.is_file():
                _paths.append(_p)
            else:
                _missing.append(_n)
        if not _paths:
            structure_viewer_panel = mo.callout(
                mo.md("**No PDB files on disk** for the selected design(s): "
                      + ", ".join(_missing)),
                kind="warn")
        else:
            # The binding partner is ALWAYS included as the alignment
            # reference (Kabsch-fit every design's target chain onto the
            # partner frame) and ALWAYS appears in the Structures legend,
            # but loads with visible=False by default. Toggling visibility
            # in the legend keeps the camera in place; the prior dedicated
            # `show_binding_partner` switch reset the viewport on each flip.
            _partner_warning = None
            _partner_path: _Path | None = None
            _cand = (rd.run_dir / "intermediate" / "0_import"
                     / "processed.pdb")
            if _cand.is_file():
                _partner_path = _cand
            else:
                _partner_warning = (
                    f"**Binding partner PDB missing** at "
                    f"`{_cand.relative_to(rd.run_dir)}`.")
            _superpose_input = (
                [_partner_path, *_paths] if _partner_path is not None
                else _paths)
            try:
                if _partner_path is not None:
                    # Sequence-aware: designs are Boltz-renumbered 1..N while
                    # the partner uses crystallographic numbering with gaps,
                    # so residue-number or index-order matching mis-pairs.
                    _aligned = A.superpose_pdbs_by_sequence(_superpose_input)
                else:
                    _aligned = A.superpose_pdbs(_superpose_input)
            except Exception as _e:
                _aligned = [p.read_text() for p in _superpose_input]
            if _partner_path is not None:
                _structures = [{
                    "name": "original binding partner",
                    "data": _aligned[0].encode("utf-8"),
                    "format": "pdb",
                    "visible": False,
                }]
                _opacities = [float(partner_opacity.value)]
                _aligned_designs = _aligned[1:]
            else:
                _structures = []
                _opacities = []
                _aligned_designs = _aligned
            _structures.extend({
                "name": _n, "data": _txt.encode("utf-8"), "format": "pdb",
            } for _n, _txt in zip(selected_design_names, _aligned_designs))
            _opacities.extend(1.0 for _ in _aligned_designs)
            # All per-residue color modes drive the widget's residue-values
            # overpaint; chain-id is the fallback.
            _color_scheme = (
                "residue-values"
                if struct_color_mode.value in ("mmgbsa-hotspots", "rmsf",
                                               "contact-frequency")
                else "chain-id"
            )
            _kwargs = dict(
                structures=_structures,
                structure_opacities=_opacities,
                color_scheme=_color_scheme,
                show_legend=True,
                representation="cartoon",
                height=540,
                residue_values=residue_values_payload,
            )
            _viewer = None
            _fallback_msg = None
            try:
                import molstar_marimo as _mm
                try:
                    _viewer = _mm.MolstarViewer(**_kwargs)
                except TypeError as _te:
                    # Older widget without `structures` / `residue_values`
                    # kwargs: fall back to single-structure topology_data.
                    _viewer = _mm.MolstarViewer(
                        topology_data=_structures[0]["data"],
                        topology_format="pdb",
                        color_scheme="chain-id",
                        representation="cartoon",
                        height=520,
                    )
                    _fallback_msg = (
                        f"Mol* widget is on an older API "
                        f"(`{type(_te).__name__}: {_te}`). Showing only the "
                        "first selected design without per-residue coloring.")
            except ImportError:
                structure_viewer_panel = mo.callout(
                    mo.md("**Mol* viewer not installed.** Run "
                          "`pip install -e projects/molstar-marimo` in the "
                          "`cpm2` env."), kind="warn")
                _viewer = None
            except Exception as _e:
                structure_viewer_panel = mo.callout(
                    mo.md(f"**Mol* viewer failed:** "
                          f"`{type(_e).__name__}: {_e}`"), kind="danger")
                _viewer = None

            if _viewer is not None:
                _bits = []
                if _fallback_msg:
                    _bits.append(mo.callout(mo.md(_fallback_msg),
                                            kind="warn"))
                if mmgbsa_warning:
                    _bits.append(mo.callout(mo.md(mmgbsa_warning),
                                            kind="warn"))
                if _partner_warning:
                    _bits.append(mo.callout(mo.md(_partner_warning),
                                            kind="warn"))
                _names_md = mo.md(
                    "**Overlay:** " + ", ".join(f"`{n}`"
                                                for n in selected_design_names)
                    + ("  *(missing on disk: "
                       + ", ".join(_missing) + ")*" if _missing else ""))
                _bits.extend([_names_md, _viewer])
                structure_viewer_panel = mo.vstack(_bits)
    return (structure_viewer_panel,)


@app.cell(hide_code=True)
def _(mo):
    # Drilldown sub-controls split per marimo's one-widget-per-cell rule.
    _opts_by = ["iptm", "plddt", "iplddt", "cpepmatch_fit_rmsd",
                "peptide_drift_ca_rmsd", "target_drift_ca_rmsd"]
    rank_by = mo.ui.dropdown(options=_opts_by, value="iptm",
                             label="rank by")
    return (rank_by,)


@app.cell(hide_code=True)
def _(mo):
    rank_n = mo.ui.slider(start=5, stop=50, step=1, value=20, label="top-n")
    return (rank_n,)


@app.cell(hide_code=True)
def _(mo):
    rank_cyclic = mo.ui.switch(value=True, label="cyclic only")
    return (rank_cyclic,)


@app.cell(hide_code=True)
def _(mo):
    rank_asc = mo.ui.switch(value=False, label="ascending")
    return (rank_asc,)


@app.cell(hide_code=True)
def _(mo, rd):
    _N = max(1, rd.n_designs) if rd is not None else 1
    conv_n = mo.ui.slider(start=1, stop=min(20, _N), step=1,
                          value=min(8, _N), label="convergence traces")
    return (conv_n,)


@app.cell(hide_code=True)
def _(mo):
    sl_n = mo.ui.slider(start=1, stop=20, step=1, value=3,
                        label="shortlist n")
    return (sl_n,)


@app.cell(hide_code=True)
def _(mo):
    sl_per_topo = mo.ui.switch(value=True, label="per topology")
    return (sl_per_topo,)


@app.cell(hide_code=True)
def _(mo):
    sl_min_iptm = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.85,
                               label="min ipTM")
    return (sl_min_iptm,)


@app.cell(hide_code=True)
def _(mo):
    write_btn = mo.ui.run_button(
        label="Write mmgbsa_shortlist.csv to the run")
    return (write_btn,)


@app.cell(hide_code=True)
def _(mo):
    spotlight = mo.ui.dropdown(
        options=[
            "page (everything)",
            "overview text",
            "score / RMSD distributions",
            "topology charts",
            "topology audit table",
            "triage plane",
            "ranked table",
            "PH convergence",
            "MM-GBSA shortlist",
            "design vs natural partner",
            "specificity matrix",
            "ipTM vs MM-GBSA",
        ],
        value="page (everything)",
        label="spotlight (enlarge one panel)",
    )
    return (spotlight,)


@app.cell(hide_code=True)
def _(
    A,
    conv_n,
    mmgbsa_only_filter,
    mo,
    partner_opacity,
    plane,
    rank_asc,
    rank_by,
    rank_cyclic,
    rank_n,
    rd,
    show_md_trajectory,
    md_traces_panel,
    md_traj_drilldown_stride,
    sl_min_iptm,
    sl_n,
    sl_per_topo,
    struct_color_mode,
    structure_viewer_panel,
    topo_filter,
    write_btn,
    x_axis,
    y_axis,
):
    if rd is None:
        panels = {}
    else:
        _ranked_all = A.rank(rd, by=rank_by.value, n=rank_n.value,
                             cyclic_only=rank_cyclic.value,
                             ascending=rank_asc.value)
        # Brush on the triage plane filters the ranked table by `name`.
        try:
            _brushed = plane.value
        except AttributeError:
            _brushed = None
        if (_brushed is not None and hasattr(_brushed, "empty")
                and not _brushed.empty and "name" in _brushed.columns):
            _ranked = _ranked_all[_ranked_all["name"]
                                  .isin(_brushed["name"])].reset_index(
                drop=True)
            _ranked_note = mo.md(
                f"*Filtered to {len(_ranked)} brushed design(s) from the "
                "triage plane above.*")
        else:
            _ranked = _ranked_all
            _ranked_note = mo.md(
                "*Brush the triage plane above to filter this table.*")

        _picks = A.shortlist(rd, n=sl_n.value,
                             per_topology=sl_per_topo.value,
                             min_iptm=sl_min_iptm.value)
        if write_btn.value:
            _out = rd.run_dir / "output" / "mmgbsa_shortlist.csv"
            _picks.to_csv(_out, index=False)
            _write_status = mo.callout(
                mo.md(f"Wrote **{len(_picks)}** rows to `{_out}`"),
                kind="success")
        else:
            _write_status = mo.md(
                "*Click the button to persist the shortlist below.*")

        panels = {
            "overview text": mo.md(f"```\n{rd.overview()}\n```"),
            "score / RMSD distributions": A.plot_distributions(rd),
            "topology charts": mo.hstack(
                [A.plot_topology(rd), A.plot_topology_fidelity(rd)],
                widths="equal", wrap=True, gap=2),
            "topology audit table": mo.ui.table(
                rd.topology_audit().round(3), selection=None,
                pagination=True),
            "triage plane": mo.vstack([
                # Plane filters.
                mo.hstack([topo_filter, x_axis, y_axis,
                           mmgbsa_only_filter],
                          justify="start", gap=2, wrap=True),
                # Structure-viewer controls hoisted above the side-by-side
                # row so the triage chart (left) and structure viewer
                # (right) share the same top edge -- prior layout put these
                # controls in the right vstack, pushing the canvas top
                # below the chart top.
                mo.hstack([struct_color_mode, partner_opacity,
                           show_md_trajectory, md_traj_drilldown_stride],
                          justify="start", gap=2, wrap=True),
                mo.hstack(
                    [plane, structure_viewer_panel],
                    widths=[1, 1], gap=2, wrap=False, align="stretch",
                ),
                # MD traces under the panels: peptide_rmsd_A +
                # com_distance_A vs time for the single selected
                # design's MD job (when available). Stays empty /
                # informational when no/multiple designs are picked.
                md_traces_panel,
            ]),
            "ranked table": mo.vstack([
                mo.hstack([rank_by, rank_n, rank_cyclic, rank_asc],
                          justify="start", gap=2, wrap=True),
                _ranked_note,
                mo.ui.table(_ranked.round(3), selection=None,
                            pagination=True),
            ]),
            "PH convergence": mo.vstack([
                conv_n,
                mo.ui.altair_chart(A.convergence_altair(rd, n=conv_n.value)),
                mo.md(
                    "*Only the **target** CA-RMSD is logged per cycle, so a "
                    "flat trace means a self-consistent basin, not a "
                    "descended loss. With `template_force` on the trace is "
                    "forcing-bounded. The best ipTM was still climbing at "
                    "cycle 10 (the loop's max), so the reported best is "
                    "partly **max-of-N inflation** -- the convergence-100 "
                    "experiment in section C tests exactly this.*"),
            ]),
            "MM-GBSA shortlist": mo.vstack([
                mo.hstack([sl_n, sl_per_topo, sl_min_iptm],
                          justify="start", gap=2, wrap=True),
                mo.md(f"**{len(_picks)} design(s)** in the current "
                      "shortlist:"),
                mo.ui.table(_picks.round(3), selection=None,
                            pagination=True),
                write_btn,
                _write_status,
            ]),
        }
    return (panels,)


# ==========================================================================
# Critical-evaluation backend panels (section B): design-vs-reference,
# specificity matrix, ipTM<->MMGBSA agreement, queued experiments.
# These do not depend on the active `rd`; they read the cross-run MD
# benchmark CSVs through `critique` (C) and `analysis` (A).
# ==========================================================================
@app.cell(hide_code=True)
def _(C, mo):
    # Design vs natural-partner dG, per target. Grouped bar with honest
    # +-5 kcal/mol error bars (NOT the MMPBSA sqrt-of-frames SE).
    _dvr = C.design_vs_reference()
    if _dvr.empty:
        dvr_panel = mo.md(
            "*No cognate dG values parsed yet (the 05-26 batch CSV is "
            "empty or absent).*")
        dvr_table = mo.md("")
    else:
        _chart = mo.ui.altair_chart(C.plot_design_vs_reference(_dvr))
        _cav = _dvr.dropna(subset=["caveat"])
        _cav_md = ""
        if not _cav.empty:
            _cav_md = "\n\n" + "\n".join(
                f"- **{r.target_label}:** {r.caveat}"
                for r in _cav.itertuples())
        dvr_panel = mo.vstack([
            _chart,
            mo.md(
                "Negative = stronger predicted binding. The honest error "
                "bars are +-5 kcal/mol (autocorrelated MD error), not the "
                "0.5 kcal/mol the MMPBSA frame-SE would suggest. A design "
                "only clearly beats its natural partner if the bars do not "
                "overlap." + _cav_md),
        ])
        _show = _dvr[["target_label", "design_dG", "reference_dG",
                      "reference_label", "gap"]].copy()
        dvr_table = mo.ui.table(_show, selection=None, pagination=False)
    return dvr_panel, dvr_table


@app.cell(hide_code=True)
def _(C, mo):
    # Negative-control specificity matrix: ddG vs cognate over
    # (condition x target). Green = control lost binding (discriminates),
    # red = near-zero ddG (composition-driven, not sequence-specific).
    _mtx = C.specificity_matrix()
    if _mtx.empty:
        spec_panel = mo.md(
            "*No specificity matrix yet (negative-control batch CSV is "
            "empty or absent).*")
        spec_table = mo.md("")
    else:
        _chart = mo.ui.altair_chart(C.plot_specificity_heatmap(_mtx))
        _cav = _mtx.dropna(subset=["caveat"])
        _cav_md = ""
        if not _cav.empty:
            _cav_md = "\n\n" + "\n".join(
                f"- **{r.target_label} / {r.condition_label}:** {r.caveat}"
                for r in _cav.itertuples())
        spec_panel = mo.vstack([
            _chart,
            mo.md(
                "ddG = dG(control) - dG(cognate). Positive (green) means "
                "the perturbed peptide lost binding, so the design is "
                "sequence- and target-specific. Near-zero on scrambled or "
                "ala-scan (red) means the apparent win is **composition-"
                "driven**, the CypA failure mode." + _cav_md),
        ])
        _show = _mtx[["target_label", "condition_label", "dG",
                      "ddg_vs_cognate", "caveat"]].copy()
        spec_table = mo.ui.table(_show, selection=None, pagination=False)
    return spec_panel, spec_table


@app.cell(hide_code=True)
def _(A, mo):
    # ipTM <-> MM-GBSA agreement. Reuses the analysis MMGBSA join. Robust
    # to the running CSVs being absent / empty -> degrade to a note.
    try:
        joined, refs = A.load_mmgbsa_join()
    except (FileNotFoundError, OSError, KeyError, ValueError):
        joined, refs = None, None

    if joined is None or joined.empty:
        iptm_panel = mo.md(
            "*No MM-GBSA join available yet "
            "(`mmgbsa results` + `shortlist` CSVs absent or empty).*")
    else:
        _corr = A.mmgbsa_correlations(joined)
        _flips = A.mmgbsa_ranking_flips(joined)
        _chart = mo.ui.altair_chart(A.plot_mmgbsa_vs_iptm(joined, refs))
        _flip_any = (bool(_flips["ranking_flipped"].any())
                     if not _flips.empty else False)
        iptm_panel = mo.vstack([
            _chart,
            mo.md(
                "*Read this panel as **under-powered**: small n per target "
                "and a narrow ipTM spread (most designs cluster in a tight "
                "0.8-0.95 band), so a per-target correlation here is "
                "suggestive at best. The expected sign is negative (higher "
                "ipTM -> more negative dG). "
                + ("At least one target's best-by-ipTM design is **not** its "
                   "best-by-MM-GBSA design (a ranking flip), which is "
                   "exactly why the cheap filter cannot stand in for the "
                   "expensive one.*" if _flip_any else
                   "No ranking flips in the current join, but do not "
                   "over-read that given the small n.*")),
            mo.accordion({
                "Per-target ipTM<->dG correlation":
                    mo.ui.table(_corr, selection=None, pagination=False),
                "Best-by-ipTM vs best-by-MM-GBSA (ranking flips)":
                    mo.ui.table(_flips, selection=None, pagination=False),
            }),
        ])
    return (iptm_panel,)


@app.cell(hide_code=True)
def _(C, mo):
    # Queued experiments: designed-but-pending critical tests. If a result
    # artefact is absent we MUST NOT claim any outcome.
    _q = C.queued_experiments()
    if not _q:
        queued_panel = mo.md("*No queued experiments registered.*")
    else:
        _rows = []
        for _e in _q:
            if _e["has_results"]:
                _badge = ("<span style='color:#55A868;font-weight:600'>"
                          "RESULTS IN</span>")
            else:
                _badge = ("<span style='color:#C0A000;font-weight:600'>"
                          "QUEUED -- no outcome to report</span>")
            _rows.append(
                f"**{_e['name']}** &nbsp; {_badge}\n\n"
                f"<span style='color:#888'>{_e['question']}</span>")
        queued_panel = mo.callout(
            mo.md("**Designed, results pending**\n\n"
                  + "\n\n---\n\n".join(_rows)
                  + "\n\nThese two experiments are scaffolded on Slurm but "
                  "have not returned. Until their result files exist on "
                  "disk, no outcome is claimed here."),
            kind="info")
    return (queued_panel,)


@app.cell(hide_code=True)
def _(mo, repo):
    # Topology-bug post-mortem: a background agent is writing
    # benchmarks/TOPOLOGY_BUG_POSTMORTEM.md. If present, surface its final
    # summary section; if absent, a neutral pending note. Robust to absence.
    topo_postmortem_panel = None
    try:
        _pm = repo / "benchmarks" / "TOPOLOGY_BUG_POSTMORTEM.md"
        if _pm.is_file():
            _text = _pm.read_text(encoding="utf-8", errors="replace")
            # Pull the last "## ..." section as the final summary; if no
            # headings, show the whole (short) doc tail.
            _lines = _text.splitlines()
            _heads = [i for i, ln in enumerate(_lines)
                      if ln.lstrip().startswith("## ")]
            if _heads:
                # Skip the heading line itself so the callout reads as prose,
                # not a stray "## 5. ..." section title.
                _summary = "\n".join(_lines[_heads[-1] + 1:]).strip()
            else:
                _summary = "\n".join(_lines[-40:]).strip()
            # Guard against an enormous tail.
            if len(_summary) > 4000:
                _summary = _summary[:4000] + "\n\n*(truncated)*"
            topo_postmortem_panel = mo.callout(
                mo.md("**Topology-bug post-mortem** "
                      "(`benchmarks/TOPOLOGY_BUG_POSTMORTEM.md`)\n\n"
                      + _summary),
                kind="warn")
        else:
            topo_postmortem_panel = mo.callout(
                mo.md("**Topology-bug post-mortem:** *pending.* The audit "
                      "table above flags any design built with a topology "
                      "unlike its cPEPmatch DB declaration; the written "
                      "post-mortem (`benchmarks/TOPOLOGY_BUG_POSTMORTEM.md`) "
                      "is not on disk yet."),
                kind="neutral")
    except Exception as _e:
        topo_postmortem_panel = mo.md(
            f"*Topology-bug post-mortem unavailable: "
            f"`{type(_e).__name__}: {_e}`.*")
    return (topo_postmortem_panel,)


@app.cell(hide_code=True)
def _(mo):
    # Honest epistemics ledger. Hideable (accordion), default collapsed so
    # the page is not a wall of caveats, but always one click away.
    epistemics_panel = mo.accordion({
        "Epistemics: what these numbers can and cannot bear (read me)":
            mo.callout(
                mo.md(
                    "Every MD-derived number on this page rests on a "
                    "**single trajectory per cell (n=1)**. The direction of "
                    "an effect (does binding survive, does a control lose "
                    "it) is fairly robust; the **magnitudes are not**.\n\n"
                    "- **True uncertainty is +-3 to 5 kcal/mol**, not the "
                    "~0.5 kcal/mol the MMPBSA frame-SE reports. The frame-SE "
                    "assumes independent frames; MD frames are "
                    "autocorrelated, so it under-estimates the real error "
                    "by 3-10x.\n"
                    "- **Bcl-xL natural reference is not converged at 10 ns** "
                    "-- treat its dG as an under-confident floor, not a "
                    "settled value.\n"
                    "- **CypA has no usable cyclic reference**: the "
                    "cyclosporin natural binder is a GAFF / N-methylation "
                    "artefact, so we fall back to the ALY-fixed capsid "
                    "peptide and flag it.\n"
                    "- **MDM2 scrambled control carries a CYX->ALA "
                    "mutagenesis artefact** (a lost disulfide), which "
                    "overstates its ddG.\n"
                    "- All MM-GBSA here is **GB-only, not PB**: cheaper, "
                    "noisier, and not directly comparable to published PB "
                    "numbers.\n\n"
                    "Read the page as a *direction-of-effect* instrument, "
                    "not a calibrated affinity predictor."),
                kind="warn"),
    })
    return (epistemics_panel,)


# ==========================================================================
# MD loading + trajectory viewer cells (reused verbatim). The active bundle
# `mdb` feeds the triage-plane stars, the structure overlay color modes,
# and the MD traces panel above; the bundle-level viewer/status panels feed
# section B's MD-stability block.
# ==========================================================================
@app.cell(hide_code=True)
def _(Q, mo):
    _bundles = Q.list_md_overnight()
    md_bundles_select = mo.ui.multiselect(
        options={b.name: str(b) for b in _bundles},
        value=[b.name for b in _bundles],
        label="bundles to compare",
    )
    return (md_bundles_select,)


@app.cell(hide_code=True)
def _(Q, mo):
    _bundles = Q.list_md_overnight()
    md_picker = mo.ui.dropdown(
        options={b.name: str(b) for b in _bundles},
        value=_bundles[-1].name if _bundles else None,
        label="active bundle (drilldown + trajectory viewer)",
    )
    return (md_picker,)


@app.cell(hide_code=True)
def _(mo):
    md_refresh = mo.ui.button(label="reload from disk")
    return (md_refresh,)


@app.cell(hide_code=True)
def _(mo):
    md_autorefresh = mo.ui.refresh(default_interval="60s",
                                   label="auto-refresh (60s)")
    return (md_autorefresh,)


@app.cell(hide_code=True)
def _(Q, md_autorefresh, md_picker, md_refresh):
    _ = md_refresh.value
    _ = md_autorefresh.value
    mdb = Q.load_md_overnight(md_picker.value) if md_picker.value else None
    return (mdb,)


@app.cell(hide_code=True)
def _(Q, md_bundles_select, md_refresh, pd):
    _ = md_refresh.value
    # Joined DataFrame of all selected bundles' jobs for comparative
    # plots. `bundle` column added; we keep the rest as-is.
    _frames = []
    for _p in (md_bundles_select.value or []):
        try:
            _b = Q.load_md_overnight(_p)
        except FileNotFoundError:
            continue
        if _b.jobs.empty:
            continue
        _j = _b.jobs.copy()
        _j["bundle"] = _b.name
        _frames.append(_j)
    md_compare_df = (pd.concat(_frames, ignore_index=True)
                     if _frames else pd.DataFrame())
    return (md_compare_df,)


@app.cell(hide_code=True)
def _(alt, md_compare_df, mo):
    # Faceted dG strip + status bars across selected bundles.
    if md_compare_df.empty:
        md_compare_panel = mo.md(
            "*No bundles selected, or selected bundles have no jobs.*")
    else:
        _dg = md_compare_df.dropna(subset=["dG_kcal_mol"]).copy()
        if _dg.empty:
            _dg_chart = mo.md("*No MMGBSA dG values parsed yet across "
                              "selected bundles.*")
        else:
            _dg["kind"] = _dg["is_reference"].map(
                {True: "reference", False: "design"})
            _c = (
                alt.Chart(_dg)
                .mark_circle(opacity=0.85, size=140, stroke="white",
                             strokeWidth=0.7)
                .encode(
                    x=alt.X("target:N",
                            axis=alt.Axis(labelAngle=-30, title="target")),
                    y=alt.Y("dG_kcal_mol:Q",
                            title="MMGBSA dG  [kcal/mol]",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "kind:N",
                        scale=alt.Scale(domain=["design", "reference"],
                                        range=["#4c78a8", "#e45756"])),
                    column=alt.Column("bundle:N", title=None),
                    tooltip=["bundle", "job", "target", "kind",
                            "dG_kcal_mol", "SD", "SE", "stage"],
                )
                .properties(width=280, height=360,
                            title="MMGBSA dG per target (faceted by "
                                  "bundle, shared y)")
            )
            _dg_chart = mo.ui.altair_chart(_c)

        _sc = (md_compare_df.groupby(["bundle", "status"])
               .size().reset_index(name="n"))
        _status_chart = mo.ui.altair_chart(
            alt.Chart(_sc)
            .mark_bar()
            .encode(
                x=alt.X("status:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("n:Q", title="jobs"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["done", "running", "queued", "failed"],
                        range=["#55A868", "#4c78a8", "#bbbbbb", "#C44E52"])),
                xOffset="bundle:N",
                tooltip=["bundle", "status", "n"],
            )
            .properties(width=420, height=280,
                        title="Job status counts (grouped by bundle)")
        )
        md_compare_panel = mo.vstack([
            mo.md("### Across selected bundles"),
            _status_chart,
            _dg_chart,
        ])
    return (md_compare_panel,)


@app.cell(hide_code=True)
def _(mdb, mo):
    if mdb is None or mdb.jobs.empty:
        _opts = {}
    else:
        _opts = {}
        for r in mdb.jobs.itertuples():
            nc = getattr(r, "free_run_nc", None)
            prm = getattr(r, "prmtop", None)
            if nc and prm:
                _opts[r.job] = f"{nc}|{prm}"
    md_traj_pick = mo.ui.dropdown(
        options=_opts,
        value=next(iter(_opts.keys()), None) if _opts else None,
        label="job to view (Mol* trajectory)",
    )
    return (md_traj_pick,)


@app.cell(hide_code=True)
def _(mo):
    md_traj_stride = mo.ui.slider(start=1, stop=200, step=1, value=1,
                                  label="frame stride (subsample)")
    return (md_traj_stride,)


@app.cell(hide_code=True)
def _(md_traj_pick, md_traj_stride, mo):
    if not md_traj_pick.value:
        traj_viewer = mo.md(
            "*This bundle has no jobs with both `free_run/run.nc` and "
            "`system_wb.prmtop` on disk.*")
    else:
        try:
            import molstar_marimo as mm
            _nc, _prm = md_traj_pick.value.split("|", 1)
            # amber_to_bytes splits chains by molecular connectivity
            # in descending atom count: chain A == target, chain B ==
            # cyclic peptide for CPM2 complexes. PyMOL-style default:
            # target as surface, peptide as ball-and-stick.
            traj_viewer = mm.MolstarViewer.from_amber(
                nc_path=_nc, prmtop_path=_prm,
                stride=md_traj_stride.value,
                strip="!:WAT,Na+,Cl-",
                height=520,
                representation="cartoon",
                color_scheme="chain-id",
                show_legend=True,
                autoplay=True,
                fps=12,
                loop=True,
                chain_representations={
                    "A": "surface",
                    "B": "ball-and-stick",
                },
            )
        except ImportError:
            traj_viewer = mo.callout(
                mo.md(
                    "**Mol* viewer not installed.** Run "
                    "`pip install -e projects/molstar-marimo` plus "
                    "`pip install anywidget mdtraj` in the `cpm2` env."),
                kind="warn")
        except Exception as _e:
            traj_viewer = mo.callout(
                mo.md(f"**Mol* viewer failed:** `{type(_e).__name__}: "
                      f"{_e}`"),
                kind="danger")
    return (traj_viewer,)


@app.cell(hide_code=True)
def _(mdb, mo):
    if mdb is None or mdb.jobs.empty:
        md_status_panel = mo.md("*No jobs found.*")
    else:
        _cols = ["job", "target", "stage", "status", "dG_kcal_mol",
                 "SD", "SE", "is_reference"]
        _cols = [c for c in _cols if c in mdb.jobs.columns]
        _j = (mdb.jobs[_cols]
              .sort_values(["status", "is_reference", "target", "job"],
                           ascending=[True, True, True, True])
              .reset_index(drop=True))
        md_status_panel = mo.ui.table(_j, selection=None,
                                      pagination=True, page_size=20)
    return (md_status_panel,)


@app.cell(hide_code=True)
def _(mdb, mo):
    if mdb is None:
        md_join_panel = mo.md("")
        md_log_panel = mo.md("")
        md_images_panel = mo.md("")
    else:
        _joined = mdb.joined()
        if _joined.empty:
            md_join_panel = mo.md("*No shortlist to join.*")
        else:
            _jcols = ["name", "target", "topology", "iptm", "plddt",
                      "cpepmatch_fit_rmsd", "peptide_drift_ca_rmsd",
                      "stage", "status", "dG_kcal_mol", "SD"]
            _jcols = [c for c in _jcols if c in _joined.columns]
            _show = (_joined[_jcols]
                     .sort_values("dG_kcal_mol", na_position="last")
                     .reset_index(drop=True))
            md_join_panel = mo.ui.table(_show, selection=None,
                                        pagination=True, page_size=20)

        if not mdb.monitor_log:
            md_log_panel = mo.md("*No monitor.log in this bundle.*")
        else:
            _tail = "\n".join(mdb.monitor_log.strip().splitlines()[-30:])
            md_log_panel = mo.md(f"```\n{_tail}\n```")

        _imgs = [mo.vstack([mo.md(f"**{p.name}**"), mo.image(str(p))])
                 for p in mdb.verification_images]
        if _imgs:
            md_images_panel = mo.hstack(_imgs, widths="equal", wrap=True,
                                        gap=2)
        else:
            md_images_panel = mo.md("*No verification images.*")
    return md_images_panel, md_join_panel, md_log_panel


@app.cell(hide_code=True)
def _(
    dvr_panel,
    dvr_table,
    epistemics_panel,
    iptm_panel,
    md_autorefresh,
    md_bundles_select,
    md_compare_panel,
    md_images_panel,
    md_join_panel,
    md_log_panel,
    md_picker,
    md_refresh,
    md_status_panel,
    md_traj_pick,
    md_traj_stride,
    mdb,
    mo,
    panels,
    queued_panel,
    rd,
    run_info_pane,
    drilldown_runs,
    run_picker,
    spec_panel,
    spec_table,
    spotlight,
    topo_postmortem_panel,
    traj_viewer,
):
    # ---- single unified page -------------------------------------------
    # No top-level tabs. One flowing page in three movements (A / B / C),
    # with depth tucked into accordions and a spotlight to enlarge any one
    # panel for a live walk-through.

    # Spotlight short-circuit: enlarge a single panel.
    _spot = spotlight.value
    if _spot != "page (everything)":
        _spot_map = dict(panels) if panels else {}
        _spot_map.update({
            "design vs natural partner": dvr_panel,
            "specificity matrix": spec_panel,
            "ipTM vs MM-GBSA": iptm_panel,
        })
        _spot_body = _spot_map.get(
            _spot, mo.md("*Spotlight target not available for this run.*"))
        page = mo.vstack([
            mo.hstack([run_picker, spotlight],
                      justify="start", gap=2, wrap=True),
            mo.md(f"### Spotlight: {_spot}"),
            _spot_body,
        ])
    elif rd is None:
        page = mo.vstack([
            mo.md(f"**{len(drilldown_runs)} runs** under "
                  "`data/runs/` and `archives/`."),
            mo.hstack([run_picker, spotlight],
                      justify="start", gap=2, wrap=True),
            mo.md("*No run selected.*"),
        ])
    else:
        # MD bundle controls + status, used by section B's stability block.
        _md_toolbar = mo.hstack(
            [md_bundles_select, md_picker, md_refresh, md_autorefresh],
            justify="start", gap=2, wrap=True)
        if mdb is None:
            _md_header = mo.md("*No MD overnight bundle loaded.*")
        else:
            _live = "LIVE" if mdb.is_live else "complete"
            _d = (mdb.jobs["status"] == "done").sum() \
                if not mdb.jobs.empty else 0
            _r = (mdb.jobs["status"] == "running").sum() \
                if not mdb.jobs.empty else 0
            _q = (mdb.jobs["status"] == "queued").sum() \
                if not mdb.jobs.empty else 0
            _f = (mdb.jobs["status"] == "failed").sum() \
                if not mdb.jobs.empty else 0
            _md_header = mo.md(
                f"**Active MD bundle:** `{mdb.name}` *({_live})* -- "
                f"done {_d} / running {_r} / queued {_q} / failed {_f}. "
                f"Shortlist rows: **{len(mdb.shortlist)}**.")

        _md_traj_block = mo.vstack([
            mo.md("#### Mol* trajectory player  (autoplay, chain-id colors)"),
            mo.hstack([md_traj_pick, md_traj_stride],
                      justify="start", gap=2, wrap=True),
            traj_viewer,
        ])

        page = mo.vstack([
            # --- header / run selector ---
            mo.md(f"**{len(drilldown_runs)} runs** on disk. Pick one to "
                  "examine; the whole page reacts."),
            mo.hstack([run_picker, spotlight],
                      justify="start", gap=2, wrap=True),
            run_info_pane,

            # ======================================================
            mo.md(
                "## A. Does the pipeline produce good-looking designs?\n\n"
                "Before physics, the cheap checks: are the topologies built "
                "the way the cPEPmatch database declares them, do the scores "
                "and geometric drifts cluster sensibly, and which designs "
                "rise to the top? The triage plane below is the working "
                "surface: brush to filter the ranked table, click a point to "
                "overlay its structure (★ = an MD trajectory is on disk)."),

            mo.md("### A.1 Triage plane + structure overlay"),
            panels["triage plane"],

            mo.md("### A.2 Topology fidelity (the topology-bug audit)\n\n"
                  "Each design's as-built topology is compared against the "
                  "cPEPmatch DB declaration for its source PDB. A `bug` "
                  "verdict means the complex was built with a topology the "
                  "database did not declare."),
            panels["topology charts"],
            mo.accordion({
                "Per-match topology audit table":
                    panels["topology audit table"],
            }),
            topo_postmortem_panel,

            mo.md("### A.3 Ranked designs + MM-GBSA shortlist"),
            panels["ranked table"],
            mo.accordion({
                "Score / RMSD distributions":
                    panels["score / RMSD distributions"],
                "ProteinHunter refinement convergence":
                    panels["PH convergence"],
                "MM-GBSA shortlist (writeable)":
                    panels["MM-GBSA shortlist"],
                "Run overview (text)":
                    panels["overview text"],
            }),

            # ======================================================
            mo.md(
                "## B. Do those scores survive contact with physics?\n\n"
                "A good-looking ipTM is a hypothesis, not a result. Here we "
                "ask whether the designs hold up under MD + MM-GBSA: do they "
                "match natural binders, are they sequence- and target-"
                "specific, does the cheap ipTM filter agree with the "
                "expensive physics, and do the winners stay bound over a "
                "trajectory?"),

            mo.md("### B.1 Designed binders vs natural partners"),
            dvr_panel,
            mo.accordion({"design-vs-reference table": dvr_table}),

            mo.md("### B.2 Negative-control specificity"),
            spec_panel,
            mo.accordion({"specificity matrix (full)": spec_table}),

            mo.md("### B.3 ipTM <-> MM-GBSA agreement"),
            iptm_panel,

            mo.md("### B.4 MD trajectory stability\n\n"
                  "The per-target winners, simulated. The triage plane stars "
                  "(section A.1) mark designs whose trajectory is on disk: "
                  "select one there and toggle *play MD trajectory* to watch "
                  "it in the overlay, or pick any finished job in the player "
                  "below. The traces (peptide RMSD vs frame 0, peptide-target "
                  "CoM distance) live under the triage plane."),
            _md_toolbar,
            _md_header,
            mo.accordion({
                "Cross-bundle MM-GBSA dG + job status":
                    md_compare_panel,
                "Per-job status (active bundle)":
                    md_status_panel,
                "Shortlist x MD outcome":
                    md_join_panel,
                "Trajectory player":
                    _md_traj_block,
                "Monitor log tail":
                    md_log_panel,
                "Verification images":
                    md_images_panel,
            }),

            # ======================================================
            mo.md(
                "## C. What we cannot yet claim\n\n"
                "Two critical experiments are scaffolded but not back; until "
                "their results land, their questions stay open. And every "
                "MD-derived magnitude on this page rests on n=1, so read the "
                "epistemics ledger before quoting any number."),
            queued_panel,
            epistemics_panel,

            # ======================================================
            mo.md("## Verdict"),
            mo.callout(
                mo.md(
                    "The pipeline is **internally consistent and "
                    "discriminating**: on 3 of 4 targets the designed "
                    "peptide loses predicted binding when its sequence is "
                    "scrambled or ala-scanned and when it is pushed onto the "
                    "wrong target, so the wins are sequence- *and* target-"
                    "specific, not artefacts of the scoring set-up. It is "
                    "**not clearly competitive with natural binders on dG**: "
                    "within the honest +-5 kcal/mol error, the designs sit at "
                    "or behind their natural partners rather than ahead of "
                    "them. **CypA is a cautionary case** -- its apparent win "
                    "is composition-driven (scrambling barely changes the "
                    "dG), so it does not count as a specific binder. All of "
                    "this rests on **n=1 MD per cell**: the *direction* of "
                    "each effect is robust, the *magnitudes* are not. A "
                    "promising warm-up that has earned the right to a "
                    "properly-powered follow-up, not a finished result."),
                kind="success"),
        ])
    page
    return


if __name__ == "__main__":
    app.run()
