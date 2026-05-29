"""cpm2.critique -- the "does the pipeline survive contact with physics" lens.

Read-only loaders + Altair plots for the critical-evaluation surface of the
presentation notebook (``notebooks/CPM2_presentation.py``). Distinct from
``quality.py`` (per-bundle MD job status) because the negative-control batch
``benchmarks/md_20260528_negcontrols/`` is *not* an ``md_*_overnight`` bundle
and its scientific content is a per-target cognate-vs-control *matrix*, not a
job list.

Data sources (both are *running* CSVs -- read fresh every call):

* ``benchmarks/md_20260526_overnight/results_running.csv`` -- cognate design
  dG_GB per target (3 designs each) + natural-reference rows.
* ``benchmarks/md_20260528_negcontrols/results_running.csv`` -- alascan /
  scrambled / wrongtarget_pushout controls for the per-target winner.

Sign convention for dG_GB (kcal/mol): more negative = stronger predicted
binding. ddG vs cognate is ``dG_control - dG_cognate``; **positive ddG means
the control lost binding = the pipeline discriminates** (good). A ddG near
zero on scrambled/alascan means the "win" is composition-driven, not
sequence-specific (the CypA failure mode).

Honest error: the per-cell MMPBSA SE assumes independent frames and
under-estimates the true autocorrelated error by ~3-10x. Per the batch's
own final_report.md, treat per-cell uncertainty as +-HONEST_SE kcal/mol;
anything within +-HONEST_SE of zero ddG is noise.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
COGNATE_CSV = BENCHMARKS_DIR / "md_20260526_overnight" / "results_running.csv"
NEGCTRL_CSV = BENCHMARKS_DIR / "md_20260528_negcontrols" / "results_running.csv"

# Honest per-cell uncertainty (kcal/mol). See module docstring + the batch
# final_report.md "Caveats" section. NOT the MMPBSA sqrt-of-frames SE.
HONEST_SE = 5.0

# Display order + human labels for the four targets.
TARGETS = ["mdm2_p53", "bclxl", "14_3_3", "cypA"]
TARGET_LABELS = {
    "mdm2_p53": "MDM2 / p53",
    "bclxl": "Bcl-xL / Bad",
    "14_3_3": "14-3-3 / Cdc25C",
    "cypA": "CypA / HIV-cap",
}

# The per-target winner benchmarked in the negative-control batch. Its cognate
# dG is read from the 05-26 CSV by this exact design name.
WINNERS = {
    "mdm2_p53": "match15_2nb6_model_3_design2",
    "bclxl": "match51_3uc7_model_3_design3",
    "14_3_3": "match109_6dzb_model_2_design0",
    "cypA": "match37_5eoc_model_0_design0",
}

# Natural-reference choice per target. The reference rows in the 05-26 CSV use
# these `target`-column tags. Choices follow the final_report Caveats:
#   - cypA: the ALY->Lys-fixed capsid ref (not the broken cyclosporin GAFF one).
#   - bclxl: the 10 ns Bad ref is NOT converged -> flagged, treat as a floor.
REF_CHOICE = {
    "mdm2_p53": ("ref_mdm2", "1ycr p53 TAD", None),
    "bclxl": ("ref_bclxl", "1g5j Bad",
              "10 ns ref not converged; treat as an under-confident floor"),
    "14_3_3": ("ref_14_3_3", "5m36 phospho-Cdc25C", None),
    "cypA": ("ref_cypA_nat_fixed", "2x2d HIV capsid (ALY-fixed)",
             "no usable cyclic reference (cyclosporin GAFF artefact)"),
}

# Condition display order for the specificity matrix.
CONDITIONS = ["cognate", "alascan", "scrambled", "wrongtarget", "reference"]
CONDITION_LABELS = {
    "cognate": "cognate (design)",
    "alascan": "ala-scan",
    "scrambled": "scrambled seq",
    "wrongtarget": "wrong target",
    "reference": "natural partner",
}

# Per-cell caveats surfaced on the matrix (target, condition) -> note.
_CELL_CAVEATS = {
    ("mdm2_p53", "scrambled"):
        "CYX->ALA mutagenesis artefact: lost an SS bridge, overstates ddG",
    ("mdm2_p53", "wrongtarget"):
        "high noise (SE 1.3) in the Bcl-xL groove; anecdotal",
    ("cypA", "scrambled"):
        "ddG ~0: composition-driven, NOT sequence-specific (key finding)",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_cognates() -> pd.DataFrame:
    """All design + reference rows from the 05-26 cognate batch."""
    return _read_csv(COGNATE_CSV)


def load_negcontrols() -> pd.DataFrame:
    """Control rows (alascan / scrambled / wrongtarget_pushout)."""
    df = _read_csv(NEGCTRL_CSV)
    if df.empty:
        return df
    # Normalise the wrongtarget_pushout label to a short 'wrongtarget'.
    df = df.copy()
    df["control_type"] = df["control_type"].replace(
        {"wrongtarget_pushout": "wrongtarget"})
    return df


def _cognate_dg(cognates: pd.DataFrame, target: str) -> float | None:
    if cognates.empty:
        return None
    hit = cognates[cognates["name"].astype(str) == WINNERS.get(target, "")]
    if hit.empty:
        return None
    return float(hit.iloc[0]["dG_GB_kcal_mol"])


def _reference_dg(cognates: pd.DataFrame, target: str) -> float | None:
    if cognates.empty:
        return None
    ref_tag = REF_CHOICE.get(target, (None,))[0]
    if ref_tag is None or "target" not in cognates.columns:
        return None
    hit = cognates[cognates["target"].astype(str) == ref_tag]
    if hit.empty:
        return None
    return float(hit.iloc[0]["dG_GB_kcal_mol"])


def specificity_matrix() -> pd.DataFrame:
    """Tidy (target, condition) matrix of dG_GB + ddG vs cognate.

    One row per (target, condition) for condition in CONDITIONS. Columns:
    target, target_label, condition, condition_label, dG, ddg_vs_cognate,
    honest_se, caveat. Missing cells are dropped (e.g. a control not yet run).
    """
    cognates = load_cognates()
    controls = load_negcontrols()
    rows: list[dict] = []
    for target in TARGETS:
        cog = _cognate_dg(cognates, target)
        if cog is None:
            continue
        # cognate row (ddG 0 by definition)
        rows.append(_row(target, "cognate", cog, cog))
        # control rows
        if not controls.empty:
            sub = controls[controls["target"].astype(str) == target]
            for cond in ("alascan", "scrambled", "wrongtarget"):
                hit = sub[sub["control_type"].astype(str) == cond]
                if not hit.empty:
                    dg = float(hit.iloc[0]["dG_GB_kcal_mol"])
                    rows.append(_row(target, cond, dg, cog))
        # reference row
        ref = _reference_dg(cognates, target)
        if ref is not None:
            rows.append(_row(target, "reference", ref, cog))
    return pd.DataFrame(rows)


def _row(target: str, condition: str, dg: float, cognate: float) -> dict:
    return {
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "dG": round(dg, 2),
        "ddg_vs_cognate": round(dg - cognate, 2),
        "honest_se": HONEST_SE,
        "caveat": _CELL_CAVEATS.get((target, condition)),
    }


def design_vs_reference() -> pd.DataFrame:
    """Per-target headline: cognate design dG vs natural-partner dG.

    Columns: target, target_label, design_dG, reference_dG, reference_label,
    gap (reference - design; negative = natural partner is stronger), caveat.
    """
    cognates = load_cognates()
    rows: list[dict] = []
    for target in TARGETS:
        cog = _cognate_dg(cognates, target)
        ref = _reference_dg(cognates, target)
        if cog is None:
            continue
        ref_tag, ref_label, ref_caveat = REF_CHOICE.get(
            target, (None, None, None))
        rows.append({
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "design_dG": round(cog, 2),
            "reference_dG": round(ref, 2) if ref is not None else None,
            "reference_label": ref_label,
            "gap": round(ref - cog, 2) if ref is not None else None,
            "honest_se": HONEST_SE,
            "caveat": ref_caveat,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Queued / pending experiments (scaffolded on Slurm, results not yet in)
# --------------------------------------------------------------------------
def queued_experiments() -> list[dict]:
    """Status of the two designed-but-pending critical experiments.

    Each is detected by the presence of its result artefact; absent -> the
    experiment is queued and its outcome must NOT be claimed.
    """
    sens = BENCHMARKS_DIR / "cpepmatch_sensitivity_20260529"
    phc = BENCHMARKS_DIR / "ph_convergence_20260529"
    out = [
        {
            "name": "cPEPmatch fit-RMSD sensitivity",
            "question": "Does the geometric match quality propagate to final "
                        "design quality, or does ProteinHunter rescue poor "
                        "matches (cPEPmatch adds little)?",
            "dir": sens,
            "result_glob": "results_joined.csv",
            "has_results": bool(list(sens.glob("**/results_joined.csv")))
                           if sens.is_dir() else False,
        },
        {
            "name": "ProteinHunter convergence (100 cycles)",
            "question": "Does the refinement loop reach a self-consistent "
                        "basin, or is the reported best ipTM just max-of-N "
                        "inflation?",
            "dir": phc,
            "result_glob": "convergence.csv",
            "has_results": bool(list(phc.glob("**/convergence.csv")))
                           if phc.is_dir() else False,
        },
    ]
    return out


# --------------------------------------------------------------------------
# Altair plots (return alt.Chart; the notebook wraps in mo.ui.altair_chart)
# --------------------------------------------------------------------------
def plot_design_vs_reference(dvr: pd.DataFrame | None = None):
    """Grouped bar: cognate design dG vs natural-partner dG per target.

    Honest +-HONEST_SE error bars (NOT the MMPBSA SE). Returns alt.Chart.
    """
    import altair as alt

    if dvr is None:
        dvr = design_vs_reference()
    if dvr.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    long = dvr.melt(
        id_vars=["target", "target_label"],
        value_vars=["design_dG", "reference_dG"],
        var_name="kind", value_name="dG").dropna(subset=["dG"])
    long["kind"] = long["kind"].map(
        {"design_dG": "design (best)", "reference_dG": "natural partner"})
    long["lo"] = long["dG"] - HONEST_SE
    long["hi"] = long["dG"] + HONEST_SE

    base = alt.Chart(long).encode(
        x=alt.X("target_label:N", title=None,
                axis=alt.Axis(labelAngle=-15)),
        xOffset=alt.XOffset("kind:N"),
    )
    bars = base.mark_bar().encode(
        y=alt.Y("dG:Q", title="MM-GBSA dG  (kcal/mol, more negative = stronger)",
                scale=alt.Scale(zero=True)),
        color=alt.Color("kind:N", title=None,
                        scale=alt.Scale(
                            domain=["design (best)", "natural partner"],
                            range=["#4c78a8", "#e45756"]),
                        legend=alt.Legend(orient="top")),
        tooltip=["target_label", "kind", "dG"],
    )
    err = base.mark_errorbar().encode(
        y=alt.Y("lo:Q", title=""),
        y2="hi:Q",
        strokeWidth=alt.value(1.5),
    )
    return (bars + err).properties(
        width="container", height=360,
        title="Designed binders vs natural partners "
              "(+-5 kcal/mol honest error)")


def plot_specificity_heatmap(matrix: pd.DataFrame | None = None):
    """Heatmap of ddG vs cognate over (condition x target).

    Diverging color: large positive ddG (control lost binding) = good
    discrimination; near-zero (esp. scrambled/alascan) = composition-driven.
    Returns alt.Chart.
    """
    import altair as alt

    if matrix is None:
        matrix = specificity_matrix()
    if matrix.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    # Drop the cognate (ddG==0 by construction) and reference rows; the matrix
    # is about how controls degrade relative to the design.
    m = matrix[matrix["condition"].isin(
        ["alascan", "scrambled", "wrongtarget"])].copy()
    cond_order = ["alascan", "scrambled", "wrongtarget"]
    m["condition_label"] = pd.Categorical(
        m["condition_label"],
        [CONDITION_LABELS[c] for c in cond_order], ordered=True)

    base = alt.Chart(m).encode(
        x=alt.X("target_label:N", title=None,
                axis=alt.Axis(labelAngle=-15, orient="top")),
        y=alt.Y("condition_label:N", title=None, sort=None),
    )
    # No domainMid: 0 maps to the RED end so a near-zero ddG (control kept
    # binding = no discrimination, the CypA failure) is the most alarming
    # cell, and a large positive ddG (control lost binding = good) is green.
    heat = base.mark_rect().encode(
        color=alt.Color(
            "ddg_vs_cognate:Q",
            title="ddG vs cognate (+ = lost binding)",
            scale=alt.Scale(scheme="redyellowgreen", domain=[0, 64]),
            legend=alt.Legend(orient="bottom")),
        tooltip=["target_label", "condition_label", "dG",
                 "ddg_vs_cognate", "caveat"],
    )
    # White text on the saturated red (low) and dark-green (high) ends;
    # black on the light yellow/orange middle.
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("ddg_vs_cognate:Q", format="+.0f"),
        color=alt.condition(
            "datum.ddg_vs_cognate < 15 || datum.ddg_vs_cognate > 50",
            alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(
        width="container", height=210,
        title="Negative-control specificity: ddG vs the cognate design "
              "(green = discriminates, red = does not)")
