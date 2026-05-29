# CPM2 Snakemake DAG.
#
# Drives the same three-stage pipeline as notebooks/CPM2.ipynb but as a
# declarative DAG with per-rule conda envs and checkpoint-based dynamic
# match expansion. The notebook stays the canonical interactive entry point;
# this Snakefile is the canonical headless entry point.
#
# Invoke via:
#   snakemake --use-conda --cores 8 --config config_name=mdm2_p53_v1 run_id=<id>
# Or via the wrapper:
#   scripts/run.py mdm2_p53_v1
#
# Fan-out:
#   Stages 2a (build_boltz_yaml), 2b (boltz_predict), and 3
#   (proteinhunter_refine) all run per-match. This pays the ~30-60s
#   Boltz model-load warmup once per match — acceptable trade for a
#   simple DAG (every output explicitly tied to a rule, native retry
#   granularity, native fan-out). To amortise the warmup later, a per-
#   chunk rule with `output: directory(...)` + a downstream per-match
#   shim rule is the right shape; the previous sentinel-only attempt
#   broke DAG resolution.
#
# Conda env activation:
#   We do not use snakemake's --use-conda (which would recreate envs
#   from envs/*.yml under .snakemake/conda/<hash>/, slow and duplicative).
#   The runners themselves shell out via `conda run -n <env>` to the
#   existing named envs. The `conda:` directives in this file remain as
#   documentation of intent.
#
# Stage 0 + 1 + summary + archive are unchanged from Phase C/D.
# All pipeline state lives at data/runs/<run_id>/. Re-running snakemake on a
# partially complete run re-enters at the first missing target.

from pathlib import Path
import sys

PIPELINE_ROOT = Path(workflow.basedir).resolve()
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

# Absolute path to the cpm2 env's python. Snakemake spawns bash for each
# shell: directive, and bash uses base conda's python by default; we need
# our editable-installed cpm2 package, so resolve explicitly. Also: do
# NOT prepend our env's bin dir to PATH — that would break `conda run -n
# <env> python ...` invocations made by the runners (conda picks up the
# first `python` on PATH and ignores `-n`).
import os as _os
CPM2_PYTHON = _os.environ.get("CPM2_PYTHON") or sys.executable

CONFIG_NAME = config.get("config_name")
RUN_ID = config.get("run_id")
if not CONFIG_NAME or not RUN_ID:
    raise ValueError("snakemake --config config_name=<name> run_id=<id> required")

RUN_ROOT = PIPELINE_ROOT / "data" / "runs" / RUN_ID
INTERMEDIATE = RUN_ROOT / "intermediate"
OUTPUT = RUN_ROOT / "output"
CACHE = PIPELINE_ROOT / "data" / "cache"

ENVS = PIPELINE_ROOT / "envs"
HELPERS = PIPELINE_ROOT / "scripts" / "_snake_helpers"


wildcard_constraints:
    match = r"match[^/]+",


# ---------------------------------------------------------------------------
# Top-level target: archive sentinel (so a single `snakemake --cores N`
# invocation produces archives/<id>/ end-to-end). The summary CSV remains
# the upstream dependency.
# ---------------------------------------------------------------------------

ARCHIVE_SENTINEL = RUN_ROOT / "output" / ".archived"
MLFLOW_SENTINEL = RUN_ROOT / ".mlflow_run_id"
MLFLOW_FINALIZED = RUN_ROOT / ".mlflow_finalized"
BENCHMARK = RUN_ROOT / "benchmark"

rule all:
    input:
        MLFLOW_FINALIZED,


# ---------------------------------------------------------------------------
# Phase F2: init_run writes the manifest, mlflow_start opens the MLflow run.
# These run before stage0_import so MLflow has provenance even on early
# crashes. mlflow_finalize runs after archive and reads summary.csv +
# benchmark TSVs, logging final metrics + per-match series.
# ---------------------------------------------------------------------------

rule init_run:
    output:
        manifest = RUN_ROOT / "manifest.json",
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "init_run.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        config_name = CONFIG_NAME,
        run_root = str(RUN_ROOT),
        helper = str(HELPERS / "init_run.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root}"


rule mlflow_start:
    input:
        manifest = RUN_ROOT / "manifest.json",
    output:
        sentinel = MLFLOW_SENTINEL,
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "mlflow_start.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        config_name = CONFIG_NAME,
        run_root = str(RUN_ROOT),
        helper = str(HELPERS / "mlflow_start.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root}"


# ---------------------------------------------------------------------------
# Stage 0: import (manifest now written by init_run, see Phase F2)
# ---------------------------------------------------------------------------

rule stage0_import:
    input:
        manifest = RUN_ROOT / "manifest.json",
        mlflow_sentinel = MLFLOW_SENTINEL,
    output:
        processed = INTERMEDIATE / "0_import" / "processed.pdb",
        template = INTERMEDIATE / "0_import" / "template.cif",
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "stage0_import.tsv")
    params:
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        run_id = RUN_ID,
        pipeline_root = str(PIPELINE_ROOT),
    shell:
        # Inline driver: load config, run import_complex + extract_chain_to_cif.
        # Manifest is now written by the init_run rule (Phase F2).
        """
        "$CPM2_PYTHON" -c '
import sys
sys.path.insert(0, "{params.pipeline_root}/src")
from pathlib import Path
from cpm2.config_loader import load_config
from cpm2.utils.pdb_utils import import_complex, extract_chain_to_cif

run_root = Path("{params.run_root}")
config = load_config(Path("{params.pipeline_root}"), "{params.config_name}", run_root=run_root)
out = run_root / "intermediate" / "0_import"
out.mkdir(parents=True, exist_ok=True)
processed = out / "processed.pdb"
import_complex(config["complex_pdb"], config["input_ligand_chain"], config["input_target_chain"], processed)
extract_chain_to_cif(processed, "T", out / "template.cif")
'
        """


# ---------------------------------------------------------------------------
# Stage 1: cPEPmatch (checkpoint — number of matches is dynamic)
# ---------------------------------------------------------------------------

checkpoint stage1_cpepmatch:
    input:
        processed = INTERMEDIATE / "0_import" / "processed.pdb",
    output:
        match_list = INTERMEDIATE / "1_cpepmatch" / "match_list.txt",
        renamed_dir = directory(INTERMEDIATE / "1_cpepmatch_renamed"),
    conda:
        str(ENVS / "cpepmatch.yml")
    benchmark:
        str(BENCHMARK / "stage1_cpepmatch.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
    shell:
        """
        "$CPM2_PYTHON" -c '
import sys
sys.path.insert(0, "{params.pipeline_root}/src")
from pathlib import Path
from cpm2.config_loader import load_config
from cpm2.runners import cpepmatch
from cpm2.filters import cpepmatch_filters
from cpm2.utils.pdb_utils import rename_chain

run_root = Path("{params.run_root}")
config = load_config(Path("{params.pipeline_root}"), "{params.config_name}", run_root=run_root)
intermediate = run_root / "intermediate"
processed = intermediate / "0_import" / "processed.pdb"
out = intermediate / "1_cpepmatch"
out.mkdir(parents=True, exist_ok=True)

filter_keys = ("min_residues", "max_residues", "unique_sources", "mutated_only", "fit_rmsd_threshold")
run_params = {{k: v for k, v in config["cpepmatch"].items() if k not in filter_keys}}
matches = cpepmatch.run(pdb_file=processed, protein_chain="L", target_chain="T", output_dir=out, **run_params)
filtered = cpepmatch_filters.apply_all_filters(
    matches,
    min_residues=config["cpepmatch"]["min_residues"],
    max_residues=config["cpepmatch"]["max_residues"],
    unique_sources=config["cpepmatch"]["unique_sources"],
    mutated_only=config["cpepmatch"]["mutated_only"],
    max_fit_rmsd=config["cpepmatch"]["fit_rmsd_threshold"],
)
renamed_dir = intermediate / "1_cpepmatch_renamed"
renamed_dir.mkdir(parents=True, exist_ok=True)
for f in renamed_dir.glob("*.pdb"):
    f.unlink()
for m in filtered:
    fname = m.pdb_path.name
    src = out / fname
    dst = renamed_dir / fname
    old = "B" if "NotMutated" in fname else "A"
    rename_chain(src, old, "P", dst)
'
        """


def _filtered_match_names(wildcards):
    """Read filtered match stems from the cpepmatch checkpoint output dir.

    Sorted for deterministic ordering across re-runs.
    """
    co = checkpoints.stage1_cpepmatch.get(**wildcards).output.renamed_dir
    return sorted(p.stem for p in Path(co).glob("match*.pdb"))


# ---------------------------------------------------------------------------
# Stage 2a: per-match Boltz YAML construction
# ---------------------------------------------------------------------------

rule build_boltz_yaml:
    input:
        match_pdb = INTERMEDIATE / "1_cpepmatch_renamed" / "{match}.pdb",
        processed = INTERMEDIATE / "0_import" / "processed.pdb",
    output:
        yaml = INTERMEDIATE / "2_boltz" / "yaml_input" / "{match}.yaml",
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "build_boltz_yaml" / "{match}.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "build_boltz_yaml_match.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} {input.match_pdb} {output.yaml}"


# ---------------------------------------------------------------------------
# Stage 2b: per-match Boltz prediction
#
# One `boltz predict` subprocess per match. Pays ~30-60s model-load warmup
# per match, but the DAG is dead simple (every output explicitly tied to a
# rule) and Snakemake handles fan-out + retry granularity natively. To
# re-introduce chunking later for warmup amortisation, the right pattern
# is a per-chunk rule with `output: directory(...)` and a downstream
# per-match shim rule that depends on the chunk dir; the previous sentinel-
# based approach broke DAG resolution.
# ---------------------------------------------------------------------------

rule boltz_predict:
    input:
        yaml = INTERMEDIATE / "2_boltz" / "yaml_input" / "{match}.yaml",
        match_pdb = INTERMEDIATE / "1_cpepmatch_renamed" / "{match}.pdb",
    output:
        cif = INTERMEDIATE / "2_boltz" / "predictions" / "boltz_results_{match}"
              / "predictions" / "{match}" / "{match}_model_0.cif",
        confidence = INTERMEDIATE / "2_boltz" / "predictions" / "boltz_results_{match}"
                     / "predictions" / "{match}" / "confidence_{match}_model_0.json",
    conda:
        str(ENVS / "boltz.yml")
    benchmark:
        str(BENCHMARK / "boltz_predict" / "{match}.tsv")
    resources:
        gpu = 1
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "boltz_predict_chunk.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} {wildcards.match}"


# ---------------------------------------------------------------------------
# Stage 3: per-match ProteinHunter refinement
# ---------------------------------------------------------------------------

rule proteinhunter_refine:
    input:
        cif = INTERMEDIATE / "2_boltz" / "predictions" / "boltz_results_{match}"
              / "predictions" / "{match}" / "{match}_model_0.cif",
        confidence = INTERMEDIATE / "2_boltz" / "predictions" / "boltz_results_{match}"
                     / "predictions" / "{match}" / "confidence_{match}_model_0.json",
    output:
        results = INTERMEDIATE / "3_proteinhunter" / "{match}" / "refine_results.json",
    conda:
        str(ENVS / "proteinhunter.yml")
    benchmark:
        str(BENCHMARK / "proteinhunter_refine" / "{match}.tsv")
    resources:
        gpu = 1
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "proteinhunter_refine_match.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} {wildcards.match}"


# ---------------------------------------------------------------------------
# Stage 3 aggregation: collect all per-match refine_results into summary.csv
# ---------------------------------------------------------------------------

def _all_refine_results(wildcards):
    matches = _filtered_match_names(wildcards)
    return [str(INTERMEDIATE / "3_proteinhunter" / m / "refine_results.json") for m in matches]


rule collect_designs:
    input:
        _all_refine_results,
    output:
        csv = OUTPUT / "summary.csv",
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "collect_designs.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
    shell:
        """
        "$CPM2_PYTHON" -c '
import sys, json
from pathlib import Path
import pandas as pd

run_root = Path("{params.run_root}")
ph_out = run_root / "intermediate" / "3_proteinhunter"
all_designs = []
for d in sorted(ph_out.iterdir()):
    if not d.is_dir():
        continue
    rj = d / "refine_results.json"
    if rj.exists():
        all_designs.extend(json.loads(rj.read_text()))
df = pd.DataFrame(all_designs)
out = run_root / "output"
out.mkdir(parents=True, exist_ok=True)
if len(df):
    df = df.sort_values("iptm", ascending=False)
df.to_csv(out / "summary.csv", index=False)
'
        """


# ---------------------------------------------------------------------------
# Stage 3.5: alignment grid (default-on)
#
# Renders one PNG per (match, design): light-pink template (experimental
# input target) + blue→red refined target (Boltz-refolded + PH-refined,
# coloured by per-residue CA displacement vs template) + yellow CP sticks.
# Then assembles all PNGs into output/alignments/all_grid.png — rows sorted
# by best ipTM, cols by design number.
#
# Renders shell out to alignment_render.py inside the `pym` micromamba env
# (pymol2 lives there). The output is `directory(...)` because the per-design
# count is dynamic; the grid + per-design PNGs are produced together.
# ---------------------------------------------------------------------------

rule render_alignments:
    input:
        csv = OUTPUT / "summary.csv",
        template = INTERMEDIATE / "0_import" / "template.cif",
    output:
        alignments = directory(OUTPUT / "alignments"),
        grid = OUTPUT / "alignments" / "all_grid.png",
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "render_alignments.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "render_alignments.py"),
        mode = config.get("alignment_mode", "all"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} --mode {params.mode}"


# ---------------------------------------------------------------------------
# Archive: produce archives/<id>/ result card + sentinel under output/.
# Default invocation is the lightweight "result card" (no full intermediates
# tree); the heavy reproducibility data continues to live at data/runs/<id>/.
# ---------------------------------------------------------------------------

rule archive:
    input:
        csv = OUTPUT / "summary.csv",
        manifest = RUN_ROOT / "manifest.json",
        grid = OUTPUT / "alignments" / "all_grid.png",
    output:
        sentinel = ARCHIVE_SENTINEL,
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "archive.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "archive_run_cli.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} {output.sentinel}"


# ---------------------------------------------------------------------------
# Phase F2: mlflow_finalize closes the MLflow run with final metrics +
# per-match metric series + benchmark TSV scalars + slim-archive artifacts.
# ---------------------------------------------------------------------------

rule mlflow_finalize:
    input:
        archive_sentinel = ARCHIVE_SENTINEL,
        mlflow_sentinel = MLFLOW_SENTINEL,
        summary = OUTPUT / "summary.csv",
    output:
        sentinel = MLFLOW_FINALIZED,
    conda:
        str(ENVS / "cpm2.yml")
    benchmark:
        str(BENCHMARK / "mlflow_finalize.tsv")
    params:
        pipeline_root = str(PIPELINE_ROOT),
        run_root = str(RUN_ROOT),
        config_name = CONFIG_NAME,
        helper = str(HELPERS / "mlflow_finalize.py"),
    shell:
        "\"$CPM2_PYTHON\" {params.helper} {params.pipeline_root} {params.config_name} "
        "{params.run_root} {output.sentinel}"
