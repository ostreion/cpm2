"""Default pipeline configuration.

The inline dict is authoritative. Tweak values at the call site:

    config = make_default_config(PIPELINE_ROOT)
    config["cpepmatch"]["motif_size"] = 5
    config["boltz"]["iptm_threshold"] = 0.5

Validation is intentionally minimal: `_require_keys` checks top-level and
per-stage keys that the notebook relies on. Stage runners (BoltzPredictConfig,
ProteinHunterConfig) apply their own stricter validation via `.from_dict(...)`.
"""

from pathlib import Path

_REQUIRED_TOP = {"complex_pdb", "input_ligand_chain", "input_target_chain",
                 "cpepmatch", "boltz", "proteinhunter"}
_REQUIRED_CPEPMATCH = {"motif_size", "min_residues", "max_residues",
                       "fit_rmsd_threshold"}
_REQUIRED_BOLTZ = {"iptm_threshold", "plddt_threshold", "rmsd_threshold",
                   "cyclization_distance", "disulfide_distance"}
_REQUIRED_PH = {"num_designs", "num_cycles", "gpu_id", "iptm_threshold",
                "template_path"}


def _require_keys(d: dict, required: set, label: str) -> None:
    missing = required - set(d)
    if missing:
        raise KeyError(f"config[{label!r}] missing required keys: {sorted(missing)}")


def validate_config(config: dict) -> None:
    _require_keys(config, _REQUIRED_TOP, "<top>")
    _require_keys(config["cpepmatch"], _REQUIRED_CPEPMATCH, "cpepmatch")
    _require_keys(config["boltz"], _REQUIRED_BOLTZ, "boltz")
    _require_keys(config["proteinhunter"], _REQUIRED_PH, "proteinhunter")


def make_default_config(pipeline_root: Path, run_root: Path | None = None) -> dict:
    """Return the default CPM2 pipeline config dict.

    The dict is mutable; override fields after the call for a run.

    Args:
        pipeline_root: Path to the cpepmatch2 pipeline root directory.
        run_root: Optional per-run root. When given, intermediate path defaults
            (e.g. ``proteinhunter.template_path``) route through
            ``<run_root>/intermediate/...``. When None, falls back to the
            legacy global ``<pipeline_root>/data/intermediate/...`` so
            interactive notebook use without a run_id still works.
    """
    pipeline_root = Path(pipeline_root)
    data = pipeline_root / "data"
    input_dir = data / "input"
    if run_root is not None:
        intermediate_dir = Path(run_root) / "intermediate"
    else:
        intermediate_dir = data / "intermediate"

    config = {
        # Stage 0: PDB import
        "complex_pdb": input_dir / "2cfh.pdb",
        "input_ligand_chain": "C",
        "input_target_chain": "A",

        # Stage 1: cPEPmatch
        "cpepmatch": {
            # Run parameters
            "motif_size": 4,
            "consecutive": True,
            "interface_cutoff": 5,
            "frmsd_threshold": 0.3,
            "protein_specific_residues": None,
            "cyclization_type": None,
            "exclude_non_standard": False,
            # Post-run filter parameters
            "fit_rmsd_threshold": 0.7,
            "min_residues": 6,
            "max_residues": 22,
            "unique_sources": False,
            "mutated_only": True,
        },

        # Stage 2: Boltz2
        "boltz": {
            # boltz predict CLI args
            "devices": 1,
            "accelerator": "gpu",
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "step_scale": 1.638,
            "output_format": "mmcif",
            "num_workers": 2,
            "preprocessing_threads": None,
            "method": None,
            "cache": None,
            "checkpoint": None,
            "override": True,
            "write_full_pae": False,
            "write_full_pde": False,
            # MSA options
            "use_msa_server": True,
            "msa_server_url": "https://api.colabfold.com",
            "msa_pairing_strategy": "greedy",
            "max_msa_seqs": 8192,
            "subsample_msa": False,
            "num_subsampled_msa": 1024,
            # Inference options
            "use_potentials": False,
            "no_kernels": False,
            # Affinity options
            "affinity_mw_correction": False,
            "sampling_steps_affinity": 200,
            "diffusion_samples_affinity": 5,
            "affinity_checkpoint": None,
            # Detection thresholds
            "cyclization_distance": 1.5,
            "disulfide_distance": 2.5,
            # Filter thresholds (0-1 scale matching Boltz2 native output)
            "iptm_threshold": 0.7,
            "plddt_threshold": 0.7,
            "rmsd_threshold": 3.0,
        },

        # Stage 3: ProteinHunter refiner
        "proteinhunter": {
            "num_designs": 2,
            "num_cycles": 10,
            "gpu_id": 0,
            "seed": 42,
            "iptm_threshold": 0.7,
            # Model params (mirrored from refiner_boltz.ipynb cell 2)
            "diffuse_steps": 200,
            "recycling_steps": 3,
            "boltz_model_path": Path("~/.boltz/boltz2_conf.ckpt").expanduser(),
            "ccd_path": Path("~/.boltz/mols").expanduser(),
            "temperature": 0.1,
            "omit_AA": "C",
            "alanine_bias": False,
            "alanine_bias_start": -0.2,
            "alanine_bias_end": 0.0,
            "high_iptm_threshold": 0.8,
            "high_plddt_threshold": 0.8,
            # Template target (written by Stage 0, CIF)
            "template_path": str(intermediate_dir / "0_import" / "template.cif"),
            # When True, Boltz applies TemplateReferencePotential (CB-distance guidance
            # pulling target chain to template at every sampling step). When False,
            # template is a soft prior via the featurizer only.
            "template_force": False,
            "template_force_threshold": 1.0,
        },

        # Alignment visualisation
        "render_alignments": True,
        "alignment_mode": "best",
        "alignment_rmsd_cap": 3.0,
    }

    validate_config(config)
    return config
