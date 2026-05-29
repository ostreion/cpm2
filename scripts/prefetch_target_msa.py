"""Pre-fetch a Boltz-compatible MSA for a target protein chain.

Runs in the `boltz` conda env (where boltz is installed). The output is a
.csv file Boltz natively understands (key,sequence) so ProteinHunter's
internal Boltz call can pass it as the `msa:` value in the data dict.

Usage (via micromamba/conda):
    micromamba run -n boltz python scripts/prefetch_target_msa.py \
        --pdb data/input/1ycr.pdb \
        --chain A \
        --target-id mdm2 \
        --out data/msa_cache/mdm2.csv

The script is idempotent: if the output file already exists, it is left
alone unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _extract_chain_sequence(pdb_path: Path, chain: str) -> str:
    """Single-letter sequence of one chain of a PDB file (CA-only, no HETATM)."""
    aa_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seq_parts: dict[int, str] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        resname = line[17:20].strip().upper()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if resname in aa_map and resnum not in seq_parts:
            seq_parts[resnum] = aa_map[resname]
    return "".join(seq_parts[k] for k in sorted(seq_parts))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, required=True, help="Input PDB file")
    p.add_argument("--chain", required=True, help="Chain ID to extract sequence from")
    p.add_argument("--target-id", required=True, help="Identifier used inside Boltz; arbitrary")
    p.add_argument("--out", type=Path, required=True, help="Output .csv path")
    p.add_argument("--msa-server-url", default="https://api.colabfold.com")
    p.add_argument("--msa-pairing-strategy", default="greedy")
    p.add_argument("--force", action="store_true", help="Re-fetch even if --out already exists")
    args = p.parse_args()

    if args.out.exists() and not args.force:
        print(f"OK: {args.out} already exists; pass --force to re-fetch")
        return 0

    seq = _extract_chain_sequence(args.pdb, args.chain)
    if not seq:
        print(f"ERROR: no CA residues found for chain {args.chain} in {args.pdb}", file=sys.stderr)
        return 1
    print(f"Target sequence (chain {args.chain}, {len(seq)} aa): {seq}")

    try:
        from boltz.main import compute_msa
    except ImportError as e:
        print(f"ERROR: cannot import boltz.main.compute_msa - is the boltz env active? ({e})",
              file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.out.parent / f"_msa_work_{args.target_id}"
    work_dir.mkdir(exist_ok=True)

    print(f"Calling Boltz compute_msa (server: {args.msa_server_url})")
    compute_msa(
        data={args.target_id: seq},
        target_id=args.target_id,
        msa_dir=work_dir,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
    )

    produced = work_dir / f"{args.target_id}.csv"
    if not produced.exists():
        print(f"ERROR: expected MSA at {produced} but it was not written", file=sys.stderr)
        return 3
    produced.replace(args.out)
    print(f"OK: wrote {args.out}")

    # Also dump a Boltz-pre-processed .npz alongside the .csv, since PH's
    # vendored Boltz wrapper expects pre-processed numpy data and calls
    # np.load() directly on whatever path it receives. Stage 2 (Boltz CLI)
    # consumes the .csv; Stage 3 (PH) consumes the sibling .npz.
    try:
        from boltz.data.parse.csv import parse_csv
    except ImportError as e:
        print(f"WARN: cannot import boltz.data.parse.csv to build .npz: {e}", file=sys.stderr)
        return 0
    npz_path = args.out.with_suffix(".npz")
    msa = parse_csv(args.out, max_seqs=8192)
    msa.dump(npz_path)
    print(f"OK: wrote {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
