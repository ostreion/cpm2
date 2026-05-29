"""mmCIF fetch + load helpers."""

import urllib.request
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def download_mmcif(pdb_id: str, cache_dir: Path) -> Path | None:
    """Download an mmCIF from RCSB with on-disk caching.

    Returns the cached path, or None if the download fails.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cif_path = cache_dir / f"{pdb_id.lower()}.cif"
    if cif_path.exists():
        return cif_path
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    try:
        urllib.request.urlretrieve(url, cif_path)
        return cif_path
    except Exception as e:
        print(f"    WARNING: Failed to download {url}: {e}")
        return None


def load_mmcif(path: Path) -> dict:
    """Parse an mmCIF into a flat dict via BioPython's MMCIF2Dict."""
    return MMCIF2Dict(str(path))
