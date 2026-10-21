"""Ion-site TDA descriptor wrapper around the ion-site pipeline.

The underlying script `_ion_site_tda_pipeline_v4_2_3.py` is the production
TDA pipeline. It includes three bug fixes over v4.2.1:

1. PI_total_energy: log1p compression (raw sums could reach ~25,000 for
   complex supercells, multiple orders of magnitude above other features).
2. Persistence entropy histogram: `density=False` (removes double normalization).
3. rho_all_dimless / rho_an_rel / rho_cat_rel: cap at 1e6 to prevent
   ~1e27 overflow when CIFs contain near-coincident atomic positions.

This module exposes a high-level function `compute_tda_descriptors` that
accepts a sequence of (material_id, ASE Atoms) pairs and returns a tidy
DataFrame keyed by material_id. The CLI behaviour of the original script
is preserved (callable via `python -m tcml_workflow.descriptors._ion_site_tda_pipeline_v4_2_3 ...`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def compute_tda_descriptors(
    structures: Iterable[tuple[str, "ase.Atoms"]],   # noqa: F821 (lazy import)
    *,
    supercell: tuple[int, int, int] = (2, 2, 2),
    scale: str = "nn",
    norm: str = "per_atom",
    include_h0: bool = True,
    anion_mode: str = "relative",
    anion_gap: float = 0.15,
    hydride_rule: bool = True,
    force_bipartition: bool = True,
    cif_workdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute ion-site TDA descriptors for a batch of structures.

    Parameters
    ----------
    structures : iterable of (material_id, ase.Atoms)
        Identified crystal structures.
    supercell : (nx, ny, nz)
        Replication used to build the periodic point cloud.
    scale, norm, include_h0, anion_mode, anion_gap, hydride_rule,
    force_bipartition : forwarded to the underlying pipeline.
    cif_workdir : optional Path
        Directory used to write temporary CIFs (one per structure).
        Defaults to a temp directory created via `tempfile.mkdtemp`.

    Returns
    -------
    pd.DataFrame
        One row per material with columns produced by the pipeline
        (rho_*, ALL/CAT/AN_H{0,1,2}_*, X_*).

    Notes
    -----
    The underlying script is CLI-driven (CIF directory + Excel sheet). We
    wrap it by serializing each ASE Atoms to a temporary CIF, writing a
    minimal Excel with `material id for cif` and a dummy κ column, then
    calling the pipeline in-process. This keeps the production code path
    in one place, without duplicating the algorithm here.
    """
    import re
    import tempfile
    from types import SimpleNamespace
    from ase.io import write as ase_write
    from openpyxl import Workbook

    from tcml_workflow.descriptors import _ion_site_tda_pipeline_v4_2_3 as core

    workdir = Path(cif_workdir or tempfile.mkdtemp(prefix="tcml_tda_"))
    cif_root = workdir / "cifs"
    cif_root.mkdir(parents=True, exist_ok=True)

    # The pipeline's main() matches CIFs by the digit-suffix of the material id
    # (legacy behaviour assuming numeric-only filenames from MP). Write CIFs
    # using just the digit substring so the matcher finds them.
    def _digits(s: str) -> str | None:
        m = re.search(r"(\d+)", str(s))
        return m.group(1) if m else None

    rows = []
    seen_digits = set()
    for mid, atoms in structures:
        d = _digits(mid)
        if d is None or d in seen_digits:
            # Skip mp_ids that share digit substrings with already-written
            # ones (e.g. mp-1 and mp-100 both reduce to first match "1");
            # The pipeline's matcher cannot disambiguate these. In practice Phonix
            # mp_ids have distinct digit suffixes so this branch is rare.
            continue
        seen_digits.add(d)
        cif_path = cif_root / f"{d}.cif"
        ase_write(cif_path, atoms, format="cif")
        rows.append({"material id for cif": mid, "thermal conductivity (W/mK)": np.nan})

    # Minimal driver Excel for the pipeline's CLI loader
    drv = workdir / "driver.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["material id for cif", "thermal conductivity (W/mK)"])
    for r in rows:
        # κ column must be numeric for core.main to accept the row; the value
        # is unused for descriptor computation (only kept for downstream tagging).
        ws.append([r["material id for cif"], 1.0])
    wb.save(drv)

    out_dir = workdir / "tda_out"
    out_dir.mkdir(exist_ok=True)

    # Build the args namespace the pipeline's main() expects (mirrors the script's argparse).
    args_ns = SimpleNamespace(
        excel=str(drv),
        sheet=0,
        cif_root=str(cif_root),
        kappa_col="thermal conductivity (W/mK)",
        id_col="material id for cif",
        out_dir=str(out_dir),
        limit=None,
        make_ordered=True,
        supercell=list(supercell),
        scale=scale,
        norm=norm,
        include_h0=include_h0,
        anion_mode=anion_mode,
        anion_gap=anion_gap,
        anion_fixed=["O", "N", "F", "Cl", "Br", "I", "S", "Se", "Te", "P", "C"],
        hydride_rule="on" if hydride_rule else "off",
        force_bipartition="on" if force_bipartition else "off",
    )
    core.main(args_ns)

    feats = pd.read_csv(out_dir / "ion_site_tda_features.csv")
    return feats.set_index("material_id" if "material_id" in feats.columns else feats.columns[0])
