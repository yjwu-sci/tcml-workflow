"""Phonix dataset loader with quality filters and structure reconstruction.

Phonix-summary parquet schema (Ohnishi et al., npj Comput. Mater. 12, 150, 2026):
- mp_id, formula, spg_number  : identification
- structure                   : JSON {cell, positions, symbols, pbc} (ASE-compatible)
- klat[W/mK]                  : lattice thermal conductivity at 300 K (training target)
- kp[W/mK], kc[W/mK]          : particle / coherence contributions (klat = kp + kc)
- fc2_error[%], fc3_error[%]  : force-constant fit quality (lower is better)
- qmesh, nac                  : computational settings (used for dedup)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ase import Atoms


@dataclass
class PhonixFilterConfig:
    """Quality filters for Phonix entries before training."""

    klat_min: float = 0.01      # W/mK; below this is fit failure
    klat_max: float = 2000.0    # W/mK; above this is unphysical
    fc3_error_max: float = 5.0  # %; third-order fit residual threshold
    require_nac: bool = False   # If True, keep only nac in {1,2,3}
    dedup_by: str = "mp_id"     # one row per mp_id (keep best fc3_error)


class PhonixLoader:
    """Load and filter the Phonix-summary parquet into an ML-ready DataFrame.

    Example
    -------
    >>> loader = PhonixLoader("data/phonix/summary.parquet")
    >>> df = loader.clean()
    >>> print(len(df))            # 6,716 unique materials
    >>> atoms = loader.get_structure(df.iloc[0])
    """

    REQUIRED_COLS = (
        "mp_id", "formula", "spg_number", "structure",
        "klat[W/mK]", "fc3_error[%]", "nac", "qmesh",
    )

    def __init__(self, parquet_path: str | Path, config: Optional[PhonixFilterConfig] = None):
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `python scripts/01_download_phonix.py` first."
            )
        self.config = config or PhonixFilterConfig()
        self._raw: Optional[pd.DataFrame] = None
        self._clean: Optional[pd.DataFrame] = None

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            self._raw = pd.read_parquet(self.path)
            missing = set(self.REQUIRED_COLS) - set(self._raw.columns)
            if missing:
                raise ValueError(f"Missing expected columns in parquet: {missing}")
        return self._raw

    def clean(self) -> pd.DataFrame:
        """Apply quality filters and dedup. Cached after first call."""
        if self._clean is not None:
            return self._clean

        df = self.raw.copy()
        cfg = self.config

        klat = df["klat[W/mK]"]
        mask = klat.between(cfg.klat_min, cfg.klat_max)
        mask &= df["fc3_error[%]"] < cfg.fc3_error_max
        if cfg.require_nac:
            mask &= df["nac"].isin([1.0, 2.0, 3.0])

        df = df.loc[mask].copy()

        # Dedup: keep the row with the lowest fc3_error per mp_id
        df = df.sort_values("fc3_error[%]", ascending=True).drop_duplicates(
            subset=cfg.dedup_by, keep="first"
        )
        df = df.reset_index(drop=True)

        # Pre-compute log10(klat) as the regression target
        df["log_k"] = np.log10(df["klat[W/mK]"].astype(float))

        self._clean = df
        return df

    @staticmethod
    def get_structure(row: pd.Series) -> Atoms:
        """Parse the `structure` JSON column into an ASE Atoms object.

        Phonix stores `positions` as FRACTIONAL coordinates (every value lies
        in [0, 1)), so they must be handed to ASE as `scaled_positions`.
        Passing them to `positions=`, which expects Cartesian coordinates in
        angstrom, collapses each structure into a sub-angstrom cluster in the
        corner of its own cell: cubic BN comes out with a 0.43 A
        nearest-neighbour distance instead of the correct 1.56 A.
        """
        s = json.loads(row["structure"])
        return Atoms(
            symbols=s["symbols"],
            scaled_positions=s["positions"],
            cell=s["cell"],
            pbc=s.get("pbc", [True, True, True]),
        )

    def train_test_split(
        self,
        test_frac: float = 0.2,
        random_state: int = 0,
        stratify_log_k_bins: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Random split with optional log_k stratification.

        Stratification preserves the distribution of κ across train and test —
        important because the dataset is skewed toward low κ (4,811 / 6,716 ≤ 5 W/m·K).
        Falls back to plain random split if the dataset is too small to populate
        the requested number of strata with ≥ 2 samples each.
        """
        from sklearn.model_selection import train_test_split

        df = self.clean()
        strata = None
        if stratify_log_k_bins and stratify_log_k_bins > 1 and len(df) >= 2 * stratify_log_k_bins:
            candidate = pd.qcut(
                df["log_k"], q=stratify_log_k_bins, labels=False, duplicates="drop"
            )
            # Skip stratification if any resulting bin has < 2 members (sklearn requirement)
            if candidate.value_counts().min() >= 2:
                strata = candidate

        train_df, test_df = train_test_split(
            df, test_size=test_frac, random_state=random_state, stratify=strata,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def summary(self) -> dict:
        """Return a small stats dict for logging."""
        df = self.clean()
        klat = df["klat[W/mK]"]
        return {
            "n_unique_materials": len(df),
            "n_low_kappa_5": int((klat <= 5).sum()),
            "n_high_kappa_50": int((klat > 50).sum()),
            "klat_median": float(klat.median()),
            "klat_range": [float(klat.min()), float(klat.max())],
            "log_k_range": [float(df["log_k"].min()), float(df["log_k"].max())],
        }
