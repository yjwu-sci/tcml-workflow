"""Predict κ for out-of-distribution candidate materials (with CIFs).

Intended for candidates that are absent from Phonix, for example a screened
structure that is later checked by a first-principles calculation.

Each candidate is supplied as (mp_id_or_label, CIF path). The script:
  1. Loads the final model produced by scripts/03_train.py
  2. Builds periodic + TDA descriptors for the candidate set
  3. Predicts log_k → κ in W/mK at 300 K
  4. Writes predictions.csv alongside DFT ground-truth columns (if provided)

Usage:
    python scripts/05_predict_candidates.py \\
        --model results/xgboost_periodic_tda_final_model.pkl \\
        --candidates data/candidates_example.csv \\
        --out_dir results/candidate_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from joblib import load


PERIODIC_PREFIX = ("coef_", "prop_")
TDA_PREFIX = ("rho_", "ALL_H", "CAT_H", "AN_H", "X_H")


def _sanitize(X: np.ndarray) -> np.ndarray:
    """Same numerical guard as scripts/03_train.py."""
    col_max = np.nanmax(np.abs(X), axis=0)
    for j in np.where(col_max > 1e6)[0]:
        X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
    return np.clip(X, -1e9, 1e9)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True,
                   help="Path to *_final_model.pkl from scripts/03_train.py")
    p.add_argument("--candidates", type=Path, required=True,
                   help="CSV with columns: label, formula, cif_path[, dft_kappa_total]")
    p.add_argument("--out_dir", type=Path, default=Path("results/candidate_predictions"))
    p.add_argument("--no_tda", action="store_true",
                   help="Skip TDA (must match the model's training setting)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(args.candidates)
    required = {"label", "formula"}
    if not args.no_tda:
        required = required | {"cif_path"}
    missing = required - set(cand.columns)
    if missing:
        raise ValueError(f"candidates CSV missing columns: {missing}")

    from tcml_workflow.descriptors import compute_periodic_dataframe
    labels = cand["label"].astype(str).tolist()
    print(f"Computing periodic descriptors for {len(cand):,} candidates...")
    periodic = compute_periodic_dataframe(cand["formula"].tolist(), index=labels, n_groups=17)
    periodic = periodic.reset_index().rename(columns={"index": "label"})

    if not args.no_tda:
        from tcml_workflow.descriptors import compute_tda_descriptors
        # The TDA wrapper matches structure files by the digits of each id, so pass
        # synthetic ids "1", "2", ... and restore the labels by row order.
        structures = [(str(i), ase_read(path)) for i, path in enumerate(cand["cif_path"], start=1)]
        print(f"Computing TDA descriptors for {len(cand):,} structures...")
        tda = compute_tda_descriptors(structures).reset_index(drop=True)
        tda.insert(0, "label", labels)
        feats = periodic.merge(tda, on="label", how="left")
    else:
        feats = periodic

    model = load(args.model)
    print(f"Loaded model: {args.model.name}")

    # same feature selection and column order as scripts/03_train.py
    prefixes = PERIODIC_PREFIX if args.no_tda else PERIODIC_PREFIX + TDA_PREFIX
    feat_cols = [c for c in feats.columns if c.startswith(prefixes)]
    X = _sanitize(feats[feat_cols].to_numpy(dtype=float))

    log_k_pred = model.predict(X)
    kappa_pred = np.power(10.0, log_k_pred)

    out = cand.copy()
    out["log_k_pred"] = log_k_pred
    out["kappa_pred_W_mK_300K"] = kappa_pred

    out_path = args.out_dir / "predictions.csv"
    out.to_csv(out_path, index=False)

    print(f"\n=== Predictions @ 300 K ===")
    print(out[["label", "formula", "kappa_pred_W_mK_300K"] +
              (["dft_kappa_total"] if "dft_kappa_total" in out.columns else [])
              ].to_string(index=False))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
