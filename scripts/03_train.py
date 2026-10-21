"""Train one algorithm (XGBoost / SVR / GPR) with Nested-CV.

- one entry point for the three algorithms (--algorithm)
- SVR and GPR are wrapped with a StandardScaler (tcml_workflow.pipeline)
- hyperparameter search ranges are defined in tcml_workflow.models
- input and output paths come from the command line
- metrics are written to JSON
- two descriptor settings: --no_tda (periodic only) or default (periodic + TDA)

Usage examples:
    python scripts/03_train.py --algorithm XGBoost --train data/processed/train.parquet
    python scripts/03_train.py --algorithm SVR --train data/processed/train.parquet --no_tda
    python scripts/03_train.py --algorithm GPR --train data/processed/train.parquet --n_iter 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcml_workflow import ModelConfig, NestedCVPipeline


PERIODIC_PREFIX = ("coef_", "prop_")
TDA_PREFIX = ("rho_", "ALL_H", "CAT_H", "AN_H", "X_H")


def select_feature_columns(df: pd.DataFrame, include_tda: bool) -> list[str]:
    cols = ["Temperature"] if "Temperature" in df.columns else []
    for c in df.columns:
        if c.startswith(PERIODIC_PREFIX):
            cols.append(c)
        elif include_tda and c.startswith(TDA_PREFIX):
            cols.append(c)
    return cols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--algorithm", required=True, choices=("XGBoost", "SVR", "GPR"))
    p.add_argument("--train", type=Path, required=True,
                   help="Path to train.parquet (from scripts/02_compute_descriptors.py)")
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    p.add_argument("--no_tda", action="store_true",
                   help="Train on periodic + Temperature only (no TDA features)")
    p.add_argument("--n_iter", type=int, default=None,
                   help="Override RandomizedSearchCV iterations (default: XGB 1000, others 200)")
    p.add_argument("--subsample", type=int, default=None,
                   help="Train on a stratified log_k subsample of this size (use for GPR — full Phonix "
                        "is intractable due to O(n³) kernel inversion)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.train)
    if "log_k" not in df.columns:
        raise ValueError("train.parquet must contain 'log_k' column")

    include_tda = not args.no_tda
    feat_cols = select_feature_columns(df, include_tda)
    if not feat_cols:
        raise ValueError("No feature columns found. Did you run scripts/02_compute_descriptors.py?")

    # Auto-detect if TDA features are actually present in the data; otherwise
    # the tag would be misleading ("periodic_tda" with zero TDA cols present).
    has_tda_cols = any(c.startswith(TDA_PREFIX) for c in feat_cols)
    if include_tda and not has_tda_cols:
        print("  ⓘ --no_tda not set, but no TDA columns found in training data — labelling as periodic-only.")
        include_tda = False
    print(f"Features ({'periodic + TDA' if include_tda else 'periodic only'}): {len(feat_cols)} cols")

    # Drop rows with any NaN in features or target
    sub = df[feat_cols + ["log_k", "mp_id"]].dropna()
    if len(sub) < len(df):
        print(f"  Dropped {len(df) - len(sub):,} rows with NaN in features/target")

    # Optional stratified subsample (essential for GPR; recommended for SVR on n > 5k)
    if args.subsample is not None and args.subsample < len(sub):
        from sklearn.model_selection import train_test_split
        strata = pd.qcut(sub["log_k"], q=5, labels=False, duplicates="drop")
        keep, _ = train_test_split(
            sub, train_size=args.subsample, random_state=args.seed, stratify=strata,
        )
        sub = keep.reset_index(drop=True)
        print(f"  Stratified subsample: {len(sub):,} samples (log_k-binned, seed={args.seed})")

    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub["log_k"].to_numpy(dtype=float)

    # Sanitize features: a small number of pathological structures (degenerate
    # AlphaComplex geometry in the pipeline) produce per-feature values
    # up to ~1e34, which trip XGBoost's "value too large" guard and inflate
    # variance in distance-based learners. Apply a signed log1p compression to
    # any column whose typical scale exceeds 1e6, then clip to [-1e9, 1e9] as a
    # safety net. Periodic descriptors (coefs/props in [0,1] / [Z]) are left
    # untouched by both transforms.
    col_max = np.nanmax(np.abs(X), axis=0)
    heavy = np.where(col_max > 1e6)[0]
    if len(heavy) > 0:
        for j in heavy:
            v = X[:, j]
            X[:, j] = np.sign(v) * np.log1p(np.abs(v))
        print(f"  log1p-compressed {len(heavy)} heavy-tailed feature columns "
              f"(max before: {col_max.max():.2e})")
    X = np.clip(X, -1e9, 1e9)
    print(f"  Feature matrix after sanitization: shape={X.shape}, "
          f"max_abs={np.max(np.abs(X)):.3e}")

    cfg_kwargs = {"algorithm": args.algorithm, "random_state": args.seed}
    if args.n_iter is not None:
        cfg_kwargs["n_iter_search"] = args.n_iter
    config = ModelConfig(**cfg_kwargs)

    pipe = NestedCVPipeline(config)
    metrics = pipe.fit(X, y)

    tag = f"{args.algorithm.lower()}_{'periodic_tda' if include_tda else 'periodic'}"
    if args.subsample is not None:
        tag += f"_n{args.subsample}"
    pipe.save(args.out_dir, tag)

    print(f"\n=== Final mean metrics ({args.algorithm}, {'+TDA' if include_tda else 'no TDA'}) ===")
    print(f"  R     = {metrics.r_mean:.4f} ± {metrics.r_std:.4f}")
    print(f"  R²    = {metrics.r2_mean:.4f} ± {metrics.r2_std:.4f}")
    print(f"  RMSE  = {metrics.rmse_mean:.4f} ± {metrics.rmse_std:.4f}")
    print(f"  Best outer fold = {metrics.best_fold_index + 1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
