"""Compute periodic + TDA descriptors for the Phonix training set.

Reads `<phonix_dir>/summary.parquet`, applies quality filters via
PhonixLoader, then computes descriptors for every retained material:

  output/
    train.parquet            ← features X + target log_k, one row per mp_id
    periodic_features.csv    ← 34-dim periodic
    tda_features.csv         ← 107-dim TDA
    descriptor_log.json      ← summary stats + failed entries

Usage:
    python scripts/02_compute_descriptors.py \\
        --phonix data/phonix/summary.parquet \\
        --out_dir data/processed \\
        --tda            # omit to skip TDA computation (periodic only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tcml_workflow import PhonixLoader
from tcml_workflow.data_loader import PhonixFilterConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phonix", type=Path, required=True,
                   help="Path to Phonix summary.parquet")
    p.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--tda", action="store_true",
                   help="Compute TDA descriptors (requires gudhi installed)")
    p.add_argument("--fc3_error_max", type=float, default=5.0)
    p.add_argument("--klat_max", type=float, default=2000.0)
    p.add_argument("--max_materials", type=int, default=None,
                   help="Limit for smoke testing (default: all)")
    p.add_argument("--n_groups", type=int, default=17, choices=(17, 18),
                   help="Periodic-table groups: 17 (default, 34-dim) or 18 (36-dim)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + filter Phonix
    cfg = PhonixFilterConfig(klat_max=args.klat_max, fc3_error_max=args.fc3_error_max)
    loader = PhonixLoader(args.phonix, config=cfg)
    df = loader.clean()
    if args.max_materials:
        df = df.head(args.max_materials)
    summary = loader.summary()
    print(f"Phonix clean: {summary}")

    # 2. Periodic descriptors
    from tcml_workflow.descriptors import compute_periodic_dataframe
    print(f"\nComputing periodic descriptors for {len(df):,} formulas (n_groups={args.n_groups})...")
    periodic = compute_periodic_dataframe(
        df["formula"].tolist(), n_groups=args.n_groups, index=df["mp_id"].tolist(),
    )
    periodic = periodic.reset_index().rename(columns={"index": "mp_id"})
    periodic.to_csv(args.out_dir / "periodic_features.csv", index=False)
    print(f"  → {args.out_dir / 'periodic_features.csv'}")

    # 3. TDA descriptors (optional, requires gudhi)
    if args.tda:
        from tcml_workflow.descriptors import compute_tda_descriptors
        print(f"\nComputing TDA descriptors for {len(df):,} structures...")
        structures = [(row["mp_id"], PhonixLoader.get_structure(row)) for _, row in df.iterrows()]
        tda = compute_tda_descriptors(structures)
        tda = tda.reset_index().rename(columns={tda.index.name or "index": "mp_id"})
        tda.to_csv(args.out_dir / "tda_features.csv", index=False)
        print(f"  → {args.out_dir / 'tda_features.csv'}")
    else:
        tda = pd.DataFrame({"mp_id": df["mp_id"].values})

    # 4. Merge into the training table
    train = df[["mp_id", "formula", "spg_number", "klat[W/mK]", "log_k"]].copy()
    if periodic.shape[1] > 1:
        train = train.merge(periodic, on="mp_id", how="left")
    if tda.shape[1] > 1:
        train = train.merge(tda, on="mp_id", how="left")
    train.to_parquet(args.out_dir / "train.parquet", index=False)
    print(f"\nWrote training table: {args.out_dir / 'train.parquet'} "
          f"({len(train):,} rows × {train.shape[1]} cols)")

    with open(args.out_dir / "descriptor_log.json", "w") as f:
        json.dump({
            "phonix_summary": summary,
            "n_materials_processed": len(train),
            "has_periodic": periodic.shape[1] > 1,
            "has_tda": tda.shape[1] > 1,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
