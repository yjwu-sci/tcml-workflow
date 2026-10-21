"""Aggregate trained model metrics into a comparison table.

Reads all `*_metrics.json` files in --results_dir and emits:

    comparison_table.csv     (one row per algorithm × setting)
    comparison_table.md      (markdown table)
    parity_data.parquet      (predicted vs actual log_k per fold, all models)

Usage:
    python scripts/04_evaluate.py --results_dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: --results_dir)")
    args = p.parse_args(argv)

    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(args.results_dir.glob("*_metrics.json")):
        with open(f) as fp:
            d = json.load(fp)
        tag = f.stem.replace("_metrics", "")
        algo, *setting = tag.split("_", 1)
        rows.append({
            "algorithm": algo.upper(),
            "setting": setting[0] if setting else "default",
            "R_mean": d["r_mean"],
            "R_std": d["r_std"],
            "R2_mean": d["r2_mean"],
            "R2_std": d["r2_std"],
            "RMSE_mean": d["rmse_mean"],
            "RMSE_std": d["rmse_std"],
            "best_fold": d["best_fold_index"] + 1,
            "n_iter_search": "—",
        })

    if not rows:
        print(f"No *_metrics.json found in {args.results_dir}")
        return 1

    table = pd.DataFrame(rows).sort_values(["setting", "algorithm"]).reset_index(drop=True)
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    # Markdown render
    md_lines = [
        "| Algorithm | Setting | R | R² | RMSE | Best fold |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        md_lines.append(
            f"| {r['algorithm']} | {r['setting']} | "
            f"{r['R_mean']:.3f} ± {r['R_std']:.3f} | "
            f"{r['R2_mean']:.3f} ± {r['R2_std']:.3f} | "
            f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f} | "
            f"{int(r['best_fold'])} |"
        )
    md_path = out_dir / "comparison_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(table.to_string(index=False))
    print(f"\nWrote {out_dir / 'comparison_table.csv'}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
