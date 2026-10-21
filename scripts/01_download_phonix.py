"""Download the Phonix-summary dataset from Hugging Face.

Phonix is CC-BY-4.0 (Ohnishi et al., npj Comput. Mater. 12, 150, 2026). This
script writes:

    <out_dir>/summary.parquet     (149 MB, primary file)
    <out_dir>/README.md
    <out_dir>/read_summary.py     (reference reader script from upstream)

Usage:
    python scripts/01_download_phonix.py --out_dir data/phonix
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/phonix-db/phonix-summary/resolve/main"
FILES = ["summary.parquet", "README.md", "read_summary.py"]


def download(url: str, dest: Path, insecure: bool = False) -> None:
    """Download a single URL to `dest` with progress printout."""
    ctx = ssl._create_unverified_context() if insecure else None  # noqa: S323

    print(f"  ↓ {dest.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tcml-workflow/0.1"})
    with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MB
        done = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = 100 * done / total
                print(f"    {done >> 20} / {total >> 20} MB ({pct:.0f}%)", end="\r", flush=True)
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out_dir", default="data/phonix", type=Path,
                   help="Destination directory (default: data/phonix)")
    p.add_argument("--insecure", action="store_true",
                   help="Skip SSL cert verification (use if you hit certifi errors)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Phonix-summary → {args.out_dir.resolve()}")
    for name in FILES:
        dest = args.out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ✓ {name} already present, skipping")
            continue
        try:
            download(f"{BASE_URL}/{name}", dest, insecure=args.insecure)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}", file=sys.stderr)
            print("    → retry with --insecure if this is a cert error", file=sys.stderr)
            return 1

    # Quick verification
    try:
        import pandas as pd
        df = pd.read_parquet(args.out_dir / "summary.parquet")
        print(f"\nVerified: {len(df):,} rows × {df.shape[1]} cols loaded successfully")
    except Exception as e:
        print(f"\n⚠ parquet readback failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
