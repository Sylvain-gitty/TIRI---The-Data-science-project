"""Export a git-committable "slim" copy of a feature-engineered dataset.

Why this exists
---------------
`data/processed/papers_fe.parquet` is 106 MB, and 101 MB of that is three raw
embedding blocks (`emb_qwen8b_*` 57 MB, `emb_jasper_*` 29 MB, `emb_qwen4b_*` 15 MB —
8,704 columns between them). The remaining 38 columns — labels, fold groups, the
Tier-1b lexical block, cosine-to-brief, and admissible metadata — weigh ~0.5 MB.

That asymmetry is the whole point. Dropping the `emb_*` block turns an
un-committable file into one small enough to ship in git, and what survives is
enough to reproduce this repo's *central* finding without any embedding at all:
brief-relative scalars beat a 384-dim embedding on leave-one-use-case-out
(0.642 vs 0.537 ROC-AUC, see `CONTEXT.md` §2). A reader who clones the repo can
re-run that, plus all of the EDA and the fold/validation design, from the
committed data alone.

What it does NOT give you: anything that needs the embedding block itself — the
PCA(50) pipeline shared by the baseline, CatBoost and ensemble notebooks. Those
need the full file, regenerated via `notebooks/04_feature_engineering.ipynb`
(which needs `data/processed/embeddings_cache/`, ~295 MB, and paid API / Modal
GPU calls). `README.md` §Data states that split explicitly rather than letting a
reader discover it at cell 1.

Usage
-----
    python scripts/export_slim_fe.py                       # defaults below
    python scripts/export_slim_fe.py --data data/processed/papers_fe_v2.parquet

The `--drop-prefix` flag is deliberately a *list*, so that when a future FE run
adds a fourth embedding model (or any other wide block), the slim export keeps
working without an edit here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_IN = Path("data/processed/papers_fe.parquet")
DEFAULT_OUT = Path("data/processed/papers_fe_slim.parquet")
DEFAULT_DROP_PREFIXES = ("emb_",)


def slim(df: pd.DataFrame, drop_prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Return `df` without any column starting with one of `drop_prefixes`."""
    keep = [c for c in df.columns if not c.startswith(tuple(drop_prefixes))]
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--drop-prefix",
        nargs="+",
        default=list(DEFAULT_DROP_PREFIXES),
        help="Column prefixes to drop (default: emb_). Wide blocks only.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    out = slim(df, tuple(args.drop_prefix))

    dropped = df.shape[1] - out.shape[1]
    print(f"in : {args.data}  {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"     dropped {dropped:,} cols matching {tuple(args.drop_prefix)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    in_mb = args.data.stat().st_size / 1e6
    out_mb = args.out.stat().st_size / 1e6
    print(f"out: {args.out}  {out.shape[0]:,} rows x {out.shape[1]:,} cols")
    print(f"     {in_mb:.1f} MB -> {out_mb:.1f} MB")
    print(f"     columns kept: {list(out.columns)}")


if __name__ == "__main__":
    main()
