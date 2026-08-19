"""Does the induced brief survive into the per-silo ensemble?

`wf_llm_benchset_a_findings.md` §5b measured the induced rule set lifting the BM25/overlap
block by **+0.065 AUC** in isolation, as much as it lifted an LLM reader, and the cosine by
nothing. The block is one branch of a two-branch ensemble that also sees 4,608 raw embedding
dimensions. So the question this answers is not "is the lexical block better" — that is
settled — but **is the improvement redundant with what the embeddings already encode.**

Three variants, identical in width (4,625 columns) so nothing is confounded with feature count:

    supplied      supplied lex + supplied cos + metadata + Jasper/Qwen3-4B embeddings
    induced_lex   induced  lex + supplied cos + metadata + embeddings
    induced_all   induced  lex + induced  cos + metadata + embeddings

`induced_lex` carries the hypothesis; `induced_all` is the honest end-to-end swap, and the
gap between them is the cosine's contribution, measured at ±0.005 in isolation.

Per-silo, never pooled (`CONTEXT.md` §1), grouped by `first_author`, `StratifiedGroupKFold`
5-fold, repeated across seeds because the repo's measured seed-to-seed sd is ~0.010 ROC-AUC
and gaps under ~0.03 are not established.

Fitted locally, not on Modal
----------------------------
`run_ensemble_candidate.py` warns CatBoost can be pathologically slow here — "a known
thread-oversubscription issue on some Apple Silicon Macs when CatBoost's default
thread_count=-1 tries to use every core". Measured on this feature table: with an explicit
`thread_count`, the largest silo (1,660 rows x 4,625 columns, iterations=50, depth=4) fits in
**15.5s**. The pathology is the autodetection, not the workload, and `catboost_fn` has always
taken the parameter — Modal was passing it all along. So this runs locally and free.

Blending is a plain 50/50 average, not a learned weight: prediction-level stacking has tied
plain averaging twice in this repo (`CONTEXT.md` §6) and a fitted weight would add a second
fitted quantity to an experiment about a first.

Usage:
    python scripts/run_setA_brief_ensemble.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import within_silo_oof  # noqa: E402
from run_ensemble_candidate import catboost_fn, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FEATURES = REPO / "data" / "processed" / "setA_ensemble_features.parquet"
OUT_JSON = REPO / "reports" / "wf_llm_setA_brief_ensemble_oof.json"

META_COLS = ["year", "n_authors", "citation_count"]
THREAD_COUNT = 8          # explicit, never -1 — see the module docstring
ITERATIONS, DEPTH = 50, 4  # the screening setting run_ensemble_candidate.py documents


def variant_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    emb = sorted(c for c in df.columns if c.startswith("emb_"))
    lex_sup = sorted(c for c in df.columns if c.startswith("lex_"))
    lex_ind = sorted(c for c in df.columns if c.startswith("lexind_"))
    cos_sup = sorted(c for c in df.columns
                     if c.startswith(("cos_brief_", "rank_cos_brief_")))
    cos_ind = sorted(c for c in df.columns if c.startswith("cosind_"))
    variants = {
        "supplied": lex_sup + cos_sup + META_COLS + emb,
        "induced_lex": lex_ind + cos_sup + META_COLS + emb,
        "induced_all": lex_ind + cos_ind + META_COLS + emb,
    }
    widths = {k: len(v) for k, v in variants.items()}
    if len(set(widths.values())) != 1:
        raise RuntimeError(f"variants differ in width, the comparison would be confounded: "
                           f"{widths}")
    return variants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="+",
                    default=["supplied", "induced_lex", "induced_all"])
    ap.add_argument("--branches", nargs="+", default=["catboost", "logreg"])
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES)
    variants = variant_cols(df)
    y = df["y"].to_numpy().astype(int)
    use_case, groups = df["use_case_key"], df["first_author"]
    print(f"{len(df):,} rows / {use_case.nunique()} silos / {y.sum():,} positive · "
          f"{len(next(iter(variants.values())))} features per variant")

    results: dict[str, dict] = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    for variant in args.variants:
        cols = variants[variant]
        X = df[cols]
        for branch in args.branches:
            for seed in range(args.seeds):
                key = f"{variant}|{branch}|{seed}"
                if key in results:
                    print(f"  {key}: cached")
                    continue
                model_fn = (catboost_fn(cols, thread_count=THREAD_COUNT,
                                        iterations=ITERATIONS, depth=DEPTH)
                            if branch == "catboost" else logreg_fn(cols))
                t0 = time.time()
                oof = within_silo_oof(X, y, use_case, groups, seed, model_fn)
                results[key] = {uc: arr.tolist() for uc, arr in oof.items()}
                print(f"  {key}: {time.time() - t0:.0f}s", flush=True)
                # Written after every cell: an hour of fits should never be lost to a
                # crash in the next one, and re-running skips what is already on disk.
                OUT_JSON.write_text(json.dumps(results), encoding="utf-8")

    print(f"\nwrote {OUT_JSON}  ({len(results)} cells)")


if __name__ == "__main__":
    main()
