"""LLM screening bake-off runner - can a prompted open-weights LLM beat the ensemble on F2?

Plan, stop conditions and cost model: `reports/wf_llm_screening_plan.md`.

Scope of the pilot this script runs: TIRI's own 1,848 labelled rows only. The 28-collection
benchset corpus is deliberately deferred - it is where a *positive* result would have to be
confirmed, because CONTEXT.md §4 is explicit that our 26-77%-positive pools are "a poor
surface for judging a recall-oriented system". That asymmetry is the point of running here
first: this pilot can kill the idea cheaply and confidently, but it cannot confirm it.

Data
----
Text (`title`/`abstract`) and the use-case brief columns live in `papers_combined.parquet`;
labels, folds and the engineered features live in `papers_fe_slim.parquet`. They join
cleanly on `(paper_id, use_case_key)` - 1,848 rows, unique both sides, no nulls. The slim
file is the row-order authority because `reports/wf_ensemble_v2_hparam_oof.json` holds the
tuned CatBoost out-of-fold probabilities aligned to it, which is what makes the
LLM-vs-ensemble comparison (and the branch-correlation question) a zero-refit analysis.

Metrics
-------
Imported from `ensemble_eval_utils`/`analyze_ensemble_v2_experiments`, never re-implemented -
`f2_optimal` in particular, so the LLM is scored by exactly the rule that produced the
ensemble's published 0.893. Note that rule is an *oracle*: it sweeps 91 thresholds and keeps
the best on the same predictions it selected on. It is reported for both sides or neither.
`f2_at_own` (the model's own hard verdict, no threshold tuning at all) is reported alongside
it, and is the honest operating point.

Usage
-----
    python scripts/run_llm_screening.py --smoke          # ~360 calls, ~$0.16
    python scripts/run_llm_screening.py --stage grid     # full pilot grid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_pipeline_utils import PROMPT_VARIANTS, score_frame  # noqa: E402

try:  # optional: repo convention is a gitignored .env at the root
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

SLIM = Path("data/processed/papers_fe_slim.parquet")
COMBINED = Path("data/processed/papers_combined.parquet")
OUT_DIR = Path("reports")

TEXT_COLS = [
    "title", "abstract", "use_case_name", "objective", "problem_statement",
    "terms_must_include", "terms_nice_to_have", "terms_exclude",
    "domain_industry", "domain_application", "domain_technology_focus",
]

# Four slots bracketing 20B -> 397B across four families, not a survey. The 20B-vs-397B
# gap is the informative one: if the ceiling model cannot clear the bar no open-weights
# model will, and if the floor model matches it then scale is not the lever and the
# shippable choice is the one that costs $0.08 per pass.
PILOT_MODELS = [
    "openai/gpt-oss-20b",
    "google/gemma-4-31b-it",
    "nvidia/nemotron-3-super-120b-a12b",
    "qwen/qwen3.5-397b-a17b",
]
PILOT_VARIANTS = ["P1", "P2", "P4"]


def load_pilot_frame() -> pd.DataFrame:
    """Slim (row-order authority, carries `y`) left-joined to combined (text + brief)."""
    slim = pd.read_parquet(SLIM)
    combined = pd.read_parquet(COMBINED, columns=["paper_id", "use_case_key", *TEXT_COLS])
    df = slim[["paper_id", "use_case_key", "y"]].merge(
        combined, on=["paper_id", "use_case_key"], how="left", validate="one_to_one"
    )
    if len(df) != len(slim):
        raise RuntimeError(f"join changed row count: {len(slim)} -> {len(df)}")
    if df["title"].isna().any():
        raise RuntimeError("join produced null titles - schema drift, do not proceed")
    return df


def stratified_sample(df: pd.DataFrame, per_use_case: int, seed: int = 0) -> pd.DataFrame:
    """`per_use_case` rows from each use case, stratified on the label.

    Stratified rather than random because at 26-77% prevalence a small random draw from
    Soil Microbiome (26% positive) can come back with almost no positives, and a smoke
    test that never shows the model a positive tells you nothing about whether it can
    find one.
    """
    parts = []
    for _, grp in df.groupby("use_case_key"):
        take = min(per_use_case // 2, grp["y"].sum(), (grp["y"] == 0).sum())
        take = max(int(take), 1)
        for label in (1, 0):
            pool = grp[grp["y"] == label]
            parts.append(pool.sample(n=min(take, len(pool)), random_state=seed))
    return pd.concat(parts).sort_index().reset_index(drop=True)


def summarise(res: pd.DataFrame) -> dict:
    """Compliance, cost and latency for one cell. Deliberately reported before any
    accuracy metric: a model that cannot return parseable JSON is disqualified on
    engineering grounds regardless of how well the rows it did parse happened to score.
    """
    n = len(res)
    errors = int(res["error"].notna().sum())
    parsed = int(res["parsed"].sum())
    scored = int(res["score"].notna().sum())
    verdict = int(res["pred"].notna().sum())
    cost = float(pd.to_numeric(res["cost"], errors="coerce").fillna(0).sum())
    return {
        "n": n,
        "http_errors": errors,
        "parse_rate": round(parsed / n, 3) if n else 0.0,
        "score_rate": round(scored / n, 3) if n else 0.0,
        "verdict_rate": round(verdict / n, 3) if n else 0.0,
        "truncated": int((res["finish_reason"] == "length").sum()),
        "insufficient": int(pd.to_numeric(res["insufficient"], errors="coerce").fillna(0).sum()),
        "param_drops": int((res["params_dropped"].fillna("") != "").sum()),
        "median_latency_s": float(pd.to_numeric(res["latency_s"], errors="coerce").median()),
        "median_prompt_tok": float(pd.to_numeric(res["prompt_tokens"], errors="coerce").median()),
        "median_completion_tok": float(
            pd.to_numeric(res["completion_tokens"], errors="coerce").median()
        ),
        "cost_usd": round(cost, 4),
        "providers": ",".join(sorted(res["provider"].dropna().unique().tolist())[:4]),
    }


def run_cells(df: pd.DataFrame, models: list[str], variants: list[str], concurrency: int,
              out_stem: str) -> pd.DataFrame:
    all_res, summary = [], []
    for model in models:
        for variant in variants:
            print(f"\n>>> {model}  {variant}  ({len(df)} rows)", flush=True)
            res = score_frame(df, model=model, variant=variant, concurrency=concurrency)
            all_res.append(res)
            row = {"model": model, "variant": variant, **summarise(res)}
            summary.append(row)
            print(f"    parse={row['parse_rate']:.0%} score={row['score_rate']:.0%} "
                  f"errors={row['http_errors']} cost=${row['cost_usd']:.4f} "
                  f"p50={row['median_latency_s']}s via {row['providers']}", flush=True)
    res_df = pd.concat(all_res, ignore_index=True)
    sum_df = pd.DataFrame(summary)
    OUT_DIR.mkdir(exist_ok=True)
    res_df.to_parquet(OUT_DIR / f"{out_stem}_responses.parquet", index=False)
    sum_df.to_csv(OUT_DIR / f"{out_stem}_summary.csv", index=False)
    return sum_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="5 rows/use case, all cells")
    ap.add_argument("--stage", default="grid", choices=["grid"])
    ap.add_argument("--models", nargs="+", default=PILOT_MODELS)
    ap.add_argument("--variants", nargs="+", default=PILOT_VARIANTS)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    df = load_pilot_frame()
    print(f"loaded {len(df):,} rows across {df.use_case_key.nunique()} use cases")

    if args.smoke:
        sample = stratified_sample(df, per_use_case=6, seed=0)
        print(f"smoke subset: {len(sample)} rows "
              f"({sample.y.sum()} positive, {(sample.y == 0).sum()} negative)")
        sum_df = run_cells(sample, args.models, args.variants, args.concurrency, "wf_llm_smoke")
        print("\n=== SMOKE SUMMARY ===")
        print(sum_df.to_string(index=False))
        total = sum_df["cost_usd"].sum()
        print(f"\ntotal spend this run: ${total:.4f}")
        print(f"projected per full 1,848-row cell: ${total / len(sum_df) * (1848 / len(sample)):.3f}")
        return

    sum_df = run_cells(df, args.models, args.variants, args.concurrency, "wf_llm_grid")
    print("\n=== GRID SUMMARY ===")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
