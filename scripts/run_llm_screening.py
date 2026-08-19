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

    # override=True is load-bearing, not tidiness. `load_dotenv` defaults to leaving an
    # already-set environment variable alone, so a stale OPENROUTER_API_KEY exported from
    # a shell profile silently wins over the .env file - which is where requirements.txt
    # says the key lives. That failure mode is near-undiagnosable from the symptom: after
    # a key rotation, .env holds the new key, every request 401s with "User not found",
    # and the file you are staring at looks correct. It cost most of a grid run once.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
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
    # `cost` is an audit field replayed from the cached response, so summing it gives what
    # the cell cost to produce - not what this run spent. Those diverge to 100% of the total
    # on a fully cached re-run, which is exactly when someone is watching the number to
    # decide whether a cache change silently started re-buying responses. Report both.
    cost_all = pd.to_numeric(res["cost"], errors="coerce").fillna(0)
    fresh = ~res["cached"].fillna(False).astype(bool)
    return {
        "n": n,
        "cached": int((~fresh).sum()),
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
        "cost_usd": round(float(cost_all.sum()), 4),
        "spend_usd": round(float(cost_all[fresh].sum()), 4),
        "providers": ",".join(sorted(res["provider"].dropna().unique().tolist())[:4]),
    }


def run_cells(df: pd.DataFrame, models: list[str], variants: list[str], concurrency: int,
              out_stem: str, brief_map: dict | None = None,
              brief_tag: str | None = None) -> pd.DataFrame:
    all_res, summary = [], []
    for model in models:
        for variant in variants:
            print(f"\n>>> {model}  {variant}  ({len(df)} rows)", flush=True)
            res = score_frame(df, model=model, variant=variant, concurrency=concurrency,
                              brief_map=brief_map, brief_tag=brief_tag)
            all_res.append(res)
            row = {"model": model, "variant": variant, **summarise(res)}
            summary.append(row)
            print(f"    parse={row['parse_rate']:.0%} score={row['score_rate']:.0%} "
                  f"errors={row['http_errors']} spend=${row['spend_usd']:.4f} "
                  f"(cached {row['cached']}/{row['n']}) "
                  f"p50={row['median_latency_s']}s via {row['providers']}", flush=True)
    res_df = pd.concat(all_res, ignore_index=True)
    sum_df = pd.DataFrame(summary)
    OUT_DIR.mkdir(exist_ok=True)
    res_df.to_parquet(OUT_DIR / f"{out_stem}_responses.parquet", index=False)
    sum_df.to_csv(OUT_DIR / f"{out_stem}_summary.csv", index=False)
    return sum_df


def shuffled_brief_map(df: pd.DataFrame, seed: int = 0) -> dict[str, str]:
    """Every use case gets a DIFFERENT use case's brief - a derangement, never a fixed point.

    This is the falsification control the rest of the repo already runs on anything that
    claims to read the brief (`build_lexical_features(df, brief_map=...)` exists for exactly
    this). The logic: an LLM scoring papers against deliberately wrong criteria should
    collapse. If it does not, it is scoring "is this a good paper" rather than "does this
    paper match this brief", and every number in the grid is measuring the wrong thing.

    Worth remembering that this control is not a formality here - it passes decisively on
    TIRI's six use cases for the lexical block (5/6, +0.155) and *fails* on SYNERGY, so it
    genuinely discriminates.
    """
    from llm_pipeline_utils import render_brief

    keys = sorted(df["use_case_key"].unique())
    briefs = {k: render_brief(dict(zip(df.columns, df[df.use_case_key == k].iloc[0])))
              for k in keys}
    rng = np.random.RandomState(seed)
    for _ in range(100):
        perm = rng.permutation(len(keys))
        if all(i != j for i, j in enumerate(perm)):  # derangement: no key keeps its own
            return {keys[i]: briefs[keys[perm[i]]] for i in range(len(keys))}
    raise RuntimeError("could not find a derangement")


def load_corpus(corpus: str) -> pd.DataFrame:
    """TIRI's 1,848 labelled rows, or the set-A case-control sample. Same columns either way.

    `benchset_a` returns the 9,993-row sample, not all 62,229: every positive plus a random
    share of the negatives, carrying the weight `w` that puts the metrics back at the true
    2.19% prevalence. `scripts/validate_benchset_sampling.py` is the gate on that design and
    must pass before any cell is bought here.
    """
    if corpus == "tiri":
        return load_pilot_frame()
    from benchset_loader import case_control_sample, load_set_a

    return case_control_sample(load_set_a())


def resolve_briefs(corpus: str, brief: str, df: pd.DataFrame) -> tuple[dict | None, str | None]:
    """The brief-format ladder: one prompt, four brief sets.

    `own` is the supplied brief and needs no map at all — importantly, that path must keep
    producing a cache tag byte-identical to the pilot's, or every response already paid for
    becomes unreachable and the grid silently re-buys itself.
    """
    if brief == "own":
        return None, None
    if brief == "shuffled":
        return shuffled_brief_map(df, seed=0), "shuffled"
    if brief == "raw":
        from benchset_loader import raw_brief_map

        return raw_brief_map(df), "raw"
    if brief == "induced":
        import json

        path = Path("reports/wf_llm_setA_rules_v1.json")
        if not path.exists():
            raise SystemExit(f"{path} missing - run scripts/induce_rule_set.py first")
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob["briefs"], f"induced-{blob['rule_set_version']}"
    raise ValueError(brief)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="a few rows/use case, all cells")
    ap.add_argument("--stage", default="grid", choices=["grid", "control"])
    ap.add_argument("--corpus", default="tiri", choices=["tiri", "benchset_a"])
    ap.add_argument("--brief", default="own", choices=["own", "raw", "induced", "shuffled"])
    ap.add_argument("--models", nargs="+", default=PILOT_MODELS)
    ap.add_argument("--variants", nargs="+", default=PILOT_VARIANTS)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--tag", default="",
                    help="appended to the output stem. Needed when running one model per "
                         "process - different models sit on different providers with "
                         "independent rate limits, so cells run far faster in parallel, but "
                         "they would otherwise all write to the same parquet.")
    args = ap.parse_args()

    df = load_corpus(args.corpus)
    print(f"loaded {len(df):,} rows across {df.use_case_key.nunique()} use cases "
          f"({df.y.sum():,} positive)")

    # Every output name carries corpus, variant set and brief arm. The TIRI + own-brief +
    # full-variants path resolves to exactly the old names so nothing already produced moves.
    # (`--variants P2lp` once wrote to wf_llm_grid_responses.parquet and destroyed the 12-cell
    # grid it took an hour to produce. The cache made recovery free; the name is the real fix.)
    corpus_tag = "" if args.corpus == "tiri" else "_setA"
    suffix = "" if sorted(args.variants) == sorted(PILOT_VARIANTS) else "_" + "-".join(args.variants)
    brief_sfx = "" if args.brief == "own" else f"_{args.brief}"
    tag_sfx = f"_{args.tag}" if args.tag else ""

    if args.smoke:
        per_uc = 6 if args.corpus == "tiri" else 30
        sample = stratified_sample(df, per_use_case=per_uc, seed=0)
        print(f"smoke subset: {len(sample)} rows "
              f"({sample.y.sum()} positive, {(sample.y == 0).sum()} negative)")
        bm, bt = resolve_briefs(args.corpus, args.brief, df)
        sum_df = run_cells(sample, args.models, args.variants, args.concurrency,
                           f"wf_llm{corpus_tag}_smoke{brief_sfx}{tag_sfx}", brief_map=bm, brief_tag=bt)
        print("\n=== SMOKE SUMMARY ===")
        print(sum_df.to_string(index=False))
        total = sum_df["cost_usd"].sum()
        print(f"\nspend this run: ${sum_df['spend_usd'].sum():.4f} "
              f"(cell cost incl. cache hits: ${total:.4f})")
        print(f"projected per full {len(df):,}-row cell: "
              f"${total / len(sum_df) * (len(df) / len(sample)):.3f}")
        return

    if args.stage == "control":
        bm = shuffled_brief_map(df, seed=0)
        print("shuffled-brief control - each use case scored against another's criteria")
        sum_df = run_cells(df, args.models, args.variants, args.concurrency,
                           f"wf_llm{corpus_tag}_control{suffix}{tag_sfx}", brief_map=bm,
                           brief_tag="shuffled")
        print("\n=== CONTROL SUMMARY ===")
        print(sum_df.to_string(index=False))
        return

    bm, bt = resolve_briefs(args.corpus, args.brief, df)
    sum_df = run_cells(df, args.models, args.variants, args.concurrency,
                       f"wf_llm{corpus_tag}_grid{suffix}{brief_sfx}{tag_sfx}", brief_map=bm, brief_tag=bt)
    print("\n=== GRID SUMMARY ===")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
