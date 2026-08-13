"""Corpus loader for `benchset_v1_large_set_a` — 8 collections, 62,229 rows, 2.19% positive.

Why this file exists
--------------------
The screening pilot ran on TIRI's own six use cases at 26-77% prevalence, where the
all-positive baseline already scores F2 = 0.872 and the metric cannot separate anything.
`CONTEXT.md` §4 says so twice. Set A is the surface that fixes that: at 2.19% positive the
all-positive floor is **0.101**, so F2 discriminates for the first time in this project.

Set A is entirely SYNERGY collections, which matters for one specific reason. `CONTEXT.md`
§4 records that the shuffled-brief control *fails* on SYNERGY (1/3, -0.012) because "SYNERGY
briefs are a published review's title and abstract ... and carry no curated term lists".
That is no longer true of this export: `brief_provenance` is still `review_abstract`, but
`terms_must_include` / `terms_nice_to_have` / `terms_exclude` are populated. So the old
failure is a hypothesis to re-test here, not a reason to distrust the surface — and
`raw_brief_map()` below exists precisely to reconstruct the old condition and measure the
difference.

Two files, joined
-----------------
`benchset_v1_large_set_a.parquet` is the label/split authority but has **4,648 columns**,
4,608 of them `emb_*`. Never read it whole - the column list below keeps it to a few MB.
`papers_benchset_v1.parquet` (all 181,199 benchset rows) carries the text and the nine
brief columns `llm_pipeline_utils.BRIEF_FIELDS` expects, unchanged. They join one-to-one on
`(row_key, use_case_key)`.

Note `paper_id` alone is NOT unique here: 1,221 papers appear in more than one set-A
collection. `(paper_id, use_case_key)` is unique (verified) and is what `score_frame`
emits, so it is the join key back to the sample manifest.

Usage:
    python scripts/benchset_loader.py --check       # integrity gate, no LLM calls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parent.parent
SET_A = REPO / "data" / "processed" / "benchset_v1_large_set_a.parquet"
PAPERS = REPO / "data" / "processed" / "papers_benchset_v1.parquet"
MANIFEST = REPO / "reports" / "wf_llm_setA_sample.parquet"

# Everything except the 4,608 emb_* columns that make this file 1 GB. The `cos_*`/`lex_*`
# columns come along because they ARE the incumbent bar - the cold-start cosine-to-brief
# ranker this run has to beat is already computed and on disk, and costs nothing to read.
KEY_COLS = ["row_key", "paper_id", "use_case_key", "first_author", "y"]
BASELINE_COLS = [
    "cos_brief_jasper", "cos_briefpre_jasper", "cos_brief_qwen4b", "cos_briefpre_qwen4b",
    "rank_cos_brief_jasper", "rank_cos_brief_qwen4b",
    "lex_bm25_obj", "lex_bm25_must", "lex_bm25_nice", "lex_overlap_must_frac",
]

TEXT_COLS = [
    "title", "abstract", "use_case_name", "objective", "problem_statement",
    "terms_must_include", "terms_nice_to_have", "terms_exclude",
    "domain_industry", "domain_application", "domain_technology_focus",
]

N_NEG_DEFAULT = 1500
SEED = 0


def load_set_a(with_baselines: bool = True) -> pd.DataFrame:
    """Set A keys + labels joined to text and brief. 62,229 rows, one-to-one, no nulls."""
    cols = KEY_COLS + (BASELINE_COLS if with_baselines else [])
    keys = pd.read_parquet(SET_A, columns=cols)
    papers = pd.read_parquet(PAPERS, columns=["row_key", "use_case_key", *TEXT_COLS])
    df = keys.merge(papers, on=["row_key", "use_case_key"], how="left", validate="one_to_one")
    if len(df) != len(keys):
        raise RuntimeError(f"join changed row count: {len(keys)} -> {len(df)}")
    if df["title"].isna().any():
        raise RuntimeError("join produced null titles - schema drift, do not proceed")
    df["y"] = df["y"].astype(int)
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """60/20/20 train/test/validate, grouped by first author, stratified on use case x label.

    Same instrument as `analyze_llm_splits.py`: nothing in a zero-shot prompt is fitted, so
    the split exists for the two things that ARE fitted here - the decision threshold, and
    the induced rule set (B2/P5). Both are chosen on train and applied unchanged elsewhere.
    Grouping by first author stops an author's papers straddling a boundary, which would
    leak into the induced rules in particular.
    """
    strat = df["use_case_key"].astype(str) + "_" + df["y"].astype(str)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(skf.split(df, strat, groups=df["first_author"])):
        folds[idx] = i
    return pd.Series(np.where(folds < 3, "train", np.where(folds == 3, "test", "validate")),
                     index=df.index)


def case_control_sample(df: pd.DataFrame, n_neg: int = N_NEG_DEFAULT,
                        seed: int = SEED) -> pd.DataFrame:
    """Every positive, plus `n_neg` random negatives per use case, with a recovery weight.

    Scoring all 62,229 rows across four models is ~$130 and almost all of it buys negatives.
    This buys 9,993 rows per cell instead and recovers the population numbers analytically:

      - **Recall is exact.** Every positive is scored; nothing about it is estimated.
      - **ROC-AUC is unbiased unweighted.** AUC is P(score+ > score-), a function of the two
        class-conditional score distributions and not of the prior, and a uniform sample of
        negatives estimates the negative distribution without bias.
      - **Precision, F2, WSS@95 and recall@k are NOT.** They all depend on how many negatives
        sit above the line, so each sampled negative has to stand for the `w` negatives it
        was drawn on behalf of. `w = n_negatives_in_population / n_negatives_sampled`,
        positives carry `w = 1`. Forgetting this is the one way to get a badly wrong - and
        flatteringly wrong - answer out of this design.

    Every cell scores the identical sample, so model-vs-model differences are **paired** and
    much tighter than the absolute confidence intervals. That matters most on
    `brouwer_2019`, where 62 positives against a 4% negative sample gives a wide absolute
    band but a perfectly usable ordering.
    """
    parts = []
    for uc, grp in df.groupby("use_case_key", sort=True):
        pos = grp[grp.y == 1]
        neg = grp[grp.y == 0]
        take = min(n_neg, len(neg))
        drawn = neg.sample(n=take, random_state=seed)
        pos = pos.assign(w=1.0)
        # NULL is not 0 and neither is this: a use case where every negative was drawn gets
        # w = 1.0 exactly, not an approximation, and its metrics are population numbers.
        drawn = drawn.assign(w=len(neg) / take)
        parts.append(pd.concat([pos, drawn]))
    out = pd.concat(parts).sort_values(["use_case_key", "row_key"]).reset_index(drop=True)
    out["split"] = make_splits(out, seed=seed)
    return out


def raw_brief_map(df: pd.DataFrame) -> dict[str, str]:
    """B0 - the review's own title and abstract, with the curated term lists stripped.

    This reconstructs the brief format that made the shuffled-brief control fail on old
    SYNERGY. `brief_provenance` is `review_abstract` for every row: `objective` holds the
    review's abstract and `problem_statement` its title. Everything else in the brief -
    the must/nice/exclude term lists and the domain fields - was written by an LLM from
    that abstract. B0 vs B1 therefore measures exactly what that derived terminology is
    worth, which `CONTEXT.md` §4 flags as an open risk and nobody has quantified.

    Field labels are copied from `render_brief` so the only difference between the arms is
    which lines are present.
    """
    briefs = {}
    for uc, grp in df.groupby("use_case_key", sort=True):
        row = grp.iloc[0]
        lines = [f"Topic: {row['use_case_name']}"]
        if str(row["objective"]).strip():
            lines.append(f"Objective: {row['objective']}")
        if str(row["problem_statement"]).strip():
            lines.append(f"Problem being solved: {row['problem_statement']}")
        briefs[uc] = "\n".join(lines)
    return briefs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="integrity gate")
    ap.add_argument("--write-manifest", action="store_true",
                    help="persist the case-control sample so analysis can join weights back")
    ap.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    args = ap.parse_args()

    df = load_set_a()
    print(f"loaded {len(df):,} rows / {df.use_case_key.nunique()} use cases / "
          f"{df.y.sum():,} positive ({df.y.mean():.4f})")

    if args.check:
        assert len(df) == 62_229, len(df)
        assert df.y.sum() == 1_362, df.y.sum()
        assert df.title.isna().sum() == 0
        assert (df.abstract.fillna("").str.len() > 0).all(), "set A should be 100% abstracts"
        assert df.duplicated(["paper_id", "use_case_key"]).sum() == 0
        assert df.first_author.isna().sum() == 0
        print("integrity: OK")

    sample = case_control_sample(df, n_neg=args.n_neg)
    t = sample.groupby("use_case_key").agg(
        n=("y", "size"), pos=("y", "sum"), w_neg=("w", "max"))
    t["prev_sampled"] = (t.pos / t.n).round(3)
    print(f"\ncase-control sample: {len(sample):,} rows per cell")
    print(t.to_string())
    print(f"\nsplits: {sample.split.value_counts().to_dict()}")

    if args.write_manifest:
        keep = ["row_key", "paper_id", "use_case_key", "first_author", "y", "w", "split",
                *BASELINE_COLS]
        MANIFEST.parent.mkdir(exist_ok=True)
        sample[keep].to_parquet(MANIFEST, index=False)
        print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
