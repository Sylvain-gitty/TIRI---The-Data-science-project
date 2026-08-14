"""Corpus loader for the `benchset_v1_large_*` splits — set A (8 collections, 62,229 rows,
2.19% positive) and set B (7 collections, 82,350 rows, 1.87% positive).

Set B, and why it was worth generalising for
--------------------------------------------
Set B existed on disk for a day and **nothing had ever opened it** — `grep set_b scripts/*.py`
returned nothing, and notebook `04` said so in as many words: "Nothing here has been scored.
The first number measured on `large_set_b` should be the..." That made "estimate on A, confirm
on B" an instruction no code could follow, which is the reason `wf_spec_quality_plan.md`'s
central discipline rule was unimplementable until now.

Its schema is **identical** to set A's — 4,648 columns, every one of `KEY_COLS + BASELINE_COLS`
present, 0 `first_author` nulls, 0 duplicate `(paper_id, use_case_key)`, 100% abstracts — so
this is a parameterisation, not a build. Two differences that are NOT cosmetic:

  - **7 collections, not 8.** Any win-count bar written as "≥5 of 8" is unreadable here.
  - **`nykvist_evcharging` is not a SYNERGY collection.** `brief_provenance` still reads
    `review_abstract`, but its `objective` is 160 characters against ~1,200 for every SYNERGY
    collection — i.e. someone wrote it. `raw_brief_map()`'s assumption below therefore does not
    hold for that one collection; treat it as an exception, do not average it in.

Set B is the last clean surface in the repo (`CONTEXT.md` §4 ring-fences the unused SYNERGY
reviews). **Every read of it burns it.** Spend it on one pre-registered question at a time.

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
    python scripts/benchset_loader.py --check              # integrity gate on set A, no LLM calls
    python scripts/benchset_loader.py --check --set b      # ... and on set B
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
PAPERS = PROCESSED / "papers_benchset_v1.parquet"
AB_CROSSING = REPO / "reports" / "benchset_v1_ab_crossing_papers.csv"


@dataclass(frozen=True)
class SetSpec:
    """One split, with the integrity numbers `--check` asserts against.

    The expected counts live here rather than inline in `main()` because they were previously
    hardcoded to set A, which is exactly the kind of assertion that silently passes on the
    wrong data once a second set exists.
    """
    name: str
    path: Path
    manifest: Path
    n_rows: int
    n_pos: int
    n_use_cases: int
    split_set: str          # how this set is spelled in `benchset_v1_ab_crossing_papers.csv`


SETS: dict[str, SetSpec] = {
    "a": SetSpec("a", PROCESSED / "benchset_v1_large_set_a.parquet",
                 REPO / "reports" / "wf_llm_setA_sample.parquet",
                 n_rows=62_229, n_pos=1_362, n_use_cases=8, split_set="large_set_a"),
    "b": SetSpec("b", PROCESSED / "benchset_v1_large_set_b.parquet",
                 REPO / "reports" / "wf_llm_setB_sample.parquet",
                 n_rows=82_350, n_pos=1_542, n_use_cases=7, split_set="large_set_b"),
}

# Kept as module constants because three scripts import them by name.
SET_A = SETS["a"].path
MANIFEST = SETS["a"].manifest

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


def load_set(name: str, with_baselines: bool = True) -> pd.DataFrame:
    """One split's keys + labels joined to text and brief. One-to-one, no nulls.

    `name` is "a" or "b". Both parquets carry the identical 4,648-column schema, so the only
    thing that varies is which file is read.
    """
    try:
        spec = SETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown set {name!r} - expected one of {sorted(SETS)}") from None
    cols = KEY_COLS + (BASELINE_COLS if with_baselines else [])
    keys = pd.read_parquet(spec.path, columns=cols)
    papers = pd.read_parquet(PAPERS, columns=["row_key", "use_case_key", *TEXT_COLS])
    df = keys.merge(papers, on=["row_key", "use_case_key"], how="left", validate="one_to_one")
    if len(df) != len(keys):
        raise RuntimeError(f"join changed row count: {len(keys)} -> {len(df)}")
    if df["title"].isna().any():
        raise RuntimeError("join produced null titles - schema drift, do not proceed")
    df["y"] = df["y"].astype(int)
    return df


def load_set_a(with_baselines: bool = True) -> pd.DataFrame:
    """Set A keys + labels joined to text and brief. 62,229 rows, one-to-one, no nulls."""
    return load_set("a", with_baselines=with_baselines)


def load_set_b(with_baselines: bool = True) -> pd.DataFrame:
    """Set B keys + labels joined to text and brief. 82,350 rows, one-to-one, no nulls."""
    return load_set("b", with_baselines=with_baselines)


def drop_ab_crossing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove rows whose paper also appears in the *other* large split.

    154 papers occupy both sets (313 rows: 158 on A's side, 155 on B's). They are not
    duplicates - each is a legitimately different (paper, use case) pair - but if set A was
    used to estimate and set B to confirm, a paper the estimate already saw is not held out.
    The effect is small by construction; the point is that "confirmed on a clean surface"
    should mean it, and this makes the claim checkable rather than approximately true.
    """
    spec = SETS[name.lower()]
    crossing = pd.read_csv(AB_CROSSING, usecols=["row_key", "split_set"])
    drop = set(crossing.loc[crossing.split_set == spec.split_set, "row_key"])
    out = df[~df["row_key"].isin(drop)].reset_index(drop=True)
    return out


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

    ⚠️ **Does not hold for `nykvist_evcharging` (set B).** That collection's `objective` is 160
    characters, against ~1,200 for every SYNERGY collection — it is a written scope statement,
    not a review abstract, despite `brief_provenance` claiming otherwise. B0 vs B1 on that one
    collection therefore measures something different from what it measures on the other six,
    so report it as an exception rather than folding it into a mean.
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
    ap.add_argument("--set", default="a", choices=sorted(SETS), help="which large split to load")
    ap.add_argument("--check", action="store_true", help="integrity gate")
    ap.add_argument("--write-manifest", action="store_true",
                    help="persist the case-control sample so analysis can join weights back")
    ap.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    args = ap.parse_args()

    spec = SETS[args.set]
    df = load_set(args.set)
    print(f"set {spec.name.upper()}: loaded {len(df):,} rows / {df.use_case_key.nunique()} "
          f"use cases / {df.y.sum():,} positive ({df.y.mean():.4f})")
    print(f"  ELI18: {df.y.mean():.2%} of these papers are relevant, so a model that guessed "
          f"'no' every time would be right {1 - df.y.mean():.1%} of the time and useless.")

    if args.check:
        assert len(df) == spec.n_rows, f"{len(df)} != expected {spec.n_rows}"
        assert df.y.sum() == spec.n_pos, f"{df.y.sum()} != expected {spec.n_pos}"
        assert df.use_case_key.nunique() == spec.n_use_cases, df.use_case_key.nunique()
        assert df.title.isna().sum() == 0
        assert (df.abstract.fillna("").str.len() > 0).all(), \
            f"set {spec.name.upper()} should be 100% abstracts"
        assert df.duplicated(["paper_id", "use_case_key"]).sum() == 0
        assert df.first_author.isna().sum() == 0
        strict = drop_ab_crossing(df, args.set)
        print(f"integrity: OK  ({len(df) - len(strict)} rows also appear in the other large "
              f"split; drop_ab_crossing() removes them for a strict held-out read)")

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
        spec.manifest.parent.mkdir(exist_ok=True)
        sample[keep].to_parquet(spec.manifest, index=False)
        print(f"wrote {spec.manifest}")


if __name__ == "__main__":
    main()
