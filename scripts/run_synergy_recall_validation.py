"""The recall comparison, repeated on SYNERGY — the only prevalence-realistic surface we have.

WHY THIS RUN MATTERS MORE THAN THE IN-REPO ONE
----------------------------------------------
TIRI's six pools run 26-77% positive, because they are what survived retrieval and a
human's attention. Production will not look like that. SYNERGY's reviews run 1.7-14.8%
included, which is the regime a 100k-paper pool actually lives in, and the labels come
from published systematic reviews this project had no hand in creating.

Two things only this surface can test:

1. Whether the query-conditioned result survives realistic imbalance. The shuffled-brief
   control is repeated here (briefs rotated between the three reviews) so the claim is
   falsifiable on external data, not just on ours.
2. Whether "a new use case arrives with labels" is a safe assumption. At 1.7% prevalence
   a random draw of 25 labels contains 0.4 positives on average, so most such draws
   contain no positive at all and cannot train anything. The warm-start section reports
   the share of unusable draws per budget, which is the number that decides whether the
   bootstrap budget must be *actively selected* rather than randomly sampled.

The brief here is the review's own title plus abstract, from SYNERGY's publication
metadata. That is label-free and known before screening starts, which is the same
standing TIRI's `objective`/`problem_statement` have. Deliberately NOT used: the
`concepts.included` field, which is derived from the inclusion decisions and would leak.

WHAT IS NOT COVERED
-------------------
Dense cosine-to-brief needs a vector for each review's brief, and only paper vectors are
cached. Embedding three short texts means a Modal GPU spin-up, so it is left out rather
than spent unasked; the lexical BM25-to-brief column below is the free analogue and tests
the same claim. To add it: embed the three brief strings with embedding_utils.embed_texts
and cache them as synergy_<review>_<model>_brief.parquet.

Usage:
    python scripts/run_synergy_recall_validation.py [--seeds 5] [--ws-seeds 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_pipeline_utils import ranking_metrics  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402
from run_embedding_recall_comparison import MODELS, _fit_scores, metrics  # noqa: E402
from run_tier1b_control import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "processed" / "embeddings_cache"
OUT = REPO / "reports" / "wf_synergy_recall_validation.md"
REVIEWS = ["Sep_2021", "Menon_2022", "van_der_Waal_2022"]


def inverted_index_to_text(index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the running text."""
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        positions.extend((p, word) for p in spots)
    return " ".join(w for _, w in sorted(positions))


def load_reviews() -> tuple[pd.DataFrame, dict[str, str]]:
    """One long frame shaped like papers_combined.parquet, so build_lexical_features and
    the fold code run unchanged. `use_case_key` becomes the review name: a SYNERGY review
    is exactly a silo -- one customer, one question, one pool."""
    from synergy_dataset import Dataset

    frames, briefs = [], {}
    for name in REVIEWS:
        ds = Dataset(name)
        # reset_index() (not drop=True) -- the frame's index IS openalex_id, and that is
        # the key the cached SYNERGY vectors are stored under.
        rdf = ds.to_frame().reset_index()
        rdf = rdf.dropna(subset=["title"]).copy()
        rdf["abstract"] = rdf["abstract"].fillna("")
        rdf["y"] = rdf["label_included"].astype(int)
        rdf["use_case_key"] = name

        pub = ds.metadata.get("publication", {})
        brief = f"{pub.get('title') or name}. {inverted_index_to_text(pub.get('abstract_inverted_index'))}"
        briefs[name] = brief

        # The TIRI brief columns build_lexical_features expects. SYNERGY has no curated
        # term lists, so they stay empty -- which exercises the NULL-is-not-0 path rather
        # than working around it, and is itself worth knowing about.
        rdf["objective"] = pub.get("title") or name
        rdf["problem_statement"] = inverted_index_to_text(pub.get("abstract_inverted_index"))
        for col in ("terms_must_include", "terms_nice_to_have", "terms_exclude"):
            rdf[col] = [[] for _ in range(len(rdf))]
        for col in ("domain_industry", "domain_application"):
            rdf[col] = ""
        rdf["domain_technology_focus"] = [[] for _ in range(len(rdf))]
        frames.append(rdf)
    return pd.concat(frames, ignore_index=True), briefs


def load_embeddings(df: pd.DataFrame, short: str) -> pd.DataFrame | None:
    """Cached SYNERGY paper vectors, concatenated across reviews in df row order."""
    blocks = []
    for name in REVIEWS:
        path = CACHE / f"synergy_{name}_{MODELS[short]}_papers.parquet"
        if not path.exists():
            return None
        cached = pd.read_parquet(path).set_index("id")
        ids = df.loc[df.use_case_key == name, "openalex_id"]
        blocks.append(np.vstack(cached.loc[ids].embedding.values).astype("float32"))
    return pd.DataFrame(np.vstack(blocks), index=df.index).add_prefix(f"{short}_")


def within_review(spec: dict, y: np.ndarray, review: pd.Series, seeds: int) -> pd.DataFrame:
    """Stratified 5-fold inside each review. No author grouping: SYNERGY exports carry no
    author column, so this is very slightly optimistic versus the in-repo runs."""
    rows = []
    for name in REVIEWS:
        m = (review == name).to_numpy()
        Xu, yu = spec["X"][m], y[m]
        per_seed = []
        for seed in range(seeds):
            oof = np.full(len(yu), np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xu, yu):
                oof[te] = _fit_scores(spec, Xu.iloc[tr], yu[tr], Xu.iloc[te])
            per_seed.append(metrics(yu, oof))
        rows.append({"review": name, **pd.DataFrame(per_seed).mean().to_dict()})
    return pd.DataFrame(rows).set_index("review")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ws-seeds", type=int, default=40)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    args = ap.parse_args()

    df, briefs = load_reviews()
    y = df["y"].to_numpy()
    review = df.use_case_key

    lex = build_lexical_features(df)
    rotated = {a: b for a, b in zip(REVIEWS, REVIEWS[1:] + REVIEWS[:1])}
    lex_shuffled = build_lexical_features(df, brief_map=rotated)

    variants: dict[str, dict] = {
        "lexical only (Tier 1b)": {"X": lex, "n_pca": None},
        "lexical — SHUFFLED briefs (control)": {"X": lex_shuffled, "n_pca": None},
    }
    for short in MODELS:
        E = load_embeddings(df, short)
        if E is None:
            continue
        variants[f"{short}: embedding"] = {"X": E, "n_pca": None}
        variants[f"{short}: embedding + lexical"] = {
            "X": pd.concat([E, lex], axis=1), "n_pca": None, "pca_on": list(E.columns)}

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# SYNERGY validation — the recall lens at realistic prevalence")
    emit()
    prev = df.groupby("use_case_key").y.agg(["size", "sum", "mean"])
    prev.columns = ["n", "n_included", "prevalence"]
    prev["wss_ceiling"] = 0.95 * (1 - prev.prevalence)
    prev["exp_pos_in_25_random"] = (25 * prev.prevalence).round(2)
    emit(to_md(prev.round(3), "review"))
    emit()
    emit("`exp_pos_in_25_random` is the expected number of positives in a random 25-label "
         "bootstrap. Below 1.0 it means most such draws contain no positive at all and "
         "cannot train a classifier — see §3.")
    emit()

    emit("## 1. Within-review (each SYNERGY review is a silo)")
    emit()
    silo = {name: within_review(spec, y, review, args.seeds) for name, spec in variants.items()}
    for metric in ("wss_at_95", "recall_at_10pct", "roc_auc"):
        piv = pd.DataFrame({n: r[metric] for n, r in silo.items()}).T
        piv["MEAN"] = piv.mean(axis=1)
        emit(f"**{metric}**")
        emit()
        emit(to_md(piv.sort_values("MEAN", ascending=False).round(3), "variant"))
        emit()

    emit("## 2. Shuffled-brief control, at realistic prevalence")
    emit()
    real = silo["lexical only (Tier 1b)"]
    ctrl = silo["lexical — SHUFFLED briefs (control)"]
    cmp = pd.DataFrame({"real_brief": real.roc_auc, "wrong_brief": ctrl.roc_auc})
    cmp["gap"] = cmp.real_brief - cmp.wrong_brief
    emit(to_md(cmp.round(3), "review"))
    emit()
    emit(f"Real briefs beat wrong briefs on **{int((cmp.gap > 0).sum())}/{len(cmp)}** reviews, "
         f"mean gap **{cmp.gap.mean():+.3f}** ROC-AUC.")
    emit()

    emit("## 3. Warm-start at low prevalence — and how often 25 random labels are unusable")
    emit()
    rows = []
    for name, spec in variants.items():
        for rev in REVIEWS:
            m = (review == rev).to_numpy()
            Xu, yu = spec["X"][m], y[m]
            for n in args.budgets:
                if n >= len(yu):
                    continue
                per_seed, unusable = [], 0
                for seed in range(args.ws_seeds):
                    order = np.random.default_rng(seed).permutation(len(yu))
                    tr, te = order[:n], order[n:]
                    if len(np.unique(yu[tr])) < 2:
                        unusable += 1
                        continue
                    per_seed.append(metrics(yu[te], _fit_scores(spec, Xu.iloc[tr], yu[tr], Xu.iloc[te])))
                rows.append({"variant": name, "review": rev, "n_labels": n,
                             "pct_draws_unusable": unusable / args.ws_seeds,
                             **(pd.DataFrame(per_seed).mean().to_dict() if per_seed else {})})
    ws = pd.DataFrame(rows)
    ws.to_csv(REPO / "reports" / "wf_synergy_warm_start.csv", index=False)

    emit("Share of random label draws with no positive at all (so no model can be fit):")
    emit()
    emit(to_md(ws.pivot_table(index="review", columns="n_labels",
                              values="pct_draws_unusable").round(2), "review"))
    emit()
    emit("WSS@95 by budget (mean over reviews, unusable draws excluded):")
    emit()
    emit(to_md(ws.pivot_table(index="variant", columns="n_labels",
                              values="wss_at_95").round(3), "variant"))
    emit()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
