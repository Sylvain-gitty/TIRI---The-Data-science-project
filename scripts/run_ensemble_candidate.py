"""Ensemble v1 candidate ablation — does CatBoost, and does adding a raw embedding block,
beat the already-published Tier1b+embedding LogisticRegression baseline?

Two gaps this closes, both explicitly flagged in reports/wf_query_conditioned_findings.md
§8 ("no tree/boosted model was tried on these feature blocks") and never tested anywhere
in this repo:

1. Every within-silo number published so far (reports/wf_tier1b_lexical_control.md) uses
   plain LogisticRegression. CatBoost with Ordered boosting is already this repo's chosen
   tree model (scripts/fold_pipeline_utils.py's build_tree_pipeline, picked specifically
   because RandomForestClassifier/HistGradientBoostingClassifier hit train AUC=1.000 under
   LOGO) but has never been run against the lexical/cosine-to-brief feature blocks.
2. Which embedding block to ship is genuinely unresolved: within-silo, Jasper+Qwen3-4B
   concat / Qwen3-8B alone / Qwen3-4B alone are a statistical tie (spread 0.027, seed noise
   0.015-0.027 - CONTEXT.md §5), but on SYNERGY (the only prevalence-realistic surface)
   Qwen3-8B wins clearly and Qwen3-4B is *last*. This script tests both fresh, on the new
   combined feature set, rather than picking one from that mixed evidence.

Explicitly NOT tested here (see the ensemble-modelling plan for why): Matryoshka
truncation, a learned low-rank bilinear interaction layer, PCA of any kind - CONTEXT.md /
wf_query_conditioned_findings.md §5 already found PCA-64 hurts within a silo, and neither
of the other two has ever been validated on this data. Raw embeddings only.

Same two evaluation surfaces as scripts/run_tier1b_control.py, via the shared
scripts/ensemble_eval_utils.py harness, so numbers are directly comparable:

- LOGO (leave-one-use-case-out) - chooses central defaults, not a production number.
- Within-silo (grouped k-fold inside one use case) - the production shape. **This is the
  surface the gate criterion below is measured on.**

Usage:
    python scripts/run_ensemble_candidate.py [--seeds 5]

If CatBoost is pathologically slow to fit here (a single tiny fit taking minutes instead
of seconds - a known thread-oversubscription issue on some Apple Silicon Macs when CatBoost's
default thread_count=-1 tries to use every core), run the same ablation on Modal instead:
    modal run scripts/modal_ensemble_candidate.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import logo, scores, to_md, within_silo  # noqa: E402
from fold_pipeline_utils import build_pipeline, build_tree_pipeline  # noqa: E402
from lexical_features import BRIEF_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed" / "papers_fe.parquet"
OUT = REPO / "reports" / "wf_ensemble_v1_candidate.md"

# Already-published within-silo ROC-AUC for "Tier 1b + embedding", plain LogisticRegression
# (reports/wf_tier1b_lexical_control.md §2) - the number any new combination has to beat by
# more than the ~0.03 noise floor (CONTEXT.md §5) to be worth carrying into Phase 2.
PUBLISHED_BASELINE = {
    "carbon_capture": 0.786, "cement_binders": 0.848, "ner": 0.733,
    "soil_microbiome": 0.692, "solar_leo": 0.675, "tech_forecasting": 0.711,
}
NOISE_FLOOR = 0.03

# BM25 alone is 0.637 of the full 22-column Tier 1b block's 0.642 LOGO ROC-AUC
# (reports/wf_query_conditioned_findings.md §4) - ship ~10 columns, not 22, given how small
# each silo is (260-360 rows).
LEX_BM25_COLS = [f"lex_bm25_{k}" for k in BRIEF_KEYS] + [f"lex_rank_bm25_{k}" for k in BRIEF_KEYS]
METADATA_COLS = ["year", "paper_age", "has_abstract", "n_authors", "citation_count"]


def cos_brief_cols(*models: str) -> list[str]:
    return [c for m in models for c in (f"cos_brief_{m}", f"rank_cos_brief_{m}")]


def embedding_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(prefix))


def build_variants(df: pd.DataFrame) -> dict[str, list[str]]:
    jasper = embedding_cols(df, "emb_jasper_")
    qwen4b = embedding_cols(df, "emb_qwen4b_")
    qwen8b = embedding_cols(df, "emb_qwen8b_")
    base = LEX_BM25_COLS + cos_brief_cols("jasper", "qwen4b", "qwen8b") + METADATA_COLS
    return {
        "lex (BM25 subset) only": LEX_BM25_COLS,
        "lex + cos-brief + metadata (no embedding)": base,
        "lex + cos-brief + metadata + Qwen3-8B": base + qwen8b,
        "lex + cos-brief + metadata + Jasper+Qwen3-4B concat": base + jasper + qwen4b,
    }


def catboost_fn(
    feature_cols: list[str], thread_count: int | None = None,
    iterations: int = 50, depth: int = 4,
):
    """iterations=50/depth=4, not CatBoost's own defaults of 1000/6: measured directly
    (scripts/modal_ensemble_candidate.py, on ordinary x86_64 Modal hardware, not just this
    machine's Apple Silicon) — the full within-silo sweep (150 sequential fits: 5 seeds x
    5 folds x 6 use cases) on the widest variant (~4600 raw embedding columns, ~300 rows)
    twice exceeded an 1800s per-cell timeout at 100 iterations, confirmed as a genuine
    per-fit cost (not memory - an 8GB/8-CPU container hit the same wall). This is a
    screening/ablation pass across 8 (variant, branch) cells, not the final tuned model —
    Phase 2 revisits both knobs for whichever combination this ablation picks, once it's
    down to one feature set instead of four.

    thread_count is left at CatBoost's own default (auto) unless overridden.
    scripts/modal_ensemble_candidate.py passes the container's allocated CPU count rather
    than relying on autodetection — see that file's docstring for why autodetection is
    unsafe on this development machine specifically (a severe, confirmed thread-
    oversubscription slowdown, not a real workload cost).
    """
    config = {
        "_embedding_feature_cols": [], "numeric_feature_cols": feature_cols,
        "categorical_feature_cols": [], "random_state": 0,
    }

    def _fn():
        from catboost import CatBoostClassifier

        kwargs = dict(
            auto_class_weights="Balanced", boosting_type="Ordered",
            random_state=0, verbose=False, iterations=iterations, depth=depth,
        )
        if thread_count is not None:
            kwargs["thread_count"] = thread_count
        return build_tree_pipeline(config, model=CatBoostClassifier(**kwargs))

    return _fn


def logreg_fn(feature_cols: list[str]):
    config = {
        "_embedding_feature_cols": [], "numeric_feature_cols": feature_cols,
        "categorical_feature_cols": [],
    }
    return lambda: build_pipeline(config)


def logreg_c_fn(feature_cols: list[str], C: float = 1.0):
    """Same as logreg_fn but with C exposed - every LogReg comparison in this repo's v2
    ensemble work has used sklearn's default C=1.0 unchecked; this lets the central
    hyperparameter search (scripts/run_central_hparam_search.py) actually test it."""
    config = {
        "_embedding_feature_cols": [], "numeric_feature_cols": feature_cols,
        "categorical_feature_cols": [],
    }
    return lambda: build_pipeline(
        config, model=LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    )


def svm_fn(feature_cols: list[str], kernel: str = "linear"):
    """SVC with `probability=True` (Platt-scaled via internal 5-fold CV) so its output is
    a `predict_proba` comparable to the other two branches' - a bare `decision_function`
    can't be blended with calibrated-ish probabilities on the same 0-1 scale. Reuses
    `build_pipeline`'s existing median-impute + StandardScaler preprocessing (SVC, unlike
    CatBoost, cannot accept raw NaN or unscaled features).
    """
    config = {
        "_embedding_feature_cols": [], "numeric_feature_cols": feature_cols,
        "categorical_feature_cols": [],
    }
    return lambda: build_pipeline(
        config,
        model=SVC(kernel=kernel, probability=True, class_weight="balanced", random_state=0),
    )


def knn_fn(feature_cols: list[str], n_neighbors: int = 15, metric: str = "cosine", weights: str = "distance"):
    """KNeighborsClassifier - a mechanistically different branch from CatBoost/LogReg/SVM
    (all three fit one global decision surface; k-NN decides locally, by who's nearby, not
    by a fitted boundary). `weights="distance"` (not the default "uniform") because uniform
    voting only produces coarse, bucketed probabilities (e.g. k=5 uniform can only output
    {0, 0.2, 0.4, 0.6, 0.8, 1.0}), which would hurt both the F2-threshold sweep and blending
    with the other two branches' continuous-valued probabilities. `metric="cosine"` (not
    Euclidean) because the ~4600-dim embedding block dominates this feature set by column
    count and embeddings are optimized for cosine similarity, not Euclidean distance.
    `algorithm="brute"` explicitly, since cosine isn't a valid metric for the tree-based
    algorithms sklearn would otherwise pick. Reuses `build_pipeline`'s median-impute +
    StandardScaler preprocessing (k-NN, like SVC, needs scaled, non-NaN input; the scaler
    doesn't affect cosine distance's direction-only comparison, but the imputer is required).
    """
    config = {
        "_embedding_feature_cols": [], "numeric_feature_cols": feature_cols,
        "categorical_feature_cols": [],
    }
    return lambda: build_pipeline(
        config,
        model=KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights, algorithm="brute",
        ),
    )


BRANCHES = {"CatBoost (Ordered)": catboost_fn, "LogisticRegression": logreg_fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="seeds for within-silo CV")
    args = parser.parse_args()

    df = pd.read_parquet(DATA)
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]
    # Int64 (pandas nullable) -> float64: sklearn/CatBoost expect plain float arrays, and
    # NaN must survive the cast (never fillna - NULL is not 0, CONTRIBUTING.md §1).
    df = df.copy()
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    variants = build_variants(df)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Ensemble v1 candidate — CatBoost vs LogisticRegression, feature/embedding ablation")
    emit()
    emit(f"`{len(df)}` labelled papers, {use_case.nunique()} use cases. Gate: a combination "
         f"only carries into Phase 2 if its within-silo ROC-AUC beats the published "
         f"Tier1b+embedding LogisticRegression baseline "
         f"(`reports/wf_tier1b_lexical_control.md` §2) by more than the "
         f"~{NOISE_FLOOR:.2f} noise floor (CONTEXT.md §5) on a majority of use cases.")
    emit()

    logo_results: dict[str, pd.DataFrame] = {}
    silo_results: dict[str, pd.DataFrame] = {}
    for vname, cols in variants.items():
        X = df[cols]
        for bname, fn_factory in BRANCHES.items():
            key = f"{vname} [{bname}]"
            print(f"  running {key} ...", file=sys.stderr)
            logo_results[key] = logo(X, y, use_case, fn_factory(cols))
            silo_results[key] = within_silo(X, y, use_case, groups, args.seeds, fn_factory(cols))

    # ---- 1. LOGO summary (defaults-selection surface, not production) -----------------
    emit("## 1. Leave-one-use-case-out (chooses central defaults, not a production number)")
    emit()
    logo_summary = pd.DataFrame({
        name: res[["roc_auc", "pr_auc", "recall_at_10pct"]].mean() for name, res in logo_results.items()
    }).T.round(3)
    emit(to_md(logo_summary, "variant [branch]"))
    emit()

    # ---- 2. Within-silo summary (the production surface, and the gate) ----------------
    emit("## 2. Within-silo, author-grouped 5-fold (the production shape, and the gate)")
    emit()
    silo_summary = pd.DataFrame({
        name: res[["roc_auc", "pr_auc", "recall_at_10pct"]].mean() for name, res in silo_results.items()
    }).T.round(3)
    emit(to_md(silo_summary, "variant [branch]"))
    emit()

    emit("Per use case, within-silo ROC-AUC:")
    emit()
    per_uc_raw = pd.DataFrame({name: res["roc_auc"] for name, res in silo_results.items()})
    per_uc_csv = REPO / "reports" / "wf_ensemble_v1_candidate_within_silo.csv"
    per_uc_raw.to_csv(per_uc_csv)
    per_uc = per_uc_raw.round(3)
    emit(to_md(per_uc, "use_case"))
    emit()

    # ---- 3. Gate check against the published baseline ---------------------------------
    emit("## 3. Gate: delta vs. published Tier1b+embedding LogisticRegression baseline")
    emit()
    baseline = pd.Series(PUBLISHED_BASELINE)
    delta = per_uc.sub(baseline, axis=0).round(3)
    emit(to_md(delta, "use_case (delta ROC-AUC)"))
    emit()
    for name in per_uc.columns:
        clears = int((delta[name] > NOISE_FLOOR).sum())
        emit(f"- **{name}**: clears the noise floor on {clears}/6 use cases "
             f"(mean delta {delta[name].mean():+.3f}).")
    emit()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
