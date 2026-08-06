"""Shared, config-driven building blocks for notebooks/pipelines/*.ipynb.

Both sf_generalized_fold_pipeline.ipynb (pooled, whole-dataset splits) and
sf_logo_fold_pipeline.ipynb (leave-one-use-case-out) import this module instead of
redefining the same preprocessing/metrics code twice. The point: the two notebooks' fold
*structure* differs (random grouped/stratified split vs. rotating use-case holdout), but
"how do we turn a row into something a LogisticRegression can use, without leaking" should
live in exactly one place, so it can't drift between them as the upstream feature-engineering
dataset changes shape.

Nothing in this module reads data/ or fits anything at import time - every function here is
called explicitly by the notebooks, at the point in their own fold logic where it's safe to
call it (see each function's docstring for which side of the train/test boundary it belongs on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, roc_auc_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_schema(df: pd.DataFrame, config: dict) -> None:
    """Raise loudly if a column CONFIG names isn't actually present in df.

    This repo's own convention for anything that consumes a dataset another workstream
    produces (notebooks/README.md's "raises loudly on anything unexpected instead of
    silently guessing" rule) - meant to be the first thing that breaks when a new
    engineered dataset's schema doesn't match what CONFIG expects, with a message that
    says exactly which CONFIG entries to fix, instead of a KeyError three cells later with
    no context.

    `config.get("expand_embedding", True)` gates whether `embedding_col` is required at
    all - a text-driven model (e.g. a prompted LLM, see `notebooks/pipelines/
    sf_llm_fold_pipeline.ipynb`) has no use for the embedding column and can set this to
    False rather than being forced to point CONFIG at a column it never reads.
    """
    required = [config["target_col"], config["use_case_col"]]
    if config.get("expand_embedding", True):
        required.append(config["embedding_col"])
    required += config["numeric_feature_cols"] + config["categorical_feature_cols"]
    if config["group_col"] not in df.columns:
        required.append(config["group_source_col"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CONFIG expects column(s) {missing} that aren't in this dataset - update "
            f"CONFIG to match its actual schema before continuing. Columns present: "
            f"{sorted(df.columns)}"
        )


def derive_first_author(series: pd.Series) -> pd.Series:
    """`authors` free-text column -> lowercased first-author surname.

    Rows with no parseable author each get a unique singleton value (`no_author_<index>`)
    rather than colliding into one shared "unknown" group - grouping missing-author rows
    together would let the CV group-splitter treat them as one (fake) shared author,
    which is worse than treating each as its own group of one.
    """
    def _first(value):
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        return str(value).split(",")[0].strip().lower()

    parsed = series.apply(_first)
    missing = parsed.isna()
    parsed = parsed.astype(object)
    parsed.loc[missing] = "no_author_" + parsed.index[missing].astype(str)
    return parsed


def prepare_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Row-wise, parameter-free prep - safe to run on the WHOLE dataset before any split
    exists, because nothing here estimates anything from the data's distribution (no
    fitted mean/std/quantile/vocabulary - see the pipeline notebooks' own "split boundary"
    section for the full argument). Three things happen, all per-row:

    1. Binarize the target and drop rows outside `positive_label`/`negative_labels`
       (e.g. this repo's `pass` triage class, which isn't a real relevance judgement).
    2. If `config.get("expand_embedding", True)`: expand the embedding column into named
       `emb_0..emb_{d-1}` scalar columns, inferring the dimensionality at runtime rather
       than hardcoding it - a differently-sized embedding in a future dataset needs no
       code change here. Set this to False for a model that doesn't consume the embedding
       at all (e.g. a prompted LLM working off raw title/abstract text) - `embedding_col`
       is then never even read, and `_embedding_feature_cols` comes back `[]`.
    3. Derive the grouping key (`config["group_col"]`) from `config["group_source_col"]`
       if the incoming dataset doesn't already provide it directly.

    Mutates `config` in place to record the discovered embedding column names under the
    `_embedding_feature_cols` key (leading underscore: derived, not something to set by
    hand) - `build_preprocessor`/`build_tree_preprocessor` read it back. Nothing here fits
    a scaler, imputer, or encoder; that happens per-fold, after the split, in
    `build_pipeline`/`build_tree_pipeline`.
    """
    df = df.copy()
    target_col = config["target_col"]
    keep_labels = [config["positive_label"], *config["negative_labels"]]
    df = df[df[target_col].isin(keep_labels)].copy()
    df["y"] = (df[target_col] == config["positive_label"]).astype(int)

    if config.get("expand_embedding", True):
        emb_col = config["embedding_col"]
        emb_matrix = np.vstack(df[emb_col].to_numpy())
        emb_cols = [f"emb_{i}" for i in range(emb_matrix.shape[1])]
        # pd.concat, not a per-column df[emb_cols] = emb_matrix assignment - the latter
        # inserts one column at a time under the hood at this width (300+ embedding dims)
        # and pandas warns loudly (PerformanceWarning: highly fragmented) for every one
        emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
        df = pd.concat([df, emb_df], axis=1)
        config["_embedding_feature_cols"] = emb_cols
    else:
        config["_embedding_feature_cols"] = []

    group_col = config["group_col"]
    if group_col not in df.columns:
        df[group_col] = derive_first_author(df[config["group_source_col"]])

    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """The one place that decides how each declared column group becomes safe input for
    LogisticRegression: median-impute + scale the numeric/embedding columns, most-frequent
    impute + one-hot encode any categorical columns CONFIG names (empty list is fine - no
    categorical branch is built at all if so). Every step is a real scikit-learn
    Transformer, not a manual fit/transform call, so wrapping this in the Pipeline
    `build_pipeline` returns guarantees every fitted statistic (imputer medians, scaler
    mean/std, one-hot categories) is fit on whatever rows `.fit()` is called with, and only
    those rows. This function can't enforce which rows that is - the caller (the pipeline
    notebooks) still has to call `.fit()` with training rows alone, at the correct point
    in the split.
    """
    numeric_cols = config["_embedding_feature_cols"] + config["numeric_feature_cols"]
    categorical_cols = config["categorical_feature_cols"]

    transformers = [
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ]
    if categorical_cols:
        transformers.append((
            "categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols,
        ))
    return ColumnTransformer(transformers)


def build_pipeline(config: dict) -> Pipeline:
    """Preprocessing + LogisticRegression as one fit-once object.

    Calling `.fit(X_train, y_train)` on the returned Pipeline fits the imputer/scaler/
    encoder AND the classifier on exactly those rows in one call - the standard
    scikit-learn idiom for "fit only on train," instead of hand-rolling separate
    fit/transform calls per fold (what the earlier fold-strategy notebooks did) and
    risking a step getting fit on the wrong rows as the feature set grows. `class_weight=
    "balanced"` matches every other classifier in this repo's `sf_*` notebook lineage.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(config)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _cast_categoricals(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Cast the declared categorical columns to pandas `category` dtype, leave everything
    else untouched (including NaNs - gradient-boosted trees split on "is this missing"
    natively, so filling it in with `SimpleImputer` would throw away real signal, not just
    tidy the data - this repo's own NULL != 0 convention applies here too)."""
    X = X.copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    return X


def build_tree_preprocessor(config: dict) -> FunctionTransformer:
    """The tree-ensemble counterpart to `build_preprocessor`: CatBoost (and LightGBM/
    XGBoost) need no scaling and no imputation - tree splits are scale-invariant, and
    modern gradient boosters handle missing numeric values as a first-class case, usually
    better than any fixed imputed value would. The only real preprocessing step left is
    casting `config["categorical_feature_cols"]` to pandas `category` dtype, so CatBoost
    (or whichever gradient-boosted tree library `build_tree_pipeline` is pointed at)
    recognises them as categorical natively instead of (wrongly) treating them as
    continuous numeric codes. Wrapped as a `FunctionTransformer`, not a bare function, so
    it still composes inside a `Pipeline` the same way `build_preprocessor`'s
    `ColumnTransformer` does - "fit only on train" is moot for this particular step (it
    fits nothing), but keeping the same Pipeline shape means a future numeric transform
    (e.g. a log-transform on a skewed count) slots in later without restructuring anything
    upstream.
    """
    categorical_cols = config["categorical_feature_cols"]
    return FunctionTransformer(
        _cast_categoricals, kw_args={"categorical_cols": categorical_cols}
    )


def build_tree_pipeline(config: dict, model=None) -> Pipeline:
    """Preprocessing (categorical casting only, see `build_tree_preprocessor`) +
    a gradient-boosted tree classifier, as one fit-once object - same calling convention
    as `build_pipeline`, so a fold loop written against one works against the other
    unchanged; only the CONFIG and the model differ.

    `model` defaults to a `catboost.CatBoostClassifier`, not LightGBM/XGBoost - chosen
    specifically because `RandomForestClassifier` and `HistGradientBoostingClassifier`
    (already tested on this exact dataset, in `sf_logo_fold_strategy.ipynb`/
    `sf_logo_fold_pipeline.ipynb`) both hit **train AUC = 1.000 in every LOGO rotation**:
    severe overfitting driven in part by ordinary gradient boosting computing each tree's
    target statistics from the full training set, a well-documented source of in-training
    target leakage on smaller datasets. CatBoost's ordered boosting computes those
    statistics on a permuted prefix instead, specifically to suppress that leakage -
    `boosting_type="Ordered"` is pinned explicitly below rather than left to CatBoost's
    own (also-Ordered-by-default-at-this-size) heuristic, so the choice is guaranteed, not
    incidental. `auto_class_weights="Balanced"` is CatBoost's equivalent of every other
    classifier's `class_weight="balanced"` in this repo's `sf_*` lineage. `cat_features`
    is read from `config["categorical_feature_cols"]` directly - CatBoost needs to be told
    which columns are categorical (by name, matching the columns it's actually fit on),
    the same information `build_tree_preprocessor` uses to decide what to cast.

    Importing catboost only inside this function, not at module level, so the rest of
    this module - and every LogisticRegression-only notebook that imports it - has no hard
    dependency on a package the LR pipelines never use. Pass a pre-configured estimator
    (e.g. a LightGBM/XGBoost equivalent) to swap it out without touching this function.
    """
    if model is None:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            cat_features=config["categorical_feature_cols"] or None,
            auto_class_weights="Balanced",
            boosting_type="Ordered",
            random_state=config.get("random_state", 0),
            verbose=False,
        )
    return Pipeline([
        ("preprocess", build_tree_preprocessor(config)),
        ("clf", model),
    ])


def ranking_metrics(y_true, scores, fractions=(0.10, 0.20), wss_target_recall=0.95) -> dict:
    """recall_at_Xpct / wss_at_95 - identical definition to sf_eda_v2.ipynb §13 and
    sf_logo_fold_strategy.ipynb, centralised here so both pipeline notebooks share one
    implementation instead of two copies that can silently drift apart."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    n_pos = int(y_true.sum())
    order = np.argsort(-np.asarray(scores))
    cum_pos = np.cumsum(y_true[order])
    out = {}
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        out[f"recall_at_{int(frac * 100)}pct"] = float(cum_pos[k - 1] / n_pos)
    target_count = int(np.ceil(wss_target_recall * n_pos))
    cutoff_idx = int(np.searchsorted(cum_pos, target_count, side="left"))
    screened = cutoff_idx + 1
    out["wss_at_95"] = float((n - screened) / n - (1 - wss_target_recall))
    return out


def classification_metrics(y_true, proba_pos, pred) -> dict:
    """ROC-AUC, Average Precision, Recall, F2 (beta=2, per reports/wf_ensemble_report.md
    §2 - recall weighted over precision for this screening use case) - the standard metric
    bundle both pipeline notebooks report at every stage (train / validation / holdout)."""
    return {
        "auc": float(roc_auc_score(y_true, proba_pos)),
        "ap": float(average_precision_score(y_true, proba_pos)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
    }


def bootstrap_auc_ci(y_true, scores, n_boot=1000, random_state=0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on ROC-AUC, resampling rows of already-computed
    predictions (no refitting) - cheap enough to run for every fold/rotation, and
    necessary at a few-hundred-row holdout where a bare point estimate invites
    over-reading a small gap between stages or models as a real difference."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)
