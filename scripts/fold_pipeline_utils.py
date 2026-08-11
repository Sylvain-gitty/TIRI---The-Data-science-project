"""Shared, config-driven building blocks for the config-driven fold-pipeline
notebooks (notebooks/experiments/sf_logo_fold_pipeline.ipynb and the two templates
in notebooks/future_work/).

sf_generalized_fold_pipeline.ipynb (pooled, whole-dataset splits), sf_logo_fold_pipeline.ipynb
(leave-one-use-case-out), and sf_catboost_fold_pipeline.ipynb all import this module instead
of redefining the same preprocessing/metrics code twice - the point: each notebook's fold
*structure* differs (random grouped/stratified split vs. rotating use-case holdout) and
model differs (LogisticRegression, LogisticRegression-on-PCA-reduced-embeddings, CatBoost),
but "how do we turn a row into something the model can use, without leaking" should live in
exactly one place per model family, so it can't drift between notebooks as the upstream
feature-engineering dataset changes shape. `build_pipeline` (plain LogisticRegression),
`build_pca_pipeline` (LogisticRegression on a PCA-reduced embedding block, for datasets
whose embedding is too wide to feed a linear model directly), and `build_tree_pipeline`
(gradient-boosted trees) are the three model-family entry points; `validate_schema`,
`prepare_dataset`, and the metrics functions are model-agnostic and used by all of them.

Nothing in this module reads data/ or fits anything at import time - every function here is
called explicitly by the notebooks, at the point in their own fold logic where it's safe to
call it (see each function's docstring for which side of the train/test boundary it belongs on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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
    all - a text-driven model (e.g. a prompted LLM, see `notebooks/future_work/
    sf_llm_fold_pipeline.ipynb`) has no use for the embedding column and can set this to
    False rather than being forced to point CONFIG at a column it never reads. When it's
    False AND `config["embedding_feature_cols"]` is set instead (a dataset that already
    ships its embedding(s) as separate wide scalar columns, e.g. `emb_jasper_0000...`,
    rather than one array-valued column), those columns are required instead.
    """
    required = [config["target_col"], config["use_case_col"]]
    if config.get("expand_embedding", True):
        required.append(config["embedding_col"])
    else:
        required += config.get("embedding_feature_cols", [])
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
       code change here. If `False`, `_embedding_feature_cols` is set from
       `config.get("embedding_feature_cols", [])` instead - either `[]` for a model that
       doesn't consume the embedding at all (e.g. a prompted LLM working off raw text), or
       an explicit list of already-wide embedding columns a dataset ships directly (e.g.
       several concatenated embedding models' `emb_<model>_####` columns) - either way,
       nothing here re-derives or reshapes those columns, they're used as-is.
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
        config["_embedding_feature_cols"] = config.get("embedding_feature_cols", [])

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


def build_pipeline(config: dict, model=None) -> Pipeline:
    """Preprocessing + a linear-friendly classifier (default LogisticRegression) as one
    fit-once object.

    Calling `.fit(X_train, y_train)` on the returned Pipeline fits the imputer/scaler/
    encoder AND the classifier on exactly those rows in one call - the standard
    scikit-learn idiom for "fit only on train," instead of hand-rolling separate
    fit/transform calls per fold (what the earlier fold-strategy notebooks did) and
    risking a step getting fit on the wrong rows as the feature set grows. `class_weight=
    "balanced"` matches every other classifier in this repo's `sf_*` notebook lineage.

    `model` defaults to `None` (LogisticRegression) but accepts any estimator that needs
    imputed/scaled numeric input - e.g. SVC - same swap-in convention as
    `build_tree_pipeline`'s own `model=` parameter.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([
        ("preprocess", build_preprocessor(config)),
        ("clf", model),
    ])


def build_pca_preprocessor(config: dict) -> ColumnTransformer:
    """The PCA-reduced-embedding counterpart to `build_preprocessor`, for a dataset whose
    embedding block is too wide to feed a linear model directly (e.g.
    `notebooks/main/06_baseline_logreg.ipynb` against `papers_fe.parquet`'s
    8,704 concatenated embedding columns). Two independently-fit branches:

    - **`config["_embedding_feature_cols"]`** -> `StandardScaler` -> `PCA(n_components=
      config["pca_n_components"], whiten=True)`. Scaling *before* PCA is not optional
      here: several embedding models are concatenated into one block, each with its own
      raw scale/variance convention, and PCA finds directions of maximum *variance* - fit
      it on the raw concatenation and the top components would just be "whichever model
      happens to have the largest raw magnitudes," not whichever directions are actually
      most informative. `whiten=True` rescales each retained component to unit variance
      afterward (sklearn's `PCA` only centers by default, it doesn't equalize component
      scale) - without it, an L2-regularized `LogisticRegression`'s uniform coefficient
      penalty would implicitly shrink the lower-variance (later) components harder than
      the higher-variance (earlier) ones, for a reason that has nothing to do with which
      components actually carry predictive signal.
    - **`config["numeric_feature_cols"]`** (+ `categorical_feature_cols`, one-hot encoded,
      if any) -> median-impute + scale, unchanged from `build_preprocessor` - these are
      ordinary engineered features, not part of the embedding block, and don't go through
      PCA at all.

    Both branches - and every fitted statistic inside them (scaler mean/std, PCA
    components/explained variance, imputer medians) - are fit together inside a single
    `ColumnTransformer`, so calling the `Pipeline` `build_pca_pipeline` returns with
    `.fit(X_train, y_train)` fits all of it on the training rows alone, in one call. The
    same discipline applies here as everywhere else in this module: this function cannot
    enforce *which* rows those are - the caller (the pipeline notebook) still has to call
    `.fit()` only on the current fold/split's training rows, never on pooled data.
    """
    embedding_cols = config["_embedding_feature_cols"]
    other_numeric_cols = config["numeric_feature_cols"]
    categorical_cols = config["categorical_feature_cols"]

    transformers = [
        ("embedding_pca", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=config["pca_n_components"], whiten=True,
                         random_state=config.get("random_state", 0))),
        ]), embedding_cols),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), other_numeric_cols),
    ]
    if categorical_cols:
        transformers.append((
            "categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols,
        ))
    return ColumnTransformer(transformers)


def build_pca_pipeline(config: dict) -> Pipeline:
    """PCA-reduced-embedding preprocessing (see `build_pca_preprocessor`) +
    `LogisticRegression`, as one fit-once object - same calling convention as
    `build_pipeline`/`build_tree_pipeline`, so a fold loop written against any of the
    three works against the others unchanged. Requires `config["pca_n_components"]` and
    `config["_embedding_feature_cols"]` to be set (the latter via `prepare_dataset`, from
    either `embedding_col` expansion or a pre-existing `embedding_feature_cols` list - see
    that function's docstring).
    """
    return Pipeline([
        ("preprocess", build_pca_preprocessor(config)),
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
