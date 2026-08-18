"""modal_benchset_v1_ensemble.py — trains the 08_ensemble_pooled.ipynb stacking architecture
(CatBoost + LogReg + RandomForest -> LogisticRegression meta-learner) on `train_filename`'s
rows CONCATENATED with `papers_fe.parquet`'s rows, entirely on Modal.

Why this exists: `benchset_v1_large_set_a.parquet` / `_large_set_b.parquet` are 62k/82k rows
by ~4,600 columns (~1.0GB/~1.4GB) - one to two orders of magnitude bigger than
`papers_fe.parquet` (1,848 rows), which the notebooks this project already fits locally
without issue. Each fold fit involves 3 base models, each owning its own PCA(50) on the
~4,600-dim embedding block, at this row count - CONTEXT.md's own note that CatBoost fits on
this project route through Modal, not locally, applies with even more force here. Every
hyperparameter below (RF_PARAMS, CB_PARAMS, META_C, DECISION_THRESHOLD) is copied verbatim
from `08_ensemble_pooled.ipynb`'s own tuning run (its §5 catalog, also carried into
`10_logo_ensemble.ipynb` unchanged) - **nothing is retuned here**, per the project's own rule
that hyperparameter search happens once, centrally (CONTEXT.md §1).

Feature columns are the common intersection across all three parquet files this module reads
(`benchset_v1_large_set_a/b.parquet`, `benchset_v1_small_test.parquet`, `papers_fe.parquet`):
`papers_fe.parquet` carries a Qwen3-Embedding-8B block (`emb_qwen8b_*`, `cos_brief_qwen8b`,
`rank_cos_brief_qwen8b`) plus `has_abstract`/`lex_has_exclude_terms` that benchset_v1 doesn't
have; benchset_v1 carries `cos_briefpre_*`/`rank_cos_briefpre_*` (a brief variant) that
papers_fe doesn't have. Dropping both non-overlapping groups is the only way one model fit on
the combined pool can be scored on `benchset_v1_small_test.parquet` without retraining - see
NON_EMBEDDING_FEATURE_COLS below for the exact resulting list.

This module does the TRAINING only (load, combine, fold-validate, refit, score) and returns
raw probabilities/labels + metrics as a JSON-safe dict. All hypothesis testing, tables and
plots happen in the calling notebook, matching how 08_ensemble_pooled.ipynb and
10_logo_ensemble.ipynb split "compute" from "analysis" elsewhere in this project.

`run_ensemble_combined`'s design, deliberately different from a plain pooled/holdout split:
`train_filename`'s rows are CONCATENATED with `papers_fe.parquet`'s rows into one training
pool (papers_fe is training data here, not a test surface) - no outer holdout at all. Fold
validation is a single StratifiedGroupKFold(5) over the entire combined pool; the final model,
refit on 100% of it, is scored only against `benchset_v1_small_test.parquet` (the one file
neither half of the training data has ever touched).

**Note on repo history:** an earlier version of this module also had a `run_ensemble` design
(benchset_v1 alone, with an outer holdout plus `papers_fe.parquet`/`benchset_v1_small_test.parquet`
as external test surfaces) and a `main` CLI entrypoint for it. Both were removed — per the
project owner's explicit instruction — once the combined-training-set design above became the
only one worth keeping; the notebooks and cached results for the benchset_v1-alone design were
deleted at the same time. If you're looking for that design, it's gone deliberately, not by
accident.

Deploy once (or after any edit here):
    modal deploy scripts/modal_benchset_v1_ensemble.py

Upload the four parquet files this module reads (once, or after they change locally):
    modal run scripts/modal_benchset_v1_ensemble.py::upload_data

Run end-to-end from the CLI (writes reports/sf_ensemble_<training_file_stem>_plus_papers_fe.json):
    modal run scripts/modal_benchset_v1_ensemble.py::main_combined --training-file benchset_v1_large_set_a.parquet
    modal run scripts/modal_benchset_v1_ensemble.py::main_combined --training-file benchset_v1_large_set_b.parquet

Or call it directly from a notebook (after `modal deploy`, see above):
    import modal
    fn_combined = modal.Function.from_name("tiri-benchset-v1-ensemble", "run_ensemble_combined")
    result_combined = fn_combined.remote("benchset_v1_large_set_a.parquet")
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tiri-benchset-v1-ensemble"
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed"
DATA_MOUNT = "/data"

TRAIN_FILENAMES = ("benchset_v1_large_set_a.parquet", "benchset_v1_large_set_b.parquet")
PAPERS_FE_FILENAME = "papers_fe.parquet"           # folded INTO training by run_ensemble_combined
SMALL_TEST_FILENAME = "benchset_v1_small_test.parquet"  # the only test surface run_ensemble_combined uses

CPU_PER_CONTAINER = 16
MEMORY_PER_CONTAINER = 32768  # MB. Generous, not measured yet - PCA(50) on a ~4,600-dim
                              # block over a ~64k-84k row combined pool is a real SVD, and
                              # CatBoost's own fold fits run in parallel via joblib inside the
                              # container (see run_ensemble_combined below) - revisit if this
                              # OOMs or times out.
FIT_TIMEOUT = 21600  # 6h. Calibrated (not guessed) off a local smoke test at CB_PARAMS's real
                     # values: a single CatBoost fold fit at thread_count=1 took ~274s on
                     # ~3,100 rows (a 20x-smaller sample than a real inner fold here). CatBoost
                     # fit cost is superlinear in rows, so this project's own "generous
                     # timeout, then measure" convention (modal_ensemble_candidate.py) argues
                     # for a wide margin on the first real run - narrow this once one actually
                     # completes and its wall-clock is known.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.2.0", "pyarrow>=16.0.0", "numpy>=2.0.0",
        "scikit-learn>=1.4.0", "catboost>=1.2.0", "joblib>=1.3.0",
    )
)

# New volume, deliberately separate from the existing `tiri-fe-data` volume other Modal
# scripts in this repo use - this work's ~2.8GB of benchset_v1 data (plus its own copy of
# papers_fe.parquet, so this module has no dependency on what's already uploaded elsewhere)
# stays self-contained.
data_vol = modal.Volume.from_name("tiri-benchset-v1-data", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

TARGET_COL = "triage_label"
POSITIVE_LABEL = "positive"
NEGATIVE_LABELS = ["negative"]
USE_CASE_COL = "use_case_key"
GROUP_COL = "first_author"

# The common intersection across benchset_v1_large_set_a/b.parquet, benchset_v1_small_test.parquet,
# and papers_fe.parquet - see this file's module docstring for exactly what was dropped and why.
NON_EMBEDDING_FEATURE_COLS = [
    "lex_n_tokens", "lex_bm25_obj", "lex_bm25_prob", "lex_bm25_must", "lex_bm25_nice", "lex_bm25_dom",
    "lex_rank_bm25_obj", "lex_rank_bm25_prob", "lex_rank_bm25_must", "lex_rank_bm25_nice", "lex_rank_bm25_dom",
    "lex_overlap_must_n", "lex_overlap_must_frac", "lex_rank_overlap_must",
    "lex_overlap_nice_n", "lex_overlap_nice_frac", "lex_rank_overlap_nice",
    "lex_overlap_excl_n", "lex_overlap_excl_frac",
    "lex_overlap_must_per_1k", "lex_overlap_nice_per_1k",
    "cos_brief_jasper", "rank_cos_brief_jasper",
    "cos_brief_qwen4b", "rank_cos_brief_qwen4b",
    "year", "paper_age", "n_authors", "citation_count",
]

PCA_N_COMPONENTS = 50
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5
RANDOM_STATE = 0
CB_THREAD_COUNT = 1  # for the 5 parallel inner-fold fits; the lone final refit overrides this

# --- 08_ensemble_pooled.ipynb's own winning hyperparameters (its §5 catalog), copied verbatim -
# --- also reused unchanged by 10_logo_ensemble.ipynb. Nothing here is retuned.
RF_PARAMS = {"n_estimators": 527, "max_depth": 13, "min_samples_leaf": 7, "max_features": "sqrt"}
CB_PARAMS = {"depth": 6, "learning_rate": 0.06119889729184578, "l2_leaf_reg": 7.474245667651606, "iterations": 656}
META_C = 0.1
DECISION_THRESHOLD = 0.2


def _load_and_label(path: str):
    import pandas as pd

    df = pd.read_parquet(path)
    df["lex_overlap_excl_n"] = df["lex_overlap_excl_n"].fillna(0.0)
    df["lex_overlap_excl_frac"] = df["lex_overlap_excl_frac"].fillna(0.0)
    df = df[df[TARGET_COL].isin([POSITIVE_LABEL, *NEGATIVE_LABELS])].copy()
    df["y"] = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    return df


def _build_preprocessor(embedding_cols, non_embedding_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    embedding_branch = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_N_COMPONENTS, whiten=True, random_state=RANDOM_STATE)),
    ])
    non_embedding_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("embedding_pca", embedding_branch, embedding_cols),
        ("non_embedding", non_embedding_branch, non_embedding_cols),
    ])


def _build_logreg_pipeline(embedding_cols, non_embedding_cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", _build_preprocessor(embedding_cols, non_embedding_cols)), ("clf", clf)])


def _build_catboost_pipeline(embedding_cols, non_embedding_cols, thread_count=CB_THREAD_COUNT):
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    clf = CatBoostClassifier(
        auto_class_weights="Balanced", boosting_type="Ordered", random_state=RANDOM_STATE,
        thread_count=thread_count, verbose=False, allow_writing_files=False, **CB_PARAMS,
    )
    return Pipeline([("preprocess", _build_preprocessor(embedding_cols, non_embedding_cols)), ("clf", clf)])


def _build_rf_pipeline(embedding_cols, non_embedding_cols):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    # n_jobs=-1 is safe here (unlike CatBoost's thread_count): every RF fit in
    # run_ensemble_combined below runs alone, one at a time, never concurrently with another
    # RF fit - no oversubscription risk, just free parallelism across RF_PARAMS's 527 trees.
    clf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **RF_PARAMS)
    return Pipeline([("preprocess", _build_preprocessor(embedding_cols, non_embedding_cols)), ("clf", clf)])


def _score(y_true, proba_pos, threshold):
    import numpy as np
    from sklearn.metrics import (
        average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score,
    )

    pred = (np.asarray(proba_pos) >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba_pos)),
        "ap": float(average_precision_score(y_true, proba_pos)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2_at_0.5": float(fbeta_score(y_true, (np.asarray(proba_pos) >= 0.5).astype(int), beta=2, zero_division=0)),
        "f2_at_threshold": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
    }


def _ranking_metrics(y_true, scores_arr, fractions=(0.10, 0.20), wss_target_recall=0.95):
    import numpy as np

    y_true = np.asarray(y_true)
    n = len(y_true)
    n_pos = int(y_true.sum())
    order = np.argsort(-np.asarray(scores_arr))
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


def _fit_predict_fold(builder_fn, pool, feature_cols, tr_idx, te_idx):
    pipe = builder_fn()
    fold_train = pool.iloc[tr_idx]
    pipe.fit(fold_train[feature_cols], fold_train["y"])
    pos_col = list(pipe.classes_).index(1)
    proba = pipe.predict_proba(pool.iloc[te_idx][feature_cols])[:, pos_col]
    return te_idx, proba


def _build_oof(builder_fn, pool, inner_splits, feature_cols, n_jobs):
    import numpy as np
    from joblib import Parallel, delayed

    if n_jobs == 1:
        results = [_fit_predict_fold(builder_fn, pool, feature_cols, tr, te) for tr, te in inner_splits]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict_fold)(builder_fn, pool, feature_cols, tr, te) for tr, te in inner_splits
        )
    oof = np.full(len(pool), np.nan)
    for te_idx, proba in results:
        oof[te_idx] = proba
    assert not np.isnan(oof).any()
    return oof


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=FIT_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_ensemble_combined(train_filename: str) -> dict:
    """Trains the ensemble on `train_filename`'s rows CONCATENATED with `papers_fe.parquet`'s
    rows into one training pool - a deliberately different design from run_ensemble (see this
    module's docstring). No outer holdout: fold validation is a single StratifiedGroupKFold(5)
    (grouped by first_author, stratified on use_case_key + y) over the ENTIRE combined pool,
    giving leakage-safe out-of-fold validation metrics exactly like run_ensemble's inner split
    did, just spanning the whole training set rather than an 80% slice of it. The final model
    is refit on 100% of the combined pool and scored only against `benchset_v1_small_test.parquet`
    - papers_fe is training data here, not a test surface, so (unlike run_ensemble) it plays no
    further role after this point. All hyperparameters are frozen (see module docstring).
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    df_train = _load_and_label(f"{DATA_MOUNT}/{train_filename}")
    embedding_cols = [c for c in df_train.columns if c.startswith("emb_")]
    missing = [c for c in NON_EMBEDDING_FEATURE_COLS if c not in df_train.columns]
    if missing:
        raise ValueError(f"Expected non-embedding feature column(s) not found in {train_filename}: {missing}")
    feature_cols = embedding_cols + NON_EMBEDDING_FEATURE_COLS

    df_fe = _load_and_label(f"{DATA_MOUNT}/{PAPERS_FE_FILENAME}")
    missing_fe = [c for c in feature_cols if c not in df_fe.columns]
    if missing_fe:
        raise ValueError(f"{PAPERS_FE_FILENAME} is missing feature column(s) required by this model: {missing_fe}")

    keep_cols = feature_cols + ["y", USE_CASE_COL, GROUP_COL]
    combined = pd.concat([df_train[keep_cols], df_fe[keep_cols]], ignore_index=True)
    n_from_train_file, n_from_papers_fe = len(df_train), len(df_fe)

    strat_key = combined[USE_CASE_COL].astype(str) + "__" + combined["y"].astype(str)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(combined, strat_key, groups=combined[GROUP_COL]))

    builders = {
        "logreg": lambda: _build_logreg_pipeline(embedding_cols, NON_EMBEDDING_FEATURE_COLS),
        "rf": lambda: _build_rf_pipeline(embedding_cols, NON_EMBEDDING_FEATURE_COLS),
        "catboost": lambda: _build_catboost_pipeline(embedding_cols, NON_EMBEDDING_FEATURE_COLS),
    }
    oof = {
        "logreg": _build_oof(builders["logreg"], combined, splits, feature_cols, n_jobs=1),
        "rf": _build_oof(builders["rf"], combined, splits, feature_cols, n_jobs=1),
        "catboost": _build_oof(builders["catboost"], combined, splits, feature_cols, n_jobs=CPU_PER_CONTAINER),
    }
    meta_feature_names = ["catboost", "logreg", "rf"]
    meta_X_oof = np.column_stack([oof[name] for name in meta_feature_names])
    y_all = combined["y"].to_numpy()

    meta_oof_proba = np.full(len(combined), np.nan)
    for tr_idx, te_idx in splits:
        meta_fold = LogisticRegression(C=META_C, class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        meta_fold.fit(meta_X_oof[tr_idx], y_all[tr_idx])
        pos_col = list(meta_fold.classes_).index(1)
        meta_oof_proba[te_idx] = meta_fold.predict_proba(meta_X_oof[te_idx])[:, pos_col]
    assert not np.isnan(meta_oof_proba).any()

    validation_rows = []
    for tr_idx, te_idx in splits:
        validation_rows.append(_score(y_all[te_idx], meta_oof_proba[te_idx], DECISION_THRESHOLD))

    # OOF quality of each base branch alone (pre-stacking), same threshold, for the
    # "does stacking actually help" comparison in the calling notebook.
    branch_oof_metrics = {name: _score(y_all, oof[name], DECISION_THRESHOLD) for name in meta_feature_names}

    # --- refit every base model + the meta-learner once on the WHOLE combined pool ---
    final_base = {
        "logreg": builders["logreg"](),
        "rf": builders["rf"](),
        "catboost": _build_catboost_pipeline(embedding_cols, NON_EMBEDDING_FEATURE_COLS, thread_count=-1),
    }
    for pipe in final_base.values():
        pipe.fit(combined[feature_cols], combined["y"])

    final_meta = LogisticRegression(C=META_C, class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    final_meta.fit(meta_X_oof, y_all)
    meta_pos_col = list(final_meta.classes_).index(1)
    meta_coefs = {name: float(c) for name, c in zip(meta_feature_names, final_meta.coef_[0])}

    def _branch_probas(rows):
        return {
            name: final_base[name].predict_proba(rows[feature_cols])[:, list(final_base[name].classes_).index(1)]
            for name in meta_feature_names
        }

    def _score_surface(rows):
        branch_proba = _branch_probas(rows)
        meta_X = np.column_stack([branch_proba[name] for name in meta_feature_names])
        proba = final_meta.predict_proba(meta_X)[:, meta_pos_col]
        y_true = rows["y"].to_numpy()
        out = _score(y_true, proba, DECISION_THRESHOLD)
        out.update(_ranking_metrics(y_true, proba))
        out["y"] = y_true.tolist()
        out["proba"] = proba.tolist()
        out["n"] = int(len(rows))
        out["pos_rate"] = float(y_true.mean())
        # each base branch's own (un-stacked) prediction on this surface, at the same fixed
        # threshold - lets the calling notebook test "does stacking actually help" per surface.
        out["branch_proba"] = {name: p.tolist() for name, p in branch_proba.items()}
        out["branch_metrics"] = {name: _score(y_true, p, DECISION_THRESHOLD) for name, p in branch_proba.items()}
        return out

    train_branch_proba = _branch_probas(combined)
    train_meta_X = np.column_stack([train_branch_proba[name] for name in meta_feature_names])
    train_proba = final_meta.predict_proba(train_meta_X)[:, meta_pos_col]
    train_metrics = _score(y_all, train_proba, DECISION_THRESHOLD)

    small_test_df = _load_and_label(f"{DATA_MOUNT}/{SMALL_TEST_FILENAME}")
    missing_st = [c for c in feature_cols if c not in small_test_df.columns]
    if missing_st:
        raise ValueError(f"{SMALL_TEST_FILENAME} is missing feature column(s) required by this model: {missing_st}")

    return {
        "train_filename": train_filename,
        "combined_with": PAPERS_FE_FILENAME,
        "test_filename": SMALL_TEST_FILENAME,
        "n_total": int(len(combined)),
        "n_from_train_file": int(n_from_train_file),
        "n_from_papers_fe": int(n_from_papers_fe),
        "n_features": len(feature_cols),
        "n_embedding_cols": len(embedding_cols),
        "use_cases": sorted(combined[USE_CASE_COL].unique().tolist()),
        "hyperparameters": {"RF_PARAMS": RF_PARAMS, "CB_PARAMS": CB_PARAMS, "META_C": META_C,
                             "DECISION_THRESHOLD": DECISION_THRESHOLD, "PCA_N_COMPONENTS": PCA_N_COMPONENTS},
        "meta_coefficients": meta_coefs,
        "train_metrics": train_metrics,
        "validation_rows": validation_rows,
        "branch_oof_metrics": branch_oof_metrics,
        # Raw out-of-fold arrays (not just aggregated validation_rows/branch_oof_metrics) -
        # there is no holdout in this design, so the calling notebook's paired-bootstrap
        # hypothesis tests (ensemble vs. branch alone, threshold effect) run on THESE
        # cross-fitted predictions instead of a held-out surface. Every row's prediction
        # here already comes from a model that never saw that row (same leakage argument
        # as run_ensemble's holdout), so a paired test on them is exactly as valid.
        "y_validation": y_all.tolist(),
        "meta_oof_proba": meta_oof_proba.tolist(),
        "branch_oof_proba": {name: arr.tolist() for name, arr in oof.items()},
        "test": _score_surface(small_test_df),
    }


def _upload_file(local_path: Path, remote_name: str) -> None:
    print(f"Uploading {local_path} ({local_path.stat().st_size / 1e6:.1f} MB) to "
          f"tiri-benchset-v1-data as /{remote_name} ...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), f"/{remote_name}")
    print("  done.")


@app.local_entrypoint()
def upload_data() -> None:
    """Uploads all four parquet files this module reads. Run once, or again after any of
    them changes locally:
        modal run scripts/modal_benchset_v1_ensemble.py::upload_data
    """
    for filename in (*TRAIN_FILENAMES, SMALL_TEST_FILENAME, PAPERS_FE_FILENAME):
        local_path = DATA_DIR / filename
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} not found - nothing to upload for it.")
        _upload_file(local_path, filename)


@app.local_entrypoint()
def main_combined(training_file: str = "benchset_v1_large_set_a.parquet") -> None:
    """CLI entrypoint: runs run_ensemble_combined remotely and writes the result JSON to
    reports/, for testing this module independently of the notebooks that otherwise call
    run_ensemble_combined directly via modal.Function.from_name(...).remote(...).
    """
    if training_file not in TRAIN_FILENAMES:
        raise ValueError(f"training_file must be one of {TRAIN_FILENAMES}, got {training_file!r}")

    print(f"Running the ensemble on {training_file} combined with {PAPERS_FE_FILENAME} on Modal "
          f"({CPU_PER_CONTAINER} CPUs, {MEMORY_PER_CONTAINER}MB) ...")
    result = run_ensemble_combined.remote(training_file)

    stem = Path(training_file).stem
    out_path = REPO / "reports" / f"sf_ensemble_{stem}_plus_papers_fe.json"
    out_path.write_text(json.dumps(result))
    test = result["test"]
    print(f"n_total={result['n_total']:,} ({result['n_from_train_file']:,} from {training_file} + "
          f"{result['n_from_papers_fe']:,} from {PAPERS_FE_FILENAME})")
    print(f"test ({SMALL_TEST_FILENAME}): auc={test['auc']:.3f} f2@0.2={test['f2_at_threshold']:.3f} "
          f"recall={test['recall']:.3f} precision={test['precision']:.3f}")
    print(f"\nWritten to {out_path}")
