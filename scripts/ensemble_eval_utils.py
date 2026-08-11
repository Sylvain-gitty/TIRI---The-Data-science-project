"""Shared LOGO / within-silo evaluation harness behind scripts/run_tier1b_control.py and
scripts/run_ensemble_candidate.py.

Extracted out of run_tier1b_control.py so a second script (comparing base-learner choices
instead of feature blocks) reuses the exact same fold construction and scoring instead of a
second copy that can silently drift apart — same reasoning embedding_utils.py/
latent_space_utils.py/fold_pipeline_utils.py already give for their own shared logic.

`logo`/`within_silo` took a hardcoded LogisticRegression pipeline before this extraction;
they now take a `model_fn` (a zero-arg callable returning a fresh, unfitted sklearn-
compatible estimator) so the same two evaluation surfaces work for any base learner. Passing
run_tier1b_control.py's own `model()` reproduces its published numbers unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def to_md(frame: pd.DataFrame, index_name: str = "") -> str:
    """Minimal markdown table. Hand-rolled rather than pulling in `tabulate` for one
    call — the repo has no test/build tooling and this is not worth a dependency."""
    frame = frame.reset_index()
    frame.columns = [index_name if i == 0 and not str(c).strip() else str(c)
                     for i, c in enumerate(frame.columns)]
    cells = [[f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in frame.itertuples(index=False)]
    header = list(frame.columns)
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) if cells else len(header[i])
              for i in range(len(header))]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join(lines)


def scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC for ranking; Recall@10% because that is the operational
    question (how much does the reviewer find in the first tenth of the pile); F2 at the
    default threshold as a reminder that raw probabilities are uncalibrated out-of-domain."""
    k = max(1, int(round(0.10 * len(y_true))))
    top_k = np.argsort(-y_score)[:k]
    n_pos = int(y_true.sum())
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "recall_at_10pct": float(y_true[top_k].sum() / n_pos) if n_pos else np.nan,
        "f2_at_0.5": fbeta_score(y_true, (y_score >= 0.5).astype(int), beta=2, zero_division=0),
    }


def logo(X: pd.DataFrame, y: np.ndarray, use_case: pd.Series, model_fn) -> pd.DataFrame:
    """Leave-one-use-case-out. Everything is fitted inside the fold.

    Chooses central defaults (which feature/model combination to ship) — not a production
    estimate, since customers are siloed and no pooled model ever ships (CONTEXT.md §1).
    """
    rows = []
    for uc in sorted(use_case.unique()):
        test = (use_case == uc).to_numpy()
        pipe = model_fn().fit(X[~test], y[~test])
        proba = pipe.predict_proba(X[test])[:, 1]
        rows.append({"use_case": uc, **scores(y[test], proba)})
    return pd.DataFrame(rows).set_index("use_case")


def within_silo(
    X: pd.DataFrame, y: np.ndarray, use_case: pd.Series, groups: pd.Series, seeds: int, model_fn
) -> pd.DataFrame:
    """Grouped, stratified k-fold inside each use case — the production shape.

    Grouped by first author so the same author's papers never straddle the split, and
    repeated across seeds because 260-360 rows per silo makes any single split noisy.
    """
    rows = []
    for uc in sorted(use_case.unique()):
        mask = (use_case == uc).to_numpy()
        Xu, yu, gu = X[mask], y[mask], groups[mask].to_numpy()
        per_seed = []
        for seed in range(seeds):
            oof = np.full(len(yu), np.nan)
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in splitter.split(Xu, yu, groups=gu):
                pipe = model_fn().fit(Xu.iloc[train_idx], yu[train_idx])
                oof[test_idx] = pipe.predict_proba(Xu.iloc[test_idx])[:, 1]
            per_seed.append(scores(yu, oof))
        mean = pd.DataFrame(per_seed).mean().to_dict()
        mean["roc_auc_sd"] = float(pd.DataFrame(per_seed)["roc_auc"].std())
        rows.append({"use_case": uc, **mean})
    return pd.DataFrame(rows).set_index("use_case")


def within_silo_oof(
    X: pd.DataFrame, y: np.ndarray, use_case: pd.Series, groups: pd.Series, seed: int, model_fn
) -> dict[str, np.ndarray]:
    """Single-seed within-silo OOF probabilities per use case — the raw predictions behind
    `within_silo`'s aggregated scores, needed by diagnostics that need actual predictions
    rather than a summary metric (calibration curves, branch-disagreement correlation,
    Recall@k curves). One seed, not repeated across several: a diagnostic visualization
    doesn't need the same repeat-and-average discipline as a number being compared against
    the noise floor (CONTEXT.md §5) — that discipline already happened in Phase 1.
    """
    out: dict[str, np.ndarray] = {}
    for uc in sorted(use_case.unique()):
        mask = (use_case == uc).to_numpy()
        Xu, yu, gu = X[mask], y[mask], groups[mask].to_numpy()
        oof = np.full(len(yu), np.nan)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(Xu, yu, groups=gu):
            pipe = model_fn().fit(Xu.iloc[train_idx], yu[train_idx])
            oof[test_idx] = pipe.predict_proba(Xu.iloc[test_idx])[:, 1]
        out[uc] = oof
    return out


def within_silo_oof_with_axis(
    X_other: pd.DataFrame, X_emb: pd.DataFrame, y: np.ndarray, use_case: pd.Series,
    groups: pd.Series, seed: int, model_fn,
) -> dict[str, np.ndarray]:
    """Like `within_silo_oof`, but adds ONE extra feature per fold: a supervised
    "relevance axis" — the direction between the positive-class and negative-class mean
    embedding, fit on the training rows only and refit fresh every fold (never seen the
    fold's own test rows), then every row's embedding is projected onto that axis as a
    single scalar (`emb_axis_score`).

    Why this is a different question from PCA-64 (already measured hurting within a silo,
    reports/wf_query_conditioned_findings.md §5): that was an unsupervised, wholesale
    compression of the embedding before concatenating. This is a supervised, single-
    direction *addition* (or, in the "replace" variant a caller can construct by passing an
    empty X_other, a comparison of that one direction against the full embedding) — a
    direction chosen to separate the classes, not to preserve variance.

    X_other: the non-embedding feature columns (lex/cos-brief/metadata), already selected.
    X_emb: the raw embedding columns only, same row index as X_other.
    model_fn's own feature-column configuration must expect X_other's columns plus one
    named "emb_axis_score" (X_emb's own columns are never passed to the model directly).
    """
    out: dict[str, np.ndarray] = {}
    for uc in sorted(use_case.unique()):
        mask = (use_case == uc).to_numpy()
        Xo, Xe, yu, gu = X_other[mask], X_emb[mask], y[mask], groups[mask].to_numpy()
        emb_all = Xe.to_numpy()
        oof = np.full(len(yu), np.nan)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(Xo, yu, groups=gu):
            emb_train, y_train = emb_all[train_idx], yu[train_idx]
            axis = emb_train[y_train == 1].mean(axis=0) - emb_train[y_train == 0].mean(axis=0)
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
            X_full = Xo.copy()
            X_full["emb_axis_score"] = emb_all @ axis
            pipe = model_fn().fit(X_full.iloc[train_idx], y_train)
            oof[test_idx] = pipe.predict_proba(X_full.iloc[test_idx])[:, 1]
        out[uc] = oof
    return out
