"""
train_baseline_classifier.py — Week 3 baseline: does a plain LogisticRegression on paper
embeddings predict the analyst's triage label, cross-validated honestly?

WHY THIS MODEL, BY DEFAULT (full trail: reports/model_shortlist.md §5)
--------------------------------------------------------------------------
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — the winner picked after
comparing 4 shortlisted models on 2 real, differently-sized/differently-domain corpora
(scripts/compare_embeddings.py): the only model that was never chance-level and never the
most collapsed on either corpus, it's symmetric (no prefix-config fragility bge/nomic
carry), it matches the real — if small — multilingual fraction present in both corpora,
and it's the SAME model academic_research_agent's own exports already ship precomputed
embeddings for, so this script reuses those vectors directly instead of re-embedding
(see embedding_utils.resolve_paper_vectors) whenever they match --model.

WHAT THIS SCRIPT DOES
----------------------
1. loads a labelled export,
2. gets paper vectors for --model (reused from the export if they match, embedded fresh
   otherwise),
3. cross-validates a LogisticRegression (StratifiedKFold, same fold logic as
   academic_research_agent's model.py and compare_embeddings.py) to get an honest ROC-AUC
   AND out-of-fold predictions — every prediction comes from a fold that never trained on
   that row, so the confusion matrix isn't just the model grading its own homework,
4. reports a confusion matrix + precision/recall/F1 from those out-of-fold predictions,
5. fits ONE final model on all labelled data and scores EVERY paper with a usable vector
   in the export (labelled or not) — so "pass"/unlabelled papers get a predicted
   probability too, the same "score everything embedded" idea as the agent's model.py.

This is a BASELINE — Week 3's own framing (see README.md): the floor an ensemble has to
beat later. No feature engineering beyond the raw embedding, no ensembling here yet —
that scope boundary is intentional, not an oversight.

OUTPUTS (named after the input file, same convention as compare_embeddings.py)
------------------------------------------------------------------------------------
    reports/<data filename>_baseline_metrics.json          headline numbers, confusion
                                                            matrix, full classification report
    reports/<data filename>_baseline_confusion_matrix.png  a small heatmap of the OOF
                                                            confusion matrix
    reports/<data filename>_baseline_scored_pool.csv       every paper with a usable
                                                            vector: its actual label (if
                                                            any) and the model's predicted
                                                            P(positive), sorted highest first

HOW TO RUN IT
--------------
    python scripts/train_baseline_classifier.py --data data/raw/your-export.parquet

Useful flags:
    --model "BAAI/bge-small-en-v1.5"    use a different embedding model as the features
    --label-col review_label            use the review-stage label instead of triage

A LIMITATION TO KNOW ABOUT
-----------------------------
The 0.5 probability threshold used for the confusion matrix/scored labels is a plain
default, not a tuned decision boundary — read predicted_probability in the scored pool as
a ranking signal, and treat the confusion matrix as illustrative of that threshold, not
as the only valid cut point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — this is a batch script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from embedding_utils import (
    MAX_CV_FOLDS,
    NEGATIVE_VALUES,
    POSITIVE_VALUES,
    cross_validated_oof_proba,
    cross_validated_roc_auc,
    drop_empty_rows,
    get_title_abstract,
    load_export,
    resolve_paper_vectors,
)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def plot_confusion_matrix(cm: np.ndarray, model_name: str, roc_auc: float | None, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    roc_txt = f"roc_auc={roc_auc:.3f}" if roc_auc is not None else "roc_auc: n/a"
    ax.set_title(f"Baseline confusion matrix (out-of-fold)\n{model_name}\n{roc_txt}", fontsize=8, wrap=True)
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to an academic_research_agent .parquet or .csv export")
    parser.add_argument("--label-col", default="triage_label", help="Column with positive/negative/pass labels (default: triage_label)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model to use as features (default: {DEFAULT_MODEL})")
    parser.add_argument("--out-metrics", type=Path, default=None, help="Where to save the metrics JSON (default: reports/<data filename>_baseline_metrics.json)")
    parser.add_argument("--out-plot", type=Path, default=None, help="Where to save the confusion matrix figure (default: reports/<data filename>_baseline_confusion_matrix.png)")
    parser.add_argument("--out-scored", type=Path, default=None, help="Where to save the scored pool CSV (default: reports/<data filename>_baseline_scored_pool.csv)")
    args = parser.parse_args()

    if args.out_metrics is None:
        args.out_metrics = Path("reports") / f"{args.data.stem}_baseline_metrics.json"
    if args.out_plot is None:
        args.out_plot = Path("reports") / f"{args.data.stem}_baseline_confusion_matrix.png"
    if args.out_scored is None:
        args.out_scored = Path("reports") / f"{args.data.stem}_baseline_scored_pool.csv"

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    titles, abstracts = get_title_abstract(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)

    print(f"Getting paper vectors for {args.model} ...")
    vectors, elapsed, source_note = resolve_paper_vectors(df, titles, abstracts, args.model)
    print(f"  {source_note} ({vectors.shape[0]} papers, dim={vectors.shape[1]}, {elapsed:.1f}s)")

    if args.label_col not in df.columns:
        raise SystemExit(f"Column '{args.label_col}' not found. Available columns: {list(df.columns)}")

    y_mask = df[args.label_col].isin(POSITIVE_VALUES | NEGATIVE_VALUES).to_numpy()
    y_full = df[args.label_col].isin(POSITIVE_VALUES).astype(int).to_numpy()
    n_pos, n_neg = int(y_full[y_mask].sum()), int(y_mask.sum() - y_full[y_mask].sum())
    print(f"  Labels: {n_pos} positive, {n_neg} negative (column '{args.label_col}')")

    k = min(MAX_CV_FOLDS, n_pos, n_neg)
    if k < 2:
        raise SystemExit(
            f"Not enough labels of both classes to cross-validate honestly "
            f"(have {n_pos} positive, {n_neg} negative — need at least 2 of each)."
        )

    X_labelled, y_labelled = vectors[y_mask], y_full[y_mask]

    roc = cross_validated_roc_auc(X_labelled, y_labelled)
    oof_proba = cross_validated_oof_proba(X_labelled, y_labelled, k)
    oof_pred = (oof_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_labelled, oof_pred)
    report = classification_report(y_labelled, oof_pred, target_names=["negative", "positive"], output_dict=True)

    print("\n" + "=" * 70)
    print("BASELINE: LogisticRegression on embeddings, cross-validated (out-of-fold)")
    print("=" * 70)
    print(f"model: {args.model}")
    print(f"roc_auc: {roc['roc_auc']:.4f} (n_folds={roc['n_folds']})" if roc["roc_auc"] is not None else "roc_auc: n/a")
    print("confusion matrix (rows=actual, cols=predicted) [negative, positive]:")
    print(cm)
    for cls in ["negative", "positive"]:
        r = report[cls]
        print(f"  {cls}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1-score']:.3f} support={int(r['support'])}")

    plot_confusion_matrix(cm, args.model, roc["roc_auc"], args.out_plot)
    print(f"\nSaved confusion matrix to {args.out_plot}")

    metrics = {
        "model": args.model,
        "embedding_source": source_note,
        "n_papers_total": int(len(df)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "roc_auc": roc["roc_auc"],
        "n_folds": roc["n_folds"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.out_metrics}")

    # Fit ONE final model on all labelled data, then score EVERY paper with a vector —
    # labelled or not — mirroring academic_research_agent's own "score everything
    # embedded" approach (model.py:score_papers) rather than only the labelled subset.
    final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    final_clf.fit(X_labelled, y_labelled)
    pos_col = list(final_clf.classes_).index(1)
    all_proba = final_clf.predict_proba(vectors)[:, pos_col]

    scored = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(len(df)),
        "title": df.get("title", ""),
        "actual_label": df[args.label_col],
        "predicted_probability": all_proba,
        "predicted_label": np.where(all_proba >= 0.5, "positive", "negative"),
    }).sort_values("predicted_probability", ascending=False)
    args.out_scored.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_scored, index=False)
    print(f"Saved scored pool ({len(scored)} papers) to {args.out_scored}")


if __name__ == "__main__":
    main()
