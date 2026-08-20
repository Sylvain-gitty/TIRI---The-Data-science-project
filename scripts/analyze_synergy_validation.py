"""analyze_synergy_validation.py — scores an ensemble config (CatBoost 150/4 +
LogisticRegression, 50/50 blend — SYNERGY reviews are new/unseen "use cases", so none of
the per-TIRI-use-case nested weights from run_nested_weight_selection.py transfer; 50/50 is
the correct general default here) against SYNERGY: F2, ROC-AUC, Recall@10/20%, WSS@95.

Reads whichever reports/wf_ensemble_v2_synergy_oof*.json is passed via --oof-json, written by:
    modal run scripts/modal_ensemble_experiments.py::synergy_validation --iterations 150 --depth 4 --seeds 5 [--filename ... --out-name ...]

Usage:
    python scripts/analyze_synergy_validation.py [--label "Jasper+Qwen3-8B"]
        [--oof-json reports/wf_ensemble_v2_synergy_oof_qwen8b.json]
        [--features data/processed/papers_fe_synergy_jasper_qwen8b.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md  # noqa: E402
from fold_pipeline_utils import ranking_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
REVIEWS = ["Sep_2021", "Menon_2022", "van_der_Waal_2022"]


def f2_at(y_true, y_score, t=0.5):
    return fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)


def f2_optimal(y_true, y_score, thresholds=np.linspace(0.05, 0.95, 91)):
    best_t, best_f2 = 0.5, -1.0
    for t in thresholds:
        f2 = f2_at(y_true, y_score, t)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
    return best_t, best_f2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="Jasper+Qwen3-4B")
    ap.add_argument("--oof-json", default="wf_ensemble_v2_synergy_oof.json")
    ap.add_argument("--features", default="papers_fe_synergy.parquet")
    ap.add_argument("--section", default="10")
    args = ap.parse_args()

    raw = json.load(open(REPO / "reports" / args.oof_json, encoding="utf-8"))
    df = pd.read_parquet(REPO / "data" / "processed" / args.features)
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    y_by_review = {r: y[(use_case == r).to_numpy()] for r in REVIEWS}

    n_seeds = len(raw["catboost"])
    print(f"Loaded SYNERGY OOF: {n_seeds} seeds, branches {list(raw.keys())}")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"## {args.section}. SYNERGY validation — {args.label}")
    emit()
    emit(f"Config core (Tier-1b BM25 + raw {args.label} embedding), CatBoost "
         "iterations=150/depth=4 + LogisticRegression, 50/50 blend — SYNERGY reviews are "
         "new/unseen \"use cases\" so none of the per-TIRI-use-case nested weights transfer "
         "here) against SYNERGY's 3 reviews at 1.7-14.8% prevalence, vs. 26-77% across "
         "TIRI's own six pools. No cos_brief or metadata (not available for SYNERGY - same "
         "scope limit `scripts/run_synergy_recall_validation.py` already documents).")
    emit()

    prev = pd.Series({r: y_by_review[r].mean() for r in REVIEWS})
    n = pd.Series({r: len(y_by_review[r]) for r in REVIEWS})
    emit("Review sizes and prevalence:")
    emit()
    emit(to_md(pd.DataFrame({"n": n, "prevalence": prev.round(3)}), "review"))
    emit()

    rows = []
    for review in REVIEWS:
        yu = y_by_review[review]
        for seed in range(n_seeds):
            cb = np.array(raw["catboost"][str(seed)][review])
            lr = np.array(raw["logreg"][str(seed)][review])
            blend = 0.5 * cb + 0.5 * lr
            rm = ranking_metrics(yu, blend, fractions=(0.10, 0.20))
            _, f2_star = f2_optimal(yu, blend)
            rows.append({
                "review": review, "seed": seed,
                "roc_auc": roc_auc_score(yu, blend),
                "f2_at_0.5": f2_at(yu, blend, 0.5),
                "f2_at_t_star": f2_star,
                **rm,
            })
    all_df = pd.DataFrame(rows)
    metric_cols = ["roc_auc", "f2_at_0.5", "f2_at_t_star", "recall_at_10pct", "recall_at_20pct", "wss_at_95"]
    grouped = all_df.groupby("review")[metric_cols].agg(["mean", "std"]).round(3)
    # Flatten the (metric, stat) MultiIndex into one "metric (mean / sd)" string column per
    # metric, rather than to_md() rendering raw ('metric', 'stat') tuples as headers.
    summary = pd.DataFrame(index=grouped.index)
    for col in metric_cols:
        summary[f"{col} (mean/sd)"] = (
            grouped[(col, "mean")].astype(str) + " / " + grouped[(col, "std")].astype(str)
        )
    emit("Per review, mean/sd across seeds:")
    emit()
    emit(to_md(summary, "review"))
    emit()

    # Compare against the existing published SYNERGY baseline (embedding-only / embedding+lexical
    # LogisticRegression, reports/wf_synergy_recall_validation.md) if that report exists.
    emit(f"Overall mean ROC-AUC: **{all_df.roc_auc.mean():.3f}**, mean F2@t*: "
         f"**{all_df.f2_at_t_star.mean():.3f}**, mean Recall@20%: "
         f"**{all_df.recall_at_20pct.mean():.3f}**, mean WSS@95: "
         f"**{all_df.wss_at_95.mean():.3f}**.")
    emit()
    emit("Compare against `reports/wf_synergy_recall_validation.md` §1 (embedding-alone and "
         "embedding+lexical LogisticRegression numbers already published there) for whether "
         "adding CatBoost + the ensemble blend moves the needle on the one surface that "
         "actually matters for production prevalence.")
    emit()

    existing = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
