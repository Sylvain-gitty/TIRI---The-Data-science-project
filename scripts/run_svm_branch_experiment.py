"""run_svm_branch_experiment.py — does adding SVM as a third branch to the CatBoost(150,4)
+ LogisticRegression ensemble give a genuine edge?

Prompted directly by the user asking "did we miss SVM, as our ensemble is only 2 models" -
this tests it properly rather than reasoning about it from first principles. A linear-kernel
SVM was screened first (single seed) and dropped: it was dominated by an RBF kernel in all 6
use cases (mean within-silo ROC-AUC 0.819 vs 0.859), so only RBF is carried through the full
analysis below.

Three questions, in the same nested/multi-seed discipline as
scripts/run_nested_weight_selection.py and scripts/run_multiseed_verification.py:

1. Standalone: how does SVM(rbf) compare to CatBoost(150,4)/LogisticRegression alone?
2. Diversity: is SVM's OOF probability correlation with the other two branches in the
   healthy 0.40-0.85 disagreement band, or is it redundant with one of them (most likely
   candidate for redundancy: LogisticRegression, since both are just decision boundaries in
   the same ~4600-dim embedding-heavy feature space - RBF's implicit kernel space and a
   linear-in-original-space boundary are not the same hypothesis, but it's an empirical
   question, not an assumed one)?
3. Does a nested (non-leaky) 3-way blend beat the already-established, already-nested 2-way
   CatBoost/LogReg blend (reports/wf_ensemble_v2_chosen_weights.json) by more than this
   project's ~0.03 noise floor, on seeds the weight search never saw?

CatBoost(150,4) OOF is fetched from the already-deployed `tiri-ensemble-ablation` Modal app
(`run_catboost_hparam_oof`) - CatBoost is pathologically slow to fit locally on this Apple
Silicon machine (documented thread-oversubscription bug), unlike LogReg/SVM which fit
locally in seconds at this row count (260-360 rows/silo).

Usage:
    python scripts/run_svm_branch_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md, within_silo_oof  # noqa: E402
from run_ensemble_candidate import build_variants, logreg_fn, svm_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
WEIGHTS_JSON = REPO / "reports" / "wf_ensemble_v2_chosen_weights.json"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
N_SEEDS = 5
SELECTION_SEEDS = [0, 1, 2]
VALIDATION_SEEDS = [3, 4]
NOISE_FLOOR = 0.03

USE_CASE_DISPLAY = {
    "carbon_capture": "Carbon Capture",
    "cement_binders": "Low-Carbon Cement",
    "solar_leo": "Solar Cells for Satellites",
    "soil_microbiome": "Soil Microbiome",
    "ner": "Named Entity Recognition",
    "tech_forecasting": "Technology Prediction",
}


def f2_at(y_true, y_score, t=0.5):
    return fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)


def f2_optimal(y_true, y_score, thresholds=np.linspace(0.05, 0.95, 91)):
    best_t, best_f2 = 0.5, -1.0
    for t in thresholds:
        f2 = f2_at(y_true, y_score, t)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
    return best_t, best_f2


def weight_simplex(step: float = 0.1):
    """All (w_cb, w_lr, w_svm) triples on a step-0.1 grid summing to 1."""
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    out = []
    for w_cb in vals:
        for w_lr in vals:
            w_svm = round(1 - w_cb - w_lr, 2)
            if -1e-9 <= w_svm <= 1 + 1e-9:
                out.append((float(w_cb), float(w_lr), float(max(0.0, w_svm))))
    return out


def main() -> None:
    import json

    df = pd.read_parquet(DATA_PATH)
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]
    USE_CASE_ORDER = sorted(use_case.unique())
    y_by_uc = {uc: y[(use_case == uc).to_numpy()] for uc in USE_CASE_ORDER}
    cols = build_variants(df)[WINNING_VARIANT]
    X = df[cols]

    print(f"Fetching CatBoost (150/4) OOF for {N_SEEDS} seeds ...")
    fn = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_hparam_oof")
    catboost_oof = {}
    for seed, res in enumerate(fn.starmap([(150, 4, s) for s in range(N_SEEDS)])):
        catboost_oof[seed] = {uc: np.array(v) for uc, v in res.items()}
        print(f"  done: catboost seed={seed}")

    print("Fitting LogisticRegression OOF locally for each seed ...")
    logreg_oof = {}
    for seed in range(N_SEEDS):
        logreg_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_fn(cols))
        print(f"  done: logreg seed={seed}")

    print("Fitting SVM(rbf) OOF locally for each seed (linear kernel screened and dropped "
          "- dominated by rbf in all 6 use cases at seed 0) ...")
    svm_oof = {}
    for seed in range(N_SEEDS):
        svm_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, svm_fn(cols, kernel="rbf"))
        print(f"  done: svm(rbf) seed={seed}")

    chosen_2way = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8")) if WEIGHTS_JSON.exists() else {}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 12. Does a third branch (SVM) add an edge?")
    emit()
    emit("Prompted by: \"did we miss SVM, as our ensemble is only 2 models, I wondered if a "
         "third would give another edge.\" A linear-kernel SVM was screened first (seed 0 "
         "only) and dropped - RBF beat it in all 6 use cases (mean within-silo ROC-AUC 0.819 "
         "vs 0.859) - so only `SVC(kernel=\"rbf\", probability=True, class_weight=\"balanced\")` "
         "is carried through the analysis below, on the same "
         f"`{WINNING_VARIANT}` feature set every other v2 experiment uses.")
    emit()

    # ---- 1. Standalone performance ------------------------------------------------------
    emit("### Standalone performance, mean +/- sd across 5 seeds")
    emit()
    rows = []
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        for branch_name, oof_dict in [("CatBoost (150/4)", catboost_oof),
                                       ("LogisticRegression", logreg_oof),
                                       ("SVM (rbf)", svm_oof)]:
            aucs, f2s = [], []
            for seed in range(N_SEEDS):
                oof = oof_dict[seed][uc]
                aucs.append(roc_auc_score(yu, oof))
                _, f2 = f2_optimal(yu, oof)
                f2s.append(f2)
            rows.append({
                "use_case": USE_CASE_DISPLAY[uc], "branch": branch_name,
                "roc_auc_mean": np.mean(aucs), "roc_auc_sd": np.std(aucs),
                "f2_at_t_star_mean": np.mean(f2s), "f2_at_t_star_sd": np.std(f2s),
            })
    standalone = pd.DataFrame(rows).set_index(["use_case", "branch"]).round(4)
    emit(to_md(standalone.reset_index().set_index("use_case"), "use_case"))
    emit()

    # ---- 2. Correlation diagnostic -------------------------------------------------------
    emit("### Branch-disagreement diagnostic — Pearson rho of OOF probabilities, averaged "
         "across 5 seeds")
    emit()
    corr_rows = []
    for uc in USE_CASE_ORDER:
        rhos = {"svm_vs_catboost": [], "svm_vs_logreg": [], "catboost_vs_logreg": []}
        for seed in range(N_SEEDS):
            cb, lr, sv = catboost_oof[seed][uc], logreg_oof[seed][uc], svm_oof[seed][uc]
            rhos["svm_vs_catboost"].append(np.corrcoef(sv, cb)[0, 1])
            rhos["svm_vs_logreg"].append(np.corrcoef(sv, lr)[0, 1])
            rhos["catboost_vs_logreg"].append(np.corrcoef(cb, lr)[0, 1])
        corr_rows.append({"use_case": USE_CASE_DISPLAY[uc],
                           **{k: np.mean(v) for k, v in rhos.items()}})
    corr_df = pd.DataFrame(corr_rows).set_index("use_case").round(3)
    emit(to_md(corr_df, "use_case"))
    emit()
    emit(f"Mean rho across use cases: SVM-vs-CatBoost **{corr_df['svm_vs_catboost'].mean():.3f}**, "
         f"SVM-vs-LogReg **{corr_df['svm_vs_logreg'].mean():.3f}**, CatBoost-vs-LogReg "
         f"(reference, the pair already in production) **{corr_df['catboost_vs_logreg'].mean():.3f}**.")
    emit()

    # ---- 3. Nested 3-way blend vs the existing nested 2-way blend ------------------------
    emit("### Nested (non-leaky) 3-way blend vs. the existing nested 2-way blend")
    emit()
    emit(f"Weights chosen on seeds {SELECTION_SEEDS}'s OOF (grid search, step 0.1, over the "
         f"full 3-branch simplex), evaluated on seeds {VALIDATION_SEEDS}'s OOF - never seen "
         f"during selection. Compared against the already-nested 2-way CatBoost/LogReg "
         f"weight from `reports/wf_ensemble_v2_chosen_weights.json`, evaluated on the same "
         f"held-out seeds.")
    emit()

    simplex = weight_simplex(0.1)
    result_rows = []
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        disp = USE_CASE_DISPLAY[uc]
        w2 = chosen_2way.get(disp, 0.5)

        sel_scores = []
        for w_cb, w_lr, w_svm in simplex:
            aucs_this_w = []
            for seed in SELECTION_SEEDS:
                blend = (w_cb * catboost_oof[seed][uc] + w_lr * logreg_oof[seed][uc]
                         + w_svm * svm_oof[seed][uc])
                aucs_this_w.append(roc_auc_score(yu, blend))
            sel_scores.append(np.mean(aucs_this_w))
        best_idx = int(np.argmax(sel_scores))
        w_cb, w_lr, w_svm = simplex[best_idx]

        val_auc_2way, val_auc_3way, val_f2_2way, val_f2_3way = [], [], [], []
        for seed in VALIDATION_SEEDS:
            cb, lr, sv = catboost_oof[seed][uc], logreg_oof[seed][uc], svm_oof[seed][uc]
            blend_2way = w2 * cb + (1 - w2) * lr
            blend_3way = w_cb * cb + w_lr * lr + w_svm * sv
            val_auc_2way.append(roc_auc_score(yu, blend_2way))
            val_auc_3way.append(roc_auc_score(yu, blend_3way))
            _, f2_2 = f2_optimal(yu, blend_2way)
            _, f2_3 = f2_optimal(yu, blend_3way)
            val_f2_2way.append(f2_2)
            val_f2_3way.append(f2_3)

        result_rows.append({
            "use_case": disp,
            "w_cb": w_cb, "w_lr": w_lr, "w_svm": w_svm,
            "held_out_auc_2way": np.mean(val_auc_2way),
            "held_out_auc_3way": np.mean(val_auc_3way),
            "auc_gain": np.mean(val_auc_3way) - np.mean(val_auc_2way),
            "held_out_f2_2way": np.mean(val_f2_2way),
            "held_out_f2_3way": np.mean(val_f2_3way),
            "f2_gain": np.mean(val_f2_3way) - np.mean(val_f2_2way),
        })

    result_df = pd.DataFrame(result_rows).set_index("use_case")
    emit(to_md(result_df.round(4), "use_case"))
    emit()

    mean_auc_gain = result_df["auc_gain"].mean()
    mean_f2_gain = result_df["f2_gain"].mean()
    clears = "YES" if mean_auc_gain > NOISE_FLOOR else "no"
    emit(f"Overall: mean AUC gain **{mean_auc_gain:+.4f}**, mean F2@t* gain "
         f"**{mean_f2_gain:+.4f}**, evaluated on held-out seeds never used for weight "
         f"selection. Clears the noise floor: **{clears}**.")
    emit()

    verdict_auc_positive = int((result_df["auc_gain"] > 0).sum())
    max_gain = result_df["auc_gain"].max()
    max_gain_uc = result_df["auc_gain"].idxmax()
    verdict = "ADOPT" if mean_auc_gain > NOISE_FLOOR else "REJECT"
    emit(f"3-way blend beats the 2-way blend's held-out AUC on {verdict_auc_positive}/6 use "
         f"cases; the single largest gain is {max_gain:+.4f} ({max_gain_uc}). "
         f"**Verdict: {verdict}** — see correlation table above for whether this is a "
         f"diversity problem (high rho with an existing branch) or a sample-size/ceiling "
         f"problem (genuinely decorrelated but the gain still doesn't clear noise).")
    emit()

    existing = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
