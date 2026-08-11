"""run_lexmeta_branch_experiment.py — does adding a lexical/metadata-only branch (no
embedding at all) to the CatBoost(150,4) + LogisticRegression ensemble give a genuine edge?

Follows up on `scripts/run_svm_branch_experiment.py` (reports/wf_ensemble_v2_experiments.md
§12), which rejected SVM as a third branch precisely because it saw the SAME ~4600-dim
embedding-dominated feature set as the other two branches (correlation 0.86-0.91, higher
than the existing CatBoost/LogReg pair's own 0.780) - the problem was the feature view, not
the algorithm. This tests the opposite lever: same kind of branch (CatBoost, Ordered
boosting), but a genuinely disjoint feature view - `lex + cos-brief + metadata (no
embedding)` from `run_ensemble_candidate.build_variants`, ~21 columns, the raw embedding
block dropped entirely.

Motivating prior signal: `scripts/run_lean_vs_embedding_arms.py` /
`reports/wf_lean_vs_embedding_arms.md` (2026-08-06) already found a "LEAN" arm (33 cols,
cos_brief + rank + lexical + metadata, no embeddings, plain LogisticRegression both sides)
had mean Spearman rank correlation of only 0.535 with an "EMBEDDING only" (qwen3-8b) arm -
well below the production CatBoost/LogReg pair's Pearson 0.780, and far below SVM's 0.86-
0.91. That result did NOT settle this question though: no CatBoost was involved, the
"embedding" side wasn't the actual production feature set, no Pearson-on-probabilities or F2
was measured, and no nested weight-selection blend against the real 2 production branches
was ever run. This script closes that gap, at the same rigor as the SVM check.

A local timing check (this file's own history, not re-run at import time) found CatBoost
with `thread_count=1` fits the 21-column lean feature set in ~38s for all 6 use cases x
5 folds (one seed) - fast enough to run entirely locally, unlike the full ~4600-dim
embedding-inclusive feature set, which needs Modal. CatBoost-lean and LogReg-lean were
statistically tied standalone at a seed-0 screen (mean ROC-AUC 0.778 vs 0.781) - CatBoost-
lean is carried forward as the single third-branch candidate, both to keep this a clean
3-way (not 4-way) blend question and because it isolates the feature-view effect: same
algorithm as the existing tree branch, different feature view only.

Same three questions as run_svm_branch_experiment.py, same nested/multi-seed discipline:

1. Standalone: how does CatBoost-lean compare to CatBoost(150,4)/LogisticRegression on the
   full `WINNING_VARIANT` feature set?
2. Diversity: is the lean branch's OOF probability correlation with the two full-feature
   branches lower than those two branches' own 0.780 correlation with each other?
3. Does a nested (non-leaky) 3-way blend beat the already-established, already-nested 2-way
   CatBoost/LogReg blend (reports/wf_ensemble_v2_chosen_weights.json) by more than this
   project's ~0.03 noise floor, on seeds the weight search never saw?

Usage:
    python scripts/run_lexmeta_branch_experiment.py
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
from run_ensemble_candidate import build_variants, catboost_fn, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
WEIGHTS_JSON = REPO / "reports" / "wf_ensemble_v2_chosen_weights.json"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
LEAN_VARIANT = "lex + cos-brief + metadata (no embedding)"
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
    """All (w_cb, w_lr, w_lean) triples on a step-0.1 grid summing to 1."""
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    out = []
    for w_cb in vals:
        for w_lr in vals:
            w_lean = round(1 - w_cb - w_lr, 2)
            if -1e-9 <= w_lean <= 1 + 1e-9:
                out.append((float(w_cb), float(w_lr), float(max(0.0, w_lean))))
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

    variants = build_variants(df)
    cols_full = variants[WINNING_VARIANT]
    cols_lean = variants[LEAN_VARIANT]
    X_full = df[cols_full]
    X_lean = df[cols_lean]

    print(f"Fetching CatBoost (150/4) OOF on the full feature set for {N_SEEDS} seeds ...")
    fn = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_hparam_oof")
    catboost_oof = {}
    for seed, res in enumerate(fn.starmap([(150, 4, s) for s in range(N_SEEDS)])):
        catboost_oof[seed] = {uc: np.array(v) for uc, v in res.items()}
        print(f"  done: catboost(full) seed={seed}")

    print("Fitting LogisticRegression OOF locally (full feature set) for each seed ...")
    logreg_oof = {}
    for seed in range(N_SEEDS):
        logreg_oof[seed] = within_silo_oof(X_full, y, use_case, groups, seed, logreg_fn(cols_full))
        print(f"  done: logreg(full) seed={seed}")

    print("Fitting CatBoost (150/4, thread_count=1) OOF locally on the LEAN (no-embedding) "
          "feature set for each seed (~38s/seed measured locally - no Modal needed at only "
          "21 columns) ...")
    lean_oof = {}
    for seed in range(N_SEEDS):
        lean_oof[seed] = within_silo_oof(
            X_lean, y, use_case, groups, seed,
            catboost_fn(cols_lean, thread_count=1, iterations=150, depth=4),
        )
        print(f"  done: catboost(lean) seed={seed}")

    chosen_2way = json.loads(WEIGHTS_JSON.read_text()) if WEIGHTS_JSON.exists() else {}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 13. Does a lexical/metadata-only branch (no embedding) add an edge?")
    emit()
    emit("Motivated by `reports/wf_lean_vs_embedding_arms.md` (2026-08-06), which found a "
         "\"LEAN\" arm (cos_brief + rank + lexical + metadata, no embeddings, plain "
         "LogisticRegression) had mean Spearman rank correlation of only **0.535** with an "
         "embedding-only arm - well below the production CatBoost/LogReg pair's Pearson "
         "0.780, and far below SVM's 0.86-0.91 (§12). That check never involved CatBoost, "
         "never used the actual production feature/weight baseline, and never ran a nested "
         "weight-selection blend - this closes that gap, at the same rigor as the SVM check. "
         "A linear-kernel screen isn't relevant here (there is no kernel choice); a seed-0 "
         "screen instead compared CatBoost vs. LogisticRegression *on the lean feature set "
         "itself* (mean ROC-AUC 0.778 vs 0.781, a statistical tie) - CatBoost-lean is "
         f"carried forward as the single third branch, on `{LEAN_VARIANT}` "
         f"(21 columns, no embedding at all), against the two production branches on "
         f"`{WINNING_VARIANT}`.")
    emit()

    # ---- 1. Standalone performance ------------------------------------------------------
    emit("### Standalone performance, mean +/- sd across 5 seeds")
    emit()
    rows = []
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        for branch_name, oof_dict in [("CatBoost (150/4, full)", catboost_oof),
                                       ("LogisticRegression (full)", logreg_oof),
                                       ("CatBoost (150/4, lean/no-embedding)", lean_oof)]:
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
        rhos = {"lean_vs_catboost_full": [], "lean_vs_logreg_full": [], "catboost_vs_logreg_full": []}
        for seed in range(N_SEEDS):
            cb, lr, lean = catboost_oof[seed][uc], logreg_oof[seed][uc], lean_oof[seed][uc]
            rhos["lean_vs_catboost_full"].append(np.corrcoef(lean, cb)[0, 1])
            rhos["lean_vs_logreg_full"].append(np.corrcoef(lean, lr)[0, 1])
            rhos["catboost_vs_logreg_full"].append(np.corrcoef(cb, lr)[0, 1])
        corr_rows.append({"use_case": USE_CASE_DISPLAY[uc],
                           **{k: np.mean(v) for k, v in rhos.items()}})
    corr_df = pd.DataFrame(corr_rows).set_index("use_case").round(3)
    emit(to_md(corr_df, "use_case"))
    emit()
    emit(f"Mean rho across use cases: lean-vs-CatBoost(full) "
         f"**{corr_df['lean_vs_catboost_full'].mean():.3f}**, lean-vs-LogReg(full) "
         f"**{corr_df['lean_vs_logreg_full'].mean():.3f}**, CatBoost-vs-LogReg (reference, "
         f"the pair already in production) **{corr_df['catboost_vs_logreg_full'].mean():.3f}**. "
         "The older Spearman-rank 0.535 finding and this Pearson-on-probabilities number are "
         "different metrics on a different exact feature set (33 cols there vs. 21 here, "
         "qwen3-8b-alone vs. the actual production Jasper+Qwen3-4B feature set) so an exact "
         "match isn't expected - the question is only whether the direction (lean decorrelates "
         "more than the existing pair decorrelates from itself) holds up.")
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
        for w_cb, w_lr, w_lean in simplex:
            aucs_this_w = []
            for seed in SELECTION_SEEDS:
                blend = (w_cb * catboost_oof[seed][uc] + w_lr * logreg_oof[seed][uc]
                         + w_lean * lean_oof[seed][uc])
                aucs_this_w.append(roc_auc_score(yu, blend))
            sel_scores.append(np.mean(aucs_this_w))
        best_idx = int(np.argmax(sel_scores))
        w_cb, w_lr, w_lean = simplex[best_idx]

        val_auc_2way, val_auc_3way, val_f2_2way, val_f2_3way = [], [], [], []
        for seed in VALIDATION_SEEDS:
            cb, lr, lean = catboost_oof[seed][uc], logreg_oof[seed][uc], lean_oof[seed][uc]
            blend_2way = w2 * cb + (1 - w2) * lr
            blend_3way = w_cb * cb + w_lr * lr + w_lean * lean
            val_auc_2way.append(roc_auc_score(yu, blend_2way))
            val_auc_3way.append(roc_auc_score(yu, blend_3way))
            _, f2_2 = f2_optimal(yu, blend_2way)
            _, f2_3 = f2_optimal(yu, blend_3way)
            val_f2_2way.append(f2_2)
            val_f2_3way.append(f2_3)

        result_rows.append({
            "use_case": disp,
            "w_cb": w_cb, "w_lr": w_lr, "w_lean": w_lean,
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
    mean_lean_rho = (corr_df["lean_vs_catboost_full"].mean() + corr_df["lean_vs_logreg_full"].mean()) / 2
    ref_rho = corr_df["catboost_vs_logreg_full"].mean()
    diversity_note = (
        f"the lean branch IS more decorrelated than the existing pair (mean rho vs. the two "
        f"full-feature branches {mean_lean_rho:.3f} vs. their own {ref_rho:.3f} with each "
        f"other) - so this is a sample-size/ceiling problem, not a diversity problem: the "
        f"disagreement is real but the lean branch is too much weaker standalone for its "
        f"disagreement to move the blend beyond noise"
        if mean_lean_rho < ref_rho else
        f"the lean branch is NOT more decorrelated than the existing pair (mean rho "
        f"{mean_lean_rho:.3f} vs. their own {ref_rho:.3f}) - the same diversity problem as SVM, "
        f"just less severe"
    )
    emit(f"3-way blend beats the 2-way blend's held-out AUC on {verdict_auc_positive}/6 use "
         f"cases; the single largest gain is {max_gain:+.4f} ({max_gain_uc}). "
         f"**Verdict: {verdict}** — {diversity_note}.")
    emit()

    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
