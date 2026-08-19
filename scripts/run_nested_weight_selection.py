"""run_nested_weight_selection.py — a properly nested (non-leaky) version of Experiment B.

The single-seed and multi-seed rounds both picked the combiner weight by looking at the
same OOF the weight was then scored on - an optimistic upper bound, flagged as such both
times. This script fixes that: for the three use cases with a stable (low-variance)
preference for weighting CatBoost heavily (Soil Microbiome, Low-Carbon Cement, Solar Cells
for Satellites - reports/wf_ensemble_v2_multiseed_verification.md), the weight is chosen
using ONLY seeds {0,1,2}'s OOF, then evaluated on seeds {3,4}'s OOF, which the selection
step never saw. The other three use cases keep the fixed 50/50 default (no stable
preference found - reports/wf_ensemble_v2_multiseed_verification.md).

CatBoost config fixed at iterations=150, depth=4 (reports/wf_ensemble_v2_experiments.md §8 -
nothing tested since beat this setting).

Usage:
    python scripts/run_nested_weight_selection.py
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
from run_ensemble_candidate import build_variants, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
N_SEEDS = 5
SELECTION_SEEDS = [0, 1, 2]
VALIDATION_SEEDS = [3, 4]
STABLE_UCS = {"soil_microbiome", "cement_binders", "solar_leo"}  # favor CatBoost weight
UNSTABLE_UCS = {"carbon_capture", "ner", "tech_forecasting"}  # keep 50/50

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


def main() -> None:
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

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 9. Properly nested weight selection (no in-sample peeking)")
    emit()
    emit(f"Weight chosen on seeds {SELECTION_SEEDS}'s OOF, evaluated on seeds "
         f"{VALIDATION_SEEDS}'s OOF (never seen during selection) - for the three use cases "
         f"with a stable per-seed preference. The other three keep fixed 50/50.")
    emit()

    rows = []
    weights = np.linspace(0, 1, 21)
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        if uc in STABLE_UCS:
            sel_aucs = []
            for w in weights:
                aucs_this_w = []
                for seed in SELECTION_SEEDS:
                    blend = w * catboost_oof[seed][uc] + (1 - w) * logreg_oof[seed][uc]
                    aucs_this_w.append(roc_auc_score(yu, blend))
                sel_aucs.append(np.mean(aucs_this_w))
            chosen_w = float(weights[int(np.argmax(sel_aucs))])
        else:
            chosen_w = 0.5

        val_auc_chosen, val_auc_5050, val_f2_chosen, val_f2_5050 = [], [], [], []
        for seed in VALIDATION_SEEDS:
            cb, lr = catboost_oof[seed][uc], logreg_oof[seed][uc]
            blend_chosen = chosen_w * cb + (1 - chosen_w) * lr
            blend_5050 = 0.5 * cb + 0.5 * lr
            val_auc_chosen.append(roc_auc_score(yu, blend_chosen))
            val_auc_5050.append(roc_auc_score(yu, blend_5050))
            _, f2_chosen = f2_optimal(yu, blend_chosen)
            _, f2_5050 = f2_optimal(yu, blend_5050)
            val_f2_chosen.append(f2_chosen)
            val_f2_5050.append(f2_5050)

        rows.append({
            "use_case": USE_CASE_DISPLAY[uc],
            "chosen_w": chosen_w,
            "held_out_auc_50_50": np.mean(val_auc_5050),
            "held_out_auc_chosen_w": np.mean(val_auc_chosen),
            "auc_gain": np.mean(val_auc_chosen) - np.mean(val_auc_5050),
            "held_out_f2_at_t_star_50_50": np.mean(val_f2_5050),
            "held_out_f2_at_t_star_chosen_w": np.mean(val_f2_chosen),
            "f2_gain": np.mean(val_f2_chosen) - np.mean(val_f2_5050),
        })

    result_table = pd.DataFrame(rows).set_index("use_case")
    emit(to_md(result_table.round(4), "use_case"))
    emit()
    stable_rows = result_table.loc[[USE_CASE_DISPLAY[uc] for uc in STABLE_UCS]]
    emit(f"On the 3 use cases with a nested (honest, held-out) weight: mean F2 gain "
         f"**{stable_rows['f2_gain'].mean():+.4f}**, mean AUC gain "
         f"**{stable_rows['auc_gain'].mean():+.4f}** — evaluated on seeds the weight "
         f"selection never saw. This is the trustworthy number, not the in-sample one "
         f"from the earlier rounds.")
    emit()

    existing = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")

    # Save the chosen weights for the SYNERGY step (or any downstream use).
    import json
    weights_out = {row["use_case"]: row["chosen_w"] for row in rows}
    (REPO / "reports" / "wf_ensemble_v2_chosen_weights.json").write_text(json.dumps(weights_out), encoding="utf-8")


if __name__ == "__main__":
    main()
