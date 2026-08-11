"""analyze_ensemble_v2_experiments.py — Experiments A/B/C from reports/wf_ensemble_v2_experiments.md.

Reads the CatBoost OOF fetched by `modal run scripts/modal_ensemble_experiments.py::hparam_sweep`
(reports/wf_ensemble_v2_hparam_oof.json), fits LogisticRegression OOF locally (fast, no
Modal needed - only CatBoost has the Apple Silicon problem), and runs:

  A. Screening vs. tuned CatBoost hyperparameters — ROC-AUC, F2@0.5, F2@t*.
  B. Weighted combiner sweep (tuned CatBoost + LogReg) vs. fixed 50/50.
  C. Platt calibration's effect on F2@0.5, using the best-weighted ensemble from B.

Everything here is single-seed OOF - a diagnostic sweep, not Phase 1's proper 5-seed CV.
Appends a results section to reports/wf_ensemble_v2_experiments.md rather than overwriting it
(the reflection section, written first, stays put).

Usage:
    python scripts/analyze_ensemble_v2_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import within_silo_oof  # noqa: E402
from run_ensemble_candidate import build_variants, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
HPARAM_JSON = REPO / "reports" / "wf_ensemble_v2_hparam_oof.json"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
OOF_SEED = 0

USE_CASE_DISPLAY = {
    "carbon_capture": "Carbon Capture",
    "cement_binders": "Low-Carbon Cement",
    "solar_leo": "Solar Cells for Satellites",
    "soil_microbiome": "Soil Microbiome",
    "ner": "Named Entity Recognition",
    "tech_forecasting": "Technology Prediction",
}


def f2_optimal(y_true, y_score, thresholds=np.linspace(0.05, 0.95, 91)):
    best_t, best_f2 = 0.5, -1.0
    for t in thresholds:
        f2 = fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
    return best_t, best_f2


def f2_at(y_true, y_score, t):
    return fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)


def to_md(frame: pd.DataFrame, index_name: str = "") -> str:
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


def main() -> None:
    if not HPARAM_JSON.exists():
        raise FileNotFoundError(
            f"{HPARAM_JSON} not found — run `modal run "
            f"scripts/modal_ensemble_experiments.py::hparam_sweep --seed 0` first."
        )
    hparam_oof = json.load(open(HPARAM_JSON))
    configs = list(hparam_oof.keys())
    print(f"Loaded CatBoost OOF for {len(configs)} configs: {configs}")

    df = pd.read_parquet(DATA_PATH)
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]
    USE_CASE_ORDER = sorted(use_case.unique())

    print("Fitting LogisticRegression OOF locally ...")
    cols = build_variants(df)[WINNING_VARIANT]
    X = df[cols]
    logreg_oof = within_silo_oof(X, y, use_case, groups, OOF_SEED, logreg_fn(cols))

    y_by_uc = {uc: y[(use_case == uc).to_numpy()] for uc in USE_CASE_ORDER}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---- Experiment A: screening vs tuned CatBoost -------------------------------------
    emit("## 2. Experiment A — does un-cheaping CatBoost's hyperparameters help?")
    emit()
    a_rows = []
    for name in configs:
        oof = {uc: np.array(v) for uc, v in hparam_oof[name].items()}
        for uc in USE_CASE_ORDER:
            yu = y_by_uc[uc]
            t_star, f2_star = f2_optimal(yu, oof[uc])
            a_rows.append({
                "config": name, "use_case": USE_CASE_DISPLAY[uc],
                "roc_auc": roc_auc_score(yu, oof[uc]),
                "f2_at_0.5": f2_at(yu, oof[uc], 0.5),
                "f2_at_t_star": f2_star,
            })
    a_table = pd.DataFrame(a_rows)
    a_pivot_auc = a_table.pivot(index="use_case", columns="config", values="roc_auc").round(3)
    a_pivot_auc = a_pivot_auc[configs]
    emit("ROC-AUC by config:")
    emit()
    emit(to_md(a_pivot_auc, "use_case"))
    emit()
    baseline, tuned = configs[0], configs[1]
    delta = (a_pivot_auc[tuned] - a_pivot_auc[baseline])
    emit(f"Mean ROC-AUC delta (tuned - screening): **{delta.mean():+.3f}**, "
         f"wins on {int((delta > 0).sum())}/6 use cases.")
    emit()

    # ---- Experiment B: weighted combiner sweep -----------------------------------------
    emit("## 3. Experiment B — weighted combiner vs. fixed 50/50")
    emit()
    best_catboost_name = tuned if delta.mean() > 0 else baseline
    emit(f"Using **{best_catboost_name}** CatBoost OOF (the stronger of the two from "
         f"Experiment A) for the combiner sweep.")
    emit()
    catboost_best_oof = {uc: np.array(v) for uc, v in hparam_oof[best_catboost_name].items()}

    b_rows = []
    weights = np.linspace(0, 1, 21)
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        aucs = []
        for w in weights:
            blend = w * catboost_best_oof[uc] + (1 - w) * logreg_oof[uc]
            aucs.append(roc_auc_score(yu, blend))
        aucs = np.array(aucs)
        best_idx = int(np.argmax(aucs))
        fixed_50_idx = int(np.argmin(np.abs(weights - 0.5)))
        b_rows.append({
            "use_case": USE_CASE_DISPLAY[uc],
            "auc_at_50_50": aucs[fixed_50_idx],
            "best_w": float(weights[best_idx]),
            "auc_at_best_w": aucs[best_idx],
            "gain_vs_50_50": aucs[best_idx] - aucs[fixed_50_idx],
        })
    b_table = pd.DataFrame(b_rows).set_index("use_case")
    emit(to_md(b_table.round(3), "use_case"))
    emit()
    emit(f"Mean gain from a per-silo weight over fixed 50/50: **{b_table['gain_vs_50_50'].mean():+.4f}** "
         f"ROC-AUC. **Diagnostic caveat:** this weight is chosen by looking at the same OOF "
         f"it's scored on (no inner/outer split) — an optimistic upper bound on what a "
         f"properly nested weight-selection would get, not a number to ship as-is.")
    emit()

    # ---- Experiment C: Platt calibration's effect on F2@0.5 ----------------------------
    emit("## 4. Experiment C — does calibration make the naive 0.5 threshold usable?")
    emit()
    ensemble_oof = {
        uc: 0.5 * catboost_best_oof[uc] + 0.5 * logreg_oof[uc] for uc in USE_CASE_ORDER
    }
    c_rows = []
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        raw = ensemble_oof[uc]
        t_star, f2_star = f2_optimal(yu, raw)
        f2_before = f2_at(yu, raw, 0.5)

        # In-sample Platt scaling (diagnostic - the same OOF calibrates and scores itself,
        # same caveat as Experiment B): a 1-D logistic regression of y on the raw score.
        platt = LogisticRegression().fit(raw.reshape(-1, 1), yu)
        calibrated = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        f2_after = f2_at(yu, calibrated, 0.5)

        c_rows.append({
            "use_case": USE_CASE_DISPLAY[uc],
            "f2_at_0.5_raw": f2_before,
            "f2_at_0.5_calibrated": f2_after,
            "f2_at_t_star (raw, uncalibrated)": f2_star,
        })
    c_table = pd.DataFrame(c_rows).set_index("use_case")
    emit(to_md(c_table.round(3), "use_case"))
    emit()
    closed = (c_table["f2_at_0.5_calibrated"] - c_table["f2_at_0.5_raw"]).mean()
    gap_left = (c_table["f2_at_t_star (raw, uncalibrated)"] - c_table["f2_at_0.5_calibrated"]).mean()
    emit(f"Calibration lifts mean F2@0.5 by **{closed:+.3f}**; the remaining gap to F2@t* "
         f"averages **{gap_left:+.3f}**. **Diagnostic caveat:** Platt scaling fit and scored "
         f"on the same OOF sample, same as Experiment B — a real deployment needs the "
         f"calibrator fit on a held-out slice, not the same rows it's evaluated on.")
    emit()

    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
