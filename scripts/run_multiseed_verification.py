"""run_multiseed_verification.py — settles whether Experiments A/B/C from
reports/wf_ensemble_v2_experiments.md are real signal or noise.

The last round measured everything at a single seed (seed=0) and found +0.01-ish gains
everywhere — all of them below this project's own established noise floor (CONTEXT.md §5:
seed-to-seed sd ~0.010 on within-silo ROC-AUC; treat any gap below ~0.03 as not established).
This script repeats the same three comparisons across 5 independent seeds (matching Phase
1's own discipline) and reports mean +/- sd, not a single point estimate.

Fetches CatBoost OOF for both the screening (iterations=50, depth=4) and tuned
(iterations=150, depth=4) configs, 5 seeds each, via the already-deployed
`run_catboost_hparam_oof` Modal function (parallelized: 10 calls, wall-clock bounded by the
slowest single call, not the sum). LogisticRegression OOF is fit locally per seed (fast, no
Modal needed - only CatBoost has the Apple Silicon problem, see
scripts/modal_ensemble_candidate.py's module docstring).

Usage:
    python scripts/run_multiseed_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import within_silo_oof  # noqa: E402
from run_ensemble_candidate import build_variants, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_multiseed_verification.md"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
N_SEEDS = 5
NOISE_FLOOR = 0.03

CONFIGS = {
    "screening": (50, 4),
    "tuned": (150, 4),
}

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

    # ---- Fetch CatBoost OOF: 2 configs x N_SEEDS, all in parallel on Modal -------------
    fn = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_hparam_oof")
    calls = [
        (config_name, seed, it, d)
        for config_name, (it, d) in CONFIGS.items()
        for seed in range(N_SEEDS)
    ]
    print(f"Fetching CatBoost OOF for {len(calls)} (config, seed) combinations on Modal "
          f"(parallelized) ...")
    catboost_oof = {name: {} for name in CONFIGS}
    for (config_name, seed, it, d), res in zip(
        calls, fn.starmap([(it, d, seed) for _, seed, it, d in calls], return_exceptions=True)
    ):
        if isinstance(res, Exception):
            print(f"  FAILED: {config_name} seed={seed}: {res!r}")
            continue
        print(f"  done: {config_name} seed={seed}")
        catboost_oof[config_name][seed] = {uc: np.array(v) for uc, v in res.items()}

    # ---- Fetch LogReg OOF locally: N_SEEDS, fast ---------------------------------------
    print("Fitting LogisticRegression OOF locally for each seed ...")
    logreg_oof = {}
    for seed in range(N_SEEDS):
        logreg_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_fn(cols))

    # ---- Per-seed comparisons -----------------------------------------------------------
    a_rows, b_rows, c_rows = [], [], []
    for seed in range(N_SEEDS):
        if seed not in catboost_oof["screening"] or seed not in catboost_oof["tuned"]:
            continue  # a failed fetch for this seed - skip rather than silently impute
        for uc in USE_CASE_ORDER:
            yu = y_by_uc[uc]
            screening_auc = roc_auc_score(yu, catboost_oof["screening"][seed][uc])
            tuned_auc = roc_auc_score(yu, catboost_oof["tuned"][seed][uc])
            a_rows.append({
                "seed": seed, "use_case": USE_CASE_DISPLAY[uc],
                "screening_auc": screening_auc, "tuned_auc": tuned_auc,
                "delta": tuned_auc - screening_auc,
            })

            tuned_cb = catboost_oof["tuned"][seed][uc]
            lr = logreg_oof[seed][uc]
            blend_50 = 0.5 * tuned_cb + 0.5 * lr
            weights = np.linspace(0, 1, 21)
            aucs = [roc_auc_score(yu, w * tuned_cb + (1 - w) * lr) for w in weights]
            best_idx = int(np.argmax(aucs))
            auc_50 = roc_auc_score(yu, blend_50)
            b_rows.append({
                "seed": seed, "use_case": USE_CASE_DISPLAY[uc],
                "auc_50_50": auc_50, "best_w": float(weights[best_idx]),
                "auc_best_w": aucs[best_idx], "delta": aucs[best_idx] - auc_50,
            })

            f2_before = f2_at(yu, blend_50, 0.5)
            platt = LogisticRegression().fit(blend_50.reshape(-1, 1), yu)
            calibrated = platt.predict_proba(blend_50.reshape(-1, 1))[:, 1]
            f2_after = f2_at(yu, calibrated, 0.5)
            _, f2_star = f2_optimal(yu, blend_50)
            c_rows.append({
                "seed": seed, "use_case": USE_CASE_DISPLAY[uc],
                "f2_before": f2_before, "f2_after": f2_after,
                "f2_at_t_star": f2_star, "delta": f2_after - f2_before,
            })

    a_df, b_df, c_df = pd.DataFrame(a_rows), pd.DataFrame(b_rows), pd.DataFrame(c_rows)
    n_seeds_completed = a_df["seed"].nunique() if not a_df.empty else 0

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Multi-seed verification of Ensemble v2 experiments")
    emit()
    emit(f"Same three comparisons as `reports/wf_ensemble_v2_experiments.md`, repeated across "
         f"{n_seeds_completed}/{N_SEEDS} seeds instead of one, applying this project's own "
         f"noise-floor discipline (`CONTEXT.md` §5: treat any gap below ~{NOISE_FLOOR:.2f} "
         f"ROC-AUC as not established).")
    emit()

    # Experiment A
    emit("## Experiment A — tuned vs. screening CatBoost, across seeds")
    emit()
    a_by_uc = a_df.groupby("use_case")["delta"].agg(["mean", "std"]).round(4)
    emit(to_md(a_by_uc, "use_case"))
    emit()
    a_mean, a_sd = a_df["delta"].mean(), a_df["delta"].std()
    a_clears = "YES" if a_mean > NOISE_FLOOR else "no"
    emit(f"Overall: mean delta **{a_mean:+.4f}** (sd across all seed x use_case rows: {a_sd:.4f}). "
         f"Clears the noise floor: **{a_clears}**.")
    emit()

    # Experiment B
    emit("## Experiment B — weighted combiner vs. fixed 50/50, across seeds")
    emit()
    b_by_uc = b_df.groupby("use_case")["delta"].agg(["mean", "std"]).round(4)
    emit(to_md(b_by_uc, "use_case"))
    emit()
    b_mean, b_sd = b_df["delta"].mean(), b_df["delta"].std()
    b_clears = "YES" if b_mean > NOISE_FLOOR else "no"
    emit(f"Overall: mean delta **{b_mean:+.4f}** (sd: {b_sd:.4f}). Clears the noise floor: "
         f"**{b_clears}**. Still an in-sample weight choice (no inner/outer split) — an "
         f"upper bound even at 5 seeds.")
    emit()
    weight_by_uc = b_df.groupby("use_case")["best_w"].agg(["mean", "std"]).round(2)
    emit("Best weight per use case, mean +/- sd across seeds (0 = pure LogReg, 1 = pure CatBoost):")
    emit()
    emit(to_md(weight_by_uc, "use_case"))
    emit()

    # Experiment C
    emit("## Experiment C — Platt calibration's effect on F2@0.5, across seeds")
    emit()
    c_by_uc = c_df.groupby("use_case")["delta"].agg(["mean", "std"]).round(4)
    emit(to_md(c_by_uc, "use_case"))
    emit()
    c_mean, c_sd = c_df["delta"].mean(), c_df["delta"].std()
    gap_left = (c_df["f2_at_t_star"] - c_df["f2_after"]).mean()
    emit(f"Overall: mean F2@0.5 lift from calibration **{c_mean:+.4f}** (sd: {c_sd:.4f}); "
         f"mean remaining gap to F2@t* after calibration: **{gap_left:+.4f}**.")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
