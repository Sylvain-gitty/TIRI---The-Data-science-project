"""run_central_hparam_search.py — the "search happens once, centrally, in the LOGO loop"
rule (CONTEXT.md §1), actually followed, for the two hyperparameters this project's ensemble
work has never properly explored: CatBoost's (iterations, depth) and LogisticRegression's C.

THE GAP THIS CLOSES
--------------------
Every CatBoost hyperparameter change so far this session (50 -> 150 -> 300 iterations,
reports/wf_ensemble_v2_experiments.md §§2, 7-8) was selected by scoring on **within-silo**
CV - the production surface - not LOGO. CONTEXT.md §1 is explicit that this is backwards:

    "Hyperparameter search happens once, centrally, in the LOGO loop, where six use cases
    of evidence exist. Each silo then just fits weights. Per-customer grid search at a few
    hundred labels would overfit."
    "Leave-one-use-case-out (LOGO) is a defaults-selection instrument, not a production
    estimate... Within-silo (grouped k-fold inside one use case) is the production surface."

Picking a setting on the same surface later reported as "the result" is a mild form of
post-selection bias - the same shape of issue the nested combiner-weight selection
(run_nested_weight_selection.py) was built to avoid, never previously extended to
hyperparameters themselves. On top of that, CatBoost's depth has never been explored past 4
(6 blew a within-silo Modal timeout; nothing in between or beyond was ever tried), and
LogisticRegression's C has never been touched from sklearn's default (1.0) anywhere in this
repo's ensemble work.

METHOD
------
1. Search via LOGO (as the rule requires), not within-silo.
2. Pick the LOGO-winning CatBoost (iterations, depth) and LogReg C.
3. Honestly VALIDATE each winner against the currently-shipped setting on fresh 5-seed
   within-silo OOF (the actual production surface) - only adopt if the LOGO-chosen setting
   clears this project's NOISE_FLOOR = 0.03 there. If it's flat, the honest conclusion is
   "LOGO and within-silo already agree the current default was fine" - a real, useful result
   of closing this methodological gap, not a failed experiment.

CALIBRATION (already run, hardcoded below, not re-run at import time)
-----------------------------------------------------------------------
A LOGO fit trains on ~1550 pooled rows (5 of 6 use cases), not the ~250-360 rows a
within-silo fold trains on - NOT automatically cheaper than within-silo per config. Measured
directly via scripts/modal_ensemble_experiments.py::run_catboost_logo: depth=4/iterations=150
(current shipped setting) took 190.9s; depth=6/iterations=100 took 446.3s. CatBoost's
per-iteration cost scales ~linearly at fixed depth (each iteration adds one more tree of the
same shape), so depth=6/iterations=350 (this grid's worst case) projects to roughly
446.3 * 3.5 ~= 1561s - comfortably under the 3600s EXPERIMENT_TIMEOUT. depth=6 is therefore
INCLUDED in the grid below (the original plan for this experiment said to trim it if unsafe;
calibration shows it is safe).

CatBoost grid: depth in {3,4,5,6} x iterations in {50,100,150,250,350} = 20 configs, via
scripts/modal_ensemble_experiments.py::run_catboost_logo (Modal, parallelized, deployed
2026-08-10). learning_rate/l2_leaf_reg are deliberately OUT OF SCOPE - this pass targets only
the two previously-unexplored axes (depth range, iterations resolution).

LogReg grid: C in {0.001, 0.01, 0.1, 1.0, 10, 100} via ensemble_eval_utils.logo() directly,
local (cheap - 6 fits per C, no Modal needed).

Usage:
    python scripts/run_central_hparam_search.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import logo, to_md, within_silo_oof  # noqa: E402
from run_ensemble_candidate import build_variants, catboost_fn, logreg_c_fn, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
GRID_JSON = REPO / "reports" / "wf_ensemble_v2_central_hparam_grid.json"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
NOISE_FLOOR = 0.03
N_SEEDS = 5

CATBOOST_DEPTHS = [3, 4, 5, 6]
CATBOOST_ITERATIONS = [50, 100, 150, 250, 350]
CURRENT_CATBOOST = (150, 4)  # (iterations, depth), the currently-shipped setting
LOGREG_C_GRID = [0.001, 0.01, 0.1, 1.0, 10, 100]
CURRENT_LOGREG_C = 1.0

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

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 15. Central hyperparameter search (CatBoost depth/iterations, LogReg C), via LOGO")
    emit()
    emit("Every CatBoost hyperparameter change so far this session (50->150->300 iterations, "
         "§§2, 7-8) was selected on **within-silo** CV - the production surface - not LOGO, "
         "which `CONTEXT.md` §1 specifies as the correct defaults-selection surface "
         "(\"hyperparameter search happens once, centrally, in the LOGO loop\"). Depth was "
         "never explored past 4 (6 blew a within-silo Modal timeout); LogisticRegression's "
         "`C` has never been touched from sklearn's default (1.0) anywhere in this repo's "
         "ensemble work. This closes both gaps: search via LOGO, then honestly validate the "
         "LOGO-chosen winner on fresh 5-seed within-silo OOF before adopting anything - "
         "extending this session's nested-selection discipline (§9, combiner weights) to "
         "hyperparameters themselves.")
    emit()
    emit("**Calibration** (`scripts/modal_ensemble_experiments.py::run_catboost_logo`, "
         "measured directly, not assumed): a LOGO fit trains on ~1550 pooled rows (5 of 6 "
         "use cases), not the ~250-360 rows a within-silo fold trains on - depth=4/"
         "iterations=150 (current shipped setting) took **190.9s**; depth=6/iterations=100 "
         "took **446.3s**. CatBoost's per-iteration cost scales ~linearly at fixed depth, so "
         "depth=6/iterations=350 (this grid's worst case) projects to ~1561s - comfortably "
         "under the 3600s timeout. depth=6 is therefore included in the grid below.")
    emit()

    # ---- 1. CatBoost grid via LOGO (Modal, parallelized) --------------------------------
    emit("### CatBoost grid via LOGO (mean across 6 use cases)")
    emit()
    fn = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_logo")
    configs = [(it, d) for d in CATBOOST_DEPTHS for it in CATBOOST_ITERATIONS]
    print(f"Fetching CatBoost LOGO scores for {len(configs)} (iterations, depth) configs "
          f"on Modal (parallelized) ...")
    cb_grid = {}
    if GRID_JSON.exists():
        cached = json.loads(GRID_JSON.read_text())
        cb_grid = {tuple(map(int, k.split("_"))): v for k, v in cached.get("catboost", {}).items()}
    remaining = [(it, d) for it, d in configs if (it, d) not in cb_grid]
    if remaining:
        for (it, d), res in zip(
            remaining,
            fn.starmap([(it, d) for it, d in remaining], return_exceptions=True),
        ):
            if isinstance(res, Exception):
                print(f"  FAILED: iterations={it} depth={d}: {res!r}")
                continue
            cb_grid[(it, d)] = res
            print(f"  done: iterations={it} depth={d}")
            GRID_JSON.write_text(json.dumps({
                "catboost": {f"{it}_{d}": v for (it, d), v in cb_grid.items()},
            }))

    cb_rows = []
    for (it, d), res in cb_grid.items():
        aucs = [res[uc]["roc_auc"] for uc in USE_CASE_ORDER]
        f2s = [res[uc]["f2_at_0.5"] for uc in USE_CASE_ORDER]
        cb_rows.append({
            "iterations": it, "depth": d,
            "mean_logo_roc_auc": float(np.mean(aucs)), "mean_logo_f2_at_0.5": float(np.mean(f2s)),
        })
    cb_grid_df = pd.DataFrame(cb_rows).sort_values(
        ["depth", "iterations"]
    ).set_index(["depth", "iterations"])
    emit(to_md(cb_grid_df.round(4).reset_index().set_index("depth"), "depth"))
    emit()

    best_cb_row = pd.DataFrame(cb_rows).sort_values("mean_logo_roc_auc", ascending=False).iloc[0]
    best_it, best_d = int(best_cb_row["iterations"]), int(best_cb_row["depth"])
    current_it, current_d = CURRENT_CATBOOST
    current_cb_row = next(r for r in cb_rows if r["iterations"] == current_it and r["depth"] == current_d)
    emit(f"LOGO winner: **iterations={best_it}, depth={best_d}** (mean LOGO ROC-AUC "
         f"{best_cb_row['mean_logo_roc_auc']:.4f}, mean LOGO F2@0.5 "
         f"{best_cb_row['mean_logo_f2_at_0.5']:.4f}) vs. the currently-shipped "
         f"iterations={current_it}/depth={current_d} (mean LOGO ROC-AUC "
         f"{current_cb_row['mean_logo_roc_auc']:.4f}, mean LOGO F2@0.5 "
         f"{current_cb_row['mean_logo_f2_at_0.5']:.4f}).")
    emit()

    # ---- 2. LogReg C grid via LOGO (local) -----------------------------------------------
    emit("### LogisticRegression C grid via LOGO")
    emit()
    lr_rows = []
    for C in LOGREG_C_GRID:
        res = logo(X, y, use_case, logreg_c_fn(cols, C=C))
        lr_rows.append({
            "C": C, "mean_logo_roc_auc": res["roc_auc"].mean(), "mean_logo_f2_at_0.5": res["f2_at_0.5"].mean(),
        })
        print(f"  done: C={C}")
    lr_grid_df = pd.DataFrame(lr_rows).set_index("C")
    emit(to_md(lr_grid_df.round(4), "C"))
    emit()

    best_lr_row = lr_grid_df.reset_index().sort_values("mean_logo_roc_auc", ascending=False).iloc[0]
    best_C = float(best_lr_row["C"])
    current_lr_row = lr_grid_df.loc[CURRENT_LOGREG_C]
    emit(f"LOGO winner: **C={best_C}** (mean LOGO ROC-AUC {best_lr_row['mean_logo_roc_auc']:.4f}, "
         f"mean LOGO F2@0.5 {best_lr_row['mean_logo_f2_at_0.5']:.4f}) vs. the currently-shipped "
         f"C={CURRENT_LOGREG_C} (mean LOGO ROC-AUC {current_lr_row['mean_logo_roc_auc']:.4f}, "
         f"mean LOGO F2@0.5 {current_lr_row['mean_logo_f2_at_0.5']:.4f}).")
    emit()

    # ---- 3. Honest within-silo validation of both winners --------------------------------
    emit("### Within-silo validation (the production surface, and the actual adoption gate)")
    emit()
    emit(f"5-seed within-silo OOF for the LOGO winner vs. the currently-shipped setting, for "
         f"both CatBoost and LogisticRegression. `NOISE_FLOOR = {NOISE_FLOOR}` applied exactly "
         f"as every other v2 experiment - only adopt if the LOGO winner clears it.")
    emit()

    cb_fn_mod = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_hparam_oof")
    print(f"Fetching within-silo OOF: CatBoost LOGO-winner ({best_it}/{best_d}) x {N_SEEDS} seeds ...")
    cb_winner_oof = {}
    for seed, res in enumerate(cb_fn_mod.starmap([(best_it, best_d, s) for s in range(N_SEEDS)])):
        cb_winner_oof[seed] = {uc: np.array(v) for uc, v in res.items()}
    print(f"Fetching within-silo OOF: CatBoost current ({current_it}/{current_d}) x {N_SEEDS} seeds ...")
    cb_current_oof = {}
    for seed, res in enumerate(cb_fn_mod.starmap([(current_it, current_d, s) for s in range(N_SEEDS)])):
        cb_current_oof[seed] = {uc: np.array(v) for uc, v in res.items()}

    print("Fitting within-silo OOF locally: LogReg LOGO-winner C and current C=1.0 ...")
    lr_winner_oof, lr_current_oof = {}, {}
    for seed in range(N_SEEDS):
        lr_winner_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_c_fn(cols, C=best_C))
        lr_current_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_fn(cols))

    def validate(winner_oof, current_oof, label):
        rows = []
        for uc in USE_CASE_ORDER:
            yu = y_by_uc[uc]
            w_aucs, c_aucs, w_f2s, c_f2s = [], [], [], []
            for seed in range(N_SEEDS):
                w, c = winner_oof[seed][uc], current_oof[seed][uc]
                w_aucs.append(roc_auc_score(yu, w))
                c_aucs.append(roc_auc_score(yu, c))
                _, wf2 = f2_optimal(yu, w)
                _, cf2 = f2_optimal(yu, c)
                w_f2s.append(wf2)
                c_f2s.append(cf2)
            rows.append({
                "use_case": USE_CASE_DISPLAY[uc],
                "current_auc": np.mean(c_aucs), "winner_auc": np.mean(w_aucs),
                "auc_gain": np.mean(w_aucs) - np.mean(c_aucs),
                "current_f2": np.mean(c_f2s), "winner_f2": np.mean(w_f2s),
                "f2_gain": np.mean(w_f2s) - np.mean(c_f2s),
            })
        vdf = pd.DataFrame(rows).set_index("use_case")
        emit(f"**{label}**")
        emit()
        emit(to_md(vdf.round(4), "use_case"))
        emit()
        mean_auc_gain, mean_f2_gain = vdf["auc_gain"].mean(), vdf["f2_gain"].mean()
        clears = "YES" if mean_auc_gain > NOISE_FLOOR else "no"
        verdict = "ADOPT" if mean_auc_gain > NOISE_FLOOR else "KEEP CURRENT"
        emit(f"Mean AUC gain **{mean_auc_gain:+.4f}**, mean F2@t* gain **{mean_f2_gain:+.4f}**. "
             f"Clears the noise floor: **{clears}**. **Verdict: {verdict}**.")
        emit()
        return verdict, mean_auc_gain

    cb_verdict, cb_gain = validate(cb_winner_oof, cb_current_oof,
                                    f"CatBoost: LOGO winner (iterations={best_it}, depth={best_d}) "
                                    f"vs. current (iterations={current_it}, depth={current_d})")
    lr_verdict, lr_gain = validate(lr_winner_oof, lr_current_oof,
                                    f"LogisticRegression: LOGO winner (C={best_C}) vs. current (C={CURRENT_LOGREG_C})")

    emit("### Summary")
    emit()
    emit(f"- CatBoost: {'adopt iterations=' + str(best_it) + '/depth=' + str(best_d) if cb_verdict == 'ADOPT' else f'keep current iterations={current_it}/depth={current_d}'} "
         f"({cb_verdict}, within-silo mean AUC gain {cb_gain:+.4f}).")
    emit(f"- LogisticRegression: {'adopt C=' + str(best_C) if lr_verdict == 'ADOPT' else f'keep current C={CURRENT_LOGREG_C}'} "
         f"({lr_verdict}, within-silo mean AUC gain {lr_gain:+.4f}).")
    if cb_verdict == "ADOPT" or lr_verdict == "ADOPT":
        emit("- If adopted, combiner weights (`reports/wf_ensemble_v2_chosen_weights.json`, §9) "
             "would need re-deriving against the new base-learner OOF as a follow-up - not done "
             "in this pass.")
    emit()

    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")

    return cb_verdict, best_it, best_d, lr_verdict, best_C


if __name__ == "__main__":
    main()
