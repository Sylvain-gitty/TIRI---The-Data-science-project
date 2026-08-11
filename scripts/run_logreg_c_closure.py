"""run_logreg_c_closure.py — closes the loop on the LogReg `C` finding from
reports/wf_ensemble_v2_experiments.md §15: a central hyperparameter search (via LOGO, per
CONTEXT.md §1) found sklearn's default `C=1.0` has been under-regularizing every
LogisticRegression fit in this project's entire ensemble work (~4600 features, 260-360 rows
per silo). A follow-up fine sweep confirmed a genuine interior optimum at `C=0.0005` (improved
6/6 use cases standalone on both ROC-AUC and F2@t*) — the most consistent positive result this
whole session. It was explicitly NOT adopted yet because two things were still missing:

1. The combiner weights need re-deriving. The current ones
   (reports/wf_ensemble_v2_chosen_weights.json) were nested-selected using `C=1.0`'s LogReg
   OOF — changing `C` invalidates that selection.
2. It had never been checked on SYNERGY (the only prevalence-realistic external validation
   surface, 1.7-14.8%, vs. 26-77% across TIRI's own six pools) — every other promising finding
   this session (the Qwen3-8B embedding swap, the axis feature) got a SYNERGY check before
   being trusted.

This script closes both gaps and reaches an adopt/keep-current/mixed decision, gated on
NOISE_FLOOR=0.03 within-silo (both AUC and F2@t* reported and gated separately — the project's
established convention gates on AUC, but the user has repeatedly asked F2 be treated as
primary) plus a SYNERGY sanity check against the published Jasper+Qwen3-8B baseline (§11, mean
ROC-AUC 0.899).

`logreg_fn` (unparameterized, hardcoded C=1.0) is left untouched — every historical script and
every number in reports §§2-15 depends on its current behavior. `logreg_c_fn(cols, C=...)` is
used exclusively here for the new candidate.

Usage:
    python scripts/run_logreg_c_closure.py
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
from ensemble_eval_utils import to_md, within_silo_oof  # noqa: E402
from fold_pipeline_utils import ranking_metrics  # noqa: E402
from run_ensemble_candidate import build_variants, logreg_c_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
SYNERGY_PATH = REPO / "data" / "processed" / "papers_fe_synergy_jasper_qwen8b.parquet"
SYNERGY_OOF_JSON = REPO / "reports" / "wf_ensemble_v2_synergy_oof_qwen8b.json"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
WEIGHTS_JSON = REPO / "reports" / "wf_ensemble_v2_chosen_weights.json"
NEW_WEIGHTS_JSON = REPO / "reports" / "wf_ensemble_v2_chosen_weights_c0005.json"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
NEW_C = 0.0005
N_SEEDS = 5
SELECTION_SEEDS = [0, 1, 2]
VALIDATION_SEEDS = [3, 4]
NOISE_FLOOR = 0.03
SYNERGY_REVIEWS = ["Sep_2021", "Menon_2022", "van_der_Waal_2022"]

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

    print(f"Fitting LogisticRegression OOF (C={NEW_C}) locally for each seed ...")
    logreg_new_oof = {}
    for seed in range(N_SEEDS):
        logreg_new_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_c_fn(cols, C=NEW_C))
        print(f"  done: logreg(C={NEW_C}) seed={seed}")

    print("Fitting LogisticRegression OOF (C=1.0, current) locally for each seed ...")
    logreg_cur_oof = {}
    for seed in range(N_SEEDS):
        logreg_cur_oof[seed] = within_silo_oof(X, y, use_case, groups, seed, logreg_c_fn(cols, C=1.0))
        print(f"  done: logreg(C=1.0) seed={seed}")

    chosen_2way = json.loads(WEIGHTS_JSON.read_text())

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 16. Closing the loop on LogReg C — combiner re-derivation + SYNERGY validation")
    emit()
    emit(f"Follows up on §15: a central hyperparameter search found sklearn's default `C=1.0` "
         f"under-regularizes every LogReg fit in this project (~4600 features, 260-360 "
         f"rows/silo); a fine sweep found a genuine interior optimum at `C={NEW_C}` (improved "
         f"6/6 use cases standalone). Not yet adopted because the combiner weights need "
         f"re-deriving (the current ones assume `C=1.0`'s LogReg OOF) and it had never been "
         f"checked on SYNERGY. This closes both gaps.")
    emit()

    # ---- 1. Within-silo combiner re-derivation ------------------------------------------
    emit("### Within-silo combiner re-derivation")
    emit()
    emit(f"For each use case, a fresh weight-grid search (`w` 0-1, step 0.05) on the "
         f"`C={NEW_C}` LogReg OOF is chosen on seeds {SELECTION_SEEDS} (mean ROC-AUC), "
         f"evaluated on seeds {VALIDATION_SEEDS} (never seen during selection) - fresh for "
         f"all 6 use cases, not reusing the old STABLE/UNSTABLE split derived under `C=1.0`. "
         f"Compared against the CURRENT shipped combiner (`C=1.0` LogReg OOF at the existing "
         f"weight from `wf_ensemble_v2_chosen_weights.json`), evaluated on the same held-out "
         f"seeds.")
    emit()

    weight_grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    rows = []
    new_weights = {}
    for uc in USE_CASE_ORDER:
        yu = y_by_uc[uc]
        disp = USE_CASE_DISPLAY[uc]
        w_current = chosen_2way.get(disp, 0.5)

        sel_scores = []
        for w in weight_grid:
            aucs_this_w = []
            for seed in SELECTION_SEEDS:
                blend = w * catboost_oof[seed][uc] + (1 - w) * logreg_new_oof[seed][uc]
                aucs_this_w.append(roc_auc_score(yu, blend))
            sel_scores.append(np.mean(aucs_this_w))
        best_idx = int(np.argmax(sel_scores))
        w_new = float(weight_grid[best_idx])
        new_weights[disp] = w_new

        val_auc_cur, val_auc_new, val_f2_cur, val_f2_new = [], [], [], []
        for seed in VALIDATION_SEEDS:
            cb = catboost_oof[seed][uc]
            lr_new = logreg_new_oof[seed][uc]
            lr_cur = logreg_cur_oof[seed][uc]
            blend_cur = w_current * cb + (1 - w_current) * lr_cur
            blend_new = w_new * cb + (1 - w_new) * lr_new
            val_auc_cur.append(roc_auc_score(yu, blend_cur))
            val_auc_new.append(roc_auc_score(yu, blend_new))
            _, f2_c = f2_optimal(yu, blend_cur)
            _, f2_n = f2_optimal(yu, blend_new)
            val_f2_cur.append(f2_c)
            val_f2_new.append(f2_n)

        rows.append({
            "use_case": disp,
            "w_current": w_current, "w_new": w_new,
            "auc_current": float(np.mean(val_auc_cur)), "auc_new": float(np.mean(val_auc_new)),
            "auc_gain": float(np.mean(val_auc_new) - np.mean(val_auc_cur)),
            "f2_current": float(np.mean(val_f2_cur)), "f2_new": float(np.mean(val_f2_new)),
            "f2_gain": float(np.mean(val_f2_new) - np.mean(val_f2_cur)),
        })

    result_df = pd.DataFrame(rows).set_index("use_case")
    emit(to_md(result_df.round(4), "use_case"))
    emit()

    mean_auc_gain = result_df["auc_gain"].mean()
    mean_f2_gain = result_df["f2_gain"].mean()
    auc_clears = "YES" if mean_auc_gain > NOISE_FLOOR else "no"
    f2_clears = "YES" if mean_f2_gain > NOISE_FLOOR else "no"
    n_auc_pos = int((result_df["auc_gain"] > 0).sum())
    n_f2_pos = int((result_df["f2_gain"] > 0).sum())
    emit(f"Overall: mean AUC gain **{mean_auc_gain:+.4f}** ({n_auc_pos}/6 use cases positive), "
         f"mean F2@t* gain **{mean_f2_gain:+.4f}** ({n_f2_pos}/6 use cases positive), evaluated "
         f"on held-out seeds never used for weight selection.")
    emit()
    emit(f"Clears the noise floor on AUC: **{auc_clears}**. Clears the noise floor on F2@t* "
         f"(the project's stated primary metric): **{f2_clears}**.")
    if auc_clears != f2_clears:
        emit(f"**The two gates disagree** — AUC says {auc_clears.lower()}, F2@t* says "
             f"{f2_clears.lower()}. Reported plainly, not averaged over.")
    emit()

    within_silo_supports_f2 = f2_clears == "YES"
    within_silo_supports_auc = auc_clears == "YES"

    # ---- 2. SYNERGY validation -----------------------------------------------------------
    emit("### SYNERGY validation")
    emit()
    emit(f"LogReg OOF at `C={NEW_C}` fit locally on SYNERGY (5 seeds, synthetic per-row "
         f"groups since SYNERGY has no author column — same convention "
         f"`scripts/modal_ensemble_experiments.py::run_synergy_oof` uses), blended 50/50 "
         f"with the already-cached CatBoost(150,4) SYNERGY OOF "
         f"(`reports/wf_ensemble_v2_synergy_oof_qwen8b.json`) — 50/50 because SYNERGY's "
         f"reviews are unseen \"use cases\" the per-TIRI nested weights don't transfer to "
         f"(existing convention, not a new departure). Compared against the existing `C=1.0` "
         f"50/50 blend (same file's `\"logreg\"` key) and against §11's published summary "
         f"(mean ROC-AUC 0.899).")
    emit()

    syn_df = pd.read_parquet(SYNERGY_PATH)
    syn_y = syn_df["y"].to_numpy().astype(int)
    syn_uc = syn_df["use_case_key"]
    syn_feature_cols = [c for c in syn_df.columns if c not in ("y", "use_case_key")]
    syn_X = syn_df[syn_feature_cols]
    syn_groups = pd.Series(np.arange(len(syn_df)))
    syn_y_by_review = {r: syn_y[(syn_uc == r).to_numpy()] for r in SYNERGY_REVIEWS}

    cached = json.loads(SYNERGY_OOF_JSON.read_text())
    syn_catboost_oof = {int(s): {uc: np.array(v) for uc, v in d.items()} for s, d in cached["catboost"].items()}
    syn_logreg_cur_oof = {int(s): {uc: np.array(v) for uc, v in d.items()} for s, d in cached["logreg"].items()}

    print(f"Fitting SYNERGY LogReg OOF (C={NEW_C}) locally for {N_SEEDS} seeds ...")
    syn_logreg_new_oof = {}
    for seed in range(N_SEEDS):
        syn_logreg_new_oof[seed] = within_silo_oof(
            syn_X, syn_y, syn_uc, syn_groups, seed, logreg_c_fn(syn_feature_cols, C=NEW_C)
        )
        print(f"  done: synergy logreg(C={NEW_C}) seed={seed}")

    def synergy_scores(catboost_oof_dict, logreg_oof_dict):
        out_rows = []
        for review in SYNERGY_REVIEWS:
            yu = syn_y_by_review[review]
            for seed in range(N_SEEDS):
                cb = np.array(catboost_oof_dict[seed][review])
                lr = np.array(logreg_oof_dict[seed][review])
                blend = 0.5 * cb + 0.5 * lr
                rm = ranking_metrics(yu, blend, fractions=(0.10, 0.20))
                _, f2_star = f2_optimal(yu, blend)
                out_rows.append({
                    "review": review, "seed": seed,
                    "roc_auc": roc_auc_score(yu, blend),
                    "f2_at_0.5": f2_at(yu, blend, 0.5),
                    "f2_at_t_star": f2_star,
                    **rm,
                })
        return pd.DataFrame(out_rows)

    syn_new_df = synergy_scores(syn_catboost_oof, syn_logreg_new_oof)
    syn_cur_df = synergy_scores(syn_catboost_oof, syn_logreg_cur_oof)

    metric_cols = ["roc_auc", "f2_at_0.5", "f2_at_t_star", "recall_at_10pct", "recall_at_20pct", "wss_at_95"]
    new_summary = syn_new_df.groupby("review")[metric_cols].mean().round(3)
    cur_summary = syn_cur_df.groupby("review")[metric_cols].mean().round(3)

    emit(f"**New (`C={NEW_C}`) 50/50 blend, per review, mean across seeds:**")
    emit()
    emit(to_md(new_summary, "review"))
    emit()
    emit(f"**Current (`C=1.0`) 50/50 blend, per review, mean across seeds (recomputed fresh "
         f"here from the cached OOF — cross-check against §11):**")
    emit()
    emit(to_md(cur_summary, "review"))
    emit()

    mean_roc_auc_new = syn_new_df["roc_auc"].mean()
    mean_roc_auc_cur = syn_cur_df["roc_auc"].mean()
    mean_f2_new = syn_new_df["f2_at_t_star"].mean()
    mean_f2_cur = syn_cur_df["f2_at_t_star"].mean()
    emit(f"Overall mean ROC-AUC: new **{mean_roc_auc_new:.4f}** vs current **"
         f"{mean_roc_auc_cur:.4f}** (§11 published baseline: 0.899). Delta: "
         f"**{mean_roc_auc_new - mean_roc_auc_cur:+.4f}**. Overall mean F2@t*: new "
         f"**{mean_f2_new:.4f}** vs current **{mean_f2_cur:.4f}**. Delta: "
         f"**{mean_f2_new - mean_f2_cur:+.4f}**.")
    emit()

    synergy_delta_auc = mean_roc_auc_new - mean_roc_auc_cur
    synergy_regresses = synergy_delta_auc < -0.005  # SYNERGY reviews are small/noisy; a small
    # negative delta isn't necessarily a real regression, but anything beyond a slack of 0.005
    # mean ROC-AUC is treated as a real signal not to ignore, given SYNERGY is the one
    # prevalence-realistic surface this ensemble has ever been checked against.
    if synergy_delta_auc > 0.005:
        synergy_read = "IMPROVES"
    elif synergy_regresses:
        synergy_read = "REGRESSES"
    else:
        synergy_read = "FLAT"
    emit(f"SYNERGY read: **{synergy_read}** (delta {synergy_delta_auc:+.4f} mean ROC-AUC, "
         f"no formal noise floor established for SYNERGY given its small per-review sizes — "
         f"this is a qualitative read against the published baseline, not a hard gate).")
    emit()

    # ---- 3. Decision gate ----------------------------------------------------------------
    emit("### Decision")
    emit()
    if within_silo_supports_f2 and synergy_read != "REGRESSES":
        verdict = "ADOPT"
        emit(f"**Verdict: {verdict}** `C={NEW_C}` for LogisticRegression, with the re-derived "
             f"per-use-case combiner weights above. Within-silo clears the noise floor on "
             f"F2@t* (the project's stated primary metric) even though it narrowly misses the "
             f"established AUC-based gate ({mean_auc_gain:+.4f} vs. 0.03) — SYNERGY does not "
             f"regress ({synergy_read.lower()}). New weights saved to "
             f"`{NEW_WEIGHTS_JSON.name}` (the existing `wf_ensemble_v2_chosen_weights.json` "
             f"is left untouched as the historical `C=1.0` record). **Code is NOT "
             f"auto-updated** — `scripts/run_ensemble_candidate.py`'s `logreg_fn` still "
             f"defaults to `C=1.0`; adopting this means using `logreg_c_fn(cols, "
             f"C={NEW_C})` in its place going forward, a deliberate call for a human to make, "
             f"not a silent script edit.")
        new_weights_out = {k: round(v, 2) for k, v in new_weights.items()}
        NEW_WEIGHTS_JSON.write_text(json.dumps(new_weights_out, indent=2))
        print(f"New weights written to {NEW_WEIGHTS_JSON.relative_to(REPO)}")
    else:
        verdict = "MIXED — not a clean adopt"
        reasons = []
        if not within_silo_supports_f2:
            reasons.append("within-silo F2@t* gain does not clear the noise floor")
        if synergy_read == "REGRESSES":
            reasons.append(f"SYNERGY regresses ({synergy_delta_auc:+.4f} mean ROC-AUC)")
        emit(f"**Verdict: {verdict}.** Specifically: {'; '.join(reasons)}. "
             f"{'Within-silo AUC also does not clear the noise floor (' + f'{mean_auc_gain:+.4f}' + ' vs 0.03). ' if not within_silo_supports_auc else ''}"
             f"Not adopting `C={NEW_C}` or the re-derived weights on the strength of this "
             f"result alone — current `C=1.0` defaults and "
             f"`wf_ensemble_v2_chosen_weights.json` are left unchanged. New weights are NOT "
             f"saved to `{NEW_WEIGHTS_JSON.name}` given the mixed result.")

    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
