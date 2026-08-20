"""Does a prompted-LLM branch add a genuine edge to the CatBoost(150,4) + LogReg ensemble?

Fourth candidate in the "does a third branch add diversity" line, after SVM (§12),
lexical/metadata-only (§13) and k-NN (§14) — all three rejected, each for a documented
reason. The one-line summary of those three:

    SVM      rho 0.86-0.91  strong standalone   REJECTED - too correlated
    lexical  rho 0.52-0.67  weak standalone     REJECTED - disagreement too weak a vote
    k-NN     rho 0.74-0.87  never best          REJECTED - mechanism difference too small

Read together they say the blocker is a *joint* condition: a third branch has to be both
decorrelated from the existing pair AND strong enough standalone for its disagreement to
carry weight. Nothing tested so far has been both. An LLM reads raw text against the brief
and never touches the ~4600-dim feature matrix at all, so it is the first candidate with a
genuinely different feature view that is also competitive standalone (mean within-silo
ROC-AUC 0.819 vs the ensemble's 0.865, winning outright on 3/6 use cases).

Two deliberate departures from the earlier three scripts, both forced by what an LLM is
------------------------------------------------------------------------------------
1. **No Modal, no 5-seed refit.** This runs at seed 0 only, using the CatBoost(150,4) OOF
   already on disk (`wf_ensemble_v2_hparam_oof.json`) and a local LogReg fit. It is a
   screening run: if the 3-way gain is clearly inside the noise floor here, the answer is
   reject and the full 5-seed Modal protocol is not worth buying. Escalate only on a
   promising result.

2. **Weight selection is leave-one-use-case-out, not leave-seeds-out.** This matters and is
   not a shortcut. The earlier scripts chose weights on seeds [0,1,2] and validated on
   [3,4], which is honest for branches that are *refit* per seed — their OOF genuinely
   changes. An LLM branch has no seed: the same prompt gives the same score for row i in
   every "seed", so a weight tuned to how well the LLM happens to fit these specific rows
   would carry that fit straight into validation and the seed split would never catch it.
   Selecting the weight on five use cases and applying it to the held-out sixth removes
   that optimism, because the validation rows played no part in choosing the weight.
   It also happens to be the deployment-shaped question (CONTEXT.md §1: what crosses a silo
   boundary is a configuration, never weights fitted on that silo's own data).

Usage:
    python scripts/run_llm_branch_experiment.py
    python scripts/run_llm_branch_experiment.py --variant P1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md, within_silo_oof  # noqa: E402
from run_ensemble_candidate import build_variants, logreg_fn  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
HPARAM_JSON = REPO / "reports" / "wf_ensemble_v2_hparam_oof.json"
WEIGHTS_JSON = REPO / "reports" / "wf_ensemble_v2_chosen_weights.json"
RESPONSES = REPO / "reports" / "wf_llm_grid_responses.parquet"
OUT_MD = REPO / "reports" / "wf_llm_branch_experiment.md"
CATBOOST_KEY = "tuned (iterations=150, depth=4)"
WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
NOISE_FLOOR = 0.03
SEED = 0

USE_CASE_DISPLAY = {
    "carbon_capture": "Carbon Capture",
    "cement_binders": "Low-Carbon Cement",
    "solar_leo": "Solar Cells for Satellites",
    "soil_microbiome": "Soil Microbiome",
    "ner": "Named Entity Recognition",
    "tech_forecasting": "Technology Prediction",
}
CANDIDATES = [
    "google/gemma-4-31b-it",
    "qwen/qwen3.5-397b-a17b",
    "openai/gpt-oss-20b",
]


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
    """(w_cb, w_lr, w_llm) triples on a step grid summing to 1."""
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    out = []
    for w_cb in vals:
        for w_lr in vals:
            w_llm = round(1 - w_cb - w_lr, 2)
            if -1e-9 <= w_llm <= 1 + 1e-9:
                out.append((float(w_cb), float(w_lr), float(max(0.0, w_llm))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="P2", help="prompt variant to use as the branch")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PATH)
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]
    order = sorted(use_case.unique())
    cols = build_variants(df)[WINNING_VARIANT]
    X = df[cols]

    y_by_uc = {uc: y[(use_case == uc).to_numpy()] for uc in order}
    pid_by_uc = {uc: df.loc[(use_case == uc).to_numpy(), "paper_id"].tolist() for uc in order}

    print(f"CatBoost(150,4) OOF from disk (seed {SEED}) ...")
    cb = {uc: np.asarray(v, dtype=float)
          for uc, v in json.loads(HPARAM_JSON.read_text(encoding="utf-8"))[CATBOOST_KEY].items()}

    print("Fitting LogisticRegression OOF locally ...")
    lr = within_silo_oof(X, y, use_case, groups, SEED, logreg_fn(cols))

    resp = pd.read_parquet(RESPONSES)
    chosen_2way = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8")) if WEIGHTS_JSON.exists() else {}
    grid = weight_simplex()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Does a prompted-LLM branch earn a place in the ensemble? (variant {args.variant})")
    emit()
    emit("Screening run: seed 0 only, CatBoost OOF read from disk, weights chosen "
         "leave-one-use-case-out. See this script's docstring for why the weight protocol "
         "differs from the SVM/lexical/k-NN experiments (an LLM branch has no seed, so a "
         "leave-seeds-out split cannot catch weight overfitting to these rows).")
    emit()

    for model in CANDIDATES:
        sub = resp[(resp.model == model) & (resp.variant == args.variant)]
        score_by_pid = dict(zip(sub.paper_id, sub.score))

        llm, mask = {}, {}
        for uc in order:
            s = np.array([score_by_pid.get(p, np.nan) for p in pid_by_uc[uc]], dtype=float)
            llm[uc] = s
            mask[uc] = ~np.isnan(s)  # never impute a missing verdict; drop the row

        dropped = sum(int((~m).sum()) for m in mask.values())
        emit(f"## {model}")
        emit()
        if dropped:
            emit(f"{dropped} row(s) had no parseable LLM score and are excluded from **both** "
                 f"the 2-way and 3-way numbers below, so the comparison is on identical rows.")
            emit()

        # ---- diversity -------------------------------------------------------------
        rows = []
        for uc in order:
            m = mask[uc]
            rows.append({
                "use_case": USE_CASE_DISPLAY[uc],
                "rho_llm_cb": spearmanr(llm[uc][m], cb[uc][m]).statistic,
                "rho_llm_lr": spearmanr(llm[uc][m], lr[uc][m]).statistic,
                "rho_cb_lr": spearmanr(cb[uc][m], lr[uc][m]).statistic,
            })
        div = pd.DataFrame(rows).set_index("use_case")
        emit("### Branch disagreement (Spearman on OOF scores, within silo)")
        emit()
        emit(to_md(div.round(3), "use_case"))
        emit()
        emit(f"Mean: LLM-vs-CatBoost **{div.rho_llm_cb.mean():.3f}**, LLM-vs-LogReg "
             f"**{div.rho_llm_lr.mean():.3f}**, and the existing production pair's own "
             f"CatBoost-vs-LogReg **{div.rho_cb_lr.mean():.3f}** on these same rows.")
        emit()

        # ---- nested 3-way vs the shipped 2-way -------------------------------------
        def blend(uc, w, m):
            w_cb, w_lr, w_llm = w
            return w_cb * cb[uc][m] + w_lr * lr[uc][m] + w_llm * llm[uc][m]

        def score_w(uc, w, metric):
            m = mask[uc]
            yv = y_by_uc[uc][m]
            p = blend(uc, w, m)
            return roc_auc_score(yv, p) if metric == "auc" else f2_optimal(yv, p)[1]

        res_rows = []
        for uc in order:
            m = mask[uc]
            yv = y_by_uc[uc][m]
            w_cb2 = chosen_2way.get(USE_CASE_DISPLAY[uc], 0.5)
            base = w_cb2 * cb[uc][m] + (1 - w_cb2) * lr[uc][m]

            # weights chosen on the OTHER five use cases only
            best, best_val = None, -np.inf
            for w in grid:
                val = np.mean([score_w(o, w, "auc") for o in order if o != uc])
                if val > best_val:
                    best, best_val = w, val
            three = blend(uc, best, m)
            res_rows.append({
                "use_case": USE_CASE_DISPLAY[uc],
                "w_cb": best[0], "w_lr": best[1], "w_llm": best[2],
                "auc_2way": roc_auc_score(yv, base),
                "auc_3way": roc_auc_score(yv, three),
                "auc_gain": roc_auc_score(yv, three) - roc_auc_score(yv, base),
                "f2_2way": f2_optimal(yv, base)[1],
                "f2_3way": f2_optimal(yv, three)[1],
                "f2_gain": f2_optimal(yv, three)[1] - f2_optimal(yv, base)[1],
            })
        out = pd.DataFrame(res_rows).set_index("use_case")
        emit("### Nested 3-way blend vs the shipped 2-way "
             "(weights chosen on the other five use cases)")
        emit()
        emit(to_md(out.round(3), "use_case"))
        emit()
        ag, fg = out.auc_gain.mean(), out.f2_gain.mean()
        emit(f"Mean AUC gain **{ag:+.3f}**, mean F2@t* gain **{fg:+.3f}** "
             f"(noise floor {NOISE_FLOOR}). "
             f"Positive on {int((out.auc_gain > 0).sum())}/6 use cases by AUC, "
             f"{int((out.f2_gain > 0).sum())}/6 by F2.")
        emit()
        verdict = ("CLEARS the noise floor" if max(ag, fg) > NOISE_FLOOR
                   else "does NOT clear the noise floor")
        emit(f"**Verdict: {verdict}.**")
        emit()

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
