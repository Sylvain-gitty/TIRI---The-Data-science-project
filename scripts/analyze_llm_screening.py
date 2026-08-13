"""Score the LLM screening grid against the incumbent ensemble.

Reads `reports/wf_llm_grid_responses.parquet` (written by `run_llm_screening.py`) and
answers the five questions the pilot was set up to answer, with the stop conditions from
`reports/wf_llm_screening_plan.md` §9.

Metric discipline carried over from the rest of the repo
-------------------------------------------------------
`f2_optimal` is imported from `analyze_ensemble_v2_experiments`, not re-implemented, so
the LLM is scored by the identical rule that produced the ensemble's published 0.893.
That rule sweeps 91 thresholds and keeps the best **on the same predictions it selected
on** - it is an oracle, and an upper bound. It is reported for both sides or neither.

Two honest operating points are reported alongside it:

- `f2_at_own`  - the LLM's own hard verdict, no threshold fitted at all. This is the
  number an LLM actually delivers in production, and the one that should survive a
  prevalence shift when a swept threshold does not.
- `f2_at_0.5`  - a fixed threshold on the graded score, for comparability with the
  ensemble's own `f2_at_0.5` column.

Ties matter here in a way they do not for a classifier. A model asked for a 0-100 score
answers in round numbers, so its ranking carries large tied blocks; `n_distinct` and
`tie_frac` are reported next to every AUC because a heavily tied score is penalised by
ROC-AUC in a way that is a property of the *elicitation method*, not the model's judgement.

Unparsed rows are dropped from metrics and reported as `coverage`, never imputed to 0.5.
NULL is not 0: "the model did not answer" is a different fact from "the model was unsure",
and averaging a fabricated midpoint into a recall metric would flatter exactly the models
that failed most often.

Usage
-----
    python scripts/analyze_llm_screening.py
    python scripts/analyze_llm_screening.py --responses reports/wf_llm_smoke_responses.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ensemble_v2_experiments import f2_at, f2_optimal  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SLIM = REPO / "data" / "processed" / "papers_fe_slim.parquet"
HPARAM_JSON = REPO / "reports" / "wf_ensemble_v2_hparam_oof.json"
CATBOOST_KEY = "tuned (iterations=150, depth=4)"

# Published within-silo held-out-seed numbers for the shipped 2-branch ensemble
# (reports/wf_ensemble_v2_experiments.md §9). Quoted, not recomputed - recomputing the
# blend needs a Modal CatBoost refit across 5 seeds, and these are the numbers the
# recommendation doc actually stands on.
ENSEMBLE_PUBLISHED = {
    "carbon_capture": {"roc_auc": 0.905, "f2_at_t_star": 0.896},
    "cement_binders": {"roc_auc": 0.911, "f2_at_t_star": 0.949},
    "ner": {"roc_auc": 0.834, "f2_at_t_star": 0.930},
    "soil_microbiome": {"roc_auc": 0.827, "f2_at_t_star": 0.726},
    "solar_leo": {"roc_auc": 0.875, "f2_at_t_star": 0.950},
    "tech_forecasting": {"roc_auc": 0.835, "f2_at_t_star": 0.907},
}
# CONTEXT.md §5: seed-to-seed sd is ~0.010 on within-silo ROC-AUC. Treat anything under
# this as not established, rather than reaching for a p-value.
NOISE_FLOOR = 0.03


def load_ensemble_oof() -> pd.DataFrame:
    """Tuned-CatBoost out-of-fold probabilities, row-aligned to papers_fe_slim.

    This is the CatBoost *branch* at seed 0, not the shipped CatBoost+LogReg blend - it is
    what exists on disk without a refit. Used for the branch-correlation question (which
    only needs one branch's opinions) and as a like-for-like same-rows reference. The
    headline "did we beat the ensemble" comparison uses ENSEMBLE_PUBLISHED instead.
    """
    slim = pd.read_parquet(SLIM, columns=["paper_id", "use_case_key", "y"])
    oof = json.loads(HPARAM_JSON.read_text())[CATBOOST_KEY]
    frames = []
    for uc, probs in oof.items():
        rows = slim[slim.use_case_key == uc]
        if len(rows) != len(probs):
            raise RuntimeError(f"{uc}: {len(rows)} rows vs {len(probs)} oof values")
        frames.append(rows.assign(cb_oof=np.asarray(probs, dtype=float)))
    return pd.concat(frames, ignore_index=True)


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray | None = None) -> dict:
    """Ranking + threshold metrics for one (cell, use case)."""
    out: dict[str, float] = {}
    if len(np.unique(y)) < 2:
        return out
    out["roc_auc"] = roc_auc_score(y, score)
    out["pr_auc"] = average_precision_score(y, score)
    k = max(1, int(round(0.10 * len(y))))
    top = np.argsort(-score)[:k]
    out["recall_at_10pct"] = float(y[top].sum() / y.sum()) if y.sum() else np.nan
    out["f2_at_t_star"] = f2_optimal(y, score)[1]
    out["f2_at_0.5"] = f2_at(y, score, 0.5)
    if pred is not None and not np.isnan(pred).all():
        ok = ~np.isnan(pred)
        from sklearn.metrics import fbeta_score

        out["f2_at_own"] = fbeta_score(y[ok], pred[ok].astype(int), beta=2, zero_division=0)
        out["pred_pos_rate"] = float(np.nanmean(pred))
    out["n_distinct"] = float(len(np.unique(score)))
    _, counts = np.unique(score, return_counts=True)
    out["tie_frac"] = float((counts[counts > 1].sum()) / len(score))
    return out


def elicitation_compare(res: pd.DataFrame) -> pd.DataFrame:
    """P2 (verbalised 0-100) vs P2lp (token log-odds), per model and use case.

    The one comparison the pilot's ranking result hinged on. Asked for a number, every model
    answered in round figures - 9-16 distinct values across 1,848 rows, tie fraction >0.99 -
    and ROC-AUC scores ties at half credit, so the LLM was penalised for how the score was
    *elicited* rather than for how well it judged. Log-odds are continuous by construction.

    `score_logodds` is used for the P2lp ranking, not `score`: renormalising to a probability
    saturates at temperature 0 and throws the granularity away again (see
    `parse_logprob_screening`).
    """
    rows = []
    for (model, variant), g in res.groupby(["model", "variant"]):
        if variant not in ("P2", "P2lp"):
            continue
        col = "score_logodds" if variant == "P2lp" and "score_logodds" in g else "score"
        for uc, sub in g.groupby("use_case_key"):
            ok = sub[sub[col].notna()]
            y = ok["y"].to_numpy().astype(int)
            if len(np.unique(y)) < 2 or len(ok) < 10:
                continue
            s = ok[col].to_numpy()
            _, counts = np.unique(s, return_counts=True)
            rows.append({
                "model": model.split("/")[-1], "variant": variant, "use_case": uc,
                "roc_auc": roc_auc_score(y, s),
                "n_distinct": len(np.unique(s)),
                "tie_frac": float(counts[counts > 1].sum() / len(s)),
            })
    return pd.DataFrame(rows)


def per_cell(res: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    """Per (model, variant, use_case) metrics, plus coverage.

    Cells with fewer than `min_rows` parsed responses get coverage only and no metrics -
    an ROC-AUC over a handful of rows is noise wearing a number's clothes. That is the
    normal outcome on the smoke subset (6 rows per use case) and a red flag on the grid.
    """
    rows = []
    for (model, variant, uc), g in res.groupby(["model", "variant", "use_case_key"]):
        ok = g[g["score"].notna()]
        base = {
            "model": model, "variant": variant, "use_case": uc,
            "n": len(g), "coverage": round(len(ok) / len(g), 3),
        }
        if len(ok) < min_rows:
            rows.append(base)
            continue
        pred = ok["pred"].astype(float).to_numpy()
        rows.append({**base, **metrics(ok["y"].to_numpy().astype(int),
                                       ok["score"].to_numpy(), pred)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--responses", type=Path,
                    default=REPO / "reports" / "wf_llm_grid_responses.parquet")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "wf_llm_pilot_results.md")
    ap.add_argument("--min-rows", type=int, default=10,
                    help="minimum parsed rows per (cell, use case) before metrics are computed")
    args = ap.parse_args()

    res = pd.read_parquet(args.responses)
    ens = load_ensemble_oof()
    res = res.merge(ens[["paper_id", "use_case_key", "cb_oof"]],
                    on=["paper_id", "use_case_key"], how="left")

    cell = per_cell(res, min_rows=args.min_rows)
    if "roc_auc" not in cell.columns:
        raise SystemExit(
            f"no (cell, use case) had >= {args.min_rows} parsed responses - nothing to "
            "score. Expected on the smoke subset; on the grid it means the run failed."
        )
    metric_cols = [c for c in ("roc_auc", "pr_auc", "recall_at_10pct", "f2_at_t_star",
                               "f2_at_own", "f2_at_0.5", "pred_pos_rate", "coverage",
                               "n_distinct", "tie_frac") if c in cell.columns]

    # ---- headline: mean over the 6 use cases, per cell -------------------------------
    summary = cell.groupby(["model", "variant"])[metric_cols].mean().round(3)
    ens_auc = np.mean([v["roc_auc"] for v in ENSEMBLE_PUBLISHED.values()])
    ens_f2 = np.mean([v["f2_at_t_star"] for v in ENSEMBLE_PUBLISHED.values()])
    summary["auc_vs_ens"] = (summary["roc_auc"] - ens_auc).round(3)
    summary["f2_vs_ens"] = (summary["f2_at_t_star"] - ens_f2).round(3)

    # ---- per-use-case win counts (CONTEXT.md §5: never a mean alone) ------------------
    wins = []
    for (model, variant), g in cell.groupby(["model", "variant"]):
        w_auc = sum(
            1 for _, r in g.iterrows()
            if r.use_case in ENSEMBLE_PUBLISHED and pd.notna(r.get("roc_auc"))
            and r["roc_auc"] > ENSEMBLE_PUBLISHED[r.use_case]["roc_auc"]
        )
        w_f2 = sum(
            1 for _, r in g.iterrows()
            if r.use_case in ENSEMBLE_PUBLISHED and pd.notna(r.get("f2_at_t_star"))
            and r["f2_at_t_star"] > ENSEMBLE_PUBLISHED[r.use_case]["f2_at_t_star"]
        )
        wins.append({"model": model, "variant": variant,
                     "auc_wins": f"{w_auc}/6", "f2_wins": f"{w_f2}/6"})
    summary = summary.join(pd.DataFrame(wins).set_index(["model", "variant"]))

    # ---- branch correlation: is this a genuinely different opinion? -------------------
    corr = []
    for (model, variant), g in res.groupby(["model", "variant"]):
        ok = g[g["score"].notna() & g["cb_oof"].notna()]
        # Correlated *within* a silo, never pooled: a pooled correlation would be inflated
        # by between-use-case differences in difficulty, which is not the question. The
        # question is whether, for one customer, the LLM disagrees with CatBoost enough to
        # be worth a vote (CONTEXT.md §1: no pooled model, ever).
        per_uc = [
            sub["score"].corr(sub["cb_oof"], method="spearman")
            for _, sub in ok.groupby("use_case_key") if len(sub) > 10
        ]
        per_uc = [v for v in per_uc if v == v]  # drop NaN (constant score in a silo)
        corr.append({"model": model, "variant": variant,
                     "rho_vs_catboost": round(float(np.mean(per_uc)), 3) if per_uc else np.nan})
    corr_df = pd.DataFrame(corr).set_index(["model", "variant"])
    summary = summary.join(corr_df)

    print("\n=== HEADLINE (mean over 6 use cases) ===")
    print(f"incumbent ensemble: ROC-AUC {ens_auc:.3f}  F2@t* {ens_f2:.3f}")
    print(summary.to_string())

    print("\n=== PER USE CASE, best cell by F2@t* ===")
    best = summary["f2_at_t_star"].idxmax()
    bc = cell[(cell.model == best[0]) & (cell.variant == best[1])].set_index("use_case")
    bc = bc[[c for c in ("roc_auc", "f2_at_t_star", "f2_at_own", "coverage") if c in bc]]
    bc["ens_auc"] = [ENSEMBLE_PUBLISHED[u]["roc_auc"] for u in bc.index]
    bc["ens_f2"] = [ENSEMBLE_PUBLISHED[u]["f2_at_t_star"] for u in bc.index]
    print(f"best cell: {best[0]} / {best[1]}")
    print(bc.round(3).to_string())

    print("\n=== STOP CONDITIONS (plan §9) ===")

    def _best(pattern: str, col: str = "roc_auc"):
        """Best value for models matching `pattern`, or None if that slot never ran.

        Returning None rather than NaN matters: a NaN silently propagates through the
        comparisons below and prints a verdict ("scale matters") derived from nothing.
        A missing slot should say so.
        """
        sel = summary.loc[summary.index.get_level_values(0).str.contains(pattern), col]
        sel = sel.dropna()
        return float(sel.max()) if len(sel) else None

    top_auc = _best(".")
    top_f2 = _best(".", "f2_at_t_star")
    rho = summary["rho_vs_catboost"].dropna()
    small, big = _best("gpt-oss"), _best("397b")

    if top_auc is None:
        print("   no scored cells - cannot evaluate")
    else:
        print(f"1. ranking      : best LLM ROC-AUC {top_auc:.3f} vs ensemble {ens_auc:.3f} "
              f"({'BELOW 0.75 -> STOP' if top_auc < 0.75 else 'above floor'})")
    if small is None or big is None:
        print("2. scale lever  : incomplete - need both the 20B and 397B slots")
    else:
        gap = big - small
        print(f"2. scale lever  : 20B {small:.3f} vs 397B {big:.3f} -> gap {gap:+.3f} "
              f"({'scale is NOT the lever' if abs(gap) < NOISE_FLOOR else 'scale matters'})")
    if not len(rho):
        print("3. decorrelation: not computable")
    else:
        print(f"3. decorrelation: min rho vs CatBoost {rho.min():.3f} "
              f"({'THIRD-BRANCH CASE' if rho.min() < 0.7 else 'too correlated'})")
    if top_f2 is not None:
        print(f"4. F2           : best LLM F2@t* {top_f2:.3f} vs ensemble {ens_f2:.3f} "
              f"-> {top_f2 - ens_f2:+.3f} (noise floor {NOISE_FLOOR})")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        "# LLM screening pilot - results\n\n"
        f"Incumbent ensemble (published, within-silo held-out seeds): "
        f"ROC-AUC **{ens_auc:.3f}**, F2@t* **{ens_f2:.3f}**.\n\n"
        "`f2_at_t_star` is an oracle threshold on both sides. `f2_at_own` is the LLM's "
        "own verdict with no threshold fitted - the honest operating point.\n\n"
        "## Headline\n\n" + to_md(summary.reset_index(), "") +
        "\n\n## Per use case (best cell)\n\n" + to_md(bc.round(3).reset_index(), "use_case") + "\n"
    )
    cell.to_csv(REPO / "reports" / "wf_llm_pilot_per_use_case.csv", index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
