"""F2 across train / test / validate, pooled over all six use cases.

Why a split at all, when the model is zero-shot
-----------------------------------------------
Nothing in P1/P2/P4 is fitted, so in the usual sense every row is already held out and
three splits of an unfitted scorer would differ only by sampling noise. But one parameter
*is* fitted, and it is the one that decides F2: **the threshold**. `f2_at_t_star` (the
number the ensemble's published 0.893 uses) sweeps 91 thresholds and keeps the best on the
same rows it scored — an oracle, and an upper bound.

So the split is over the threshold, not the model:

    train     pick t* here (the only thing being learned)
    test      apply that t*, never re-picked
    validate  apply that t*, never re-picked

train-vs-test is then a direct measure of how much the oracle flatters, and test-vs-validate
is a stability check. `f2_at_own` — the model's own verdict, nothing fitted — is reported
alongside as the reference that should NOT move across splits.

Splits are grouped by `first_author` and stratified on use case x label, the same
discipline `within_silo` uses, so an author's papers never straddle a boundary.

Pooled vs per-use-case
----------------------
"Combined" is reported as a genuine pool: all rows in one pile, ONE global threshold. That
is the deployment-shaped question if a single config ships. It is not the same as the mean
of per-use-case F2s, and at 26-77% prevalence the pooled figure is partly a function of the
mix, so both are printed. CONTEXT.md §1's "no pooled model ever ships" is about fitting a
model across silos; this pools only the *evaluation*, which is why it is admissible here.

Usage:
    python scripts/analyze_llm_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESPONSES = REPO / "reports" / "wf_llm_grid_responses.parquet"
SLIM = REPO / "data" / "processed" / "papers_fe_slim.parquet"
OUT_MD = REPO / "reports" / "wf_llm_split_results.md"
THRESHOLDS = np.linspace(0.01, 0.99, 99)
SEED = 0


def f2(y, pred):
    return fbeta_score(y, pred, beta=2, zero_division=0)


def make_splits(df: pd.DataFrame) -> pd.Series:
    """60/20/20 train/test/validate, grouped by first author, stratified on use case x label."""
    strat = df["use_case_key"].astype(str) + "_" + df["y"].astype(str)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(skf.split(df, strat, groups=df["first_author"])):
        folds[idx] = i
    return pd.Series(np.where(folds < 3, "train", np.where(folds == 3, "test", "validate")),
                     index=df.index)


def main() -> None:
    resp = pd.read_parquet(RESPONSES)
    slim = pd.read_parquet(SLIM, columns=["paper_id", "use_case_key", "first_author", "y"])
    slim["split"] = make_splits(slim)
    split_by = dict(zip(zip(slim.paper_id, slim.use_case_key), slim.split))
    resp["split"] = [split_by.get((p, u)) for p, u in zip(resp.paper_id, resp.use_case_key)]

    counts = slim.groupby("split").agg(n=("y", "size"), positive=("y", "sum"))
    counts["prevalence"] = (counts.positive / counts.n).round(3)
    # The number that decides how to read everything below. F2 weights recall 5:1, so at
    # high prevalence "mark everything relevant" already scores well: precision = prevalence,
    # recall = 1, F2 = 5p/(4p+1). At the pooled 57.7% that is 0.872. Any F2 near it is not
    # evidence of screening ability - it is evidence of the prevalence. CONTEXT.md §3 warns
    # these pools run ~20x production prevalence; this is what that warning looks like in a
    # metric.
    counts["f2_all_positive"] = (5 * counts.prevalence / (4 * counts.prevalence + 1)).round(3)

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# LLM screening — F2 across train / test / validate (all six use cases pooled)")
    emit()
    emit("The threshold is the only fitted parameter: **t\\* is chosen on train and applied "
         "unchanged to test and validate.** `f2_own` is the model's own verdict with nothing "
         "fitted, and should not move across splits.")
    emit()
    emit(to_md(counts.reset_index(), "split"))
    emit()

    rows = []
    for (model, variant), g in resp.groupby(["model", "variant"]):
        g = g[g["score"].notna() & g["split"].notna()]
        tr = g[g.split == "train"]
        if len(tr) < 50:
            continue
        y_tr, s_tr = tr["y"].to_numpy().astype(int), tr["score"].to_numpy()
        # the ONLY thing learned, and it is learned on train alone
        t_star = max(THRESHOLDS, key=lambda t: f2(y_tr, (s_tr >= t).astype(int)))

        rec = {"model": model.split("/")[-1], "variant": variant, "t_star": round(float(t_star), 2)}
        for name in ("train", "test", "validate"):
            sub = g[g.split == name]
            y = sub["y"].to_numpy().astype(int)
            s = sub["score"].to_numpy()
            pred = sub["pred"].astype(float).to_numpy()
            m = ~np.isnan(pred)
            rec[f"f2_{name}"] = f2(y, (s >= t_star).astype(int))
            # Rank on log-odds where they exist. For a logprob variant, `score` is the
            # sigmoid of the margin and saturates at temperature 0, so an AUC computed on it
            # measures the squashing rather than the ordering. F2 still uses `score`, which
            # is a genuine probability and is what a threshold has to be applied to.
            rank_col = ("score_logodds"
                        if "score_logodds" in sub and sub["score_logodds"].notna().any()
                        else "score")
            rs = sub[rank_col].to_numpy()
            keep = ~np.isnan(rs)
            rec[f"auc_{name}"] = (roc_auc_score(y[keep], rs[keep])
                                  if len(np.unique(y[keep])) > 1 else np.nan)
            rec[f"f2own_{name}"] = f2(y[m], pred[m].astype(int))
        rec["oracle_train_minus_test"] = rec["f2_train"] - rec["f2_test"]
        rows.append(rec)

    res = pd.DataFrame(rows)
    base_test = float(counts.loc["test", "f2_all_positive"])
    res["lift_over_all_pos"] = (res["f2_test"] - base_test).round(3)
    # How close is the tuned threshold to simply saying yes to everything?
    res["pos_rate_at_t"] = [
        float((resp[(resp.model.str.endswith(m)) & (resp.variant == v)
                    & (resp.split == "test") & resp.score.notna()]["score"] >= t).mean())
        for m, v, t in zip(res.model, res.variant, res.t_star)
    ]

    emit("## All cells")
    emit()
    show = res[["model", "variant", "t_star", "f2_train", "f2_test", "f2_validate",
                "lift_over_all_pos", "pos_rate_at_t", "f2own_test", "auc_test"]].round(3)
    emit(to_md(show.sort_values(["model", "variant"]), ""))
    emit()

    emit("## Best variant per model (selected on **train** F2, never on test)")
    emit()
    best = res.loc[res.groupby("model")["f2_train"].idxmax()]
    b = best[["model", "variant", "t_star", "f2_train", "f2_test", "f2_validate",
              "lift_over_all_pos", "f2own_test", "auc_test"]].round(3)
    emit(to_md(b.sort_values("f2_test", ascending=False), ""))
    emit()
    emit(f"### Read the `lift_over_all_pos` column first")
    emit()
    emit(f"At the pooled test prevalence of **{counts.loc['test', 'prevalence']:.3f}**, "
         f"marking *every* paper relevant scores **F2 = {base_test:.3f}**. That is the floor "
         "these numbers sit on. The ensemble's 0.893 is **+0.021** over it and the best cell "
         f"here is **{res.f2_test.max() - base_test:+.3f}**. Pooled F2 at this prevalence "
         "barely discriminates - `pos_rate_at_t` shows the tuned threshold is driving the "
         "models toward saying yes to nearly everything, because at 58% positive that is "
         "very nearly the right answer. This is exactly the ~20x-production-prevalence "
         "distortion CONTEXT.md §3 warns about, made visible. Rank metrics (`auc_test`) and "
         "the per-use-case numbers in `wf_llm_pilot_findings.md` carry the real signal.")
    emit()
    emit(f"Mean train→test drop from re-using a fitted threshold: "
         f"**{best.oracle_train_minus_test.mean():+.3f}** "
         f"(max {best.oracle_train_minus_test.max():+.3f}). That is the size of the optimism "
         "in any F2 quoted at its own optimal threshold.")
    emit()
    emit("Ensemble reference: within-silo **F2@t\\* 0.893**, itself an oracle number "
         "(threshold swept on the rows it scores), so it carries the same optimism as the "
         "`f2_train` column, not the `f2_test` one.")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n")
    res.to_csv(REPO / "reports" / "wf_llm_split_results.csv", index=False)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
