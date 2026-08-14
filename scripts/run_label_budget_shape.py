"""How many labels, in what composition, before the supervised rung is worth switching on?

`CONTEXT.md` §1 ships a ladder with one number on it: *cosine-to-brief below ~25 in-silo
labels, supervised in-silo model above*. That 25 comes from `wf_query_conditioned_findings.md`
§5's warm-start curve, which was measured on TIRI's own pools at **26-77% prevalence** — where
a random 25-label draw contains ~10 positives. At set A's 2.19% it contains **0.4**, and 57% of
such draws contain none at all (§6 of the same report). So the shipped rule is stated in the one
unit that does not survive the prevalence change it will be deployed across.

This script restates it in units that do. Two measurements, both on `benchset_v1_large_set_a`,
both on the same held-out rows as `compare_setA_low_label.py` so the numbers are comparable:

**A. Yield.** Ranking the pool by cosine-to-brief, how many papers must a human read to reach
   k positives? Against the random baseline (k / prevalence). This is the number the labelling
   UX actually needs and it appears in no report: a label budget is spent in *cards*, and only
   the yield curve converts cards into the positives a model needs.

**B. Shape.** A (n_pos x n_neg) grid — train on exactly n_pos positives and n_neg negatives,
   score the held-out rows. `compare_setA_low_label.py` measured one cell of this (30/30) and
   read it as "60 labels". Whether that 60 was really 30 positives, or really 60 labels, is the
   whole question for a UX that cannot choose the ratio directly — it can only choose what to
   deal, and the ranking decides the rest.

Why LogisticRegression on the embedding and not the full ensemble: it is the arm that reached
0.844 at 60 labels in `wf_llm_benchset_a_low_label.md` (against the cosine's 0.774), it is one
of the two shipped branches, and at 12-130 training rows a CatBoost fit is measuring its own
priors. `C=0.0005` follows `wf_ensemble_final_recommendations.md` #9, which recommends exactly
that for a silo with no history — which is definitionally every silo at this label count.

What this cannot say, declared up front:

- **Set A's briefs derive from each review's own abstract** (`docs/BENCHSETS.md`, quoted in
  S-FF §2a). The cosine ranking is therefore a paraphrase of the answer key's cover page, and
  **experiment A's yield is contaminated upward**. It is an optimistic bound on how well a
  cold-start ranker seeds a labelling session. Reported because the *shape* of the curve and
  its gap to random are still informative, and because a bound is worth more than a guess.
- **Eight SYNERGY collections, 26 of 28 of which are clinical.** `CONTEXT.md` §4's warning
  against carrying anything across a prevalence regime applies to this script's own output.
- Experiment B draws negatives uniformly from the case-control sample's negatives, which is a
  fair draw from the population negatives (they were sampled uniformly). It is **not** a
  simulation of what the deck would deal — that is what A is for, and joining the two is a
  product decision, not a measurement.

    python scripts/run_label_budget_shape.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import benchset_metrics as bm  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402
from run_ensemble_candidate import logreg_c_fn  # noqa: E402

FEATURES = REPO / "data" / "processed" / "setA_ensemble_features.parquet"
OUT_MD = REPO / "reports" / "wf_label_budget_shape.md"
OUT_YIELD = REPO / "reports" / "wf_label_budget_shape_yield.csv"
OUT_GRID = REPO / "reports" / "wf_label_budget_shape_grid.csv"
OUT_NEGPOL = REPO / "reports" / "wf_label_budget_shape_negpolicy.csv"
FIG = REPO / "reports" / "wf_label_budget_shape.png"

COS = "cos_brief_qwen4b"
LOGREG_C = 0.0005  # wf_ensemble_final_recommendations.md #9 - the no-history starting point

N_POS_GRID = [2, 3, 5, 10, 20, 30]
N_NEG_GRID = [10, 20, 50, 100]
K_POSITIVES = [1, 2, 3, 5, 10, 25]


# --------------------------------------------------------------------------- A. yield


def cards_to_k(y: np.ndarray, score: np.ndarray, w: np.ndarray, k: int) -> float:
    """Population papers read, ranking by `score`, before the k-th positive is found.

    Ties are resolved by expectation, matching `benchset_metrics.wss_at`: inside a block of
    equal scores the positives are assumed uniformly spread. Without that the answer depends
    on the sort's arbitrary order inside a tie block, which is not a measurement.
    """
    n_pos = y.sum()
    if n_pos < k:
        return float("nan")
    seen_pos = seen_w = 0.0
    for by, bw in bm._blocks(score, y, w):
        p, wt = by.sum(), bw.sum()
        if seen_pos + p >= k:
            need = k - seen_pos
            return float(seen_w + wt * (need / p if p else 0.0))
        seen_pos += p
        seen_w += wt
    return float("nan")


def yield_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uc, g in df.groupby("use_case_key"):
        y = g["y"].to_numpy(float)
        w = g["w"].to_numpy(float)
        pop_n, prev = w.sum(), y.sum() / w.sum()
        rec = {"use_case": uc, "pop_n": int(round(pop_n)), "prevalence": prev}
        for k in K_POSITIVES:
            ranked = cards_to_k(y, g[COS].to_numpy(float), w, k)
            rec[f"cos_to_{k}pos"] = ranked
            rec[f"rand_to_{k}pos"] = k / prev if prev else float("nan")
        # the direct UX number: positives in the first 50 cards, either way
        rec["pos_per_50_cos"] = bm.recall_at(y, g[COS].to_numpy(float), w, 50 / pop_n) * y.sum()
        rec["pos_per_50_rand"] = 50 * prev
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("prevalence").reset_index(drop=True)


# --------------------------------------------------------------------------- B. shape


def fit_score(train: pd.DataFrame, evalset: pd.DataFrame, cols: list[str]) -> np.ndarray:
    pipe = logreg_c_fn(cols, C=LOGREG_C)()
    pipe.fit(train[cols], train["y"].to_numpy())
    return pipe.predict_proba(evalset[cols])[:, 1]


def shape_grid(df: pd.DataFrame, emb_cols: list[str], seeds: int) -> pd.DataFrame:
    rows = []
    for uc, g in df.groupby("use_case_key"):
        train_pool = g[g["split"] == "train"]
        ev = g[g["split"].isin(["test", "validate"])]
        y_ev, w_ev = ev["y"].to_numpy(float), ev["w"].to_numpy(float)
        if y_ev.sum() < 5:
            continue

        pos_pool = train_pool[train_pool["y"] == 1]
        neg_pool = train_pool[train_pool["y"] == 0]

        # the 0-label floor, on the identical eval rows
        cos_ev = ev[COS].to_numpy(float)
        floor = {
            "auc": roc_auc_score(y_ev, cos_ev),
            "recall10": bm.recall_at(y_ev, cos_ev, w_ev, 0.10),
        }

        for n_pos in N_POS_GRID:
            for n_neg in N_NEG_GRID:
                if len(pos_pool) < n_pos or len(neg_pool) < n_neg:
                    continue
                aucs, recs = [], []
                for seed in range(seeds):
                    rng = np.random.default_rng(1000 * seed + 7)
                    tr = pd.concat([
                        pos_pool.iloc[rng.choice(len(pos_pool), n_pos, replace=False)],
                        neg_pool.iloc[rng.choice(len(neg_pool), n_neg, replace=False)],
                    ])
                    s = fit_score(tr, ev, emb_cols)
                    aucs.append(roc_auc_score(y_ev, s))
                    recs.append(bm.recall_at(y_ev, s, w_ev, 0.10))
                rows.append({
                    "use_case": uc, "n_pos": n_pos, "n_neg": n_neg, "n_labels": n_pos + n_neg,
                    "auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                    "recall10": float(np.mean(recs)), "recall10_sd": float(np.std(recs)),
                    "auc_floor": floor["auc"], "recall10_floor": floor["recall10"],
                    "auc_gain": float(np.mean(aucs)) - floor["auc"],
                    "recall10_gain": float(np.mean(recs)) - floor["recall10"],
                })
        print(f"  {uc}: done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- D. which negatives


def negative_policy_grid(df: pd.DataFrame, emb_cols: list[str], seeds: int) -> pd.DataFrame:
    """Does it matter *which* negatives get labelled?

    Experiment B draws negatives uniformly, which is the deep-pool draw. Two other policies are
    what actually happens, or what an analyst might do deliberately:

      - `hard`  - the highest-ranked negatives, i.e. the ones a ranked deck deals first. This is
                  the negative an analyst labels *by default*, not by choice.
      - `easy`  - the lowest-ranked negatives. What deliberately sampling the deep pool gives you.
      - `random`- uniform over the silo's negatives. Experiment B's policy, kept as the control.

    The question this settles for the product: is "go and label some negatives" a distinct act
    worth a step in the flow, or is it something a ranked deck already does for free?
    """
    rows = []
    for uc, g in df.groupby("use_case_key"):
        train_pool = g[g["split"] == "train"]
        ev = g[g["split"].isin(["test", "validate"])]
        y_ev, w_ev = ev["y"].to_numpy(float), ev["w"].to_numpy(float)
        if y_ev.sum() < 5:
            continue
        pos_pool = train_pool[train_pool["y"] == 1]
        neg_pool = train_pool[train_pool["y"] == 0].sort_values(COS, ascending=False)
        floor_auc = roc_auc_score(y_ev, ev[COS].to_numpy(float))

        for n_pos in (10, 20):
            if len(pos_pool) < n_pos:
                continue
            for n_neg in (10, 20, 50):
                if len(neg_pool) < n_neg:
                    continue
                for policy in ("hard", "mixed", "random", "easy"):
                    aucs = []
                    for seed in range(seeds):
                        rng = np.random.default_rng(1000 * seed + 7)
                        p = pos_pool.iloc[rng.choice(len(pos_pool), n_pos, replace=False)]
                        if policy == "hard":
                            n = neg_pool.head(n_neg)
                        elif policy == "easy":
                            n = neg_pool.tail(n_neg)
                        elif policy == "mixed":
                            # what a ranked deck plus some deeper labelling actually gives you:
                            # half from the boundary, half from the interior.
                            half = n_neg // 2
                            rest = neg_pool.iloc[half:]
                            n = pd.concat([
                                neg_pool.head(half),
                                rest.iloc[rng.choice(len(rest), n_neg - half, replace=False)],
                            ])
                        else:
                            n = neg_pool.iloc[rng.choice(len(neg_pool), n_neg, replace=False)]
                        s = fit_score(pd.concat([p, n]), ev, emb_cols)
                        aucs.append(roc_auc_score(y_ev, s))
                    rows.append({
                        "use_case": uc, "n_pos": n_pos, "n_neg": n_neg, "policy": policy,
                        "auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                        "auc_gain": float(np.mean(aucs)) - floor_auc,
                    })
        print(f"  neg-policy {uc}: done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def chart(yield_df: pd.DataFrame, grid: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.scatter(yield_df.prevalence * 100, yield_df.cos_to_5pos, label="cosine-ranked", s=60)
    ax.scatter(yield_df.prevalence * 100, yield_df.rand_to_5pos, label="random order",
               s=60, marker="x")
    for _, r in yield_df.iterrows():
        ax.plot([r.prevalence * 100] * 2, [r.cos_to_5pos, r.rand_to_5pos], color="grey",
                lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pool prevalence (%)"); ax.set_ylabel("cards read to reach 5 positives")
    ax.set_title("A. Yield — what the query costs you in cards")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    common = set.intersection(*(set(grid[grid.n_pos == n].use_case) for n in grid.n_pos.unique()))
    piv = grid[grid.use_case.isin(common)].groupby(["n_pos", "n_neg"]).auc_gain.mean().unstack()
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("negatives labelled"); ax.set_ylabel("positives labelled")
    ax.set_title("B. Mean ROC-AUC gain over the 0-label cosine")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    for n_neg, g in grid[grid.use_case.isin(common)].groupby("n_neg"):
        m = g.groupby("n_pos").auc_gain.mean()
        ax.plot(m.index, m.values, marker="o", label=f"{n_neg} negatives")
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-0.03, 0.03, color="grey", alpha=0.2, label="±0.03 noise floor")
    ax.set_xlabel("positives labelled"); ax.set_ylabel("mean ROC-AUC gain over cosine")
    ax.set_title("C. Positives are the axis that moves it")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    meta = ["use_case_key", "y", "w", "split", COS]
    import pyarrow.parquet as pq

    emb_cols = sorted(
        c for c in pq.ParquetFile(FEATURES).schema_arrow.names if c.startswith("emb_qwen4b_")
    )
    print(f"loading {len(emb_cols)} embedding columns + {len(meta)} meta ...", flush=True)
    df = pd.read_parquet(FEATURES, columns=meta + emb_cols)

    print("A. yield ...", flush=True)
    ydf = yield_table(df[meta])
    ydf.to_csv(OUT_YIELD, index=False)

    print(f"B. shape grid, {args.seeds} seeds ...", flush=True)
    grid = shape_grid(df, emb_cols, args.seeds)
    grid.to_csv(OUT_GRID, index=False)

    print("D. which negatives ...", flush=True)
    negpol = negative_policy_grid(df, emb_cols, args.seeds)
    negpol.to_csv(OUT_NEGPOL, index=False)

    chart(ydf, grid)

    ydf["lift_50"] = ydf.pos_per_50_cos / ydf.pos_per_50_rand
    ycols = ["use_case", "pop_n", "prevalence", "cos_to_1pos", "cos_to_5pos", "cos_to_25pos",
             "rand_to_5pos", "pos_per_50_cos", "pos_per_50_rand", "lift_50"]

    # `sep_2021` has only 24 train positives, so it drops out of the n_pos=30 row and a raw
    # mean over n_pos compares different collection sets. CONTEXT.md §5: report win counts
    # beside means, and never let a mean change its own denominator.
    common = set.intersection(*(set(grid[grid.n_pos == n].use_case) for n in grid.n_pos.unique()))
    matched = grid[grid.use_case.isin(common)]
    by_pos = matched.groupby("n_pos")[["auc", "auc_gain", "recall10_gain"]].mean()
    by_neg = matched.groupby("n_neg")[["auc", "auc_gain", "recall10_gain"]].mean()
    by_tot = grid.groupby("n_labels")[["auc", "auc_gain"]].mean()
    wins = (grid.assign(win=grid.auc_gain > 0.03).groupby(["n_pos", "n_neg"]).win.sum()
            .unstack().fillna(0).astype(int))

    # Heterogeneity, and the diagnostic it suggests. Exploratory — not pre-registered.
    ref = grid[(grid.n_pos == 10) & (grid.n_neg == 50)].merge(ydf, on="use_case")
    from scipy.stats import spearmanr  # noqa: PLC0415

    rho_hidden = spearmanr(ref.auc_floor, ref.auc_gain)
    rho_lift = spearmanr(ref.lift_50, ref.auc_gain)
    rho_raw = spearmanr(ref.pos_per_50_cos, ref.auc_gain)
    het = ref[["use_case", "prevalence", "pos_per_50_rand", "pos_per_50_cos", "lift_50",
               "auc_floor", "auc", "auc_gain", "auc_sd"]].sort_values("lift_50")

    OUT_MD.write_text(
        "# Label budget and shape — how many positives, how many negatives, and what the query costs\n\n"
        f"Generated by `scripts/run_label_budget_shape.py --seeds {args.seeds}` on "
        "`benchset_v1_large_set_a` (8 SYNERGY collections, 0.17–22.3% prevalence). "
        "Held-out rows are `test`+`validate`, identical to `compare_setA_low_label.py`.\n\n"
        "> **Read the script's docstring before any number here.** Set A's briefs derive from "
        "each review's own abstract, so experiment A's cosine yield is an **optimistic bound**, "
        "not an estimate.\n\n"
        "## A. Yield — cards a human must read to reach k positives\n\n"
        + to_md(ydf[ycols].round(3)) + "\n\n"
        f"## B. Shape — by positives labelled (matched subset, {len(common)} collections)\n\n"
        + to_md(by_pos.round(4)) + "\n\n"
        f"## B. Shape — by negatives labelled (matched subset, {len(common)} collections)\n\n"
        + to_md(by_neg.round(4)) + "\n\n"
        "## B. Shape — by total labels (the unit the shipped ladder is stated in)\n\n"
        "Non-monotone, and that is the point: 52 labels as 2 positives + 50 negatives scores "
        "**below** the zero-label ranker, while 30 labels as 20 + 10 clears it. A total label "
        "count does not determine the answer.\n\n"
        + to_md(by_tot.round(4)) + "\n\n"
        "## B. Collections clearing the +0.03 noise floor (of 8; 7 at n_pos=30)\n\n"
        + to_md(wins) + "\n\n"
        "## C. Heterogeneity — where does labelling actually pay? (10 positives, 50 negatives)\n\n"
        f"Spearman against ROC-AUC gain, n=8, **exploratory — not pre-registered**:\n\n"
        f"- cold-start cosine AUC (**not observable at label 0**): rho "
        f"{rho_hidden.statistic:+.3f}, p={rho_hidden.pvalue:.4f}\n"
        f"- **lift of the first 50 cards over the pool base rate** (observable): rho "
        f"{rho_lift.statistic:+.3f}, p={rho_lift.pvalue:.4f}\n"
        f"- raw positives in the first 50 cards (observable, unnormalised): rho "
        f"{rho_raw.statistic:+.3f}, p={rho_raw.pvalue:.4f} — **not established**\n\n"
        + to_md(het.round(3)) + "\n\n"
        "## D. Which negatives? (mean ROC-AUC gain over the 0-label ranker)\n\n"
        "`hard` = the highest-ranked negatives, i.e. the ones a ranked deck deals anyway. "
        "`easy` = the lowest-ranked, i.e. what deliberately sampling the deep pool gives you.\n\n"
        + to_md(negpol.pivot_table(index=["n_pos", "n_neg"], columns="policy",
                                   values="auc_gain").round(4)) + "\n\n"
        + "Per-collection win counts, `hard` vs `random`: "
        + str(int((negpol.pivot_table(index=["use_case", "n_pos", "n_neg"], columns="policy",
                                      values="auc_gain").eval("hard > random")).sum()))
        + f" of {negpol.use_case.nunique() * 6} cells\n\n"
        "## B. Full grid\n\n"
        + to_md(grid.groupby(["n_pos", "n_neg"])[["auc", "auc_gain", "auc_sd", "recall10_gain"]]
                .mean().round(4)) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
