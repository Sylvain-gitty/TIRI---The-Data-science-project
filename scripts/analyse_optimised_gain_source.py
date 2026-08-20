"""Where does the optimised-spec gain actually come from?

`wf_optimised_usecase_baseline.md` established *that* enriching a use-case objective is worth
+0.014 mean zero-label ROC-AUC, clearing the 0.03 floor on 2 of 6 use cases. It did not establish
*why those two*. This script tests the obvious hypothesis:

    "it improves the least well-defined use cases and has little effect on the rest"

which, if true, predicts two things that can be measured separately:

  **(a) spread narrows** — the six use cases end up closer together, because the weak ones come up
       and the strong ones stay put.
  **(b) gain tracks how badly-defined the spec was** — on some independent measure of that, not on
       the outcome itself.

Part (b) is where this is easy to fool yourself. "Least well-defined" is not one thing, and the
tempting proxy — *the use case with the lowest baseline score* — is not a measure of the spec at
all. It is the outcome, and ranking gain against it is regression to the mean by construction: any
noisy quantity gains most where it started lowest. So six proxies are used, five of which are
properties of the **spec text alone** and never see a paper or a label:

    baseline_auc        the outcome. Included ONLY to show what the tempting answer looks like
    linter_findings     scripts/spec_linter.py - this repo's own definition of a badly-written spec
    checkability        wf_checkability_audit.md - share of criteria that are checkable predicates
    objective_words     how thin the objective was before the rewrite
    n_terms             must_include + nice_to_have, before
    reservoir_chars     SUPPLY: how much analyst-written material sat in fields no feature reads
    dose_chars          how much the objective actually grew - supply x whether the rule used it

⚠️ **n = 6. This is descriptive and nothing here licenses a correlation claim** (`CONTEXT.md` §5,
and S-UCQ Q5's rule that 34 does not license one either). Spearman rho is printed because ranking
six things by two orders and seeing whether the orders agree is a legitimate description of six
data points; it is not evidence of a relationship in the population. Read the ranks, not the rho.

Part (c), the mechanism, is separate and is not a hypothesis test: for each use case it asks which
papers moved when the brief changed — did relevant papers climb, did irrelevant ones fall, and did
the movement happen at the top of the ranked list (where an analyst actually reads) or deep in the
tail (where it flatters AUC and changes nobody's day).

    python scripts/analyse_optimised_gain_source.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_optimised_usecase_baseline as ob  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402
from spec_linter import lint, normalise  # noqa: E402

OUT_MD = REPO / "reports" / "wf_optimised_gain_source.md"
OUT_CSV = REPO / "reports" / "wf_optimised_gain_source.csv"
CHECKABILITY = REPO / "reports" / "wf_checkability_audit.csv"

ENCODERS = ["jasper", "qwen4b", "qwen8b"]
# Fields no feature path reads - the reservoir the rewrite is allowed to draw on.
RESERVOIR = ["performance_criteria", "constraints", "decision_criteria", "notes"]
TOP_FRAC = 0.10   # "the top of the list" = what an analyst would actually read first


def build_arms(fe: pd.DataFrame, text: pd.DataFrame) -> dict[str, pd.DataFrame]:
    arms = {"baseline": fe}
    for arm in ("prose_only",):
        spec = ob.arm_fields(arm)
        frame = ob.add_lexical(fe, spec, text)
        frame = ob.add_cos_brief(frame, ob.brief_vectors(spec, arm))
        frame[ob.EXCL_FILL_COLS] = frame[ob.EXCL_FILL_COLS].fillna(0.0)
        arms[arm] = frame
    return arms


# ------------------------------------------------------------------ (a) spread


def spread(arms: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (encoder, use case) zero-label AUC, and how far apart the six use cases sit."""
    rows = []
    for enc in ENCODERS:
        col = f"cos_brief_{enc}"
        for uc in sorted(arms["baseline"][ob.USE_CASE_COL].unique()):
            rec = {"encoder": enc, "use_case": uc}
            for arm, frame in arms.items():
                g = frame[frame[ob.USE_CASE_COL] == uc]
                rec[arm] = roc_auc_score(g["y"], g[col])
            rec["gain"] = rec["prose_only"] - rec["baseline"]
            rows.append(rec)
    per = pd.DataFrame(rows)

    agg = []
    for enc, g in per.groupby("encoder"):
        agg.append({"encoder": enc,
                    "sd_before": g["baseline"].std(ddof=1), "sd_after": g["prose_only"].std(ddof=1),
                    "range_before": g["baseline"].max() - g["baseline"].min(),
                    "range_after": g["prose_only"].max() - g["prose_only"].min(),
                    "worst_before": g["baseline"].min(), "worst_after": g["prose_only"].min(),
                    "mean_before": g["baseline"].mean(), "mean_after": g["prose_only"].mean()})
    agg = pd.DataFrame(agg).set_index("encoder")
    agg["d_sd"] = agg["sd_after"] - agg["sd_before"]
    agg["d_range"] = agg["range_after"] - agg["range_before"]
    agg["d_worst"] = agg["worst_after"] - agg["worst_before"]
    agg["d_mean"] = agg["mean_after"] - agg["mean_before"]
    return per, agg.round(4)


# ------------------------------------------------- (b) how badly-defined was the spec?


def spec_measures(per: pd.DataFrame) -> pd.DataFrame:
    """Six ways to ask "how badly written was this spec?", five of them text-only."""
    base = ob.arm_fields("baseline")
    prose = ob.arm_fields("prose_only")
    check = pd.read_csv(CHECKABILITY).set_index("use_case")["checkability"]

    rows = []
    for uc in ob.USE_CASES:
        raw = json.loads((REPO / "data" / "raw" / f"{uc}.usecase.json").read_text(encoding="utf-8"))
        reservoir = sum(len(json.dumps(raw.get(f) or "")) for f in RESERVOIR)
        rows.append({
            "use_case": uc,
            "linter_findings": len(lint(normalise(raw))),
            "checkability": float(check.loc[uc]),
            "objective_words": len(str(base[uc]["objective"]).split()),
            "n_terms": len(base[uc]["terms_must_include"]) + len(base[uc]["terms_nice_to_have"]),
            "reservoir_chars": reservoir,
            "dose_chars": len(str(prose[uc]["objective"])) - len(str(base[uc]["objective"])),
        })
    m = pd.DataFrame(rows).set_index("use_case")
    g = per.groupby("use_case")
    m.insert(0, "baseline_auc", g["baseline"].mean().round(3))
    m.insert(1, "gain", g["gain"].mean().round(3))
    m.insert(2, "encoders_up", g["gain"].apply(lambda s: f"{int((s > 0).sum())}/3"))
    return m


# `worse` = the direction that means "less well defined", so a POSITIVE rho below is the
# hypothesis being supported: worse spec -> bigger gain.
WORSE_IS = {"baseline_auc": "low", "linter_findings": "high", "checkability": "low",
            "objective_words": "low", "n_terms": "low", "reservoir_chars": "high",
            "dose_chars": "high"}


def loo_stability(m: pd.DataFrame) -> pd.DataFrame:
    """Drop each use case in turn and re-rank. At n=6 a single point can carry a rho, and
    `solar_leo` ranks worst-defined on five of the seven measures, so this is not optional.

    Note the limit of the test, from `wf_foreign_brief_validity_setb.md`: leave-one-out shows
    whether one *point* carries a correlation. It cannot show whether the whole *surface* does —
    only a second surface can, and that is why §5 proposes one.
    """
    rows = []
    for col, worse in WORSE_IS.items():
        sign = -1.0 if worse == "low" else 1.0
        rec = {"measure": col, "all 6": round(float(spearmanr(sign * m[col], m["gain"]).statistic), 2)}
        loo = {}
        for uc in m.index:
            d = m.drop(index=uc)
            loo[uc] = float(spearmanr(sign * d[col], d["gain"]).statistic)
        rec["min (drop 1)"] = round(min(loo.values()), 2)
        rec["max (drop 1)"] = round(max(loo.values()), 2)
        rec["worst when dropping"] = min(loo, key=loo.get)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("measure")


DEPTHS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75)


def depth_profile(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Change in recall at increasing read depths. ROC-AUC is the area under this whole curve, so
    a gain in AUC says nothing about WHERE it happened — and only the shallow end is a screening
    workflow. If the whole gain sits past 50% depth it is real, measurable, and operationally
    invisible."""
    rows = []
    for enc in ENCODERS:
        col = f"cos_brief_{enc}"
        for uc in sorted(arms["baseline"][ob.USE_CASE_COL].unique()):
            b = arms["baseline"][arms["baseline"][ob.USE_CASE_COL] == uc]
            p = arms["prose_only"].loc[b.index]
            y = b["y"].to_numpy().astype(bool)
            ob_, op_ = np.argsort(-b[col].to_numpy()), np.argsort(-p[col].to_numpy())
            rec = {"encoder": enc, "use_case": uc}
            for f in DEPTHS:
                k = max(1, int(round(f * len(y))))
                rec[f"{int(f * 100)}%"] = (y[op_][:k].sum() - y[ob_][:k].sum()) / y.sum()
            rows.append(rec)
    return pd.DataFrame(rows)


def hypothesis_table(m: pd.DataFrame) -> pd.DataFrame:
    """For each proxy: rank the six use cases worst-spec-first, and see whether that order matches
    the gain order. Rank 1 = the worst-defined spec on that measure."""
    out = []
    for col, worse in WORSE_IS.items():
        sign = -1.0 if worse == "low" else 1.0
        oriented = sign * m[col].astype(float)
        rho = spearmanr(oriented, m["gain"]).statistic
        order = list(oriented.sort_values(ascending=False).index)
        out.append({
            "measure": col,
            "'worse' means": f"{worse}er value",
            "worst-defined spec": order[0],
            "2nd worst": order[1],
            "best-defined spec": order[-1],
            "rho vs gain": round(float(rho), 2),
        })
    return pd.DataFrame(out).set_index("measure")


# ------------------------------------------------------------------ (c) mechanism


def mechanism(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which papers moved, and where in the list.

    Percentile rank is zero-sum inside a use case, so "positives rose" and "negatives fell" are the
    same fact stated twice; both are printed because which one a reader finds intuitive varies.
    `d_recall_at_10pct` is the one that matters operationally: AUC counts a positive climbing from
    rank 900 to 600 exactly as much as one climbing from 50 to 20, and only the second changes what
    an analyst sees.
    """
    rows = []
    for enc in ENCODERS:
        col = f"cos_brief_{enc}"
        for uc in sorted(arms["baseline"][ob.USE_CASE_COL].unique()):
            b = arms["baseline"][arms["baseline"][ob.USE_CASE_COL] == uc]
            p = arms["prose_only"].loc[b.index]
            y = b["y"].to_numpy().astype(bool)
            rb = b[col].rank(pct=True).to_numpy()
            rp = p[col].rank(pct=True).to_numpy()
            d = rp - rb
            k = max(1, int(round(TOP_FRAC * len(y))))
            rec_b = y[np.argsort(-b[col].to_numpy())][:k].sum() / y.sum()
            rec_p = y[np.argsort(-p[col].to_numpy())][:k].sum() / y.sum()
            rows.append({
                "encoder": enc, "use_case": uc, "n": len(y), "n_pos": int(y.sum()),
                "d_auc": roc_auc_score(y, p[col]) - roc_auc_score(y, b[col]),
                "pos_rank_shift": float(d[y].mean()),
                "neg_rank_shift": float(d[~y].mean()),
                "share_pos_up": float((d[y] > 0).mean()),
                "d_recall_at_10pct": float(rec_p - rec_b),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()

    fe = pd.read_parquet(ob.FE)
    fe = fe[fe[ob.TARGET_COL].isin([ob.POSITIVE_LABEL, *ob.NEGATIVE_LABELS])].copy() \
        .reset_index(drop=True)
    fe["y"] = (fe[ob.TARGET_COL] == ob.POSITIVE_LABEL).astype(int)
    fe[ob.EXCL_FILL_COLS] = fe[ob.EXCL_FILL_COLS].fillna(0.0)
    pc = pd.read_parquet(ob.PC, columns=["paper_id", ob.USE_CASE_COL, "title", "abstract"])
    text = fe[["paper_id", ob.USE_CASE_COL]].merge(pc, on=["paper_id", ob.USE_CASE_COL],
                                                   how="left", validate="one_to_one")
    arms = build_arms(fe, text)

    per, agg = spread(arms)
    m = spec_measures(per)
    hyp = hypothesis_table(m)
    loo = loo_stability(m)
    mech = mechanism(arms)
    depth = depth_profile(arms)
    per.to_csv(OUT_CSV, index=False)
    mech.to_csv(str(OUT_CSV).replace(".csv", "_mechanism.csv"), index=False)
    depth.to_csv(str(OUT_CSV).replace(".csv", "_depth.csv"), index=False)

    print("=== (a) does the spread narrow? ===")
    print(to_md(agg[["sd_before", "sd_after", "d_sd", "range_before", "range_after", "d_range",
                     "worst_before", "worst_after", "d_worst", "d_mean"]]))

    print("\n=== (b) does the gain land on the worst-defined specs? ===")
    print(to_md(m.round(3)))
    print("\nrho > 0 would support the hypothesis (worse spec -> bigger gain). n=6, DESCRIPTIVE.")
    print(to_md(hyp))
    print("\nleave-one-out: can any single use case be dropped without the ordering collapsing?")
    print(to_md(loo))

    print("\n=== (c) which papers moved, mean over the three encoders ===")
    mm = mech.groupby("use_case")[["d_auc", "pos_rank_shift", "neg_rank_shift", "share_pos_up",
                                   "d_recall_at_10pct"]].mean().round(3)
    print(to_md(mm.sort_values("d_auc", ascending=False)))

    print("\n=== (c2) WHERE in the ranked list - change in recall by read depth ===")
    dd = depth.groupby("use_case")[[f"{int(f * 100)}%" for f in DEPTHS]].mean().round(3)
    dd.loc["ALL SIX (mean)"] = depth[[f"{int(f * 100)}%" for f in DEPTHS]].mean().round(3)
    print(to_md(dd))

    OUT_MD.write_text(_report(agg, m, hyp, loo, mm, dd, per, mech), encoding="utf-8")
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")


def _report(agg, m, hyp, loo, mm, dd, per, mech) -> str:
    worst_gain = m.loc[m["baseline_auc"].idxmin(), "gain"]
    best_gain = m.loc[m["baseline_auc"].idxmax(), "gain"]
    top_gainer = m["gain"].idxmax()
    return (
        "# Where does the optimised-spec gain come from?\n\n"
        "Status: **measured.** Follow-up to "
        "[`wf_optimised_usecase_baseline.md`](wf_optimised_usecase_baseline.md), which established "
        "that enriching a use-case objective is worth **+0.014** mean zero-label ROC-AUC and "
        "clears the 0.03 floor on 2 of 6 use cases — but not *which* 2, or why.\n\n"
        "🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding\n\n"
        "## 0. The hypothesis, and how it can be tested without fooling yourself\n\n"
        "> *\"It improves the least well-defined use cases and has less effect on the rest.\"*\n\n"
        "That predicts two separable things:\n\n"
        "**(a) the six use cases end up closer together** — the weak ones come up, the strong ones "
        "stay put, so the spread narrows.\n"
        "**(b) the gain lands on the badly-written specs** — measured on something independent of "
        "the outcome.\n\n"
        "🔴 **The tempting version of (b) is circular and this report keeps it visible so it can be "
        "dismissed on the record.** \"Which use case scored worst before?\" is not a measure of the "
        "spec — it is the outcome. Ranking gain against it is regression to the mean by "
        "construction: *any* noisy quantity gains most where it started lowest, whether or not "
        "briefs matter at all. So five of the six measures below are properties of the **spec text "
        "alone**, and never see a paper or a label.\n\n"
        "⚠️ **n = 6, and nothing here licenses a correlation claim.** Spearman rho is printed "
        "because ranking six things two ways and asking whether the orders agree is a fair "
        "description of six data points. It is not evidence about a population. Read the ranks.\n\n"
        "⚠️ Everything below is the **zero-label** regime, which is the only one where the answer "
        "is non-zero — the fitted baseline cannot separate the arms at all "
        "(`wf_optimised_usecase_baseline.md` §4).\n\n"
        "## 1. (a) Does the spread narrow?\n\n"
        "`sd` and `range` are taken across the six use cases' zero-label ROC-AUC, once per "
        "encoder. If the rewrite lifts the laggards, both shrink.\n\n"
        + to_md(agg[["sd_before", "sd_after", "d_sd", "range_before", "range_after", "d_range",
                     "worst_before", "worst_after", "d_worst", "d_mean"]]) + "\n\n"
        "*ELI18: `sd` is how far apart the six use cases' scores are. A negative `d_sd` means they "
        "converged; a positive one means the rewrite pushed them further apart. `d_worst` is what "
        "happened to the worst-performing use case specifically — the one the hypothesis says "
        "should benefit most.*\n\n"
        "## 2. (b) Does the gain land on the worst-defined specs?\n\n"
        + to_md(m.round(3)) + "\n\n"
        "**Rank agreement.** Each measure orders the six use cases worst-spec-first; `rho` says how "
        "well that order matches the gain order. **Positive rho supports the hypothesis.**\n\n"
        + to_md(hyp) + "\n\n"
        f"For the record, the two numbers the circular version turns on: the use case with the "
        f"**lowest** baseline gained **{worst_gain:+.3f}**, and the one with the **highest** gained "
        f"**{best_gain:+.3f}**. The largest gain of all went to **`{top_gainer}`**.\n\n"
        "### 2a. ⚠️ How much of that rests on one use case\n\n"
        "`solar_leo` ranks worst-defined on five of the seven measures *and* is a large gainer, so "
        "it could be carrying every positive rho on its own. Dropping each use case in turn:\n\n"
        + to_md(loo) + "\n\n"
        "🔴 **The limit of this test, stated because it has bitten this repo before.** "
        "`wf_foreign_brief_validity_setb.md` records it: leave-one-out shows whether a single "
        "*point* carries a correlation. It cannot show whether the whole *surface* does. Six "
        "analyst-written specs from one team is a surface, and it took a second surface to kill the "
        "foreign-brief margin after leave-one-out had passed it. §5 says what the second surface "
        "would be here.\n\n"
        "## 3. (c) Which papers moved, and where in the list\n\n"
        "Percentile rank is zero-sum inside a use case, so \"relevant papers rose\" and \"irrelevant "
        "papers fell\" are one fact stated twice. `d_recall_at_10pct` is the one that matters "
        "operationally: ROC-AUC counts a relevant paper climbing from rank 900 to 600 exactly as "
        "much as one climbing from 50 to 20, and only the second changes what an analyst sees.\n\n"
        + to_md(mm.sort_values("d_auc", ascending=False)) + "\n\n"
        "*ELI18: `pos_rank_shift` +0.02 means the average relevant paper moved up 2 percentiles — "
        "out of 100 papers, past two of them. `share_pos_up` is the fraction of relevant papers "
        "that moved up at all; 0.500 would mean the rewrite shuffled them without helping.*\n\n"
        "### 3a. The depth profile — the finding that outranks the rest\n\n"
        "ROC-AUC is the area under the whole ranked list, so an AUC gain says nothing about "
        "**where** it happened. Only the shallow end is a screening workflow: an analyst reads the "
        "top of the list and stops. Change in recall, by how far down the list you read:\n\n"
        + to_md(dd) + "\n\n"
        "*ELI18: each cell is \"how many more of the relevant papers you would have found, as a "
        "share of all of them, if you read that far down the list with the new brief instead of the "
        "old one\". 0.000 means the rewrite made no difference to what you would have found by "
        "then.*\n"
    )


if __name__ == "__main__":
    main()
