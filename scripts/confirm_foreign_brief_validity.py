"""Does the foreign-brief margin predict labelling gain on a surface nobody has selected against?

`P-FB` (`reports/wf_foreign_brief_detector.md`) found two things. The margin
(`own_brief_auc - best_foreign_brief_auc`) is **not** a brief defect — it fails the
cross-instrument test, so it is a property of one embedding's geometry rather than of the text.
But it predicts how much a collection **gains from labelling** at rho **-0.929**, surviving every
leave-one-out.

That second result is the only salvageable thing in the probe, and it was measured on **n=7, on
set A, with both sides of the correlation derived from set A** — a surface with five prior
selection passes. `CONTEXT.md` §4 is explicit that this is the position from which a finding
stops being trustworthy, so the finding was reported as exploratory and explicitly *not* as
independent confirmation.

This script buys the confirmation, and it is the one question `benchset_v1_large_set_b` was
reserved for. Set B has never been selected against — nothing in the repo had even opened it
until this week — so a relationship that holds there is a relationship, not a residue of tuning.

What is and is not new here
---------------------------
The **margin** is not recomputed: it comes off the same cached 34x34 matrices, which already
cover all 7 set-B collections (they are 7 of the 28 benchset use cases). What is new is the
**labelling-gain** side, which had only ever been measured on set A's 8 collections — hence
`run_label_budget_shape.py --set b`.

Pre-registered before looking (and the direction is the whole claim)
--------------------------------------------------------------------
  PASS: Spearman rho <= -0.5 on set B, same sign as set A, with the bootstrap CI reported
        beside it. Negative means: the worse a collection's margin, the MORE labelling buys.
  FAIL: rho above -0.5, or the sign flips. Either kills the caption - a predictor that
        reverses direction between surfaces predicts nothing.

Read the CI, not just the point estimate: at n=7 a bootstrap interval is wide by construction
(the same data shape gives rho 0.74 with a CI of 0.14 to 1.00 elsewhere in this repo), so "clears
the bar on the point estimate" and "established" are different statements and are kept apart.

Three instrument caveats, all inherited and none fixed here
----------------------------------------------------------
1. `auc_gain`'s floor is `cos_brief_qwen4b` on **weighted, held-out** rows; the matrix diagonal
   is **jasper over the whole collection**. Different instruments. Both margins are therefore
   reported (jasper and qwen4b) so the reader can see whether the choice matters, and the
   lexical margin too, since that is the representation the specificity test showed disagreeing.
2. n=7 on B and n=7 on A. Two small samples agreeing is not a large sample.
3. `nykvist_evcharging` is not a SYNERGY collection and its brief is written rather than derived
   from a review abstract, so it is the one set-B row whose margin means something slightly
   different. It is reported, never silently dropped, and the jackknife shows what it carries.

    python scripts/confirm_foreign_brief_validity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ensemble_eval_utils import partial_rho, spearman_ci, to_md  # noqa: E402

REPORTS = REPO / "reports"
OUT_MD = REPORTS / "wf_foreign_brief_validity_setb.md"
OUT_CSV = REPORTS / "wf_foreign_brief_validity_setb.csv"
BAR = -0.5
N_POS_REF = 30   # the grid cell the set-A result was read at; held fixed here


def margin(matrix: str) -> pd.Series:
    T = pd.read_csv(REPORTS / f"wf_foreign_brief_matrix_{matrix}.csv", index_col=0)
    own = pd.Series(np.diag(T), index=T.columns)
    off = T.where(~np.eye(len(T), dtype=bool))
    return own - off.max(axis=0)


def gain(set_name: str) -> pd.Series:
    """Mean AUC gained from N_POS_REF labelled positives, per collection."""
    sfx = "" if set_name == "a" else f"_set_{set_name}"
    g = pd.read_csv(REPORTS / f"wf_label_budget_shape{sfx}_grid.csv")
    return g[g.n_pos == N_POS_REF].groupby("use_case").auc_gain.mean()


def prevalence(set_name: str) -> pd.Series:
    sfx = "" if set_name == "a" else f"_set_{set_name}"
    y = pd.read_csv(REPORTS / f"wf_label_budget_shape{sfx}_yield.csv")
    return y.set_index("use_case").prevalence


def table(set_name: str) -> pd.DataFrame:
    g, p = gain(set_name), prevalence(set_name)
    out = pd.DataFrame({"auc_gain": g, "prevalence": p.reindex(g.index)})
    for m in ("jasper", "qwen4b", "lexical"):
        out[f"margin_{m}"] = margin(m).reindex(g.index)
    return out.dropna(subset=["auc_gain"])


def assess(df: pd.DataFrame, col: str) -> dict:
    d = df.dropna(subset=[col, "auc_gain"])
    r = spearman_ci(d[col], d.auc_gain)
    pr = partial_rho(d[col], d.auc_gain, d.prevalence) if d.prevalence.notna().all() else np.nan
    jack = [spearmanr(*d.drop(index=k)[[col, "auc_gain"]].values.T).statistic for k in d.index]
    return {"predictor": col, "n": r["n"], "rho": r["rho"], "p": r["p"],
            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "rho_given_prevalence": pr,
            "jack_min": min(jack), "jack_max": max(jack),
            "clears_bar": bool(r["rho"] <= BAR)}


def main() -> None:
    A, B = table("a"), table("b")
    print(f"set A: n={len(A)}   set B: n={len(B)}")

    rows = []
    for name, df in (("A (burned)", A), ("B (clean)", B)):
        for col in ("margin_jasper", "margin_qwen4b", "margin_lexical"):
            rows.append({"surface": name, **assess(df, col)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    head = res[res.predictor == "margin_jasper"].set_index("surface")
    a, b = head.loc["A (burned)"], head.loc["B (clean)"]
    same_sign = np.sign(a.rho) == np.sign(b.rho)
    passed = bool(b.clears_bar and same_sign)

    print(f"\nset A jasper: rho {a.rho:+.3f}  set B jasper: rho {b.rho:+.3f}  -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(to_md(res.set_index(["surface", "predictor"]).round(3)))

    verdict = "PASS" if passed else "FAIL"
    OUT_MD.write_text(
        "# Does the foreign-brief margin predict labelling gain on a clean surface?\n\n"
        f"Status: **{verdict}**. Generated by `scripts/confirm_foreign_brief_validity.py`. $0 — no "
        "LLM, no GPU. Read `reports/wf_foreign_brief_detector.md` first; this confirms (or kills) "
        "the one salvageable result in it.\n\n"
        "🟢 clears the pre-registered bar · 🟡 real but not established · ⚪ engineering finding\n\n"
        "## 0. How to read any number here\n\n"
        "**Spearman rho** measures whether two rankings move together, from **−1** (perfectly "
        "opposite) through **0** (unrelated) to **+1** (identical). Here it asks: *do the "
        "collections with the worst foreign-brief margins tend to be the ones that gain most from "
        "labelling papers by hand?* If the margin were meaningless, rho would sit near **0**.\n\n"
        "**The sign is the claim, not the size.** A negative rho means a bad margin goes with a big "
        "labelling payoff — which is the direction that makes the number useful as a *budget* "
        "warning. A positive rho would mean the opposite and would kill it outright.\n\n"
        f"**The bar was fixed before looking: rho ≤ {BAR} on set B, same sign as set A.**\n\n"
        "⚠️ **Read the confidence interval, not the point estimate.** At n=7 a bootstrap interval "
        "is wide by construction. \"Clears the bar\" and \"established\" are different claims and "
        "are kept apart below. `jack_min`/`jack_max` show the range when each collection is dropped "
        "in turn — if one collection is carrying the whole result, that range is where it shows.\n\n"
        "**Label which critic is speaking.** rho, the CI and the jackknife are hard numbers. "
        "Whether n=7 on two surfaces amounts to replication is a judgement call, and §2 marks it.\n\n"
        "## 1. The result\n\n"
        + to_md(res.set_index(["surface", "predictor"]).round(3)) + "\n\n"
        f"**Pre-registered bar: {verdict}.** It required rho ≤ {BAR} on set B with the same sign as "
        f"set A. Set B's jasper margin gives **rho {b.rho:+.3f}** (n={int(b.n)}, CI "
        f"{b.ci_lo:+.3f} to {b.ci_hi:+.3f}) against set A's **{a.rho:+.3f}**.\n\n"
        "## 2. Per-collection inputs\n\n"
        "### Set B (clean)\n\n" + to_md(B.round(3)) + "\n\n"
        "### Set A (burned, for comparison)\n\n" + to_md(A.round(3)) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_MD}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
