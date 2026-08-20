"""Set A results: does a prompted LLM beat the cold-start ranker at 2.19% prevalence?

Everything here is computed at **population prevalence** from the case-control sample, via
`benchset_metrics`. Read that module's docstring before trusting a number: recall and AUC
come straight off the sample, but precision, F2, WSS@95 and recall@k all need the weight
`w`, and a reader who forgets it gets a flatteringly wrong answer.

What is fitted where
--------------------
Nothing in P1/P2 is fitted, so the zero-shot arms are reported on all 9,993 sampled rows -
every row is held out for a prompt that learned nothing. Two things ARE fitted and are
handled separately:

  - **The threshold.** `f2_star` sweeps 91 thresholds and keeps the best on the rows it
    scored: an oracle, quoted for the LLM and the cosine baseline alike or for neither.
    `f2_own` - the model's own verdict, nothing tuned - is the honest operating point.
  - **The induced rule set (B2).** It saw train labels, so the whole brief ladder is scored
    on **test + validate only**, where every arm is on identical unseen rows.

Usage:
    python scripts/analyze_benchset_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_metrics import (  # noqa: E402
    all_positive_f2, f2_at, f2_optimal, recall_at, work_at_own, wss_at,
)
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
MANIFEST = REPORTS / "wf_llm_setA_sample.parquet"
OUT_MD = REPORTS / "wf_llm_benchset_a.md"
BASELINE = "cos_brief_qwen4b"
FLOOR = 0.03  # CONTEXT.md §5: gaps under this are not established
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def arm_of(stem: str) -> str:
    """Which brief a response file was produced under. Encoded in the filename by design."""
    if "control" in stem or "shuffled" in stem:
        return "B✗ shuffled"
    if "_raw" in stem:
        return "B0 raw"
    if "_induced" in stem:
        return "B2 induced"
    return "B1 supplied"


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} missing - regenerate with\n"
                         "    python scripts/benchset_loader.py --write-manifest")
    man = pd.read_parquet(MANIFEST)
    frames = []
    for path in sorted(REPORTS.glob("wf_llm_setA_*_responses.parquet")):
        stem = path.stem.replace("_responses", "")
        if "smoke" in stem:
            continue
        d = pd.read_parquet(path)
        d["arm"] = arm_of(stem)
        d["source"] = stem
        frames.append(d)
    if not frames:
        raise SystemExit("no set-A response files in reports/ - run the grid first")
    resp = pd.concat(frames, ignore_index=True)
    resp = resp.drop(columns=["y"]).merge(
        man[["paper_id", "use_case_key", "y", "w", "split", BASELINE]],
        on=["paper_id", "use_case_key"], how="left", validate="many_to_one")
    if resp["w"].isna().any():
        raise RuntimeError("responses not covered by the sample manifest - regenerate it")
    return resp, man


def cell_metrics(g: pd.DataFrame) -> dict:
    """One (model, variant, arm, use case) cell at population prevalence.

    `f2_star` picks its threshold on the same rows it scores - an oracle and an upper bound,
    quoted only because the cosine baseline is quoted the same way. `f2_own` is the honest
    number: the model's own verdict, nothing fitted, and the only one of the two a ranker
    cannot produce at all.
    """
    y, w = g.y.to_numpy(), g.w.to_numpy()
    s = g.score.to_numpy(dtype=float)
    keep = ~np.isnan(s)
    if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
        return {}
    y, w, s = y[keep], w[keep], s[keep]
    pred = g.pred.to_numpy(dtype=float)[keep]
    pm = ~np.isnan(pred)

    _, t_star = f2_optimal(y, s, w, THRESHOLDS)
    own = work_at_own(y[pm], pred[pm], w[pm])
    rank = wss_at(y, s, w)
    return {
        "n": int(keep.sum()),
        "auc": float(roc_auc_score(y, s)),
        "wss95": rank["wss"],
        "tie_frac": rank["tie_frac"],
        "recall@10%": recall_at(y, s, w, 0.10),
        "t_star": float(t_star),
        "f2_star": f2_at(y, s, w, t_star),
        "f2_own": own["f2"],
        "recall_own": own["recall"],
        "precision_own": own["precision"],
        "screened_own": own["screened_frac"],
        "coverage": float(keep.sum() / len(g)),
        "floor_f2": all_positive_f2(y, w),
    }


def per_use_case(resp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant, arm, uc), g in resp.groupby(["model", "variant", "arm", "use_case_key"]):
        m = cell_metrics(g)
        if m:
            rows.append({"model": model.split("/")[-1], "variant": variant, "arm": arm,
                         "use_case": uc.replace("synergy_", ""), **m})
    return pd.DataFrame(rows)


def baseline_table(man: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uc, g in man.groupby("use_case_key"):
        y, w, s = g.y.to_numpy(), g.w.to_numpy(), g[BASELINE].to_numpy()
        f2s, _ = f2_optimal(y, (s - s.min()) / (s.max() - s.min()), w, THRESHOLDS)
        rows.append({"use_case": uc.replace("synergy_", ""),
                     "auc": round(roc_auc_score(y, s), 3),
                     "wss95": round(wss_at(y, s, w)["wss"], 3),
                     "recall@10%": round(recall_at(y, s, w, 0.10), 3),
                     "f2_star": round(f2s, 3),
                     "floor_f2": round(all_positive_f2(y, w), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    resp, man = load_all()
    base = baseline_table(man).set_index("use_case")
    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Set A — LLM screening at 2.19% prevalence")
    emit()
    emit(f"8 SYNERGY collections, 62,229 papers, 1,362 relevant. Scored on the "
         f"{man.shape[0]:,}-row case-control sample and reweighted to population prevalence "
         "(`scripts/benchset_metrics.py`); the sampling gate is "
         "`reports/wf_llm_benchset_a_baselines.md`.")
    emit()
    emit("**The bar:** the cold-start cosine-to-brief ranker already on disk — mean AUC "
         f"**{base.auc.mean():.3f}**, mean WSS@95 **{base.wss95.mean():.3f}**. "
         f"Marking everything relevant scores F2 **{all_positive_f2(man.y.to_numpy(), man.w.to_numpy()):.3f}**.")
    emit()

    # ---------------- zero-shot contest, all sampled rows ----------------
    zs = resp[resp.arm == "B1 supplied"]
    puc = per_use_case(zs)
    if puc.empty:
        raise SystemExit("no B1 cells found")

    emit("## 1. Ranking — per collection, against the cosine baseline")
    emit()
    piv = puc[puc.variant == "P2"].pivot_table(index="use_case", columns="model", values="auc")
    piv.insert(0, "cosine", base.auc)
    order = base.sort_values("auc", ascending=False).index
    emit(to_md(piv.reindex(order).round(3)))
    emit()
    models = [m for m in piv.columns if m != "cosine"]
    wins = {m: int((piv[m] > piv["cosine"] + FLOOR).sum()) for m in models}
    losses = {m: int((piv[m] < piv["cosine"] - FLOOR).sum()) for m in models}
    # Wins alone are the number that flatters. A model that wins 3 and loses 3 has not
    # "won 3 collections", it has traded - and on a per-silo product where each collection
    # ships its own model, a trade is worth nothing unless you can tell in advance which
    # side of it you are on.
    emit(f"Against the 0.03 noise floor, of {len(piv)} collections:")
    emit()
    emit(to_md(pd.DataFrame({"beats cosine": wins, "loses to cosine": losses,
                             "within floor": {m: len(piv) - wins[m] - losses[m]
                                              for m in models}})))
    emit()

    emit("## 2. Headline, P2, all 9,993 sampled rows")
    emit()
    head = puc[puc.variant == "P2"].groupby("model").agg(
        mean_auc=("auc", "mean"), mean_wss95=("wss95", "mean"),
        mean_f2_star=("f2_star", "mean"), mean_f2_own=("f2_own", "mean"),
        recall_own=("recall_own", "mean"), screened_own=("screened_own", "mean"),
        tie_frac=("tie_frac", "mean"), coverage=("coverage", "mean"))
    head["beats_cosine"] = [wins.get(m, 0) for m in head.index]
    head.loc["cosine baseline"] = {
        "mean_auc": base.auc.mean(), "mean_wss95": base.wss95.mean(),
        "mean_f2_star": base.f2_star.mean(), "mean_f2_own": np.nan,
        "recall_own": np.nan, "screened_own": np.nan, "tie_frac": 0.0,
        "coverage": 1.0, "beats_cosine": np.nan}
    emit(to_md(head.round(3)))
    emit()
    emit("`f2_star` is an **oracle** on both sides (91 thresholds swept on the rows scored). "
         "`f2_own` is the model's own verdict with nothing tuned — the cosine has no "
         "equivalent, which is the point: a ranker cannot decide, only order. "
         "`screened_own` is the fraction of the corpus a reviewer would read at that verdict.")
    emit()
    emit("`mean_auc` is dominated by `brouwer_2019`, which is 60% of set A and which the "
         "cosine already scores 0.99 on. Read the win/loss table above it, not the mean.")
    emit()

    # ---------------- where does the LLM's advantage live? ----------------
    emit("### Where the advantage lives")
    emit()
    emit("The pilot's rule was **diversity only pays where the dissenting branch is "
         "competent** — the LLM helped where it was strong, not where it was different. "
         "The same question here is whether the LLM covers the cosine's weak collections or "
         "merely re-wins its strong ones.")
    emit()
    delta = pd.DataFrame({m: piv[m] - piv["cosine"] for m in models})
    delta.insert(0, "cosine_auc", piv["cosine"])
    emit(to_md(delta.round(3)))
    emit()
    corrs = {m: float(np.corrcoef(piv["cosine"], delta[m])[0, 1]) for m in models}
    emit("Correlation between the cosine's own AUC and the LLM's gain over it: "
         + " · ".join(f"**{k}** {v:+.2f}" for k, v in corrs.items())
         + ". A strong negative means the LLM picks up exactly where the cheap ranker "
           "gives out — which is the complementarity a second rung would need.")
    emit()

    # ---------------- P1 vs P2 at low prevalence ----------------
    both = puc[puc.model.isin(puc[puc.variant == "P1"].model.unique())]
    if not both.empty and both.variant.nunique() > 1:
        emit("## 3. Does \"when uncertain, include\" survive the prevalence drop?")
        emit()
        emit("The pilot's most transferable finding was that stating the F2 asymmetry moved "
             "F2@own by +0.31 to +0.37 — at 26–77% prevalence, where including a doubtful "
             "paper is nearly free. At 2.19% each marginal inclusion costs ~45 false "
             "positives per true one. This is that instruction re-priced.")
        emit()
        pv = both.pivot_table(index=["model", "use_case"], columns="variant",
                              values=["f2_own", "recall_own", "screened_own", "auc"])
        agg = pd.DataFrame({
            "f2_own_P1": pv[("f2_own", "P1")], "f2_own_P2": pv[("f2_own", "P2")],
            "recall_P1": pv[("recall_own", "P1")], "recall_P2": pv[("recall_own", "P2")],
            "screened_P1": pv[("screened_own", "P1")], "screened_P2": pv[("screened_own", "P2")],
            "auc_P1": pv[("auc", "P1")], "auc_P2": pv[("auc", "P2")],
        }).groupby("model").mean()
        agg["d_f2_own"] = agg.f2_own_P2 - agg.f2_own_P1
        agg["d_auc"] = agg.auc_P2 - agg.auc_P1
        emit(to_md(agg.round(3)))
        emit()

    # ---------------- brief ladder, unseen rows only ----------------
    ladder_arms = resp.arm.unique()
    if len(ladder_arms) > 1:
        emit("## 4. The brief-format ladder")
        emit()
        emit("Same prompt (P2), four brief sets. Scored on **test + validate only**, because "
             "B2 was induced from train labels and every arm must sit on identical unseen "
             "rows. B0 strips the LLM-written term lists back to the review's own title and "
             "abstract — the format `CONTEXT.md` §4 says made the shuffled-brief control "
             "fail on old SYNERGY.")
        emit()
        held = resp[(resp.split != "train") & (resp.variant == "P2")]
        lp = per_use_case(held)
        lad = lp.groupby(["model", "arm"]).agg(
            mean_auc=("auc", "mean"), mean_f2_own=("f2_own", "mean"),
            recall_own=("recall_own", "mean"), screened_own=("screened_own", "mean"),
            mean_wss95=("wss95", "mean"), coverage=("coverage", "mean"))
        emit(to_md(lad.round(3)))
        emit()

        # Per-collection is the only honest read of the ladder: the two biggest effects here
        # are collections where the supplied brief drove the model to reject almost
        # everything, and a mean hides that completely.
        for m in sorted(lp.model.unique()):
            g = lp[lp.model == m]
            if g.arm.nunique() < 3:
                continue
            emit(f"#### `{m}` per collection")
            emit()
            for metric, label in (("auc", "ROC-AUC"), ("recall_own", "recall at own verdict"),
                                  ("screened_own", "fraction of corpus read")):
                p = g.pivot_table(index="use_case", columns="arm", values=metric)
                p = p.reindex(order)
                if metric == "auc":
                    p.insert(0, "cosine", base.auc)
                emit(f"**{label}**")
                emit()
                emit(to_md(p.round(3)))
                emit()
            arms_present = [a for a in ("B0 raw", "B1 supplied", "B2 induced") if a in g.arm.values]
            pa = g.pivot_table(index="use_case", columns="arm", values="auc")
            for a in arms_present:
                w = int((pa[a] > base.auc.reindex(pa.index) + FLOOR).sum())
                lo = int((pa[a] < base.auc.reindex(pa.index) - FLOOR).sum())
                emit(f"- **{a}** vs cosine: {w} wins, {lo} losses, "
                     f"{len(pa) - w - lo} inside the floor")
            emit()

        if "B✗ shuffled" in ladder_arms:
            sh = lad.xs("B✗ shuffled", level="arm", drop_level=False)
            emit(f"**Falsification check.** The shuffled arm must collapse: mean AUC "
                 f"**{sh.mean_auc.mean():.3f}** (0.5 = no ranking), predicted-positive rate "
                 f"**{sh.screened_own.mean():.3f}**. If that number is not near chance, every "
                 "other result on this page is measuring generic paper quality and should be "
                 "discarded.")
            emit()

    # ---------------- stop conditions ----------------
    emit("## 5. Pre-registered bar")
    emit()
    best = head.drop(index="cosine baseline").mean_auc.max()
    best_model = head.drop(index="cosine baseline").mean_auc.idxmax()
    n_win = int(head.drop(index="cosine baseline").beats_cosine.max())
    rows = [
        {"condition": f"Zero-shot replaces the cold-start rung: beat cosine "
                      f"{base.auc.mean():.3f} by >0.03 on ≥6 of 8 collections",
         "result": f"best is {best_model} at {best:.3f}, winning {n_win}/8",
         "verdict": "PASS" if n_win >= 6 else "FAIL"},
        {"condition": "Stop early if best mean AUC < 0.70",
         "result": f"{best:.3f}",
         "verdict": "continue" if best >= 0.70 else "STOP"},
    ]
    # The bar was written before the ladder existed and asks about a *zero-shot* prompt, so
    # the induced arm is scored against it separately rather than quietly folded in. It is a
    # fitted configuration and has to be judged as one.
    if "B2 induced" in resp.arm.values:
        held2 = resp[(resp.split != "train") & (resp.variant == "P2")]
        l2 = per_use_case(held2)
        for m in sorted(l2[l2.arm == "B2 induced"].model.unique()):
            pa = l2[l2.model == m].pivot_table(index="use_case", columns="arm", values="auc")
            if "B2 induced" not in pa:
                continue
            bcos = base.auc.reindex(pa.index)
            w = int((pa["B2 induced"] > bcos + FLOOR).sum())
            lo = int((pa["B2 induced"] < bcos - FLOOR).sum())
            rows.append({
                "condition": f"Induced brief (B2, fitted on train) replaces the rung — `{m}`",
                "result": f"mean AUC {pa['B2 induced'].mean():.3f}, {w} wins / {lo} losses of "
                          f"{len(pa)}",
                "verdict": "PASS" if w >= 6 else "FAIL"})
    emit(to_md(pd.DataFrame(rows).set_index("condition")))
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    puc.to_csv(REPORTS / "wf_llm_benchset_a_per_use_case.csv", index=False)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
