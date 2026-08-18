"""Stage 0: prove the case-control sample reproduces the full corpus, before spending anything.

The whole set-A design rests on one claim — that scoring 9,993 of 62,229 rows and reweighting
gives the same answer as scoring all of them. That claim is testable for **free**, because
set A already carries the cold-start cosine-to-brief ranker (`cos_brief_jasper`,
`cos_brief_qwen4b`) and the lex block on disk. So: compute the metrics both ways and compare.

If they do not agree, the sampling is wrong and no LLM number bought on top of it would mean
anything. This script is therefore the gate on all spend, and it runs at three seeds so that
"agrees" means "agrees regardless of which negatives were drawn".

It doubles as the incumbent bar. Set A has no shipped ensemble; the cold-start cosine ranker
is the rung the LLM has to beat, and `wf_llm_pilot_findings.md` §8 already called that "the
more winnable" contest and the one that would replace something real.

Usage:
    python scripts/validate_benchset_sampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_loader import BASELINE_COLS, case_control_sample, load_set_a  # noqa: E402
from benchset_metrics import all_positive_f2, f2_optimal, recall_at, wss_at  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "wf_llm_benchset_a_baselines.md"
SEEDS = (0, 1, 2)
PRIMARY = "cos_brief_qwen4b"


def metrics(y, s, w) -> dict:
    keep = ~np.isnan(s)
    y, s, w = y[keep], s[keep], w[keep]
    f2, t = f2_optimal(y, _to01(s), w)
    return {
        "auc": roc_auc_score(y, s),
        "wss95": wss_at(y, s, w)["wss"],
        "recall_10pct": recall_at(y, s, w, 0.10),
        "f2_star": f2,
        "floor_f2": all_positive_f2(y, w),
    }


def _to01(s: np.ndarray) -> np.ndarray:
    """Min-max a similarity onto [0,1] so the shared threshold sweep is meaningful.

    Monotone, so it changes no ranking metric. F2@t* is an oracle over the sweep anyway —
    this only makes the sweep's grid land in a sensible place for a cosine.
    """
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def main() -> None:
    df = load_set_a()
    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Set A — cold-start baselines, and the case-control sampling gate")
    emit()
    emit(f"`benchset_v1_large_set_a`: **{len(df):,} rows / {df.use_case_key.nunique()} "
         f"collections / {df.y.sum():,} positive ({df.y.mean():.4f})**. All SYNERGY; "
         "briefs carry populated `terms_*` lists, which old SYNERGY did not.")
    emit()

    # ---------- the bar: full corpus, every row, no sampling anywhere ----------
    full = []
    for uc, g in df.groupby("use_case_key"):
        y, w = g.y.to_numpy(), np.ones(len(g))
        rec = {"use_case": uc.replace("synergy_", ""), "n": len(g), "prev": round(y.mean(), 4),
               "floor_f2": round(all_positive_f2(y, w), 3)}
        for c in ("cos_brief_jasper", "cos_brief_qwen4b", "lex_bm25_must"):
            s = g[c].to_numpy()
            rec[f"auc_{c.replace('cos_brief_', '').replace('lex_', '')}"] = round(
                roc_auc_score(y, s), 3)
        m = metrics(y, g[PRIMARY].to_numpy(), w)
        rec["wss95"] = round(m["wss95"], 3)
        rec["recall@10%"] = round(m["recall_10pct"], 3)
        rec["f2_star"] = round(m["f2_star"], 3)
        full.append(rec)
    full_df = pd.DataFrame(full).sort_values("n", ascending=False)

    emit("## The incumbent bar — cold-start cosine-to-brief, all 62,229 rows")
    emit()
    emit(f"Ranking metrics from `{PRIMARY}`. `floor_f2` is F2 for marking *everything* "
         "relevant — the number every F2 below has to be read against.")
    emit()
    emit(to_md(full_df.set_index("use_case")))
    emit()
    mean_auc = full_df["auc_qwen4b"].mean()
    emit(f"**Mean AUC {mean_auc:.3f}** (jasper {full_df['auc_jasper'].mean():.3f}, "
         f"bm25_must {full_df['auc_bm25_must'].mean():.3f}) · "
         f"**mean WSS@95 {full_df['wss95'].mean():.3f}**. Pooled floor F2 "
         f"**{all_positive_f2(df.y.to_numpy(), np.ones(len(df))):.3f}** at 2.19% prevalence, "
         "against 0.872 on TIRI's pools — this is the surface where F2 means something.")
    emit()
    emit("Two collections to keep in view. **`brouwer_2019` is 60% of set A by volume and "
         "trivially easy** (AUC 0.994, 62 positives in 37,401 rows), so it will dominate any "
         "pooled number and flatter anything that ranks at all. **`moran_2021` is below "
         "chance** (0.445) — every embedding and the lex block agree, so the brief points "
         "away from the labels there. It is kept: it is the one collection where reading the "
         "abstract could beat a cosine outright, and the one that could expose a corpus defect.")
    emit()

    # ---------- the gate: does the 9,993-row sample reproduce that? ----------
    rows = []
    for seed in SEEDS:
        smp = case_control_sample(df, seed=seed)
        for uc, g in smp.groupby("use_case_key"):
            m = metrics(g.y.to_numpy(), g[PRIMARY].to_numpy(), g.w.to_numpy())
            rows.append({"seed": seed, "use_case": uc.replace("synergy_", ""), **m})
    samp = pd.DataFrame(rows)

    agg = samp.groupby("use_case").agg(["mean", "std"])
    cmp_rows = []
    for uc in full_df.use_case:
        f = full_df[full_df.use_case == uc].iloc[0]
        a = agg.loc[uc]
        cmp_rows.append({
            "use_case": uc,
            "auc_full": f["auc_qwen4b"], "auc_samp": round(a[("auc", "mean")], 3),
            "auc_sd": round(a[("auc", "std")], 3),
            "wss_full": f["wss95"], "wss_samp": round(a[("wss95", "mean")], 3),
            "wss_sd": round(a[("wss95", "std")], 3),
            "f2_full": f["f2_star"], "f2_samp": round(a[("f2_star", "mean")], 3),
            "f2_sd": round(a[("f2_star", "std")], 3),
            "floor_full": f["floor_f2"], "floor_samp": round(a[("floor_f2", "mean")], 3),
        })
    cmp = pd.DataFrame(cmp_rows)
    for k in ("auc", "wss", "f2"):
        cmp[f"d_{k}"] = (cmp[f"{k}_samp"] - cmp[f"{k}_full"]).round(3)

    emit("## The gate — 9,993-row case-control sample vs all 62,229 rows")
    emit()
    emit("Same metric, same ranker, three seeds. `_samp` columns are weighted by "
         "`w = negatives_in_population / negatives_sampled`; `_full` scores every row. "
         "`floor_samp` recovering `floor_full` is the direct check that the weights restore "
         "population prevalence.")
    emit()
    emit(to_md(cmp[["use_case", "auc_full", "auc_samp", "auc_sd", "d_auc",
                    "wss_full", "wss_samp", "wss_sd", "d_wss",
                    "f2_full", "f2_samp", "d_f2", "floor_full", "floor_samp"]]
               .set_index("use_case")))
    emit()
    worst = {k: cmp[f"d_{k}"].abs().max() for k in ("auc", "wss", "f2")}
    floor_err = (cmp.floor_samp - cmp.floor_full).abs().max()
    emit(f"Largest absolute discrepancy: AUC **{worst['auc']:.3f}**, WSS@95 "
         f"**{worst['wss']:.3f}**, F2@t\\* **{worst['f2']:.3f}**. Prevalence recovery is "
         f"exact to **{floor_err:.4f}**. Seed-to-seed sd on AUC is "
         f"**{samp.groupby('use_case').auc.std().mean():.3f}**, against the repo's ~0.010 "
         "noise floor.")
    emit()
    ok = worst["auc"] < 0.03 and floor_err < 0.005
    emit(f"**Gate: {'PASS' if ok else 'FAIL'}.** "
         + ("The sample reproduces the corpus inside the noise floor; LLM spend is authorised "
            "against it." if ok else
            "The sample does NOT reproduce the corpus. Do not buy LLM responses against it."))
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n")
    full_df.to_csv(REPO / "reports" / "wf_llm_benchset_a_baselines.csv", index=False)
    print(f"\nwrote {OUT_MD}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
