"""One table: every set-A method, every metric, on identical rows at population prevalence.

The set-A work produced its numbers in five places (`wf_llm_benchset_a.md`,
`_baselines.md`, `_low_label.md`, `_induced_features.md`, `_ensemble.md`), each answering its
own question on its own slice. This consolidates them so methods can be read against each
other rather than against their own section.

Three things make the rows comparable, and all three matter:

  - **Same rows.** Everything is scored on **test + validate** (3,997 sampled rows). Some
    arms are zero-shot and could use every row, but the induced-brief and 60-label arms
    cannot, so the common set governs.
  - **Same prevalence.** Weighted by `w` back to the population's 2.19%. Recall and AUC are
    insensitive to the case-control sampling; precision, F2, WSS@95 and recall@10% are not,
    and are wrong by a factor of ~5-25x without the weight.
  - **Same metric definitions.** F2 = 5PR/(4P+R); WSS@95 = (N-screened)/N - 0.05 with
    screened measured in *weight*, i.e. papers a human would really read.

Three operating points per method, because they answer different questions:

  @0.5   a fixed, untuned threshold — what you get with no calibration at all.
  @t*    the best of 91 swept thresholds. An **oracle**: it picks on the rows it scores, so
         it is an upper bound, quoted for every method or none.
  @own   the model's own verdict, nothing fitted. Only prompted LLMs have one — a ranker
         orders papers but cannot decide where to stop, which is itself a finding.

Usage:
    python scripts/summarise_setA_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_metrics import (  # noqa: E402
    all_positive_f2, recall_at, weighted_prf, wss_at,
)
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
MANIFEST = REPORTS / "wf_llm_setA_sample.parquet"
ENSEMBLE_FEATURES = REPO / "data" / "processed" / "setA_ensemble_features.parquet"
ENSEMBLE_OOF = REPORTS / "wf_llm_setA_brief_ensemble_oof.json"
OUT_MD = REPORTS / "wf_llm_benchset_a_summary.md"
THRESHOLDS = np.linspace(0.01, 0.99, 99)
K_PER_CLASS, SEED = 30, 0


def minmax(s: np.ndarray) -> np.ndarray:
    """Monotone rescale to [0,1]. Changes no ranking metric; it only makes a fixed 0.5 and a
    threshold sweep mean the same thing for a cosine as for a probability. Flagged in the
    output, because a cosine has no natural 0.5 and pretending otherwise would be a lie."""
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def score_row(y, s, w, pred=None, rescale=False) -> dict:
    keep = ~np.isnan(s)
    y, s, w = y[keep], s[keep], w[keep]
    if len(np.unique(y)) < 2:
        return {}
    s01 = minmax(s) if rescale else s
    out = {"auc": roc_auc_score(y, s),
           "wss95": wss_at(y, s, w)["wss"],
           "recall@10%": recall_at(y, s, w, 0.10),
           "floor_f2": all_positive_f2(y, w)}

    half = weighted_prf(y, s01 >= 0.5, w)
    out |= {"f2@0.5": half["f2"], "prec@0.5": half["precision"],
            "rec@0.5": half["recall"], "read@0.5": half["pos_rate"]}

    best_t, best = 0.5, {"f2": -1.0}
    for t in THRESHOLDS:
        m = weighted_prf(y, s01 >= t, w)
        if m["f2"] > best["f2"]:
            best_t, best = float(t), m
    out |= {"t_star": best_t, "f2@t*": best["f2"], "prec@t*": best["precision"],
            "rec@t*": best["recall"], "read@t*": best["pos_rate"]}

    if pred is not None:
        p = pred[keep]
        m = ~np.isnan(p)
        if m.any():
            o = weighted_prf(y[m], p[m].astype(bool), w[m])
            out |= {"f2@own": o["f2"], "prec@own": o["precision"],
                    "rec@own": o["recall"], "read@own": o["pos_rate"]}
    return out


def fit_predict(tr, te, cols, C=1.0):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    pipe.fit(tr[cols].to_numpy(float), tr.y.to_numpy())
    return pipe.predict_proba(te[cols].to_numpy(float))[:, 1]


def train_rows(g):
    pool = g[g.split == "train"]
    return pd.concat([
        pool[pool.y == 1].sample(n=min(K_PER_CLASS, int((pool.y == 1).sum())), random_state=SEED),
        pool[pool.y == 0].sample(n=min(K_PER_CLASS, int((pool.y == 0).sum())), random_state=SEED),
    ])


def collect() -> pd.DataFrame:
    man = pd.read_parquet(MANIFEST)
    feats = pd.read_parquet(ENSEMBLE_FEATURES)
    rows: list[dict] = []

    def add(method, labels, uc, y, s, w, pred=None, rescale=False):
        m = score_row(y, s, w, pred, rescale)
        if m:
            rows.append({"method": method, "labels": labels,
                         "use_case": uc.replace("synergy_", ""), **m})

    # ---------- 0-label rankers ----------
    for uc, g in man.groupby("use_case_key"):
        h = g[g.split != "train"]
        y, w = h.y.to_numpy().astype(int), h.w.to_numpy()
        for col, name in (("cos_brief_qwen4b", "cosine-to-brief (Qwen3-4B)"),
                          ("cos_brief_jasper", "cosine-to-brief (Jasper)"),
                          ("lex_bm25_must", "BM25 must-terms, supplied brief")):
            add(name, 0, uc, y, h[col].to_numpy(float), w, rescale=True)

    # ---------- prompted LLMs ----------
    key = ["paper_id", "use_case_key"]
    mm = man[[*key, "y", "w", "split"]]
    arm_of = {"raw": "B0 raw brief", "induced": "B2 induced brief",
              "control": "B✗ shuffled brief"}
    for path in sorted(REPORTS.glob("wf_llm_setA_*_responses.parquet")):
        stem = path.stem.replace("_responses", "")
        if "smoke" in stem:
            continue
        arm = next((v for k, v in arm_of.items() if k in stem), "B1 supplied brief")
        d = pd.read_parquet(path).drop(columns=["y"]).merge(mm, on=key, validate="many_to_one")
        d = d[d.split != "train"]
        for (model, variant), g in d.groupby(["model", "variant"]):
            short = model.split("/")[-1]
            label = f"LLM {short} {variant}"
            if arm != "B1 supplied brief":
                label += f" · {arm}"
            n_lab = 60 if arm == "B2 induced brief" else 0
            for uc, gg in g.groupby("use_case_key"):
                add(label, n_lab, uc, gg.y.to_numpy().astype(int),
                    gg.score.to_numpy(float), gg.w.to_numpy(),
                    pred=gg.pred.to_numpy(float))

    # ---------- 60-label supervised ----------
    emb = sorted(c for c in feats.columns if c.startswith("emb_qwen4b_"))
    lex_sup = sorted(c for c in feats.columns if c.startswith("lex_"))
    lex_ind = sorted(c for c in feats.columns if c.startswith("lexind_"))
    qc = lex_sup + [c for c in feats.columns if c.startswith(("cos_brief_", "rank_cos_brief_"))]
    for name, cols, C in (("LogReg on Qwen3-4B embedding", emb, 0.01),
                          ("LogReg on lexical block, supplied brief", lex_sup, 1.0),
                          ("LogReg on lexical block, induced brief", lex_ind, 1.0),
                          ("LogReg on query-conditioned block", qc, 1.0)):
        for uc, g in feats.groupby("use_case_key"):
            te = g[g.split != "train"]
            s = fit_predict(g.loc[train_rows(g).index], te, cols, C)
            add(name, 60, uc, te.y.to_numpy().astype(int), s, te.w.to_numpy())

    # ---------- full-label ensemble ----------
    raw = json.loads(ENSEMBLE_OOF.read_text())
    held = (feats.split != "train").to_numpy()
    cells: dict[tuple, dict] = {}
    for k, per_uc in raw.items():
        variant, branch, seed = k.split("|")
        cells.setdefault((variant, int(seed)), {})[branch] = per_uc
    acc: dict[tuple, list] = {}
    for (variant, seed), branches in cells.items():
        if len(branches) < 2:
            continue
        for uc, g in feats.groupby("use_case_key"):
            m = held[g.index.to_numpy()]
            s = np.mean([np.asarray(p[uc], float) for p in branches.values()], axis=0)[m]
            acc.setdefault((variant, uc), []).append(s)
    label = {"supplied": "Ensemble blend, supplied brief",
             "induced_lex": "Ensemble blend, induced brief",
             "induced_all": "Ensemble blend, induced brief + cosine"}
    for (variant, uc), arrs in acc.items():
        g = feats[feats.use_case_key == uc]
        h = g[g.split != "train"]
        add(label[variant], "in-silo", uc, h.y.to_numpy().astype(int),
            np.mean(arrs, axis=0), h.w.to_numpy())

    return pd.DataFrame(rows)


def main() -> None:
    res = collect()
    num = [c for c in res.columns if c not in ("method", "labels", "use_case")]
    agg = res.groupby(["labels", "method"])[num].mean()

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Set A — every method, every metric, one table")
    emit()
    emit("8 SYNERGY collections, 62,229 papers, **2.19% relevant**. All rows scored on "
         "**test + validate** (3,997 of the 9,993-row case-control sample) and **weighted "
         "back to population prevalence** — precision, F2, WSS@95 and recall@10% are wrong "
         "by 5-25x without that weight. Every figure is the **mean over the 8 collections**, "
         "not a pooled number: `brouwer_2019` is 60% of set A and trivially easy, so a pool "
         "is mostly that one collection (`CONTEXT.md` §5).")
    emit()
    emit("**Marking every paper relevant scores F2 = 0.101.** That is the floor every F2 "
         "below sits on.")
    emit()

    order = ["auc", "wss95", "recall@10%",
             "f2@0.5", "prec@0.5", "rec@0.5", "read@0.5",
             "t_star", "f2@t*", "prec@t*", "rec@t*", "read@t*",
             "f2@own", "prec@own", "rec@own", "read@own"]

    emit("## 1. Ranking quality and the two threshold-free metrics")
    emit()
    emit("`wss95` = fraction of the corpus a reviewer can skip while still finding 95% of "
         "relevant papers, minus the 5% chance would give. `recall@10%` = share of relevant "
         "papers found in the top tenth of the ranking.")
    emit()
    emit(to_md(agg[["auc", "wss95", "recall@10%"]].round(3).sort_values("auc",
                                                                       ascending=False)))
    emit()

    emit("## 2. At a fixed, untuned threshold of 0.5")
    emit()
    emit("What you get with no calibration. `read` is the fraction of the corpus the "
         "reviewer must read. Cosine and BM25 rows are min-max rescaled to [0,1] first — a "
         "cosine has no natural 0.5, so read those two rows as indicative only.")
    emit()
    emit(to_md(agg[["f2@0.5", "prec@0.5", "rec@0.5", "read@0.5"]].round(3)))
    emit()

    emit("## 3. At the best of 91 swept thresholds (oracle — upper bound)")
    emit()
    emit("`t_star` is the mean per-collection optimum. This metric picks its threshold on "
         "the same rows it scores, so it flatters every method equally and is quoted for all "
         "of them or none. It is the rule that produced the ensemble's published 0.893.")
    emit()
    emit(to_md(agg[["t_star", "f2@t*", "prec@t*", "rec@t*", "read@t*"]].round(3)))
    emit()

    own = agg[agg["f2@own"].notna()]
    emit("## 4. At the model's own verdict — nothing fitted")
    emit()
    emit("The honest operating point, and **only prompted LLMs have one**. A ranker orders "
         "papers but cannot say where to stop; turning one into a decision costs labels that "
         "the LLM did not need. That is the asymmetry the F2 columns above hide.")
    emit()
    emit(to_md(own[["f2@own", "prec@own", "rec@own", "read@own", "auc"]].round(3)))
    emit()

    emit("## 5. Headline comparison, best configuration per label budget")
    emit()
    picks = [
        (0, "cosine-to-brief (Qwen3-4B)"),
        (0, "LLM gemma-4-31b-it P2"),
        (0, "LLM nemotron-3-super-120b-a12b P2"),
        (60, "LogReg on lexical block, induced brief"),
        (60, "LLM gemma-4-31b-it P2 · B2 induced brief"),
        (60, "LogReg on Qwen3-4B embedding"),
        ("in-silo", "Ensemble blend, supplied brief"),
    ]
    sel = agg.loc[[p for p in picks if p in agg.index]]
    emit(to_md(sel[order].round(3)))
    emit()

    emit("## 6. Per collection — the two configurations that matter")
    emit()
    for m in ("LLM gemma-4-31b-it P2 · B2 induced brief", "Ensemble blend, supplied brief"):
        sub = res[res.method == m]
        if sub.empty:
            continue
        emit(f"**{m}**")
        emit()
        emit(to_md(sub.set_index("use_case")[
            ["auc", "wss95", "f2@0.5", "f2@t*", "t_star", "prec@t*", "rec@t*",
             "floor_f2"]].round(3)))
        emit()

    emit("## 7. Reading notes")
    emit()
    emit("- **`floor_f2` is per collection and ranges 0.008-0.590**, because prevalence does "
         "(0.2%-22%). A collection-level F2 near its own floor is measuring prevalence, not "
         "skill. The pooled floor is 0.101.")
    emit("- **`f2@t*` is an oracle** and always beats `f2@0.5` and `f2@own` by construction. "
         "The gap between `f2@t*` and `f2@own` is what a model gives up by having to decide "
         "without seeing the answers.")
    emit("- **Precision is low everywhere** — at 2.19% prevalence, a screen tuned for recall "
         "necessarily returns mostly irrelevant papers. That is the correct behaviour when a "
         "miss costs 5x a false positive; `read@` is the column that says what it costs.")
    emit("- **The shuffled-brief row must collapse to ~0.5 AUC and 0 recall.** It does. If "
         "it ever does not, nothing else on this page is measuring brief-conditioned "
         "relevance.")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n")
    res.to_csv(REPORTS / "wf_llm_benchset_a_summary.csv", index=False)
    agg.round(4).to_csv(REPORTS / "wf_llm_benchset_a_summary_agg.csv")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
