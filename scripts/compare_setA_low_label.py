"""The comparison B2 actually has to win: a supervised model on the same 60 labels.

Why this exists
---------------
The induced rule set (B2 / "P5") is not zero-shot. It was distilled from **30 positive + 30
negative train rows per collection** — 60 labels. `CONTEXT.md` §1's operating rule is
explicit about what that number buys:

    cosine-to-brief below ~25 in-silo labels, supervised in-silo model above.

Sixty labels is above the line. So measuring B2 against the *cold-start* cosine flatters it:
the cosine is the rung for a collection with **no** labels, and B2 is not that situation. If
you have 60 labels the repo's own rule says fit a model, and the honest question is whether
spending them on a brief beats spending them on a classifier.

This script fits that classifier on **exactly the rows the inducer saw** — same collections,
same train split, same 30+30 draw with the same seed — and scores it on the same
test+validate rows the ladder is reported on. Features are already on disk in
`benchset_v1_large_set_a.parquet`, so this costs nothing and refits nothing upstream.

Two branches, both cheap and both defensible at n=60:
  - **LogReg on the query-conditioned block** (`cos_*`, `rank_*`, `lex_*`) — 21ish features,
    the arm `CONTEXT.md` §1 has the most evidence for at low label counts.
  - **LogReg on the Qwen3-4B embedding** (2,560-d, heavily regularised) — the representation
    the cosine baseline itself is built from, given a supervised read.

Both get the standard treatment: median-impute + standardise inside the fold (NULL is not 0,
but a linear model cannot take a NaN, so imputation is declared here and an indicator is not
worth it at n=60), `class_weight="balanced"` because 2.19% prevalence otherwise collapses the
decision to all-negative.

Usage:
    python scripts/compare_setA_low_label.py
"""

from __future__ import annotations

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
from benchset_loader import SET_A, case_control_sample, load_set_a  # noqa: E402
from benchset_metrics import all_positive_f2, f2_optimal, wss_at  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "wf_llm_benchset_a_low_label.md"
K_PER_CLASS = 30      # must match induce_rule_set.K_PER_CLASS
INDUCE_SEED = 0       # must match the `random_state` in induce_rule_set._examples
THRESHOLDS = np.linspace(0.01, 0.99, 99)

QC_PREFIXES = ("cos_", "rank_", "lex_")


def _emb_cols() -> list[str]:
    names = pq.ParquetFile(SET_A).schema_arrow.names
    return [n for n in names if n.startswith("emb_qwen4b_")] or \
           [n for n in names if n.startswith("emb_jasper_")]


def _fit_score(Xtr, ytr, Xte, C: float):
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1], pipe.predict(Xte)


def main() -> None:
    base = load_set_a()
    sample = case_control_sample(base)

    qc = [c for c in base.columns if c.startswith(QC_PREFIXES)]
    emb = _emb_cols()
    print(f"query-conditioned block: {len(qc)} features · embedding block: {len(emb)}")
    embs = pd.read_parquet(SET_A, columns=["row_key", "use_case_key", *emb])
    frame = sample.merge(embs, on=["row_key", "use_case_key"], how="left", validate="one_to_one")

    rows = []
    for uc, g in frame.groupby("use_case_key", sort=True):
        tr_pool = g[g.split == "train"]
        # The identical draw the inducer saw: same k, same seed, positives then negatives.
        tr = pd.concat([
            tr_pool[tr_pool.y == 1].sample(n=min(K_PER_CLASS, int((tr_pool.y == 1).sum())),
                                           random_state=INDUCE_SEED),
            tr_pool[tr_pool.y == 0].sample(n=min(K_PER_CLASS, int((tr_pool.y == 0).sum())),
                                           random_state=INDUCE_SEED),
        ])
        te = g[g.split != "train"]
        y_te, w_te = te.y.to_numpy(), te.w.to_numpy()
        rec = {"use_case": uc.replace("synergy_", ""), "n_train": len(tr),
               "n_eval": len(te), "floor_f2": all_positive_f2(y_te, w_te)}

        for name, cols, C in (("qc_block", qc, 1.0), ("embedding", emb, 0.01)):
            s, pred = _fit_score(tr[cols].to_numpy(float), tr.y.to_numpy(),
                                 te[cols].to_numpy(float), C)
            f2s, _ = f2_optimal(y_te, s, w_te, THRESHOLDS)
            from benchset_metrics import work_at_own
            own = work_at_own(y_te, pred, w_te)
            rec[f"auc_{name}"] = roc_auc_score(y_te, s)
            rec[f"f2star_{name}"] = f2s
            rec[f"f2own_{name}"] = own["f2"]
            rec[f"recall_{name}"] = own["recall"]
            rec[f"screened_{name}"] = own["screened_frac"]
            rec[f"wss_{name}"] = wss_at(y_te, s, w_te)["wss"]
        # the no-label rung, on the same eval rows
        s0 = te.cos_brief_qwen4b.to_numpy()
        rec["auc_cosine"] = roc_auc_score(y_te, s0)
        rec["wss_cosine"] = wss_at(y_te, s0, w_te)["wss"]
        rows.append(rec)

    res = pd.DataFrame(rows).set_index("use_case")
    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Set A — what 60 labels per collection buy, spent three ways")
    emit()
    emit("The induced rule set (B2) saw **30 positive + 30 negative train rows per "
         "collection**. `CONTEXT.md` §1's rule — *cosine-to-brief below ~25 in-silo labels, "
         "supervised in-silo model above* — puts that above the line, so the cold-start "
         "cosine is **not** B2's fair opponent. This is the fair one: a supervised model on "
         "exactly the same 60 rows, scored on exactly the same held-out rows.")
    emit()
    emit(to_md(res[["n_train", "n_eval", "auc_cosine", "auc_qc_block", "auc_embedding",
                    "wss_cosine", "wss_qc_block", "wss_embedding"]].round(3)))
    emit()
    emit(to_md(res[["floor_f2", "f2own_qc_block", "recall_qc_block", "screened_qc_block",
                    "f2own_embedding", "recall_embedding", "screened_embedding"]].round(3)))
    emit()
    means = res.mean(numeric_only=True)
    emit(f"**Means** — AUC: cosine (no labels) **{means.auc_cosine:.3f}** · "
         f"LogReg on the query-conditioned block **{means.auc_qc_block:.3f}** · "
         f"LogReg on the Qwen3-4B embedding **{means.auc_embedding:.3f}**. "
         f"F2@own: qc **{means.f2own_qc_block:.3f}** · embedding "
         f"**{means.f2own_embedding:.3f}**.")
    emit()
    emit("Read alongside `reports/wf_llm_benchset_a.md` §4, which reports the induced-brief "
         "arm on the same held-out rows. Whichever wins, the comparison is now the right one: "
         "three ways to spend the same 60 labels, not a fitted method against an unfitted one.")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n")
    res.to_csv(REPO / "reports" / "wf_llm_benchset_a_low_label.csv")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
