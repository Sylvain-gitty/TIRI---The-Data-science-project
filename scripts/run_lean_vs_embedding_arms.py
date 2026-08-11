"""Does the ensemble need the raw embeddings, or do cos_brief + lexical carry it?

THE QUESTION
------------
`cos_brief` is a 2560-or-4096-dimension embedding compressed to one scalar: "how close is
this paper to the brief". Keeping it while dropping the raw vectors would shrink the
ensemble's input from ~8,700 columns to ~35. That is very attractive for a per-silo model
fitted on 260-360 rows, for determinism, and for cost. The question is what it costs in
recall.

Two sub-questions, and only the second one decides the ensemble design:

1. How much worse is a no-embedding ("lean") arm standing alone?
2. Are the lean arm's errors *decorrelated* from the embedding arm's? A weaker arm still
   earns an ensemble slot if it is wrong about different papers -- that is the entire
   mechanism by which ensembling helps recall. `reports/wf_ensemble_report.md` §3 proposed
   an embedding-free base learner on exactly this reasoning but never measured the
   correlation, and its proposed feature set (the metadata punch list) has since been
   rejected. cos_brief + lexical is a much better candidate for that slot.

Scored on WSS@95 within-silo, because siloed per-customer models are what ships.

Usage:
    python scripts/run_lean_vs_embedding_arms.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexical_features import build_lexical_features  # noqa: E402
from run_embedding_recall_comparison import CACHE, MODELS, metrics  # noqa: E402
from run_tier1b_control import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "wf_lean_vs_embedding_arms.md"
EMB_MODEL = "qwen3-8b"  # best at realistic prevalence on SYNERGY; see wf_query_conditioned_findings §6


def pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def build_blocks(lab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lex = build_lexical_features(lab).add_prefix("lex_")

    cos_cols = {}
    embeddings = {}
    for short, cache_name in MODELS.items():
        papers = pd.read_parquet(CACHE / f"{cache_name}_papers.parquet").set_index("paper_id")
        E = np.vstack(papers.loc[lab.paper_id].embedding.values).astype("float32")
        briefs = pd.read_parquet(CACHE / f"{cache_name}_usecases.parquet").set_index("use_case_key")
        B = np.vstack(briefs.loc[lab.use_case_key].embedding.values).astype("float32")
        cos = ((E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9))
               * (B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9))).sum(1)
        cos_cols[f"cos_brief_{short}"] = cos
        # Percentile rank within the pool: what makes the scalar mean the same thing in
        # every silo, and label-free, so it stays a row-local fact.
        ranks = np.zeros_like(cos)
        for uc in lab.use_case_key.unique():
            m = (lab.use_case_key == uc).to_numpy()
            ranks[m] = pd.Series(cos[m]).rank(pct=True).to_numpy()
        cos_cols[f"rank_cos_brief_{short}"] = ranks
        embeddings[short] = pd.DataFrame(E, index=lab.index).add_prefix(f"emb_{short}_")

    meta = pd.DataFrame({
        "year": lab.year.astype("float64"),
        "paper_age": 2026.0 - lab.year.astype("float64"),
        "has_abstract": lab.has_abstract.astype(float),
        "n_authors": lab.authors.fillna("").astype(str).str.count(",") + 1.0,
        "citation_count": lab.citation_count.astype("float64"),
    }, index=lab.index)

    return {
        "lex": lex,
        "cos": pd.DataFrame(cos_cols, index=lab.index),
        "meta": meta,
        "emb": embeddings[EMB_MODEL],
    }


def oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        out[te] = pipe().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(REPO / "data" / "processed" / "papers_combined.parquet")
    lab = df[df.triage_label.isin(["positive", "negative"])].reset_index(drop=True)
    y = (lab.triage_label == "positive").to_numpy().astype(int)
    groups = lab.authors.fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()
    groups = groups.where(groups != "", pd.Series([f"__s{i}" for i in range(len(lab))])).to_numpy()

    b = build_blocks(lab)
    arms = {
        "LEAN (cos_brief + rank + lexical + metadata, no embeddings)":
            pd.concat([b["cos"], b["lex"], b["meta"]], axis=1),
        f"EMBEDDING only ({EMB_MODEL})": b["emb"],
        "ALL (embedding + lean)": pd.concat([b["emb"], b["cos"], b["lex"], b["meta"]], axis=1),
    }

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Does the ensemble need the raw embeddings?")
    emit()
    emit(f"Within-silo, author-grouped 5-fold, {args.seeds} seeds, plain LogisticRegression. "
         f"Embedding arm uses {EMB_MODEL} alone ({b['emb'].shape[1]} cols); the lean arm is "
         f"{arms[list(arms)[0]].shape[1]} cols.")
    emit()

    rows, corr_rows, ens_rows = [], [], []
    for uc in sorted(lab.use_case_key.unique()):
        m = (lab.use_case_key == uc).to_numpy()
        yu, gu = y[m], groups[m]
        per_arm_scores: dict[str, list[np.ndarray]] = {k: [] for k in arms}
        for name, X in arms.items():
            Xu = X[m]
            per_seed = []
            for seed in range(args.seeds):
                s = oof(Xu, yu, gu, seed)
                per_arm_scores[name].append(s)
                per_seed.append(metrics(yu, s))
            rows.append({"arm": name, "use_case": uc, **pd.DataFrame(per_seed).mean().to_dict()})

        lean_name, emb_name = list(arms)[0], list(arms)[1]
        # Spearman on out-of-fold scores: do the two arms rank papers the same way? Low
        # correlation is what would justify keeping both as ensemble members.
        cors = [spearmanr(a, e).statistic
                for a, e in zip(per_arm_scores[lean_name], per_arm_scores[emb_name])]
        corr_rows.append({"use_case": uc, "spearman_lean_vs_emb": float(np.mean(cors))})

        # Two cheap recall-oriented combiners over the two arms' OOF probabilities.
        for label, fn in (("mean of arms", lambda a, e: (a + e) / 2),
                          ("max of arms (union-flavoured)", np.maximum)):
            per_seed = [metrics(yu, fn(a, e))
                        for a, e in zip(per_arm_scores[lean_name], per_arm_scores[emb_name])]
            ens_rows.append({"arm": label, "use_case": uc, **pd.DataFrame(per_seed).mean().to_dict()})

    res = pd.DataFrame(rows + ens_rows)
    res.to_csv(REPO / "reports" / "wf_lean_vs_embedding_arms.csv", index=False)

    emit("## 1. Standing alone, and combined")
    emit()
    for metric in ("wss_at_95", "recall_at_10pct", "roc_auc"):
        piv = res.pivot_table(index="arm", columns="use_case", values=metric)
        piv["MEAN"] = piv.mean(axis=1)
        emit(f"**{metric}**")
        emit()
        emit(to_md(piv.sort_values("MEAN", ascending=False).round(3), "arm"))
        emit()

    emit("## 2. Do the two arms make the same mistakes?")
    emit()
    corr = pd.DataFrame(corr_rows).set_index("use_case")
    emit(to_md(corr.round(3), "use_case"))
    emit()
    emit(f"Mean Spearman correlation between the lean and embedding arms' out-of-fold "
         f"rankings: **{corr.spearman_lean_vs_emb.mean():.3f}**. Near 1.0 would mean the "
         "lean arm is redundant; a moderate value means it ranks different papers highly "
         "and can contribute to an ensemble even while scoring lower alone.")
    emit()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
