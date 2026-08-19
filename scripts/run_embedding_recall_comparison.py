"""Qwen3-4B / Qwen3-8B / Jasper, scored through the recall lens, on the folds we ship on.

WHY THIS EXISTS
---------------
reports/wf_embedding_bakeoff.md picked Jasper+Qwen3-4B on mean holdout PR-AUC, by a
0.004 margin. Two things about that decision need re-testing rather than inheriting:

1. The objective is high recall (F2), not ranking area. Those are different questions,
   and the bake-off's own calibration table already ranks the models differently under
   F2 than under PR-AUC.
2. The fold context has since changed. Use cases are siloed per customer -- no pooled
   model ever ships -- so leave-one-use-case-out is a *defaults-selection* instrument,
   not a production estimate. The production number is within-silo.

So every model here is scored on WSS@95 and Recall@k as primaries, on both surfaces,
plus the warm-start curve that reflects how a customer actually arrives (with labels).

WSS@95 (Cohen et al. 2006) is the headline: the fraction of the pool a reviewer can skip
while still catching 95% of the relevant papers, minus what they'd have saved by chance.
It is the right primary here because it is prevalence-aware -- unlike Recall@10%, whose
ceiling collapses as prevalence rises (these pools run 26-77% positive, so reviewing 10%
of a pool cannot recover more than ~0.13-0.38 of the positives even with perfect ranking).
Its own ceiling is 0.95*(1-prevalence), reported alongside so a number is never read as
though 1.0 were reachable.

Metric helpers come from scripts/fold_pipeline_utils.py rather than being redefined --
same reason that module exists (CONTRIBUTING.md §7: one implementation, so the two pipelines
cannot drift apart).

Usage:
    python scripts/run_embedding_recall_comparison.py [--seeds 5] [--ws-seeds 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_pipeline_utils import ranking_metrics  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402
from run_tier1b_control import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "processed" / "embeddings_cache"
DATA = REPO / "data" / "processed" / "papers_combined.parquet"
OUT = REPO / "reports" / "wf_embedding_recall_comparison.md"

MODELS = {
    "jasper": "infgrad__Jasper-Token-Compression-600M",
    "qwen3-4b": "Qwen__Qwen3-Embedding-4B",
    "qwen3-8b": "qwen__qwen3-embedding-8b",
}
PCA_COMPONENTS = 64


def supervised_pipeline(n_pca: int | None) -> Pipeline:
    """Plain LogisticRegression, optionally on PCA-reduced embeddings.

    PCA sits *inside* the pipeline so it is refit per fold. Fitting it once over all rows
    would leak test-fold structure into the components -- the exact mistake
    reports/wf_featureengineering_review.md §7 warns about, and the reason its 0.520->0.714
    result is trustworthy.
    """
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler())]
    if n_pca:
        # n_components is capped by the caller, not here: a 25-label warm-start fold cannot
        # support 64 components, and shrinking to fit is the right behaviour for a curve
        # whose whole point is small label budgets.
        steps.append(("pca", PCA(n_components=n_pca, random_state=0)))
    steps.append(("clf", LogisticRegression(max_iter=5000, class_weight="balanced")))
    return Pipeline(steps)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Recall lens first, AUC kept for continuity with everything already measured."""
    out = ranking_metrics(y_true, scores)
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    # The reachable maximum for this pool, so WSS is never read as though 1.0 were on
    # the table. At 77% positive (solar_leo) the best possible WSS@95 is ~0.22.
    out["wss_ceiling"] = float(0.95 * (1.0 - y_true.mean()))
    return out


def build_variants(lab: pd.DataFrame, lex: pd.DataFrame) -> dict[str, dict]:
    """name -> {X, n_pca, cosine}. `cosine` variants need no classifier at all."""
    variants: dict[str, dict] = {"lexical only (Tier 1b)": {"X": lex, "n_pca": None}}
    for short, cache_name in MODELS.items():
        papers = pd.read_parquet(CACHE / f"{cache_name}_papers.parquet").set_index("paper_id")
        E = pd.DataFrame(
            np.vstack(papers.loc[lab.paper_id].embedding.values).astype("float32"),
            index=lab.index,
        ).add_prefix(f"{short}_")
        briefs = pd.read_parquet(CACHE / f"{cache_name}_usecases.parquet").set_index("use_case_key")
        B = np.vstack(briefs.loc[lab.use_case_key].embedding.values).astype("float32")

        Ev = E.to_numpy()
        cos = (
            (Ev / (np.linalg.norm(Ev, axis=1, keepdims=True) + 1e-9))
            * (B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9))
        ).sum(1)

        variants[f"{short}: cosine-to-brief (0 labels)"] = {"cosine": pd.Series(cos, index=lab.index)}
        variants[f"{short}: embedding"] = {"X": E, "n_pca": None}
        variants[f"{short}: embedding PCA-{PCA_COMPONENTS}"] = {"X": E, "n_pca": PCA_COMPONENTS}
        variants[f"{short}: PCA-{PCA_COMPONENTS} + lexical"] = {
            "X": pd.concat([E, lex], axis=1), "n_pca": None, "pca_on": list(E.columns)}
    return variants


def _fit_scores(spec: dict, X_train, y_train, X_test) -> np.ndarray:
    """One fit. `pca_on` reduces only the embedding block, leaving the ~22 lexical
    columns at full width -- otherwise the lexical features get folded into the same
    64 components and the whole point of concatenating them is lost."""
    if "pca_on" in spec:
        emb_cols = spec["pca_on"]
        lex_cols = [c for c in X_train.columns if c not in set(emb_cols)]
        k = min(PCA_COMPONENTS, len(X_train) - 1, len(emb_cols))
        pre = Pipeline([("scale", StandardScaler()),
                        ("pca", PCA(n_components=k, random_state=0))]).fit(X_train[emb_cols])
        tr = np.hstack([pre.transform(X_train[emb_cols]),
                        SimpleImputer(strategy="median", keep_empty_features=True)
                        .fit(X_train[lex_cols]).transform(X_train[lex_cols])])
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train[lex_cols])
        te = np.hstack([pre.transform(X_test[emb_cols]), imp.transform(X_test[lex_cols])])
        sc = StandardScaler().fit(tr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(tr), y_train)
        return clf.predict_proba(sc.transform(te))[:, 1]
    n_pca = spec["n_pca"]
    if n_pca:
        n_pca = min(n_pca, len(X_train) - 1, X_train.shape[1])
    pipe = supervised_pipeline(n_pca).fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def run_logo(variants, y, use_case) -> pd.DataFrame:
    rows = []
    for name, spec in variants.items():
        for uc in sorted(use_case.unique()):
            te = (use_case == uc).to_numpy()
            if "cosine" in spec:
                s = spec["cosine"][te].to_numpy()
            else:
                s = _fit_scores(spec, spec["X"][~te], y[~te], spec["X"][te])
            rows.append({"variant": name, "use_case": uc, **metrics(y[te], s)})
    return pd.DataFrame(rows)


def run_within_silo(variants, y, use_case, groups, seeds) -> pd.DataFrame:
    rows = []
    for name, spec in variants.items():
        for uc in sorted(use_case.unique()):
            m = (use_case == uc).to_numpy()
            yu = y[m]
            if "cosine" in spec:
                # No training involved, so there is nothing to hold out.
                rows.append({"variant": name, "use_case": uc, **metrics(yu, spec["cosine"][m].to_numpy())})
                continue
            Xu, gu = spec["X"][m], groups[m].to_numpy()
            per_seed = []
            for seed in range(seeds):
                oof = np.full(len(yu), np.nan)
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xu, yu, gu):
                    oof[te] = _fit_scores(spec, Xu.iloc[tr], yu[tr], Xu.iloc[te])
                per_seed.append(metrics(yu, oof))
            rows.append({"variant": name, "use_case": uc, **pd.DataFrame(per_seed).mean().to_dict(),
                         "wss_sd": float(pd.DataFrame(per_seed)["wss_at_95"].std())})
    return pd.DataFrame(rows)


def run_warm_start(variants, y, use_case, budgets, seeds) -> pd.DataFrame:
    rows = []
    for name, spec in variants.items():
        if "cosine" in spec:
            # Flat in n by construction (no training), but emitted at every budget so the
            # supervised curves can be read against the zero-label floor they must beat.
            for uc in sorted(use_case.unique()):
                m = (use_case == uc).to_numpy()
                row = metrics(y[m], spec["cosine"][m].to_numpy())
                rows += [{"variant": name, "use_case": uc, "n_labels": n, **row}
                         for n in budgets]
            continue
        for uc in sorted(use_case.unique()):
            m = (use_case == uc).to_numpy()
            Xu, yu = spec["X"][m], y[m]
            for n in budgets:
                if n >= len(yu):
                    continue
                per_seed = []
                for seed in range(seeds):
                    order = np.random.default_rng(seed).permutation(len(yu))
                    tr, te = order[:n], order[n:]
                    if len(np.unique(yu[tr])) < 2:
                        continue
                    per_seed.append(metrics(yu[te], _fit_scores(spec, Xu.iloc[tr], yu[tr], Xu.iloc[te])))
                if per_seed:
                    rows.append({"variant": name, "use_case": uc, "n_labels": n,
                                 **pd.DataFrame(per_seed).mean().to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ws-seeds", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    lab = df[df.triage_label.isin(["positive", "negative"])].reset_index(drop=True)
    y = (lab.triage_label == "positive").to_numpy().astype(int)
    use_case = lab.use_case_key
    groups = lab.authors.fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()
    groups = groups.where(groups != "", pd.Series([f"__solo_{i}" for i in range(len(lab))]))

    lex = build_lexical_features(lab)
    variants = build_variants(lab, lex)

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Embedding comparison through the recall lens")
    emit()
    emit(f"{len(lab)} labelled papers, 6 use cases. Primary metric **WSS@95**: the share of "
         "the pool a reviewer can skip while still finding 95% of the relevant papers. "
         "Its ceiling is `0.95 x (1 - prevalence)`, shown per use case, because these pools "
         "are 26-77% positive and a WSS of 0.30 means very different things at each end.")
    emit()
    prev = pd.DataFrame({"prevalence": lab.groupby("use_case_key").apply(
        lambda g: (g.triage_label == "positive").mean())})
    prev["wss_ceiling"] = 0.95 * (1 - prev.prevalence)
    emit(to_md(prev.round(3), "use_case"))
    emit()

    logo = run_logo(variants, y, use_case)
    silo = run_within_silo(variants, y, use_case, groups, args.seeds)
    for frame, path in ((logo, "wf_recall_logo.csv"), (silo, "wf_recall_within_silo.csv")):
        frame.to_csv(REPO / "reports" / path, index=False)

    emit("## 1. Within-silo — the production surface")
    emit()
    emit("One customer, one model, its own labels, author-grouped 5-fold, "
         f"{args.seeds} seeds.")
    emit()
    for metric in ("wss_at_95", "recall_at_10pct", "roc_auc"):
        piv = silo.pivot_table(index="variant", columns="use_case", values=metric)
        piv["MEAN"] = piv.mean(axis=1)
        emit(f"**{metric}**")
        emit()
        emit(to_md(piv.sort_values("MEAN", ascending=False).round(3), "variant"))
        emit()

    emit("## 2. Leave-one-use-case-out — chooses central defaults, not a production number")
    emit()
    for metric in ("wss_at_95", "roc_auc"):
        piv = logo.pivot_table(index="variant", columns="use_case", values=metric)
        piv["MEAN"] = piv.mean(axis=1)
        emit(f"**{metric}**")
        emit()
        emit(to_md(piv.sort_values("MEAN", ascending=False).round(3), "variant"))
        emit()

    emit("## 3. Warm-start — WSS@95 vs. labels from the target use case")
    emit()
    ws = run_warm_start(variants, y, use_case, tuple(args.budgets), args.ws_seeds)
    ws.to_csv(REPO / "reports" / "wf_recall_warm_start.csv", index=False)
    piv = ws.pivot_table(index="variant", columns="n_labels", values="wss_at_95")
    emit(to_md(piv.round(3), "variant"))
    emit()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
