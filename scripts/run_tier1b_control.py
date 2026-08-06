"""Run the Tier 1b lexical block and its shuffled-brief control.

The question this answers is not "do these features score well" but "do they score well
*because they read the brief*". A feature block that secretly measures abstract length
or lexical richness would score well against any brief at all, so the block is rebuilt
against deliberately wrong briefs (a derangement over use cases) and re-scored. If the
wrong-brief score holds up, the block is measuring something generic and should be
thrown away regardless of how good the real-brief number looks.

Two evaluation surfaces, because they answer different questions (see the split
strategy discussion in reports/):

- LOGO (leave-one-use-case-out) — *not* a production simulation, since customers are
  siloed and no pooled model ever ships. Its job is choosing central defaults: which
  feature families are worth shipping to a customer we have no labels for yet.
- Within-silo (grouped k-fold inside one use case) — this *is* production shape: one
  customer, one model, its own labels.

Usage:
    python scripts/run_tier1b_control.py [--shuffles 5] [--seeds 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexical_features import BRIEF_KEYS, build_lexical_features, derangements  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed" / "papers_combined.parquet"
OUT = REPO / "reports" / "wf_tier1b_lexical_control.md"

# Feature groups, for the ablation. The point of splitting them out is to find which
# part of the block carries the signal, so the shipped default is as small as it can be.
GROUPS = {
    "bm25": [f"bm25_{k}" for k in BRIEF_KEYS] + [f"rank_bm25_{k}" for k in BRIEF_KEYS],
    "overlap": [
        "overlap_must_n", "overlap_must_frac", "rank_overlap_must",
        "overlap_nice_n", "overlap_nice_frac", "rank_overlap_nice",
        "overlap_excl_n", "overlap_excl_frac", "has_exclude_terms",
    ],
    "length_normalised": ["overlap_must_per_1k", "overlap_nice_per_1k", "n_tokens"],
}


def to_md(frame: pd.DataFrame, index_name: str = "") -> str:
    """Minimal markdown table. Hand-rolled rather than pulling in `tabulate` for one
    call — the repo has no test/build tooling and this is not worth a dependency."""
    frame = frame.reset_index()
    frame.columns = [index_name if i == 0 and not str(c).strip() else str(c)
                     for i, c in enumerate(frame.columns)]
    cells = [[f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in frame.itertuples(index=False)]
    header = list(frame.columns)
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) if cells else len(header[i])
              for i in range(len(header))]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join(lines)


def model() -> Pipeline:
    """Deliberately plain and deterministic: a difference in score is then about the
    features, not about which block got a fancier learner. `keep_empty_features` matters
    because the exclusion columns are all-NaN whenever no training use case specified
    exclusion terms."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC for ranking; Recall@10% because that is the operational
    question (how much does the reviewer find in the first tenth of the pile); F2 at the
    default threshold as a reminder that raw probabilities are uncalibrated out-of-domain."""
    k = max(1, int(round(0.10 * len(y_true))))
    top_k = np.argsort(-y_score)[:k]
    n_pos = int(y_true.sum())
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "recall_at_10pct": float(y_true[top_k].sum() / n_pos) if n_pos else np.nan,
        "f2_at_0.5": fbeta_score(y_true, (y_score >= 0.5).astype(int), beta=2, zero_division=0),
    }


def logo(X: pd.DataFrame, y: np.ndarray, use_case: pd.Series) -> pd.DataFrame:
    """Leave-one-use-case-out. Everything is fitted inside the fold."""
    rows = []
    for uc in sorted(use_case.unique()):
        test = (use_case == uc).to_numpy()
        pipe = model().fit(X[~test], y[~test])
        proba = pipe.predict_proba(X[test])[:, 1]
        rows.append({"use_case": uc, **scores(y[test], proba)})
    return pd.DataFrame(rows).set_index("use_case")


def within_silo(
    X: pd.DataFrame, y: np.ndarray, use_case: pd.Series, groups: pd.Series, seeds: int
) -> pd.DataFrame:
    """Grouped, stratified k-fold inside each use case — the production shape.

    Grouped by first author so the same author's papers never straddle the split, and
    repeated across seeds because 260-360 rows per silo makes any single split noisy.
    """
    rows = []
    for uc in sorted(use_case.unique()):
        mask = (use_case == uc).to_numpy()
        Xu, yu, gu = X[mask], y[mask], groups[mask].to_numpy()
        per_seed = []
        for seed in range(seeds):
            oof = np.full(len(yu), np.nan)
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in splitter.split(Xu, yu, groups=gu):
                pipe = model().fit(Xu.iloc[train_idx], yu[train_idx])
                oof[test_idx] = pipe.predict_proba(Xu.iloc[test_idx])[:, 1]
            per_seed.append(scores(yu, oof))
        mean = pd.DataFrame(per_seed).mean().to_dict()
        mean["roc_auc_sd"] = float(pd.DataFrame(per_seed)["roc_auc"].std())
        rows.append({"use_case": uc, **mean})
    return pd.DataFrame(rows).set_index("use_case")


def warm_start(
    variants: dict[str, pd.DataFrame],
    y: np.ndarray,
    use_case: pd.Series,
    budgets: tuple[int, ...],
    seeds: int,
) -> pd.DataFrame:
    """How good is each representation at n labels from the target use case?

    This is the question production actually asks. A new customer arrives with some
    labels, and the only thing that matters is how few of them are needed before the
    ranking is useful. Labels are drawn at RANDOM rather than stratified, because that
    is what arriving labels look like -- stratifying would quietly hand the model a
    balanced sample it will not get.

    The n=0 row is the transfer baseline: trained on the other five use cases, zero
    labels from the target. It is the honest reference point for "was the cross-domain
    work worth it", and it is the number every LOGO experiment in this repo reports.
    """
    rows = []
    for name, X in variants.items():
        for uc in sorted(use_case.unique()):
            mask = (use_case == uc).to_numpy()
            Xu, yu = X[mask], y[mask]

            transfer = model().fit(X[~mask], y[~mask]).predict_proba(Xu)[:, 1]
            rows.append({"representation": name, "use_case": uc, "n_labels": 0,
                         **scores(yu, transfer)})

            for n in budgets:
                if n >= len(yu):
                    continue
                per_seed = []
                for seed in range(seeds):
                    rng = np.random.default_rng(seed)
                    order = rng.permutation(len(yu))
                    train_idx, test_idx = order[:n], order[n:]
                    if len(np.unique(yu[train_idx])) < 2:
                        continue  # a single-class draw teaches nothing; skip, don't impute
                    pipe = model().fit(Xu.iloc[train_idx], yu[train_idx])
                    per_seed.append(scores(yu[test_idx], pipe.predict_proba(Xu.iloc[test_idx])[:, 1]))
                if per_seed:
                    rows.append({"representation": name, "use_case": uc, "n_labels": n,
                                 **pd.DataFrame(per_seed).mean().to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffles", type=int, default=5, help="wrong-brief derangements")
    parser.add_argument("--seeds", type=int, default=5, help="seeds for within-silo CV")
    parser.add_argument("--ws-seeds", type=int, default=25, help="seeds for warm-start curve")
    parser.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200],
                        help="target-use-case label budgets for the warm-start curve")
    args = parser.parse_args()

    df = pd.read_parquet(DATA)
    lab = df[df.triage_label.isin(["positive", "negative"])].reset_index(drop=True)
    y = (lab.triage_label == "positive").to_numpy().astype(int)
    use_case = lab.use_case_key
    # Author grouping mirrors notebooks/modelling/wf_fold_pca_test.ipynb: papers by the
    # same first author must not straddle a split, or a within-silo score is inflated.
    groups = lab.authors.fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()
    groups = groups.where(groups != "", pd.Series([f"__solo_{i}" for i in range(len(lab))]))

    embeddings = pd.DataFrame(
        np.vstack(lab.embedding.values).astype("float32"), index=lab.index
    ).add_prefix("emb_")

    real = build_lexical_features(lab)
    maps = [derangements(sorted(use_case.unique()), seed=s) for s in range(args.shuffles)]
    shuffled = [(m, build_lexical_features(lab, brief_map=m)) for m in maps]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Tier 1b lexical block — results and the shuffled-brief control")
    emit()
    emit(f"`{len(lab)}` labelled papers, {use_case.nunique()} use cases, "
         f"{real.shape[1]} lexical features. Plain LogisticRegression throughout.")
    emit()
    ceiling = float((0.10 / (y.mean())) if y.mean() else np.nan)
    emit(f"> Read `recall_at_10pct` against its ceiling, not against 1.0. These pools are "
         f"~{y.mean():.0%} positive, so reviewing the top 10% of a pool can recover at most "
         f"~{ceiling:.2f} of the positives even with perfect ranking. The number becomes "
         f"informative at production prevalence, not here.")
    emit()

    # ---- 1. LOGO: the defaults-selection surface -------------------------------------
    emit("## 1. Leave-one-use-case-out (chooses central defaults, not a production number)")
    emit()
    variants = {
        "paper embedding (384-d, current baseline)": embeddings,
        "Tier 1b lexical — real briefs": real,
        "Tier 1b + embedding": pd.concat([real, embeddings], axis=1),
    }
    logo_results = {name: logo(X, y, use_case) for name, X in variants.items()}

    shuffled_runs = [logo(feats, y, use_case) for _, feats in shuffled]
    shuffled_mean = pd.concat(shuffled_runs).groupby(level=0).mean()
    logo_results["Tier 1b lexical — SHUFFLED briefs (control)"] = shuffled_mean

    summary = pd.DataFrame({
        name: res[["roc_auc", "pr_auc", "recall_at_10pct"]].mean()
        for name, res in logo_results.items()
    }).T.round(3)
    emit(to_md(summary, "representation"))
    emit()
    emit("Per use case, ROC-AUC:")
    emit()
    per_uc = pd.DataFrame({n: r["roc_auc"] for n, r in logo_results.items()}).round(3)
    emit(to_md(per_uc, "use_case"))
    emit()

    real_auc = logo_results["Tier 1b lexical — real briefs"]["roc_auc"]
    ctrl_auc = shuffled_mean["roc_auc"]
    gap = float(real_auc.mean() - ctrl_auc.mean())
    wins = int((real_auc > ctrl_auc).sum())
    spread = float(pd.concat([r["roc_auc"] for r in shuffled_runs], axis=1).std(axis=1).mean())
    emit(f"**Control:** real briefs beat wrong briefs on **{wins}/6** use cases, "
         f"mean ROC-AUC gap **{gap:+.3f}** (between-derangement sd {spread:.3f}).")
    emit()

    # ---- 2. Within-silo: the production surface --------------------------------------
    emit("## 2. Within-silo, author-grouped 5-fold (this is the production shape)")
    emit()
    silo = {
        "paper embedding": within_silo(embeddings, y, use_case, groups, args.seeds),
        "Tier 1b lexical": within_silo(real, y, use_case, groups, args.seeds),
        "Tier 1b + embedding": within_silo(
            pd.concat([real, embeddings], axis=1), y, use_case, groups, args.seeds),
        "Tier 1b — SHUFFLED (control)": within_silo(
            shuffled[0][1], y, use_case, groups, args.seeds),
    }
    emit(to_md(pd.DataFrame({n: r["roc_auc"] for n, r in silo.items()}).round(3), "use_case"))
    emit()
    emit("Seed-to-seed sd (Tier 1b): " + ", ".join(
        f"{uc} {v:.3f}" for uc, v in silo["Tier 1b lexical"]["roc_auc_sd"].items()))
    emit()

    # ---- 3. Ablation: what inside the block is doing the work -------------------------
    emit("## 3. Which part of the block carries it (LOGO mean ROC-AUC)")
    emit()
    ablation = {name: logo(real[cols], y, use_case)["roc_auc"].mean()
                for name, cols in GROUPS.items()}
    ablation["all"] = real_auc.mean()
    ablation["all minus length_normalised"] = logo(
        real.drop(columns=GROUPS["length_normalised"]), y, use_case)["roc_auc"].mean()
    emit(to_md(pd.Series(ablation).round(3).to_frame("mean LOGO ROC-AUC"), "feature group"))
    emit()

    # ---- 4. Warm-start curve: the label-efficiency claim ------------------------------
    emit("## 4. Warm-start curve — ROC-AUC vs. number of labels from the target use case")
    emit()
    emit("`n=0` is the transfer baseline (trained on the other five use cases). Labels are "
         "drawn at random, not stratified, because that is what arriving labels look like. "
         f"Mean of {args.ws_seeds} seeds.")
    emit()
    curve = warm_start(
        {"embedding": embeddings, "Tier 1b": real,
         "Tier 1b + embedding": pd.concat([real, embeddings], axis=1)},
        y, use_case, tuple(args.budgets), args.ws_seeds,
    )
    curve.to_csv(REPO / "reports" / "wf_tier1b_warm_start_curve.csv", index=False)

    pooled = curve.pivot_table(index="n_labels", columns="representation",
                               values="roc_auc", aggfunc="mean").round(3)
    emit("Mean across all six use cases:")
    emit()
    emit(to_md(pooled, "n_labels"))
    emit()
    for name in ("Tier 1b + embedding", "Tier 1b", "embedding"):
        block = curve[curve.representation == name].pivot_table(
            index="use_case", columns="n_labels", values="roc_auc").round(3)
        emit(f"Per use case — {name}:")
        emit()
        emit(to_md(block, "use_case"))
        emit()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
