"""
compare_embeddings.py — visually compare the LATENT SPACE of different embedding/NLP
models on your labelled corpus.

THE THREE QUESTIONS THIS SCRIPT ANSWERS
-----------------------------------------
1. Is a given embedding model's latent space "sane" — or is it dense/collapsed (every
   paper's vector points in roughly the same direction, so the model can't actually tell
   papers apart) versus disperse (vectors spread out, genuinely encoding differences)?
2. How CENTRAL is the use case's own embedding relative to the corpus of abstracts —
   does it sit near the corpus's centre of mass (i.e. is the use case's phrasing
   "typical" of the papers found for it), or off to one side?
3. How do different models — and different TYPES of model (classic lexical TF-IDF vs.
   modern neural sentence embedders) — compare on both of the above, side by side?

For each model you ask it to try, the script:
    1. embeds every paper's "title. abstract" text,
    2. embeds the use case's own text the same way,
    3. computes latent-space diagnostics (see "THE METRICS" below),
    4. draws a 3-panel row: a 2D PCA map of the whole corpus, a pairwise-similarity
       histogram (the dense/disperse check), and a use-case-vs-abstracts centrality
       histogram,
    5. (if there are enough triage labels) also reports cross-validated ROC-AUC, as a
       "does this space support classification" sanity check — kept from the previous
       version of this script, now secondary to the visual diagnostics above.

All model rows are stacked into ONE figure so you can compare them by eye.

THE METRICS (and why these specific ones)
--------------------------------------------
avg_pairwise_cosine — mean cosine similarity between every pair of paper vectors.
    This is the single number behind "dense vs. disperse". A well-known pathology of
    sentence embedding models ("representation degeneration" / anisotropy, Ethayarajh
    2019) is that ALL sentences end up crammed into a narrow cone of the vector space —
    every pair looks similar (~0.7+) regardless of topic, which quietly wrecks anything
    downstream that relies on cosine similarity (search, clustering, the agent's own
    triage ranking). Low-to-moderate average similarity (roughly 0.05-0.5) means the
    model is actually spreading papers out by content. These bands are a judgement call
    (there's no formal threshold in the literature), not a certified cutoff — read the
    histogram shape, not just the one number.

participation_ratio — an "effective dimensionality": (sum of PCA eigenvalues)^2 /
    sum(eigenvalues^2). A model that only truly varies along a couple of directions
    (everything else is noise) gets a low ratio even if its raw dimension is 768 —
    another angle on the same collapse question as avg_pairwise_cosine, from the
    variance side rather than the similarity side.

top1_variance_ratio — the fraction of variance explained by the single largest PCA
    component. A dominant PC1 usually means one thing (often something boring, like
    text length or a dominant topic) is swamping every other distinction the model
    could be making.

use_case_to_centroid_sim / use_case_centroid_percentile — cosine similarity between the
    embedded use case and the corpus's centroid vector (the mean of all abstract
    vectors), and what percentile that similarity falls at relative to every abstract's
    own similarity to that same centroid. A high percentile says the use case's wording
    sits near the "centre" of the papers found for it — as central as a typical
    abstract, or more so. A low percentile says the use case is an outlier relative to
    its own corpus, worth a second look at whether the query terms actually match the
    literature's vocabulary.

roc_auc (secondary) — cross-validated LogisticRegression AUC predicting the triage
    label from the embeddings, IDENTICAL to academic_research_agent's own
    model.py:_cross_validated_roc_auc. Kept because "does this space separate my labels"
    is still a useful downstream check, but the dense/disperse and centroid questions
    above are the ones this version of the script is built around.

WHERE THE USE CASE TEXT COMES FROM
--------------------------------------
An academic_research_agent ML export only carries the use case's short NAME in its
`use_case` column (e.g. "Climate change extremes and agriculture practices"), not the
full objective/key-terms JSON the agent itself embeds for ranking. That name is used by
default. If you have the fuller wording (objective + must-include terms), pass it with
--use-case-text for a closer match to what the agent actually ranks against.

MODEL TYPES YOU CAN COMPARE
-------------------------------
- "tfidf" — a classic lexical baseline: TF-IDF followed by TruncatedSVD (dense, so PCA/
  cosine still apply). No neural network at all — the useful contrast case for "is my
  baseline embedding model's latent space actually earning its keep over plain word
  statistics?"
- any fastembed model name (see --list-models) — local ONNX sentence embedders, e.g.
  "BAAI/bge-small-en-v1.5" or the multilingual model academic_research_agent uses.

HOW TO RUN IT
--------------
    python scripts/compare_embeddings.py --data data/raw/your-export.parquet

Useful flags:
    --models "tfidf,BAAI/bge-small-en-v1.5"     which models/types to compare
    --label-col review_label                    use the review-stage label instead
    --use-case-text "..."                       override the export's short use_case name
    --projection tsne                           use t-SNE instead of PCA for the 2D map
    --list-models                               print fastembed's full model catalogue

OUTPUTS
--------
    reports/latent_space_comparison.png   the multi-model visual comparison (main output)
    reports/embedding_comparison.csv      the scalar metrics table behind the plot

A LIMITATION TO KNOW ABOUT
-----------------------------
All of this is descriptive of ONE corpus (typically a few hundred papers at most). With
a small n, the PCA map and histograms show real structure but the scalar metrics have
wide error bars — read them as "which model looks meaningfully different", not as
precise measurements.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — this is a batch script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# A deliberately mixed default set: a classic lexical baseline (no neural network at
# all), a small English neural model, and the multilingual model academic_research_agent
# itself uses by default — enough spread to see whether "neural" and "bigger/multilingual"
# actually buy anything on YOUR corpus, not just in general.
DEFAULT_MODELS = [
    "tfidf",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# Triage/review label values the agent's exports use.
POSITIVE_VALUES = {"positive"}
NEGATIVE_VALUES = {"negative"}
MAX_CV_FOLDS = 5

LABEL_COLORS = {
    "positive": "#2ca02c",
    "negative": "#d62728",
    "pass": "#9467bd",
}
UNLABELLED_COLOR = "#c7c7c7"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_export(path: Path) -> pd.DataFrame:
    """Load an academic_research_agent export (.parquet or .csv), no filtering — every
    row is kept, since the latent-space diagnostics want the WHOLE corpus, not just the
    labelled subset (labels are only needed later, for colouring the plot and for the
    secondary ROC-AUC check)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")


def build_text(df: pd.DataFrame) -> list[str]:
    """Build the "title. abstract" string embedded for each paper — the same shape of
    text academic_research_agent embeds (embeddings.paper_embedding_text)."""
    titles = df.get("title", pd.Series([""] * len(df))).fillna("")
    abstracts = df.get("abstract", pd.Series([""] * len(df))).fillna("")
    return [f"{t}. {a}".strip(". ").strip() for t, a in zip(titles, abstracts)]


def get_use_case_text(df: pd.DataFrame, use_case_col: str, override: str | None) -> str:
    """Return the text to embed as "the use case". An explicit --use-case-text always
    wins; otherwise fall back to the export's use_case column (see the module docstring
    for why that's a short name, not the full objective)."""
    if override:
        return override
    if use_case_col in df.columns:
        values = df[use_case_col].dropna().unique()
        if len(values) > 0:
            return str(values[0])
    raise ValueError(
        f"No use case text found: column '{use_case_col}' is missing/empty and "
        "--use-case-text was not given."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backends — fastembed neural models, or a classic TF-IDF+SVD baseline
# ─────────────────────────────────────────────────────────────────────────────

def embed_with_model(model_name: str, texts: list[str]) -> tuple[np.ndarray, float]:
    """Embed `texts` (+ implicitly, the use-case text is embedded via the SAME call by
    the caller appending it to `texts`) with either fastembed or the "tfidf" baseline.
    Returns (vectors, seconds_elapsed). Imports are local so --list-models stays cheap.
    """
    start = time.monotonic()
    if model_name == "tfidf":
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        sparse = vectorizer.fit_transform(texts)
        n_components = max(2, min(100, sparse.shape[0] - 1, sparse.shape[1] - 1))
        vectors = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(sparse)
    else:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=model_name)
        vectors = np.array([v for v in model.embed(texts)])
    elapsed = time.monotonic() - start
    return vectors, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Latent-space diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rows(V: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def dispersion_metrics(V: np.ndarray) -> dict:
    """The dense-vs-disperse diagnostics: pairwise cosine similarities (full
    distribution, for the histogram, plus their mean) and PCA-based effective
    dimensionality (see module docstring for what each number means)."""
    V_norm = normalize_rows(V)
    sim_matrix = V_norm @ V_norm.T
    iu = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[iu]

    n_components = max(2, min(50, V.shape[0] - 1, V.shape[1]))
    pca = PCA(n_components=n_components)
    pca.fit(V_norm)
    eigenvalues = pca.explained_variance_
    participation_ratio = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    return {
        "pairwise_sims": pairwise_sims,
        "avg_pairwise_cosine": float(pairwise_sims.mean()),
        "participation_ratio": participation_ratio,
        "top1_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def interpret_dispersion(avg_cos: float) -> str:
    """A judgement-call reading of avg_pairwise_cosine — not a certified threshold, a
    documented rule of thumb (same spirit as academic_research_agent's own
    SIMILAR_KEPT_THRESHOLD judgement call in model.py)."""
    if avg_cos > 0.7:
        return "dense / likely collapsed (anisotropic) - vectors point in similar directions"
    if avg_cos > 0.35:
        return "moderately dense"
    if avg_cos > 0.05:
        return "sane / reasonably disperse"
    return "very disperse (near-orthogonal) - check this isn't noise"


def centroid_analysis(V: np.ndarray, use_case_vec: np.ndarray) -> dict:
    """How central the use case's embedding is relative to the corpus of abstracts —
    see the module docstring's use_case_to_centroid_sim / percentile explanation."""
    V_norm = normalize_rows(V)
    centroid = V_norm.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid) or 1.0

    abstract_to_centroid = (V_norm @ centroid) / centroid_norm

    uc_norm = use_case_vec / (np.linalg.norm(use_case_vec) or 1.0)
    use_case_to_centroid = float(np.dot(uc_norm, centroid) / centroid_norm)

    percentile = float((abstract_to_centroid < use_case_to_centroid).mean() * 100)

    return {
        "centroid": centroid,
        "abstract_to_centroid_sims": abstract_to_centroid,
        "use_case_to_centroid_sim": use_case_to_centroid,
        "use_case_centroid_percentile": percentile,
    }


def cross_validated_roc_auc(X: np.ndarray, y: np.ndarray) -> dict:
    """Mean cross-validated ROC-AUC for a fresh LogisticRegression, mirroring
    academic_research_agent's model.py:_cross_validated_roc_auc exactly."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(MAX_CV_FOLDS, n_pos, n_neg)
    if k < 2:
        return {"roc_auc": None, "n_folds": 0}

    folds = StratifiedKFold(n_splits=k, shuffle=False)
    aucs = []
    for train_idx, test_idx in folds.split(X, y):
        y_test = y[test_idx]
        if len(set(y_test.tolist())) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pos_col = list(clf.classes_).index(1)
        y_score = clf.predict_proba(X[test_idx])[:, pos_col]
        aucs.append(roc_auc_score(y_test, y_score))

    if not aucs:
        return {"roc_auc": None, "n_folds": 0}
    return {"roc_auc": float(np.mean(aucs)), "n_folds": len(aucs)}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — one row per model: [PCA/t-SNE map | dispersion histogram | centroid histogram]
# ─────────────────────────────────────────────────────────────────────────────

def project_2d(V: np.ndarray, use_case_vec: np.ndarray, centroid: np.ndarray, method: str):
    """Project the corpus + use-case + centroid into 2D with ONE shared fit, so all
    three land in the same coordinate space and are visually comparable."""
    combined = np.vstack([V, use_case_vec[None, :], centroid[None, :]])
    if method == "tsne":
        perplexity = max(5, min(30, (combined.shape[0] - 1) // 3))
        coords = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(combined)
        var_ratio = (None, None)  # t-SNE has no explained-variance concept
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(combined)
        var_ratio = tuple(pca.explained_variance_ratio_)
    return coords[:-2], coords[-2], coords[-1], var_ratio


def plot_model_row(fig, axes_row, model_name: str, df: pd.DataFrame, label_col: str,
                    coords_2d: np.ndarray, uc_xy: np.ndarray, centroid_xy: np.ndarray,
                    var_ratio: tuple, disp: dict, cent: dict, roc: dict) -> None:
    ax_map, ax_disp, ax_cent = axes_row

    # --- panel 1: 2D map of the whole corpus, coloured by label ---
    labels = df[label_col] if label_col in df.columns else pd.Series([None] * len(df))
    drawn_any_legend_label = set()
    order = [None, "pass", "negative", "positive"]  # draw important points last (on top)
    for lab in order:
        mask = labels.isna() if lab is None else (labels == lab)
        mask = mask.to_numpy()
        if not mask.any():
            continue
        color = UNLABELLED_COLOR if lab is None else LABEL_COLORS.get(lab, UNLABELLED_COLOR)
        legend_label = "unlabelled" if lab is None else lab
        ax_map.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=18, alpha=0.7,
                        color=color, label=legend_label if legend_label not in drawn_any_legend_label else None)
        drawn_any_legend_label.add(legend_label)

    ax_map.scatter(*centroid_xy, marker="X", s=200, color="black", edgecolors="white",
                   linewidths=1, zorder=5, label="centroid")
    ax_map.scatter(*uc_xy, marker="*", s=350, color="#8c00ff", edgecolors="white",
                   linewidths=1, zorder=6, label="use case")

    axis_label = "PC" if var_ratio[0] is not None else "dim"
    if var_ratio[0] is not None:
        ax_map.set_xlabel(f"{axis_label}1 ({var_ratio[0]:.1%} var)")
        ax_map.set_ylabel(f"{axis_label}2 ({var_ratio[1]:.1%} var)")
    ax_map.set_title(f"{model_name}\ncorpus map", fontsize=10)
    ax_map.legend(fontsize=7, loc="best")

    # --- panel 2: pairwise-cosine histogram — the dense/disperse check ---
    ax_disp.hist(disp["pairwise_sims"], bins=30, color="#4c72b0")
    ax_disp.axvline(disp["avg_pairwise_cosine"], color="black", linestyle="--", linewidth=1)
    ax_disp.set_title(
        f"dispersion: avg={disp['avg_pairwise_cosine']:.2f}\n"
        f"({interpret_dispersion(disp['avg_pairwise_cosine'])})",
        fontsize=8,
    )
    ax_disp.set_xlabel("pairwise cosine similarity")
    ax_disp.set_ylabel("pair count")

    # --- panel 3: abstract-to-centroid histogram — how central is the use case? ---
    ax_cent.hist(cent["abstract_to_centroid_sims"], bins=30, color="#55a868")
    ax_cent.axvline(cent["use_case_to_centroid_sim"], color="#8c00ff", linestyle="--", linewidth=2)
    ax_cent.set_title(
        f"use case @ {ordinal(round(cent['use_case_centroid_percentile']))} pct of abstracts\n"
        f"(sim={cent['use_case_to_centroid_sim']:.2f})",
        fontsize=8,
    )
    ax_cent.set_xlabel("abstract-to-centroid cosine similarity")
    ax_cent.set_ylabel("paper count")

    roc_txt = f"roc_auc={roc['roc_auc']:.3f} (n_folds={roc['n_folds']})" if roc["roc_auc"] is not None else "roc_auc: n/a (too few labels)"
    ax_map.annotate(roc_txt, xy=(0, -0.18), xycoords="axes fraction", fontsize=7, color="#555555")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to an academic_research_agent .parquet or .csv export")
    parser.add_argument("--label-col", default="triage_label", help="Column with positive/negative/pass labels (default: triage_label)")
    parser.add_argument("--use-case-col", default="use_case", help="Column holding the use case text (default: use_case)")
    parser.add_argument("--use-case-text", default=None, help="Override the use case text instead of reading --use-case-col")
    parser.add_argument("--models", default=None, help="Comma-separated model names/types (default: tfidf + two fastembed models)")
    parser.add_argument("--projection", choices=["pca", "tsne"], default="pca", help="2D projection method for the corpus map (default: pca)")
    parser.add_argument("--out", type=Path, default=Path("reports/embedding_comparison.csv"), help="Where to save the scalar metrics table")
    parser.add_argument("--out-plot", type=Path, default=Path("reports/latent_space_comparison.png"), help="Where to save the comparison figure")
    parser.add_argument("--list-models", action="store_true", help="Print every model fastembed supports, then exit")
    args = parser.parse_args()

    if args.list_models:
        from fastembed import TextEmbedding
        for m in TextEmbedding.list_supported_models():
            print(f"{m['model']}  (dim={m['dim']})")
        return

    if args.data is None:
        parser.error("--data is required unless --list-models is passed")

    model_names = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    texts = build_text(df)
    non_empty = [i for i, t in enumerate(texts) if t]
    if len(non_empty) < len(texts):
        print(f"  Dropping {len(texts) - len(non_empty)} rows with no title/abstract text.")
        df = df.iloc[non_empty].reset_index(drop=True)
        texts = [texts[i] for i in non_empty]

    use_case_text = get_use_case_text(df, args.use_case_col, args.use_case_text)
    print(f"  {len(df)} papers. Use case text: \"{use_case_text}\"")

    has_labels = args.label_col in df.columns
    if has_labels:
        y_mask = df[args.label_col].isin(POSITIVE_VALUES | NEGATIVE_VALUES).to_numpy()
        y_full = df[args.label_col].isin(POSITIVE_VALUES).astype(int).to_numpy()
        n_pos, n_neg = int(y_full[y_mask].sum()), int((y_mask.sum() - y_full[y_mask].sum()))
        print(f"  Labels: {n_pos} positive, {n_neg} negative (column '{args.label_col}')")
    else:
        y_mask, y_full = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=int)
        print(f"  No '{args.label_col}' column found — plots will skip label colouring and ROC-AUC.")

    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4.6 * n_models), squeeze=False)

    rows = []
    for row_idx, model_name in enumerate(model_names):
        print(f"\nEmbedding with {model_name} ...")
        try:
            vectors, elapsed = embed_with_model(model_name, texts + [use_case_text])
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"model": model_name, "dim": None, "embed_seconds": None,
                         "avg_pairwise_cosine": None, "participation_ratio": None,
                         "top1_variance_ratio": None, "use_case_to_centroid_sim": None,
                         "use_case_centroid_percentile": None, "roc_auc": None, "n_folds": 0,
                         "note": f"embedding failed: {exc}"})
            for ax in axes[row_idx]:
                ax.set_title(f"{model_name}\nFAILED: {exc}", fontsize=8, color="red")
            continue

        paper_vectors, use_case_vec = vectors[:-1], vectors[-1]

        disp = dispersion_metrics(paper_vectors)
        cent = centroid_analysis(paper_vectors, use_case_vec)

        roc = {"roc_auc": None, "n_folds": 0}
        if has_labels and y_mask.sum() >= 6:
            roc = cross_validated_roc_auc(paper_vectors[y_mask], y_full[y_mask])

        coords_2d, uc_xy, centroid_xy, var_ratio = project_2d(
            paper_vectors, use_case_vec, cent["centroid"], args.projection
        )
        plot_model_row(fig, axes[row_idx], model_name, df, args.label_col,
                        coords_2d, uc_xy, centroid_xy, var_ratio, disp, cent, roc)

        rows.append({
            "model": model_name,
            "dim": paper_vectors.shape[1],
            "embed_seconds": round(elapsed, 2),
            "avg_pairwise_cosine": round(disp["avg_pairwise_cosine"], 4),
            "participation_ratio": round(disp["participation_ratio"], 2),
            "top1_variance_ratio": round(disp["top1_variance_ratio"], 4),
            "use_case_to_centroid_sim": round(cent["use_case_to_centroid_sim"], 4),
            "use_case_centroid_percentile": round(cent["use_case_centroid_percentile"], 1),
            "roc_auc": round(roc["roc_auc"], 4) if roc["roc_auc"] is not None else None,
            "n_folds": roc["n_folds"],
            "note": "",
        })
        print(f"  dim={paper_vectors.shape[1]}  avg_pairwise_cosine={disp['avg_pairwise_cosine']:.3f} "
              f"({interpret_dispersion(disp['avg_pairwise_cosine'])})  "
              f"use_case_percentile={cent['use_case_centroid_percentile']:.0f}  "
              f"roc_auc={roc['roc_auc']}")

    fig.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"\nSaved comparison figure to {args.out_plot}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SCALAR METRICS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out, index=False)
    print(f"\nSaved metrics table to {args.out}")


if __name__ == "__main__":
    main()
