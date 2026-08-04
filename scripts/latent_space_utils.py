"""
latent_space_utils.py — shared latent-space diagnostics + plotting behind BOTH
compare_embeddings.py and compare_ner_models.py.

Extracted out of compare_embeddings.py so a second script (NER-derived representations
instead of sentence embeddings) can reuse the exact same dense/disperse, use-case-centroid,
and 2D-map machinery instead of a second copy quietly drifting apart — same reasoning as
embedding_utils.py already gives for the embedding-specific logic. Nothing in here is
specific to embeddings OR NER: it's "given some vectors (any representation) plus a
use-case vector in the same space, produce the same diagnostics" — see compare_embeddings.py
or compare_ner_models.py's own docstring for what each metric means and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

LABEL_COLORS = {
    "positive": "#2ca02c",
    "negative": "#d62728",
    "pass": "#9467bd",
}
UNLABELLED_COLOR = "#c7c7c7"


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rows(V: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def dispersion_metrics(V: np.ndarray) -> dict:
    """The dense-vs-disperse diagnostics: pairwise cosine similarities (full
    distribution, for the histogram, plus their mean) and PCA-based effective
    dimensionality (see compare_embeddings.py's docstring for what each number means —
    the same reading applies regardless of what produced the vectors)."""
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
    """How central the use case's vector is relative to the corpus — see
    compare_embeddings.py's docstring for the use_case_to_centroid_sim / percentile
    explanation."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — one row per model/representation: [2D map | dispersion hist | centroid hist]
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


def plot_model_row(axes_row, model_name: str, df: pd.DataFrame, label_col: str,
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
