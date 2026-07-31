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

THE 4 MODELS SHORTLISTED FOR THIS ROUND (see reports/model_shortlist.md for the full
reasoning behind each pick) — this is what DEFAULT_MODELS below actually is:
    1. sentence-transformers/allenai-specter   — domain-specific: trained on academic
       paper citation pairs, the closest match to this exact task of anything available.
    2. BAAI/bge-small-en-v1.5                  — re-run WITH its required query prefix
       this time (see PREFIXES below) — last round's run of this model, unprefixed,
       looked collapsed; this isolates whether the prefix was the actual problem.
    3. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — the CONTROL: this is
       academic_research_agent's current default model, unchanged, so every other row has
       something fixed to compare against.
    4. sentence-transformers/all-MiniLM-L6-v2  — the tiny/fast floor: is anything above
       actually earning the extra size over the smallest reasonable model?

(nomic-ai/nomic-embed-text-v1.5 was in this shortlist and has been REMOVED after testing:
on a 100-paper corpus it took ~795s to embed vs. 4-85s for every other model here — on
CPU, fastembed's ONNX graph for this model is dramatically slower than the rest, which
outweighs what it offered (Matryoshka dims) for a script meant to be re-run often while
iterating. Still registered in MODEL_CONFIGS if you want to bring it back via --models.)

PREFIXES — WHY SOME TEXT GETS A PREFIX BEFORE EMBEDDING
------------------------------------------------------------
bge and nomic are *asymmetric* models: they were trained expecting the query and the
documents to be marked differently in the input text (e.g. bge wants
"Represent this sentence for searching relevant passages: " glued in front of a QUERY,
never a document). MODEL_CONFIGS below is the one place that records, per model, which
prefix (if any) goes on the use-case text ("query_prefix") vs. every paper's
title+abstract ("passage_prefix"). Symmetric models (specter, the multilingual control,
all-MiniLM-L6-v2) get empty strings for both — they were trained to treat query and
document text identically, so adding a prefix would only introduce noise they weren't
trained to ignore.

CORRECTION FROM THE PREVIOUS ROUND, TESTED: bge's own convention prefixes ONLY the
query, never the passages — so re-running BAAI/bge-small-en-v1.5 with the query prefix
added did NOT change its corpus-wide dispersion at all (avg pairwise cosine: 0.7692
before, 0.7692 after — identical, because every PAPER embedding was already unprefixed
both times). The earlier hypothesis ("the collapse was probably a missing-prefix
artifact") is therefore NOT confirmed for the corpus itself: bge-small's document
embeddings really do look collapsed on this corpus, prefix or no prefix. What the prefix
DID change was the use-case vector specifically (use_case_to_centroid_sim: 0.85 -> 0.83,
percentile: 22nd -> 7th) — worth knowing, but a different, smaller effect than originally
guessed. Report what you actually see per run rather than assuming this pattern repeats
on a different corpus or a different asymmetric model.

TITLE_ABSTRACT_SEP — the same "wrong input shape" problem, for text formatting
------------------------------------------------------------------------------------
Specter's own model card documents joining title and abstract with the tokenizer's
[SEP] token, not free-form prose ("title[SEP]abstract", not "title. abstract"). The
first live run of this script fed Specter "title. abstract" like every other model —
and Specter came back with the densest corpus of all (avg pairwise cosine 0.80) and a
ROC-AUC of 0.49 (no better than chance), despite being the one model actually trained
for this domain. That looked like an input-shape artifact worth fixing, so
MODEL_CONFIGS' title_abstract_sep now gives Specter its documented "[SEP]" join.

TESTED, RESULT: fixing the separator barely moved either number (avg pairwise cosine
0.80 -> 0.81, ROC-AUC 0.4947 -> 0.4996) — on the climate/agriculture test corpus,
Specter's collapsed, chance-level result was NOT a formatting artifact. Whether that
holds on a different corpus (a better domain match, or a bigger one) is untested —
don't assume the verdict on Specter transfers; re-run per corpus.

MODEL BACKENDS — fastembed vs. sentence-transformers
---------------------------------------------------------
4 of these 5 are in fastembed's own catalogue (local, ONNX, no PyTorch). Specter is not
(fastembed simply doesn't ship it), so it runs through the `sentence-transformers`
package instead — still fully local/offline, just a heavier dependency (pulls in
PyTorch). MODEL_CONFIGS' "backend" field is what routes each model name to the right
loader; unregistered model names default to fastembed with a plain ". " join (so you can
still try any fastembed model ad hoc via --models without editing this file), and the
"tfidf" special case runs neither.

HOW TO RUN IT
--------------
    python scripts/compare_embeddings.py --data data/raw/your-export.parquet

That one command runs all 4 shortlisted models above — no extra flags needed.

Useful flags:
    --models "tfidf,BAAI/bge-small-en-v1.5"     run a different set instead of the 4 above
    --label-col review_label                    use the review-stage label instead
    --use-case-text "..."                       override the export's short use_case name
    --projection tsne                           use t-SNE instead of PCA for the 2D map
    --list-models                               print fastembed's full model catalogue
                                                 (+ this script's own MODEL_CONFIGS)
    --out / --out-plot                          override the default output paths below

OUTPUTS (one pair of files per export, so different datasets never overwrite each other)
------------------------------------------------------------------------------------------
    reports/<data filename>_latent_space_comparison.png   the visual comparison (main output)
    reports/<data filename>_embedding_comparison.csv      the scalar metrics behind the plot

A LIMITATION TO KNOW ABOUT
-----------------------------
All of this is descriptive of ONE corpus (typically a few hundred papers at most). With
a small n, the PCA map and histograms show real structure but the scalar metrics have
wide error bars — read them as "which model looks meaningfully different", not as
precise measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — this is a batch script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embedding_utils import (
    MODEL_CONFIGS,
    NEGATIVE_VALUES,
    POSITIVE_VALUES,
    cross_validated_roc_auc,
    drop_empty_rows,
    embed_corpus_and_use_case,
    get_title_abstract,
    get_use_case_text,
    load_export,
)

# Run these 4 shortlisted models by a single bare command; --models overrides this.
# (nomic-ai/nomic-embed-text-v1.5 is registered in embedding_utils.MODEL_CONFIGS but not
# listed here — see that module's comment for why: ~795s to embed vs. 4-85s for these 4.)
DEFAULT_MODELS = [
    "sentence-transformers/allenai-specter",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

LABEL_COLORS = {
    "positive": "#2ca02c",
    "negative": "#d62728",
    "pass": "#9467bd",
}
UNLABELLED_COLOR = "#c7c7c7"


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
    parser.add_argument("--models", default=None, help="Comma-separated model names/types (default: the 4 shortlisted models, see --list-models)")
    parser.add_argument("--projection", choices=["pca", "tsne"], default="pca", help="2D projection method for the corpus map (default: pca)")
    parser.add_argument("--out", type=Path, default=None, help="Where to save the scalar metrics table (default: reports/<data filename>_embedding_comparison.csv)")
    parser.add_argument("--out-plot", type=Path, default=None, help="Where to save the comparison figure (default: reports/<data filename>_latent_space_comparison.png)")
    parser.add_argument("--list-models", action="store_true", help="Print every model fastembed supports, then exit")
    args = parser.parse_args()

    if args.list_models:
        print("Models registered in THIS script (MODEL_CONFIGS) - the shortlist:")
        for name, cfg in MODEL_CONFIGS.items():
            marker = " <- default set" if name in DEFAULT_MODELS else ""
            print(f"  {name}  [{cfg['backend']}]{marker}")
        print("\nEvery model fastembed itself supports (any of these also work via --models,")
        print("unprefixed, unless you add it to MODEL_CONFIGS first):")
        from fastembed import TextEmbedding
        for m in TextEmbedding.list_supported_models():
            print(f"  {m['model']}  (dim={m['dim']})")
        return

    if args.data is None:
        parser.error("--data is required unless --list-models is passed")

    # Default output paths are derived from the input filename so results from different
    # exports land in different files instead of silently overwriting each other.
    if args.out is None:
        args.out = Path("reports") / f"{args.data.stem}_embedding_comparison.csv"
    if args.out_plot is None:
        args.out_plot = Path("reports") / f"{args.data.stem}_latent_space_comparison.png"

    model_names = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    titles, abstracts = get_title_abstract(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)

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
            paper_vectors, use_case_vec, elapsed = embed_corpus_and_use_case(model_name, titles, abstracts, use_case_text)
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
