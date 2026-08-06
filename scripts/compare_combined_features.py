"""
compare_combined_features.py — does concatenating an NER-derived block onto an embedding
block change the latent-space diagnostics, versus either one standalone?

SCOPE (same as compare_embeddings.py / compare_ner_models.py): a diagnostic exercise
against an already-labelled dataset — dispersion, cosine similarity, use-case centrality
— not a classifier-baseline-selection pipeline. Nothing downstream in this repo is wired
to whatever this script reports.

WHY THIS SCRIPT
------------------
compare_ner_models.py found NER-derived features score close to chance standalone
(ROC-AUC 0.55-0.62) — meaningfully weaker than every embedding model tested in
compare_embeddings.py (0.65-0.80, see reports/model_shortlist.md §4). But "worse alone"
doesn't answer "worse as an ADDED signal" — a survey of open-source paper-scoring tools
(cross-referenced against this repo in HANDOFF.md) found every multi-stage tool in that
survey wins by combining a cheap signal with a precise one, not by picking one winner
standalone (e.g. Paper-QA-RAG-LoRA's cross-encoder re-ranking stage moved hit@5 from
0.776 to 0.928 over bi-encoder retrieval alone). This script tests that directly:
concatenate an embedding vector with an NER-derived vector for every paper, run the EXACT
same dispersion/centroid/roc_auc diagnostics as the other two comparison scripts (via the
shared latent_space_utils.py), and see whether the combined space looks any different
from either block alone.

THE ONE REAL DESIGN DECISION: block weighting
--------------------------------------------------
An embedding vector (384-dim, say) and an entity_type_counts vector (18-dim) have very
different raw magnitudes. Naive np.hstack would let whichever block happens to have the
larger norm dominate every downstream cosine similarity — silently, not because either
block is more informative. Fix: L2-normalise each block to unit length INDEPENDENTLY
(latent_space_utils.normalize_rows) before concatenating, so both blocks contribute
comparably by construction. This is a deliberate 50/50 weighting choice, not a tuned
hyperparameter — an --embedding-weight flag is the natural follow-up if this looks
promising, not attempted here.

DEFAULT EMBEDDING MODEL
---------------------------
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 by default — chosen only
because it's the model both existing exports ship precomputed vectors for. A convenience
default, NOT a pick informed by compare_embeddings.py's results (this repo already
retired that framing — see HANDOFF.md). Override with --embedding-model to test any
other registered model (see embedding_utils.MODEL_CONFIGS).

TWO NER REPRESENTATIONS, SAME AS compare_ner_models.py
------------------------------------------------------------
entity_type_counts and entity_text_tfidf — see that script's docstring for what each
means. Each is combined with the SAME embedding block in its own row, so the two rows
answer "does combining help THIS NER representation" independently.

HOW TO RUN IT
--------------
Preferred: open notebooks/comparisons/run_comparisons.ipynb (from inside
notebooks/comparisons/) and run all cells — output renders inline, nothing written to
disk. Run this script directly only for scripting/automation:

    python scripts/compare_combined_features.py --data data/raw/your-export.parquet

Prints the scalar-metrics table to the console. No file is written.

Useful flags:
    --embedding-model "BAAI/bge-small-en-v1.5"   use a different embedding model
    --ner-representations entity_type_counts     run one NER representation instead of both
    --spacy-model en_core_web_sm                 local spaCy pipeline for the NER block
    --use-case-text "..."                        override the export's short use_case name
    --label-col review_label                     use the review-stage label instead
    --projection tsne                            use t-SNE instead of PCA for the 2D map
    --out path.csv                               ALSO write the metrics table to this CSV
    --out-plot path.png                          ALSO build + write the comparison figure

OUTPUTS: none, by default — see compare_embeddings.py's docstring for why (short version:
reports/ holds decision-trail .md files in this repo, not a CSV/PNG per run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — this is a batch script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embedding_utils import (
    build_label_masks,
    drop_empty_rows,
    embed_corpus_and_use_case,
    evaluate_representation,
    get_title_abstract,
    get_use_case_text,
    load_export,
    round_floats,
)
from compare_ner_models import DEFAULT_SPACY_MODEL, embed_representation, load_spacy_model
from latent_space_utils import (
    centroid_analysis,
    dispersion_metrics,
    interpret_dispersion,
    normalize_rows,
    plot_model_row,
    project_2d,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_NER_REPRESENTATIONS = ["entity_type_counts", "entity_text_tfidf"]


# ─────────────────────────────────────────────────────────────────────────────
# Block concatenation
# ─────────────────────────────────────────────────────────────────────────────

def concat_blocks(
    embedding_papers: np.ndarray, embedding_use_case: np.ndarray,
    ner_papers: np.ndarray, ner_use_case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalise each block independently, then concatenate — see module docstring's
    "block weighting" section for why. Returns (combined_paper_vectors, combined_use_case_vector)."""
    paper_vectors = np.hstack([normalize_rows(embedding_papers), normalize_rows(ner_papers)])
    use_case_vector = np.hstack([
        normalize_rows(embedding_use_case[None, :])[0],
        normalize_rows(ner_use_case[None, :])[0],
    ])
    return paper_vectors, use_case_vector


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to an academic_research_agent .parquet or .csv export")
    parser.add_argument("--label-col", default="triage_label", help="Column with positive/negative/pass labels (default: triage_label)")
    parser.add_argument("--use-case-col", default="use_case", help="Column holding the use case text (default: use_case)")
    parser.add_argument("--use-case-text", default=None, help="Override the use case text instead of reading --use-case-col")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help=f"Embedding model for the embedding block (default: {DEFAULT_EMBEDDING_MODEL} — a convenience default, not a pick, see module docstring)")
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL, help=f"Local spaCy pipeline for the NER block (default: {DEFAULT_SPACY_MODEL})")
    parser.add_argument("--ner-representations", default=None, help="Comma-separated NER representation names (default: entity_type_counts,entity_text_tfidf)")
    parser.add_argument("--projection", choices=["pca", "tsne"], default="pca", help="2D projection method for the corpus map (default: pca)")
    parser.add_argument("--out", type=Path, default=None, help="Save the scalar metrics table to this CSV path (default: not saved — printed to the console only)")
    parser.add_argument("--out-plot", type=Path, default=None, help="Save the comparison figure to this PNG path (default: not built/saved at all — see notebooks/comparisons/ for an inline alternative)")
    args = parser.parse_args()

    # Nothing is written to disk unless explicitly asked (--out/--out-plot) — reports/
    # holds this repo's decision-trail .md files, not a per-run CSV/PNG pile. See
    # notebooks/comparisons/run_comparisons.ipynb for the inline-output equivalent.
    make_plot = args.out_plot is not None

    ner_representation_names = (
        [r.strip() for r in args.ner_representations.split(",")] if args.ner_representations else DEFAULT_NER_REPRESENTATIONS
    )

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    titles, abstracts = get_title_abstract(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)

    use_case_text = get_use_case_text(df, args.use_case_col, args.use_case_text)
    print(f"  {len(df)} papers. Use case text: \"{use_case_text}\"")

    print(f"Embedding with {args.embedding_model} ...")
    embedding_papers, embedding_use_case, embed_elapsed = embed_corpus_and_use_case(
        args.embedding_model, titles, abstracts, use_case_text
    )
    print(f"  dim={embedding_papers.shape[1]}, {embed_elapsed:.1f}s")

    print(f"Loading spaCy model {args.spacy_model} ...")
    nlp = load_spacy_model(args.spacy_model)
    label_vocab = list(nlp.get_pipe("ner").labels)

    masks = build_label_masks(df, args.label_col)
    if masks["has_labels"]:
        print(f"  Labels: {masks['n_pos']} positive, {masks['n_neg']} negative, "
              f"{masks['n_pass']} pass (column '{args.label_col}')")
    else:
        print(f"  No '{args.label_col}' column found — plots will skip label colouring and evaluation metrics.")

    n_rows = len(ner_representation_names)
    if make_plot:
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.6 * n_rows), squeeze=False)

    rows = []
    for row_idx, ner_rep_name in enumerate(ner_representation_names):
        row_label = f"{args.embedding_model.split('/')[-1]} + {ner_rep_name}"
        print(f"\nBuilding combined representation '{row_label}' ...")
        try:
            ner_papers, ner_use_case, ner_elapsed, extra = embed_representation(
                ner_rep_name, nlp, label_vocab, titles, abstracts, use_case_text
            )
            paper_vectors, use_case_vec = concat_blocks(embedding_papers, embedding_use_case, ner_papers, ner_use_case)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"combination": row_label, "dim": None, "seconds": None,
                         "note": f"failed: {exc}"})
            if make_plot:
                for ax in axes[row_idx]:
                    ax.set_title(f"{row_label}\nFAILED: {exc}", fontsize=8, color="red")
            continue

        disp = dispersion_metrics(paper_vectors)
        cent = centroid_analysis(paper_vectors, use_case_vec, masks["pos_mask"], masks["neg_mask"])
        metrics = evaluate_representation(paper_vectors, use_case_vec, masks)

        if make_plot:
            coords_2d, uc_xy, centroid_xy, var_ratio = project_2d(
                paper_vectors, use_case_vec, cent["centroid"], args.projection
            )
            plot_model_row(axes[row_idx], row_label, df, args.label_col,
                            coords_2d, uc_xy, centroid_xy, var_ratio, disp, cent, metrics)

        rows.append({
            "combination": row_label,
            "dim": paper_vectors.shape[1],
            "seconds": round(embed_elapsed + ner_elapsed, 2),
            **round_floats({
                "avg_pairwise_cosine": disp["avg_pairwise_cosine"],
                "participation_ratio": disp["participation_ratio"],
                "top1_variance_ratio": disp["top1_variance_ratio"],
                "use_case_to_centroid_sim": cent["use_case_to_centroid_sim"],
                "use_case_centroid_percentile": cent["use_case_centroid_percentile"],
                "use_case_to_positive_centroid_sim": cent["use_case_to_positive_centroid_sim"],
                "use_case_to_negative_centroid_sim": cent["use_case_to_negative_centroid_sim"],
                "use_case_discriminative_gap": cent["use_case_discriminative_gap"],
                **metrics,
            }),
            "note": "",
        })
        print(f"  dim={paper_vectors.shape[1]}  avg_pairwise_cosine={disp['avg_pairwise_cosine']:.3f} "
              f"({interpret_dispersion(disp['avg_pairwise_cosine'])})  "
              f"use_case_percentile={cent['use_case_centroid_percentile']:.0f}  "
              f"roc_auc_strict={metrics['roc_auc_strict']}  "
              f"query_auc_strict={metrics['query_auc_strict']}")

    if make_plot:
        fig.tight_layout()
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_plot, dpi=150)
        print(f"\nSaved comparison figure to {args.out_plot}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("SCALAR METRICS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out, index=False)
        print(f"\nSaved metrics table to {args.out}")
    else:
        print("\n(--out not given — nothing written to disk; the table above is the only output.)")


if __name__ == "__main__":
    main()
