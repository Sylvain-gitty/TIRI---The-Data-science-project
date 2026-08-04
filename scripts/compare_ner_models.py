"""
compare_ner_models.py — the same latent-space / cosine-similarity comparison as
compare_embeddings.py (read that script's docstring first, this one assumes it), but for
NER-DERIVED representations instead of sentence embeddings.

SCOPE (same as compare_embeddings.py): a diagnostic exercise against an already-labelled
dataset — latent-space sanity, cosine similarity, use-case centrality — not a
classifier-baseline-selection pipeline. Nothing downstream in this repo is wired to
whatever "wins" a run of this script.

WHY spaCy, WHY NER-DERIVED VECTORS AT ALL
------------------------------------------------
HANDOFF.md's "where this is heading next" flagged NER as a way to pull structured
metadata (organisms, techniques, locations, ...) out of title/abstract text that isn't a
column in the export today. Before treating NER output as a feature source, this script
asks the same question compare_embeddings.py asks of sentence embeddings: does turning a
paper into a NER-derived vector produce a SANE latent space at all (papers spread out by
content, not collapsed), and does the use case's own text land somewhere sensible in that
space relative to the corpus it was used to find?

spaCy (en_core_web_sm by default) was picked over GLiNER/scispaCy for this first pass:
fully local/offline, no PyTorch dependency for the small model, mature and fast on CPU —
the same "local, no API key, CPU throughput" priorities compare_embeddings.py's model
shortlist (reports/model_shortlist.md §1) used for embeddings. The tradeoff, stated up
front and honestly: en_core_web_sm's entity types (PERSON, ORG, GPE, DATE, ...) are
GENERIC-purpose, not domain-trained — it will NOT recognise organism/species names,
chemical compounds, or lab techniques as anything other than generic PERSON/ORG/nothing.
See "A LIMITATION TO KNOW ABOUT" below for what testing this actually found, and
reports/ner_model_notes.md for the fuller candidate-comparison writeup (what a
domain-specific alternative like scispaCy or GLiNER-with-custom-labels would buy instead).

TWO REPRESENTATIONS COMPARED (mirrors compare_embeddings.py running several models)
--------------------------------------------------------------------------------------
1. entity_type_counts — per paper, a vector of HOW MANY entities of each spaCy label
   (ORG, GPE, DATE, PERSON, ...) were found, L1-normalised so it reads as "what mix of
   entity types does this paper mention" rather than being dominated by abstract length.
   Tests whether papers cluster/separate by entity-TYPE profile.
2. entity_text_tfidf — per paper, TF-IDF (+ SVD, same recipe as embedding_utils.py's
   "tfidf" backend) over the actual entity TEXT spans extracted (e.g. "University of
   Leeds", "2024", "Yorkshire"), not the type labels. Tests whether papers cluster/
   separate by WHICH specific entities recur across the corpus.

Both get the exact same dispersion / centroid-percentile / (secondary) roc_auc
diagnostics and the same 3-panel-per-row plot as compare_embeddings.py, via the shared
scripts/latent_space_utils.py module (extracted from compare_embeddings.py so both
scripts share identical diagnostic math instead of two copies quietly drifting apart).

A LIMITATION TO KNOW ABOUT
-----------------------------
Papers with no abstract text, or where spaCy finds no entities at all, get an all-zero
vector for entity_type_counts (cosine similarity 0 against everything — reads as
"maximally different", which isn't really true, just "nothing found"). The script prints
how many papers that affected per run; a high count means this representation isn't
really being tested on your corpus. en_core_web_sm's general-purpose entity types are
also a known mismatch for domain scientific text — soil-microbiome or climate abstracts
mostly surface ORG/GPE/DATE/CARDINAL, not anything like "organism" or "technique" (those
label types don't exist in this model at all) — a real, if unflattering, first result to
report honestly, not a bug to tune away.

TESTED, RESULT: on the climate/agriculture sample export (100 papers), 20/100 papers
produced zero entities, AND the export's own use_case text ("Climate change extremes and
agriculture practices") also produced zero entities — a short use-case NAME (see
embedding_utils.get_use_case_text's docstring) is exactly the kind of generic phrase
general-purpose NER isn't built to tag. That collapses use_case_to_centroid_sim to
exactly 0.0 for BOTH representations on this corpus — not a bug (the zero-vector handling
in latent_space_utils.centroid_analysis is deliberate), but a genuine finding: as tested,
NER-derived "how central is the use case" is not a meaningful question for a use-case
NAME this short, whatever the corpus. It might be more meaningful with the agent's
fuller objective/key-terms wording (--use-case-text) — untested here, don't assume.

METRICS REWORK — see compare_embeddings.py's "THE METRICS" docstring section for the
full explanation (this script reuses the exact same evaluate_representation/
centroid_analysis machinery): roc_auc is now roc_auc_strict/roc_auc_conservative (two
label readings, plus roc_auc_std) and comes with a query-conditioned counterpart
(query_auc_strict/conservative, recall_at_Xpct, wss_at_95) that this script's ROC-AUC
never had — the classifier version is fit on paper vectors alone and never looks at the
use-case vector at all, which matters most exactly here: a NER representation whose
use-case vector collapsed to all-zero (see above) could still, in principle, score a
normal-looking roc_auc, since that metric wouldn't have noticed the collapse either way.

HOW TO RUN IT
--------------
Preferred: open notebooks/comparisons/run_comparisons.ipynb (from inside
notebooks/comparisons/) and run all cells — output renders inline, nothing written to
disk. Run this script directly only for scripting/automation:

    python scripts/compare_ner_models.py --data data/raw/your-export.parquet

Prints the scalar-metrics table to the console. No file is written.

Useful flags:
    --spacy-model en_core_web_md          use a different local spaCy pipeline
    --representations entity_type_counts  run one representation instead of both
    --label-col review_label              use the review-stage label instead of triage
    --use-case-text "..."                 override the export's short use_case name
    --projection tsne                     use t-SNE instead of PCA for the 2D map
    --list-entity-labels                  print the loaded spaCy model's entity types
    --out path.csv                        ALSO write the metrics table to this CSV
    --out-plot path.png                   ALSO build + write the comparison figure

OUTPUTS: none, by default — see compare_embeddings.py's docstring for why (short version:
reports/ holds decision-trail .md files in this repo, not a CSV/PNG per run).
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from embedding_utils import (
    build_label_masks,
    build_paper_texts,
    drop_empty_rows,
    evaluate_representation,
    get_title_abstract,
    get_use_case_text,
    load_export,
    round_floats,
)
from latent_space_utils import (
    centroid_analysis,
    dispersion_metrics,
    interpret_dispersion,
    plot_model_row,
    project_2d,
)

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_REPRESENTATIONS = ["entity_type_counts", "entity_text_tfidf"]

# NER-only pipes needed — tagger/parser/attribute_ruler/lemmatizer don't feed spaCy's NER
# component and only slow down nlp.pipe() for no benefit here.
_PIPES_TO_DISABLE = ["tagger", "parser", "attribute_ruler", "lemmatizer"]


# ─────────────────────────────────────────────────────────────────────────────
# spaCy loading + entity extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_spacy_model(name: str):
    import spacy
    try:
        nlp = spacy.load(name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{name}' isn't installed locally. Run:\n"
            f"    python -m spacy download {name}\n"
            f"(original error: {exc})"
        )
    to_disable = [p for p in _PIPES_TO_DISABLE if p in nlp.pipe_names]
    if to_disable:
        nlp.select_pipes(disable=to_disable)
    return nlp


def extract_entities(nlp, texts: list[str]) -> list[list[tuple[str, str]]]:
    """Run NER over every text, batched via nlp.pipe for throughput. Returns, per text, a
    list of (entity_text, entity_label) pairs — empty list if nothing was found."""
    return [[(ent.text, ent.label_) for ent in doc.ents] for doc in nlp.pipe(texts)]


# ─────────────────────────────────────────────────────────────────────────────
# Representation 1: entity type-count profile
# ─────────────────────────────────────────────────────────────────────────────

def entity_type_count_vectors(ents_lists: list[list[tuple[str, str]]], label_vocab: list[str]) -> np.ndarray:
    """One row per doc: count of each entity TYPE in label_vocab, L1-normalised so the
    vector reads as a type-mix profile rather than being dominated by document length.
    Docs with zero entities get an all-zero row — see module docstring's limitation."""
    label_index = {label: i for i, label in enumerate(label_vocab)}
    counts = np.zeros((len(ents_lists), len(label_vocab)))
    for row, ents in enumerate(ents_lists):
        for _, label in ents:
            idx = label_index.get(label)
            if idx is not None:
                counts[row, idx] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals


# ─────────────────────────────────────────────────────────────────────────────
# Representation 2: TF-IDF over the extracted entity text spans
# ─────────────────────────────────────────────────────────────────────────────

def entity_text_pseudo_docs(ents_lists: list[list[tuple[str, str]]]) -> list[str]:
    """One pseudo-document per doc: every entity TEXT span found, lowercased and
    space-joined (duplicates kept deliberately — a repeated entity mention is a real
    repetition signal for TF-IDF, not noise to dedupe away)."""
    return [" ".join(text.lower() for text, _ in ents) for ents in ents_lists]


def tfidf_embed(corpus_texts: list[str], query_text: str) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF + SVD over corpus texts AND the use-case query jointly (same recipe as
    embedding_utils.py's "tfidf" backend), so both land in the same reduced space from one
    fit rather than the query being projected into a space it didn't help define."""
    all_texts = corpus_texts + [query_text]
    vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
    sparse = vectorizer.fit_transform(all_texts)
    n_components = max(2, min(100, sparse.shape[0] - 1, sparse.shape[1] - 1))
    vectors = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(sparse)
    return vectors[:-1], vectors[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch: representation name -> paper_vectors, use_case_vector, elapsed, extra
# ─────────────────────────────────────────────────────────────────────────────

def embed_representation(
    name: str, nlp, label_vocab: list[str], titles: list[str], abstracts: list[str], use_case_text: str
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    paper_texts = build_paper_texts(titles, abstracts, ". ")
    start = time.monotonic()
    ents_lists = extract_entities(nlp, paper_texts)
    use_case_ents = extract_entities(nlp, [use_case_text])[0]

    if name == "entity_type_counts":
        paper_vectors = entity_type_count_vectors(ents_lists, label_vocab)
        use_case_vector = entity_type_count_vectors([use_case_ents], label_vocab)[0]
        n_zero = sum(1 for ents in ents_lists if not ents)
    elif name == "entity_text_tfidf":
        corpus_pseudo_docs = entity_text_pseudo_docs(ents_lists)
        query_pseudo_doc = entity_text_pseudo_docs([use_case_ents])[0]
        paper_vectors, use_case_vector = tfidf_embed(corpus_pseudo_docs, query_pseudo_doc)
        n_zero = sum(1 for t in corpus_pseudo_docs if not t.strip())
    else:
        raise ValueError(f"Unknown representation: {name!r} (expected one of {DEFAULT_REPRESENTATIONS})")

    elapsed = time.monotonic() - start
    return paper_vectors, use_case_vector, elapsed, {"n_zero_entity_papers": n_zero}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to an academic_research_agent .parquet or .csv export")
    parser.add_argument("--label-col", default="triage_label", help="Column with positive/negative/pass labels (default: triage_label)")
    parser.add_argument("--use-case-col", default="use_case", help="Column holding the use case text (default: use_case)")
    parser.add_argument("--use-case-text", default=None, help="Override the use case text instead of reading --use-case-col")
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL, help=f"Local spaCy pipeline to run NER with (default: {DEFAULT_SPACY_MODEL})")
    parser.add_argument("--representations", default=None, help="Comma-separated representation names (default: entity_type_counts,entity_text_tfidf)")
    parser.add_argument("--projection", choices=["pca", "tsne"], default="pca", help="2D projection method for the corpus map (default: pca)")
    parser.add_argument("--out", type=Path, default=None, help="Save the scalar metrics table to this CSV path (default: not saved — printed to the console only)")
    parser.add_argument("--out-plot", type=Path, default=None, help="Save the comparison figure to this PNG path (default: not built/saved at all — see notebooks/comparisons/ for an inline alternative)")
    parser.add_argument("--list-entity-labels", action="store_true", help="Print the loaded spaCy model's entity label set, then exit")
    args = parser.parse_args()

    if args.list_entity_labels:
        nlp = load_spacy_model(args.spacy_model)
        print(f"Entity labels in {args.spacy_model}:")
        for label in nlp.get_pipe("ner").labels:
            print(f"  {label}")
        return

    if args.data is None:
        parser.error("--data is required unless --list-entity-labels is passed")

    # Nothing is written to disk unless explicitly asked (--out/--out-plot) — reports/
    # holds this repo's decision-trail .md files, not a per-run CSV/PNG pile. See
    # notebooks/comparisons/run_comparisons.ipynb for the inline-output equivalent.
    make_plot = args.out_plot is not None

    representation_names = (
        [r.strip() for r in args.representations.split(",")] if args.representations else DEFAULT_REPRESENTATIONS
    )

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    titles, abstracts = get_title_abstract(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)

    use_case_text = get_use_case_text(df, args.use_case_col, args.use_case_text)
    print(f"  {len(df)} papers. Use case text: \"{use_case_text}\"")

    print(f"Loading spaCy model {args.spacy_model} ...")
    nlp = load_spacy_model(args.spacy_model)
    label_vocab = list(nlp.get_pipe("ner").labels)

    masks = build_label_masks(df, args.label_col)
    if masks["has_labels"]:
        print(f"  Labels: {masks['n_pos']} positive, {masks['n_neg']} negative, "
              f"{masks['n_pass']} pass (column '{args.label_col}')")
    else:
        print(f"  No '{args.label_col}' column found — plots will skip label colouring and evaluation metrics.")

    n_reps = len(representation_names)
    if make_plot:
        fig, axes = plt.subplots(n_reps, 3, figsize=(15, 4.6 * n_reps), squeeze=False)

    rows = []
    for row_idx, rep_name in enumerate(representation_names):
        print(f"\nBuilding representation '{rep_name}' ...")
        try:
            paper_vectors, use_case_vec, elapsed, extra = embed_representation(
                rep_name, nlp, label_vocab, titles, abstracts, use_case_text
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"representation": rep_name, "dim": None, "seconds": None,
                         "note": f"failed: {exc}"})
            if make_plot:
                for ax in axes[row_idx]:
                    ax.set_title(f"{rep_name}\nFAILED: {exc}", fontsize=8, color="red")
            continue

        disp = dispersion_metrics(paper_vectors)
        cent = centroid_analysis(paper_vectors, use_case_vec, masks["pos_mask"], masks["neg_mask"])
        metrics = evaluate_representation(paper_vectors, use_case_vec, masks)

        if make_plot:
            coords_2d, uc_xy, centroid_xy, var_ratio = project_2d(
                paper_vectors, use_case_vec, cent["centroid"], args.projection
            )
            plot_model_row(axes[row_idx], rep_name, df, args.label_col,
                            coords_2d, uc_xy, centroid_xy, var_ratio, disp, cent, metrics)

        rows.append({
            "representation": rep_name,
            "dim": paper_vectors.shape[1],
            "seconds": round(elapsed, 2),
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
            "n_zero_entity_papers": extra["n_zero_entity_papers"],
            "note": "",
        })
        print(f"  dim={paper_vectors.shape[1]}  avg_pairwise_cosine={disp['avg_pairwise_cosine']:.3f} "
              f"({interpret_dispersion(disp['avg_pairwise_cosine'])})  "
              f"use_case_percentile={cent['use_case_centroid_percentile']:.0f}  "
              f"roc_auc_strict={metrics['roc_auc_strict']}  "
              f"query_auc_strict={metrics['query_auc_strict']}  "
              f"zero_entity_papers={extra['n_zero_entity_papers']}/{len(df)}")

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
