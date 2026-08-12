"""
compare_embeddings.py — visually compare the LATENT SPACE of different embedding/NLP
models on your labelled corpus.

SCOPE: this is a diagnostic exercise, not a model-selection pipeline. The goal is to test
how a use-case embedding relates to an already-labelled dataset — latent-space sanity,
cosine similarity, use-case centrality — for its own sake. It does NOT pick a model for
any future classifier baseline (Week 3 or otherwise); nothing downstream in this repo is
wired to whatever "wins" a run of this script. See scripts/compare_ner_models.py for the
same diagnostic exercise applied to NER-derived representations instead of embeddings.

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
    5. (if there are enough triage labels) also reports cross-validated ROC-AUC — a
       secondary "how well does this space separate the labels you already have" reading
       on the labelled dataset, not a signal for picking a future classifier's model.

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
    embedded use case and the corpus's centroid vector (the mean of ALL abstract
    vectors — positive, negative, AND pass pooled together, unweighted by label), and
    what percentile that similarity falls at relative to every abstract's own similarity
    to that same centroid. A high percentile says the use case's wording sits near the
    "centre" of the papers this search retrieved — as central as a typical retrieved
    paper, or more so. A low percentile says the use case is an outlier relative to its
    own retrieved pool, worth a second look at whether the query terms actually match the
    literature's vocabulary. NOTE what this does NOT tell you: because the pool is
    unweighted by label, "typical of everything retrieved" is not the same claim as
    "typical of the papers actually accepted" — see use_case_discriminative_gap below for
    the label-aware version of this same question.

use_case_to_positive_centroid_sim / use_case_to_negative_centroid_sim /
    use_case_discriminative_gap — the label-aware counterpart to the corpus-wide
    centrality above: a separate centroid for positive-labelled rows and one for
    negative-labelled rows, and how similar the use case is to each. The gap (positive
    minus negative) is a single query-aware, label-aware number: positive means the use
    case's own wording sits closer to the papers that got KEPT than to the ones that got
    REJECTED — a claim corpus-wide centrality alone cannot make, since it never looks at
    labels at all. Needs >=2 rows in each class to compute; None below that.

roc_auc_strict / roc_auc_conservative (secondary) — cross-validated LogisticRegression
    AUC predicting the triage label from the embeddings, IDENTICAL to
    academic_research_agent's own model.py:_cross_validated_roc_auc. "strict" = positive
    vs. negative only (pass/unlabelled excluded, the reading this script always reported
    before this rework). "conservative" = positive vs. everything NOT accepted
    (negative + pass) — a harsher second reading that doesn't let the ambiguous middle
    get skipped, since a deployed tool can't skip it either. Both come with roc_auc_std
    (the standard deviation ACROSS folds, not just the mean) so a "0.006 difference is
    noise" claim can be checked against a number instead of asserted. NOTE what this
    metric does NOT measure, in either reading: the classifier is fit on paper vectors
    ALONE — the use-case/query vector never enters this computation. A space can be
    perfectly label-separable here while being useless for ranking-by-similarity-to-a-
    specific-query — see query_auc below for the metric that actually uses the query.

query_auc_strict / query_auc_conservative, recall_at_10pct / recall_at_20pct, wss_at_95
    (secondary, query-conditioned) — rank papers by cosine similarity to the USE-CASE
    VECTOR ITSELF (the actual nearest-neighbour mechanism a retrieval step uses, and the
    thing roc_auc above never looks at), then score that ranking against the labels.
    query_auc is on the same 0.5/1.0 scale as roc_auc but answers a different question
    ("does ranking BY THIS QUERY separate relevant from not" vs. "is this space linearly
    separable by label at all"). recall_at_Xpct: of the top X% of the ranked list, what
    fraction of the true positives an analyst would already have seen. wss_at_95: Work
    Saved over Sampling at 95% recall (Cohen et al. 2006, a standard citation-screening-
    automation metric) — the fraction of the corpus an analyst could skip while still
    catching 95% of the positives, minus the 5% skippable by chance alone.

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
Preferred: open notebooks/experiments/run_comparisons.ipynb (from inside
notebooks/experiments/) and run all cells — every table and plot renders inline, nothing
is written to disk. This script is the library that notebook imports, not a separate
tool; run it directly only for scripting/automation, or to sanity-check a change:

    python scripts/compare_embeddings.py --data data/raw/your-export.parquet

That one command runs all 4 shortlisted models above — no extra flags needed — and prints
the scalar-metrics table to the console. No file is written.

Useful flags:
    --models "tfidf,BAAI/bge-small-en-v1.5"     run a different set instead of the 4 above
    --label-col review_label                    use the review-stage label instead
    --use-case-text "..."                       override the export's short use_case name
    --projection tsne                           use t-SNE instead of PCA for the 2D map
    --list-models                               print fastembed's full model catalogue
                                                 (+ this script's own MODEL_CONFIGS)
    --out path.csv                              ALSO write the metrics table to this CSV
    --out-plot path.png                         ALSO build + write the comparison figure
                                                 (skipped entirely without this flag — no
                                                 PCA/t-SNE projection is even computed)

OUTPUTS: none, by default — the metrics table above is printed, not saved. `reports/` in
this repo holds decision-trail `.md` files, not a CSV/PNG per run; pass --out/--out-plot
explicitly if you want a file for some other purpose (e.g. feeding another tool).

A LIMITATION TO KNOW ABOUT
-----------------------------
All of this is descriptive of ONE corpus (typically a few hundred papers at most). With
a small n, the PCA map and histograms show real structure but the scalar metrics have
wide error bars — read them as "which model looks meaningfully different", not as
precise measurements. roc_auc_std now gives that error bar a number instead of a hand-
wave; a "the two runs are basically the same" claim should be checked against it, not
asserted from the point estimate alone.

METRICS REWORK (see reports/metrics_rework_and_rerun.md for the full trail) — a
methodology review flagged that the original roc_auc never used the query vector at all
(classifier-on-paper-vectors, query-blind) while the centrality percentile never used
labels at all (query-aware, label-blind) — neither one actually answered "does this
representation retrieve the right papers for THIS query", which is the tool's actual
job. query_auc/recall/wss_at_95 and the positive/negative centroid split close that gap;
roc_auc_conservative and roc_auc_std address two further gaps (the "pass" middle being
silently dropped, and fold variance being computed then discarded). Old CSVs from before
this rework only have the single roc_auc/n_folds columns — don't assume a numeric column
name means the same thing across an old vs. new run without checking the header.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display — this is a batch script
import matplotlib.pyplot as plt
import pandas as pd

from embedding_utils import (
    MODEL_CONFIGS,
    build_label_masks,
    drop_empty_rows,
    embed_corpus_and_use_case,
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

# Run these 4 shortlisted models by a single bare command; --models overrides this.
# (nomic-ai/nomic-embed-text-v1.5 is registered in embedding_utils.MODEL_CONFIGS but not
# listed here — see that module's comment for why: ~795s to embed vs. 4-85s for these 4.)
DEFAULT_MODELS = [
    "sentence-transformers/allenai-specter",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

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
    parser.add_argument("--out", type=Path, default=None, help="Save the scalar metrics table to this CSV path (default: not saved — printed to the console only)")
    parser.add_argument("--out-plot", type=Path, default=None, help="Save the comparison figure to this PNG path (default: not built/saved at all — see notebooks/experiments/ for an inline alternative)")
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

    # Nothing is written to disk unless explicitly asked (--out/--out-plot) — reports/
    # holds this repo's decision-trail .md files, not a per-run CSV/PNG pile. See
    # notebooks/experiments/run_comparisons.ipynb for the inline-output equivalent.
    make_plot = args.out_plot is not None

    model_names = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    print(f"Loading {args.data} ...")
    df = load_export(args.data)
    titles, abstracts = get_title_abstract(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)

    use_case_text = get_use_case_text(df, args.use_case_col, args.use_case_text)
    print(f"  {len(df)} papers. Use case text: \"{use_case_text}\"")

    masks = build_label_masks(df, args.label_col)
    if masks["has_labels"]:
        print(f"  Labels: {masks['n_pos']} positive, {masks['n_neg']} negative, "
              f"{masks['n_pass']} pass (column '{args.label_col}')")
    else:
        print(f"  No '{args.label_col}' column found — plots will skip label colouring and evaluation metrics.")

    n_models = len(model_names)
    if make_plot:
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 4.6 * n_models), squeeze=False)

    rows = []
    for row_idx, model_name in enumerate(model_names):
        print(f"\nEmbedding with {model_name} ...")
        try:
            paper_vectors, use_case_vec, elapsed = embed_corpus_and_use_case(model_name, titles, abstracts, use_case_text)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"model": model_name, "dim": None, "embed_seconds": None,
                         "note": f"embedding failed: {exc}"})
            if make_plot:
                for ax in axes[row_idx]:
                    ax.set_title(f"{model_name}\nFAILED: {exc}", fontsize=8, color="red")
            continue

        disp = dispersion_metrics(paper_vectors)
        cent = centroid_analysis(paper_vectors, use_case_vec, masks["pos_mask"], masks["neg_mask"])
        metrics = evaluate_representation(paper_vectors, use_case_vec, masks)

        if make_plot:
            coords_2d, uc_xy, centroid_xy, var_ratio = project_2d(
                paper_vectors, use_case_vec, cent["centroid"], args.projection
            )
            plot_model_row(axes[row_idx], model_name, df, args.label_col,
                            coords_2d, uc_xy, centroid_xy, var_ratio, disp, cent, metrics)

        rows.append({
            "model": model_name,
            "dim": paper_vectors.shape[1],
            "embed_seconds": round(elapsed, 2),
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
