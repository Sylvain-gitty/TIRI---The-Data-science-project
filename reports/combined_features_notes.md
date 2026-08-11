# Combined features notes — does concatenating NER onto an embedding help?

The decision trail behind `scripts/compare_combined_features.py`: why it exists, what it
tested, and what came back — including the parts that didn't confirm the hypothesis that
motivated building it.

## 1. Scope and motivation

Same scope note as the other two comparison scripts: this is a diagnostic against an
already-labelled dataset (dispersion, cosine similarity, use-case centrality), not a
classifier-baseline-selection step. `reports/ner_model_notes.md` found NER-derived
features score close to chance standalone (ROC-AUC 0.55–0.62) — meaningfully weaker than
every embedding tested (0.65–0.80, `model_shortlist.md` §4). The field-guide artifact
cross-referenced against 11 open-source paper-scoring tools found that several of them
win by *combining* a cheap signal with a precise one (e.g. Paper-QA-RAG-LoRA's
cross-encoder re-ranking moved hit@5 from 0.776 to 0.928 over bi-encoder retrieval
alone), rather than picking one winner standalone. This script tests the direct,
simplest version of that idea: concatenate an embedding vector with an NER-derived
vector per paper (each block L2-normalised independently before `np.hstack`, so neither
dominates by raw magnitude — see the script's own docstring), and re-run the same
diagnostics.

Default embedding block: `paraphrase-multilingual-MiniLM-L12-v2` — a convenience default
(the model both exports ship precomputed vectors for), not a pick.

> **Corpus retired (2026-08-04):** `data/raw/sample-export.parquet` — the
> "climate/agriculture" 100-paper corpus §2 below is about — has been removed from this
> repo. It was a placeholder used to start testing before this project had real use-case
> briefs: no real use-case text of its own (just a generic short name) and too few
> observations to trust a comparison drawn from it. §2's numbers are kept as historical
> record and can no longer be reproduced by re-running against that file — §3 (soil
> corpus) is unaffected. New comparisons should use the soil-microbiome corpus or
> `data/processed/papers_combined.parquet`'s 6 real research questions — see
> `notebooks/experiments/run_comparisons.ipynb`.

## 2. Results: climate/agriculture corpus (100 papers, short use-case name)

| Signal | ROC-AUC | avg pairwise cosine |
|---|---|---|
| Embedding alone (control) | 0.650 | 0.56 — moderate |
| NER alone (`entity_type_counts`) | 0.553 | 0.22 — sane |
| NER alone (`entity_text_tfidf`) | 0.590 | 0.01 — very disperse |
| **Combined: embedding + `entity_type_counts`** | **0.605** | 0.43 — moderate |
| **Combined: embedding + `entity_text_tfidf`** | **0.629** | 0.33 — sane |

**On this corpus, combining made things WORSE than the embedding alone**, not better —
0.605/0.629 sits between the embedding-alone and NER-alone scores, closer to their
average than to either extreme. A naive 50/50-weighted concatenation diluted the
embedding's real signal with the NER block's much weaker one, rather than adding to it.

## 3. Results: soil microbiome corpus (602 papers, short use-case name)

| Signal | ROC-AUC | avg pairwise cosine |
|---|---|---|
| Embedding alone (control) | 0.768 | 0.56 — moderate |
| NER alone (`entity_type_counts`) | 0.551 | 0.24 — sane |
| NER alone (`entity_text_tfidf`) | 0.616 | 0.01 — very disperse |
| **Combined: embedding + `entity_type_counts`** | **0.760** | 0.47 — moderate |
| **Combined: embedding + `entity_text_tfidf`** | **0.762** | 0.36 — moderate |

**On this corpus, combining was neutral, not harmful** — 0.760/0.762 vs. 0.768 alone is a
~0.006–0.008 dip, well inside cross-validation noise for `n_folds=5`. The NER block
neither helped nor meaningfully hurt here.

**The two corpora disagree on whether combining is worth it at all** — same pattern as
`model_shortlist.md` §4's ranking flip: don't generalize a "combining helps/hurts"
verdict from one dataset. On the evidence so far, naive equal-weighted concatenation is
not a free win on either corpus tested — at best neutral, at worst a real dilution.

## 4. An unexpected side benefit: combining rescues a degenerate use-case vector

`ner_model_notes.md` reported that the climate corpus's short use-case text
("Climate change extremes and agriculture practices") produces **zero NER entities**,
forcing `use_case_to_centroid_sim` to exactly 0.0 and the percentile to 0th for BOTH
standalone NER representations — the diagnostic has nothing to say about the use case at
all in that case.

In the combined run on the same corpus, that same zero-entity use case landed at the
53rd percentile (`entity_type_counts` combo) and 90th percentile (`entity_text_tfidf`
combo) — not degenerate. Why: when the NER half of the use-case vector is entirely zero,
`latent_space_utils.centroid_analysis`'s centroid comparison is driven almost entirely by
whichever block DID produce a real vector for the use case — here, the embedding block.
Concatenation doesn't just average two scores; it can rescue a diagnostic from a
degenerate all-zero corner case in one block, for free, as long as the other block still
has signal. This is a genuine, useful side effect of combining, distinct from (and,
on this evidence, more reliable than) any hoped-for ROC-AUC improvement.

## 5. Results: soil corpus, richer use-case text (the real `.usecase.json`)

Same combined representations, re-run with the fuller use-case string built from
`data/raw/high-quality-microbial-and-fungal-community-in-soil.usecase.json`'s
`objective` + `terms.must_include` + `terms.nice_to_have` (see `model_shortlist.md` §4b
and `ner_model_notes.md` §4 for the standalone versions of this same test):

| Combined representation | percentile (short name) | percentile (full objective+terms) |
|---|---|---|
| embedding + `entity_type_counts` | 92.9th (sim 0.81) | **21.3th (sim 0.61)** — big drop |
| embedding + `entity_text_tfidf` | 62.6th (sim 0.59) | 68.1th (sim 0.62) — modest rise |

ROC-AUC and dispersion were unchanged (expected — confirmed invariant to
`--use-case-text` in every comparison script in this repo, see `model_shortlist.md`
§4b). The centrality swing direction matches the standalone NER result in
`ner_model_notes.md` §4 (`entity_type_counts` got LESS central with richer text,
`entity_text_tfidf` stayed roughly flat/slightly more central) — but the MAGNITUDE
differs: standalone `entity_type_counts` dropped 70th→39th (31 points), while combined
dropped a much larger 92.9th→21.3rd (71.6 points). Combining doesn't just add the two
effects — it can amplify them, at least in this one case. Not enough evidence here to
say whether that generalizes.

## 6. Where this leaves things

Naive, equal-weighted concatenation is not, on this evidence, a reliable way to improve
discriminative power (ROC-AUC) — it was neutral-to-harmful on both corpora tested, the
opposite of what the field-guide's multi-stage-funnel comparison hoped for. Two honest
caveats before ruling the idea out entirely: (1) the field guide's real wins (e.g.
Paper-QA-RAG-LoRA's +20% hit@5) came from a trained *cross-encoder re-ranking* stage, not
a static, unweighted concatenation — a genuinely different, more expensive technique
than what this script tests; (2) a tuned `--embedding-weight` (rather than the deliberate
50/50 split used here) is untested and could change the picture. What combining DOES
reliably do, on this evidence, is rescue the use-case centrality diagnostic from a
degenerate all-zero-block case — a real, if narrower, benefit than the one originally
hoped for.

**Run it yourself, with output inline:** `notebooks/experiments/run_comparisons.ipynb`
§3 — see `notebooks/README.md`.

`scripts/compare_combined_features.py` still works as a CLI (`python
scripts/compare_combined_features.py --data data/raw/your-export.parquet`) for
scripting/automation, printing the same table to the console; pass `--out`/`--out-plot`
explicitly if you also want a CSV/PNG on disk (neither is written by default —
`reports/` holds this decision trail, not per-run artifacts).
