# NER model notes — properties, choice, and what testing it actually found

The decision trail behind `scripts/compare_ner_models.py`, in the same spirit as
`reports/model_shortlist.md` for embeddings: why this NER tool, what it was tested
against, and what came back — including the unflattering parts.

## 1. Scope of this comparison

Same goal as the embedding comparison (see `HANDOFF.md` and `scripts/compare_embeddings.py`'s
docstring): test how a use-case representation relates to an already-labelled dataset —
latent-space sanity (dense/disperse), cosine similarity structure, use-case centrality —
**not** select a model for a future classifier baseline. Nothing downstream is wired to
whatever "wins" a run of this script.

## 2. Why spaCy, over GLiNER / scispaCy

| Property | spaCy (`en_core_web_sm`) | GLiNER | scispaCy |
|---|---|---|---|
| Local / offline, no API key | Yes | Yes | Yes |
| PyTorch dependency | No (small pipeline) | Yes | Yes |
| Domain fit for academic/scientific text | No — generic entity types | Zero-shot, could target custom labels (organism, technique, ...) at inference time | Yes — trained on biomedical/scientific text |
| Maturity / CPU throughput | Mature, fast on CPU | Newer, transformer-backed, slower on CPU | Mature, but still a heavier model download |
| Entity label set | Fixed, generic (`PERSON`, `ORG`, `GPE`, `DATE`, `NORP`, ... — 18 labels, see `--list-entity-labels`) | Arbitrary, chosen at inference time | Fixed, biomedical (genes, chemicals, diseases, ...) |

Picked `en_core_web_sm` for this first pass on the same "local, no API key, CPU
throughput" priorities `reports/model_shortlist.md` §1 used for embeddings — the
lightest, fastest, no-extra-dependency floor. **The known tradeoff, up front:** its entity
types are generic-purpose, not scientific-domain — it will not recognise organism/species
names, chemical compounds, or lab techniques as anything other than generic
`PERSON`/`ORG`/nothing. GLiNER (custom domain labels) or scispaCy (biomedical-trained) are
the natural next candidates if this floor turns out too weak — not evaluated in this round.

## 3. Two representations tested

`scripts/compare_ner_models.py`'s `DEFAULT_REPRESENTATIONS`:

1. **`entity_type_counts`** — per paper, a vector of how many entities of each spaCy
   label were found, L1-normalised (a type-mix profile, not raw counts, so document
   length doesn't dominate). Tests whether papers separate by entity-TYPE profile.
2. **`entity_text_tfidf`** — TF-IDF (+ SVD, same recipe as `embedding_utils.py`'s `tfidf`
   backend) over the actual entity TEXT spans extracted (e.g. "University of Leeds",
   "2024"). Tests whether papers separate by WHICH specific entities recur.

Both get the exact same dispersion / centroid-percentile / (secondary) `roc_auc`
diagnostics as `compare_embeddings.py`, via the shared `scripts/latent_space_utils.py`.

## 4. Tested on the same 2 real corpora as the embedding comparison

| Corpus | Representation | avg cosine | dim | ROC-AUC | use-case percentile | zero-entity papers |
|---|---|---|---|---|---|---|
| climate/agriculture (100 papers) | `entity_type_counts` | 0.22 — sane | 18 | 0.553 | 0th (sim=0.00) | 20/100 |
| climate/agriculture (100 papers) | `entity_text_tfidf` | 0.01 — very disperse | 100 | 0.590 | 0th (sim=0.00) | 20/100 |
| soil microbiome (602 papers) | `entity_type_counts` | 0.24 — sane | 18 | 0.551 | 70th (sim=0.89) | 172/602 |
| soil microbiome (602 papers) | `entity_text_tfidf` | 0.01 — very disperse | 100 | 0.616 | 89th (sim=0.29) | 172/602 |

**Non-obvious findings worth carrying forward (reported honestly, not just the flattering
parts):**

- **The use case's own text produced ZERO entities on the climate/agriculture corpus.**
  The export's `use_case` column only carries a short NAME (see
  `embedding_utils.get_use_case_text`'s docstring) — "Climate change extremes and
  agriculture practices" has no `PERSON`/`ORG`/`GPE`/etc. for a general-purpose NER model
  to find. That collapses `use_case_to_centroid_sim` to exactly 0.0 for BOTH
  representations on this corpus — a genuine finding, not a bug (the zero-vector handling
  in `latent_space_utils.centroid_analysis` is deliberate). On the soil-microbiome corpus's
  use case name ("High quality microbial and fungal community in Soil") entities WERE
  found, and the use case landed at a real, non-degenerate percentile (70th / 89th) — so
  this isn't universal, it depends on whether the use-case NAME happens to contain
  something spaCy recognises. As-tested, NER-derived "how central is the use case"
  is not reliable for a use-case name this short.
- **Tested with the real fuller use-case text — and it moved the two representations in
  OPPOSITE directions, not both toward "better."** Using the analyst's actual objective/
  key-terms JSON (`data/raw/high-quality-microbial-and-fungal-community-in-soil.usecase.json`)
  instead of the short name, re-run on the same soil corpus
  (`reports/high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN_richer_usecase_ner_representation_comparison.csv`):
  `entity_type_counts` percentile DROPPED from 70th to 39th (sim 0.89→0.30), while
  `entity_text_tfidf` percentile ROSE from 89th to 100th (sim 0.29→0.46) — the most
  extreme value the metric can report. ROC-AUC and dispersion were unchanged for both
  (expected — see `model_shortlist.md` §4b: those metrics never see the use-case vector).
  Reading the two representations side by side explains the split: the fuller text adds
  many domain WORDS (matching what `entity_text_tfidf` looks for) but doesn't add more
  recognisable spaCy entity TYPES (what `entity_type_counts` looks for) — if anything it
  dilutes the type-mix profile with more non-entity domain vocabulary. So "give the use
  case richer text" is not a universal fix for NER-derived centrality — it depends on
  which half of the representation (types vs. specific text) the richer wording actually
  feeds.
- **A large fraction of papers produced zero entities at all** — 20% on the smaller
  corpus, 29% (172/602) on the larger one. Both entity_type_counts and
  entity_text_tfidf inherit the same zero-entity papers (same underlying NER pass) —
  this isn't a representation-specific weakness, it's a spaCy-model-on-this-text
  weakness. A domain-trained NER model (scispaCy) would be the natural thing to test to
  see whether that fraction shrinks.
- **`entity_text_tfidf` looks far more disperse than `entity_type_counts`** on both
  corpora (avg cosine 0.01 vs. 0.22-0.24) — expected, since TF-IDF over specific entity
  text is a much higher-dimensional, sparser signal than an 18-dimensional type-mix
  profile. Neither looks collapsed; both are on the "reasonably-to-very disperse" side,
  the opposite failure mode from what several embedding models showed (bge/Specter's
  0.7+ dense cones, `reports/model_shortlist.md` §3-4).
- **ROC-AUC stayed close to chance on both representations, on both corpora** (0.55-0.62)
  — meaningfully weaker than every embedding model tested in `model_shortlist.md` §4
  (0.65-0.80 range). As a diagnostic reading (not a baseline-selection signal — see
  scope note above), this says generic NER-derived features alone don't separate these
  labels as well as sentence embeddings do on either corpus tested. Whether NER features
  add anything WHEN COMBINED with an embedding (concatenated, per `HANDOFF.md`'s "likely
  intersects with the existing baseline" note) is untested here — this script only tests
  each representation standalone.

## 5. Where this leaves things

This is a first-pass floor test, from 2 data points, same judgement-call caveat as
`model_shortlist.md` §5: not a proof. If NER-derived features are pursued further for
Week 2/3 feature engineering, the natural next steps (not started) are: (a) try a
domain-trained NER model (scispaCy) to see whether the zero-entity fraction and ROC-AUC
both improve on scientific text, and (b) test `entity_type_counts`/`entity_text_tfidf`
concatenated onto an embedding vector rather than standalone, since standalone
performance being weaker than embeddings doesn't rule out an additive contribution.

Run it yourself:

```bash
python scripts/compare_ner_models.py --data data/raw/your-export.parquet
```

Outputs: `reports/<data filename>_ner_latent_space_comparison.png` (visual comparison) and
`reports/<data filename>_ner_representation_comparison.csv` (scalar metrics) — see the
script's own docstring for what each metric means.
