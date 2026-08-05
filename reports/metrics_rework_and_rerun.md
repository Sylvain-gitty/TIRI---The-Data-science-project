# Evaluation-metrics rework, and what changed when we re-ran everything

A methodology review of `compare_embeddings.py`, `compare_ner_models.py`, and
`compare_combined_features.py` found that the two metrics behind every "model X beats
model Y" claim in this repo — `roc_auc` and the use-case centroid percentile — had a
structural gap between them: **`roc_auc` never looks at the use-case query, and the
centroid percentile never looks at the labels.** Neither one, alone or together, actually
answers the tool's real question: *does this representation retrieve the right papers
for THIS query.* Three smaller gaps came with it: fold-level uncertainty was computed and
then thrown away, the ambiguous `pass` label was silently dropped from every AUC, and nobody
had ever run a paired significance check before calling a point difference "noise."

This report is the fix, in code, then re-run on both real corpora this repo has ever
tested. Everything below is honest about what changed and what didn't — some of this
repo's most confident-sounding old claims turned out not to survive the new metrics.

## Part 1 — what changed, and why (the workflow)

| # | Gap found | Fix | Where |
|---|---|---|---|
| 1 | `roc_auc` is a classifier fit on paper vectors alone — the use-case vector never enters the computation | Added `query_similarity_auc` — rank papers by cosine similarity to the use-case vector itself (the actual nearest-neighbour retrieval mechanism), score that ranking against the labels | `embedding_utils.retrieval_ranking_metrics` |
| 2 | The centroid percentile pools positive/negative/pass together, unweighted by label — "typical of everything retrieved" isn't "typical of what got accepted" | Split the centroid by class: `use_case_to_positive_centroid_sim`, `use_case_to_negative_centroid_sim`, and their difference, `use_case_discriminative_gap` | `latent_space_utils.centroid_analysis` |
| 3 | `pass` (borderline) rows were dropped from every AUC — flattering, since the deployed tool can't skip the ambiguous middle either | Every classifier/ranking metric now computes twice: `_strict` (positive vs. negative only, the old behaviour) and `_conservative` (positive vs. negative+pass) | `embedding_utils.build_label_masks` |
| 4 | Per-fold AUCs were computed, then only the mean was kept — a "0.006 difference is noise" claim in `combined_features_notes.md` §3 was never actually checked against a number | `cross_validated_roc_auc` now returns `roc_auc_std` and the raw `fold_aucs` list | `embedding_utils.cross_validated_roc_auc` |
| 5 | No representation-vs-representation comparison had ever run a significance test — point differences were eyeballed | `notebooks/comparisons/run_comparisons.ipynb` §4 runs a **paired** t-test across the now-exposed `fold_aucs` (paired, not independent, because `StratifiedKFold(shuffle=False)` makes fold row-membership depend only on the label column, identical across representations scored on the same corpus) | notebook + `scipy.stats.ttest_rel` |
| 6 | This class of problem (triage a candidate pool, decide how much of it to keep reading) has a standard domain metric this repo never used | Added `recall_at_10pct` / `recall_at_20pct` and `wss_at_95` (Work Saved over Sampling at 95% recall, Cohen et al. 2006 — the standard citation-screening-automation metric) | `embedding_utils.retrieval_ranking_metrics` |

All three comparison scripts (`compare_embeddings.py`, `compare_ner_models.py`,
`compare_combined_features.py`) now call the same two entry points —
`build_label_masks` and `evaluate_representation` — instead of each copy-pasting the
label-mask logic (three copies, silently able to drift) and each computing only the old
`roc_auc`/`n_folds` pair. `latent_space_utils.plot_model_row`'s annotation now shows both
label readings' AUCs with their std, plus `query_auc`, on every PNG this repo produces.

**The notebook is now the primary way to run and see any of this.**
`notebooks/comparisons/run_comparisons.ipynb` imports these same functions (doesn't
reimplement them) and shows every table, every plot, and the paired-significance check
directly in the notebook — nothing needs to be written to disk first. See
`notebooks/README.md`. `reports/` holds this decision trail (the `.md` files) and nothing
else — the CLI scripts (`compare_embeddings.py` etc.) still run standalone for
scripting/automation, but no longer write a CSV/PNG per run unless you explicitly pass
`--out`/`--out-plot`; by default they just print the same table the notebook shows.

**What this rework did NOT touch:** the underlying models, the use-case text (still the
export's short NAME column, not the richer `.usecase.json` string — that's a separate,
already-documented axis, see `model_shortlist.md` §4b), or the block-weighting in
`compare_combined_features.py` (still a fixed 50/50 split). This is a metrics fix, not a
new modelling attempt.

## Part 2 — re-run on both real corpora tested at the time

Same two corpora, same default models/representations as every prior round in this repo
up to that point, so old-vs-new was a fair comparison:

- **climate/agriculture** — `data/raw/sample-export.parquet`, 100 papers, 39 positive /
  56 negative / 5 pass. **Retired since (2026-08-04) — see the note below §2a.**
- **soil microbiome** — `data/raw/high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN.parquet`,
  602 papers, 94 positive / 263 negative / 245 pass.

The tables below are the full scalar output of that rerun, copied in directly so this
report stays readable on its own. §2b (soil) is reproducible today — open
`notebooks/comparisons/run_comparisons.ipynb` and set `CORPUS_KEY = "soil"`. §2a (climate)
is not — that corpus is gone; see the notice below its table. The notebook also now
supports `data/processed/papers_combined.parquet` (`CORPUS_KEY = "combined"`, pick a
`USE_CASE_KEY`) for any comparison beyond these two. All numbers below are `_strict`
(positive vs. negative only) unless labelled conservative; `roc_auc_std` is the standard
deviation ACROSS the 5 CV folds.

### 2a. Climate/agriculture (100 papers)

> **Retired (2026-08-04):** `data/raw/sample-export.parquet` has been removed from this
> repo — it was a placeholder used to start testing before this project had real
> use-case briefs (no real use-case text of its own, too few observations to trust). The
> table and paired-significance result below are kept as historical record — they were
> real findings from this rework and demonstrate why `fold_aucs` needed exposing — but
> this corpus can no longer be re-run. Use `"soil"` or `"combined"` in the notebook going
> forward.

| Representation | roc_auc | ±std | query_auc | wss@95 | gap(+/−) | dispersion |
|---|---|---|---|---|---|---|
| Specter | 0.500 | 0.121 | 0.457 | −0.040 | −0.021 | 0.81 (collapsed) |
| bge-small | 0.548 | 0.044 | 0.566 | −0.029 | −0.005 | 0.77 (collapsed) |
| paraphrase-multilingual (control) | 0.650 | 0.120 | 0.617 | −0.018 | +0.012 | 0.55 (moderate) |
| all-MiniLM-L6-v2 | 0.584 | 0.091 | 0.599 | +0.003 | −0.001 | 0.46 (best dispersion) |
| NER `entity_type_counts` | 0.553 | 0.135 | 0.500* | −0.040 | 0.000* | 0.22 (sane) |
| NER `entity_text_tfidf` | 0.590 | 0.051 | 0.500* | −0.040 | 0.000* | 0.01 (very disperse) |
| combined: control + `entity_type_counts` | 0.605 | 0.092 | 0.600 | −0.018 | +0.042 | 0.43 (moderate) |
| combined: control + `entity_text_tfidf` | 0.629 | 0.060 | 0.628 | −0.018 | +0.013 | 0.33 (sane) |

\* the use case's own short name produces zero spaCy entities on this corpus (known from
`ner_model_notes.md` §4) — the use-case vector is all-zero, so `query_auc` correctly comes
back at exact chance (0.5) and `gap` at exactly 0. This is the new metric independently
confirming the same degenerate case the old centroid percentile already flagged (0th
percentile) — good, not a discrepancy.

**Paired significance (notebook §4, `scipy.stats.ttest_rel` on the 5 paired
`roc_auc_strict` folds):** of the 28 pairwise comparisons among these 8 representations,
only **2 reach p < 0.05** — bge-small vs. combined+tfidf (p=0.016) and Specter vs.
combined+tfidf (p=0.033). Every other pair, including the ones `combined_features_notes.md`
§2 called a real result ("combining made things WORSE than the embedding alone" — control
0.650 vs. combined 0.605/0.629) **is not statistically distinguishable from noise on this
evidence**: control vs. combined+type_counts, p=0.673; control vs. combined+tfidf, p=0.475.
The direction of the old claim may still be right, but the confidence the old report's
prose implied wasn't there — this is the single clearest concrete example of why exposing
`fold_aucs` mattered.

### 2b. Soil microbiome (602 papers) — the headline finding

| Representation | roc_auc | ±std | query_auc | wss@95 | gap(+/−) | dispersion |
|---|---|---|---|---|---|---|
| Specter | 0.776 | 0.066 | **0.293** | −0.022 | −0.042 | 0.84 (collapsed) |
| bge-small | 0.803 | 0.063 | **0.392** | −0.030 | −0.029 | 0.78 (collapsed) |
| paraphrase-multilingual (control) | 0.768 | 0.064 | **0.429** | +0.006 | −0.059 | 0.56 (moderate) |
| all-MiniLM-L6-v2 | 0.793 | 0.094 | **0.380** | −0.008 | −0.095 | 0.48 (best dispersion) |
| NER `entity_type_counts` | 0.551 | 0.015 | **0.579** | −0.016 | +0.087 | 0.24 (sane) |
| NER `entity_text_tfidf` | 0.616 | 0.068 | 0.466 | −0.033 | −0.153 | 0.01 (very disperse) |
| combined: control + `entity_type_counts` | 0.760 | 0.047 | 0.567 | −0.008 | +0.013 | 0.47 (moderate) |
| combined: control + `entity_text_tfidf` | 0.762 | 0.068 | 0.410 | +0.003 | −0.061 | 0.36 (moderate) |

**Every embedding model's `query_auc` on this corpus is BELOW 0.5** — ranking papers by
similarity to the use case's own text is *worse than random* at putting accepted papers
ahead of rejected ones, for all four embedding models tested, despite `roc_auc` calling
every one of them "solid" (0.768–0.803). This is not a contradiction or a bug: the
`use_case_discriminative_gap` is negative for all four (as low as −0.095 for all-MiniLM),
meaning the use case's own embedded text sits **closer to the centroid of the rejected
papers than the accepted ones**, for every embedding model on this corpus. `roc_auc`
couldn't have caught this — it never looks at the use-case vector at all. This is exactly
the failure mode the metrics rework set out to find, and on this corpus, it's real and
large, not a rounding artifact.

The flip side: NER's crude `entity_type_counts` — the representation `ner_model_notes.md`
called "close to chance" and "meaningfully weaker than every embedding tested" — has the
only **positive** `query_auc` (0.579) and the only positive `gap` (+0.087) among every
single-representation row on this corpus. By `roc_auc` alone, NER lost to every embedding.
By `query_auc` — the metric that actually reflects how a nearest-neighbour retrieval step
would rank papers for this query — it beat every embedding tested. Concatenating it onto
the control embedding lifts the combination's `query_auc` from 0.429 (embedding alone) to
0.567, while `roc_auc` barely moves (0.768 → 0.760) — the exact opposite of
`combined_features_notes.md`'s "no free win" verdict, once you're asking the
query-conditioned question instead of the classifier one.

**`wss_at_95` is near zero or negative for every representation on both corpora** — no
representation tested here reliably lets an analyst skip a meaningful fraction of the
corpus while still catching 95% of the true positives, when ranked by raw query
similarity. Read this as the most sobering number in the whole rework: strong `roc_auc`
numbers do not, on this evidence, translate into a practically useful ranked list.

## Part 3 — conservative reading (pass counted as rejected)

The gap between `_strict` and `_conservative` AUCs is itself informative — a big gap means
the `pass` rows were doing real classification work the strict reading skipped. On the
soil corpus (`n_pass=245`, 41% of triaged papers), every embedding's `roc_auc_conservative`
comes back HIGHER than `roc_auc_strict` (e.g. bge-small 0.803 → 0.847, control 0.768 →
0.816) — the opposite of the "conservative readings should look worse" intuition, because
`pass` rows apparently look more like clear negatives than like ambiguous cases in
embedding space on this corpus. On the smaller climate corpus the pattern is mixed (Specter
0.500 → 0.568, but control 0.650 → 0.628) — another corpus-dependent result, don't
generalize from one to the other. Full `_conservative` numbers for every representation
are in the notebook's `embedding_df`/`ner_df`/`combined_df` tables.

## Where this leaves things

- The metrics rework is in production in all three comparison scripts and the new
  notebook — every future run of this repo's comparison tooling gets both label readings,
  both AUC families, fold-level uncertainty, and the label-aware centroid split by
  default, no flag required.
- The re-run **did not overturn** the standalone NER-vs-embedding classifier gap
  (`roc_auc`: NER 0.55–0.62 vs. embeddings 0.65–0.80 stands on both corpora) — but it did
  overturn the *interpretation* that this gap settles whether NER is useful, since
  `query_auc` tells a materially different story on the soil corpus.
- Two claims from before this rework do not survive scrutiny as stated: "combining made
  things worse on the climate corpus" (not statistically significant, §2a) and "combining
  is a wash on the soil corpus, no free win" (true for `roc_auc`, false for `query_auc`,
  §2b).
- Untested, still open: whether `query_auc`/`wss_at_95` improve with the richer use-case
  text (`--use-case-text`, `model_shortlist.md` §4b tested this only against the old
  metrics); whether a tuned, non-50/50 block weighting in `compare_combined_features.py`
  changes the `query_auc` story further; a genuine held-out replication (this rework's
  paired test is single-corpus, single-run — informative about internal consistency, not
  proof of generalisation).
- Since this rerun, `data/raw/sample-export.parquet` (§2a's climate corpus) has been
  removed from the repo (placeholder test file, no real use-case text, too few
  observations) and the notebook gained a second corpus source:
  `data/processed/papers_combined.parquet`, 6 real research questions in one file, each
  with an actual objective + search-term brief the notebook uses to build a richer
  use-case query automatically — no separate `.usecase.json` needed. Re-running this
  rework's method against those isn't done here; that's a natural next comparison.

**Run it yourself:** open `notebooks/comparisons/run_comparisons.ipynb` from inside
`notebooks/comparisons/` and run all cells — every table and plot renders inline.
`CORPUS_KEY = "soil"` reproduces §2b above; `CORPUS_KEY = "combined"` (with a
`USE_CASE_KEY`) runs any of the 6 research questions in `papers_combined.parquet`
instead. Nothing needs to exist in `reports/` first; the notebook loads straight from
`data/raw/`/`data/processed/`.

The same functions are also callable from the command line for scripting/automation —
`python scripts/compare_embeddings.py --data data/raw/your-export.parquet` (same for
`compare_ner_models.py`/`compare_combined_features.py`) — which prints the identical
metrics table to the console and writes nothing unless you pass `--out`/`--out-plot`
explicitly.
