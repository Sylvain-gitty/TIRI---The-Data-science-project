# What text the embedding sees — the $0 pre-check for probe P-TX

**Status: measured, 2026-08-14.** `BAAI/bge-small-en-v1.5`, fastembed, CPU
only. 3 `small_test` SYNERGY collections
(`reports/benchset_v1_split_manifest.json`) plus one TIRI silo
(`data/processed/papers_combined.parquet`). Total wall-clock: **0.4s**. $0 —
no GPU, no Modal, no network beyond fastembed's own one-time model download (this run found
the model already cached locally).

**Confidence key:** 🟢 clears the 0.03 noise floor / decisively measured · 🟡 real but under
the floor · ⚪ engineering finding — a gate result at a screening threshold, not an effect size.

---

## 0. How to read any number here

**This is a screening gate, not an effect-size claim.** `CONTEXT.md` §5 measures a
seed-to-seed sd of ~0.010 on within-silo ROC-AUC and treats any gap under ~0.03 as *not
established*. This gate's bar is **0.01**, one third of that floor. That is
deliberate: the point of a $0 pre-check is to be generous *before* spending Modal GPU money,
not to claim a text-input effect is real. A PASS here means "worth the ~$3 to look properly",
not "text input matters, decisively". A FAIL means all four collections sit inside a band a
third the width of noise the repo already treats as unreliable — which is a much stronger
statement than "no effect found", because the band it failed to clear is itself
conservative.

**Two provenances, never pooled.** The three benchset collections' briefs are LLM-derived
from each review's own abstract (`brief_provenance = review_abstract`); the TIRI silo's brief
is analyst-written from scratch. `DATA_BRIEF.md` honest-limit #2: averaging across that seam
compares two brief-generation processes, not two text-input arms. Every table below reports
them separately.

**Deterministic, no seeds.** Cosine-to-brief AUC here is a raw ranking statistic (no
classifier fit, no folds — `retrieval_ranking_metrics`'s `query_similarity_auc`), and
fastembed's ONNX forward pass has no dropout at inference. The same input text always
produces the same score, so there is no seed-to-seed spread to report for this specific
number the way `run_label_budget_shape.py`'s fitted arms need one.

**Why the wrong-axis version of this gate was not run.** The plan's original pre-check
compared `cos_brief_*` against `cos_briefpre_*` — a **brief**-side ablation
(`embed_benchsets.BRIEF_VARIANTS["pre_screening"]` drops `objective`, adds term lists), with
byte-identical paper vectors between the pair. That measurement already exists
(`03_eda_full_benchset_v1.ipynb` §8.4, cells 39-40): paired median **0.005-0.007** (dropping
the abstract) and **0.003-0.007** (the realistic pre-screening brief) across 27 collections —
below this gate's bar, but about a different axis entirely. Verified directly against that
notebook before writing this script. Running it as the P-TX gate would have produced a FAIL
on a number that says nothing about paper text, which is why this script embeds a genuine
title-only paper arm instead — the first one ever computed for this corpus.

---

## 1. The verdict

**Gate: PASS.** Gate: any collection where |AUC(title-only) − AUC(title+abstract)| ≥ 0.01 → PASS (P-TX earns the Modal run). All four inside the band → FAIL (P-TX closes, CONTEXT.md §6 gains a row).

Collection(s) clearing the bar: `synergy_donners_2021` (+0.052), `synergy_oud_2018` (-0.027), `synergy_meijboom_2021` (-0.052), `carbon_capture` (+0.056)

🔴 **ELI18.** Feeding the embedder only the paper's title, instead of its title *and* abstract,
changed how well the ranking worked by 2.7 to 5.6 points of AUC — where 0.500 is a coin flip and
0.030 is the smallest gap this repo treats as real. So *which text we hand the model* is not a
detail; it is worth as much as most of the features this project has rejected.

**Two things about the size, and the first was mis-stated in an earlier draft of this line.**

1. 🟢 **Three of the four gaps clear the 0.03 noise floor**, not merely the 0.01 gate: 0.052, 0.052
   and 0.056, against a floor of 0.030. Only `synergy_oud_2018` (0.027) sits inside it. This gate was
   deliberately set at one third of the floor to be generous before spending money — but it did not
   need the generosity, and saying it "is well under an established effect" would be wrong.
2. 🔴 **The sign is inconsistent, and that is the real finding.** Dropping the abstract makes
   `donners_2021` and `carbon_capture` **better** (+0.052, +0.056) and `meijboom_2021` and
   `oud_2018` **worse** (−0.052, −0.027). Two up, two down, at similar magnitudes.

*ELI18 on why that matters more than the average: if abstracts simply helped, all four would move the
same way and the answer would be "keep using them". Instead, for half these collections the abstract
is actively getting in the way. So the lever is not "more text is better" — it is "the right text
per use case", which is a harder and more interesting thing to build.* A mean across these four would
be ≈ +0.007 and would report "no effect" while hiding four effects, which is exactly the
mean-hides-a-split failure `CONTEXT.md` §5 demands win counts to prevent.

⚠️ **This does not license a claim about the shipped features.** The gate holds one model fixed
(`BAAI/bge-small-en-v1.5`, CPU) so the contrast is about text input alone — but that is *not* the
model behind `cos_brief_jasper` / `cos_brief_qwen4b`. The gate answers "does the text input move
within-silo AUC at all on this corpus", which is what it was designed to answer. It does not answer
"by how much would the shipped number move", and the Modal run it earns is what would.

## 2. Per-collection results

| collection            | provenance | n_papers | n_pos_strict | n_neg_strict | pct_abstract_over_4000 | auc_title_abstract | auc_title_only | diff   | gate_pass |
|-----------------------|------------|----------|--------------|--------------|------------------------|--------------------|----------------|--------|-----------|
| synergy_donners_2021  | benchset   | 258      | 15           | 243          | 0.147                  | 0.808              | 0.859          | 0.052  | True      |
| synergy_oud_2018      | benchset   | 952      | 20           | 932          | 0.009                  | 0.877              | 0.851          | -0.027 | True      |
| synergy_meijboom_2021 | benchset   | 882      | 37           | 845          | 0.009                  | 0.909              | 0.857          | -0.052 | True      |
| carbon_capture        | live       | 323      | 147          | 150          | 0.006                  | 0.626              | 0.682          | 0.056  | True      |

`pct_abstract_over_4000` is the fraction of each collection's abstracts longer than
`embed_benchsets.MAX_ABSTRACT_CHARS` (trap #2 below) — the `title_abstract` arm here
truncates them exactly as a real Modal run would; `title_only` never sees an abstract at all,
truncated or not.

## 3. Three traps recorded for whoever runs the Modal part, if this gate passes

1. **Cache filename.** `embed_benchsets.papers_path()` (~:155) is
   `benchset_v1_{use_case_key}_{safe_model}_papers.parquet` — no `text_variant` in it.
   Four text variants would silently overwrite each other, and the resume logic ("a finished
   collection is skipped outright") would then *skip* a variant that never ran. This script
   never touches that path or that pipeline — every vector here is computed directly through
   `embedding_utils.embed_papers`/`embed_texts` with report-local files only.
2. **Truncation confound.** `MAX_ABSTRACT_CHARS = 4000` truncates abstracts but never titles,
   so `title_only` is the only *fully* untruncated arm — a confound between arms, not merely
   a text change. Quantified per collection in §2 above:
   `synergy_donners_2021` has the largest share truncated; the TIRI silo and the other two
   benchset collections are under 1%.
3. **The shuffled-brief seam does not apply.** `CONTEXT.md` L258-260 mandates a
   shuffled-brief derangement control for anything claiming to read the brief, to prove the
   reader is not scoring on an artefact independent of brief content. P-TX changes the
   **paper** side, not the brief side — the brief is embedded once per collection and reused
   unchanged across both arms (see the module docstring and `evaluate_collection`) — so there
   is no brief-reading claim here for that seam to falsify. Stated explicitly, not omitted.

**Also recorded:** the `+venue` arm named in the plan is dropped — `papers_benchset_v1.parquet`
has no `venue` column, so it is unbuildable on its own confirmatory surface (only the
2,873-row `papers_combined.parquet` has one). `CONTEXT.md` §6 already rejected `has_venue` as
a *metadata* feature (part of the 11-feature punch list, +0.002 combined) — a different
operation from a text-input change, but worth saying before a reviewer asks.

## 4. The prior this cuts against

**Judgement call, not a hard metric:** `CONTEXT.md` §7 measured what a genuinely *better*
brief (an induced rule set from 60 labels) is worth to cosine-to-brief on set A — **+0.004 /
−0.005**, against +0.065 for the lexical block on the identical swap — and calls the cosine
"the unimprovable one". That is a brief-quality lever, not a text-input lever, but it is the
closest existing measurement of "can anything be done to move this particular ranker", and it
found next to nothing. This gate did not repeat that finding — at least one collection moved past the bar, so the paper-text axis is not obviously as inert as the brief-quality axis was.

## 5. What this script did not do

No GPU, no Modal, no re-embedding of any full corpus, no change to `embed_benchsets.py` or
its cache layout. `MODEL_CONFIGS`, `build_paper_texts`, and `embed_papers` in
`scripts/embedding_utils.py` gained one optional `text_variant` parameter each
(default-preserving; every existing caller is unaffected) — nothing else in the shared core
changed.
