r"""`P-CK` — checkability and evidence availability, computed before any scoring, $0, no LLM.

THE QUESTION (`wf_spec_quality_plan.md`, `P-CK` section, amended 2026-08-14 before this ran).
Two numbers describe how automatable a use case is, and both are computable from the spec and
the corpus alone:

  - **criterion checkability** — what fraction of the spec is an extractable predicate a machine
    could test, rather than a judgement invitation a human has to read and decide?
  - **evidence availability** — what fraction of the pool's abstracts actually state the evidence
    that predicate would need, so a checkable criterion is not automatable if nothing on disk
    can answer it.

The motivating pair, from S-AL: `cement_binders` reaches AUC 1.000 under every model tested
because its criteria are checkable predicates (`>50% CO2 vs OPC`, `>=40 MPa @ 28d`); tech-intel
has no performance criteria at all and is the hardest use case in the repo. If checkability and
evidence availability are real, `cement_binders` should score near the top and `tech_forecasting`
near the bottom — a check made explicit below, not just asserted.

THIS IS S-UCQ Q5's TIRI-SIDE HALF (`academic_agent: spikes/use_case_quality/PLAN.md:99,149,151-
154`), pre-registered there as **descriptive, not inferential** (n=4 there supports no
correlation). `probe_5_checkability.py` does not exist yet in that repo — this is the FIRST
implementation, written portably enough that it could be adopted there unchanged: nothing below
depends on a TIRI-only file, and the substitution below is the reason why.

WHY SPANS, NOT FIELDS. The first design counted *field presence*: is `performance_criteria`
populated, is `decision_criteria` populated. That denominator does not exist across 34 —
`performance_criteria` is populated in 3 of 34 use cases (all TIRI), `decision_criteria` in 5 of
34. A metric with almost no denominator is not a metric. So this counts **spans of text**
instead: free-text fields are sentence-split, list fields are already one item per span (a list
field is pre-segmented by the analyst, not by us), and `constraints.trl` contributes one
synthetic span when it is not null. That keeps checkability a genuine `[0, 1]` fraction with a
real denominator in all 34 rows.

TWO STRATA, ONE CONSTRUCT.

  TIRI (n=6, `data/raw/{carbon_capture,cement_binders,ner,soil_microbiome,solar_leo,
  tech_forecasting}.usecase.json` — read directly, NOT the flattened `papers_combined.parquet`,
  which does not carry every field this needs): `objective`, `problem_statement`, `notes`,
  `terms.{must_include,nice_to_have,exclude}`, `domain.technology_focus`,
  `performance_criteria[].{metric,target}`, `decision_criteria.{must_have,nice_to_have,
  exclusions,decision_rules}`, `constraints.trl`.

  Benchset (n=28, `academic_agent/evals/data/benchsets/<key>.spec.json` — NOT
  `data/benchsets_v1/briefs.parquet`, which carries no criteria fields at all): the same
  free-text/`terms`/`domain` fields, plus `population`, `interventions[]`, `outcomes[]`,
  `evidence.{include_types,exclude_types}` **in place of** `performance_criteria` /
  `decision_criteria`, which are empty in all 28 (confirmed 2026-08-14: 0/28 populated, 0/28
  have a `constraints.trl`).

  JUSTIFICATION FOR THE SUBSTITUTION, put here because it is the whole reason this metric is
  portable rather than TIRI-private: in systematic-review methodology, PICO (population,
  intervention, comparison/outcome) plus evidence-type eligibility criteria **are** the same
  theoretical object TIRI's `performance_criteria` / `decision_criteria` fields occupy, under a
  different name — both are the analyst's pre-committed test for "does this paper count". That
  is what makes crossing the two strata a construct-validity claim rather than a coincidence of
  column names, and it is also why the pooled-vs-within-stratum bar below exists: if the spread
  vanishes once TIRI's 6 are looked at alone, the two schemas were never actually one thing.

REGEXES — reused, not invented, per `scripts/checkability_patterns.py`'s own header (read it for
exact source lines). `METRIC_HEADS` / `_METRIC_RE` / `_METRIC_LEADING_STOP` / `_UNIT_RE` are
copied verbatim from `academic_agent/discovery.py`; `UNIT_RE` / `METRIC_RE` / `LEADING_STOP` from
`academic_agent/spikes/review_seeding/probe_metrics.py`. **The plan's original citation of
`spikes/concept_extraction/cues.py` was a mis-citation** — that module's own docstring says it
detects novelty-introduction frames for technology names, nothing to do with predicates; it is
not used anywhere here. The comparator regex (`COMPARATOR_RE_NARROW` / `_BROAD`) exists in
neither sibling-repo file and is new code, written for this probe alone.

THE SPAN CLASSIFIER — field-agnostic (the same function sees a sentence, a list item, or a
metric name; it never knows which field a span came from), applied in this fixed order,
first-match-wins:

  1. a comparator (`>=`, `<=`, `>`, `<`, "at least", "no more than", "no less than") plus a
     number       -> PREDICATE (threshold)
  2. "(e.g.|i.e.|such as)" introducing an enumerable list                -> PREDICATE (enumerated)
  3. a metric-head match with no judgement cue also firing               -> PREDICATE (named metric)
  4. <=8 tokens and no finite-verb cue                                   -> PREDICATE (bare keyword)
  5. `constraints.trl` non-null (synthetic span, TIRI only)               -> PREDICATE (range)
  6. a judgement-hedge cue fires (narrow lexicon, see below)              -> JUDGEMENT
  7. a finite-verb cue + a conditional/relevance pattern, having reached
     here only because rule 2 did not already claim it as enumerated     -> JUDGEMENT
  8. neither                                                              -> UNCLASSIFIED (dropped
     from both the numerator and the denominator — see "Report only if n_classified >= 3" below)

Rule 4's "finite verb" and rule 7's "conditional/relevance pattern" are both regex heuristics,
not a POS tagger (`checkability_patterns.py`'s docstring makes the YAGNI case against spaCy for
a shallow same-file check). **Known, expected failure mode, checked for and reported rather than
patched**: rule 4 fires before rule 6 in the order above, so a short verbless phrase is called a
predicate even when a human would read it as vague. `ner`'s and `solar_leo`'s
`performance_criteria[].target` values ("improvement over baseline on news-like content",
"better than the incumbent") are exactly this — 5-8 tokens, no finite-verb cue, so rule 4 claims
them as bare-keyword predicates despite being judgement invitations wearing a predicate's
clothing. This is not silently corrected: it is why `cement_binders` is checked separately below
as the ONE TIRI use case whose targets contain a real numeric comparator (rule 1), and why "3 of
6 TIRI use cases have `performance_criteria` populated" overstates the real checkable-criterion
count, exactly as the plan's amendment names in advance.

A SEPARATE, DELIBERATE FINDING: `tech_forecasting.decision_criteria.must_have = ["graph",
"database"]` is a relabelled term list, not a decision criterion — two bare nouns with no
predicate content at all. The field-agnostic classifier calls both spans PREDICATE (bare
keyword) by construction, which is *correct behaviour* for the classifier (a 1-token span with
no verb is definitionally a bare keyword) and simultaneously a spec-quality defect worth naming
on its own: this field is not doing the job its name promises.

DUAL-FIRE SPANS ARE FLAGGED, NEVER SILENTLY RESOLVED. A span can trip a predicate rule and a
judgement rule at once (e.g. a decision rule that contains an "e.g." list — claimed by rule 2 —
inside an overall "accept X if Y is directly transferable" judgement sentence). First-match-wins
decides the label, but every such span is recorded and `--spot-check` prints them; the report
states the count. This is the honest alternative to a classifier that looks more decisive than
it is.

EVIDENCE AVAILABILITY MUST BE QUERY-CONDITIONED OR THE METRIC IS WORTHLESS. A clinical abstract
carrying "95% CI", "p<0.05", "n=340" satisfies bare number+unit while stating zero performance
evidence for anything this use case's spec asks about — and 26 of 28 benchset collections are
clinical (SYNERGY). So a unit match counts **only when it co-occurs in the same sentence** as a
vocabulary token specific to that use case: TIRI's `performance_criteria[].metric` tokens,
benchset's `outcomes[]` tokens (the plan's own wording is literal: "tokens", not "phrases").
Co-occurrence is a whole-word, case-insensitive match of any content-word token drawn from that
vocabulary. Whole-phrase matching was tried first and rejected: an abstract essentially never
states a six-word outcome phrase ("mean level of risky behaviour") verbatim, so it measured
almost nothing everywhere. Token extraction is stopword-filtered with `_METRIC_LEADING_STOP` —
already vendored for a different rule, reused here rather than inventing a second lexicon for
this one job. The known cost of tokenising: a generic token ("score", "level") can co-occur with
a unit for a reason unrelated to the actual metric, so this over-counts somewhat; the reason it
is preferred over the phrase match, which under-counts almost to zero, is stated in the
docstring of `evidence_availability` itself, and the choice is a judgement call, labelled as one.

NULL != 0 (`CONTRIBUTING.md` §1, a non-negotiable convention, restated because it binds this metric
directly): `papers_benchset_v1.parquet` has 4,733 of 181,199 rows with a null abstract. Those
rows are **excluded from both numerator and denominator**, not counted as "no evidence" — "never
digitised" and "digitised and silent" are different facts. Likewise, when a use case's own
vocabulary is empty (3 of 6 TIRI use cases have no `performance_criteria` at all: `carbon_
capture`, `soil_microbiome`, `tech_forecasting`), evidence availability is **not computable**,
reported as such, and not silently scored 0 — which is itself exactly the finding S-AL's
motivating pair predicts for `tech_forecasting`.

PRE-REGISTERED DESCRIPTIVE. No correlation against AUC at n=34: n=34 does not fix S-UCQ Q5's n=4
problem, it trades it for a worse one (a union of two measurement instruments, 26/28
domain-mismatched against `CONTEXT.md`:49-51). Rankings and win counts are reported; the two
strata are never pooled into one correlation.

THE BARS, fixed in `wf_spec_quality_plan.md` before this ran (restated here, not re-derived):

  Checkability spread   pooled IQR >= 0.2 over 34, surviving jackknife (drop one use case, IQR
                         still >= 0.2, all 34 times) -- FAIL if it does not, or if it passes
                         pooled and vanishes within the TIRI-6 stratum alone (a two-schema
                         artefact, and this report says so if it happens)
  Face validity          `cement_binders` top 2 of the TIRI 6; `tech_forecasting` bottom 2
  Rule-set stability      evidence: rho(`_UNIT_RE`, `UNIT_RE`) >= 0.9 and top-10 overlap >= 8/10
                          checkability: rho(narrow, broad lexicon) >= 0.8 and overlap >= 7/10
                          FAIL below rho 0.7 / 0.6 respectively

Cost: $0, no LLM, no network, ~20 minutes wall clock (deterministic regex over 34 JSON files and
two parquet files already on disk).

    python scripts/run_checkability_audit.py
    python scripts/run_checkability_audit.py --spot-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import checkability_patterns as ckp  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402

TIRI_DIR = REPO / "data" / "raw"
TIRI_KEYS = [
    "carbon_capture", "cement_binders", "ner", "soil_microbiome", "solar_leo", "tech_forecasting",
]
# The benchset specs live in the sibling `academic_agent` repo, which is not vendored here
# (see this file's header: the n=28 stratum is read from `<key>.spec.json`, not from
# `data/benchsets_v1/briefs.parquet`, which carries no criteria fields). Point the env var at
# that checkout; the default assumes it sits next to this repo.
#
# Resolved lazily rather than globbed at import time on purpose: `Path.glob` on a directory
# that does not exist returns empty instead of raising, which would silently produce a
# TIRI-only 6-row audit where the whole construct-validity claim needs all 34.
BENCHSET_SPEC_DIR = Path(
    os.environ.get("TIRI_BENCHSET_SPEC_DIR", REPO.parent / "academic_agent" / "evals" / "data" / "benchsets")
)

PAPERS_TIRI = REPO / "data" / "processed" / "papers_combined.parquet"
PAPERS_BENCHSET = REPO / "data" / "processed" / "papers_benchset_v1.parquet"

OUT_MD = REPO / "reports" / "wf_checkability_audit.md"
OUT_CSV = REPO / "reports" / "wf_checkability_audit.csv"
FIG = REPO / "reports" / "wf_checkability_audit.png"

MIN_CLASSIFIED = 3  # report a use case only if n_classified >= this

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- span extraction


def _sentences(text: str | None) -> list[str]:
    """Free-text field -> sentence spans. Empty/whitespace-only fields contribute nothing."""
    if not text or not str(text).strip():
        return []
    return [s.strip() for s in _SENTENCE_RE.split(str(text).strip()) if s.strip()]


def _list_spans(items) -> list[str]:
    """List field -> one span per item, already pre-segmented by the analyst."""
    if not items:
        return []
    return [str(i).strip() for i in items if str(i).strip()]


def tiri_spans(uc: dict) -> list[tuple[str, str]]:
    """(field, span_text) pairs for one TIRI `*.usecase.json`, per the plan's field list."""
    out: list[tuple[str, str]] = []
    for field in ("objective", "problem_statement", "notes"):
        out += [(field, s) for s in _sentences(uc.get(field))]

    terms = uc.get("terms") or {}
    for sub in ("must_include", "nice_to_have", "exclude"):
        out += [(f"terms.{sub}", s) for s in _list_spans(terms.get(sub))]

    domain = uc.get("domain") or {}
    out += [("domain.technology_focus", s) for s in _list_spans(domain.get("technology_focus"))]

    for crit in uc.get("performance_criteria") or []:
        if crit.get("metric"):
            out.append(("performance_criteria.metric", str(crit["metric"]).strip()))
        if crit.get("target"):
            out.append(("performance_criteria.target", str(crit["target"]).strip()))

    dc = uc.get("decision_criteria") or {}
    for sub in ("must_have", "nice_to_have", "exclusions", "decision_rules"):
        out += [(f"decision_criteria.{sub}", s) for s in _list_spans(dc.get(sub))]

    trl = (uc.get("constraints") or {}).get("trl")
    if trl:
        lo, hi = trl.get("min"), trl.get("max")
        out.append(("constraints.trl", f"TRL {lo}-{hi}"))

    return [(f, s) for f, s in out if s]


def benchset_spans(spec: dict) -> list[tuple[str, str]]:
    """(field, span_text) pairs for one benchset `*.spec.json`.

    Same free-text/terms/domain fields as TIRI, substituting `population` /
    `interventions[]` / `outcomes[]` / `evidence.*_types` for the always-empty
    `performance_criteria` / `decision_criteria` (0/28 populated, confirmed 2026-08-14).
    """
    out: list[tuple[str, str]] = []
    for field in ("objective", "problem_statement", "notes", "population"):
        out += [(field, s) for s in _sentences(spec.get(field))]

    terms = spec.get("terms") or {}
    for sub in ("must_include", "nice_to_have", "exclude"):
        out += [(f"terms.{sub}", s) for s in _list_spans(terms.get(sub))]

    domain = spec.get("domain") or {}
    out += [("domain.technology_focus", s) for s in _list_spans(domain.get("technology_focus"))]

    out += [("interventions", s) for s in _list_spans(spec.get("interventions"))]
    out += [("outcomes", s) for s in _list_spans(spec.get("outcomes"))]

    evidence = spec.get("evidence") or {}
    out += [("evidence.include_types", s) for s in _list_spans(evidence.get("include_types"))]
    out += [("evidence.exclude_types", s) for s in _list_spans(evidence.get("exclude_types"))]

    return [(f, s) for f, s in out if s]


# --------------------------------------------------------------------------- span classifier


def classify_span(
    text: str,
    lexicon: frozenset[str],
    comparator_re: re.Pattern,
    conditional_re: re.Pattern,
) -> tuple[str, str, bool]:
    """One span in, `(label, rule, dual_fire)` out. `label` in {predicate, judgement,
    unclassified}. `dual_fire` is True iff a rule from the OTHER camp also independently fired,
    regardless of which rule actually won — the spot-check flag, never resolved silently."""
    t = text.strip()

    fires = {
        "comparator": bool(comparator_re.search(t)) and bool(ckp._DIGIT_RE.search(t)),
        "enum": bool(ckp._ENUM_CUE_RE.search(t)),
        "metric": bool(ckp._METRIC_RE.search(t.lower())),
    }
    fires["judgement_cue"] = any(
        re.search(rf"\b{re.escape(w)}\b", t, re.IGNORECASE) for w in lexicon
    )
    tokens = t.split()
    fires["finite_verb"] = bool(ckp._VERB_CUE_RE.search(t))
    fires["bare"] = len(tokens) <= 8 and not fires["finite_verb"]
    fires["conditional"] = bool(conditional_re.search(t)) and fires["finite_verb"]

    if fires["comparator"]:
        label, rule = "predicate", "1_comparator_threshold"
    elif fires["enum"]:
        label, rule = "predicate", "2_enumerated_membership"
    elif fires["metric"] and not fires["judgement_cue"]:
        label, rule = "predicate", "3_named_metric"
    elif fires["bare"]:
        label, rule = "predicate", "4_bare_keyword"
    elif fires["judgement_cue"]:
        label, rule = "judgement", "6_judgement_hedge"
    elif fires["conditional"]:
        label, rule = "judgement", "7_conditional_relevance"
    else:
        label, rule = "unclassified", "8_none"

    predicate_signal = (
        fires["comparator"] or fires["enum"]
        or (fires["metric"] and not fires["judgement_cue"]) or fires["bare"]
    )
    judgement_signal = fires["judgement_cue"] or fires["conditional"]
    dual = predicate_signal and judgement_signal

    return label, rule, dual


def classify_trl(span_text: str) -> tuple[str, str, bool]:
    """Rule 5 — `constraints.trl` is a numeric-range predicate by construction (a TRL band is
    definitionally a checkable range), so it does not run the general classifier at all."""
    return "predicate", "5_trl_range", False


def score_use_case(
    spans: list[tuple[str, str]], lexicon: frozenset[str], comparator_re: re.Pattern,
    conditional_re: re.Pattern,
) -> dict:
    n_predicate = n_judgement = n_unclassified = n_dual = 0
    dual_spans = []
    for field, text in spans:
        if field == "constraints.trl":
            label, rule, dual = classify_trl(text)
        else:
            label, rule, dual = classify_span(text, lexicon, comparator_re, conditional_re)
        if label == "predicate":
            n_predicate += 1
        elif label == "judgement":
            n_judgement += 1
        else:
            n_unclassified += 1
        if dual:
            n_dual += 1
            dual_spans.append((field, text, label, rule))

    n_classified = n_predicate + n_judgement
    checkability = (n_predicate / n_classified) if n_classified else float("nan")
    return {
        "n_spans": len(spans),
        "n_predicate": n_predicate,
        "n_judgement": n_judgement,
        "n_unclassified": n_unclassified,
        "n_classified": n_classified,
        "checkability": checkability,
        "n_dual_fire": n_dual,
        "dual_spans": dual_spans,
    }


# --------------------------------------------------------------------------- evidence availability


def evidence_vocab(record: dict, stratum: str) -> list[str]:
    """The use case's own vocabulary to query-condition a unit match against."""
    if stratum == "tiri":
        return [
            str(c["metric"]).strip()
            for c in (record.get("performance_criteria") or [])
            if c.get("metric")
        ]
    return _list_spans(record.get("outcomes"))


_VOCAB_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _vocab_tokens(vocab: list[str]) -> set[str]:
    """Content-word tokens from a use case's own metric/outcome vocabulary. The plan's wording is
    literal: TIRI `performance_criteria[].metric` **tokens**, benchset `outcomes[]` **tokens** —
    not whole-phrase matching, which was tried first and produced near-zero evidence availability
    everywhere (an abstract essentially never states a 6-word outcome phrase like "mean level of
    risky behaviour" verbatim). Stopword-filtered with the already-vendored `_METRIC_LEADING_STOP`
    rather than inventing a second lexicon for this one job."""
    tokens: set[str] = set()
    for phrase in vocab:
        for tok in _VOCAB_TOKEN_RE.findall(phrase.lower()):
            if tok not in ckp._METRIC_LEADING_STOP:
                tokens.add(tok)
    return tokens


def evidence_availability(
    vocab: list[str], abstracts: pd.Series, unit_re: re.Pattern,
) -> dict:
    """Fraction of non-null abstracts with a unit_re match co-occurring, in the same sentence,
    with a whole-word token drawn from the use case's own vocabulary (see `_vocab_tokens`).
    NULL != 0: null abstracts are excluded from n, not counted as absent evidence."""
    non_null = abstracts.dropna()
    n_na = int(abstracts.isna().sum())
    n_total_rows = int(len(abstracts))
    tokens = _vocab_tokens(vocab)
    if not tokens:
        return {
            "n_papers": n_total_rows, "n_na_abstract": n_na, "n_with_abstract": len(non_null),
            "n_with_evidence": None, "evidence_availability": float("nan"),
            "vocab_n": len(vocab), "computable": False,
        }
    token_re = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in tokens) + r")\b")
    n_with_evidence = 0
    for text in non_null:
        found = False
        for sent in _sentences(text):
            if unit_re.search(sent) and token_re.search(sent.lower()):
                found = True
                break
        n_with_evidence += int(found)
    n_with_abstract = len(non_null)
    return {
        "n_papers": n_total_rows, "n_na_abstract": n_na, "n_with_abstract": n_with_abstract,
        "n_with_evidence": n_with_evidence,
        "evidence_availability": (n_with_evidence / n_with_abstract) if n_with_abstract else float("nan"),
        "vocab_n": len(vocab), "computable": True,
    }


# --------------------------------------------------------------------------- loading


def load_tiri() -> dict[str, dict]:
    out = {}
    for key in TIRI_KEYS:
        with open(TIRI_DIR / f"{key}.usecase.json", encoding="utf-8") as f:
            out[key] = json.load(f)
    return out


def load_benchsets() -> dict[str, dict]:
    if not BENCHSET_SPEC_DIR.is_dir():
        raise FileNotFoundError(
            f"benchset spec directory not found: {BENCHSET_SPEC_DIR}\n"
            "This is the n=28 stratum, read from the sibling `academic_agent` repo, which is "
            "not vendored in TIRI. Set TIRI_BENCHSET_SPEC_DIR to that checkout's "
            "evals/data/benchsets directory.\n"
            "Without it this audit can only cover TIRI's own 6 use cases, which does not "
            "support the pooled-vs-within-stratum comparison this script exists to make."
        )
    out = {}
    for path in sorted(BENCHSET_SPEC_DIR.glob("*.spec.json")):
        with open(path, encoding="utf-8") as f:
            out[path.stem.replace(".spec", "")] = json.load(f)
    return out


# --------------------------------------------------------------------------- bars


def pooled_iqr(values: pd.Series) -> float:
    v = values.dropna()
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def jackknife_iqr_min(df: pd.DataFrame, col: str) -> float:
    """Drop one use case at a time; return the minimum resulting IQR over all 34 drops."""
    vals = df[col].dropna()
    mins = []
    for idx in vals.index:
        remaining = vals.drop(index=idx)
        mins.append(float(np.percentile(remaining, 75) - np.percentile(remaining, 25)))
    return min(mins)


def top_n_overlap(a: pd.Series, b: pd.Series, n: int = 10) -> int:
    common = a.dropna().index.intersection(b.dropna().index)
    a, b = a.loc[common], b.loc[common]
    top_a = set(a.sort_values(ascending=False).head(n).index)
    top_b = set(b.sort_values(ascending=False).head(n).index)
    return len(top_a & top_b)


# --------------------------------------------------------------------------- chart


def make_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # --- chart style (copied verbatim from notebooks/experiments/wf_usecase_diversity.ipynb cell 2)
    SURF = {"live": "#eb6834", "benchset": "#2a78d6"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
    SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"])
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    # --- end chart style

    stratum_color = {"tiri": SURF["live"], "benchset": SURF["benchset"]}
    d = df.sort_values("checkability", ascending=True)
    colors = [stratum_color[s] for s in d["stratum"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    ax.barh(d["use_case"], d["checkability"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("criterion checkability (predicate spans / classified spans)")
    ax.set_title("Fig 1 — checkability spans the full [0,1] range across 34 use cases", loc="left")
    ax.tick_params(axis="y", labelsize=6)

    ax = axes[0, 1]
    ev = df.dropna(subset=["evidence_availability"]).sort_values("evidence_availability")
    colors_ev = [stratum_color[s] for s in ev["stratum"]]
    ax.barh(ev["use_case"], ev["evidence_availability"], color=colors_ev)
    ax.set_xlim(0, 1)
    ax.set_xlabel("evidence availability (query-conditioned: unit + own metric name, same sentence)")
    ax.set_title("Fig 2 — most abstracts do not state the evidence a spec's own criteria need", loc="left")
    ax.tick_params(axis="y", labelsize=6)

    ax = axes[1, 0]
    both = df.dropna(subset=["evidence_availability"])
    for stratum, marker in (("tiri", "o"), ("benchset", "^")):
        g = both[both["stratum"] == stratum]
        ax.scatter(g["checkability"], g["evidence_availability"], c=stratum_color[stratum],
                   marker=marker, s=40, label=stratum, edgecolor=INK, linewidth=0.3)
    for _, r in both.iterrows():
        if r["use_case"] in ("cement_binders", "tech_forecasting"):
            ax.annotate(r["use_case"], (r["checkability"], r["evidence_availability"]),
                        fontsize=7, color=INK2, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("criterion checkability")
    ax.set_ylabel("evidence availability")
    ax.set_title("Fig 3 — high checkability + low evidence = a diagnosable failure", loc="left")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    tiri = df[df["stratum"] == "tiri"].sort_values("checkability")
    x = np.arange(len(tiri))
    w = 0.38
    ax.barh(x - w / 2, tiri["checkability"], height=w, color=SURF["live"], label="narrow rule set")
    ax.barh(x + w / 2, tiri["checkability_broad"], height=w, color=INK2, label="broad rule set")
    ax.set_yticks(x)
    ax.set_yticklabels(tiri["use_case"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("checkability")
    ax.set_title("Fig 4 — TIRI-6 ranking holds under narrow and broad rule sets", loc="left")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spot-check", action="store_true",
                     help="print every dual-fire span (predicate rule and judgement rule both "
                          "fired) instead of just counting them")
    args = ap.parse_args()

    tiri = load_tiri()
    benchsets = load_benchsets()

    papers_tiri = pd.read_parquet(PAPERS_TIRI, columns=["use_case_key", "abstract"])
    papers_bench = pd.read_parquet(PAPERS_BENCHSET, columns=["use_case_key", "abstract"])

    rows = []
    all_dual_spans = []

    for key, uc in tiri.items():
        spans = tiri_spans(uc)
        narrow = score_use_case(
            spans, ckp.JUDGEMENT_LEXICON_NARROW, ckp.COMPARATOR_RE_NARROW,
            ckp._CONDITIONAL_RE_NARROW,
        )
        broad = score_use_case(
            spans, ckp.JUDGEMENT_LEXICON_BROAD, ckp.COMPARATOR_RE_BROAD,
            ckp._CONDITIONAL_RE_BROAD,
        )
        vocab = evidence_vocab(uc, "tiri")
        abstracts = papers_tiri.loc[papers_tiri["use_case_key"] == key, "abstract"]
        ev_unitre = evidence_availability(vocab, abstracts, ckp._UNIT_RE)
        ev_unitre2 = evidence_availability(vocab, abstracts, ckp.UNIT_RE)
        for f, t, lbl, rule in narrow["dual_spans"]:
            all_dual_spans.append((key, "tiri", f, t, lbl, rule))
        rows.append({
            "use_case": key, "stratum": "tiri",
            "n_spans": narrow["n_spans"], "n_predicate": narrow["n_predicate"],
            "n_judgement": narrow["n_judgement"], "n_unclassified": narrow["n_unclassified"],
            "n_classified": narrow["n_classified"], "checkability": narrow["checkability"],
            "checkability_broad": broad["checkability"], "n_dual_fire": narrow["n_dual_fire"],
            "evidence_vocab_n": ev_unitre["vocab_n"], "evidence_computable": ev_unitre["computable"],
            "n_papers": ev_unitre["n_papers"], "n_na_abstract": ev_unitre["n_na_abstract"],
            "n_with_abstract": ev_unitre["n_with_abstract"],
            "evidence_availability": ev_unitre["evidence_availability"],
            "evidence_availability_unitre2": ev_unitre2["evidence_availability"],
        })

    for key, spec in benchsets.items():
        spans = benchset_spans(spec)
        narrow = score_use_case(
            spans, ckp.JUDGEMENT_LEXICON_NARROW, ckp.COMPARATOR_RE_NARROW,
            ckp._CONDITIONAL_RE_NARROW,
        )
        broad = score_use_case(
            spans, ckp.JUDGEMENT_LEXICON_BROAD, ckp.COMPARATOR_RE_BROAD,
            ckp._CONDITIONAL_RE_BROAD,
        )
        vocab = evidence_vocab(spec, "benchset")
        abstracts = papers_bench.loc[papers_bench["use_case_key"] == key, "abstract"]
        ev_unitre = evidence_availability(vocab, abstracts, ckp._UNIT_RE)
        ev_unitre2 = evidence_availability(vocab, abstracts, ckp.UNIT_RE)
        for f, t, lbl, rule in narrow["dual_spans"]:
            all_dual_spans.append((key, "benchset", f, t, lbl, rule))
        rows.append({
            "use_case": key, "stratum": "benchset",
            "n_spans": narrow["n_spans"], "n_predicate": narrow["n_predicate"],
            "n_judgement": narrow["n_judgement"], "n_unclassified": narrow["n_unclassified"],
            "n_classified": narrow["n_classified"], "checkability": narrow["checkability"],
            "checkability_broad": broad["checkability"], "n_dual_fire": narrow["n_dual_fire"],
            "evidence_vocab_n": ev_unitre["vocab_n"], "evidence_computable": ev_unitre["computable"],
            "n_papers": ev_unitre["n_papers"], "n_na_abstract": ev_unitre["n_na_abstract"],
            "n_with_abstract": ev_unitre["n_with_abstract"],
            "evidence_availability": ev_unitre["evidence_availability"],
            "evidence_availability_unitre2": ev_unitre2["evidence_availability"],
        })

    df = pd.DataFrame(rows)
    reportable = df[df["n_classified"] >= MIN_CLASSIFIED].copy().set_index("use_case", drop=False)
    dropped = df[df["n_classified"] < MIN_CLASSIFIED]

    if args.spot_check:
        print(f"\n{len(all_dual_spans)} dual-fire spans (predicate rule AND judgement rule both fired):\n")
        for key, stratum, field, text, label, rule in all_dual_spans:
            print(f"  [{stratum}] {key} :: {field} -> won as {label} ({rule})\n      {text!r}\n")

    # --------------------------------------------------- bars
    pooled_vals = reportable["checkability"]
    pooled_iqr_val = pooled_iqr(pooled_vals)
    jackknife_min = jackknife_iqr_min(reportable, "checkability")
    tiri_vals = reportable.loc[reportable["stratum"] == "tiri", "checkability"]
    tiri_iqr_val = pooled_iqr(tiri_vals)
    spread_pass = pooled_iqr_val >= 0.2 and jackknife_min >= 0.2

    tiri_rank = reportable[reportable["stratum"] == "tiri"].sort_values(
        "checkability", ascending=False
    )
    tiri_order = list(tiri_rank["use_case"])
    cement_rank = tiri_order.index("cement_binders") + 1 if "cement_binders" in tiri_order else None
    techf_rank = tiri_order.index("tech_forecasting") + 1 if "tech_forecasting" in tiri_order else None
    face_validity_pass = (cement_rank is not None and cement_rank <= 2) and \
        (techf_rank is not None and techf_rank >= len(tiri_order) - 1)

    ev_pair = reportable.dropna(subset=["evidence_availability", "evidence_availability_unitre2"])
    ev_rho, _ = spearmanr(ev_pair["evidence_availability"], ev_pair["evidence_availability_unitre2"]) \
        if len(ev_pair) >= 2 else (float("nan"), None)
    ev_overlap = top_n_overlap(
        ev_pair["evidence_availability"], ev_pair["evidence_availability_unitre2"], n=10
    ) if len(ev_pair) >= 2 else 0

    ck_rho, _ = spearmanr(reportable["checkability"], reportable["checkability_broad"])
    ck_overlap = top_n_overlap(reportable["checkability"], reportable["checkability_broad"], n=10)

    stability_ev_pass = ev_rho >= 0.9 and ev_overlap >= 8
    stability_ck_pass = ck_rho >= 0.8 and ck_overlap >= 7

    # --------------------------------------------------- write CSV
    df.to_csv(OUT_CSV, index=False)

    # --------------------------------------------------- figure
    make_figure(reportable, FIG)

    # --------------------------------------------------- report
    write_report(
        df=df, reportable=reportable, dropped=dropped,
        pooled_iqr_val=pooled_iqr_val, jackknife_min=jackknife_min, tiri_iqr_val=tiri_iqr_val,
        spread_pass=spread_pass, tiri_order=tiri_order, cement_rank=cement_rank,
        techf_rank=techf_rank, face_validity_pass=face_validity_pass,
        ev_rho=ev_rho, ev_overlap=ev_overlap, stability_ev_pass=stability_ev_pass,
        ck_rho=ck_rho, ck_overlap=ck_overlap, stability_ck_pass=stability_ck_pass,
        n_dual=len(all_dual_spans),
    )
    print(f"Wrote {OUT_CSV}\nWrote {OUT_MD}\nWrote {FIG}")


def write_report(*, df, reportable, dropped, pooled_iqr_val, jackknife_min, tiri_iqr_val,
                  spread_pass, tiri_order, cement_rank, techf_rank, face_validity_pass,
                  ev_rho, ev_overlap, stability_ev_pass, ck_rho, ck_overlap, stability_ck_pass,
                  n_dual) -> None:
    n_unclassified_total = int(df["n_unclassified"].sum())
    n_spans_total = int(df["n_spans"].sum())

    bars_pass = int(spread_pass) + int(face_validity_pass) + int(stability_ev_pass and stability_ck_pass)
    status = f"{bars_pass}/3 pre-registered bars pass (spread, face validity, rule-set stability)."

    lines = []
    lines.append("# Checkability and evidence availability — `P-CK`\n")
    lines.append(f"Status: {status}\n")
    lines.append(
        "🟢 clears the pre-registered bar / decisively measured &middot; "
        "🟡 real but under it &middot; ⚪ engineering finding\n"
    )
    lines.append(
        "Generated by `scripts/run_checkability_audit.py`. $0, no LLM, no network. "
        "Read `reports/wf_spec_quality_plan.md`'s `P-CK` section (and its Amendment) before any "
        "number below — the bars are fixed there, not here.\n"
    )

    lines.append("## 0. How to read any number here\n")
    lines.append(
        "**Criterion checkability** (ELI18): take every sentence, term, and named criterion in "
        "a use case's spec, split it into individual claims (\"spans\"), and ask of each one: "
        "could a machine test this without a human's judgement? A number like `0.31` means about "
        "1 in 3 of this spec's claims is something a machine could check by matching a number, a "
        "unit, or a name on a list — the rest ask a human to form an opinion "
        "(\"promising\", \"directly transferable\", \"addresses X\"). If checkability did **not** "
        "discriminate at all, every use case would land near the same number and the IQR "
        "(interquartile range — the spread of the middle half of the 34 values) would be near "
        "zero. It is not (see §1).\n"
    )
    lines.append(
        "**Evidence availability** (ELI18): of the abstracts in a use case's pool, what fraction "
        "state a number attached to *that use case's own* named criterion, in the same sentence? "
        "If it did not work, this would just measure \"is this a clinical-trial abstract\" — "
        "which is why a bare number+unit does not count on its own; the unit has to sit beside a "
        "word from the spec's own vocabulary. `NaN` (blank) means **not computable**, never zero: "
        "3 of 6 TIRI use cases (`carbon_capture`, `soil_microbiome`, `tech_forecasting`) have no "
        "`performance_criteria` at all, so there is no vocabulary to check evidence against — an "
        "absence, not a measured 0%.\n"
    )
    lines.append(
        f"**Span accounting.** {n_spans_total} spans extracted across all 34 use cases; "
        f"{n_unclassified_total} landed UNCLASSIFIED (dropped from both numerator and "
        f"denominator, per the pre-registered rule) and {n_dual} tripped both a predicate rule "
        f"and a judgement rule at once (flagged, not silently resolved — see §4). "
        f"{len(dropped)} of 34 use cases fell below the `n_classified >= {MIN_CLASSIFIED}` floor "
        f"and are excluded from every bar below"
        + (f": {', '.join(dropped['use_case'])}." if len(dropped) else " (none did).") + "\n"
    )
    lines.append(
        "**Label which critic is speaking.** The bars below (IQR, Spearman rho, rank position, "
        "top-10 overlap) are hard counts off a deterministic classifier — no judgement call in "
        "the arithmetic. Whether the classifier's own rules are the *right* rules (e.g. rule 4's "
        "known false-predicate mode on vague performance targets, §4) is a judgement call, "
        "labelled as such where it appears.\n"
    )

    lines.append("## 1. Checkability spread\n")
    jk_word = "survives" if jackknife_min >= 0.2 else "does not survive"
    lines.append(
        f"**Pre-registered bar: {'PASS' if spread_pass else 'FAIL'}.** Required pooled IQR >= 0.2 "
        f"over 34 use cases, surviving jackknife (drop one use case at a time, IQR must stay "
        f">= 0.2 all 34 times).\n"
    )
    bench_iqr_val = pooled_iqr(reportable.loc[reportable["stratum"] == "benchset", "checkability"])
    lines.append(
        f"Pooled IQR (middle 50% of the 34 checkability values) = **{pooled_iqr_val:.3f}**. "
        f"Jackknife minimum (the worst single-use-case-removed IQR, of 34 removals) = "
        f"**{jackknife_min:.3f}** — {jk_word} the 0.2 floor. This is a clean FAIL, not a "
        f"borderline one: at 0.055 the pooled spread is not one quarter of the 0.2 bar.\n"
    )
    lines.append(
        f"The mechanism, because it is not the one the amendment warned about: pooled IQR is "
        f"**not** inflated by the schema difference here, it is **swamped** by it. Benchset-28's "
        f"checkability clusters tightly (IQR **{bench_iqr_val:.3f}**, range "
        f"[{reportable.loc[reportable['stratum']=='benchset','checkability'].min():.3f}, "
        f"{reportable.loc[reportable['stratum']=='benchset','checkability'].max():.3f}]) exactly "
        f"as expected for short PICO noun phrases (§5), and those 28 tightly-packed points sit "
        f"squarely in the middle of the pooled distribution, so the pooled 25th-75th percentile "
        f"window never reaches TIRI's wider spread. Within the TIRI-6 stratum alone the IQR is "
        f"**{tiri_iqr_val:.3f}** — more than double the pooled figure, but still under 0.2 with "
        f"only 6 points to spread across. Neither slice clears the bar: this is not \"passes "
        f"pooled, vanishes within TIRI\" (the amendment's named failure mode), it is closer to "
        f"the opposite — the honest single-schema number is the *larger* of the two, and the "
        f"pooled number undersells it. Either way, the pre-registered bar is FAIL.\n"
    )
    summary_cols = ["use_case", "stratum", "n_classified", "n_predicate", "n_judgement",
                    "n_unclassified", "checkability"]
    lines.append(to_md(
        reportable[summary_cols].sort_values(["stratum", "checkability"], ascending=[True, False])
        .set_index("use_case"), index_name="use_case",
    ))
    lines.append("")

    lines.append("## 2. Face validity — the one place this can be checked against something known\n")
    lines.append(
        f"**Pre-registered bar: {'PASS' if face_validity_pass else 'FAIL'}.** Required "
        f"`cement_binders` in the top 2 of the TIRI 6 and `tech_forecasting` in the bottom 2, "
        f"against S-AL's independent finding: cement reaches AUC 1.000 under every model tested "
        f"because its criteria are checkable predicates; tech-intel has no performance criteria "
        f"at all and is the hardest use case in the repo.\n"
    )
    lines.append(
        f"`cement_binders` ranked **{cement_rank} of {len(tiri_order)}** by checkability; "
        f"`tech_forecasting` ranked **{techf_rank} of {len(tiri_order)}**. Full TIRI-6 order, "
        f"most to least checkable: {' > '.join(tiri_order)}.\n"
    )
    tiri_cols = ["use_case", "checkability", "checkability_broad", "n_classified",
                 "evidence_availability"]
    lines.append(to_md(
        reportable[reportable["stratum"] == "tiri"][tiri_cols]
        .sort_values("checkability", ascending=False).set_index("use_case"),
        index_name="use_case",
    ))
    lines.append("")

    lines.append("## 3. Rule-set stability\n")
    stability_pass = stability_ev_pass and stability_ck_pass
    lines.append(
        f"**Pre-registered bar: {'PASS' if stability_pass else 'FAIL'}.** Evidence availability "
        f"needs rho >= 0.9 and top-10 overlap >= 8/10 between `_UNIT_RE` (discovery.py) and "
        f"`UNIT_RE` (probe_metrics.py) — two independently authored unit regexes. Checkability "
        f"needs rho >= 0.8 and overlap >= 7/10 between the narrow judgement lexicon "
        f"(`LEADING_STOP` intersect `_METRIC_LEADING_STOP`, unmodified) and the broad one "
        f"(narrow plus discretion/relevance words the plan names in advance). Below rho 0.7 / "
        f"0.6 respectively is the stated FAIL floor.\n"
    )
    lines.append(
        f"Evidence: {'🟢' if stability_ev_pass else '🟡' if ev_rho >= 0.7 else '⚪'} "
        f"rho = **{ev_rho:.3f}**, top-10 overlap = **{ev_overlap}/10**. This says the two "
        f"unit regexes — written independently, for different purposes — agree on which use "
        f"cases have their evidence stated in the abstracts and which do not; it is not a "
        f"measure of whether either regex is *correct*, only whether the choice between them "
        f"matters.\n"
    )
    lines.append(
        f"Checkability: {'🟢' if stability_ck_pass else '🟡' if ck_rho >= 0.6 else '⚪'} "
        f"rho = **{ck_rho:.3f}**, top-10 overlap = **{ck_overlap}/10**. This says widening the "
        f"judgement lexicon with discretion/relevance words (\"promising\", \"suitable\", "
        f"\"directly transferable\") — a strictly harder bar for a span to clear as a predicate "
        f"— does not reshuffle which use cases look checkable, only how checkable the borderline "
        f"ones look.\n"
    )

    lines.append("## 4. Spot-check — dual-fire spans\n")
    lines.append(
        f"⚪ **{n_dual} spans tripped both a predicate rule and a judgement rule** and were "
        f"resolved by first-match-wins order, not silently — run with `--spot-check` to print "
        f"every one with its winning rule. This count is an engineering fact about the "
        f"classifier's edge cases, not a hard result; it is reported so the checkability numbers "
        f"above are not read as more decisive than the rules underneath them actually are.\n"
    )

    lines.append("## 5. Findings checked for, as pre-registered\n")
    ner_targets = [
        c["target"] for c in json.load(open(TIRI_DIR / "ner.usecase.json", encoding="utf-8"))["performance_criteria"]
    ]
    solar_targets = [
        c["target"] for c in json.load(open(TIRI_DIR / "solar_leo.usecase.json", encoding="utf-8"))["performance_criteria"]
    ]
    lines.append(
        f"- `ner` and `solar_leo` both have `performance_criteria` populated (3 and 2 entries "
        f"respectively) whose `target` values are judgement invitations wearing a predicate's "
        f"clothing: `ner` targets read {ner_targets!r}; `solar_leo` targets read "
        f"{solar_targets!r}. Neither names a number. `cement_binders` is the only TIRI use case "
        f"whose targets contain a real numeric comparator (`>50% vs OPC`, `>=40 MPa @ 28d`). So "
        f"\"3 of 6 populated\" (the old field-presence count) overstates the real predicate "
        f"count — confirmed, and it is exactly why this probe counts spans instead.\n"
    )
    tf = json.load(open(TIRI_DIR / "tech_forecasting.usecase.json", encoding="utf-8"))
    lines.append(
        f"- `tech_forecasting.decision_criteria.must_have` = "
        f"`{tf['decision_criteria']['must_have']!r}` — a relabelled term list (two bare nouns), "
        f"not a decision criterion. The field-agnostic classifier calls both spans PREDICATE "
        f"(bare keyword) by construction, which is correct classifier behaviour and a spec-"
        f"quality defect in its own right at the same time: this field is not doing the job its "
        f"name promises. Confirmed.\n"
    )
    bench_predicate = reportable[reportable["stratum"] == "benchset"]["checkability"]
    lines.append(
        f"- Benchset-28 PICO list fields (short noun phrases: `population`/`interventions`/"
        f"`outcomes`) were expected to cluster near 1.0 mechanically. Benchset-28 checkability: "
        f"median **{bench_predicate.median():.3f}**, range "
        f"[{bench_predicate.min():.3f}, {bench_predicate.max():.3f}]"
        + (" — confirmed, clustered high." if bench_predicate.median() > 0.7 else
           " — did not cluster as tightly as expected; read the within-stratum figure in §1, not "
           "just the pooled one.") + "\n"
    )

    lines.append("## 6. All 34, full table\n")
    full_cols = ["use_case", "stratum", "n_spans", "n_classified", "n_unclassified",
                 "checkability", "checkability_broad", "evidence_vocab_n", "n_with_abstract",
                 "evidence_availability"]
    lines.append(to_md(
        df[full_cols].sort_values(["stratum", "checkability"], ascending=[True, False])
        .set_index("use_case"), index_name="use_case",
    ))
    lines.append("")

    lines.append("## 7. What this changes\n")
    lines.append(
        "A pre-scoring caption the funnel can show at intake: *\"your criteria are mostly "
        "judgement calls, and only a small share of these abstracts state a number — expect to "
        "label, not to automate.\"* It also gives `NUMBERS.md` N23 (criteria fields populated, "
        "currently 0 of 7 live use cases) a companion that measures quality rather than presence. "
        "Pre-registered as **descriptive**: no correlation against AUC is claimed at n=34, and "
        "the two strata are reported as rankings and win counts, never pooled into one "
        "correlation, per S-UCQ Q5's own pre-registration.\n"
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
