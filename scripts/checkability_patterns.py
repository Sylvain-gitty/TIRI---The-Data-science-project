"""Vendored regexes for `P-CK` (checkability), copied verbatim from the sibling repo.

WHY VENDORED RATHER THAN IMPORTED. `academic_agent` is read-only from here and lives at a
sibling path (`../academic_agent`), not a package on this repo's path. A `sys.path` hop across
repos is exactly the kind of cross-repo coupling `CONTRIBUTING.md` §8's sibling-repo rule
warns against — it would silently break the day the sibling repo moves or refactors a private
name. Copying four small, already-frozen regex objects is cheaper than that fragility, and it is
copying, not reimplementing: every object below is unchanged from its source, byte for byte
(only the leading underscore on private names and the module-level comments are TIRI's own).

SOURCES, verified against the sibling repo on 2026-08-14:

    academic_agent/discovery.py
        _UNIT_RE              ~line 1484  (a number + a real unit)
        METRIC_HEADS          ~line 611   (metric head nouns, frozenset)
        _METRIC_RE            ~line 1396  (<=1 modifier word before the head noun)
        _METRIC_LEADING_STOP  ~line 1400  (words that must not start a harvested phrase)

    academic_agent/spikes/review_seeding/probe_metrics.py
        UNIT_RE      ~line 97   (adds k\\b, micro-second/-metre, hours? over discovery.py's)
        METRIC_RE    ~line 81   (<=3 modifiers — the EARLIER, looser version)
        METRIC_HEADS ~line 68   (same 49 head nouns as discovery.py, as a list not a frozenset)
        LEADING_STOP ~line 85   (same list as discovery.py's `_METRIC_LEADING_STOP`)

WHY TWO COPIES OF THE "SAME" THING, DELIBERATELY. `discovery.py`'s comment at its line ~1375
(reproduced below) explains that `_METRIC_RE`'s <=1-modifier limit is a later, MEASURED
correction to `probe_metrics.py`'s earlier <=3-modifier version: at 3 modifiers both corpora
tested there produced sentence fragments ("especially for coal-based reduction") and a longer
match stole df from a shorter one. `_METRIC_RE` (discovery.py) is therefore what this module
uses for classification (rule 3 below). `probe_metrics.METRIC_RE` is kept only as the second,
independently-authored instrument for a check this probe's plan requires: `_UNIT_RE` vs.
`UNIT_RE` is a genuine "did two people who solved this separately agree" test
(`wf_spec_quality_plan.md`, P-CK's rule-set-stability bar), and `LEADING_STOP` /
`_METRIC_LEADING_STOP` intersected is the anti-gaming judgement lexicon — fixed because
neither list was authored with this probe's result in mind.

The verbatim `discovery.py` comment on why <=1 modifier beats <=3 (its line ~1375-1391):

    "AT MOST ONE modifier word before the head noun. WHY ONE, measured on two real corpora on
    2026-07-25 rather than argued. The spike's harvester allowed up to three, which is enough
    for "power conversion efficiency" — but this list is SHOWN to the analyst, and at two or
    three modifiers both corpora produce sentence fragments: `especially for coal-based
    reduction`, `foster sustainable direct reduction`, `incorporating renewable energy`. A
    longer match also STEALS from the shorter one (allowing three modifiers drops `efficiency`
    from df 6 to 5), so it is worse in two ways at once. [...] One modifier reproduces the
    vocabulary the spike reported for cement (durability, compressive strength, workability)
    and shows nothing embarrassing."

MIS-CITATION ON RECORD. `wf_spec_quality_plan.md`'s pre-amendment draft named
`spikes/concept_extraction/cues.py` as a reuse for this probe. Its own module docstring says
what it actually does: "regex patterns that detect the LINGUISTIC FRAMES authors use to
introduce a technology" — a novelty-introduction detector for technology NAMES, nothing to do
with performance predicates. Confirmed on 2026-08-14 and not used anywhere in this module.

A comparator regex (`COMPARATOR_RE` below) exists in **neither** sibling-repo file and is new
code, written for this probe alone.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Vendored verbatim from academic_agent/discovery.py
# ─────────────────────────────────────────────────────────────────────────────

# discovery.py ~line 611 — metric HEAD NOUNS a performance phrase can end in.
METRIC_HEADS = frozenset({
    "strength", "efficiency", "capacity", "accuracy", "yield", "selectivity", "conductivity",
    "density", "lifetime", "cost", "consumption", "emissions", "penalty", "loss", "latency",
    "throughput", "purity", "durability", "stability", "reduction", "rate", "time",
    "temperature", "pressure", "power", "energy", "resistance", "toughness", "permeability",
    "porosity", "absorption", "adsorption", "degradation", "conversion", "recovery",
    "performance", "voltage", "current", "sensitivity", "specificity", "mortality",
    "prevalence", "adherence", "bioavailability", "tolerance", "endurance", "retention",
    "shrinkage", "workability", "productivity", "metallization", "corrosion",
})

# discovery.py ~line 1400 — words that must never START a harvested metric phrase.
_METRIC_LEADING_STOP = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or", "as", "by",
    "this", "these", "those", "that", "their", "its", "our", "which", "such", "both", "from",
    "high", "higher", "low", "lower", "good", "better", "best", "improved", "increased",
    "decreased", "reduced", "excellent", "superior", "optimal", "significant", "various",
    "different", "several", "many", "most", "some", "other", "also", "however", "while",
    "review", "paper", "study", "studies", "research", "results", "article", "recent",
})

# discovery.py ~line 1396 — <=1 modifier before the head noun (see docstring above for why).
_METRIC_HEAD_PATTERN = "|".join(sorted(METRIC_HEADS, key=len, reverse=True))
_METRIC_RE = re.compile(rf"\b((?:[a-z][a-z0-9\-]{{2,}}\s+)?(?:{_METRIC_HEAD_PATTERN}))\b")

# discovery.py ~line 1484 — a number followed by a real unit.
_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|mpa|gpa|kpa|w/g|w/kg|kw|mw|gw|wh|kwh|mwh|mj|gj|kj|"
    r"mg/dl|mmol/l|mg|kg|g/l|ml|°c|tops/w|gops/w|pj|nj|ms|ns|hz|khz|mhz|ghz|"
    r"mol/kg|mmol/g|m2/g|cm2|nm|mm|years?|months?|weeks?|days?|cycles?)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Vendored verbatim from academic_agent/spikes/review_seeding/probe_metrics.py
# ─────────────────────────────────────────────────────────────────────────────

# probe_metrics.py ~line 68 — same 49 head nouns as discovery.py's METRIC_HEADS, as a list.
PROBE_METRIC_HEADS = [
    "strength", "efficiency", "capacity", "accuracy", "yield", "selectivity", "conductivity",
    "density", "lifetime", "cost", "consumption", "emissions", "penalty", "loss", "latency",
    "throughput", "purity", "durability", "stability", "reduction", "rate", "time",
    "temperature", "pressure", "power", "energy", "resistance", "toughness", "permeability",
    "porosity", "absorption", "adsorption", "degradation", "conversion", "recovery",
    "performance", "voltage", "current", "sensitivity", "specificity", "mortality",
    "prevalence", "adherence", "bioavailability", "tolerance", "endurance", "retention",
    "shrinkage", "workability", "productivity", "metallization", "corrosion",
]

# probe_metrics.py ~line 84 — same list as discovery.py's `_METRIC_LEADING_STOP`.
LEADING_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or", "as", "by",
    "this", "these", "those", "that", "their", "its", "our", "which", "such", "both", "from",
    "high", "higher", "low", "lower", "good", "better", "best", "improved", "increased",
    "decreased", "reduced", "excellent", "superior", "optimal", "significant", "various",
    "different", "several", "many", "most", "some", "other", "also", "however", "while",
    "review", "paper", "study", "studies", "research", "results", "article", "recent",
}

# probe_metrics.py ~line 80-81 — <=3 modifiers, the earlier/looser version. Not used for
# classification (see docstring); kept only so it exists to be cited if ever needed.
_PROBE_MODIFIER = r"(?:[a-z][a-z0-9\-]{2,}\s+){0,3}"
METRIC_RE = re.compile(rf"\b({_PROBE_MODIFIER}(?:{'|'.join(PROBE_METRIC_HEADS)}))\b")

# probe_metrics.py ~line 97 — adds k\b, micro-second/-metre, hours? over discovery.py's _UNIT_RE.
UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|mpa|gpa|kpa|w/g|w/kg|kw|mw|gw|wh|kwh|mwh|mj|gj|kj|"
    r"mg/dl|mmol/l|mg|kg|g/l|ml|°c|k\b|tops/w|gops/w|pj|nj|ms|µs|ns|hz|khz|mhz|ghz|"
    r"mol/kg|mmol/g|m2/g|cm2|nm|µm|mm|years?|months?|weeks?|days?|hours?|cycles?)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# New code for this probe — a comparator regex exists in neither sibling-repo file.
# ─────────────────────────────────────────────────────────────────────────────

# Narrow = ASCII comparators only (`wf_spec_quality_plan.md` P-CK bars: "Narrow = ASCII
# comparators and the intersected lexicon").
COMPARATOR_RE_NARROW = re.compile(
    r"(>=|<=|>|<|\bat least\b|\bno more than\b|\bno less than\b)", re.IGNORECASE,
)

# Broad = narrow plus the unicode comparator glyphs (>= / <= spelled as single characters).
COMPARATOR_RE_BROAD = re.compile(
    r"(>=|<=|≥|≤|>|<|\bat least\b|\bno more than\b|\bno less than\b)", re.IGNORECASE,
)

_DIGIT_RE = re.compile(r"\d")

# Rule 2 — "(e.g.|i.e.|such as)" introducing an enumerable list. No trailing `\b` on the
# period-ending alternatives: "." is a non-word char, so a `\b` right after it can never match
# when (as always) a space follows — `e.g. ` has no word/non-word transition at that point. Left
# as a leading `\b` only, which still anchors the match to a word boundary on the way in.
_ENUM_CUE_RE = re.compile(r"(?:\be\.g\.,?|\bi\.e\.,?|\bsuch as\b)", re.IGNORECASE)

# Rule 7's conditional/relevance pattern (narrow). Broad widens this with discretion/relevance
# words per the plan's stability check ("if, unless, addresses, relates to, promising, suitable,
# directly transferable").
_CONDITIONAL_RE_NARROW = re.compile(
    r"\b(if|unless|address(?:es)?|relates?\s+to|concerns?)\b", re.IGNORECASE,
)
_BROAD_EXTRA_WORDS = ("promising", "suitable", "directly transferable")
_CONDITIONAL_RE_BROAD = re.compile(
    r"\b(if|unless|address(?:es)?|relates?\s+to|concerns?|promising|suitable|"
    r"directly\s+transferable)\b",
    re.IGNORECASE,
)

# A small, curated set of finite-verb cues (auxiliaries/modals plus the common main verbs that
# actually appear in these 34 specs' decision-criteria prose). NOT a POS tagger — a POS tagger
# would need spaCy, a new dependency, for a shallow same-file heuristic; `cues.py`'s own
# docstring makes exactly this YAGNI argument for regex over parsing. The known failure mode is
# stated in the module docstring of `run_checkability_audit.py`, not hidden here.
_VERB_CUE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|am|has|have|had|does|do|did|"
    r"can|could|may|might|must|shall|should|will|would|"
    r"address(?:es)?|relates?|concerns?|requires?|reports?|presents?|covers?|includes?|"
    r"benchmarks?|evaluates?|identif(?:y|ies)|predicts?|explores?|reasons?|measures?|"
    r"improves?|operates?|sustains?|allows?|enables?|uses?|needs?|targets?|exceeds?|"
    r"achieves?|reaches?|focus(?:es)?|screens?|scouts?|addresses?)\b",
    re.IGNORECASE,
)

# The narrow judgement lexicon P-CK's bars fix in advance: `LEADING_STOP` intersected with
# `_METRIC_LEADING_STOP`. Both source lists turn out to be identical (same 43 words, authored in
# two different files) so the intersection is that list unchanged — worth stating because it
# means the "anti-gaming guard" is not narrowed by the intersection itself, only by the fact
# that neither list was authored with this probe's outcome in mind.
JUDGEMENT_LEXICON_NARROW = frozenset(LEADING_STOP) & frozenset(_METRIC_LEADING_STOP)

# Broad = narrow widened with discretion/relevance verbs, per the plan's stability-check bar.
JUDGEMENT_LEXICON_BROAD = JUDGEMENT_LEXICON_NARROW | frozenset(
    {"if", "unless", "addresses", "relates to", "promising", "suitable", "directly transferable"}
)
