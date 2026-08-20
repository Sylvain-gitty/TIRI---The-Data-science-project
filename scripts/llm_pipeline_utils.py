"""Shared building blocks for prompted-LLM paper screening.

Two things live here, and they came from different places:

1. The original template helpers for `notebooks/future_work/sf_llm_fold_pipeline.ipynb`
   (`build_use_case_brief`, `select_few_shot_examples`, `build_prompt`,
   `call_llm_stub`, `parse_llm_response`) - unchanged, still imported by that notebook.
2. The screening bake-off harness below (`render_brief`, `PROMPT_VARIANTS`,
   `OpenRouterClient`, `score_frame`), added for `scripts/run_llm_screening.py`.
   See `reports/wf_llm_screening_plan.md` for what it is measuring and why.

A prompted LLM doesn't consume a numeric feature matrix, so none of
`fold_pipeline_utils.py`'s `ColumnTransformer`/`Pipeline` machinery applies here - the
input is a text prompt, not `X`. What *does* carry over unchanged is the split-then-fit
discipline: a few-shot prompt's demonstration examples are the prompt-based equivalent of
a fitted parameter (they encode information the model conditions its answer on), so they
must be drawn only from the training fold, never from validation/holdout, for exactly the
same reason `StandardScaler` must be fit on the training fold only. `select_few_shot_examples`
below is where that rule is enforced. The pilot's three variants (P1/P2/P4) are all
zero-shot, so nothing in the pilot can leak - but keep that seam intact if few-shot (P3)
or the induced rule set (P5) are ever added, because both are fitted artifacts.

`call_llm_stub` is deliberately still a `NotImplementedError` - the notebook template's
gate is unchanged. Real calls go through `OpenRouterClient`, which is opt-in per script.

Every response is cached on disk keyed by (model, variant, prompt version, params,
prompt text) so re-analysis costs nothing and every published number can be traced back to
the exact bytes the provider returned. `wf_ensemble_report.md` §0 documents what happened
the last time a scored signal in this repo was allowed to drift silently (`relevance_score`,
disqualified as a feature for exactly that reason) - the cache plus the pinned
`prompt_version` is how that is not repeated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def build_use_case_brief(row: pd.Series, config: dict) -> str:
    """Assemble the use case's own written brief (objective, search terms, TRL bounds,
    decision rules - whichever of those exist as columns) into a labelled text block for
    the prompt. Column list is config-driven (`config["use_case_brief_cols"]`), not
    hardcoded, because papers_combined.parquet's exact set of broadcast use-case-brief
    columns is exactly the kind of thing the in-progress feature-engineering track may
    rename or restructure - update that list, not this function, when it does. Any listed
    column that's missing or null for this row is skipped rather than printed as "None".
    """
    lines = []
    for col in config["use_case_brief_cols"]:
        if col not in row.index:
            continue
        value = row[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                continue
            value = ", ".join(str(v) for v in value)
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"{col}: {text}")
    return "\n".join(lines)


def select_few_shot_examples(train_pool: pd.DataFrame, config: dict, random_state: int = 0) -> pd.DataFrame:
    """Draw `config['n_few_shot']` demonstration rows from `train_pool` ONLY.

    This is the prompt-construction equivalent of "fit the scaler on the training fold
    alone" - the caller must pass the current fold/rotation's *training* rows here, never
    validation or holdout rows, or the few-shot context effectively leaks the label of the
    row(s) being evaluated. Stratifies on `y` so both classes appear (unless `n_few_shot`
    is 0, in which case this is a pure zero-shot prompt and returns an empty frame).
    """
    n = config.get("n_few_shot", 0)
    if n <= 0:
        return train_pool.iloc[0:0]
    rng = np.random.RandomState(random_state)
    per_class = max(1, n // 2)
    parts = []
    for label in (0, 1):
        pool = train_pool[train_pool["y"] == label]
        take = min(per_class, len(pool))
        if take:
            idx = rng.choice(pool.index, size=take, replace=False)
            parts.append(pool.loc[idx])
    if not parts:
        return train_pool.iloc[0:0]
    return pd.concat(parts).iloc[:n]


def build_prompt(row: pd.Series, brief_text: str, few_shot_examples: pd.DataFrame, config: dict) -> str:
    """Assemble the final prompt: task framing + the use case's own brief (§ above) +
    optional few-shot demonstrations (leakage-safe if `few_shot_examples` came from
    `select_few_shot_examples`, not a hand-picked slice) + the target row's own text,
    ending in an explicit request for structured (JSON) output so `parse_llm_response`
    has something reliable to parse.
    """
    title_col, abstract_col = config["text_cols"]
    parts = [
        "You are screening academic papers for relevance to a specific research question.",
        "Research question brief:",
        brief_text,
        "",
        "Decide whether the paper below is RELEVANT (positive) or NOT RELEVANT (negative) "
        "to this research question, using the brief's own inclusion/exclusion criteria.",
    ]
    if len(few_shot_examples) > 0:
        parts.append("\nLabelled examples from this research question, for reference:")
        for _, ex in few_shot_examples.iterrows():
            label = "RELEVANT" if ex["y"] == 1 else "NOT RELEVANT"
            parts.append(
                f"- Title: {ex.get(title_col, '')}\n"
                f"  Abstract: {ex.get(abstract_col, '')}\n"
                f"  Verdict: {label}"
            )
    parts.append("\nPaper to classify:")
    parts.append(f"Title: {row.get(title_col, '')}")
    parts.append(f"Abstract: {row.get(abstract_col, '')}")
    parts.append(
        '\nRespond with JSON only, in the form '
        '{"label": "positive"|"negative", "confidence": <0-1 float>, "rationale": "<one sentence>"}.'
    )
    return "\n".join(parts)


def call_llm_stub(prompt: str, config: dict) -> str:
    """Placeholder - deliberately not implemented.

    Wire this to a real provider (e.g. the Claude/Anthropic API, matching
    `config["llm_model_name"]`) once `sf_llm_fold_pipeline.ipynb`'s CONFIG["run_training"]
    is flipped to True. Kept as an explicit `NotImplementedError`, not a silent dummy
    return, so an accidental "Run All" with `run_training=True` fails loudly instead of
    quietly scoring against fabricated responses.

    When implemented, PIN the model name/version and the prompt template version
    together (e.g. log both alongside every response) - `reports/wf_ensemble_report.md`
    §0 already documents what happens when a scored signal is allowed to drift silently
    (`relevance_score`, disqualified as a feature for exactly this reason). Don't repeat
    that mistake with this one.
    """
    model_name = config.get("llm_model_name") or "<CONFIG['llm_model_name'] not set>"
    raise NotImplementedError(
        f"call_llm_stub is a placeholder - implement a real call to {model_name} "
        "before setting CONFIG['run_training'] = True."
    )


def parse_llm_response(raw: str) -> dict:
    """Parse the `{"label", "confidence", "rationale"}` JSON `call_llm_stub` is prompted
    to return into `{"pred": 0/1, "proba": float}`. Defensive by construction - an LLM's
    raw text output is never guaranteed to be valid JSON, so a malformed response falls
    back to `pred=0, proba=0.5` (an explicit "couldn't parse" default, not a crash) rather
    than taking down an entire evaluation loop over one bad response.
    """
    try:
        parsed = json.loads(raw)
        pred = 1 if str(parsed.get("label", "")).strip().lower() == "positive" else 0
        proba = float(parsed.get("confidence", 0.5))
        proba = min(max(proba, 0.0), 1.0)
        if pred == 0:
            proba = 1.0 - proba  # confidence was expressed toward the predicted label
        return {"pred": pred, "proba": proba}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"pred": 0, "proba": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# Screening bake-off harness (scripts/run_llm_screening.py)
# ─────────────────────────────────────────────────────────────────────────────

BRIEF_FIELDS = [
    ("use_case_name", "Topic"),
    ("objective", "Objective"),
    ("problem_statement", "Problem being solved"),
    ("terms_must_include", "Must-include terms"),
    ("terms_nice_to_have", "Nice-to-have terms"),
    ("terms_exclude", "Exclusion terms"),
    ("domain_industry", "Industry"),
    ("domain_application", "Application"),
    ("domain_technology_focus", "Technology focus"),
]


def _as_text(value) -> str:
    """Flatten a brief cell to a string. List/array cells (the `terms_*` and
    `domain_technology_focus` columns are numpy arrays) become comma-joined; empty and
    null cells become "" so the caller can skip the line entirely rather than print
    "None" or "[]" into a prompt and invite the model to treat it as a real constraint.
    NULL is not 0 here either - an absent `terms_exclude` must read as "no exclusion
    terms were specified", not "exclude nothing", and the cleanest way to say that in a
    prompt is to omit the line.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        items = [str(v).strip() for v in list(value) if str(v).strip()]
        return ", ".join(items)
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def render_brief(row) -> str:
    """Render the use case's own written brief as labelled prose for the prompt.

    Distinct from `build_use_case_brief` above (which emits raw `col: value` lines for
    the notebook template): this one uses human field labels, drops absent fields, and
    is the version whose exact wording is pinned by `BRIEF_RENDER_VERSION`. Changing the
    labels changes the measurement, so bump that version if you touch this.
    """
    lines = []
    for col, label in BRIEF_FIELDS:
        text = _as_text(row.get(col))
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


BRIEF_RENDER_VERSION = "brief-v1"

_JSON_ONLY = (
    "Respond with JSON only. No markdown fences, no commentary before or after."
)

# The three pilot variants. Each carries its own version string; both it and the model id
# are logged on every row so a number can always be traced to the prompt that produced it.
PROMPT_VARIANTS: dict[str, dict] = {
    # P1 - the reference. A graded 0-100 score (not just a label) is what makes ranking
    # metrics - ROC-AUC, Recall@k, WSS@95 - computable at all, and those are the metrics
    # that transfer across prevalence regimes when a fixed threshold does not.
    "P1": {
        "version": "P1-v1",
        "system": (
            "You are a meticulous research assistant screening academic papers for a "
            "systematic review. You judge each paper only against the brief you are given."
        ),
        "instruction": (
            "Decide whether this paper is relevant to the brief, using the brief's own "
            "inclusion and exclusion criteria - not your general sense of paper quality.\n\n"
            f"{_JSON_ONLY}\n"
            '{"relevance": <integer 0-100>, "label": "relevant" | "not_relevant", '
            '"insufficient_information": <true|false>}\n\n'
            '"relevance" is how well the paper matches the brief: 0 = unrelated, '
            "100 = squarely on target. Set \"insufficient_information\" to true only when "
            "the title and abstract genuinely do not say enough to judge."
        ),
    },
    # P2 - P1 plus the F2 asymmetry stated explicitly. This is the single highest-value
    # untested lever in the pilot: every F2 number in this repo comes from a threshold
    # swept after the fact, and nobody has ever simply *told* the model that recall is
    # worth ~5x precision at this stage.
    "P2": {
        "version": "P2-v1",
        "system": (
            "You are a meticulous research assistant screening academic papers for a "
            "systematic review. You judge each paper only against the brief you are given."
        ),
        "instruction": (
            "Decide whether this paper is relevant to the brief, using the brief's own "
            "inclusion and exclusion criteria - not your general sense of paper quality.\n\n"
            "This is a HIGH-RECALL first-pass screen. A human expert will read everything "
            "you mark relevant, but will never see anything you mark not relevant. Missing "
            "a genuinely relevant paper is therefore about five times as costly as passing "
            "through an irrelevant one. When you are genuinely uncertain, include the paper.\n\n"
            f"{_JSON_ONLY}\n"
            '{"relevance": <integer 0-100>, "label": "relevant" | "not_relevant", '
            '"insufficient_information": <true|false>}\n\n'
            '"relevance" is how well the paper matches the brief: 0 = unrelated, '
            "100 = squarely on target. Set \"insufficient_information\" to true only when "
            "the title and abstract genuinely do not say enough to judge."
        ),
    },
    # P4 - per-criterion checklist before the verdict. Tests whether reasoning scaffolding
    # helps, and produces the per-score notes as a by-product: a reviewer-facing reason,
    # and a brief-quality diagnostic (criteria the model can never evaluate are criteria
    # the brief underspecifies).
    "P4": {
        "version": "P4-v1",
        "system": (
            "You are a meticulous research assistant screening academic papers for a "
            "systematic review. You judge each paper only against the brief you are given."
        ),
        "instruction": (
            "Work through the brief's criteria one at a time before deciding. For each "
            "criterion that the brief actually states, say whether this paper meets it, "
            "fails it, or whether the title and abstract leave it unclear. Then give an "
            "overall verdict.\n\n"
            f"{_JSON_ONLY}\n"
            '{"criteria": [{"criterion": "<short name>", "verdict": "met" | "not_met" | '
            '"unclear", "note": "<max 15 words>"}], "relevance": <integer 0-100>, '
            '"label": "relevant" | "not_relevant", "insufficient_information": '
            '<true|false>, "rationale": "<one sentence>"}\n\n'
            '"relevance" is how well the paper matches the brief: 0 = unrelated, '
            "100 = squarely on target."
        ),
    },
}


# P2lp - P2's framing, but the answer is elicited as a SINGLE TOKEN so the decision can be
# read from the token distribution instead of from a number the model writes down.
#
# Why: asked for a 0-100 score, every model in the grid answered in round numbers - 9-16
# distinct values across 1,848 rows, a tie fraction above 0.99. ROC-AUC over ~10 buckets is
# a measurement of the elicitation method as much as of the model, and ties are scored at
# half credit, so the LLM was handicapped in precisely the comparison it lost (0.821 vs the
# ensemble's 0.865 on continuous probabilities). P(YES) from the logprobs is continuous by
# construction and costs nothing extra to obtain.
#
# YES/NO rather than RELEVANT/IRRELEVANT deliberately: they are single tokens in every
# vocabulary here, whereas "IRRELEVANT" fragments and its probability mass would be split
# across several first-token continuations.
PROMPT_VARIANTS["P2lp"] = {
    "version": "P2lp-v1",
    "system": PROMPT_VARIANTS["P2"]["system"],
    "instruction": (
        "Decide whether this paper is relevant to the brief, using the brief's own "
        "inclusion and exclusion criteria - not your general sense of paper quality.\n\n"
        "This is a HIGH-RECALL first-pass screen. A human expert will read everything "
        "you mark relevant, but will never see anything you mark not relevant. Missing "
        "a genuinely relevant paper is therefore about five times as costly as passing "
        "through an irrelevant one. When you are genuinely uncertain, include the paper.\n\n"
        "Answer with exactly one word and nothing else: YES if the paper is relevant to "
        "this brief, NO if it is not."
    ),
    "logprobs": True,
}

# Token-first-character prefixes that decide which side of the answer a candidate token is
# on. Checked against the uppercased, whitespace-stripped token.
_YES_PREFIXES = ("YES", "Y", "REL", "INCL", "TRUE")
_NO_PREFIXES = ("NO", "N", "IRR", "EXCL", "FALSE", "NOT")


def parse_logprob_screening(logprobs: dict | None) -> dict:
    """Turn a token logprob payload into a continuous P(YES) in [0, 1].

    Walks forward to the first token position that actually carries a decision (a model may
    emit a leading newline or space first), then softmaxes the YES-side mass against the
    NO-side mass among that position's alternatives. Returns the renormalised P(YES), which
    is a genuine probability over the binary decision rather than over the whole vocabulary -
    so leftover mass on unrelated tokens does not drag every score toward zero.

    Returns parsed=False when no position carries a recognisable decision, rather than
    guessing. Same rule as everywhere else here: NULL is not 0.
    """
    if not logprobs or not logprobs.get("content"):
        return {"score": None, "pred": None, "insufficient": None, "parsed": False, "notes": None}

    for position in logprobs["content"][:6]:
        alts = position.get("top_logprobs") or []
        lp_yes = lp_no = None
        floor = 0.0
        for alt in alts:
            tok = str(alt.get("token", "")).strip().upper()
            lp = float(alt["logprob"])
            floor = min(floor, lp)
            if not tok:
                continue
            if tok.startswith(_NO_PREFIXES):      # checked first: "NOT" must not match "N"
                lp_no = lp if lp_no is None else max(lp_no, lp)
            elif tok.startswith(_YES_PREFIXES):
                lp_yes = lp if lp_yes is None else max(lp_yes, lp)
        if lp_yes is None and lp_no is None:
            continue

        # A side missing from the top-k is censored, not absent: bound it just below the
        # least likely token we were actually shown.
        censored = lp_yes is None or lp_no is None
        y = lp_yes if lp_yes is not None else floor - 1.0
        n = lp_no if lp_no is not None else floor - 1.0

        # TWO scores, because they answer different questions.
        #
        # `score` renormalises to a probability over the binary decision - the right thing
        # for thresholding, but at temperature 0 these models are extremely peaked (a
        # measured example: NO at -0.0 with the runner-up at -14.5), so it saturates to a
        # near-binary 0/1 and destroys exactly the ranking granularity logprobs were meant
        # to recover. Measured on the smoke subset: only 3 of 36 rows landed strictly
        # between 0.01 and 0.99.
        #
        # `score_logodds` keeps the raw margin instead. A row where YES beats NO by 2 nats
        # and one where it wins by 25 are genuinely different confidences, and the sigmoid
        # flattens both to 1.0. Ranking metrics only need an ordering, so the unsquashed
        # margin is the better ranking signal - and recovering ranking is the entire point
        # of this variant.
        odds = y - n
        return {
            "score": 1.0 / (1.0 + math.exp(-max(min(odds, 700.0), -700.0))),
            "score_logodds": odds,
            "logprob_censored": censored,
            "pred": int(odds >= 0.0),
            "insufficient": False,
            "parsed": True,
            "notes": None,
        }
    return {"score": None, "score_logodds": None, "logprob_censored": None,
            "pred": None, "insufficient": None, "parsed": False, "notes": None}


def build_screening_prompt(row, variant: str, brief_text: str | None = None) -> tuple[str, str]:
    """Return (system, user) for one paper under one prompt variant.

    `brief_text` overrides the row's own brief - that is the seam the shuffled-brief
    falsification control runs through, the same role `brief_map=` plays in
    `build_lexical_features`. Keep it: any feature claiming to read the brief has to be
    rebuildable against a deliberately wrong one (CONTEXT.md, closing note).
    """
    spec = PROMPT_VARIANTS[variant]
    brief = render_brief(row) if brief_text is None else brief_text
    abstract = _as_text(row.get("abstract")) or "(no abstract available)"
    user = (
        "RESEARCH BRIEF\n"
        f"{brief}\n\n"
        "PAPER\n"
        f"Title: {_as_text(row.get('title'))}\n"
        f"Abstract: {abstract}\n\n"
        f"{spec['instruction']}"
    )
    return spec["system"], user


def _balanced_objects(text: str):
    """Yield every brace-balanced `{...}` span in `text`, longest-first from each start.

    A greedy `\\{.*\\}` fails on the shape reasoning models actually emit - working-out
    prose that itself contains braces, followed by the real answer - because it spans from
    the first `{` to the last `}` and parses as nothing. Scanning for balance instead lets
    the caller try each candidate and keep the one that is genuinely an answer.
    """
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]


def extract_json(text: str) -> dict | None:
    """Pull the answer JSON object out of a model response.

    Deliberately forgiving in one direction only: it strips markdown fences, leading prose
    and in-band reasoning (all three common even when a model is told to emit none of them)
    but never guesses at a value. Anything it cannot parse returns None and is counted as a
    compliance failure rather than silently defaulting - a model that cannot emit parseable
    JSON is a finding about that model, not a row to quietly fill in.

    Candidates are preferred by *answer-ness*, not by position: the last balanced object
    carrying a `relevance` or `label` key wins. A reasoning model often emits a worked
    example or a fragment before the real verdict, so "first object" and "last object" are
    both wrong often enough to matter.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    best = None
    for span in _balanced_objects(cleaned):
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        if "relevance" in obj or "label" in obj:
            best = obj  # keep the last answer-shaped object
        elif best is None:
            best = obj
    return best


def parse_screening(raw: str) -> dict:
    """Parse a screening response into {score, pred, insufficient, parsed, notes}.

    `score` is the 0-1 rankable signal (the LLM's counterpart to a classifier's
    `predict_proba`); `pred` is its own hard verdict at its own natural decision
    boundary. Both are kept because they answer different questions: `score` gives
    threshold-free ranking metrics comparable to the ensemble's, `pred` gives the
    operating point the model actually chose without any threshold tuning - which is the
    whole reason an LLM might survive a prevalence shift that a swept threshold does not.

    A failure to parse is recorded as `parsed=False` with `score=None`, never as a
    neutral 0.5. A NaN that propagates loudly is worth more than a fabricated midpoint:
    NULL is not 0, and "the model did not answer" is not "the model was unsure".
    """
    obj = extract_json(raw)
    if obj is None or not isinstance(obj, dict):
        return {"score": None, "pred": None, "insufficient": None, "parsed": False, "notes": None}

    score = obj.get("relevance")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = min(max(score, 0.0), 100.0) / 100.0

    label = str(obj.get("label", "")).strip().lower()
    if label in ("relevant", "positive", "include", "yes", "true"):
        pred = 1
    elif label in ("not_relevant", "not relevant", "negative", "exclude", "no", "false"):
        pred = 0
    else:
        pred = None

    notes = None
    if "criteria" in obj or "rationale" in obj:
        notes = json.dumps(
            {k: obj[k] for k in ("criteria", "rationale") if k in obj}, ensure_ascii=False
        )

    return {
        "score": score,
        "pred": pred,
        "insufficient": bool(obj.get("insufficient_information", False)),
        "parsed": True,
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter client with an on-disk response cache
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CACHE_DIR = Path("data/processed/llm_cache")

# Per-model quirks, same idea as `embedding_utils.MODEL_CONFIGS`: the things that differ
# between models and would otherwise be rediscovered as a wall of HTTP 400s. Unregistered
# models fall back to DEFAULT_QUIRKS, so a new `--models` value works without an edit here.
#
# `reasoning`: what to send in the request's reasoning block.
#   None                      -> {"enabled": False}, the default; keeps output tokens down
#                                and removes a second source of run-to-run variance.
#   {"effort": ..., ...}      -> sent verbatim. Required for reasoning-native models that
#                                reject being switched off (gpt-oss returns HTTP 400
#                                "Reasoning is mandatory for this endpoint and cannot be
#                                disabled" - measured, not guessed).
# `max_tokens`: raised where a mandatory reasoning trace shares the completion budget.
DEFAULT_QUIRKS = {"reasoning": None, "max_tokens": 700}
MODEL_QUIRKS: dict[str, dict] = {
    "openai/gpt-oss-20b": {"reasoning": {"effort": "low", "exclude": True}, "max_tokens": 2500},
    "openai/gpt-oss-120b": {"reasoning": {"effort": "low", "exclude": True}, "max_tokens": 2500},
    # Nemotron reasons *in-band* - it writes its working into the visible content and only
    # then emits the JSON, whatever the reasoning block says. At the 700-token default that
    # overran on 14 of 36 P4 rows (finish_reason="length", cut off mid-thought). The budget
    # is raised rather than the reasoning suppressed, because suppressing it is exactly what
    # was already asked for and declined.
    "nvidia/nemotron-3-super-120b-a12b": {"reasoning": None, "max_tokens": 1600},
    "nvidia/nemotron-3-nano-30b-a3b": {"reasoning": None, "max_tokens": 1600},
}

# Parameters that some providers reject outright even when OpenRouter's routing table says
# they are supported (DeepInfra's nemotron-3-super endpoint rejects `seed` as
# "extra_forbidden" - also measured). On a 4xx naming one of these, it is dropped and the
# call retried, and the drop is recorded per row: a response produced without `seed` is
# strictly less reproducible, which the determinism stage has to know about rather than
# silently average over.
DROPPABLE_PARAMS = ("seed", "reasoning", "logprobs", "top_logprobs")

# Pinned inference provider per model, with `allow_fallbacks: False`.
#
# Why pin at all: OpenRouter routes to whichever provider is cheapest per call, so an
# unpinned cell is served by 4-8 different providers and provider quality lands inside the
# measurement. This is not hypothetical - in the smoke run, 2 of 36 `gemma-4-31b` responses
# came back as corrupted JSON (`{"SSSS{"relevance": 7S75, ...}`, literal characters injected
# mid-token) and BOTH came from Phala. Unpinned, that reads as "gemma is 94% compliant".
#
# Selection rule: the cheapest endpoint with `status == 0` (healthy) that genuinely honours
# `seed`. Two consequences worth knowing:
#   - nemotron's cheapest endpoint (DeepInfra, $0.085/M) is skipped on both counts: status
#     -2, and it is the endpoint measured rejecting `seed` as `extra_forbidden` while
#     OpenRouter's routing table claims support. Nebius costs ~3.5x more and is the only
#     healthy endpoint that actually takes the parameter - and honouring `seed` is the
#     entire point of pinning.
#   - Single-endpoint providers are preferred where the price gap is trivial: several
#     providers expose the same model at 2-3 prices (quantisations), so pinning by name
#     alone does not fully pin the numerics. `provider` is recorded per row regardless.
#
# Re-derive with `--list-providers` rather than trusting this table after a few months;
# endpoint health and pricing both move.
PINNED_PROVIDERS: dict[str, list[str]] = {
    "openai/gpt-oss-20b": ["CoreWeave"],                 # $0.030/$0.130, seed+logprobs
    # Friendli, not the cheaper Chutes: chosen on a measured 6-provider probe, not on
    # price. Chutes served the first gemma run at 28.7s median and produced 586 HTTP 504s
    # on the P4 checklist variant (68% coverage) - its failure mode is the slow tail, and
    # the probe puts that tail at 57s. Friendli's *worst* call (9.3s) is faster than
    # Chutes' median, for a few cents more across the whole grid. Latency spread across
    # providers for one model is ~5x, far wider than the price spread.
    "google/gemma-4-31b-it": ["Friendli"],               # $0.140/$0.400, p50 7.6s, max 9.3s
    # Nebius, revised at set-A scale. All three of nemotron's endpoints fail a different way,
    # and none of the three failures is visible in OpenRouter's capability table:
    #   DeepInfra    ($0.085/M) rejects `seed`, and rate-limits so hard at 62k-row scale that
    #                measured throughput collapses to **4 rows/min** - 42 hours for one cell.
    #                Fine for the 1,848-row pilot, unusable here. Throughput is a capability.
    #   DigitalOcean ($0.165/M) 404s every request: its endpoint advertises neither
    #                `temperature` nor `seed`, so `require_parameters` filters it to nothing.
    #   Nebius       ($0.300/M) rejects `seed` too, but parses 100% at 434 rows/min on P2.
    # So: the only usable endpoint costs 3.5x the cheapest and still cannot honour `seed`.
    # **Nemotron's determinism numbers therefore carry a caveat no other model here does**,
    # and that is itself a finding against a requirement that says "100% deterministic".
    # (The pilot preferred DeepInfra on P4 truncation - 108/108 at 322 median completion
    # tokens vs Nebius' 60/108 at 568. P2 is short-output and shows 0 truncations on Nebius,
    # so that objection does not apply to the variant run here.)
    "nvidia/nemotron-3-super-120b-a12b": ["Nebius"],     # $0.300/$0.900, no working seed
    "qwen/qwen3.5-397b-a17b": ["Alibaba"],               # $0.390/$2.340, seed+logprobs
}

# Separate table for logprob-scored variants, because not every provider returns logprobs
# and the ones that do are not always the fastest (which is what PINNED_PROVIDERS optimises).
#
# gpt-oss and qwen keep their normal pin, so their verbalised-vs-logprob comparison is
# within-provider and clean. gemma cannot: its normal pin (Friendli) returns no logprobs at
# all, so the logprob run moves to CoreWeave — and to keep the comparison paired rather than
# confounded by provider, gemma's verbalised P2 should be re-run on CoreWeave too before the
# two are compared for that model.
#
# nvidia/nemotron-3-super is absent deliberately: it has NO endpoint offering logprobs and
# seed together, so it cannot take part. That is its second capability strike, after the
# `seed` rejection above — both matter for a system specified as 100% deterministic.
LOGPROB_PROVIDERS: dict[str, list[str]] = {
    "openai/gpt-oss-20b": ["CoreWeave"],
    "google/gemma-4-31b-it": ["CoreWeave"],
    # NOT Alibaba, despite it advertising logprobs and serving qwen fine for the grid: it
    # rejects the parameter at the endpoint, the retry ladder drops it, and the call then
    # returns a bare "YES" with no distribution at all. Third provider in this pilot whose
    # advertised capability does not survive contact. Parasail costs more and answers.
    "qwen/qwen3.5-397b-a17b": ["Parasail"],
}

# Reasoning-mandatory models need room to think before the answer token appears; a 6-token
# budget is entirely consumed by the reasoning trace and `content` comes back empty with
# finish_reason="length" and no logprobs at all (measured on gpt-oss).
LOGPROB_MAX_TOKENS = {"openai/gpt-oss-20b": 1400, "openai/gpt-oss-120b": 1400}


class OpenRouterClient:
    """Cached, retrying, thread-pooled chat client.

    Three deliberate choices:

    - **`require_parameters: True`** in the provider block. OpenRouter routes to whichever
      provider is cheapest by default, and providers differ in which sampling parameters
      they honour. Requiring them restricts routing to providers that actually respect
      `temperature`/`seed`, so "temperature 0" means what it says. The provider that
      served each call is recorded per row anyway, because at temperature 0 an MoE model
      is still only as reproducible as its routing.
    - **Reasoning explicitly disabled.** Every candidate except Ministral has a reasoning
      mode; left on, it multiplies output tokens ~20x and adds a second source of
      run-to-run variance on top of the sampler.
    - **Failures are recorded, never invented.** An exhausted retry stores an error row;
      it does not return a neutral score.
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 700,
        max_retries: int = 7,
        api_key: str | None = None,
        provider_order: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 20,
    ):
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        quirks = {**DEFAULT_QUIRKS, **MODEL_QUIRKS.get(model, {})}
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens if max_tokens != 700 else quirks["max_tokens"]
        self.reasoning = quirks["reasoning"] or {"enabled": False}
        self.max_retries = max_retries
        self.provider_order = (
            provider_order if provider_order is not None else PINNED_PROVIDERS.get(model)
        )
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (env var, .env, or api_key=).")

        cache_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
        self.cache_path = cache_dir / f"{slug}.jsonl"
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._cache[rec["key"]] = rec

    def _params(self, drop: tuple[str, ...] = ()) -> dict:
        provider = {"require_parameters": True}
        if self.provider_order:
            provider["order"] = self.provider_order
            provider["allow_fallbacks"] = False
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "provider": provider,
            "usage": {"include": True},
        }
        if self.logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = self.top_logprobs
        for name in drop:
            params.pop(name, None)
        if drop:
            # A dropped parameter is exactly the case where `require_parameters` would
            # filter every provider out - the provider that rejected it is, by
            # definition, one OpenRouter believes supports it.
            params["provider"] = {k: v for k, v in provider.items() if k != "require_parameters"}
        return params

    def _key(self, system: str, user: str, tag: str) -> str:
        """Cache key over the *requested* params, not the ones a fallback settled on.

        Deliberate: a re-run asks the same question, and should get the recorded answer
        back regardless of which parameters the provider turned out to reject. The
        params actually used are recorded in the row as `params_dropped`.
        """
        payload = json.dumps(
            {"p": self._params(), "s": system, "u": user, "t": tag}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _write(self, rec: dict) -> None:
        with self._lock:
            self._cache[rec["key"]] = rec
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def call(self, system: str, user: str, tag: str = "") -> dict:
        """One chat completion, served from cache when the exact request was made before."""
        import requests  # local import: only scripts that actually call out need it

        key = self._key(system, user, tag)
        hit = self._cache.get(key)
        # Successful responses are served from cache; failures are NOT. A cached error
        # would make every re-run replay the same failure forever and never retry it -
        # which is the opposite of what a cache is for. Failures stay on disk as an audit
        # trail but are always re-attempted.
        if hit is not None and not hit.get("error"):
            return {**hit, "cached": True}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        dropped: tuple[str, ...] = ()
        last_err = None

        for attempt in range(self.max_retries):
            body = {**self._params(drop=dropped), "messages": messages}
            try:
                t0 = time.time()
                resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=body, timeout=180)
                latency = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" not in data:  # OpenRouter surfaces upstream errors in-band
                        last_err = f"no choices: {json.dumps(data)[:300]}"
                        time.sleep(2 ** attempt)
                        continue
                    usage = data.get("usage") or {}
                    rec = {
                        "key": key,
                        "model": self.model,
                        "tag": tag,
                        "content": data["choices"][0]["message"].get("content") or "",
                        "provider": data.get("provider"),
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cost": usage.get("cost"),
                        "latency_s": round(latency, 2),
                        "params_dropped": ",".join(dropped),
                        "error": None,
                    }
                    if self.logprobs:
                        # Keep only the first few positions: the decision is in the first
                        # real token, and storing the whole payload for 1,848 rows x N cells
                        # would bloat the cache for no analytical gain.
                        lp = data["choices"][0].get("logprobs") or {}
                        rec["logprobs"] = {"content": (lp.get("content") or [])[:4]}
                    self._write(rec)
                    return {**rec, "cached": False}

                text = resp.text[:400]
                if resp.status_code in (429, 500, 502, 503, 524):
                    # Pinning to one provider with `allow_fallbacks: False` deliberately
                    # removes the escape valve, so upstream rate limits land here instead
                    # of being routed around. Back off hard, with jitter so a pool of
                    # threads that hit the wall together does not retry in lockstep.
                    last_err = f"HTTP {resp.status_code}: {text[:200]}"
                    base = 5 if resp.status_code == 429 else 2
                    time.sleep(min(base * (2 ** attempt), 60) * (0.5 + random.random()))
                    continue
                # A 4xx naming a parameter is a capability mismatch, not a transient
                # fault: drop that parameter and retry rather than burning the row.
                offender = next(
                    (p for p in DROPPABLE_PARAMS if p not in dropped and p in text.lower()), None
                )
                if offender:
                    dropped = (*dropped, offender)
                    last_err = f"HTTP {resp.status_code} (dropped {offender}): {text[:150]}"
                    continue
                last_err = f"HTTP {resp.status_code}: {text[:300]}"
                break
            except Exception as exc:  # noqa: BLE001 - network layer, anything can surface
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 20))

        rec = {
            "key": key, "model": self.model, "tag": tag, "content": "", "provider": None,
            "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
            "cost": None, "latency_s": None, "params_dropped": ",".join(dropped),
            "error": last_err,
        }
        self._write(rec)
        return {**rec, "cached": False}


def cache_tag(variant: str, brief_map: dict | None = None, brief_tag: str | None = None) -> str:
    """The `tag` component of the cache key for one cell. Single source of truth.

    Extracted so `dry_run_frame` cannot drift from `score_frame`. That drift is not a
    hypothetical: the tag is part of the cache key, so a dry run that computed the tag
    slightly differently would report 0% cached, and the honest response to that report is
    to go and buy responses that were already paid for. A cost estimator that can be wrong
    in the expensive direction is worse than none.

    Note the asymmetry, which is load-bearing: the brief tag is appended **only** when
    `brief_map is not None`. An own-brief run must produce the bare tag, byte for byte, or
    every response already on disk becomes unreachable.
    """
    tag = f"{variant}|{PROMPT_VARIANTS[variant]['version']}|{BRIEF_RENDER_VERSION}"
    if brief_map is not None:
        tag += "|" + (brief_tag or "shuffled")
    return tag


def dry_run_frame(
    df,
    model: str,
    variant: str,
    brief_map: dict | None = None,
    brief_tag: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    usd_per_row: float | None = None,
) -> dict:
    """What would `score_frame` cost, and how much of it is already paid for? No network.

    Builds the identical prompts and cache keys `score_frame` would, then looks each one up
    on disk. Nothing is sent anywhere, so this is free and safe to run before every spend.

    `usd_per_row` defaults to the median recorded cost of this model's existing successful
    cache rows — a measured unit price rather than an estimated one. Returns NaN for the
    projection when the cache holds no priced rows for the model yet, rather than guessing.
    """
    import numpy as np

    client = OpenRouterClient(model=model, cache_dir=cache_dir)
    tag = cache_tag(variant, brief_map, brief_tag)
    cols = list(df.columns)

    n_cached = 0
    for rt in df.itertuples(index=False):
        row = dict(zip(cols, rt))
        brief_text = brief_map.get(row.get("use_case_key")) if brief_map is not None else None
        system, user = build_screening_prompt(row, variant, brief_text=brief_text)
        hit = client._cache.get(client._key(system, user, tag))
        if hit is not None and not hit.get("error"):
            n_cached += 1

    if usd_per_row is None:
        priced = [r["cost"] for r in client._cache.values()
                  if not r.get("error") and isinstance(r.get("cost"), (int, float))]
        usd_per_row = float(np.median(priced)) if priced else float("nan")

    n_fresh = len(df) - n_cached
    return {
        "model": model, "variant": variant, "tag": tag,
        "n_rows": len(df), "n_cached": n_cached, "n_fresh": n_fresh,
        "hit_rate": n_cached / len(df) if len(df) else 0.0,
        "usd_per_row": usd_per_row,
        "projected_usd": n_fresh * usd_per_row,
    }


def score_frame(
    df,
    model: str,
    variant: str,
    concurrency: int = 16,
    brief_map: dict | None = None,
    brief_tag: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    progress: bool = True,
    **client_kwargs,
):
    """Score every row of `df` with one (model, variant) cell. Returns a DataFrame.

    `brief_map` maps `use_case_key` -> brief text, overriding each row's own brief. That
    is how the shuffled-brief control runs: pass a permuted map and every number below
    should collapse. If it does not, the cell is measuring generic paper quality rather
    than brief-conditioned relevance and its score means nothing.

    `brief_tag` names which brief set a swapped run used, so the cache stays readable after
    the fact. It only became necessary once the brief swap stopped being a control and
    started being an experiment: the set-A brief-format ladder runs the *same* prompt over
    the review's raw abstract (`raw`), the supplied brief, an induced rule set (`induced`)
    and a derangement (`shuffled`), and tagging the first three "shuffled" would make the
    audit trail lie. Defaults to "shuffled" so every existing call site is unchanged.
    """
    import pandas as pd

    spec = PROMPT_VARIANTS[variant]
    wants_logprobs = bool(spec.get("logprobs"))
    if wants_logprobs:
        client_kwargs.setdefault("logprobs", True)
        client_kwargs.setdefault("max_tokens", LOGPROB_MAX_TOKENS.get(model, 6))
        if model in LOGPROB_PROVIDERS:
            client_kwargs.setdefault("provider_order", LOGPROB_PROVIDERS[model])
    client = OpenRouterClient(model=model, cache_dir=cache_dir, **client_kwargs)
    # A shuffled-brief run gets a marker so it is distinguishable in the cache after the
    # fact, rather than merely non-colliding with the real-brief run. The own-brief tag is
    # left EXACTLY as it was: `tag` is part of the cache key, so appending anything to the
    # default case invalidates every response already paid for. (Adding "|own" here did
    # exactly that and started silently re-buying the whole 12-cell grid.)
    tag = cache_tag(variant, brief_map, brief_tag)
    rows = list(df.itertuples(index=False))
    cols = list(df.columns)

    def one(idx_row):
        i, rt = idx_row
        row = dict(zip(cols, rt))
        brief_text = None
        if brief_map is not None:
            brief_text = brief_map.get(row.get("use_case_key"))
        system, user = build_screening_prompt(row, variant, brief_text=brief_text)
        res = client.call(system, user, tag=tag)
        parsed = (parse_logprob_screening(res.get("logprobs")) if wants_logprobs
                  else parse_screening(res["content"]))
        return {
            "paper_id": row.get("paper_id"),
            "use_case_key": row.get("use_case_key"),
            "y": row.get("y"),
            "model": model,
            "variant": variant,
            "prompt_version": spec["version"],
            **parsed,
            "provider": res["provider"],
            "params_dropped": res.get("params_dropped", ""),
            "finish_reason": res["finish_reason"],
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "cost": res["cost"],
            "latency_s": res["latency_s"],
            "cached": res["cached"],
            "error": res["error"],
            "raw": res["content"][:4000],
        }

    out = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, rec in zip(range(len(rows)), pool.map(one, enumerate(rows))):
            out[i] = rec
            done += 1
            if progress and done % 50 == 0:
                print(f"    {model} {variant}: {done}/{len(rows)}", flush=True)
    return pd.DataFrame(out)
