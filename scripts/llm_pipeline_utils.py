"""Shared building blocks for notebooks/pipelines/sf_llm_fold_pipeline.ipynb.

A prompted LLM doesn't consume a numeric feature matrix, so none of
`fold_pipeline_utils.py`'s `ColumnTransformer`/`Pipeline` machinery applies here - the
input is a text prompt, not `X`. What *does* carry over unchanged is the split-then-fit
discipline: a few-shot prompt's demonstration examples are the prompt-based equivalent of
a fitted parameter (they encode information the model conditions its answer on), so they
must be drawn only from the training fold, never from validation/holdout, for exactly the
same reason `StandardScaler` must be fit on the training fold only. `select_few_shot_examples`
below is where that rule is enforced.

Nothing in this module calls a real LLM - `call_llm_stub` deliberately raises
`NotImplementedError`. Wire it to a real provider only once `notebooks/pipelines/
sf_llm_fold_pipeline.ipynb`'s CONFIG["run_training"] is flipped to True.
"""

from __future__ import annotations

import json

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
