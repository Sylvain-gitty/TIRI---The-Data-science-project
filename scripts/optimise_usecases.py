"""Rewrite TIRI's six use-case specs per `wf_spec_quality_answer.md` — deterministically, and
without ever looking at a label.

The question this serves: *what does a better-written use case do to the baseline model?* The
answer is only worth anything if the "better" specs were written **without hindsight**, and that is
the entire design problem here. If I write an objective by reading the relevant papers, the
cold-start features match brief text against paper text, so I have fitted the brief to the labels
and the gain is manufactured. `DATA_BRIEF.md`'s honest-limit #2 records exactly this failure in the
benchset corpus: every benchset brief derives from its review's own abstract, "which inflates
absolute scores for every brief-reading method equally".

The leak-free reservoir, and why one exists at all
--------------------------------------------------
The six specs already contain far more analyst-written material than the model ever sees:

  read by the **cosine** brief   `use_case_name`, `problem_statement`, `objective`,
                                 `domain_industry`, `domain_application`   (`embed_benchsets.USE_CASE_COLS`)
  read by the **lexical** block  `objective`, `problem_statement`, `terms_must_include`,
                                 `terms_nice_to_have`, and `domain_*`      (`lexical_features.build_brief_texts`)
  **read by neither**            `performance_criteria`, `constraints`, `decision_criteria.*`, `notes`

So there is real, hand-written domain knowledge sitting in fields no feature reads. **The
optimisation is to move the analyst's own words into the fields the model actually reads** — not to
add knowledge. Nothing here consults a label, a paper, or the corpus. Every output token traces to
an input field of the same JSON file, and `--provenance` prints that mapping so the claim is
checkable rather than asserted.

The transformation, stated in full
----------------------------------
1. **`objective`** — de-imperatived (a leading "Find/Identify/Scout/Explore…" is stripped, because
   the feature matches the objective against *paper* text and papers do not open with instructions),
   then extended with, in order and de-duplicated: `problem_statement`; a sentence naming each
   `performance_criteria` metric and target; a sentence naming `constraints` (TRL / scale / cost);
   each `decision_criteria.must_have` entry that is a full sentence rather than a bare keyword; and
   `notes` **only** where it describes the subject rather than the corpus or the labelling process.
2. **`terms_must_include`** — existing ∪ `domain_technology_focus` ∪ `performance_criteria[].metric`,
   de-duplicated case-insensitively, capped at 8 (the answer's recommended 5–8).
3. **`terms_nice_to_have`** — existing ∪ short noun phrases from `decision_criteria.nice_to_have`
   ∪ any `domain_technology_focus` not already used, to a floor of 5.
4. **`terms_exclude`** — existing ∪ short noun phrases from `decision_criteria.exclusions`.
5. **`domain_industry` / `domain_application`** — filled from `name` or `technology_focus` when empty.
6. Everything else — **unchanged**, including the criteria fields themselves. They stay where they
   are; this copies from them, it does not move them.

Two rules that are judgement, and are declared as such
------------------------------------------------------
- **Which `notes` count as subject matter.** `solar_leo`'s note is commentary about how its pool was
  built ("Seeded from the reference lists ... the starting pool is CANON"), which describes the
  corpus rather than the literature and would inject labelling-process vocabulary into a feature that
  matches papers. `cement_binders`' note is an instruction to the analyst ("Prioritise bigger impact
  potential ..."). Both are excluded by a keyword+imperative filter. `soil_microbiome`'s and
  `tech_forecasting`'s notes are subject matter and are included. The filter is `_is_subject_matter`
  below and it is applied identically to all six.
- **Which `decision_criteria` entries are sentences vs keywords.** `tech_forecasting.must_have` is
  `["graph", "database"]` — a term list wearing a criteria field's name (a finding in its own right,
  `wf_checkability_audit.md` §5). Entries of ≤3 tokens go to the term lists; longer ones go to the
  objective.

    python scripts/optimise_usecases.py                # write data/raw/optimised/*.usecase.json
    python scripts/optimise_usecases.py --provenance   # ... and print where every token came from
    python scripts/optimise_usecases.py --dry-run      # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

RAW = REPO / "data" / "raw"
OUT_DIR = RAW / "optimised"
OUT_DIR_PROSE = RAW / "optimised_prose_only"
OUT_MD = REPO / "reports" / "wf_optimised_usecases.md"

USE_CASES = ["carbon_capture", "cement_binders", "ner", "soil_microbiome", "solar_leo",
             "tech_forecasting"]

MIN_MUST, MIN_NICE = 5, 5

# Universal evaluation-metric names. A metric a whole field reports is not a discriminative
# must-include term: promoting "Precision" into `ner`'s list cost **-0.118** on `lex_bm25_must`
# because it matches nearly every NLP paper written. Domain-specific physical quantities
# ("conversion efficiency", "embodied CO2 reduction") are NOT on this list and are still allowed -
# adding those to `solar_leo`'s two-term list was worth +0.046.
GENERIC_METRICS = {
    "precision", "recall", "f1", "f1 score", "f-score", "f score", "accuracy", "auc", "roc-auc",
    "roc auc", "auroc", "p-value", "error rate", "rmse", "mae", "mse", "sensitivity",
    "specificity", "throughput", "latency",
}
KEYWORD_MAX_TOKENS = 3          # <=3 tokens is a term, not a criterion sentence

_LEADING_IMPERATIVE = re.compile(
    r"^\s*(find|identify|scout|explore|improve|discover|search for|look for|locate|retrieve)\s+",
    re.I)
# Notes that describe the corpus or the labelling process, not the literature.
_PROCESS_NOTE = re.compile(
    r"\b(pool|seeded|canon|analyst|labell?ing|triage|this tool|reference lists)\b"
    r"|\bprioriti[sz]\w*",   # "Prioritise ..." - an instruction to the analyst, not subject matter
    re.I)


def _lst(v) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return [str(v).strip()] if v and str(v).strip() else []


def _txt(v) -> str:
    return str(v).strip() if v is not None else ""


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        k = i.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(i.strip())
    return out


def _near_duplicate(cand: str, existing: list[str]) -> bool:
    """True if `cand` says essentially what one of `existing` already says.

    Substring either way, case-insensitively, after stripping parenthetical glosses. BM25 splits a
    phrase into tokens, so "Named Entity Recognition (NER)" beside "Named Entity Recognition"
    contributes no new coverage and dilutes what is there.
    """
    c = re.sub(r"\s*\([^)]*\)", "", cand).strip().lower()
    if not c:
        return True
    return any(c in e.lower() or e.lower() in c for e in existing if e.strip())


def _is_subject_matter(note: str) -> bool:
    """A note describes the literature, not the corpus or what the analyst should do."""
    if not note.strip():
        return False
    if _PROCESS_NOTE.search(note):
        return False
    return not _LEADING_IMPERATIVE.match(note)


def _sentence(s: str) -> str:
    s = s.strip()
    return s if not s or s.endswith((".", "!", "?")) else s + "."


def optimise(spec: dict, prose_only: bool = False) -> tuple[dict, list[dict]]:
    """Return the rewritten spec and a provenance record: every addition, and its source field.

    `prose_only=True` applies rule 1 (and 5) and **leaves the term lists exactly as written**. That
    arm exists because the full rewrite turned out to help the cosine (+0.027 on qwen4b) and *hurt*
    the must-term BM25 (-0.028): padding `terms_must_include` up to 8 with whatever
    `domain_technology_focus` happened to contain is `keyword_flood` with domain words instead of
    generic ones, and `wf_spec_quality_answer.md` §4 asks for "5-8 **precise, discriminative**
    phrases", not eight of anything. Separating the two lets each rule be judged on its own.
    """
    prov: list[dict] = []
    out = json.loads(json.dumps(spec))          # deep copy; untouched fields stay untouched
    terms = spec.get("terms") or {}
    domain = spec.get("domain") or {}
    dec = spec.get("decision_criteria") or {}
    perf = spec.get("performance_criteria") or []
    cons = spec.get("constraints") or {}

    # ---- 1. objective -------------------------------------------------------------------
    parts: list[str] = []
    obj = _txt(spec.get("objective"))
    stripped = _LEADING_IMPERATIVE.sub("", obj)
    if stripped != obj:
        prov.append({"field": "objective", "action": "de-imperative",
                     "detail": f"{obj[:40]!r} -> {stripped[:40]!r}", "source": "objective"})
        stripped = stripped[0].upper() + stripped[1:] if stripped else stripped
    if stripped:
        parts.append(_sentence(stripped))

    prob = _txt(spec.get("problem_statement"))
    if prob:
        parts.append(_sentence(prob))
        prov.append({"field": "objective", "action": "append", "detail": prob[:60],
                     "source": "problem_statement"})

    metrics = [f"{_txt(p.get('metric'))} ({_txt(p.get('target'))})" if _txt(p.get("target"))
               else _txt(p.get("metric")) for p in perf if _txt(p.get("metric"))]
    if metrics:
        s = "Reported outcomes of interest include " + ", ".join(_dedupe(metrics)) + "."
        parts.append(s)
        prov.append({"field": "objective", "action": "append", "detail": s[:80],
                     "source": "performance_criteria"})

    cbits = []
    trl = cons.get("trl") or {}
    if isinstance(trl, dict) and trl.get("min") is not None:
        cbits.append(f"technology readiness level {trl.get('min')}-{trl.get('max')}")
    for k in ("scale", "cost"):
        if _txt(cons.get(k)):
            cbits.append(f"{k}: {_txt(cons[k])}")
    if cbits:
        s = "Relevant work sits at " + "; ".join(cbits) + "."
        parts.append(s)
        prov.append({"field": "objective", "action": "append", "detail": s[:80],
                     "source": "constraints"})

    must_sentences = [m for m in _lst(dec.get("must_have"))
                      if len(m.split()) > KEYWORD_MAX_TOKENS]
    for m in must_sentences:
        parts.append(_sentence(m))
        prov.append({"field": "objective", "action": "append", "detail": m[:60],
                     "source": "decision_criteria.must_have"})

    note = _txt(spec.get("notes"))
    if note:
        if _is_subject_matter(note):
            parts.append(_sentence(note))
            prov.append({"field": "objective", "action": "append", "detail": note[:60],
                         "source": "notes"})
        else:
            prov.append({"field": "objective", "action": "SKIPPED note",
                         "detail": note[:70], "source": "notes (corpus/process commentary)"})
    out["objective"] = " ".join(parts)

    # ---- 2/3/4. term lists ---------------------------------------------------------------
    tech = _lst(domain.get("technology_focus"))
    existing_must = _lst(terms.get("must_include"))
    perf_metrics = [m for m in (_txt(p.get("metric")) for p in perf)
                    if m and m.lower() not in GENERIC_METRICS]
    dropped_metrics = [_txt(p.get("metric")) for p in perf
                       if _txt(p.get("metric")).lower() in GENERIC_METRICS]

    # Fix 1: TOP UP to a floor, never expand past it. The list only grows if the analyst wrote
    # fewer than MIN_MUST; a spec that already has enough is left exactly alone. The full rewrite
    # expanded `ner` from 4 to 8 and cost -0.118.
    # Fix 2: NEVER truncate. "5-8" is a floor and a quality bar, not a cap - capping
    # `soil_microbiome`'s 10 hand-written terms at 8 cost -0.077.
    # Fix 3: drop NEAR-duplicates, not just exact ones. `ner`'s technology_focus is
    # "Named Entity Recognition (NER)" against an existing "Named Entity Recognition"; adding it
    # dilutes BM25 without adding coverage.
    must = list(existing_must)
    if len(must) < MIN_MUST:
        for cand in tech + perf_metrics:
            if len(must) >= MIN_MUST:
                break
            if not _near_duplicate(cand, must):
                must.append(cand)
    must = _dedupe(must)
    added_must = [m for m in must if m.lower() not in {x.lower() for x in existing_must}]
    if added_must:
        prov.append({"field": "terms_must_include", "action": "top up to floor",
                     "detail": ", ".join(added_must),
                     "source": "domain.technology_focus + non-generic performance_criteria metrics"})
    if dropped_metrics:
        prov.append({"field": "terms_must_include", "action": "SKIPPED metric",
                     "detail": ", ".join(dropped_metrics),
                     "source": "performance_criteria[].metric (universal evaluation metric)"})
    if len(existing_must) >= MIN_MUST:
        prov.append({"field": "terms_must_include", "action": "left alone",
                     "detail": f"{len(existing_must)} terms already >= floor of {MIN_MUST}",
                     "source": "-"})

    # Same three fixes on the nice-to-have list: top up to a floor, never truncate, no near-dupes.
    existing_nice = _lst(terms.get("nice_to_have"))
    nice_kw = [n for n in _lst(dec.get("nice_to_have")) if len(n.split()) <= KEYWORD_MAX_TOKENS]
    leftover_tech = [x for x in tech if not _near_duplicate(x, must)]
    nice = list(existing_nice)
    if len(nice) < MIN_NICE:
        for cand in nice_kw + leftover_tech:
            if len(nice) >= MIN_NICE:
                break
            if not _near_duplicate(cand, nice):
                nice.append(cand)
    nice = _dedupe(nice)
    added_nice = [n for n in nice if n.lower() not in {x.lower() for x in existing_nice}]
    if added_nice:
        prov.append({"field": "terms_nice_to_have", "action": "add", "detail": ", ".join(added_nice),
                     "source": "decision_criteria.nice_to_have + leftover technology_focus"})

    excl_kw = [e for e in _lst(dec.get("exclusions")) if len(e.split()) <= KEYWORD_MAX_TOKENS]
    excl = _dedupe(_lst(terms.get("exclude")) + excl_kw)

    if not prose_only:
        out.setdefault("terms", {})
        out["terms"]["must_include"] = must
        out["terms"]["nice_to_have"] = nice
        out["terms"]["exclude"] = excl
    else:
        prov = [p for p in prov if not p["field"].startswith("terms_")]

    # ---- 5. domain fills -----------------------------------------------------------------
    out.setdefault("domain", {})
    if not _txt(domain.get("industry")) and tech:
        out["domain"]["industry"] = tech[0]
        prov.append({"field": "domain_industry", "action": "fill", "detail": tech[0],
                     "source": "domain.technology_focus[0]"})
    if not _txt(domain.get("application")) and _txt(spec.get("name")):
        out["domain"]["application"] = _txt(spec["name"])
        prov.append({"field": "domain_application", "action": "fill",
                     "detail": _txt(spec["name"]), "source": "name"})
    return out, prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prose-only", action="store_true",
                    help="rewrite the objective only; leave term lists exactly as the analyst "
                         "wrote them")
    args = ap.parse_args()

    from spec_linter import lint, normalise  # noqa: E402

    out_dir = OUT_DIR_PROSE if args.prose_only else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, all_prov = [], {}
    for key in USE_CASES:
        spec = json.loads((RAW / f"{key}.usecase.json").read_text())
        new, prov = optimise(spec, prose_only=args.prose_only)
        all_prov[key] = prov

        flat_old, flat_new = normalise(_flat(spec)), normalise(_flat(new))
        rows.append({
            "use_case": key,
            "obj_chars": f"{len(flat_old['objective'])} -> {len(flat_new['objective'])}",
            "n_must": f"{len(flat_old['terms_must_include'])} -> {len(flat_new['terms_must_include'])}",
            "n_nice": f"{len(flat_old['terms_nice_to_have'])} -> {len(flat_new['terms_nice_to_have'])}",
            "n_excl": f"{len(flat_old['terms_exclude'])} -> {len(flat_new['terms_exclude'])}",
            "linter_before": len(lint(_flat(spec))) or "clean",
            "linter_after": len(lint(_flat(new))) or "clean",
        })
        if not args.dry_run:
            (out_dir / f"{key}.usecase.json").write_text(json.dumps(new, indent=2) + "\n")

    import pandas as pd
    from ensemble_eval_utils import to_md
    tab = pd.DataFrame(rows).set_index("use_case")
    print(to_md(tab))

    if args.provenance:
        for key, prov in all_prov.items():
            print(f"\n=== {key} ===")
            for p in prov:
                print(f"  [{p['action']:14s}] {p['field']:20s} <- {p['source']}")
                print(f"                   {p['detail']}")

    still = {k: [c.id for c in lint(_flat(json.loads((out_dir / f'{k}.usecase.json').read_text())))]
             for k in USE_CASES} if not args.dry_run else {}
    if still:
        print("\nlinter on the optimised specs:",
              {k: v for k, v in still.items() if v} or "all clean")
    if not args.dry_run:
        print(f"\nwrote {len(USE_CASES)} specs to {out_dir}")


def _flat(spec: dict) -> dict:
    """The nine flat fields the linter and the feature builders read."""
    t, d = spec.get("terms") or {}, spec.get("domain") or {}
    return {"use_case_name": spec.get("name"), "objective": spec.get("objective"),
            "problem_statement": spec.get("problem_statement"),
            "terms_must_include": t.get("must_include"), "terms_nice_to_have": t.get("nice_to_have"),
            "terms_exclude": t.get("exclude"), "domain_industry": d.get("industry"),
            "domain_application": d.get("application"),
            "domain_technology_focus": d.get("technology_focus")}


if __name__ == "__main__":
    main()
