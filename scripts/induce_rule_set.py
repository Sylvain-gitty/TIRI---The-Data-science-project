"""B2 / "P5" — induce a screening rule set per use case from train-fold labels only.

The question this answers
------------------------
The pilot's §3 showed the models really do read the brief (shuffled-brief AUC collapses to
0.49 and the predicted-positive rate to zero). If the brief is load-bearing, then brief
*quality* is a lever nobody has pulled: every arm so far has used whatever brief the export
happened to ship. So write a better one — not by hand, but by showing a strong model what
the labeller actually accepted and rejected, and asking it to state the operative rule.

`CONTEXT.md` §4 sharpens why this matters here. The old SYNERGY briefs failed the
shuffled-brief control because they were "a description of what a review did, not a
statement of what to include", with no curated term lists. This export has term lists, but
they were written by an LLM from the review's abstract — still blind to the labels. An
induced rule set is the first brief on this corpus that has actually seen a label.

**This is a fitted parameter, not a prompt.** Three consequences, all enforced below:

  1. Induced from **train-fold rows only**, so the test split stays clean. The same rule
     `select_few_shot_examples` enforces for demonstrations, for the same reason.
  2. **Versioned and pinned** to a JSON artefact. `relevance_score` was disqualified as a
     feature for exactly this drift (`wf_ensemble_report.md` §0); an unversioned rule set
     would repeat the mistake with a moving target nobody could reproduce.
  3. Reported against a **zero-shot** baseline that never saw a label, so the comparison is
     not "did more information help" (it must) but "is the brief where the information is
     best spent, versus few-shot examples or a supervised model on the same labels".

Output: `reports/wf_llm_setA_rules_v1.json` — `{rule_set_version, model, briefs: {use_case: text}}`
consumable directly by `run_llm_screening.py --brief induced`.

Usage:
    python scripts/induce_rule_set.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchset_loader import case_control_sample, load_set_a  # noqa: E402
from llm_pipeline_utils import (  # noqa: E402
    BRIEF_FIELDS, OpenRouterClient, extract_json, render_brief,
)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:  # pragma: no cover
    pass

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "wf_llm_setA_rules_v1.json"
RULE_SET_VERSION = "setA-rules-v1"
INDUCER = "qwen/qwen3.5-397b-a17b"
K_PER_CLASS = 30
ABSTRACT_CHARS = 700

SYSTEM = (
    "You are a systematic-review methodologist. You write screening protocols: precise, "
    "operational inclusion and exclusion criteria that a screener can apply to a title and "
    "abstract without knowing anything else about the review."
)

INSTRUCTION = f"""\
Below is the brief currently used to screen papers for one systematic review, followed by
papers a human expert has already judged - some INCLUDED, some EXCLUDED.

Your job is to write a better brief, by working out what the expert's decisions actually
turned on. Look for the distinctions the current brief misses: study designs that were
accepted or rejected, populations, outcomes, phrasing that reliably appears in included
papers, and near-miss topics that look relevant but were excluded.

Rules:
- Write criteria that are checkable from a title and abstract alone.
- The exclusion terms matter most. Near-misses are what a screener gets wrong, so name the
  topics that look on-target but were rejected.
- Do not name or quote the specific example papers. Write the general rule they imply.
- Do not simply copy the current brief back. If a field is already right, improve its
  precision; if it is misleading, replace it.

Respond with JSON only. No markdown fences, no commentary before or after.
{{
  "use_case_name": "<short topic name>",
  "objective": "<what this review is looking for, 2-4 sentences>",
  "problem_statement": "<the screening decision being made, 1-2 sentences>",
  "terms_must_include": ["<term>", ...],
  "terms_nice_to_have": ["<term>", ...],
  "terms_exclude": ["<term>", ...],
  "domain_industry": "<field>",
  "domain_application": "<application>",
  "domain_technology_focus": ["<focus>", ...]
}}
Aim for 8-15 must-include terms, 5-12 nice-to-have and 8-15 exclusion terms.
"""


def _examples(frame, label: int, k: int, seed: int = 0) -> str:
    sub = frame[frame.y == label]
    sub = sub.sample(n=min(k, len(sub)), random_state=seed)
    out = []
    for i, r in enumerate(sub.itertuples(), 1):
        abstract = (r.abstract or "")[:ABSTRACT_CHARS]
        out.append(f"[{i}] {r.title}\n{abstract}")
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=INDUCER)
    ap.add_argument("--k", type=int, default=K_PER_CLASS)
    args = ap.parse_args()

    sample = case_control_sample(load_set_a())
    train = sample[sample.split == "train"]
    client = OpenRouterClient(model=args.model, max_tokens=2500)

    briefs, fields, meta = {}, {}, []
    for uc, grp in train.groupby("use_case_key", sort=True):
        current = render_brief(dict(zip(grp.columns, grp.iloc[0])))
        pos, neg = _examples(grp, 1, args.k), _examples(grp, 0, args.k)
        n_pos = min(args.k, int((grp.y == 1).sum()))
        n_neg = min(args.k, int((grp.y == 0).sum()))
        user = (
            f"{INSTRUCTION}\n\n=== CURRENT BRIEF ===\n{current}\n\n"
            f"=== INCLUDED BY THE EXPERT ({n_pos} papers) ===\n{pos}\n\n"
            f"=== EXCLUDED BY THE EXPERT ({n_neg} papers) ===\n{neg}\n"
        )
        res = client.call(SYSTEM, user, tag=f"induce|{RULE_SET_VERSION}")
        if res["error"]:
            raise SystemExit(f"{uc}: induction failed - {res['error']}")
        obj = extract_json(res["content"])
        if not obj:
            raise SystemExit(f"{uc}: induced rule set did not parse\n{res['content'][:600]}")
        missing = [c for c, _ in BRIEF_FIELDS if c not in obj]
        if missing:
            print(f"    {uc}: induced brief omits {missing} - those lines will be absent")
        # Rendered through render_brief so the ONLY difference between this arm and the
        # supplied-brief arm is the content of the fields, never their formatting.
        briefs[uc] = render_brief(obj)
        # The structured fields are kept as well as the rendered prose, because the rule set
        # is not only an LLM prompt: `compare_setA_induced_brief.py` feeds these same fields
        # to the BM25/overlap block and to cosine-to-brief, which need the columns and not
        # the paragraph. Storing only the render made that experiment impossible without
        # going back to the response cache.
        fields[uc] = {c: obj.get(c) for c, _ in BRIEF_FIELDS}
        meta.append({"use_case_key": uc, "n_pos_shown": n_pos, "n_neg_shown": n_neg,
                     "n_train": len(grp), "chars": len(briefs[uc]),
                     "cost": res["cost"], "cached": res["cached"]})
        print(f"  {uc}: {len(briefs[uc])} chars from {n_pos}+{n_neg} train examples "
              f"(${res['cost']:.4f}{', cached' if res['cached'] else ''})")

    OUT.write_text(json.dumps({
        "rule_set_version": RULE_SET_VERSION,
        "inducer_model": args.model,
        "k_per_class": args.k,
        "abstract_chars": ABSTRACT_CHARS,
        "split": "train",
        "briefs": briefs,
        "fields": fields,
        "per_use_case": meta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}  (total ${sum(m['cost'] or 0 for m in meta):.4f})")


if __name__ == "__main__":
    main()
