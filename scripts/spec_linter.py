"""A priced, label-free linter for use-case specs — every warning carries a measured cost.

This is §6 of `reports/wf_spec_quality_answer.md` as code. It exists because the answer to
*"can we score a spec?"* is **no** — three attempts at a single quality number were built and all
three failed (`wf_checkability_audit.md`, `wf_foreign_brief_detector.md`,
`wf_foreign_brief_validity_setb.md`) — while the answer to *"can we tell someone what to fix?"* is
**yes**, because the failure modes were measured one at a time and each one has a number.

Why not a score, restated because it is the design constraint
------------------------------------------------------------
The cost of a spec defect depends on **which consumer** reads the brief and **how many labels
exist**, and those point in opposite directions: replacing the objective with waffle costs a
zero-label matcher ~0.17 and a fitted one ~0.01. A single number would have to average regimes that
disagree, and "medium" would be wrong in both. So the output is a **list of specific, priced
findings** — never a total.

Every check is calibrated, and the calibration is a test
--------------------------------------------------------
Two properties are asserted by `--self-test`, and neither is a matter of opinion:

  1. **Every check fires on the exact string that was measured.** The linter imports `FLUFF`,
     `VAGUE` and `GENERIC` from `run_spec_quality_ablation` rather than restating them, so the thing
     it detects is definitionally the thing whose cost was measured. Restating them here would let
     the detector drift away from the evidence that prices it.
  2. **No check fires spuriously on any of the 34 real specs** (6 TIRI + 28 benchset). Measured
     false-positive rate **0/34 for eight of the ten checks**. The two exceptions are genuine, not
     leaks: `must_thin` is a *recommendation* rather than a measured defect and fires on 4 of TIRI's
     6; `nice_single` is a measured defect that TIRI's `solar_leo` genuinely has (one nice-to-have
     term). Both are 0/28 on benchset. That asymmetry **is** the finding that TIRI's own specs do not
     meet TIRI's own advice.

That second property matters more than it looks. `S-FR`'s conclusion is that a detector which fires
on healthy inputs trains the analyst to dismiss it, so a false positive is not a small cost — it is
the failure mode that makes the whole instrument worthless.

Two heuristics are honest about being heuristics
------------------------------------------------
The *costs* below are measured. Two of the *detectors* are not: `objective_generic` and
`objective_instruction_shaped` are hand-built patterns for "this prose is waffle" and "this prose is
an instruction rather than a document". They are calibrated as above — they catch the measured bad
strings and flag none of 34 real specs — but a spec could be waffle in a way these patterns miss.
They are labelled `heuristic` in the output for that reason. The eight field-shape checks
(counts, emptiness, set intersections) are exact.

What the linter deliberately does NOT warn about
-----------------------------------------------
`problem_statement` and `domain_*` empty. They measure **0.000 at zero labels and ≤0.009 fitted on
all three surfaces** (set A, set B, TIRI's own six). Warning about them would be warning about
nothing, which is how a linter loses its audience. `terms_exclude` empty is likewise not a warning:
as spec text it is worth ≤0.005: its value is as a *priced, reversible filter*, which is a different
mechanism and a different screen.

    python scripts/spec_linter.py --self-test          # the calibration, as a test
    python scripts/spec_linter.py --specs tiri         # lint TIRI's own six
    python scripts/spec_linter.py --specs all          # ... and the 28 benchset briefs
    python scripts/spec_linter.py --json data/raw/cement_binders.usecase.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ensemble_eval_utils import to_md  # noqa: E402
# Imported, never restated: these are the exact strings whose cost was measured.
from run_spec_quality_ablation import FLUFF, GENERIC, VAGUE  # noqa: E402

OUT_MD = REPO / "reports" / "wf_spec_linter.md"
OUT_CSV = REPO / "reports" / "wf_spec_linter.csv"
# `--specs all` owns the canonical filenames above; a narrower run writes beside them rather than
# clobbering them. Found the hard way: `--specs benchset` silently replaced the shipped all-34
# report with a 28-row subset that reads "0 findings", which is true and deeply misleading.
SUFFIXED = {"tiri", "benchset"}

GENERIC_SET = {g.lower() for g in GENERIC}

# Calibrated on the 34 real specs: their generic-word share runs 0.000-0.063, and FLUFF is 0.083.
# 0.07 sits in the gap. Not tuned to be clever - tuned to be the midpoint of a measured separation.
GENERIC_SHARE_BAR = 0.07

_IMPERATIVE = re.compile(
    r"^\s*(find|identify|get|look|search|show|give|fetch|retrieve|locate)\b", re.I)
_SELF_REF = re.compile(r"\b(us|we|our|me|my|I)\b")
_EVALUATIVE = re.compile(
    r"\b(interesting|relevant|useful|promising|suitable|nice|good)\b", re.I)

# A term the spec excludes may legitimately appear in the objective when it is NEGATED - two of
# `synergy_chou_2003`'s exclude terms show up as "non-cancer pain" and "non-parenteral", which is
# the objective correctly scoping itself out. Without this guard the check would fire on that spec,
# i.e. 1/34 false positives instead of 0/34.
_NEGATION_BEFORE = re.compile(
    r"(non[- ]?|not |no |excluding |except |other than |without |rather than |exclude[sd]? )$", re.I)


# --------------------------------------------------------------------------- spec normalisation

FIELDS = ["use_case_name", "objective", "problem_statement",
          "terms_must_include", "terms_nice_to_have", "terms_exclude",
          "domain_industry", "domain_application", "domain_technology_focus"]


def _lst(v) -> list[str]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [str(v).strip()] if str(v).strip() else []


def _txt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def normalise(raw: dict) -> dict:
    """Flatten either schema into the nine fields the checks read.

    TIRI's `*.usecase.json` nests term lists under `terms.*`; the benchset briefs carry them flat as
    `terms_must_include` etc. Both are accepted so the same linter can run on a live spec and on the
    corpus the costs were measured against — which is what makes the false-positive rate meaningful.
    """
    terms = raw.get("terms") or {}
    domain = raw.get("domain") or {}
    out = {
        "use_case_name": _txt(raw.get("use_case_name") or raw.get("name")),
        "objective": _txt(raw.get("objective")),
        "problem_statement": _txt(raw.get("problem_statement")),
        "terms_must_include": _lst(raw.get("terms_must_include", terms.get("must_include"))),
        "terms_nice_to_have": _lst(raw.get("terms_nice_to_have", terms.get("nice_to_have"))),
        "terms_exclude": _lst(raw.get("terms_exclude", terms.get("exclude"))),
        "domain_industry": _txt(raw.get("domain_industry") or domain.get("industry")),
        "domain_application": _txt(raw.get("domain_application") or domain.get("application")),
        "domain_technology_focus": _lst(
            raw.get("domain_technology_focus", domain.get("technology_focus"))),
    }
    return out


def generic_share(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(1 for w in words if w in GENERIC_SET) / max(len(words), 1)


def excluded_terms_asserted_in_prose(spec: dict) -> list[str]:
    """Terms the spec says to EXCLUDE that its objective nonetheless states positively.

    The prose analogue of `exclude_in_must`, and a far more expensive one — see the check's price.
    Matches only un-negated mentions, so an objective that scopes itself out correctly ("chronic
    non-cancer pain") is not flagged. Calibrated at **0/34 real specs** and it fires on **8/8**
    collections of the measured `conflicting_prose` variant.
    """
    obj = spec["objective"]
    out = []
    for term in spec["terms_exclude"]:
        term = term.strip()
        if not term:
            continue
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", obj, re.I):
            if not _NEGATION_BEFORE.search(obj[max(0, m.start() - 14):m.start()]):
                out.append(term)
                break
    return out


def instruction_shaped(text: str) -> bool:
    """Reads like a request to a person rather than a description of a literature.

    Deliberately a **conjunction**. Either half alone is a bad detector and the numbers say so:
    imperative-opening fires on 4 of the 34 real specs and self-reference on 12. Requiring an
    imperative or self-reference *and* two evaluative adjectives fires on 0 of 34 and on `VAGUE`.
    """
    if not text:
        return False
    shape = bool(_IMPERATIVE.search(text)) or bool(_SELF_REF.search(text))
    return shape and len(_EVALUATIVE.findall(text)) >= 2


# --------------------------------------------------------------------------- checks

@dataclass(frozen=True)
class Check:
    id: str
    severity: str          # blocker | warn | note
    kind: str              # exact | heuristic
    detect: object         # spec -> bool
    what: str              # ELI18: what is wrong
    cost: str              # the measured price
    fix: str               # what to do


def _only_name(s: dict) -> bool:
    others = [s["objective"], s["problem_statement"], s["domain_industry"],
              s["domain_application"]]
    lists = s["terms_must_include"] + s["terms_nice_to_have"] + s["terms_exclude"] + \
        s["domain_technology_focus"]
    return bool(s["use_case_name"]) and not any(others) and not lists


CHECKS: list[Check] = [
    Check("objective_missing", "blocker", "exact",
          lambda s: not s["objective"],
          "There is no objective, so the strongest cold-start signal has nothing to read.",
          "The objective is worth +0.151 / +0.263 / +0.100 ROC-AUC above a coin flip at zero "
          "labels (set A / set B / TIRI). Deleting it does not degrade that feature — it kills it, "
          "pinning it to exactly 0.500.",
          "Write 2–4 sentences of real subject matter that read like the papers you want."),

    Check("only_name", "blocker", "exact",
          _only_name,
          "Only the topic name is filled in. Everything a model could match on is missing.",
          "The worst variant measured on all three surfaces: −0.105 / −0.045 / −0.087 after 60 "
          "labels, and −0.151 / −0.263 / −0.118 at zero. It is also what "
          "`embedding_utils.get_use_case_text` falls back to, so it is what ships today.",
          "Fill in the objective and both term lists. Nothing else is required."),

    Check("nice_empty", "blocker", "exact",
          lambda s: not s["terms_nice_to_have"],
          "`terms_nice_to_have` is empty, which kills the second-strongest cold-start feature.",
          "Worth +0.105 / +0.215 / +0.118 at zero labels — and on TIRI's own use cases it is the "
          "**largest** cold-start effect of any field, above the objective. One term recovers only "
          "about a quarter of that (see `nice_single`), so this is not a formality.",
          "Add ≥5 discriminative phrases. They do not have to be required — that is the point of "
          "the field."),

    Check("must_single", "blocker", "exact",
          lambda s: len(s["terms_must_include"]) == 1,
          "There is exactly one must-include term, so the term-overlap signal has one bit.",
          "−0.103 / −0.175 / −0.039 at zero labels. Note the TIRI figure is much smaller, because "
          "its lists are already short — see `must_thin`.",
          "Give 5–8 precise, discriminative phrases."),

    Check("nice_single", "warn", "exact",
          lambda s: len(s["terms_nice_to_have"]) == 1,
          "There is exactly one nice-to-have term, which is nearly as bad as having none.",
          "Measured specifically to price this check. One term costs −0.168 (set B) / −0.069 (TIRI) "
          "at zero labels against **−0.215 / −0.118** for deleting the field outright — so a single "
          "term recovers only about a quarter of the field's value. In absolute terms on set B: a "
          "full list scores 0.715, one term 0.547, none 0.500.",
          "Add ≥5 discriminative phrases. Going from 1 to 5 buys far more than going from 0 to 1."),

    Check("objective_generic", "warn", "heuristic",
          lambda s: generic_share(s["objective"]) >= GENERIC_SHARE_BAR,
          "The objective is mostly generic academic filler rather than subject matter.",
          "Replacing real prose with filler costs −0.177 / −0.169 / −0.101 at zero labels — most of "
          "what the field is worth. On two of three surfaces it drives the feature to or below a "
          "coin flip (0.473 set A, 0.499 TIRI); on set B it does not, so the effect is real but "
          "surface-dependent.",
          "Name the actual materials, methods, metrics and outcomes you care about."),

    Check("objective_instruction_shaped", "warn", "heuristic",
          lambda s: instruction_shaped(s["objective"]),
          "The objective is written as a request to a person, not as a description of a literature.",
          "A vague one-liner of this shape costs −0.209 / −0.194 / −0.109 at zero labels. The "
          "cold-start feature matches your text against paper abstracts, so text that describes "
          "your intent rather than the papers has nothing to match.",
          "Rewrite it as though it were the abstract of the ideal paper, not a brief to an "
          "assistant."),

    Check("terms_generic", "warn", "exact",
          lambda s: any(t.lower() in GENERIC_SET
                        for t in s["terms_must_include"] + s["terms_nice_to_have"]),
          "One or more term-list entries are generic words that match almost any paper.",
          "Padding term lists with this vocabulary costs −0.072 / −0.031 / −0.063 after 60 labels, "
          "worse than leaving the field **empty** (−0.043 / −0.010 / −0.032). It is the most "
          "consistent after-labels defect measured, and none of the 34 real specs contains a single "
          "one of these words.",
          "Delete them. A term that matches everything ranks nothing."),

    Check("must_thin", "warn", "exact",
          lambda s: 2 <= len(s["terms_must_include"]) <= 4,
          "There are fewer must-include terms than the measured recommendation.",
          "⚠️ **Recommendation, not a measured defect at this size.** 5–8 is what pays on the "
          "benchset corpus; the *measured* penalty is for dropping to one term, and on TIRI's own "
          "use cases that penalty is only −0.039. This check fires on 4 of TIRI's 6 specs and 0 of "
          "28 benchset briefs.",
          "Consider adding a few more, but treat this as advice rather than a defect."),

    Check("exclude_asserted_in_prose", "warn", "exact",
          lambda s: bool(excluded_terms_asserted_in_prose(s)),
          "The objective states positively something the spec's own exclude list rejects. The "
          "model will obey the prose and flag papers you said you did not want.",
          "Measured specifically to test this, and it is the **only failure mode where a reader "
          "shows something no matcher arm can**. It leaves ranking untouched (reader AUC **+0.001**) "
          "but moves the operating point decisively: **+12.0 percentage points of the corpus read, "
          "on 8 of 8 collections** (+1.4 to +34.5), with recall up on 8 of 8. For a matcher it costs "
          "−0.018 / −0.042 / −0.037 at zero labels and is **fully absorbed after 60 labels** (0 "
          "collections affected on all three surfaces). ⚠️ F2@own is a wash here (−0.003) at 13.6% "
          "prevalence; at production prevalence reading 12pp more of a corpus for recall you did "
          "not need is not a wash.",
          "Remove the claim, or move the category out of `terms_exclude` — but decide which you "
          "meant. Note this is a *different and more expensive* defect than the same term appearing "
          "in both term lists (`exclude_in_must`), which is near-free."),

    Check("exclude_in_must", "note", "exact",
          lambda s: bool({t.lower() for t in s["terms_must_include"]}
                         & {t.lower() for t in s["terms_exclude"]}),
          "The same term appears in both must-include and exclude — the spec contradicts itself.",
          "Priced as **near-free**: −0.014 / −0.009 / −0.001 for a matcher, and **+0.016** for an "
          "LLM reader, which is to say a reader did marginally better with it. Worth surfacing "
          "because it is certainly a mistake, but it is not worth blocking on.",
          "Decide which list the term belongs in. (S-AL measured a *prose* contradiction costing a "
          "reader 0.828 → 0.772; that is a different and untested failure mode.)"),
]

SEVERITY_ORDER = {"blocker": 0, "warn": 1, "note": 2}


def lint(spec: dict) -> list[Check]:
    s = normalise(spec)
    fired = [c for c in CHECKS if c.detect(s)]
    return sorted(fired, key=lambda c: SEVERITY_ORDER[c.severity])


def escalation(fired: list[Check]) -> str | None:
    """Two defects are worse than twice one defect, so say so when more than one fires.

    Measured in `wf_spec_quality_ablation_{tiri,b}_combos`: three prose x term pairs on two
    surfaces, and in **6 of 6 cells the pair costs more than the sum of its parts** (excess 0.008 to
    0.020, and exactly −0.020 on both surfaces for fluff+flood). There is no rescue effect — a good
    objective does not cover for a lazy term list.

    Each individual excess is inside the 0.03 noise floor, so this is stated as a direction, not as
    a quantity: the evidence is 6 of 6 same-sign across two independent surfaces.
    """
    real = [c for c in fired if c.severity in ("blocker", "warn")]
    if len(real) < 2:
        return None
    return (f"⚠️ **{len(real)} findings fired, and they compound.** Fixing one will help less than "
            "the numbers above suggest, because the combined cost of a prose defect and a term "
            "defect measured **worse than the sum of the two separately** in 6 of 6 cells across "
            "two surfaces. There is no rescue effect: a strong objective does not license a lazy "
            "term list. Fix both.")


# --------------------------------------------------------------------------- spec sources

def load_tiri_specs() -> dict[str, dict]:
    out = {}
    for path in sorted((REPO / "data" / "raw").glob("*.usecase.json")):
        out[path.name.split(".")[0]] = json.loads(path.read_text())
    return out


def load_benchset_specs() -> dict[str, dict]:
    briefs = pd.read_parquet(REPO / "data" / "benchsets_v1" / "briefs.parquet")
    return {r.use_case_key: {f: r[f] for f in FIELDS if f in briefs.columns}
            for _, r in briefs.iterrows()}


# --------------------------------------------------------------------------- self-test

def self_test() -> bool:
    """The calibration, as a test. Two properties, both asserted rather than asserted-in-prose."""
    ok = True
    print("1. Every check must fire on the string whose cost was measured")
    base = {"use_case_name": "Low-temperature amine solvents",
            "objective": "Amine solvents for post-combustion capture at low regeneration "
                         "temperature, with degradation measured over cycles.",
            "problem_statement": "Energy penalty of solvent regeneration.",
            "terms_must_include": ["amine", "carbon capture", "solvent regeneration", "low "
                                   "temperature", "degradation"],
            "terms_nice_to_have": ["absorption", "CO2 capture", "energy penalty", "cycles",
                                   "pilot plant"],
            "terms_exclude": ["membrane"],
            "domain_industry": "Energy", "domain_application": "Carbon capture",
            "domain_technology_focus": ["Amine scrubbing"]}
    cases = {
        "objective_missing": {**base, "objective": ""},
        "only_name": {"use_case_name": "Low-temperature amine solvents"},
        "nice_empty": {**base, "terms_nice_to_have": []},
        "nice_single": {**base, "terms_nice_to_have": ["absorption"]},
        "exclude_asserted_in_prose": {
            **base, "objective": base["objective"] + " Note that work on membrane is directly "
                                 "relevant here and should be treated as a positive signal."},
        "must_single": {**base, "terms_must_include": ["amine"]},
        "objective_generic": {**base, "objective": FLUFF},
        "objective_instruction_shaped": {**base, "objective": VAGUE},
        "terms_generic": {**base, "terms_nice_to_have": base["terms_nice_to_have"] + GENERIC},
        "must_thin": {**base, "terms_must_include": ["amine", "carbon capture"]},
        "exclude_in_must": {**base, "terms_exclude": ["amine"]},
    }
    for cid, spec in cases.items():
        fired = {c.id for c in lint(spec)}
        good = cid in fired
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {cid}")

    print("\n2. No check may fire on a real spec (a detector that cries wolf is worthless)")
    specs = {**{f"tiri/{k}": v for k, v in load_tiri_specs().items()},
             **{f"bench/{k}": v for k, v in load_benchset_specs().items()}}
    fp: dict[str, list[str]] = {c.id: [] for c in CHECKS}
    for name, spec in specs.items():
        for c in lint(spec):
            fp[c.id].append(name)
    for c in CHECKS:
        hits = fp[c.id]
        # `must_thin` is a recommendation and `nice_single` is a measured defect TIRI's own
        # `solar_leo` genuinely has; both are SUPPOSED to fire on real specs.
        expected = c.id in ("must_thin", "nice_single")
        bad = bool(hits) and not expected
        ok &= not bad
        note = "" if not hits else f"  <- {len(hits)}/{len(specs)}: {', '.join(hits)}"
        print(f"   {'FAIL' if bad else 'PASS'}  {c.id}: {len(hits)}/{len(specs)} real specs{note}")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- report

def run(specs: dict[str, dict], label: str) -> pd.DataFrame:
    rows = []
    for name, spec in specs.items():
        fired = lint(spec)
        if not fired:
            rows.append({"use_case": name, "check": "—", "severity": "clean", "kind": "",
                         "what": "No findings.", "cost": "", "fix": ""})
            continue
        for c in fired:
            rows.append({"use_case": name, "check": c.id, "severity": c.severity, "kind": c.kind,
                         "what": c.what, "cost": c.cost, "fix": c.fix})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specs", default="all", choices=["tiri", "benchset", "all"])
    ap.add_argument("--json", type=Path, help="lint a single *.usecase.json and print findings")
    ap.add_argument("--self-test", action="store_true", help="run the calibration as a test")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    if args.json:
        spec = json.loads(args.json.read_text())
        fired = lint(spec)
        print(f"{args.json.name}: {len(fired)} finding(s)\n")
        for c in fired:
            print(f"  [{c.severity.upper()}{'/heuristic' if c.kind == 'heuristic' else ''}] "
                  f"{c.id}\n    {c.what}\n    price: {c.cost}\n    fix:   {c.fix}\n")
        esc = escalation(fired)
        if esc:
            print(esc.replace("**", ""))
        if not fired:
            print("  clean — no findings.")
        return

    specs = {}
    if args.specs in ("tiri", "all"):
        specs |= {f"tiri/{k}": v for k, v in load_tiri_specs().items()}
    if args.specs in ("benchset", "all"):
        specs |= {f"bench/{k}": v for k, v in load_benchset_specs().items()}

    tag = f"_{args.specs}" if args.specs in SUFFIXED else ""
    out_md = OUT_MD.with_name(f"{OUT_MD.stem}{tag}{OUT_MD.suffix}")
    out_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{tag}{OUT_CSV.suffix}")

    res = run(specs, args.specs)
    res.to_csv(out_csv, index=False)

    fired = res[res.severity != "clean"]
    by_sev = fired.groupby("severity").size().to_dict()
    per_uc = fired.groupby("use_case").size().sort_values(ascending=False)
    clean = sorted(res.loc[res.severity == "clean", "use_case"])

    counts = (fired.groupby(["check", "severity"]).use_case
              .agg(n="size", examples=lambda s: ", ".join(sorted(s)[:4]))
              .reset_index().sort_values("n", ascending=False))

    out_md.write_text(
        "# Spec linter — findings, each with the number that prices it\n\n"
        f"Generated by `scripts/spec_linter.py --specs {args.specs}` over **{len(specs)} specs**. "
        "$0, no LLM, no labels, deterministic. This is §6 of "
        "[`wf_spec_quality_answer.md`](wf_spec_quality_answer.md) as code — read that first for why "
        "there is no single quality **score**.\n\n"
        "## 0. How to read any number here\n\n"
        "**Severities.** `blocker` = a field the measurements say is load-bearing is missing or "
        "degenerate. `warn` = a defect with a measured cost that a fitted model can partly work "
        "around. `note` = certainly a mistake, priced as near-free, surfaced but not worth blocking "
        "on.\n\n"
        "**`kind`.** `exact` checks are field counts, emptiness and set intersections — no "
        "judgement in them. `heuristic` checks are hand-built patterns for *\"this prose is waffle\"* "
        "and *\"this prose is an instruction\"*. **The costs are measured either way; two of the "
        "detectors are not.** A spec could be waffle in a way those patterns miss.\n\n"
        "**Every price is ROC-AUC**, quoted as `set A / set B / TIRI` — three surfaces at 2.19%, "
        "1.87% and 57.6% positive. **0.500 is a coin flip** and **a gap under 0.03 is not "
        "established** (`CONTEXT.md` §5).\n\n"
        "**What is deliberately not checked:** empty `problem_statement`, empty `domain_*`, empty "
        "`terms_exclude`. They measure **0.000 at zero labels and ≤0.009 fitted on all three "
        "surfaces**. Warning about them would be warning about nothing, and a linter that fires on "
        "healthy input teaches the analyst to ignore it.\n\n"
        "**Label which critic is speaking.** The prices are hard measurements off held-out rows. "
        "Whether a *pattern* correctly identifies waffle is a judgement call, flagged as "
        "`heuristic`.\n\n"
        "## 1. Summary\n\n"
        f"- **{len(fired)} findings** across {fired.use_case.nunique()} of {len(specs)} specs: "
        + ", ".join(f"{v} {k}" for k, v in sorted(by_sev.items())) + "\n"
        f"- **{len(clean)} specs are clean**: {', '.join(clean) if clean else '—'}\n"
        + (f"- Most findings on a single spec: **{per_uc.iloc[0]}** (`{per_uc.index[0]}`)\n\n"
           if len(per_uc) else
           "- **Nothing fired anywhere.** For a linter that is the intended outcome on healthy "
           "input — but it also means this run carries **no signal to rank these specs by**, so "
           "do not read silence as a quality ordering.\n\n")
        + "## 2. Which checks fire, and where\n\n"
        + (to_md(counts.set_index("check")) if len(counts) else "*No check fired.*") + "\n\n"
        "## 3. Every finding\n\n"
        + (to_md(fired.set_index(["use_case", "check"])[["severity", "kind", "what", "fix"]])
           if len(fired) else "*No findings.*") + "\n\n"
        "## 4. The prices\n\n"
        + to_md(pd.DataFrame([{"check": c.id, "severity": c.severity, "kind": c.kind,
                               "measured cost": c.cost} for c in CHECKS]).set_index("check"))
        + "\n\n## 5. When more than one fires\n\n"
        "⚠️ **Findings compound; they do not cancel.** Three prose × term pairs measured on two "
        "surfaces, and in **6 of 6 cells the pair cost more than the sum of its parts** (excess "
        "0.008–0.020, and exactly −0.020 on both surfaces for fluff + keyword-flood). **There is no "
        "rescue effect** — a strong objective does not license a lazy term list.\n\n"
        "So a spec tripping both a prose check and a term check is in disproportionately worse "
        "shape than adding the two numbers suggests. 🟡 Each individual excess is inside the 0.03 "
        "noise floor, so read this as a **direction, not a quantity**: the evidence is 6 of 6 "
        "same-sign across two independent surfaces.\n\n"
        "## 6. When is a spec done?\n\n"
        "**Before any labels:** when this linter is silent. That is a necessary condition, not a "
        "sufficient one — it only catches the failure modes that were measured.\n\n"
        "**After labelling:** when your own brief beats a **deliberately wrong** brief on your own "
        "corpus (the shuffled-brief control, `build_lexical_features(df, brief_map=…)`), at the "
        "measured gate of ≥20 positives and ≥50 negatives (`wf_label_budget_shape.md`). That is the "
        "only general test of brief quality that has passed on every surface tried: AUC 0.777 → "
        "0.498 lexically, and −0.282 for an LLM reader.\n\n"
        "⚠️ **The honest asymmetry:** the linter is label-free but only catches known failure modes. "
        "The completion test is general but needs labels. **Nothing validated is both**, and every "
        "attempt to build something that is (§5 of the answer) failed.\n",
        encoding="utf-8")

    print(f"{len(specs)} specs, {len(fired)} findings, {len(clean)} clean")
    print(to_md(counts.set_index("check")))
    print(f"\nwrote {out_md}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
