"""The reader arm of the spec-quality ablation: do contradictions and fluff cost an LLM?

`run_spec_quality_ablation.py` measured 14 brief variants against a **matcher** (BM25 + term
overlap). This measures the same 14 texts against a **reader** (an LLM screening one paper at a
time). It is probe `P-R` of `reports/wf_spec_quality_plan.md`; read that file's `## Amendment`
section before changing any bar here.

Why two arms at all
-------------------
The two consumers are already known to respond in opposite directions: term lists help matchers
and cost readers recall (`wf_llm_benchset_a_findings.md` §5). The variant where that matters most
is `conflicting` — exclusion terms moved into must-include, so the spec demands what it should
reject. It cost the matcher **-0.014** on set A and **-0.009** on set B, i.e. nothing either
time, and the reason is structural rather than lucky: **BM25 cannot notice that a term list
contradicts itself.** It just matches more strings. Meanwhile S-AL measured the reader case and
it was severe — a hand-written policy scored 0.772 against 0.828 for no policy at all, because it
named as a positive signal the category its own labels rejected.

If that asymmetry is real, a spec linter needs **two critics**, because neither failure is
visible to the consumer it does not affect. That is the whole question. One number decides
whether `academic_agent`'s D43 linter is a two-critic design or a one-critic design.

Reusing `variant()` rather than re-deriving the briefs
-----------------------------------------------------
`run_spec_quality_ablation.variant()` produces the brief frames, and this script imports it. One
function, both arms, so any difference between the results is about the *consumer* and cannot be
about the texts. Re-implementing the 14 degradations here would be the fastest way to produce a
finding that is really a diff between two copies of the same idea.

⚠️ The plan claimed `build_use_case_brief` was the config-driven seam. It is not — that is the
layer-1 notebook function and `use_case_brief_cols` appears nowhere in `scripts/`. The harness
renders through `render_brief`, hardcoded to `BRIEF_FIELDS`. The reuse works because `variant()`
rewrites the **columns** `render_brief` reads. That distinction changes the instrument check: the
failure mode to guard is not "a config column name is wrong" but "a variant rewrote a column
`render_brief` does not read", which produces the same silent no-op — a variant that renders
identically to `full`, scores identically, and looks like a finding. Hence the brief-hash assert.

Money, and the three ways to waste it
-------------------------------------
1. **`full` is scored with `brief_map=None`.** gemma's set-A P2 own-brief cell is already paid for
   under cache tag `P2|P2-v1|brief-v1`, and `score_frame` appends a brief tag only when
   `brief_map is not None`. Routing `full` through `brief_map=` with any tag changes the key and
   re-buys ~$1.32 of identical answers. `score_frame`'s own docstring records this having
   happened before, on a 12-cell grid.
2. **Every other variant carries a distinct `brief_tag`** (`spec-<variant>`). The key would differ
   anyway because the brief text is hashed, but without the tag every row would read `shuffled`
   in the cache and the audit trail would lie about what was bought.
3. **Nothing is bought without `--spend`.** `--dry-run` is the default: it builds the identical
   prompts and cache keys, looks them up on disk, and prints the hit rate and the projected cost
   from the model's own median recorded price. If stage 1's `full` cell is not 100% cached, the
   key has changed and stage 2 would cost double — the run stops rather than proceeding.

Staging
-------
  Stage 1 (set A, 2,000-row subsample, ~$0.80): the 4 decisive variants. **Estimates only, no
  verdict** — set A has had five selection passes and `full` here is free.
  Stage 2 (set B, held-out rows only, ~$3.30): the same 4 plus the shuffled floor. **The verdict
  is read here.** Held-out rows are also the population the matcher arm scored, so this is the
  comparable subset and not merely the cheap one.

Ceiling $10. The remaining 10 variants are out of budget (~$6.6 more on B) and stay unbought
unless someone raises it deliberately.

    python scripts/run_spec_quality_reader.py --stage 1              # dry run, free
    python scripts/run_spec_quality_reader.py --stage 1 --spend
    python scripts/run_spec_quality_reader.py --stage 2 --spend
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from benchset_loader import case_control_sample, drop_ab_crossing, load_set  # noqa: E402
from llm_pipeline_utils import dry_run_frame, render_brief, score_frame  # noqa: E402
from run_llm_screening import shuffled_brief_map  # noqa: E402
from run_spec_quality_ablation import variant  # noqa: E402

MODEL = "google/gemma-4-31b-it"
PROMPT = "P2"

# The decisive variants: `full` is the reference and the rest are exactly the ones H1-H3 name.
# Deliberately not all 14 - at B's measured unit price all 14 is ~$21.6 against a $10 ceiling,
# and buying 10 variants nobody has a hypothesis about is not thrift.
#
# `vague_objective` is here even though the plan's staging text lists only four variants. H2 is
# about `fluff_replace` AND `vague_objective`, so four cells would have left a pre-registered
# hypothesis half-tested - and the two are not interchangeable: `fluff_replace` swaps real prose
# for confident-sounding generic prose, while `vague_objective` swaps it for an honest one-liner
# that names nothing selectable. A reader might well forgive one and not the other. ~$0.94 for a
# testable hypothesis instead of an untestable one.
STAGE_VARIANTS = ["full", "conflicting", "fluff_replace", "vague_objective", "keyword_flood"]

STAGE1_ROWS = 2000
SEED = 0


def brief_hashes(sample: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One rendered brief per (variant, collection), hashed.

    The instrument check. A variant that silently renders identically to `full` would score
    identically and read as "this degradation is free" — a finding made of nothing. Comparing
    hashes catches it before any money is spent, and it also catches the reverse error, a
    variant that changes a column `render_brief` never reads.
    """
    rows = []
    for name in names:
        d = variant(sample, name)
        for uc, g in d.groupby("use_case_key", sort=True):
            text = render_brief(g.iloc[0])
            rows.append({"variant": name, "use_case_key": uc,
                         "sha8": hashlib.sha256(text.encode()).hexdigest()[:8],
                         "chars": len(text)})
    return pd.DataFrame(rows)


def assert_variants_distinct(h: pd.DataFrame, names: list[str]) -> None:
    """Every variant must render differently from `full` on every collection.

    Raises rather than warns. A silent no-op here is indistinguishable from a real null result,
    and the whole probe rests on the 14 texts actually differing.
    """
    wide = h.pivot(index="use_case_key", columns="variant", values="sha8")
    bad = []
    for name in names:
        if name == "full":
            continue
        same = wide.index[wide[name] == wide["full"]].tolist()
        if same:
            bad.append(f"{name} renders identically to `full` on {same}")
    if bad:
        raise RuntimeError("brief-hash check FAILED - do not spend:\n  " + "\n  ".join(bad))
    per_coll = wide.nunique(axis=1)
    print(f"  brief-hash check: OK - {len(names)} variants render "
          f"{per_coll.min()}-{per_coll.max()} distinct texts per collection "
          f"(want {len(names)}); ELI18: every damaged brief really is different text, so a "
          f"score difference cannot be an accident of the code.")


def build_sample(stage: int) -> tuple[pd.DataFrame, str]:
    """The rows to score, and a human label for the surface."""
    if stage == 1:
        base = load_set("a")
        sample = case_control_sample(base).reset_index(drop=True)
        # A subsample of the SAME case-control sample the cached set-A cell was scored on, so
        # `full` stays a cache hit. Stratified within collection so every collection keeps its
        # positives - at 0.17% prevalence a naive 2,000-row draw would catch almost none.
        rng = np.random.default_rng(SEED)
        frac = STAGE1_ROWS / len(sample)
        parts = []
        for _, g in sample.groupby("use_case_key", sort=True):
            for _, gg in g.groupby("y", sort=True):
                take = max(1, int(round(len(gg) * frac)))
                parts.append(gg.iloc[rng.permutation(len(gg))[:take]])
        out = pd.concat(parts).sort_values(["use_case_key", "row_key"]).reset_index(drop=True)
        return out, f"set A subsample ({len(out):,} of {len(sample):,} rows)"

    base = drop_ab_crossing(load_set("b"), "b")
    sample = case_control_sample(base).reset_index(drop=True)
    held = sample[sample.split != "train"].reset_index(drop=True)
    return held, f"set B held-out rows ({len(held):,} of {len(sample):,} in the sample)"


def cells(sample: pd.DataFrame, names: list[str], shuffled: bool) -> list[dict]:
    """One entry per cell to score: its brief_map, its brief_tag, and its label.

    `full` gets `brief_map=None`. That is not a stylistic choice - see the module docstring.
    """
    out = []
    for name in names:
        if name == "full":
            out.append({"name": "full", "brief_map": None, "brief_tag": None})
            continue
        d = variant(sample, name)
        bm = {uc: render_brief(g.iloc[0]) for uc, g in d.groupby("use_case_key", sort=True)}
        out.append({"name": name, "brief_map": bm, "brief_tag": f"spec-{name}"})
    if shuffled:
        # The floor. A derangement of the real briefs: no collection keeps its own. It returned
        # AUC 0.777 -> 0.498 and F2@own 0.285 -> 0.000 on set A, so it is a known-good calibration
        # of "the reader is genuinely reading" rather than pattern-matching paper quality.
        # `CONTEXT.md` L258-260 requires this seam on anything claiming to read a brief.
        #
        # Imported rather than re-rolled, and that is not tidiness: a derangement is one of many
        # permutations, so a different RNG produces a different pairing, different prompts, and a
        # cache miss on a cell already bought. Rolling my own here scored 36% cached instead of
        # 100% - and worse, it would not have been the same floor the published 0.498 refers to.
        out.append({"name": "shuffled", "brief_map": shuffled_brief_map(sample, seed=SEED),
                    "brief_tag": "shuffled"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--spend", action="store_true",
                    help="actually call the API; without it this is a free dry run")
    ap.add_argument("--variants", nargs="+", default=STAGE_VARIANTS)
    ap.add_argument("--shuffled", action="store_true",
                    help="add the derangement floor (implied on stage 2). Free on set A, where "
                         "that cell is already cached - and without it a null result cannot be "
                         "told apart from a blind instrument")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--ceiling", type=float, default=10.0, help="hard USD stop for this run")
    args = ap.parse_args()

    sample, surface = build_sample(args.stage)
    print(f"stage {args.stage}: {surface}, {sample.use_case_key.nunique()} collections, "
          f"{int(sample.y.sum())} positive")

    h = brief_hashes(sample, args.variants)
    assert_variants_distinct(h, args.variants)

    plan = cells(sample, args.variants, shuffled=(args.shuffled or args.stage == 2))

    est = pd.DataFrame([
        dry_run_frame(sample, MODEL, PROMPT, brief_map=c["brief_map"],
                      brief_tag=c["brief_tag"]) | {"cell": c["name"]}
        for c in plan
    ]).set_index("cell")
    total = float(est.projected_usd.sum())
    print("\n--- dry run: what this would cost, and what is already paid for ---")
    print(est[["n_rows", "n_cached", "n_fresh", "hit_rate", "usd_per_row",
               "projected_usd"]].round(4).to_string())
    print(f"\nprojected fresh spend: ${total:.2f}  (ceiling ${args.ceiling:.2f})")
    print(f"ELI18: {int(est.n_cached.sum()):,} of {int(est.n_rows.sum()):,} answers are already "
          f"on disk from earlier runs and cost nothing to reuse; only the remaining "
          f"{int(est.n_fresh.sum()):,} would be bought.")

    # The tripwire. `full` is byte-identical to a cell already on disk, so anything less than a
    # full hit means the cache key moved and every later stage would cost double.
    if args.stage == 1 and "full" in est.index and est.loc["full", "hit_rate"] < 1.0:
        raise SystemExit(
            f"STOP: stage-1 `full` is {est.loc['full', 'hit_rate']:.1%} cached, expected 100%. "
            "The cache key has changed, so nothing here is comparable to the published set-A "
            "numbers and stage 2 would cost double. Diagnose before spending."
        )
    if total > args.ceiling:
        raise SystemExit(f"STOP: ${total:.2f} exceeds the ${args.ceiling:.2f} ceiling.")
    if not args.spend:
        print("\ndry run only - pass --spend to buy the fresh rows above.")
        return

    frames = []
    for c in plan:
        print(f"\nscoring {c['name']} ...", flush=True)
        r = score_frame(sample, MODEL, PROMPT, brief_map=c["brief_map"],
                        brief_tag=c["brief_tag"], concurrency=args.concurrency)
        r["spec_variant"] = c["name"]          # carried as a COLUMN, never in a filename:
        r["stage"] = args.stage               # `analyze_benchset_a.arm_of` maps filename
        r["surface"] = f"set_{'a' if args.stage == 1 else 'b'}"   # substrings to arms and would
        frames.append(r)                      # collapse all 14 variants into one arm.

    res = pd.concat(frames, ignore_index=True)
    out = REPO / "reports" / f"wf_spec_quality_reader_stage{args.stage}_responses.parquet"
    res.to_parquet(out, index=False)
    fresh = float(res.loc[~res.cached, "cost"].fillna(0).sum())
    print(f"\nwrote {out}  ({len(res):,} rows)")
    print(f"fresh spend this run: ${fresh:.2f} | replayed from cache: "
          f"{res.cached.mean():.1%} | parsed: {res.parsed.mean():.1%}")


if __name__ == "__main__":
    main()
