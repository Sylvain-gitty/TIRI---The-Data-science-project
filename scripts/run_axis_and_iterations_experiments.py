"""run_axis_and_iterations_experiments.py — fetches OOF for the supervised embedding-axis
test (add / replace) and the iterations=300 push, 5 seeds each, in one parallel batch.

All three new configs are compared against the already-fetched tuned (iterations=150,
depth=4) baseline in reports/wf_ensemble_v2_hparam_oof.json (screening/tuned keys) - no
need to refetch that one.

Usage:
    python scripts/run_axis_and_iterations_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reports" / "wf_ensemble_v2_axis_iterations_oof.json"
N_SEEDS = 5

# (result_key, function_name, args-without-seed)
JOBS = [
    ("axis_add", "run_catboost_axis_oof", ("add", 150, 4)),
    ("axis_replace", "run_catboost_axis_oof", ("replace", 150, 4)),
    ("iterations_300", "run_catboost_hparam_oof", (300, 4)),
]


def main() -> None:
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}

    calls = []
    for key, fn_name, args in JOBS:
        for seed in range(N_SEEDS):
            if str(seed) in results.get(key, {}):
                continue  # already fetched (e.g. a prior run got wiped mid-way) - don't redo it
            calls.append((key, fn_name, args, seed))

    if not calls:
        print("Nothing to do - all calls already present in the output file.")
        return

    # Group by function name since each is a different deployed Modal function.
    by_fn = {}
    for key, fn_name, args, seed in calls:
        by_fn.setdefault(fn_name, []).append((key, args, seed))

    print(f"Running {len(calls)} calls across {len(by_fn)} Modal functions (parallelized) ...")
    for fn_name, items in by_fn.items():
        fn = modal.Function.from_name("tiri-ensemble-ablation", fn_name)
        call_args = [(*args, seed) for _, args, seed in items]
        for (key, args, seed), res in zip(
            items, fn.starmap(call_args, return_exceptions=True)
        ):
            if isinstance(res, Exception):
                print(f"  FAILED: {key} seed={seed}: {res!r}")
                continue
            print(f"  done: {key} seed={seed}")
            results.setdefault(key, {})[str(seed)] = res
            OUT_JSON.write_text(json.dumps(results), encoding="utf-8")

    print(f"\nWritten to {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
