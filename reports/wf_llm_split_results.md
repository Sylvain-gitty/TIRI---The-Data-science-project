# LLM screening — F2 across train / test / validate (all six use cases pooled)

The threshold is the only fitted parameter: **t\* is chosen on train and applied unchanged to test and validate.** `f2_own` is the model's own verdict with nothing fitted, and should not move across splits.

| index | split    | n    | positive | prevalence | f2_all_positive |
|-------|----------|------|----------|------------|-----------------|
| 0     | test     | 369  | 213      | 0.577      | 0.872           |
| 1     | train    | 1110 | 639      | 0.576      | 0.872           |
| 2     | validate | 369  | 213      | 0.577      | 0.872           |

## All cells

| index | model                      | variant | t_star | f2_train | f2_test | f2_validate | lift_over_all_pos | pos_rate_at_t | f2own_test | auc_test |
|-------|----------------------------|---------|--------|----------|---------|-------------|-------------------|---------------|------------|----------|
| 0     | gemma-4-31b-it             | P1      | 0.060  | 0.878    | 0.874   | 0.870       | 0.002             | 0.916         | 0.500      | 0.716    |
| 1     | gemma-4-31b-it             | P2      | 0.110  | 0.879    | 0.876   | 0.866       | 0.004             | 0.862         | 0.809      | 0.731    |
| 2     | gemma-4-31b-it             | P2lp    | 0.010  | 0.796    | 0.800   | 0.799       | -0.072            | 0.688         | 0.774      | 0.728    |
| 3     | gemma-4-31b-it             | P4      | 0.010  | 0.886    | 0.885   | 0.875       | 0.013             | 0.938         | 0.442      | 0.691    |
| 7     | gpt-oss-20b                | P1      | 0.010  | 0.831    | 0.837   | 0.837       | -0.035            | 0.783         | 0.448      | 0.724    |
| 8     | gpt-oss-20b                | P2      | 0.060  | 0.865    | 0.850   | 0.874       | -0.022            | 0.816         | 0.565      | 0.745    |
| 9     | gpt-oss-20b                | P2lp    | 0.010  | 0.548    | 0.530   | 0.528       | -0.342            | 0.350         | 0.530      | 0.694    |
| 10    | gpt-oss-20b                | P4      | 0.060  | 0.803    | 0.811   | 0.838       | -0.061            | 0.783         | 0.269      | 0.668    |
| 4     | nemotron-3-super-120b-a12b | P1      | 0.060  | 0.879    | 0.864   | 0.878       | -0.008            | 0.889         | 0.578      | 0.709    |
| 5     | nemotron-3-super-120b-a12b | P2      | 0.060  | 0.883    | 0.885   | 0.886       | 0.013             | 0.938         | 0.643      | 0.730    |
| 6     | nemotron-3-super-120b-a12b | P4      | 0.060  | 0.882    | 0.875   | 0.879       | 0.003             | 0.943         | 0.484      | 0.701    |
| 11    | qwen3.5-397b-a17b          | P1      | 0.060  | 0.881    | 0.879   | 0.878       | 0.007             | 0.913         | 0.461      | 0.771    |
| 12    | qwen3.5-397b-a17b          | P2      | 0.060  | 0.881    | 0.879   | 0.872       | 0.007             | 0.913         | 0.830      | 0.767    |
| 13    | qwen3.5-397b-a17b          | P2lp    | 0.010  | 0.886    | 0.873   | 0.872       | 0.001             | 0.873         | 0.597      | 0.749    |
| 14    | qwen3.5-397b-a17b          | P4      | 0.060  | 0.880    | 0.868   | 0.874       | -0.004            | 0.908         | 0.264      | 0.701    |

## Best variant per model (selected on **train** F2, never on test)

| index | model                      | variant | t_star | f2_train | f2_test | f2_validate | lift_over_all_pos | f2own_test | auc_test |
|-------|----------------------------|---------|--------|----------|---------|-------------|-------------------|------------|----------|
| 3     | gemma-4-31b-it             | P4      | 0.010  | 0.886    | 0.885   | 0.875       | 0.013             | 0.442      | 0.691    |
| 5     | nemotron-3-super-120b-a12b | P2      | 0.060  | 0.883    | 0.885   | 0.886       | 0.013             | 0.643      | 0.730    |
| 13    | qwen3.5-397b-a17b          | P2lp    | 0.010  | 0.886    | 0.873   | 0.872       | 0.001             | 0.597      | 0.749    |
| 8     | gpt-oss-20b                | P2      | 0.060  | 0.865    | 0.850   | 0.874       | -0.022            | 0.565      | 0.745    |

### Read the `lift_over_all_pos` column first

At the pooled test prevalence of **0.577**, marking *every* paper relevant scores **F2 = 0.872**. That is the floor these numbers sit on. The ensemble's 0.893 is **+0.021** over it and the best cell here is **+0.013**. Pooled F2 at this prevalence barely discriminates - `pos_rate_at_t` shows the tuned threshold is driving the models toward saying yes to nearly everything, because at 58% positive that is very nearly the right answer. This is exactly the ~20x-production-prevalence distortion CONTEXT.md §3 warns about, made visible. Rank metrics (`auc_test`) and the per-use-case numbers in `wf_llm_pilot_findings.md` carry the real signal.

Mean train→test drop from re-using a fitted threshold: **+0.007** (max +0.015). That is the size of the optimism in any F2 quoted at its own optimal threshold.

Ensemble reference: within-silo **F2@t\* 0.893**, itself an oracle number (threshold swept on the rows it scores), so it carries the same optimism as the `f2_train` column, not the `f2_test` one.

