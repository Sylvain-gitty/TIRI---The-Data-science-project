# Token-logprob scoring vs a verbalised 0-100 score

Ranking signal for P2lp is `logprob(YES) - logprob(NO)`, not the renormalised probability, which saturates at temperature 0. Both variants use the same F2-asymmetry framing; only the elicitation differs.

## Per model

| index | model             | variant | mean_auc | n_distinct | tie_frac | f2_own | coverage | same_provider |
|-------|-------------------|---------|----------|------------|----------|--------|----------|---------------|
| 0     | gpt-oss-20b       | P2      | 0.776    | 12.333     | 0.996    | 0.552  | 1.000    | True          |
| 1     | gpt-oss-20b       | P2lp    | 0.685    | 96.167     | 0.834    | 0.540  | 1.000    | True          |
| 2     | gemma-4-31b-it    | P2      | 0.819    | 13.833     | 0.997    | 0.830  | 1.000    | False         |
| 3     | gemma-4-31b-it    | P2lp    | 0.796    | 300.500    | 0.047    | 0.779  | 1.000    | False         |
| 4     | qwen3.5-397b-a17b | P2      | 0.821    | 9.000      | 0.996    | 0.825  | 1.000    | False         |
| 5     | qwen3.5-397b-a17b | P2lp    | 0.829    | 246.833    | 0.286    | 0.593  | 1.000    | False         |

## The comparison

| model             | auc_verbalised | auc_logprob | auc_gain | distinct_verbalised | distinct_logprob | ties_verbalised | ties_logprob |
|-------------------|----------------|-------------|----------|---------------------|------------------|-----------------|--------------|
| gemma-4-31b-it    | 0.819          | 0.796       | -0.023   | 13.833              | 300.500          | 0.997           | 0.047        |
| gpt-oss-20b       | 0.776          | 0.685       | -0.091   | 12.333              | 96.167           | 0.996           | 0.834        |
| qwen3.5-397b-a17b | 0.821          | 0.829       | 0.008    | 9.000               | 246.833          | 0.996           | 0.286        |

Mean AUC change **-0.035**, positive on **1 of 3** models. Best logprob AUC **0.829** vs the ensemble's **0.865** (gap -0.036).

On the one within-provider pair (`gpt-oss-20b`, CoreWeave both sides) the change is **-0.091** — the control for the provider switch the other two models required.

