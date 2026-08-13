# Shuffled-brief control — is the LLM reading the brief?

Each use case's papers scored against a **different** use case's brief (derangement, no fixed point). A brief-conditioned screener should collapse.

## google/gemma-4-31b-it / P2

| use_case         | auc_real | auc_shuffled | auc_drop | f2own_real | f2own_shuffled | f2own_drop |
|------------------|----------|--------------|----------|------------|----------------|------------|
| carbon_capture   | 0.910    | 0.427        | 0.482    | 0.883      | 0.000          | 0.883      |
| cement_binders   | 0.913    | 0.500        | 0.413    | 0.934      | 0.000          | 0.934      |
| ner              | 0.824    | 0.500        | 0.324    | 0.765      | 0.000          | 0.765      |
| soil_microbiome  | 0.689    | 0.500        | 0.189    | 0.645      | 0.000          | 0.645      |
| solar_leo        | 0.739    | 0.500        | 0.239    | 0.932      | 0.000          | 0.932      |
| tech_forecasting | 0.840    | 0.500        | 0.340    | 0.726      | 0.000          | 0.726      |

Mean AUC 0.819 → **0.488** (drop **+0.331**), degrades on **6/6** use cases. Mean F2@own 0.814 → **0.000** (drop +0.814).

**Verdict: PASSES — collapses toward chance without its own brief.**

## qwen/qwen3.5-397b-a17b / P2

| use_case         | auc_real | auc_shuffled | auc_drop | f2own_real | f2own_shuffled | f2own_drop |
|------------------|----------|--------------|----------|------------|----------------|------------|
| carbon_capture   | 0.922    | 0.482        | 0.440    | 0.893      | 0.000          | 0.893      |
| cement_binders   | 0.925    | 0.500        | 0.425    | 0.947      | 0.000          | 0.947      |
| ner              | 0.802    | 0.502        | 0.300    | 0.736      | 0.000          | 0.736      |
| soil_microbiome  | 0.717    | 0.500        | 0.217    | 0.624      | 0.000          | 0.624      |
| solar_leo        | 0.724    | 0.500        | 0.224    | 0.909      | 0.000          | 0.909      |
| tech_forecasting | 0.836    | 0.489        | 0.347    | 0.763      | 0.008          | 0.755      |

Mean AUC 0.821 → **0.496** (drop **+0.326**), degrades on **6/6** use cases. Mean F2@own 0.812 → **0.001** (drop +0.811).

**Verdict: PASSES — collapses toward chance without its own brief.**

## Summary

| index | model                  | variant | auc_real | auc_shuffled | auc_drop | degraded |
|-------|------------------------|---------|----------|--------------|----------|----------|
| 0     | google/gemma-4-31b-it  | P2      | 0.819    | 0.488        | 0.331    | 6/6      |
| 1     | qwen/qwen3.5-397b-a17b | P2      | 0.821    | 0.496        | 0.326    | 6/6      |
