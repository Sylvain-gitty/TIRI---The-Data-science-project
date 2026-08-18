# Does a prompted-LLM branch earn a place in the ensemble? (variant P2)

Screening run: seed 0 only, CatBoost OOF read from disk, weights chosen leave-one-use-case-out. See this script's docstring for why the weight protocol differs from the SVM/lexical/k-NN experiments (an LLM branch has no seed, so a leave-seeds-out split cannot catch weight overfitting to these rows).

## google/gemma-4-31b-it

### Branch disagreement (Spearman on OOF scores, within silo)

| use_case                   | rho_llm_cb | rho_llm_lr | rho_cb_lr |
|----------------------------|------------|------------|-----------|
| Carbon Capture             | 0.699      | 0.717      | 0.904     |
| Low-Carbon Cement          | 0.800      | 0.726      | 0.874     |
| Named Entity Recognition   | 0.651      | 0.641      | 0.818     |
| Soil Microbiome            | 0.404      | 0.362      | 0.775     |
| Solar Cells for Satellites | 0.491      | 0.358      | 0.728     |
| Technology Prediction      | 0.587      | 0.575      | 0.838     |

Mean: LLM-vs-CatBoost **0.605**, LLM-vs-LogReg **0.563**, and the existing production pair's own CatBoost-vs-LogReg **0.823** on these same rows.

### Nested 3-way blend vs the shipped 2-way (weights chosen on the other five use cases)

| use_case                   | w_cb  | w_lr  | w_llm | auc_2way | auc_3way | auc_gain | f2_2way | f2_3way | f2_gain |
|----------------------------|-------|-------|-------|----------|----------|----------|---------|---------|---------|
| Carbon Capture             | 0.600 | 0.100 | 0.300 | 0.888    | 0.922    | 0.034    | 0.890   | 0.912   | 0.022   |
| Low-Carbon Cement          | 0.500 | 0.100 | 0.400 | 0.919    | 0.925    | 0.005    | 0.951   | 0.952   | 0.001   |
| Named Entity Recognition   | 0.500 | 0.100 | 0.400 | 0.833    | 0.867    | 0.034    | 0.929   | 0.930   | 0.001   |
| Soil Microbiome            | 0.500 | 0.100 | 0.400 | 0.838    | 0.835    | -0.004   | 0.736   | 0.721   | -0.016  |
| Solar Cells for Satellites | 0.500 | 0.100 | 0.400 | 0.872    | 0.879    | 0.007    | 0.953   | 0.955   | 0.003   |
| Technology Prediction      | 0.600 | 0.100 | 0.300 | 0.834    | 0.858    | 0.024    | 0.899   | 0.913   | 0.014   |

Mean AUC gain **+0.017**, mean F2@t* gain **+0.004** (noise floor 0.03). Positive on 5/6 use cases by AUC, 5/6 by F2.

**Verdict: does NOT clear the noise floor.**

## qwen/qwen3.5-397b-a17b

### Branch disagreement (Spearman on OOF scores, within silo)

| use_case                   | rho_llm_cb | rho_llm_lr | rho_cb_lr |
|----------------------------|------------|------------|-----------|
| Carbon Capture             | 0.768      | 0.762      | 0.904     |
| Low-Carbon Cement          | 0.833      | 0.747      | 0.874     |
| Named Entity Recognition   | 0.619      | 0.610      | 0.818     |
| Soil Microbiome            | 0.423      | 0.384      | 0.775     |
| Solar Cells for Satellites | 0.509      | 0.383      | 0.728     |
| Technology Prediction      | 0.613      | 0.598      | 0.838     |

Mean: LLM-vs-CatBoost **0.628**, LLM-vs-LogReg **0.581**, and the existing production pair's own CatBoost-vs-LogReg **0.823** on these same rows.

### Nested 3-way blend vs the shipped 2-way (weights chosen on the other five use cases)

| use_case                   | w_cb  | w_lr  | w_llm | auc_2way | auc_3way | auc_gain | f2_2way | f2_3way | f2_gain |
|----------------------------|-------|-------|-------|----------|----------|----------|---------|---------|---------|
| Carbon Capture             | 0.500 | 0.100 | 0.400 | 0.888    | 0.930    | 0.041    | 0.890   | 0.918   | 0.028   |
| Low-Carbon Cement          | 0.500 | 0.100 | 0.400 | 0.919    | 0.934    | 0.015    | 0.951   | 0.951   | 0.000   |
| Named Entity Recognition   | 0.500 | 0.100 | 0.400 | 0.833    | 0.865    | 0.032    | 0.929   | 0.928   | -0.000  |
| Soil Microbiome            | 0.400 | 0.100 | 0.500 | 0.838    | 0.828    | -0.010   | 0.736   | 0.725   | -0.011  |
| Solar Cells for Satellites | 0.400 | 0.100 | 0.500 | 0.872    | 0.874    | 0.003    | 0.953   | 0.952   | -0.001  |
| Technology Prediction      | 0.500 | 0.100 | 0.400 | 0.834    | 0.867    | 0.034    | 0.899   | 0.913   | 0.014   |

Mean AUC gain **+0.019**, mean F2@t* gain **+0.005** (noise floor 0.03). Positive on 5/6 use cases by AUC, 2/6 by F2.

**Verdict: does NOT clear the noise floor.**

## openai/gpt-oss-20b

### Branch disagreement (Spearman on OOF scores, within silo)

| use_case                   | rho_llm_cb | rho_llm_lr | rho_cb_lr |
|----------------------------|------------|------------|-----------|
| Carbon Capture             | 0.689      | 0.681      | 0.904     |
| Low-Carbon Cement          | 0.723      | 0.659      | 0.874     |
| Named Entity Recognition   | 0.594      | 0.565      | 0.818     |
| Soil Microbiome            | 0.253      | 0.243      | 0.775     |
| Solar Cells for Satellites | 0.298      | 0.234      | 0.728     |
| Technology Prediction      | 0.512      | 0.493      | 0.838     |

Mean: LLM-vs-CatBoost **0.512**, LLM-vs-LogReg **0.479**, and the existing production pair's own CatBoost-vs-LogReg **0.823** on these same rows.

### Nested 3-way blend vs the shipped 2-way (weights chosen on the other five use cases)

| use_case                   | w_cb  | w_lr  | w_llm | auc_2way | auc_3way | auc_gain | f2_2way | f2_3way | f2_gain |
|----------------------------|-------|-------|-------|----------|----------|----------|---------|---------|---------|
| Carbon Capture             | 0.700 | 0.100 | 0.200 | 0.888    | 0.915    | 0.027    | 0.890   | 0.903   | 0.013   |
| Low-Carbon Cement          | 0.700 | 0.100 | 0.200 | 0.919    | 0.935    | 0.016    | 0.951   | 0.952   | 0.001   |
| Named Entity Recognition   | 0.700 | 0.100 | 0.200 | 0.833    | 0.858    | 0.025    | 0.929   | 0.928   | -0.000  |
| Soil Microbiome            | 0.600 | 0.100 | 0.300 | 0.838    | 0.837    | -0.001   | 0.736   | 0.750   | 0.014   |
| Solar Cells for Satellites | 0.600 | 0.100 | 0.300 | 0.872    | 0.858    | -0.014   | 0.953   | 0.956   | 0.004   |
| Technology Prediction      | 0.700 | 0.100 | 0.200 | 0.834    | 0.840    | 0.006    | 0.899   | 0.902   | 0.003   |

Mean AUC gain **+0.010**, mean F2@t* gain **+0.006** (noise floor 0.03). Positive on 4/6 use cases by AUC, 5/6 by F2.

**Verdict: does NOT clear the noise floor.**

