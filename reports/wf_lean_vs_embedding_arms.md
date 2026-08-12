# Does the ensemble need the raw embeddings?

> **Status: current, supporting.** A v2 experiment — does the ensemble actually need the raw embedding block? Feeds the `cement_binders` exception in [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md).

Within-silo, author-grouped 5-fold, 5 seeds, plain LogisticRegression. Embedding arm uses qwen3-8b alone (4096 cols); the lean arm is 33 cols.

## 1. Standing alone, and combined

**wss_at_95**

| arm                                                         | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|-------------------------------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| ALL (embedding + lean)                                      | 0.281          | 0.209          | 0.101 | 0.117           | 0.058     | 0.150            | 0.153 |
| EMBEDDING only (qwen3-8b)                                   | 0.276          | 0.202          | 0.103 | 0.124           | 0.052     | 0.152            | 0.151 |
| mean of arms                                                | 0.248          | 0.227          | 0.108 | 0.059           | 0.073     | 0.120            | 0.139 |
| max of arms (union-flavoured)                               | 0.223          | 0.228          | 0.110 | 0.059           | 0.067     | 0.097            | 0.130 |
| LEAN (cos_brief + rank + lexical + metadata, no embeddings) | 0.163          | 0.223          | 0.081 | 0.028           | 0.067     | 0.047            | 0.101 |

**recall_at_10pct**

| arm                                                         | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|-------------------------------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| EMBEDDING only (qwen3-8b)                                   | 0.193          | 0.145          | 0.136 | 0.247           | 0.126     | 0.163            | 0.168 |
| ALL (embedding + lean)                                      | 0.192          | 0.146          | 0.136 | 0.247           | 0.126     | 0.163            | 0.168 |
| max of arms (union-flavoured)                               | 0.193          | 0.145          | 0.136 | 0.247           | 0.125     | 0.163            | 0.168 |
| mean of arms                                                | 0.193          | 0.140          | 0.128 | 0.238           | 0.128     | 0.154            | 0.164 |
| LEAN (cos_brief + rank + lexical + metadata, no embeddings) | 0.186          | 0.141          | 0.128 | 0.177           | 0.123     | 0.140            | 0.149 |

**roc_auc**

| arm                                                         | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|-------------------------------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| mean of arms                                                | 0.893          | 0.914          | 0.840 | 0.755           | 0.842     | 0.792            | 0.839 |
| ALL (embedding + lean)                                      | 0.899          | 0.896          | 0.845 | 0.756           | 0.804     | 0.825            | 0.838 |
| EMBEDDING only (qwen3-8b)                                   | 0.897          | 0.895          | 0.845 | 0.755           | 0.796     | 0.825            | 0.835 |
| max of arms (union-flavoured)                               | 0.888          | 0.904          | 0.847 | 0.748           | 0.804     | 0.804            | 0.832 |
| LEAN (cos_brief + rank + lexical + metadata, no embeddings) | 0.808          | 0.916          | 0.779 | 0.668           | 0.805     | 0.652            | 0.771 |

## 2. Do the two arms make the same mistakes?

| use_case         | spearman_lean_vs_emb |
|------------------|----------------------|
| carbon_capture   | 0.635                |
| cement_binders   | 0.783                |
| ner              | 0.589                |
| soil_microbiome  | 0.373                |
| solar_leo        | 0.475                |
| tech_forecasting | 0.355                |

Mean Spearman correlation between the lean and embedding arms' out-of-fold rankings: **0.535**. Near 1.0 would mean the lean arm is redundant; a moderate value means it ranks different papers highly and can contribute to an ensemble even while scoring lower alone.

