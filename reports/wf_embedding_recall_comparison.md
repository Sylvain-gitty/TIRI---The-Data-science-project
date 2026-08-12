# Embedding comparison through the recall lens

> **Status: current.** One of the live evidence docs listed in `CONTEXT.md` §7 — embedding choice judged through the recall lens (WSS@95) on both fold surfaces.

1852 labelled papers, 6 use cases. Primary metric **WSS@95**: the share of the pool a reviewer can skip while still finding 95% of the relevant papers. Its ceiling is `0.95 x (1 - prevalence)`, shown per use case, because these pools are 26-77% positive and a WSS of 0.30 means very different things at each end.

| use_case_key     | prevalence | wss_ceiling |
|------------------|------------|-------------|
| carbon_capture   | 0.495      | 0.480       |
| cement_binders   | 0.649      | 0.334       |
| ner              | 0.715      | 0.271       |
| soil_microbiome  | 0.263      | 0.700       |
| solar_leo        | 0.767      | 0.222       |
| tech_forecasting | 0.595      | 0.385       |

## 1. Within-silo — the production surface

One customer, one model, its own labels, author-grouped 5-fold, 5 seeds.

**wss_at_95**

| variant                              | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|--------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| qwen3-8b: PCA-64 + lexical           | 0.250          | 0.201          | 0.122 | 0.165           | 0.048     | 0.166            | 0.159 |
| qwen3-4b: embedding                  | 0.245          | 0.191          | 0.081 | 0.225           | 0.059     | 0.114            | 0.153 |
| qwen3-4b: PCA-64 + lexical           | 0.221          | 0.206          | 0.071 | 0.213           | 0.062     | 0.137            | 0.152 |
| qwen3-8b: embedding                  | 0.276          | 0.202          | 0.103 | 0.124           | 0.052     | 0.152            | 0.151 |
| jasper: embedding                    | 0.253          | 0.192          | 0.084 | 0.156           | 0.050     | 0.144            | 0.147 |
| qwen3-4b: embedding PCA-64           | 0.209          | 0.199          | 0.076 | 0.168           | 0.055     | 0.153            | 0.143 |
| qwen3-8b: embedding PCA-64           | 0.226          | 0.198          | 0.093 | 0.136           | 0.053     | 0.152            | 0.143 |
| jasper: PCA-64 + lexical             | 0.214          | 0.199          | 0.083 | 0.145           | 0.039     | 0.165            | 0.141 |
| jasper: embedding PCA-64             | 0.188          | 0.180          | 0.088 | 0.144           | 0.036     | 0.158            | 0.132 |
| jasper: cosine-to-brief (0 labels)   | 0.125          | 0.206          | 0.085 | -0.008          | 0.006     | 0.079            | 0.082 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.162          | 0.217          | 0.069 | -0.019          | 0.017     | 0.033            | 0.080 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.122          | 0.217          | 0.065 | 0.012           | 0.011     | 0.037            | 0.077 |
| lexical only (Tier 1b)               | 0.071          | 0.203          | 0.056 | 0.037           | 0.008     | 0.035            | 0.069 |

**recall_at_10pct**

| variant                              | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|--------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| qwen3-4b: embedding                  | 0.195          | 0.140          | 0.138 | 0.285           | 0.129     | 0.166            | 0.175 |
| qwen3-4b: PCA-64 + lexical           | 0.197          | 0.136          | 0.135 | 0.283           | 0.126     | 0.162            | 0.173 |
| jasper: PCA-64 + lexical             | 0.197          | 0.140          | 0.132 | 0.281           | 0.125     | 0.153            | 0.171 |
| jasper: embedding PCA-64             | 0.190          | 0.129          | 0.131 | 0.300           | 0.121     | 0.150            | 0.170 |
| qwen3-4b: embedding PCA-64           | 0.195          | 0.139          | 0.127 | 0.279           | 0.118     | 0.162            | 0.170 |
| qwen3-8b: embedding                  | 0.193          | 0.145          | 0.136 | 0.247           | 0.126     | 0.163            | 0.168 |
| jasper: embedding                    | 0.200          | 0.141          | 0.124 | 0.245           | 0.127     | 0.154            | 0.165 |
| qwen3-8b: PCA-64 + lexical           | 0.193          | 0.142          | 0.134 | 0.232           | 0.124     | 0.157            | 0.164 |
| qwen3-8b: embedding PCA-64           | 0.190          | 0.144          | 0.125 | 0.238           | 0.123     | 0.154            | 0.162 |
| jasper: cosine-to-brief (0 labels)   | 0.170          | 0.124          | 0.139 | 0.170           | 0.120     | 0.140            | 0.144 |
| lexical only (Tier 1b)               | 0.169          | 0.149          | 0.112 | 0.181           | 0.117     | 0.127            | 0.143 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.150          | 0.129          | 0.126 | 0.160           | 0.127     | 0.127            | 0.136 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.177          | 0.124          | 0.135 | 0.117           | 0.120     | 0.146            | 0.136 |

**roc_auc**

| variant                              | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|--------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| qwen3-8b: PCA-64 + lexical           | 0.884          | 0.880          | 0.848 | 0.774           | 0.803     | 0.823            | 0.836 |
| qwen3-8b: embedding                  | 0.897          | 0.895          | 0.845 | 0.755           | 0.796     | 0.825            | 0.835 |
| qwen3-4b: embedding                  | 0.887          | 0.869          | 0.840 | 0.793           | 0.799     | 0.821            | 0.835 |
| qwen3-4b: PCA-64 + lexical           | 0.874          | 0.873          | 0.825 | 0.792           | 0.795     | 0.820            | 0.830 |
| qwen3-4b: embedding PCA-64           | 0.868          | 0.879          | 0.813 | 0.793           | 0.766     | 0.830            | 0.825 |
| qwen3-8b: embedding PCA-64           | 0.875          | 0.873          | 0.814 | 0.778           | 0.784     | 0.818            | 0.824 |
| jasper: embedding                    | 0.889          | 0.865          | 0.799 | 0.758           | 0.782     | 0.830            | 0.820 |
| jasper: PCA-64 + lexical             | 0.868          | 0.864          | 0.814 | 0.784           | 0.756     | 0.821            | 0.818 |
| jasper: embedding PCA-64             | 0.859          | 0.833          | 0.805 | 0.794           | 0.749     | 0.835            | 0.812 |
| lexical only (Tier 1b)               | 0.762          | 0.881          | 0.717 | 0.679           | 0.610     | 0.657            | 0.718 |
| jasper: cosine-to-brief (0 labels)   | 0.751          | 0.828          | 0.774 | 0.603           | 0.528     | 0.672            | 0.693 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.798          | 0.834          | 0.751 | 0.520           | 0.589     | 0.629            | 0.687 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.723          | 0.843          | 0.722 | 0.527           | 0.517     | 0.639            | 0.662 |

## 2. Leave-one-use-case-out — chooses central defaults, not a production number

**wss_at_95**

| variant                              | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|--------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| jasper: cosine-to-brief (0 labels)   | 0.125          | 0.206          | 0.085 | -0.008          | 0.006     | 0.079            | 0.082 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.162          | 0.217          | 0.069 | -0.019          | 0.017     | 0.033            | 0.080 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.122          | 0.217          | 0.065 | 0.012           | 0.011     | 0.037            | 0.077 |
| lexical only (Tier 1b)               | 0.068          | 0.198          | 0.037 | 0.014           | -0.006    | -0.008           | 0.051 |
| jasper: embedding PCA-64             | -0.010         | 0.103          | 0.021 | 0.040           | 0.042     | 0.060            | 0.042 |
| qwen3-8b: embedding                  | 0.061          | 0.114          | 0.001 | -0.005          | 0.017     | 0.052            | 0.040 |
| qwen3-4b: PCA-64 + lexical           | 0.038          | 0.072          | 0.017 | 0.037           | 0.014     | 0.037            | 0.036 |
| qwen3-4b: embedding PCA-64           | 0.021          | 0.087          | 0.014 | 0.009           | 0.014     | 0.052            | 0.033 |
| qwen3-8b: embedding PCA-64           | 0.001          | 0.072          | 0.011 | 0.006           | 0.014     | 0.064            | 0.028 |
| jasper: PCA-64 + lexical             | 0.007          | 0.084          | 0.027 | 0.000           | 0.008     | 0.033            | 0.027 |
| qwen3-4b: embedding                  | 0.021          | 0.049          | 0.001 | 0.020           | 0.014     | 0.041            | 0.024 |
| qwen3-8b: PCA-64 + lexical           | 0.011          | 0.011          | 0.011 | 0.006           | 0.008     | 0.048            | 0.016 |
| jasper: embedding                    | 0.017          | 0.057          | 0.008 | -0.030          | 0.006     | 0.037            | 0.016 |

**roc_auc**

| variant                              | carbon_capture | cement_binders | ner   | soil_microbiome | solar_leo | tech_forecasting | MEAN  |
|--------------------------------------|----------------|----------------|-------|-----------------|-----------|------------------|-------|
| jasper: cosine-to-brief (0 labels)   | 0.751          | 0.828          | 0.774 | 0.603           | 0.528     | 0.672            | 0.693 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.798          | 0.834          | 0.751 | 0.520           | 0.589     | 0.629            | 0.687 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.723          | 0.843          | 0.722 | 0.527           | 0.517     | 0.639            | 0.662 |
| lexical only (Tier 1b)               | 0.744          | 0.890          | 0.673 | 0.594           | 0.413     | 0.540            | 0.642 |
| qwen3-4b: PCA-64 + lexical           | 0.616          | 0.689          | 0.573 | 0.568           | 0.581     | 0.631            | 0.610 |
| qwen3-4b: embedding PCA-64           | 0.547          | 0.698          | 0.547 | 0.582           | 0.624     | 0.643            | 0.607 |
| jasper: embedding PCA-64             | 0.420          | 0.667          | 0.503 | 0.619           | 0.650     | 0.666            | 0.588 |
| qwen3-8b: embedding                  | 0.590          | 0.729          | 0.508 | 0.427           | 0.557     | 0.712            | 0.587 |
| jasper: PCA-64 + lexical             | 0.522          | 0.655          | 0.536 | 0.580           | 0.587     | 0.628            | 0.585 |
| qwen3-4b: embedding                  | 0.627          | 0.594          | 0.500 | 0.456           | 0.608     | 0.696            | 0.580 |
| qwen3-8b: PCA-64 + lexical           | 0.565          | 0.532          | 0.565 | 0.509           | 0.566     | 0.661            | 0.566 |
| qwen3-8b: embedding PCA-64           | 0.501          | 0.562          | 0.541 | 0.488           | 0.616     | 0.675            | 0.564 |
| jasper: embedding                    | 0.491          | 0.646          | 0.462 | 0.544           | 0.526     | 0.629            | 0.550 |

## 3. Warm-start — WSS@95 vs. labels from the target use case

| variant                              | 25    | 50    | 100   | 200   |
|--------------------------------------|-------|-------|-------|-------|
| jasper: PCA-64 + lexical             | 0.083 | 0.103 | 0.118 | 0.133 |
| jasper: cosine-to-brief (0 labels)   | 0.082 | 0.082 | 0.082 | 0.082 |
| jasper: embedding                    | 0.102 | 0.121 | 0.135 | 0.147 |
| jasper: embedding PCA-64             | 0.101 | 0.121 | 0.131 | 0.136 |
| lexical only (Tier 1b)               | 0.042 | 0.054 | 0.063 | 0.075 |
| qwen3-4b: PCA-64 + lexical           | 0.083 | 0.109 | 0.124 | 0.149 |
| qwen3-4b: cosine-to-brief (0 labels) | 0.080 | 0.080 | 0.080 | 0.080 |
| qwen3-4b: embedding                  | 0.100 | 0.133 | 0.149 | 0.159 |
| qwen3-4b: embedding PCA-64           | 0.100 | 0.133 | 0.146 | 0.156 |
| qwen3-8b: PCA-64 + lexical           | 0.083 | 0.107 | 0.122 | 0.154 |
| qwen3-8b: cosine-to-brief (0 labels) | 0.077 | 0.077 | 0.077 | 0.077 |
| qwen3-8b: embedding                  | 0.107 | 0.133 | 0.150 | 0.150 |
| qwen3-8b: embedding PCA-64           | 0.107 | 0.133 | 0.139 | 0.142 |

