# Does the induced brief help features that do not read?

Rule set `setA-rules-v1`, induced from 30+30 train rows per collection. Scored on **test + validate only**, weighted to population prevalence — the same rows as `wf_llm_benchset_a.md` §4, so these numbers sit beside the LLM ladder.

Rebuild check: rebuilt `bm25_must` vs the shipped `lex_bm25_must` correlates **0.9993** (min 0.9961) across the 8 collections — the residual is the pool-IDF difference, and both arms below are built identically.

## 1. BM25 and term overlap — single features, no labels, no fitting

This is the cold-start use of the lexical block: one number per paper, straight from the brief. ROC-AUC on held-out rows.

**`bm25_obj`** — mean 0.651 → **0.698** (+0.048), improves on 3/8, degrades on 2

| use_case          | supplied | induced | delta  |
|-------------------|----------|---------|--------|
| brouwer_2019      | 0.752    | 0.951   | 0.199  |
| leenaars_2020     | 0.554    | 0.570   | 0.016  |
| moran_2021        | 0.591    | 0.607   | 0.016  |
| muthu_2021        | 0.640    | 0.604   | -0.036 |
| nelson_2002       | 0.521    | 0.676   | 0.155  |
| sep_2021          | 0.505    | 0.580   | 0.075  |
| van_der_valk_2021 | 0.782    | 0.730   | -0.052 |
| van_dis_2020      | 0.861    | 0.869   | 0.008  |

**`bm25_must`** — mean 0.692 → **0.727** (+0.035), improves on 3/8, degrades on 1

| use_case          | supplied | induced | delta  |
|-------------------|----------|---------|--------|
| brouwer_2019      | 0.933    | 0.942   | 0.009  |
| leenaars_2020     | 0.661    | 0.843   | 0.182  |
| moran_2021        | 0.509    | 0.497   | -0.011 |
| muthu_2021        | 0.726    | 0.736   | 0.010  |
| nelson_2002       | 0.583    | 0.638   | 0.054  |
| sep_2021          | 0.563    | 0.526   | -0.037 |
| van_der_valk_2021 | 0.695    | 0.763   | 0.068  |
| van_dis_2020      | 0.865    | 0.871   | 0.006  |

**`bm25_nice`** — mean 0.605 → **0.696** (+0.091), improves on 6/8, degrades on 0

| use_case          | supplied | induced | delta |
|-------------------|----------|---------|-------|
| brouwer_2019      | 0.735    | 0.833   | 0.098 |
| leenaars_2020     | 0.625    | 0.756   | 0.131 |
| moran_2021        | 0.478    | 0.502   | 0.025 |
| muthu_2021        | 0.562    | 0.619   | 0.057 |
| nelson_2002       | 0.542    | 0.753   | 0.211 |
| sep_2021          | 0.496    | 0.539   | 0.043 |
| van_der_valk_2021 | 0.693    | 0.698   | 0.005 |
| van_dis_2020      | 0.712    | 0.868   | 0.155 |

**`overlap_must_frac`** — mean 0.718 → **0.754** (+0.036), improves on 5/8, degrades on 1

| use_case          | supplied | induced | delta  |
|-------------------|----------|---------|--------|
| brouwer_2019      | 0.875    | 0.914   | 0.039  |
| leenaars_2020     | 0.748    | 0.840   | 0.092  |
| moran_2021        | 0.487    | 0.516   | 0.029  |
| muthu_2021        | 0.605    | 0.728   | 0.123  |
| nelson_2002       | 0.624    | 0.659   | 0.035  |
| sep_2021          | 0.779    | 0.722   | -0.057 |
| van_der_valk_2021 | 0.788    | 0.775   | -0.012 |
| van_dis_2020      | 0.836    | 0.875   | 0.040  |

**`rank_overlap_must`** — mean 0.718 → **0.754** (+0.036), improves on 5/8, degrades on 1

| use_case          | supplied | induced | delta  |
|-------------------|----------|---------|--------|
| brouwer_2019      | 0.875    | 0.914   | 0.039  |
| leenaars_2020     | 0.748    | 0.840   | 0.092  |
| moran_2021        | 0.487    | 0.516   | 0.029  |
| muthu_2021        | 0.605    | 0.728   | 0.123  |
| nelson_2002       | 0.624    | 0.659   | 0.035  |
| sep_2021          | 0.779    | 0.722   | -0.057 |
| van_der_valk_2021 | 0.788    | 0.775   | -0.012 |
| van_dis_2020      | 0.836    | 0.875   | 0.040  |

## 2. The whole lexical block, fitted on the same 60 labels

LogReg over all 21 lexical features, trained on the identical 30+30 rows the rule induction saw. Tests whether a better brief still pays once the block can be fitted.

| use_case          | auc|supplied | auc|induced | wss|supplied | wss|induced | f2own|supplied | f2own|induced | d_auc |
|-------------------|--------------|-------------|--------------|-------------|----------------|---------------|-------|
| brouwer_2019      | 0.941        | 0.964       | 0.732        | 0.780       | 0.048          | 0.065         | 0.023 |
| leenaars_2020     | 0.816        | 0.862       | 0.391        | 0.455       | 0.476          | 0.552         | 0.046 |
| moran_2021        | 0.525        | 0.525       | 0.046        | 0.062       | 0.096          | 0.089         | 0.000 |
| muthu_2021        | 0.712        | 0.762       | 0.150        | 0.153       | 0.485          | 0.500         | 0.050 |
| nelson_2002       | 0.583        | 0.760       | 0.013        | 0.174       | 0.446          | 0.608         | 0.177 |
| sep_2021          | 0.592        | 0.750       | 0.033        | 0.033       | 0.354          | 0.529         | 0.158 |
| van_der_valk_2021 | 0.787        | 0.828       | 0.365        | 0.380       | 0.474          | 0.565         | 0.041 |
| van_dis_2020      | 0.911        | 0.934       | 0.760        | 0.764       | 0.111          | 0.148         | 0.023 |

Mean AUC **0.733 → 0.798** (+0.065); improves on 5/8, degrades on 0.

