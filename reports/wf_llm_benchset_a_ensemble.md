# Does the induced brief survive into the per-silo ensemble?

Per-silo CatBoost + LogisticRegression on the 9,993-row case-control sample, grouped by `first_author`, 5-fold, **3 seeds**, scored at population prevalence. `induced_lex` swaps only the BM25/overlap block; `induced_all` also swaps cosine-to-brief. All three variants are 4,625 columns wide, so nothing is confounded with feature count.

The question is not whether the lexical block improved — `wf_llm_benchset_a_induced_features.md` settled that at **+0.065** in isolation — but whether that improvement is **redundant with the 4,608 embedding dimensions the ensemble already sees**.

## 1. Headline — mean over 8 silos and seeds

| arm      | variant     | auc   | auc_sd | wss95 | f2_star |
|----------|-------------|-------|--------|-------|---------|
| blend    | supplied    | 0.884 | 0.072  | 0.488 | 0.574   |
| blend    | induced_lex | 0.885 | 0.072  | 0.489 | 0.581   |
| blend    | induced_all | 0.885 | 0.072  | 0.486 | 0.578   |
| catboost | supplied    | 0.878 | 0.074  | 0.483 | 0.559   |
| catboost | induced_lex | 0.880 | 0.074  | 0.479 | 0.566   |
| catboost | induced_all | 0.879 | 0.074  | 0.471 | 0.565   |
| logreg   | supplied    | 0.849 | 0.109  | 0.433 | 0.531   |
| logreg   | induced_lex | 0.849 | 0.108  | 0.435 | 0.534   |
| logreg   | induced_all | 0.850 | 0.108  | 0.439 | 0.533   |

## 2. Paired per-silo change, `supplied` → each induced variant

Same rows, same folds, same seeds — so these differences are paired and far tighter than the 0.03 noise floor that governs unpaired comparisons. `n_better`/`n_worse` count silos whose mean moved by more than that floor.

**`supplied` → `induced_lex`**, ROC-AUC change per silo

| use_case          | blend  | catboost | logreg |
|-------------------|--------|----------|--------|
| brouwer_2019      | -0.000 | 0.000    | 0.000  |
| leenaars_2020     | 0.001  | 0.001    | 0.001  |
| moran_2021        | -0.005 | -0.006   | 0.002  |
| muthu_2021        | -0.001 | -0.000   | -0.001 |
| nelson_2002       | -0.003 | -0.000   | -0.001 |
| sep_2021          | 0.006  | 0.003    | 0.004  |
| van_der_valk_2021 | 0.004  | 0.006    | -0.000 |
| van_dis_2020      | 0.004  | 0.009    | -0.002 |

| arm      | mean_delta | n_better | n_worse | n_unchanged |
|----------|------------|----------|---------|-------------|
| blend    | 0.001      | 0        | 0       | 8           |
| catboost | 0.002      | 0        | 0       | 8           |
| logreg   | 0.000      | 0        | 0       | 8           |

**`supplied` → `induced_all`**, ROC-AUC change per silo

| use_case          | blend  | catboost | logreg |
|-------------------|--------|----------|--------|
| brouwer_2019      | 0.000  | -0.001   | -0.000 |
| leenaars_2020     | 0.002  | 0.002    | 0.001  |
| moran_2021        | -0.004 | -0.006   | 0.002  |
| muthu_2021        | 0.002  | 0.003    | -0.001 |
| nelson_2002       | 0.000  | 0.003    | -0.001 |
| sep_2021          | 0.006  | 0.001    | 0.005  |
| van_der_valk_2021 | -0.002 | -0.000   | -0.000 |
| van_dis_2020      | 0.005  | 0.008    | 0.000  |

| arm      | mean_delta | n_better | n_worse | n_unchanged |
|----------|------------|----------|---------|-------------|
| blend    | 0.001      | 0        | 0       | 8           |
| catboost | 0.001      | 0        | 0       | 8           |
| logreg   | 0.001      | 0        | 0       | 8           |

## 3. The blend, per silo

| use_case          | auc_supplied | auc_induced_lex | auc_induced_all | seed_sd_supplied | d_induced_lex | f2star_supplied | f2star_induced_lex | f2star_induced_all | floor_f2 |
|-------------------|--------------|-----------------|-----------------|------------------|---------------|-----------------|--------------------|--------------------|----------|
| brouwer_2019      | 0.996        | 0.996           | 0.996           | 0.001            | -0.000        | 0.572           | 0.603              | 0.593              | 0.008    |
| leenaars_2020     | 0.941        | 0.942           | 0.943           | 0.002            | 0.001         | 0.696           | 0.699              | 0.699              | 0.319    |
| moran_2021        | 0.833        | 0.828           | 0.828           | 0.010            | -0.005        | 0.343           | 0.343              | 0.343              | 0.099    |
| muthu_2021        | 0.841        | 0.840           | 0.843           | 0.005            | -0.001        | 0.598           | 0.590              | 0.597              | 0.415    |
| nelson_2002       | 0.886        | 0.883           | 0.886           | 0.012            | -0.003        | 0.769           | 0.765              | 0.755              | 0.590    |
| sep_2021          | 0.761        | 0.767           | 0.767           | 0.020            | 0.006         | 0.570           | 0.586              | 0.579              | 0.465    |
| van_der_valk_2021 | 0.869        | 0.873           | 0.867           | 0.010            | 0.004         | 0.643           | 0.650              | 0.639              | 0.411    |
| van_dis_2020      | 0.945        | 0.949           | 0.950           | 0.004            | 0.004         | 0.402           | 0.410              | 0.417              | 0.039    |

Mean seed-to-seed sd on the blend: **0.007** (repo's measured floor ~0.010). The paired per-silo deltas in §2 are the instrument that matters — they share folds and seeds, so they resolve changes far below this spread.

## 4. Against the isolated-block result

| | ROC-AUC gain from the induced brief |
|---|---|
| BM25 + overlap block alone, fitted on 60 labels | **+0.065** |
| LLM reader, same model and prompt | **+0.057** |
| **Full ensemble (blend), same brief swap** | **+0.001** |

## 5. The label ladder, all arms on the same held-out rows

The ensemble scored on **test + validate only** — the rows every other set-A arm reports on. Training budget is what the ladder varies; the evaluation rows are now identical throughout.

| labels per silo | method | mean ROC-AUC |
|---|---|---|
| 0 | cosine-to-brief (`qwen4b`) | 0.774 |
| 0 | LLM reader, supplied brief | 0.777 |
| 60 | BM25 + overlap block, supplied brief | 0.733 |
| 60 | BM25 + overlap block, **induced** brief | 0.798 |
| 60 | LLM reader, **induced** brief | 0.834 |
| 60 | LogReg on the Qwen3-4B embedding | 0.844 |
| full in-silo (~80% of each silo) | **ensemble blend, supplied brief** | **0.884** |
| full in-silo | ensemble blend, **induced** brief | **0.885** |

**The brief is a cold-start lever and it decays to nothing once the silo has enough labels to train on.** That is `CONTEXT.md` §1's ladder, now with numbers on every rung.

