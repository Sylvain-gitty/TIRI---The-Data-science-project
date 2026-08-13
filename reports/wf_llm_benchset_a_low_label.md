# Set A — what 60 labels per collection buy, spent three ways

The induced rule set (B2) saw **30 positive + 30 negative train rows per collection**. `CONTEXT.md` §1's rule — *cosine-to-brief below ~25 in-silo labels, supervised in-silo model above* — puts that above the line, so the cold-start cosine is **not** B2's fair opponent. This is the fair one: a supervised model on exactly the same 60 rows, scored on exactly the same held-out rows.

| use_case          | n_train | n_eval | auc_cosine | auc_qc_block | auc_embedding | wss_cosine | wss_qc_block | wss_embedding |
|-------------------|---------|--------|------------|--------------|---------------|------------|--------------|---------------|
| brouwer_2019      | 60      | 625    | 0.998      | 0.996        | 0.998         | 0.942      | 0.937        | 0.935         |
| leenaars_2020     | 60      | 830    | 0.870      | 0.892        | 0.894         | 0.486      | 0.539        | 0.533         |
| moran_2021        | 60      | 644    | 0.488      | 0.573        | 0.639         | -0.018     | 0.010        | 0.050         |
| muthu_2021        | 60      | 734    | 0.728      | 0.759        | 0.762         | 0.252      | 0.230        | 0.179         |
| nelson_2002       | 60      | 143    | 0.713      | 0.687        | 0.841         | 0.258      | 0.216        | 0.433         |
| sep_2021          | 54      | 108    | 0.618      | 0.609        | 0.806         | 0.015      | 0.024        | 0.098         |
| van_der_valk_2021 | 60      | 284    | 0.881      | 0.877        | 0.875         | 0.432      | 0.503        | 0.394         |
| van_dis_2020      | 60      | 629    | 0.897      | 0.912        | 0.941         | 0.650      | 0.724        | 0.714         |

| use_case          | floor_f2 | f2own_qc_block | recall_qc_block | screened_qc_block | f2own_embedding | recall_embedding | screened_embedding |
|-------------------|----------|----------------|-----------------|-------------------|-----------------|------------------|--------------------|
| brouwer_2019      | 0.008    | 0.143          | 1.000           | 0.052             | 0.132           | 1.000            | 0.057              |
| leenaars_2020     | 0.319    | 0.583          | 0.878           | 0.302             | 0.573           | 0.857            | 0.297              |
| moran_2021        | 0.098    | 0.122          | 0.523           | 0.373             | 0.114           | 0.477            | 0.361              |
| muthu_2021        | 0.416    | 0.536          | 0.821           | 0.455             | 0.504           | 0.694            | 0.359              |
| nelson_2002       | 0.590    | 0.588          | 0.688           | 0.413             | 0.656           | 0.750            | 0.385              |
| sep_2021          | 0.465    | 0.392          | 0.500           | 0.352             | 0.577           | 0.750            | 0.370              |
| van_der_valk_2021 | 0.413    | 0.655          | 0.943           | 0.394             | 0.612           | 0.829            | 0.342              |
| van_dis_2020      | 0.039    | 0.124          | 0.966           | 0.284             | 0.136           | 0.966            | 0.256              |

**Means** — AUC: cosine (no labels) **0.774** · LogReg on the query-conditioned block **0.788** · LogReg on the Qwen3-4B embedding **0.844**. F2@own: qc **0.393** · embedding **0.413**.

Read alongside `reports/wf_llm_benchset_a.md` §4, which reports the induced-brief arm on the same held-out rows. Whichever wins, the comparison is now the right one: three ways to spend the same 60 labels, not a fitted method against an unfitted one.

