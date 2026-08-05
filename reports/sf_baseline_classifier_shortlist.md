# Baseline classifier shortlist — decision trail

Standalone write-up of `notebooks/eda/sf_eda_firstrun.ipynb` §13–14's baseline-model
comparison, kept here as a decision-trail document per this repo's `reports/` convention
(see `HANDOFF.md`). The notebook is the source of truth — every number below is copied
from its own executed output, not recomputed separately here — this file exists so the
recommendation survives as a quick reference without re-opening the notebook.

**Scope note:** this is a **baseline-model shortlist**, not a model-selection pipeline —
same distinction this repo already draws for `scripts/compare_embeddings.py` et al.
(`README.md`'s scope note, `HANDOFF.md`). Nothing here is wired to any script; whoever
starts the actual Week-3 modelling work should re-run the comparison in
`sf_eda_firstrun.ipynb` §13 once the additional features flagged in §11 (term overlap,
cleaned `venue`, `author_count`) are actually built, rather than trusting these numbers
unchanged.

## 1. What was tested

Three real candidates plus a `DummyClassifier` floor, cross-validated with
`StratifiedGroupKFold` (5 folds, grouped by first-author, per §12's leakage finding),
feature set = the 384-dim `embedding` + engineered `citation_velocity` only (deliberately
small, so the comparison is easy to audit end-to-end). Both target framings from §2.3 are
reported — nothing here silently picks one:

| Model | Binary ROC-AUC (`positive` vs `negative`) | 3-class macro OVR ROC-AUC (`positive`/`negative`/`pass`) |
|---|---|---|
| `DummyClassifier` (floor) | 0.500 ± 0.000 | 0.500 ± 0.000 |
| `LogisticRegression` | 0.774 ± 0.028 | 0.830 ± 0.014 |
| `RandomForestClassifier` | 0.811 ± 0.015 | 0.860 ± 0.014 |
| `HistGradientBoostingClassifier` | 0.821 ± 0.019 | 0.871 ± 0.015 |

All three real candidates clear the dummy floor comfortably in both framings.
`HistGradientBoostingClassifier` ranks first, `RandomForestClassifier` a close second,
`LogisticRegression` a further step behind both — the gap between the top two is smaller
than one fold's own standard deviation, so don't over-read that ordering as decisive
(`HANDOFF.md`'s own convention: "a fold-count difference is a claim, not a given").

## 2. The shortlist

### 1st — `HistGradientBoostingClassifier` (or an equivalent gradient-boosted-trees library)

- **Cross-validated score here:** highest of the three, both framings.
- **Properties:** histogram-based gradient boosting; native missing-value handling (no
  imputation step, so `citation_count`'s real nulls stay real nulls per this repo's
  NULL≠0 convention instead of needing a workaround); captures non-linear interactions
  between metadata and embedding dimensions; no feature scaling required.
- **Why it's first:** best raw discriminative power observed, and the only candidate that
  takes missing metadata as a first-class input rather than fighting it. Consistent with
  `wf_fold_pca_test.ipynb`'s own finding that a related gradient-boosting learner matched
  a PCA+logistic stack in-distribution with less manual feature prep.
- **Caveat:** every model type tested in `wf_fold_pca_test.ipynb`, including gradient
  boosting, collapsed to 0.53–0.59 AUC on a fully held-out use case — this ranking is an
  in-distribution result, not a promise it generalizes to a 7th research question.

### 2nd — `RandomForestClassifier`

- **Cross-validated score here:** close second, both framings — within noise of 1st.
- **Properties:** bagged decision trees; robust to outliers/skew (citation counts, §5.1)
  without a log-transform first; built-in feature-importance ranking, useful as an
  ongoing leakage/confound check (watch for `has_abstract` or raw `citation_count`
  dominating importances, per `wf_eda_fe_report.md` §9); fewer hyperparameters than
  boosting, cheaper to tune.
- **Why it's second:** a structurally different model (bagging vs. boosting) scoring
  this close is valuable as a cross-check — agreement between the two increases
  confidence in either; disagreement flags something worth investigating before trusting
  either ranking.

### 3rd — `LogisticRegression` (`class_weight="balanced"`, standardized features)

- **Cross-validated score here:** lowest of the three tested, still far above the dummy
  floor.
- **Properties:** linear decision boundary; fast to fit/score at this dataset size (a few
  thousand rows); directly interpretable coefficients, especially paired with a
  PCA-reduced embedding (`wf_fold_pca_test.ipynb` found K=30 components as the sweep
  peak); well-calibrated probabilities without extra calibration machinery.
- **Why it's still shortlisted:** the eventual product is a *ranked probability* over an
  unreviewed pool (478 never-triaged rows, §2.1), not just a label — calibration matters
  as much as raw AUC there. It's also the right floor-above-the-floor: if a more complex
  model can't beat a well-tuned logistic regression by more than fold-to-fold noise,
  that's worth knowing before paying for the added complexity of tree ensembles in
  production.

## 3. Considered and not shortlisted

- **Linear SVM (`LinearSVC`)** — needs an extra calibration step (Platt scaling) for
  usable probabilities, which the three candidates above get for free, for no clear
  upside at this dataset size.
- **A small MLP / neural network** — nothing about this problem's scale (low thousands
  of rows) or feature structure (one pre-computed sentence embedding, not raw text
  needing its own representation learning) suggests it would beat a boosted-tree or
  linear model by enough to justify the extra tuning/calibration burden as a *baseline*.
  Worth trying later as a non-baseline comparison point, not ruled out for good.

## 4. Before trusting this for real modelling work

1. Re-run `sf_eda_firstrun.ipynb` §13 once the term-overlap feature and cleaned
   `venue`/`author_count` features (§11 of that notebook) are actually built and joined
   on — this comparison used only the embedding + one metadata feature on purpose, and a
   wider feature set will very likely move all three numbers by more than the gaps
   currently separating them.
2. Never skip the held-out-use-case generalization check (`wf_fold_pca_test.ipynb`'s
   own headline finding) before trusting an in-distribution score on a new research
   question.
3. Check calibration (reliability curves), not just discrimination (ROC-AUC) — the
   scoring pool's whole purpose is a ranked probability, not a bare label.
