# Winner selection

The leaderboard rule is: rank by macro-F1, break ties by accuracy, then prefer
smaller/faster artifacts when the top candidates are within `0.5` macro-F1
points. See `experiments/words/README.md` for the full rule.

Top of the leaderboard (`leaderboard.md`) on the combined MS-ASL + ASL Citizen
split (661 train / 48 test, 20 classes):

- `gbdt_xgb`           accuracy `0.8750`  macro F1 `0.7014`  p95 `1.84 ms`  size `6.95 MB`
- `rf_gbdt_ensemble`   accuracy `0.8750`  macro F1 `0.7014`  p95 `39.27 ms` size `36.37 MB`
- `extra_trees`        accuracy `0.8542`  macro F1 `0.6731`  p95 `49.71 ms` size `87.89 MB`

`gbdt_xgb` and `rf_gbdt_ensemble` are tied on both accuracy and macro F1.
Applying the tie-breaker:

- `gbdt_xgb` is ~21x faster at the p95 (`1.84 ms` vs `39.27 ms`) and ~5x smaller
  (`6.95 MB` vs `36.37 MB`).
- Inference path becomes a single XGBoost forest pass instead of RF + XGB + an
  averaging wrapper.
- Training is also faster and more reproducible.

## Winner: `gbdt_xgb`

Details:

- Algorithm: XGBoost `multi:softprob` with the `hist` tree method.
- Feature pipeline: RF-importance scout -> top-`300` features from the `1430`
  engineered landmark features.
- Hyperparameters: `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`,
  `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.0`.
- Full report: [`gbdt_xgb.json`](./gbdt_xgb.json).

## Notes on the other candidates

- The current production RF baseline (`baseline_rf`) still matches
  `scripts/train_words.py` exactly at `acc=0.8333, f1m=0.6529`, so the
  comparison is fair.
- All sequence models (temporal CNN, BiGRU, Transformer) underperformed the
  top tabular models. On ~250-700 training samples of richly engineered
  landmark features, tree ensembles are just the right tool. `temporal_cnn_v2`
  lifted the previous CNN prototype from `0.6875` to `0.8125` thanks to delta
  features and augmentation, which validates those changes as reusable
  ingredients if we ever outgrow the tabular approach.
- `knn_cosine` and `gbdt_hist` were underwhelming on this split. LGBM trails
  XGB here, likely due to small-data regularization differences.
