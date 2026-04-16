# Words model leaderboard

- Config: `combined`
- Split: `combined_split`
- Train samples: `2034`  Test samples: `257`
- Labels (50): `AGAIN, BABY, BATHROOM, BOOK, BOY, BROTHER, DOCTOR, DRINK, EAT, FAMILY, FATHER, FINISH, FRIEND, GIRL, GO, GOOD, HAPPY, HELLO, HELP, HOME, KNOW, LEARN, LIKE, LOVE, MONEY, MORE, MOTHER, NAME, NEED, NO, PHONE, PLEASE, SAD, SCHOOL, SICK, SISTER, SORRY, THANK_YOU, TIRED, TODAY, TOMORROW, UNDERSTAND, WANT, WATER, WHAT, WHERE, WHO, WORK, YES, YESTERDAY`

## Ranked entries

### rf_gbdt_ensemble  (`ensemble`)
- Accuracy: `0.7121`
- Macro F1: `0.6134`  Weighted F1: `0.7145`
- Latency (ms): mean `40.295`  p50 `39.567`  p95 `42.863`  (n=200)
- Artifact size: `258.36 MB`
- Train seconds: `39.03`
- Notes: Probability average of RandomForest and xgboost on RF-selected top-K features.

### gbdt_xgb  (`tree`)
- Accuracy: `0.6887`
- Macro F1: `0.5967`  Weighted F1: `0.6946`
- Latency (ms): mean `4.626`  p50 `4.365`  p95 `6.215`  (n=200)
- Artifact size: `19.42 MB`
- Train seconds: `31.26`
- Notes: XGBoost hist tree method on RF-selected top-K features.

### extra_trees  (`tree`)
- Accuracy: `0.6965`
- Macro F1: `0.5895`  Weighted F1: `0.6864`
- Latency (ms): mean `47.217`  p50 `46.212`  p95 `50.091`  (n=200)
- Artifact size: `678.34 MB`
- Train seconds: `12.34`
- Notes: ExtraTreesClassifier on RF-selected top-K features.

### gbdt_lgbm  (`tree`)
- Accuracy: `0.6732`
- Macro F1: `0.5778`  Weighted F1: `0.6732`
- Latency (ms): mean `9.825`  p50 `9.320`  p95 `13.616`  (n=200)
- Artifact size: `65.62 MB`
- Train seconds: `106.31`
- Notes: LightGBM classifier on RF-selected top-K features.

### baseline_rf  (`tree`)
- Accuracy: `0.6809`
- Macro F1: `0.5701`  Weighted F1: `0.6637`
- Latency (ms): mean `35.561`  p50 `34.889`  p95 `37.021`  (n=200)
- Artifact size: `238.94 MB`
- Train seconds: `9.89`
- Notes: Mirrors scripts/train_words.py: scout-RF top-K feature selection then full RF.

### rf_logreg_stack  (`ensemble`)
- Accuracy: `0.6654`
- Macro F1: `0.5677`  Weighted F1: `0.6612`
- Latency (ms): mean `36.314`  p50 `35.616`  p95 `38.251`  (n=200)
- Artifact size: `239.11 MB`
- Train seconds: `23.17`
- Notes: Stacking(RF, LogReg) -> LogReg meta on RF-selected top-K features.

### logreg_l2  (`linear`)
- Accuracy: `0.6459`
- Macro F1: `0.5588`  Weighted F1: `0.6510`
- Latency (ms): mean `0.115`  p50 `0.090`  p95 `0.152`  (n=200)
- Artifact size: `0.12 MB`
- Train seconds: `33.23`
- Notes: StandardScaler -> LogisticRegression(L2, multinomial) on RF-selected top-K features.

### gbdt_hist  (`tree`)
- Accuracy: `0.5642`
- Macro F1: `0.4823`  Weighted F1: `0.5612`
- Latency (ms): mean `18.353`  p50 `18.092`  p95 `20.548`  (n=200)
- Artifact size: `8.36 MB`
- Train seconds: `12.40`
- Notes: HistGradientBoostingClassifier with early stopping on RF-selected top-K features.

### linear_svm  (`linear`)
- Accuracy: `0.5525`
- Macro F1: `0.4569`  Weighted F1: `0.5322`
- Latency (ms): mean `2.806`  p50 `2.751`  p95 `3.059`  (n=200)
- Artifact size: `0.60 MB`
- Train seconds: `8.28`
- Notes: StandardScaler -> CalibratedClassifierCV(LinearSVC, sigmoid) on RF-selected top-K features.

### knn_cosine  (`knn`)
- Accuracy: `0.4786`
- Macro F1: `0.3995`  Weighted F1: `0.4655`
- Latency (ms): mean `13.610`  p50 `13.603`  p95 `14.301`  (n=200)
- Artifact size: `2.35 MB`
- Train seconds: `3.95`
- Notes: StandardScaler -> KNN(k=5, cosine, distance-weighted) on RF-selected top-K features.
