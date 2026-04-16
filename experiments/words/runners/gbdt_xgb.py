"""XGBoost multiclass classifier on RF-selected features.

Skipped cleanly if the ``xgboost`` package is not installed.
"""
from __future__ import annotations

import numpy as np

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    SkipRunner,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "gbdt_xgb"
FAMILY = "tree"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_weight": 1.0,
}


def run(context: ExperimentContext) -> ExperimentResult:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise SkipRunner(f"xgboost not installed: {exc}") from exc

    selected = select_top_k_by_rf(
        context.X_train,
        context.y_train,
        k=HYPERPARAMETERS["select_top_k_features"],
        seed=context.seed,
        n_estimators=HYPERPARAMETERS["scout_n_estimators"],
    )

    labels = list(context.labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def make_pipeline(X, y):
        y_int = np.array([label_to_idx[str(label)] for label in y], dtype=np.int64)
        classifier = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(labels),
            n_estimators=HYPERPARAMETERS["n_estimators"],
            max_depth=HYPERPARAMETERS["max_depth"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            subsample=HYPERPARAMETERS["subsample"],
            colsample_bytree=HYPERPARAMETERS["colsample_bytree"],
            reg_lambda=HYPERPARAMETERS["reg_lambda"],
            reg_alpha=HYPERPARAMETERS["reg_alpha"],
            min_child_weight=HYPERPARAMETERS["min_child_weight"],
            random_state=context.seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        classifier.fit(X, y_int)
        return classifier

    def predict_fn(estimator, X):
        idx = estimator.predict(X)
        preds = np.array([labels[int(i)] for i in idx])
        return preds, None

    return run_sklearn_runner(
        name=NAME,
        family=FAMILY,
        context=context,
        make_pipeline=make_pipeline,
        predict_fn=predict_fn,
        selected_indices=selected,
        hyperparameters=HYPERPARAMETERS,
        notes="XGBoost hist tree method on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
