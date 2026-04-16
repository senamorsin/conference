"""sklearn HistGradientBoostingClassifier on RF-selected features.

No external dependencies required. Historically edges RandomForest by 1-3
points on engineered tabular features and is very CPU-friendly at inference.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "gbdt_hist"
FAMILY = "tree"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 3,
    "l2_regularization": 1e-3,
    "class_weight": "balanced",
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
}


def run(context: ExperimentContext) -> ExperimentResult:
    selected = select_top_k_by_rf(
        context.X_train,
        context.y_train,
        k=HYPERPARAMETERS["select_top_k_features"],
        seed=context.seed,
        n_estimators=HYPERPARAMETERS["scout_n_estimators"],
    )

    def make_pipeline(X, y):
        model = HistGradientBoostingClassifier(
            max_iter=HYPERPARAMETERS["max_iter"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            max_leaf_nodes=HYPERPARAMETERS["max_leaf_nodes"],
            min_samples_leaf=HYPERPARAMETERS["min_samples_leaf"],
            l2_regularization=HYPERPARAMETERS["l2_regularization"],
            class_weight=HYPERPARAMETERS["class_weight"],
            early_stopping=HYPERPARAMETERS["early_stopping"],
            validation_fraction=HYPERPARAMETERS["validation_fraction"],
            n_iter_no_change=HYPERPARAMETERS["n_iter_no_change"],
            random_state=context.seed,
        )
        model.fit(X, y)
        return model

    return run_sklearn_runner(
        name=NAME,
        family=FAMILY,
        context=context,
        make_pipeline=make_pipeline,
        selected_indices=selected,
        hyperparameters=HYPERPARAMETERS,
        notes="HistGradientBoostingClassifier with early stopping on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
