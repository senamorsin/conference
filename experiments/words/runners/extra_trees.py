"""ExtraTreesClassifier on RF-selected features.

Cheap variance source; useful on its own and as an ensemble ingredient.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "extra_trees"
FAMILY = "tree"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "n_estimators": 600,
    "class_weight": "balanced_subsample",
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
        model = ExtraTreesClassifier(
            n_estimators=HYPERPARAMETERS["n_estimators"],
            class_weight=HYPERPARAMETERS["class_weight"],
            random_state=context.seed,
            n_jobs=-1,
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
        notes="ExtraTreesClassifier on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
