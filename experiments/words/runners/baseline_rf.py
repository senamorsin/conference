"""Reference RandomForest runner matching ``scripts/train_words.py``.

Features: RF importance-based top-300 feature selection (the same scout step
as the production training script), followed by a 400-tree RandomForest.
This is the ground-truth anchor every other runner is compared against.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "baseline_rf"
FAMILY = "tree"

HYPERPARAMETERS = {
    "n_estimators": 400,
    "class_weight": "balanced_subsample",
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
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
        model = RandomForestClassifier(
            n_estimators=HYPERPARAMETERS["n_estimators"],
            random_state=context.seed,
            class_weight=HYPERPARAMETERS["class_weight"],
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
        notes="Mirrors scripts/train_words.py: scout-RF top-K feature selection then full RF.",
        extras={"selected_feature_indices": selected.tolist()},
    )
