"""L2-regularized multinomial logistic regression on RF-selected features.

Good calibration source and a strong stacking input. Pipeline:
``StandardScaler -> LogisticRegression(penalty='l2', multi_class='auto')``.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "logreg_l2"
FAMILY = "linear"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": "balanced",
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
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=HYPERPARAMETERS["C"],
                        solver=HYPERPARAMETERS["solver"],
                        max_iter=HYPERPARAMETERS["max_iter"],
                        class_weight=HYPERPARAMETERS["class_weight"],
                        random_state=context.seed,
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)
        return pipeline

    return run_sklearn_runner(
        name=NAME,
        family=FAMILY,
        context=context,
        make_pipeline=make_pipeline,
        selected_indices=selected,
        hyperparameters=HYPERPARAMETERS,
        notes="StandardScaler -> LogisticRegression(L2, multinomial) on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
