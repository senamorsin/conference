"""StackingClassifier with RandomForest + LogisticRegression base models
and Logistic Regression as the meta-learner. All on RF-selected features.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "rf_logreg_stack"
FAMILY = "ensemble"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "rf_n_estimators": 400,
    "logreg_C": 1.0,
    "stack_cv": 5,
    "final_estimator_C": 1.0,
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
        logreg = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=HYPERPARAMETERS["logreg_C"],
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=context.seed,
                    ),
                ),
            ]
        )
        rf = RandomForestClassifier(
            n_estimators=HYPERPARAMETERS["rf_n_estimators"],
            class_weight="balanced_subsample",
            random_state=context.seed,
            n_jobs=-1,
        )
        stack = StackingClassifier(
            estimators=[("rf", rf), ("logreg", logreg)],
            final_estimator=LogisticRegression(
                C=HYPERPARAMETERS["final_estimator_C"],
                solver="lbfgs",
                max_iter=2000,
                random_state=context.seed,
            ),
            cv=HYPERPARAMETERS["stack_cv"],
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        )
        stack.fit(X, y)
        return stack

    return run_sklearn_runner(
        name=NAME,
        family=FAMILY,
        context=context,
        make_pipeline=make_pipeline,
        selected_indices=selected,
        hyperparameters=HYPERPARAMETERS,
        notes="Stacking(RF, LogReg) -> LogReg meta on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
