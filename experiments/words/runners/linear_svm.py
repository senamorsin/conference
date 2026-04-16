"""Calibrated LinearSVC on RF-selected features with StandardScaler.

Platt-style calibration is wrapped via ``CalibratedClassifierCV`` so the
model exposes ``predict_proba`` for downstream ensembling and for the app's
confidence/rejection logic.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "linear_svm"
FAMILY = "linear"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "C": 0.5,
    "class_weight": "balanced",
    "max_iter": 5000,
    "calibration_cv": 5,
    "calibration_method": "sigmoid",
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
        base = LinearSVC(
            C=HYPERPARAMETERS["C"],
            class_weight=HYPERPARAMETERS["class_weight"],
            max_iter=HYPERPARAMETERS["max_iter"],
            random_state=context.seed,
            dual="auto",
        )
        calibrated = CalibratedClassifierCV(
            base,
            method=HYPERPARAMETERS["calibration_method"],
            cv=HYPERPARAMETERS["calibration_cv"],
            n_jobs=-1,
        )
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", calibrated),
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
        notes="StandardScaler -> CalibratedClassifierCV(LinearSVC, sigmoid) on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
