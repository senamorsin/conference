"""k-NN with cosine distance on RF-selected features.

Often surprisingly strong on balanced small datasets. Cosine distance is a
decent fit for angle-normalized landmark vectors.
"""
from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "knn_cosine"
FAMILY = "knn"

HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "cosine",
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
                    KNeighborsClassifier(
                        n_neighbors=HYPERPARAMETERS["n_neighbors"],
                        weights=HYPERPARAMETERS["weights"],
                        metric=HYPERPARAMETERS["metric"],
                        n_jobs=-1,
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
        notes="StandardScaler -> KNN(k=5, cosine, distance-weighted) on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist()},
    )
