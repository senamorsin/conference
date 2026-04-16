"""Probability-average ensemble of RandomForest + the best available GBDT.

Picks the boosted implementation in order of historical strength:
XGBoost -> LightGBM -> sklearn HistGradientBoostingClassifier. Predictions are
the argmax of the mean of the two models' ``predict_proba`` vectors.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    run_sklearn_runner,
    select_top_k_by_rf,
)


NAME = "rf_gbdt_ensemble"
FAMILY = "ensemble"


class _ProbabilityAverageEnsemble:
    """Tiny estimator that averages probabilities from two fitted models.

    Both models must expose ``predict_proba`` and ``classes_``. We align the
    two class orderings when averaging so labels match up correctly.
    """

    def __init__(self, model_a, model_b, classes_a, classes_b):
        self._model_a = model_a
        self._model_b = model_b
        self._classes = list(classes_a)
        self._b_to_a_indices = np.array(
            [self._classes.index(cls) for cls in classes_b],
            dtype=int,
        )
        self.classes_ = np.array(self._classes)

    def predict_proba(self, X):
        proba_a = self._model_a.predict_proba(X)
        proba_b_raw = self._model_b.predict_proba(X)
        proba_b = np.zeros_like(proba_a)
        proba_b[:, self._b_to_a_indices] = proba_b_raw
        return 0.5 * (proba_a + proba_b)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fit(self, X, y):
        # No-op: sub-models are pre-fit by ``make_pipeline``. Required so the
        # sklearn-style runner harness treats this as a fitted estimator.
        return self

    _is_fitted = True


def _build_gbdt(seed: int):
    notes = ""
    try:
        import xgboost as xgb  # noqa: F401

        from experiments.words.runners.gbdt_xgb import HYPERPARAMETERS as xgb_hp
        import xgboost as xgb

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=xgb_hp["n_estimators"],
            max_depth=xgb_hp["max_depth"],
            learning_rate=xgb_hp["learning_rate"],
            subsample=xgb_hp["subsample"],
            colsample_bytree=xgb_hp["colsample_bytree"],
            reg_lambda=xgb_hp["reg_lambda"],
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        notes = "xgboost"
        return model, notes, "xgb"
    except Exception:
        pass
    try:
        import lightgbm as lgb

        from experiments.words.runners.gbdt_lgbm import HYPERPARAMETERS as lgb_hp

        model = lgb.LGBMClassifier(
            n_estimators=lgb_hp["n_estimators"],
            num_leaves=lgb_hp["num_leaves"],
            learning_rate=lgb_hp["learning_rate"],
            subsample=lgb_hp["subsample"],
            colsample_bytree=lgb_hp["colsample_bytree"],
            reg_lambda=lgb_hp["reg_lambda"],
            min_child_samples=lgb_hp["min_child_samples"],
            class_weight=lgb_hp["class_weight"],
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        notes = "lightgbm"
        return model, notes, "lgb"
    except Exception:
        pass
    return (
        HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            class_weight="balanced",
            random_state=seed,
        ),
        "sklearn-hist",
        "hist",
    )


def run(context: ExperimentContext) -> ExperimentResult:
    selected = select_top_k_by_rf(
        context.X_train,
        context.y_train,
        k=300,
        seed=context.seed,
        n_estimators=200,
    )

    def make_pipeline(X, y):
        rf = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=context.seed,
            n_jobs=-1,
        )
        gbdt, _, kind = _build_gbdt(context.seed)
        if kind == "xgb":
            labels = sorted(set(y.tolist()))
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            y_int = np.array([label_to_idx[str(label)] for label in y], dtype=np.int64)
            gbdt.fit(X, y_int)
            gbdt_classes = list(labels)
        else:
            gbdt.fit(X, y)
            gbdt_classes = list(gbdt.classes_)
        rf.fit(X, y)
        return _ProbabilityAverageEnsemble(
            rf,
            gbdt,
            classes_a=list(rf.classes_),
            classes_b=gbdt_classes,
        )

    _, gbdt_note, _ = _build_gbdt(context.seed)

    return run_sklearn_runner(
        name=NAME,
        family=FAMILY,
        context=context,
        make_pipeline=make_pipeline,
        selected_indices=selected,
        hyperparameters={
            "select_top_k_features": 300,
            "rf": {"n_estimators": 400, "class_weight": "balanced_subsample"},
            "gbdt_backend": gbdt_note,
        },
        notes=f"Probability average of RandomForest and {gbdt_note} on RF-selected top-K features.",
        extras={"selected_feature_indices": selected.tolist(), "gbdt_backend": gbdt_note},
    )
