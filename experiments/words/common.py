"""Shared harness for the whole-word classifier leaderboard.

Every candidate runner lives in ``experiments/words/runners`` and exports a
``run(context)`` function returning an ``ExperimentResult``. The leaderboard
runner loads a single word-classifier YAML config, builds identical train/test
splits via ``scripts.train_words.build_splits``, and feeds the same data into
each runner so results can be compared fairly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from scripts.train_words import build_splits  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.words.labels import WORD_FEATURE_COLUMNS  # noqa: E402


DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "model_words_combined.yaml"
DEFAULT_REPORTS_DIR: Path = ROOT / "experiments" / "words" / "reports"


class SkipRunner(RuntimeError):
    """Raise from a runner's ``run()`` function to mark it skipped with a message."""


@dataclass(slots=True)
class ExperimentContext:
    """Frozen data passed to every runner so results are apples-to-apples."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: tuple[str, ...]
    labels: tuple[str, ...]
    seed: int
    split_strategy: str
    source_description: str
    latency_repeats: int = 200
    latency_warmup: int = 20


@dataclass(slots=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    samples: int


@dataclass(slots=True)
class ExperimentResult:
    """Uniform report every runner must produce."""

    name: str
    family: str  # "tree" | "linear" | "knn" | "ensemble" | "neural" | "other"
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_f1: dict[str, float]
    support_per_class: dict[str, int]
    confusion_matrix: list[list[int]]
    labels: list[str]
    latency: LatencyStats
    artifact_size_mb: float
    hyperparameters: dict[str, Any]
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["latency"] = asdict(self.latency)
        return payload


@dataclass(slots=True)
class RunnerOutcome:
    """Either a successful result or a skip/error explanation."""

    name: str
    status: str  # "ok" | "skipped" | "error"
    result: ExperimentResult | None = None
    message: str = ""
    duration_seconds: float = 0.0


def _as_np(a: Sequence[Any]) -> np.ndarray:
    return np.array(a, copy=True)


def load_context(config_path: Path, *, seed: int = 42, latency_repeats: int = 200) -> ExperimentContext:
    config = load_yaml(config_path)
    training_config = dict(config.get("training", {}))
    training_config.setdefault("random_state", seed)
    feature_columns = list(WORD_FEATURE_COLUMNS)
    X_train, y_train, X_test, y_test, split_strategy, source_desc = build_splits(
        config,
        feature_columns,
        training_config,
    )
    labels = tuple(sorted(set(np.concatenate([y_train, y_test]).tolist())))
    return ExperimentContext(
        X_train=_as_np(X_train).astype(np.float32),
        y_train=_as_np(y_train),
        X_test=_as_np(X_test).astype(np.float32),
        y_test=_as_np(y_test),
        feature_columns=tuple(feature_columns),
        labels=labels,
        seed=seed,
        split_strategy=str(split_strategy),
        source_description=str(source_desc),
        latency_repeats=latency_repeats,
    )


def measure_inference_latency(
    predict_one: Callable[[np.ndarray], Any],
    sample: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> LatencyStats:
    """Measure single-sample CPU inference latency.

    ``predict_one`` must take a ``(feature_dim,)`` 1D array and perform a full
    prediction (including any reshape the runtime would do). We force the input
    through ``np.asarray`` each call so runners cannot accidentally cache.
    """
    if repeats <= 0:
        return LatencyStats(mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, samples=0)
    sample = np.asarray(sample, dtype=np.float32).reshape(-1)
    for _ in range(max(0, warmup)):
        predict_one(sample)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_one(sample)
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        samples=int(arr.size),
    )


def artifact_size_mb(obj: Any) -> float:
    """Estimate in-memory artifact size by pickling to a temp location.

    We write to memory via ``joblib.dump`` only if ``obj`` is picklable. Torch
    modules are sized via their state dict instead.
    """
    import io

    try:
        import torch

        if isinstance(obj, torch.nn.Module):
            buffer = io.BytesIO()
            torch.save({k: v.detach().cpu() for k, v in obj.state_dict().items()}, buffer)
            return buffer.getbuffer().nbytes / (1024 * 1024)
    except Exception:
        pass

    try:
        import joblib

        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getbuffer().nbytes / (1024 * 1024)
    except Exception:
        return 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> tuple[float, float, float, dict[str, float], dict[str, int], list[list[int]]]:
    labels_list = list(labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels_list, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_list,
        target_names=labels_list,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {label: float(report[label]["f1-score"]) for label in labels_list if label in report}
    support = {label: int(report[label]["support"]) for label in labels_list if label in report}
    cm = confusion_matrix(y_true, y_pred, labels=labels_list).tolist()
    return accuracy, f1_macro, f1_weighted, per_class_f1, support, cm


def write_result_json(result: ExperimentResult, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{result.name}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out_path


def select_top_k_by_rf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    seed: int = 42,
    n_estimators: int = 200,
) -> np.ndarray:
    """Return sorted indices of the top-``k`` features by RandomForest importance.

    Mirrors the feature-selection step in ``scripts/train_words.py`` so runners
    that consume "RF-selected features" are operating on the exact same support
    as the production RF baseline.
    """
    from sklearn.ensemble import RandomForestClassifier

    effective_k = min(int(k), X.shape[1])
    if effective_k <= 0:
        return np.arange(X.shape[1], dtype=int)
    scout = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    scout.fit(X, y)
    order = np.argsort(scout.feature_importances_)[-effective_k:]
    order.sort()
    return order.astype(int)


def run_sklearn_runner(
    *,
    name: str,
    family: str,
    context: ExperimentContext,
    make_pipeline: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    selected_indices: np.ndarray | None = None,
    hyperparameters: dict[str, Any] | None = None,
    notes: str = "",
    extras: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Train any sklearn-style estimator (has ``fit`` and usually ``predict``) and
    produce an ``ExperimentResult``.

    ``make_pipeline(X_train, y_train)`` must return an estimator already fit on
    ``X_train`` / ``y_train`` (or a Pipeline ready to be ``.fit``-ed). If the
    returned object is not yet fit, we fit it.
    """
    X_train = context.X_train
    X_test = context.X_test
    if selected_indices is not None:
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

    y_train = context.y_train
    y_test = context.y_test
    labels = list(context.labels)

    estimator = make_pipeline(X_train, y_train)
    # Allow runners to return either a fit estimator or an unfit one.
    if not getattr(estimator, "_is_fitted", False):
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(estimator)
            fitted = True
        except Exception:
            fitted = False
        if not fitted:
            estimator.fit(X_train, y_train)

    if predict_fn is not None:
        y_pred, _ = predict_fn(estimator, X_test)
    else:
        y_pred = estimator.predict(X_test)

    accuracy, f1_macro, f1_weighted, per_class_f1, support, cm = compute_metrics(y_test, y_pred, labels)

    def _predict_one(sample: np.ndarray) -> Any:
        arr = np.asarray(sample, dtype=np.float32).reshape(1, -1)
        if selected_indices is not None:
            # Simulate the production path where the full landmark vector is
            # indexed down to the selected features at inference time.
            arr = arr[:, selected_indices]
        if predict_fn is not None:
            result, _ = predict_fn(estimator, arr)
            return result
        return estimator.predict(arr)

    latency = measure_inference_latency(
        _predict_one,
        context.X_test[0],
        repeats=context.latency_repeats,
        warmup=context.latency_warmup,
    )
    size_mb = artifact_size_mb(estimator)
    return ExperimentResult(
        name=name,
        family=family,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        per_class_f1=per_class_f1,
        support_per_class=support,
        confusion_matrix=cm,
        labels=labels,
        latency=latency,
        artifact_size_mb=size_mb,
        hyperparameters=dict(hyperparameters or {}),
        notes=notes,
        extras=dict(extras or {}),
    )


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_REPORTS_DIR",
    "ExperimentContext",
    "ExperimentResult",
    "LatencyStats",
    "RunnerOutcome",
    "SkipRunner",
    "artifact_size_mb",
    "compute_metrics",
    "load_context",
    "measure_inference_latency",
    "run_sklearn_runner",
    "select_top_k_by_rf",
    "set_global_seed",
    "write_result_json",
]
