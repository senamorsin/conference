"""Run every word-classifier candidate under a shared harness.

Examples:
    python -m experiments.words.run_leaderboard
    python -m experiments.words.run_leaderboard --subset baseline_rf gbdt_hist
    python -m experiments.words.run_leaderboard --skip temporal_cnn_v2 bigru_attn transformer_small
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.words.common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORTS_DIR,
    ExperimentContext,
    ExperimentResult,
    RunnerOutcome,
    SkipRunner,
    load_context,
    set_global_seed,
    write_result_json,
)


RUNNER_REGISTRY: tuple[str, ...] = (
    "baseline_rf",
    "gbdt_hist",
    "gbdt_xgb",
    "gbdt_lgbm",
    "extra_trees",
    "logreg_l2",
    "linear_svm",
    "knn_cosine",
    "rf_gbdt_ensemble",
    "rf_logreg_stack",
    "temporal_cnn_v2",
    "bigru_attn",
    "transformer_small",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Word training config to load.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Where to write per-runner JSON and the leaderboard.")
    parser.add_argument("--seed", type=int, default=42, help="Shared random seed for all runners.")
    parser.add_argument("--latency-repeats", type=int, default=200, help="Single-sample predict() timings per runner.")
    parser.add_argument("--subset", nargs="+", default=None, help="Only run these runners (names from the registry).")
    parser.add_argument("--skip", nargs="+", default=None, help="Skip these runners by name.")
    parser.add_argument("--list", action="store_true", help="Print the runner registry and exit.")
    return parser.parse_args()


def resolve_runner_names(args: argparse.Namespace) -> list[str]:
    names = list(RUNNER_REGISTRY)
    if args.subset:
        unknown = [n for n in args.subset if n not in RUNNER_REGISTRY]
        if unknown:
            raise SystemExit(f"Unknown runner(s): {unknown}. Known: {RUNNER_REGISTRY}")
        names = [n for n in args.subset]
    if args.skip:
        skip = set(args.skip)
        names = [n for n in names if n not in skip]
    return names


def _import_runner(name: str):
    return importlib.import_module(f"experiments.words.runners.{name}")


def _execute_runner(name: str, context: ExperimentContext) -> RunnerOutcome:
    started = time.perf_counter()
    try:
        module = _import_runner(name)
    except ModuleNotFoundError as exc:
        return RunnerOutcome(name=name, status="skipped", message=f"missing runner module: {exc}", duration_seconds=0.0)

    run_fn = getattr(module, "run", None)
    if run_fn is None:
        return RunnerOutcome(name=name, status="error", message="runner has no run() function", duration_seconds=0.0)

    try:
        result = run_fn(context)
    except SkipRunner as exc:
        return RunnerOutcome(name=name, status="skipped", message=str(exc), duration_seconds=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - we want to surface anything
        traceback.print_exc()
        return RunnerOutcome(
            name=name,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.perf_counter() - started,
        )

    if result is None:
        return RunnerOutcome(name=name, status="skipped", message="runner returned None", duration_seconds=time.perf_counter() - started)
    if not isinstance(result, ExperimentResult):
        return RunnerOutcome(
            name=name,
            status="error",
            message=f"runner returned {type(result).__name__}, expected ExperimentResult",
            duration_seconds=time.perf_counter() - started,
        )
    return RunnerOutcome(
        name=name,
        status="ok",
        result=result,
        duration_seconds=time.perf_counter() - started,
    )


def _sorted_results(outcomes: Iterable[RunnerOutcome]) -> list[RunnerOutcome]:
    ok = [o for o in outcomes if o.status == "ok" and o.result is not None]
    ok.sort(
        key=lambda o: (
            -o.result.f1_macro,
            -o.result.accuracy,
            o.result.latency.p95_ms,
            o.result.artifact_size_mb,
        ),
    )
    other = [o for o in outcomes if o.status != "ok"]
    return ok + other


def _format_leaderboard_md(outcomes: list[RunnerOutcome], *, context: ExperimentContext) -> str:
    lines: list[str] = ["# Words model leaderboard", ""]
    lines.append(f"- Config: `{context.source_description}`")
    lines.append(f"- Split: `{context.split_strategy}`")
    lines.append(f"- Train samples: `{context.y_train.size}`  Test samples: `{context.y_test.size}`")
    lines.append(f"- Labels ({len(context.labels)}): `{', '.join(context.labels)}`")
    lines.append("")
    lines.append("## Ranked entries")
    lines.append("")
    for outcome in outcomes:
        if outcome.status != "ok" or outcome.result is None:
            continue
        r = outcome.result
        lines.append(f"### {r.name}  (`{r.family}`)")
        lines.append(f"- Accuracy: `{r.accuracy:.4f}`")
        lines.append(f"- Macro F1: `{r.f1_macro:.4f}`  Weighted F1: `{r.f1_weighted:.4f}`")
        lines.append(
            f"- Latency (ms): mean `{r.latency.mean_ms:.3f}`  p50 `{r.latency.p50_ms:.3f}`  p95 `{r.latency.p95_ms:.3f}`  (n={r.latency.samples})"
        )
        lines.append(f"- Artifact size: `{r.artifact_size_mb:.2f} MB`")
        lines.append(f"- Train seconds: `{outcome.duration_seconds:.2f}`")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append("")
    skipped = [o for o in outcomes if o.status == "skipped"]
    errored = [o for o in outcomes if o.status == "error"]
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for outcome in skipped:
            lines.append(f"- `{outcome.name}`: {outcome.message}")
        lines.append("")
    if errored:
        lines.append("## Errored")
        lines.append("")
        for outcome in errored:
            lines.append(f"- `{outcome.name}`: {outcome.message}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.list:
        print("\n".join(RUNNER_REGISTRY))
        return

    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    context = load_context(
        Path(args.config).expanduser().resolve(),
        seed=args.seed,
        latency_repeats=args.latency_repeats,
    )

    names = resolve_runner_names(args)

    outcomes: list[RunnerOutcome] = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        outcome = _execute_runner(name, context)
        if outcome.result is not None:
            out_path = write_result_json(outcome.result, reports_dir)
            print(
                f"  status={outcome.status}  acc={outcome.result.accuracy:.4f}  "
                f"f1m={outcome.result.f1_macro:.4f}  p95={outcome.result.latency.p95_ms:.3f}ms  "
                f"size={outcome.result.artifact_size_mb:.2f}MB  -> {out_path}",
                flush=True,
            )
        else:
            print(f"  status={outcome.status}  {outcome.message}", flush=True)
        outcomes.append(outcome)

    ranked = _sorted_results(outcomes)
    summary = {
        "config": str(Path(args.config).resolve()),
        "seed": args.seed,
        "split_strategy": context.split_strategy,
        "source_description": context.source_description,
        "train_samples": int(context.y_train.size),
        "test_samples": int(context.y_test.size),
        "labels": list(context.labels),
        "entries": [
            {
                "name": o.name,
                "status": o.status,
                "message": o.message,
                "duration_seconds": o.duration_seconds,
                "result": o.result.to_dict() if o.result is not None else None,
            }
            for o in ranked
        ],
    }
    (reports_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "leaderboard.md").write_text(_format_leaderboard_md(ranked, context=context), encoding="utf-8")
    print(f"\nLeaderboard written to {reports_dir/'leaderboard.md'}")


if __name__ == "__main__":
    main()
