# Words classifier leaderboard

A shared harness for comparing whole-word classifier candidates on identical
train/test splits, producing a ranked leaderboard and JSON per-run reports.

## How to run

From the repo root:

```bash
source .venv/bin/activate
python -m experiments.words.run_leaderboard
```

Common flags:

- `--config configs/model_words_combined.yaml` (default) - word training config to load, reused via `scripts.train_words.build_splits`.
- `--subset baseline_rf gbdt_xgb` - only run the named runners.
- `--skip temporal_cnn_v2 bigru_attn transformer_small` - skip listed runners (useful when iterating on tabular models).
- `--latency-repeats 200` - how many single-sample `predict()` calls to time.
- `--list` - print the runner registry and exit.

Outputs land in `experiments/words/reports/`:

- `<runner>.json` per run, with accuracy, macro/weighted F1, per-class F1, confusion matrix, latency stats, artifact size, hyperparameters.
- `leaderboard.json` - machine-readable combined report.
- `leaderboard.md` - human-readable ranked entries.

## Adding a runner

Each runner is a module in `experiments/words/runners/` that exports:

```python
def run(context: ExperimentContext) -> ExperimentResult: ...
```

Use `run_sklearn_runner` from `common.py` for any fit/predict estimator; use
`train_sequence_model` from `_torch_common.py` for neural sequence models.
Gate optional dependencies by raising `SkipRunner("pkg not installed")`.
Then add the runner name to `RUNNER_REGISTRY` in `run_leaderboard.py`.

## Selection rule

The leaderboard ranks by macro-F1, then accuracy, then p95 latency, then
artifact size. When integrating a winner into the app we additionally prefer
a smaller / faster model if two candidates are within `0.5` macro-F1 points
of each other, so CPU-friendly models aren't crowded out by marginally better
heavy models.
