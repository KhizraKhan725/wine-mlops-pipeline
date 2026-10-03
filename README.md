# Wine MLOps Pipeline

![CI](https://github.com/KhizraKhan725/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

MLOps Assignment 1 (FAST-NUCES, Fall 2026): a reproducible training + CI pipeline for
the sklearn Wine dataset (178 samples, 13 features, 3 cultivar classes).

## What it does
- `src/data.py` loads the data, validates it (no nulls, 13 features) and makes a stratified
  80/20 split with `random_state=42`.
- `src/train.py` runs 3 configs each of RandomForest and GradientBoosting, scored with
  5-fold stratified CV (macro F1, accuracy, log loss, train + validation). Every config is
  an MLflow run with params, metrics, tags, a signature and an input example. The run with the
  best validation macro F1 is registered as `WineClassifier` and given the alias `champion`.
- `src/evaluate.py` loads `models:/WineClassifier@champion` and scores it on the test split.
- `tests/test_model_gate.py` is the quality gate used by CI: validation macro F1 >= 0.88,
  batch inference <= 30 ms, predictions only in {0, 1, 2}.

## Usage
```bash
python -m venv .venv && source .venv/bin/activate
make install     # upgrade pip, install pinned requirements
make lint        # flake8, max line length 100
make test        # pytest -v (data checks + model gate)
make train       # run the experiments, register the champion
make evaluate    # test-split metrics for the champion
make clean       # remove caches / bytecode
```

To browse the runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
(open http://127.0.0.1:5000).

## Notes
- Seed is 42 everywhere (split, CV folds, model init).
- CI runs on Python 3.10, so `requirements.txt` pins versions that support 3.10.
- `mlflow.db` and `mlruns/` are git-ignored; the CI gate trains its own reference model
  instead of depending on them.
- Models are logged with cloudpickle because MLflow's default skops format rejects
  sklearn tree objects as untrusted.
