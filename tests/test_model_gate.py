"""Quality gate: a model must clear these checks before it can reach main.

The gate trains the reference config itself (rather than reading mlflow.db)
so it runs the same on a fresh CI checkout.
"""
import time

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import SEED, get_splits

MIN_VAL_F1 = 0.88
MAX_LATENCY_MS = 30.0


@pytest.fixture(scope="module")
def splits():
    return get_splits()


@pytest.fixture(scope="module")
def model(splits):
    X_train, _, y_train, _ = splits
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    return clf.fit(X_train, y_train)


def test_validation_f1_threshold(splits):
    X_train, _, y_train, _ = splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    assert scores.mean() >= MIN_VAL_F1, f"val macro F1 {scores.mean():.3f} < {MIN_VAL_F1}"


def test_test_split_f1_threshold(model, splits):
    _, X_test, _, y_test = splits
    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    assert f1 >= MIN_VAL_F1


def test_batch_inference_latency(model, splits):
    _, X_test, _, _ = splits
    model.predict(X_test)  # warm-up call
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        model.predict(X_test)
        timings.append((time.perf_counter() - start) * 1000)
    best = min(timings)
    assert best <= MAX_LATENCY_MS, f"batch latency {best:.1f} ms > {MAX_LATENCY_MS} ms"


def test_output_schema_only_valid_classes(model, splits):
    _, X_test, _, _ = splits
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1, 2})
