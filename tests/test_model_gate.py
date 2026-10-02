"""Automated MLOps Quality Gate tests for model operational validation."""

import time
import pytest
import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from src.data import load_and_validate_data


@pytest.fixture(scope="module")
def dataset_and_model():
    """Fixture providing loaded dataset and trained candidate model."""
    X_train, X_test, y_train, y_test = load_and_validate_data()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def test_metric_threshold_gate(dataset_and_model):
    """Quality Gate 1: Macro F1-score must be >= 0.88."""
    model = dataset_and_model["model"]
    X_test = dataset_and_model["X_test"]
    y_test = dataset_and_model["y_test"]

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    threshold = 0.88
    assert macro_f1 >= threshold, f"Model Macro F1 ({macro_f1:.4f}) failed threshold ({threshold})"


def test_inference_latency_gate(dataset_and_model):
    """Quality Gate 2: Batch inference latency must be <= 30 ms."""
    model = dataset_and_model["model"]
    X_test = dataset_and_model["X_test"]

    # Warmup
    _ = model.predict(X_test)
    latencies = []
    for _ in range(50):
        start_time = time.perf_counter()
        _ = model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

    avg_latency_ms = np.mean(latencies)
    max_threshold_ms = 30.0
    assert avg_latency_ms <= max_threshold_ms, (
        f"Inference latency ({avg_latency_ms:.2f} ms) exceeded threshold ({max_threshold_ms} ms)"
    )


def test_output_schema_integrity(dataset_and_model):
    """Quality Gate 3: Output predictions must contain class indices only (0, 1, or 2)."""
    model = dataset_and_model["model"]
    X_test = dataset_and_model["X_test"]

    y_pred = model.predict(X_test)
    valid_classes = {0, 1, 2}
    unique_preds = set(np.unique(y_pred))

    assert unique_preds.issubset(valid_classes), (
        f"Predicted invalid class indices: {unique_preds.difference(valid_classes)}"
    )
    assert np.issubdtype(y_pred.dtype, np.integer), f"Predictions not int: {y_pred.dtype}"
