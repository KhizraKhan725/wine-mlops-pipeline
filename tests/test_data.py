"""Unit tests for the Wine dataset ingestion and validation pipeline."""

from src.data import load_and_validate_data


def test_data_shapes():
    """Verify 80/20 train-test stratified split shapes."""
    X_train, X_test, y_train, y_test = load_and_validate_data(test_size=0.20, random_state=42)
    assert X_train.shape[0] == 142, f"Expected 142 train samples, got {X_train.shape[0]}"
    assert X_test.shape[0] == 36, f"Expected 36 test samples, got {X_test.shape[0]}"
    assert y_train.shape[0] == 142
    assert y_test.shape[0] == 36


def test_feature_count():
    """Verify exact 13 chemical cultivar features exist."""
    X_train, X_test, _, _ = load_and_validate_data()
    assert X_train.shape[1] == 13, f"Expected 13 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 13, f"Expected 13 features, got {X_test.shape[1]}"


def test_null_values():
    """Verify that there are zero missing or null values in datasets."""
    X_train, X_test, y_train, y_test = load_and_validate_data()
    assert X_train.isnull().sum().sum() == 0, "Null values found in X_train"
    assert X_test.isnull().sum().sum() == 0, "Null values found in X_test"
    assert y_train.isnull().sum() == 0, "Null values found in y_train"
    assert y_test.isnull().sum() == 0, "Null values found in y_test"


def test_target_classes():
    """Verify target classes are strictly 3 categories (0, 1, 2)."""
    _, _, y_train, y_test = load_and_validate_data()
    unique_train = set(y_train.unique())
    unique_test = set(y_test.unique())
    expected = {0, 1, 2}
    assert unique_train == expected, f"Expected classes {expected}, got {unique_train}"
    assert unique_test == expected, f"Expected classes {expected}, got {unique_test}"
