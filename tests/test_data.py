import pandas as pd
import pytest

from src.data import N_FEATURES, get_splits, load_data, validate_data


def test_feature_count():
    X, _ = load_data()
    assert X.shape[1] == N_FEATURES == 13


def test_no_nulls():
    X, y = load_data()
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_split_sizes_and_stratification():
    X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) + len(X_test) == 178
    assert abs(len(X_test) / 178 - 0.2) < 0.01
    # every class should show up in both splits
    assert set(y_train) == set(y_test) == {0, 1, 2}


def test_split_is_reproducible():
    a = get_splits()[0]
    b = get_splits()[0]
    assert a.equals(b)


def test_validate_rejects_nulls():
    X, y = load_data()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError):
        validate_data(X, y)


def test_validate_rejects_wrong_feature_count():
    X, y = load_data()
    with pytest.raises(ValueError):
        validate_data(X.iloc[:, :12], y)
    with pytest.raises(ValueError):
        validate_data(pd.concat([X, X["alcohol"].rename("extra")], axis=1), y)
