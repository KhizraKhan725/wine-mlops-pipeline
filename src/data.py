"""Data loading, validation and splitting for the Wine dataset."""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
N_FEATURES = 13


def load_data():
    wine = load_wine(as_frame=True)
    return wine.data, wine.target


def validate_data(X, y):
    """Basic sanity checks - raises ValueError if something is off."""
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("dataset contains null values")
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")
    return True


def get_splits(test_size=0.2, seed=SEED):
    """Stratified 80/20 split, validated before returning."""
    X, y = load_data()
    validate_data(X, y)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_splits()
    print("train:", X_train.shape, "test:", X_test.shape)
    print(pd.Series(y_train).value_counts().sort_index().to_dict())
