"""Data loading, validation, and splitting module for Wine Cultivar dataset."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_and_validate_data(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load Wine dataset from scikit-learn, perform validation checks,
    and split into stratified train and test sets.

    Returns:
        X_train (pd.DataFrame): Training features (80%)
        X_test (pd.DataFrame): Testing features (20%)
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """
    wine = load_wine(as_frame=True)
    df = wine.frame
    X = wine.data
    y = wine.target

    # Data Validation Checks
    # 1. Ensure no missing/null values exist
    null_count = int(df.isnull().sum().sum())
    if null_count != 0:
        raise ValueError(f"Data validation failed: Dataset contains {null_count} null values.")

    # 2. Ensure feature count equals exactly 13
    if X.shape[1] != 13:
        raise ValueError(
            f"Data validation failed: Expected 13 features, but found {X.shape[1]} features."
        )

    # 3. Ensure sample count equals 178
    if X.shape[0] != 178:
        raise ValueError(
            f"Data validation failed: Expected 178 samples, but found {X.shape[0]} samples."
        )

    # Stratified 80/20 train-test split using fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_and_validate_data()
    print("Data successfully loaded and validated:")
    print(f"  Training set: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f"  Test set:     {X_te.shape[0]} samples, {X_te.shape[1]} features")
