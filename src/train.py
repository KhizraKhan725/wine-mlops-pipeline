"""Model training, hyperparameter tuning, and MLflow tracking pipeline."""

import os
import sys
from typing import Dict, Any, List
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath("."))
from src.data import load_and_validate_data  # noqa: E402

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_REGISTRY_NAME = "WineClassifier"
ALIAS_NAME = "champion"


def get_hyperparameter_grids() -> List[Dict[str, Any]]:
    """Define hyperparameter configurations for Model Family A (RF) and Family B (GBM)."""
    return [
        # Model Family A: RandomForestClassifier (3 configurations)
        {
            "family": "RandomForest",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "criterion": "gini",
                "random_state": 42
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg1"}
        },
        {
            "family": "RandomForest",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "criterion": "gini",
                "random_state": 42
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg2"}
        },
        {
            "family": "RandomForest",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 150,
                "max_depth": 7,
                "criterion": "entropy",
                "random_state": 42
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg3"}
        },
        # Model Family B: GradientBoostingClassifier (3 configurations)
        {
            "family": "GradientBoosting",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg1"}
        },
        {
            "family": "GradientBoosting",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg2"}
        },
        {
            "family": "GradientBoosting",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 150,
                "learning_rate": 0.15,
                "max_depth": 4,
                "random_state": 42
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg3"}
        }
    ]


def evaluate_cv(model_class, params: Dict[str, Any], X_train, y_train, n_splits: int = 5):
    """Perform 5-fold stratified cross validation and compute metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_train_acc, cv_val_acc = [], []
    cv_train_f1, cv_val_f1 = [], []
    cv_train_loss, cv_val_loss = [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = model_class(**params)
        fold_model.fit(X_tr_f, y_tr_f)

        # Predictions
        y_tr_pred = fold_model.predict(X_tr_f)
        y_val_pred = fold_model.predict(X_val_f)
        y_tr_proba = fold_model.predict_proba(X_tr_f)
        y_val_proba = fold_model.predict_proba(X_val_f)

        # Train Metrics
        cv_train_acc.append(accuracy_score(y_tr_f, y_tr_pred))
        cv_train_f1.append(f1_score(y_tr_f, y_tr_pred, average="macro"))
        cv_train_loss.append(log_loss(y_tr_f, y_tr_proba, labels=[0, 1, 2]))

        # Validation Metrics
        cv_val_acc.append(accuracy_score(y_val_f, y_val_pred))
        cv_val_f1.append(f1_score(y_val_f, y_val_pred, average="macro"))
        cv_val_loss.append(log_loss(y_val_f, y_val_proba, labels=[0, 1, 2]))

    return {
        "train_accuracy": float(np.mean(cv_train_acc)),
        "val_accuracy": float(np.mean(cv_val_acc)),
        "train_macro_f1": float(np.mean(cv_train_f1)),
        "val_macro_f1": float(np.mean(cv_val_f1)),
        "train_log_loss": float(np.mean(cv_train_loss)),
        "val_log_loss": float(np.mean(cv_val_loss)),
    }


def run_training_pipeline():
    """Execute hyperparameter exploration, MLflow logging, and champion model registration."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_validate_data()
    configs = get_hyperparameter_grids()

    logged_runs = []
    print(f"Starting MLflow experiment: '{EXPERIMENT_NAME}' (Tracking URI: {tracking_uri})")

    for cfg in configs:
        run_name = f"{cfg['family']}_{cfg['tags']['config_id']}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log tags & hyperparams
            mlflow.set_tags(cfg["tags"])
            mlflow.log_params(cfg["params"])

            # 5-fold Cross-Validation
            metrics = evaluate_cv(cfg["model_class"], cfg["params"], X_train, y_train)
            mlflow.log_metrics(metrics)

            # Train final model on full training split
            final_model = cfg["model_class"](**cfg["params"])
            final_model.fit(X_train, y_train)

            # Model signature & input example
            signature = infer_signature(X_train, final_model.predict(X_train))
            input_example = X_train.iloc[:5]

            # Log Model artifact
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle"
            )

            logged_runs.append({
                "run_id": run_id,
                "run_name": run_name,
                "family": cfg["family"],
                "val_macro_f1": metrics["val_macro_f1"],
                "val_accuracy": metrics["val_accuracy"],
                "val_log_loss": metrics["val_log_loss"],
                "train_macro_f1": metrics["train_macro_f1"],
                "train_accuracy": metrics["train_accuracy"],
                "train_log_loss": metrics["train_log_loss"],
                "params": cfg["params"]
            })
            print(
                f"Logged '{run_name}' [{run_id[:8]}]: "
                f"Val F1={metrics['val_macro_f1']:.4f}, "
                f"Val Acc={metrics['val_accuracy']:.4f}, "
                f"Val Loss={metrics['val_log_loss']:.4f}"
            )

    # Model Registry and Champion Promotion
    best_run = max(logged_runs, key=lambda r: r["val_macro_f1"])
    print("\n" + "=" * 70)
    print(f"Champion: {best_run['run_name']} (Val F1: {best_run['val_macro_f1']:.4f})")
    print("=" * 70)

    client = MlflowClient()
    model_uri = f"runs:/{best_run['run_id']}/model"

    try:
        try:
            client.create_registered_model(MODEL_REGISTRY_NAME)
        except Exception:
            pass

        mv = client.create_model_version(
            name=MODEL_REGISTRY_NAME,
            source=model_uri,
            run_id=best_run["run_id"],
            description="Champion wine cultivar classification model chosen by validation F1."
        )

        client.set_registered_model_alias(
            name=MODEL_REGISTRY_NAME,
            alias=ALIAS_NAME,
            version=mv.version
        )
        print(f"Registered model '{MODEL_REGISTRY_NAME}' v{mv.version} alias '@{ALIAS_NAME}'.")
    except Exception as e:
        print(f"Model registry notification: {e}")

    return logged_runs, best_run


if __name__ == "__main__":
    run_training_pipeline()
