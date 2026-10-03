"""Train RandomForest and GradientBoosting candidates, track with MLflow,
then register the best run as WineClassifier@champion."""
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_splits

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"

RF_GRID = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 4},
]
GB_GRID = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
]
FAMILIES = [
    ("RandomForest", RandomForestClassifier, RF_GRID),
    ("GradientBoosting", GradientBoostingClassifier, GB_GRID),
]


def cross_validate_config(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"f1": "f1_macro", "acc": "accuracy", "ll": "neg_log_loss"}
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                         return_train_score=True)
    return {
        "train_f1_macro": res["train_f1"].mean(),
        "val_f1_macro": res["test_f1"].mean(),
        "train_accuracy": res["train_acc"].mean(),
        "val_accuracy": res["test_acc"].mean(),
        "train_log_loss": -res["train_ll"].mean(),
        "val_log_loss": -res["test_ll"].mean(),
    }


def run_search(X_train, y_train):
    results = []
    for family, cls, grid in FAMILIES:
        for i, params in enumerate(grid, start=1):
            model = cls(random_state=SEED, **params)
            metrics = cross_validate_config(model, X_train, y_train)

            with mlflow.start_run(run_name=f"{family}-cfg{i}") as run:
                mlflow.set_tags({"model_family": family, "cv": "stratified-5fold"})
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_param("random_state", SEED)
                mlflow.log_metrics(metrics)

                model.fit(X_train, y_train)
                signature = infer_signature(X_train, model.predict(X_train))
                info = mlflow.sklearn.log_model(
                    model, name="model", signature=signature,
                    input_example=X_train.head(5),
                    # tree models trip mlflow's default skops serializer
                    serialization_format="cloudpickle",
                )
            results.append({"family": family, "config": i, "params": params,
                            "run_id": run.info.run_id,
                            "model_uri": info.model_uri, **metrics})
            print(f"{family:17s} cfg{i}  val_f1={metrics['val_f1_macro']:.4f}  "
                  f"val_acc={metrics['val_accuracy']:.4f}  "
                  f"val_logloss={metrics['val_log_loss']:.4f}")
    return results


def promote_best(results):
    best = max(results, key=lambda r: r["val_f1_macro"])
    client = MlflowClient()
    mv = mlflow.register_model(best["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"\nBest: {best['family']} cfg{best['config']} "
          f"(val_f1={best['val_f1_macro']:.4f}) -> {MODEL_NAME} v{mv.version} @champion")
    return best, mv


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_test, y_train, y_test = get_splits()
    results = run_search(X_train, y_train)
    promote_best(results)
    return results


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
