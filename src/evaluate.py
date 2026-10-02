"""Inference verification and evaluation script for the registered champion model."""

import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

sys.path.insert(0, os.path.abspath("."))
from src.data import load_and_validate_data  # noqa: E402


def evaluate_champion():
    """Load champion model from MLflow registry and evaluate on test set."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    _, X_test, _, y_test = load_and_validate_data()

    model_uri = "models:/WineClassifier@champion"
    print(f"Loading registered model from: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Wine-Cultivar-Classification")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_macro_f1 DESC"],
            max_results=1
        )
        best_run_id = runs[0].info.run_id
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Fallback loading best run from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba, labels=[0, 1, 2])

    print("\n" + "=" * 50)
    print("CHAMPION MODEL TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Accuracy:   {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Test Macro F1:   {f1:.4f}")
    print(f"Test Log Loss:   {loss:.4f}")
    print("\nClassification Report:")
    target_names = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    return {"accuracy": acc, "macro_f1": f1, "log_loss": loss}


if __name__ == "__main__":
    evaluate_champion()
