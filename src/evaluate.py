"""Load the registered champion model and score it on the held-out test split."""
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_splits
from src.train import MODEL_NAME, TRACKING_URI


def evaluate_champion():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    _, X_test, _, y_test = get_splits()

    preds = model.predict(X_test)
    metrics = {
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_accuracy": accuracy_score(y_test, preds),
        "test_log_loss": log_loss(y_test, model.predict_proba(X_test)),
    }
    return metrics


if __name__ == "__main__":
    for name, value in evaluate_champion().items():
        print(f"{name}: {value:.4f}")
