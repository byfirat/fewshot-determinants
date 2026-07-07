from typing import Any

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classwise_f1": {
            "0": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
            "1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        },
    }
