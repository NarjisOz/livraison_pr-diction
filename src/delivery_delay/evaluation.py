"""Model evaluation helpers."""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None


def evaluate_classifier(model, X_test, y_test):
    """Evaluate a classifier and return metrics plus prediction arrays."""

    with _quiet_model_warnings():
        y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            with _quiet_model_warnings():
                y_proba = model.predict_proba(X_test)
            if y_proba is not None and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                metrics["average_precision"] = float(average_precision_score(y_test, y_proba[:, 1]))
        except Exception:
            y_proba = None

    metrics["selection_score"] = metrics.get("roc_auc", metrics["f1_weighted"])
    return metrics, y_pred, y_proba


def evaluate_regressor(model, X_test, y_test):
    """Evaluate a regressor and return robust regression metrics."""

    with _quiet_model_warnings():
        y_pred = model.predict(X_test)
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_test, y_pred)
    else:
        try:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, y_pred)),
    }
    metrics["selection_score"] = metrics["r2"] if np.isfinite(metrics["r2"]) else -metrics["rmse"]
    return metrics, y_pred, None


def leaderboard_from_results(results):
    """Build a sorted leaderboard table from raw model results."""

    rows = []
    for result in results:
        row = {
            "model": result["model_name"],
            "status": result["status"],
            "selection_score": result.get("metrics", {}).get("selection_score"),
            "error": result.get("error", ""),
        }
        row.update(result.get("metrics", {}))
        rows.append(row)

    board = pd.DataFrame(rows)
    if "selection_score" in board.columns:
        board = board.sort_values("selection_score", ascending=False, na_position="last")
    return board.reset_index(drop=True)


@contextmanager
def _quiet_model_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        yield
