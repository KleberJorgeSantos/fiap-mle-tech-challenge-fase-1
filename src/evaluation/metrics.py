import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    logger.info("Metrics: %s", metrics)
    return metrics


def cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 100.0,
) -> float:
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total_cost = fp * cost_fp + fn * cost_fn
    logger.info("Cost analysis → FP=%d, FN=%d, total_cost=%.2f", fp, fn, total_cost)
    return total_cost


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("roc_auc", ascending=False)
