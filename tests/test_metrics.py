"""Tests for src/evaluation/metrics.py — covers evaluate_model, cost_analysis
and comparison_table."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import comparison_table, cost_analysis, evaluate_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_arrays(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0, 1, size=n)
    return y_true, y_proba


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_returns_expected_keys():
    y_true, y_proba = _random_arrays()
    metrics = evaluate_model(y_true, y_proba)
    assert set(metrics.keys()) == {"roc_auc", "pr_auc", "f1", "accuracy"}


def test_evaluate_model_values_are_floats():
    y_true, y_proba = _random_arrays()
    metrics = evaluate_model(y_true, y_proba)
    for val in metrics.values():
        assert isinstance(val, float)


def test_evaluate_model_values_in_range():
    y_true, y_proba = _random_arrays()
    metrics = evaluate_model(y_true, y_proba)
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} está fora do intervalo [0,1]"


def test_evaluate_model_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])
    metrics = evaluate_model(y_true, y_proba)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_model_custom_threshold():
    # threshold=0.25 → pred=[1,1,1,1] → accuracy=0.5
    # threshold=0.65 → pred=[0,1,0,0] → accuracy=0.75
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.3, 0.7, 0.4, 0.6])
    metrics_low = evaluate_model(y_true, y_proba, threshold=0.25)
    metrics_high = evaluate_model(y_true, y_proba, threshold=0.65)
    assert metrics_low["accuracy"] != metrics_high["accuracy"]


def test_evaluate_model_with_larger_dataset():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=500)
    y_proba = rng.uniform(0, 1, size=500)
    metrics = evaluate_model(y_true, y_proba)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# ---------------------------------------------------------------------------
# cost_analysis
# ---------------------------------------------------------------------------

def test_cost_analysis_no_errors_zero_cost():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert cost_analysis(y_true, y_pred) == pytest.approx(0.0)


def test_cost_analysis_fp_only():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])  # 3 FP
    cost = cost_analysis(y_true, y_pred, cost_fp=10.0, cost_fn=100.0)
    assert cost == pytest.approx(30.0)


def test_cost_analysis_fn_only():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])  # 3 FN
    cost = cost_analysis(y_true, y_pred, cost_fp=10.0, cost_fn=100.0)
    assert cost == pytest.approx(300.0)


def test_cost_analysis_mixed():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1])  # 1 FP + 1 FN
    cost = cost_analysis(y_true, y_pred, cost_fp=10.0, cost_fn=100.0)
    assert cost == pytest.approx(110.0)


def test_cost_analysis_custom_costs():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])  # 1 FP + 1 FN
    cost = cost_analysis(y_true, y_pred, cost_fp=5.0, cost_fn=50.0)
    assert cost == pytest.approx(55.0)


def test_cost_analysis_returns_float():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    cost = cost_analysis(y_true, y_pred)
    assert isinstance(cost, float)


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------

def test_comparison_table_returns_dataframe():
    results = {
        "model_a": {"roc_auc": 0.80, "pr_auc": 0.70, "f1": 0.75, "accuracy": 0.82},
    }
    df = comparison_table(results)
    assert isinstance(df, pd.DataFrame)


def test_comparison_table_sorted_descending_by_roc_auc():
    results = {
        "weak": {"roc_auc": 0.60, "pr_auc": 0.50, "f1": 0.55, "accuracy": 0.65},
        "best": {"roc_auc": 0.92, "pr_auc": 0.88, "f1": 0.90, "accuracy": 0.93},
        "mid": {"roc_auc": 0.75, "pr_auc": 0.65, "f1": 0.70, "accuracy": 0.78},
    }
    df = comparison_table(results)
    assert list(df.index) == ["best", "mid", "weak"]


def test_comparison_table_has_roc_auc_column():
    results = {
        "only_model": {"roc_auc": 0.80, "pr_auc": 0.72, "f1": 0.76, "accuracy": 0.81},
    }
    df = comparison_table(results)
    assert "roc_auc" in df.columns


def test_comparison_table_index_is_model_name():
    results = {
        "logistic_regression": {"roc_auc": 0.80, "pr_auc": 0.72, "f1": 0.76, "accuracy": 0.81},
    }
    df = comparison_table(results)
    assert "logistic_regression" in df.index
