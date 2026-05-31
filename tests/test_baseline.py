"""Tests for src/models/baseline.py — covers train_baselines.

BASELINES é substituído por um único DummyClassifier para manter os testes
rápidos sem perder cobertura do fluxo principal.
"""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.models.baseline import train_baselines

# Apenas DummyClassifier para velocidade (evita treinar RF/GB de 100 árvores em CI)
_FAST_BASELINES = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=42),
}


def _synthetic_data(n: int = 200, n_features: int = 10):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Estrutura do resultado
# ---------------------------------------------------------------------------

def test_train_baselines_returns_dict():
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _FAST_BASELINES):
        results = train_baselines(X, y)
    assert isinstance(results, dict)


def test_train_baselines_keys_match_baselines():
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _FAST_BASELINES):
        results = train_baselines(X, y)
    assert set(results.keys()) == set(_FAST_BASELINES.keys())


def test_train_baselines_each_result_has_metric_keys():
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _FAST_BASELINES):
        results = train_baselines(X, y)
    expected_keys = {"roc_auc", "pr_auc", "f1", "accuracy"}
    for model_name, metrics in results.items():
        assert set(metrics.keys()) == expected_keys, f"Métricas inesperadas para {model_name}"


# ---------------------------------------------------------------------------
# Valores das métricas
# ---------------------------------------------------------------------------

def test_train_baselines_metric_values_are_floats():
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _FAST_BASELINES):
        results = train_baselines(X, y)
    for model_name, metrics in results.items():
        for key, val in metrics.items():
            assert isinstance(val, float), f"{model_name}.{key} não é float"


def test_train_baselines_metric_values_in_range():
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _FAST_BASELINES):
        results = train_baselines(X, y)
    for model_name, metrics in results.items():
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{model_name}.{key}={val} fora de [0,1]"


# ---------------------------------------------------------------------------
# Múltiplos modelos (verifica iteração sobre BASELINES)
# ---------------------------------------------------------------------------

def test_train_baselines_with_two_models():
    _two_baselines = {
        "dummy_freq": DummyClassifier(strategy="most_frequent", random_state=0),
        "dummy_uniform": DummyClassifier(strategy="uniform", random_state=0),
    }
    X, y = _synthetic_data()
    with patch("src.models.baseline.BASELINES", _two_baselines):
        results = train_baselines(X, y)
    assert len(results) == 2
    assert "dummy_freq" in results
    assert "dummy_uniform" in results
