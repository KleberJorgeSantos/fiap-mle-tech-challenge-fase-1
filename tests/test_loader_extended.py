"""Tests adicionais para src/data/loader.py — cobre load_raw e get_features_target."""

import os
import tempfile

import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.loader import get_features_target, load_raw


# ---------------------------------------------------------------------------
# Helper: cria CSV sintético compatível com o schema do projeto
# ---------------------------------------------------------------------------

def _make_csv(path: str, n: int = 15) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "customerID": f"CUST_{i:04d}",
                "tenure": i * 4,
                "MonthlyCharges": 50.0 + i * 2,
                "TotalCharges": str(200.0 + i * 80),  # str com espaço ocasional
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": i % 2,  # int 0/1
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "Churn": "Yes" if i % 3 == 0 else "No",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

def test_load_raw_returns_dataframe():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        assert isinstance(df, pd.DataFrame)
    finally:
        os.unlink(path)


def test_load_raw_row_count():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path, n=20)
        df = load_raw(path)
        assert len(df) == 20
    finally:
        os.unlink(path)


def test_load_raw_has_expected_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        assert "customerID" in df.columns
        assert "Churn" in df.columns
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# get_features_target
# ---------------------------------------------------------------------------

def test_get_features_target_returns_tuple():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        result = get_features_target(df)
        assert isinstance(result, tuple) and len(result) == 2
    finally:
        os.unlink(path)


def test_get_features_target_shapes():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path, n=20)
        df = load_raw(path)
        X, y = get_features_target(df)
        assert X.shape[0] == 20
        assert len(y) == 20
    finally:
        os.unlink(path)


def test_get_features_target_drops_customer_id():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        X, _ = get_features_target(df)
        assert "customerID" not in X.columns
    finally:
        os.unlink(path)


def test_get_features_target_senior_citizen_is_str():
    """SeniorCitizen deve ser convertido de int para str antes do pipeline."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        X, _ = get_features_target(df)
        assert X["SeniorCitizen"].dtype == object
    finally:
        os.unlink(path)


def test_get_features_target_churn_is_binary():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        _, y = get_features_target(df)
        assert set(y.unique()).issubset({0, 1})
    finally:
        os.unlink(path)


def test_get_features_target_total_charges_is_numeric():
    """TotalCharges vem como string no CSV e deve ser convertido para float."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        X, _ = get_features_target(df)
        assert pd.api.types.is_numeric_dtype(X["TotalCharges"])
    finally:
        os.unlink(path)


def test_get_features_target_has_all_feature_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _make_csv(path)
        df = load_raw(path)
        X, _ = get_features_target(df)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            assert col in X.columns, f"Coluna '{col}' ausente em X"
    finally:
        os.unlink(path)
