"""Tests for src/data/preprocessing.py — covers build_pipeline, split_data,
get_feature_names and to_numpy."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import NUMERIC_FEATURES
from src.data.preprocessing import build_pipeline, get_feature_names, split_data, to_numpy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_X(n: int = 120) -> pd.DataFrame:
    """Return a DataFrame with all expected feature columns filled with valid data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n).astype(float),
            "MonthlyCharges": rng.uniform(20, 120, size=n),
            "TotalCharges": rng.uniform(0, 8000, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.choice(["0", "1"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
        }
    )


def _synthetic_data(n: int = 120):
    X = _synthetic_X(n)
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_has_preprocessor_step():
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps


def test_pipeline_fit_transform_shape():
    X, y = _synthetic_data()
    pipeline = build_pipeline()
    X_out = pipeline.fit_transform(X)
    assert X_out.shape[0] == len(X)
    assert X_out.shape[1] > len(NUMERIC_FEATURES)  # OHE expands categoricals


def test_pipeline_transform_no_nan():
    X, y = _synthetic_data()
    pipeline = build_pipeline()
    X_out = pipeline.fit_transform(X)
    assert not np.isnan(X_out).any()


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

def test_split_data_total_rows():
    X, y = _synthetic_data(200)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.1)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 200


def test_split_data_test_size_approx():
    X, y = _synthetic_data(200)
    _, _, X_test, _, _, _ = split_data(X, y, test_size=0.2, val_size=0.1)
    assert abs(len(X_test) - 40) <= 2


def test_split_data_consistent_X_y_shapes():
    X, y = _synthetic_data(150)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


# ---------------------------------------------------------------------------
# get_feature_names
# ---------------------------------------------------------------------------

def test_get_feature_names_returns_list():
    X, _ = _synthetic_data()
    pipeline = build_pipeline()
    pipeline.fit(X)
    names = get_feature_names(pipeline)
    assert isinstance(names, list)
    assert len(names) > 0


def test_get_feature_names_includes_numeric():
    X, _ = _synthetic_data()
    pipeline = build_pipeline()
    pipeline.fit(X)
    names = get_feature_names(pipeline)
    for feat in NUMERIC_FEATURES:
        assert feat in names


def test_get_feature_names_count_matches_transform():
    X, _ = _synthetic_data()
    pipeline = build_pipeline()
    X_out = pipeline.fit_transform(X)
    names = get_feature_names(pipeline)
    assert len(names) == X_out.shape[1]


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------

def test_to_numpy_with_ndarray():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_to_numpy_with_list():
    result = to_numpy([[1, 2], [3, 4]])
    assert isinstance(result, np.ndarray)


def test_to_numpy_with_sparse_matrix():
    from scipy.sparse import csr_matrix

    sparse = csr_matrix(np.eye(4))
    result = to_numpy(sparse)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    np.testing.assert_array_equal(result, np.eye(4))
