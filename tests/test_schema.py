import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.loader import validate_schema


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [12, 24],
            "MonthlyCharges": [65.5, 80.0],
            "TotalCharges": [786.0, 1920.0],
            "gender": ["Male", "Female"],
            "InternetService": ["Fiber optic", "DSL"],
            "Contract": ["Month-to-month", "Two year"],
        }
    )


def test_valid_schema_passes():
    df = _valid_df()
    validated = validate_schema(df)
    assert len(validated) == 2


def test_invalid_negative_tenure_fails():
    df = _valid_df()
    df["tenure"] = [-1, 5]
    with pytest.raises(pa.errors.SchemaError):
        validate_schema(df)


def test_invalid_negative_charges_fails():
    df = _valid_df()
    df["MonthlyCharges"] = [-10.0, 50.0]
    with pytest.raises(pa.errors.SchemaError):
        validate_schema(df)
