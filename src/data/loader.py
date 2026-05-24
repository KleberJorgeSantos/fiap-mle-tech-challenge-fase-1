import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from src.config import CATEGORICAL_FEATURES, DROP_COLUMNS, NUMERIC_FEATURES, TARGET

logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen is already 0/1 int — convert to str for OneHotEncoder consistency
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET].str.strip().str.lower() == "yes").astype(int)

    logger.info("Features shape: %s | Churn rate: %.2f%%", X.shape, y.mean() * 100)
    return X, y


_schema = DataFrameSchema(
    {
        "tenure": Column(float, pa.Check.ge(0), nullable=True),
        "MonthlyCharges": Column(float, pa.Check.ge(0), nullable=True),
        "TotalCharges": Column(float, nullable=True),
        "gender": Column(str),
        "InternetService": Column(str),
        "Contract": Column(str),
    },
    coerce=True,
)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Validating dataframe schema")
    return _schema.validate(df)
