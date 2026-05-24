SEED = 42
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
MLFLOW_EXPERIMENT_CHURN = "churn-prediction"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET = "Churn"

DROP_COLUMNS = ["customerID"]
