import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.middleware import LatencyMiddleware
from src.api.schemas import CustomerFeatures, PredictResponse
from src.config import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES
from src.models.trainer import load_model, predict_proba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_state: dict = {"model": None, "pipeline": None, "input_dim": None}

MODEL_PATH = f"{MODEL_DIR}/churn_mlp"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pt_file = Path(MODEL_PATH + ".pt")
    if pt_file.exists():
        try:
            import joblib

            pipeline = joblib.load(MODEL_PATH + "_pipeline.joblib")
            dummy_row = pd.DataFrame(
                [[0] * (len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))],
                columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            )
            transformed = pipeline.transform(dummy_row)
            input_dim = transformed.shape[1]
            model, pipeline = load_model(MODEL_PATH, input_dim)
            _state["model"] = model
            _state["pipeline"] = pipeline
            _state["input_dim"] = input_dim
            logger.info("Model loaded successfully (input_dim=%d)", input_dim)
        except Exception as exc:
            logger.warning("Could not load model: %s", exc)
    else:
        logger.warning("No model file found at %s — /predict will fail", MODEL_PATH)
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Churn Prediction API",
    description="MLP-based churn prediction for telecom customers",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyMiddleware)

# Expõe /metrics para o Prometheus coletar (latência, throughput, status HTTP)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerFeatures):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = {
        "tenure": customer.tenure,
        "MonthlyCharges": customer.monthly_charges,
        "TotalCharges": customer.total_charges,
        "gender": customer.gender,
        "SeniorCitizen": str(customer.senior_citizen),
        "Partner": customer.partner,
        "Dependents": customer.dependents,
        "PhoneService": customer.phone_service,
        "MultipleLines": customer.multiple_lines,
        "InternetService": customer.internet_service,
        "OnlineSecurity": customer.online_security,
        "OnlineBackup": customer.online_backup,
        "DeviceProtection": customer.device_protection,
        "TechSupport": customer.tech_support,
        "StreamingTV": customer.streaming_tv,
        "StreamingMovies": customer.streaming_movies,
        "Contract": customer.contract,
        "PaperlessBilling": customer.paperless_billing,
        "PaymentMethod": customer.payment_method,
    }

    df = pd.DataFrame([row])
    X = _state["pipeline"].transform(df)
    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    proba = float(predict_proba(_state["model"], X_arr)[0])
    prediction = int(proba >= 0.5)

    logger.info("Prediction → proba=%.4f | class=%d", proba, prediction)
    return PredictResponse(churn_probability=proba, churn_prediction=prediction)
