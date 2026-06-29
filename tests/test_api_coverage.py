"""Testes que exercem os caminhos do lifespan e do /predict que dependem do
modelo carregado — fecham a cobertura de src/api/main.py."""

import pandas as pd
import pandera.pandas as pa
from fastapi.testclient import TestClient

import src.api.main as main_module
from src.api.main import app
from src.data.loader import _schema

VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def _reset_state():
    main_module._state.update({"model": None, "pipeline": None, "input_dim": None})


def _make_schema_error() -> pa.errors.SchemaError:
    """Gera um SchemaError real validando dados inválidos com o schema do loader."""
    bad = pd.DataFrame(
        [
            {
                "tenure": -1.0,
                "MonthlyCharges": 1.0,
                "TotalCharges": 1.0,
                "gender": "Male",
                "InternetService": "DSL",
                "Contract": "Month-to-month",
            }
        ]
    )
    try:
        _schema.validate(bad)
    except pa.errors.SchemaError as exc:
        return exc
    raise AssertionError("schema deveria ter falhado")  # pragma: no cover


# ---------------------------------------------------------------------------
# lifespan — caminho feliz (modelo carrega dos artefatos reais)
# ---------------------------------------------------------------------------

def test_lifespan_loads_model_and_predicts():
    _reset_state()
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True

        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] in (0, 1)


# ---------------------------------------------------------------------------
# lifespan — arquivo do modelo ausente (branch else)
# ---------------------------------------------------------------------------

def test_lifespan_no_model_file(monkeypatch):
    _reset_state()
    monkeypatch.setattr(main_module, "MODEL_PATH", "models/__does_not_exist__")
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False


# ---------------------------------------------------------------------------
# lifespan — arquivo existe mas o carregamento falha (branch except)
# ---------------------------------------------------------------------------

def test_lifespan_load_failure(monkeypatch):
    _reset_state()

    def _boom(*args, **kwargs):
        raise RuntimeError("corrupted artifact")

    monkeypatch.setattr("joblib.load", _boom)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict — falha de validação Pandera vira HTTP 422 (branch except)
# ---------------------------------------------------------------------------

def test_predict_pandera_failure_returns_422(monkeypatch):
    _reset_state()
    err = _make_schema_error()

    def _raise(_df):
        raise err

    with TestClient(app) as client:
        # garante modelo carregado para passar do guard 503
        assert client.get("/health").json()["model_loaded"] is True
        monkeypatch.setattr(main_module, "validate_schema", _raise)
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["message"] == "Dados de entrada inválidos"
        assert isinstance(detail["failures"], list)


# ---------------------------------------------------------------------------
# /predict — modelo não carregado retorna 503 (guard)
# ---------------------------------------------------------------------------

def test_predict_without_model_returns_503():
    _reset_state()
    client = TestClient(app)  # sem 'with' → lifespan não roda, modelo fica None
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Model not loaded"
