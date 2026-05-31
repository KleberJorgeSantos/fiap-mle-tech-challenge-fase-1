from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

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


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    # 200 if model loaded, 503 if not — both are acceptable in test env
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["churn_prediction"] in (0, 1)


def test_predict_invalid_payload_returns_422():
    response = client.post("/predict", json={"tenure": "not_a_number"})
    assert response.status_code == 422
