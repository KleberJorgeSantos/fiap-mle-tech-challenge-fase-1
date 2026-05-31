from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure: int = Field(ge=0, description="Months the customer has been with the company")
    monthly_charges: float = Field(ge=0.0, alias="MonthlyCharges")
    total_charges: float = Field(ge=0.0, alias="TotalCharges")
    gender: str
    senior_citizen: int = Field(ge=0, le=1, alias="SeniorCitizen")
    partner: str = Field(alias="Partner")
    dependents: str = Field(alias="Dependents")
    phone_service: str = Field(alias="PhoneService")
    multiple_lines: str = Field(alias="MultipleLines")
    internet_service: str = Field(alias="InternetService")
    online_security: str = Field(alias="OnlineSecurity")
    online_backup: str = Field(alias="OnlineBackup")
    device_protection: str = Field(alias="DeviceProtection")
    tech_support: str = Field(alias="TechSupport")
    streaming_tv: str = Field(alias="StreamingTV")
    streaming_movies: str = Field(alias="StreamingMovies")
    contract: str = Field(alias="Contract")
    paperless_billing: str = Field(alias="PaperlessBilling")
    payment_method: str = Field(alias="PaymentMethod")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
        },
    }


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str = "1.0.0"
