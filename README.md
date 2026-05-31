# Churn Prediction — FIAP Tech Challenge Fase 01

Pipeline completo de ML para previsão de churn de clientes de telecomunicações, do zero até uma API servida via FastAPI.

## Contexto

Uma operadora de telecomunicações está perdendo clientes. Este projeto constrói um modelo preditivo (MLP com PyTorch) para classificar clientes com risco de cancelamento, comparado com baselines clássicos (Scikit-Learn) e rastreado com MLflow.

**Dataset:** Telco Customer Churn (IBM) — 7.043 registros, 33 features.  
**Target:** `Churn Value` (0 = manteve, 1 = cancelou) — ~26% churn rate.

## Arquitetura

```
tech_challenge_1/
├── data/               # Dataset bruto (.xlsx)
├── src/
│   ├── config.py       # Seeds, paths, feature lists
│   ├── data/           # Loader + preprocessing pipeline
│   ├── models/         # MLP PyTorch + baselines sklearn + trainer
│   ├── evaluation/     # Métricas e análise de custo
│   ├── tracking/       # MLflow utilities
│   ├── api/            # FastAPI: /health, /predict
│   └── train_pipeline.py  # Script principal de treinamento
├── models/             # Artefatos salvos (.pt, .joblib)
├── tests/              # smoke, schema (pandera), API
├── notebooks/          # EDA e comparação de modelos
└── docs/               # Model Card
```

## Setup

Requer Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Execução

```bash
# Treinar modelos e registrar no MLflow
make train

# Subir API de inferência
make run

# Rodar testes
make test

# Linting
make lint

# Visualizar experimentos no MLflow UI
uv run mlflow ui
```

## API

Após `make run`, a API estará disponível em `http://localhost:8000`.

**`GET /health`**
```json
{"status": "ok", "model_loaded": true}
```

**`POST /predict`**
```json
{
  "tenure_months": 12,
  "monthly_charges": 65.5,
  "total_charges": 786.0,
  "gender": "Male",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "No",
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

Resposta:
```json
{"churn_probability": 0.72, "churn_prediction": 1, "model_version": "1.0.0"}
```

Documentação interativa: `http://localhost:8000/docs`

## Modelos

| Modelo | AUC-ROC | PR-AUC | F1 |
|--------|---------|--------|----|
| MLP (PyTorch) | — | — | — |
| Gradient Boosting | — | — | — |
| Random Forest | — | — | — |
| Logistic Regression | — | — | — |
| DummyClassifier | — | — | — |

*Tabela preenchida após `make train`.*

## Boas Práticas

- Seed fixado em `42` para reprodutibilidade total
- Validação cruzada estratificada (5-fold) nos baselines
- Early stopping no treinamento da MLP
- Logging estruturado (sem `print()`)
- Linting com `ruff` sem erros
- Testes automatizados: smoke, schema (pandera), API

## Dependências Principais

- `torch` — MLP
- `scikit-learn` — pipelines e baselines
- `mlflow` — tracking de experimentos
- `fastapi` + `uvicorn` — API de inferência
- `pandera` — validação de schema
- `uv` — gerenciador de pacotes
