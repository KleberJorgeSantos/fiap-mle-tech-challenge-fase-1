# Churn Prediction — FIAP Tech Challenge Fase 01

Pipeline completo de ML para previsão de churn de clientes de telecomunicações, do zero até uma API servida via FastAPI com monitoramento em produção.

## Contexto

Uma operadora de telecomunicações está perdendo clientes. Este projeto constrói um modelo preditivo (MLP com PyTorch) para classificar clientes com risco de cancelamento, comparado com baselines clássicos (Scikit-Learn) e rastreado com MLflow.

**Dataset:** Telco Customer Churn (IBM) — 7.043 registros, 19 features.  
**Target:** `Churn` (0 = manteve, 1 = cancelou) — ~26% churn rate.

---

## Arquitetura

### Pipeline de ML

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
  └─ src/data/loader.py           load_raw() → get_features_target()
  └─ src/data/preprocessing.py    build_pipeline()   [ColumnTransformer → 46 features]
  └─ src/models/baseline.py       train_baselines()  [Dummy, LR, RF, GB — 5-fold CV]
  └─ src/models/mlp.py            ChurnMLP           [128→64→32→1, BatchNorm+Dropout]
  └─ src/models/trainer.py        train_mlp()        [BCELoss, Adam, early stopping]
  └─ src/evaluation/metrics.py    evaluate_model()   [AUC-ROC, PR-AUC, F1, custo negócio]
  └─ src/tracking/mlflow_utils.py                    [rastreia params/métricas/modelos]
  └─ src/train_pipeline.py        main()             [orquestra tudo; salva em models/]
```

### Stack de Serviço e Monitoramento

```
                    ┌──────────────────────────────────────────┐
                    │             NGINX  (porta 80)             │
                    │          reverse proxy / roteador         │
                    └────────┬─────────────┬────────────┬──────┘
                             │             │            │
               /health       │  /grafana/  │  /prome-   │
               /predict      │             │  theus/    │
               /metrics      │             │            │
               /docs         │             │            │
                    ┌────────▼──────┐ ┌────▼──────┐ ┌──▼────────────┐
                    │  FastAPI      │ │  Grafana  │ │  Prometheus   │
                    │  :8000        │ │  :3000    │ │  :9090        │
                    │               │ │           │ │               │
                    │  /metrics ────┼─┼───────────┼─► scrape :15s  │
                    └───────────────┘ └─────▲─────┘ └──────────────┘
                                            │
                                      datasource +
                                      dashboard
                                      provisionados
```

### Estrutura de arquivos

```
tech_challenge_1/
├── data/                        # Dataset bruto (.csv)
├── src/
│   ├── config.py                # Seeds, paths, feature lists (source of truth)
│   ├── data/                    # Loader + preprocessing pipeline
│   ├── models/                  # MLP PyTorch + baselines sklearn + trainer
│   ├── evaluation/              # Métricas e análise de custo de negócio
│   ├── tracking/                # MLflow utilities
│   ├── api/
│   │   ├── main.py              # FastAPI: /health, /predict, /metrics
│   │   ├── schemas.py           # Pydantic CustomerFeatures / PredictResponse
│   │   └── middleware.py        # LatencyMiddleware (loga ms por request)
│   └── train_pipeline.py        # Script principal de treinamento
├── models/                      # Artefatos salvos (.pt, .joblib) — gerados por make train
├── nginx/
│   └── nginx.conf               # Reverse proxy: 80 → API + /grafana/ + /prometheus/
├── monitoring/
│   ├── prometheus.yml           # Scrape config (coleta /metrics a cada 15 s)
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/     # Conecta Grafana → Prometheus automaticamente
│       │   └── dashboards/      # Provider: lê JSONs da pasta abaixo
│       └── dashboards/
│           └── churn_api.json   # Dashboard pronto (req/s, latência p95/p99, erros)
├── tests/                       # smoke, schema, API, preprocessing, métricas
├── notebooks/                   # EDA e comparação de modelos
├── docs/                        # Model Card e ML Canvas
├── Dockerfile                   # Multi-stage build (torch CPU-only, non-root)
├── docker-compose.yml           # API + NGINX + Prometheus + Grafana
└── .dockerignore
```

---

## Setup (desenvolvimento local)

Requer Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

### Comandos disponíveis

```bash
# ML
make train          # treina modelos e registra no MLflow → salva em models/
make run            # sobe a API em http://localhost:8000
make test           # roda todos os testes pytest
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make mlflow         # UI de experimentos em http://localhost:5000

# Docker
make docker-up      # sobe toda a stack (API + NGINX + Prometheus + Grafana)
make docker-down    # para e remove os containers (preserva volumes)
make docker-logs    # acompanha logs da API em tempo real
make docker-ps      # lista status de todos os containers
make docker-build   # build das imagens sem subir
make docker-clean   # para containers + remove volumes (reset completo)
```

---

## Deploy com Docker

> **Pré-requisito:** rodar `make train` pelo menos uma vez para gerar  
> `models/churn_mlp.pt` e `models/churn_mlp_pipeline.joblib` antes do build.

### Fluxo completo

```bash
# 1. Treinar o modelo (primeira vez ou ao atualizar o modelo)
make train

# 2. Subir toda a stack em background
make docker-up

# 3. Verificar se todos os containers subiram saudáveis
make docker-ps

# 4. Testar a API
curl http://localhost/health
```

### Status esperado (`make docker-ps`)

| Container | Imagem | Status |
|---|---|---|
| `api` | `tech_challenge_1-api` | `Up (healthy)` |
| `nginx` | `nginx:1.27-alpine` | `Up` — porta `0.0.0.0:80` |
| `prometheus` | `prom/prometheus:v2.52.0` | `Up` — rede interna |
| `grafana` | `grafana/grafana:11.0.0` | `Up` — rede interna |

### URLs de acesso

| Serviço | Local | GCP (após deploy) |
|---|---|---|
| API + Swagger | http://localhost/docs | `http://<IP_EXTERNO>/docs` |
| Health check | http://localhost/health | `http://<IP_EXTERNO>/health` |
| Grafana | http://localhost/grafana | `http://<IP_EXTERNO>/grafana` |
| Prometheus | http://localhost/prometheus | `http://<IP_EXTERNO>/prometheus` |

Credenciais padrão do Grafana: **admin / admin** — troque em produção.

### Logs e diagnóstico

```bash
# Logs em tempo real por serviço
docker compose logs -f api
docker compose logs -f nginx
docker compose logs -f prometheus
docker compose logs -f grafana

# Inspecionar o health check da API
docker inspect $(docker compose ps -q api) --format '{{json .State.Health}}'
```

### Notas técnicas do Dockerfile

- **Multi-stage build**: stage `builder` instala deps com `uv`, stage `runtime` copia apenas o `.venv`
- **torch CPU-only**: `pyproject.toml` fixa o torch no índice `https://download.pytorch.org/whl/cpu` — sem libs NVIDIA (~2.5 GB economizados); imagem final ~3.8 GB
- **Usuário não-root**: container roda como `appuser` (UID 1001)
- **Health check interno**: `/health` é consultado a cada 30 s pelo Docker Engine; containers `depends_on` aguardam status `healthy`

### Notas técnicas do docker-compose

- **Rede `internal`**: nenhum container (exceto NGINX) expõe porta para o host — todo tráfego passa pelo proxy
- **Prometheus sub-path**: flags `--web.external-url` e `--web.route-prefix=/` necessários para o UI funcionar via `/prometheus/`. O NGINX já remove o prefixo antes de repassar (via `proxy_pass http://prometheus/`)
- **Grafana sub-path**: variáveis `GF_SERVER_ROOT_URL` e `GF_SERVER_SERVE_FROM_SUB_PATH=true` habilitam o UI em `/grafana/`
- **Volumes persistentes**: `prometheus_data` e `grafana_data` preservam dados entre `docker compose down/up`

---

## API

### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

Os campos usam os aliases do CSV original (camelCase). O campo `SeniorCitizen` é `int` (0 ou 1).

```json
{
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
  "PaymentMethod": "Electronic check"
}
```

Resposta:
```json
{"churn_probability": 0.572, "churn_prediction": 1, "model_version": "1.0.0"}
```

### `GET /metrics`
Endpoint Prometheus — expõe throughput, latência e status HTTP de todas as rotas.

Documentação interativa completa: **http://localhost/docs**

---

## Monitoramento

O Grafana sobe **pré-configurado** (zero config manual): datasource e dashboard são
provisionados automaticamente ao iniciar o container.

### Dashboard "Churn Prediction API"

| Painel | Query Prometheus |
|---|---|
| Requisições / s | `rate(http_requests_total[1m])` |
| Latência p95 | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` |
| Erros 5xx / s | `rate(http_requests_total{status=~"5.."}[1m])` |
| Total de predições | `http_requests_total{handler="/predict", method="POST"}` |
| Throughput por endpoint | breakdown por `handler` |
| Latência p50 / p95 / p99 em `/predict` | série temporal |
| Requisições por status HTTP | breakdown por `status` |

### Targets ativos no Prometheus

| Job | Endpoint | Intervalo |
|---|---|---|
| `churn-api` | `http://api:8000/metrics` | 15 s |
| `prometheus` | `http://localhost:9090/metrics` | 15 s |

---

## Modelos

| Modelo | AUC-ROC | PR-AUC | F1 |
|--------|---------|--------|----|
| MLP (PyTorch) | — | — | — |
| Gradient Boosting | — | — | — |
| Random Forest | — | — | — |
| Logistic Regression | — | — | — |
| DummyClassifier | — | — | — |

*Tabela preenchida após `make train`.*

---

## Boas Práticas

- Seed fixado em `42` para reprodutibilidade total
- Validação cruzada estratificada (5-fold) nos baselines
- Early stopping no treinamento da MLP
- Logging estruturado (sem `print()`)
- Linting com `ruff` sem erros
- Testes automatizados: smoke, schema (pandera), API, métricas, preprocessing
- Imagem Docker multi-stage, torch CPU-only, usuário não-root
- `uv.lock` com torch CPU fixado — sem libs NVIDIA na imagem
- Monitoramento out-of-the-box: Prometheus + Grafana provisionados

---

## Dependências Principais

| Pacote | Uso |
|---|---|
| `torch` (CPU-only) | MLP de churn — inferência sem GPU |
| `scikit-learn` | pipelines e baselines |
| `mlflow` | tracking de experimentos |
| `fastapi` + `uvicorn` | API de inferência |
| `prometheus-fastapi-instrumentator` | endpoint `/metrics` |
| `pandera` | validação de schema |
| `uv` | gerenciador de pacotes e lock file |
