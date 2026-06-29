<div align="center">

# Churn Prediction

### FIAP Tech Challenge — Fase 01

Pipeline completo de Machine Learning para previsão de churn em telecomunicações —
do dado bruto até uma API em produção com monitoramento em tempo real.

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.13+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20us--east--2-FF9900?style=for-the-badge&logo=amazonwebservices&logoColor=white)](http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/docs)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)

<br/>

🎬 **[Vídeo de Demonstração (YouTube)](https://youtu.be/BVvGkcEbC0Q)** &nbsp;|&nbsp;
🚀 **[API — Swagger UI](http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/docs)** &nbsp;|&nbsp;
📊 **[Grafana Dashboard](http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/grafana)** &nbsp;|&nbsp;
🔬 **[Prometheus](http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/prometheus)**

</div>

---

## 📋 Sumário

- [Vídeo de Demonstração](#-vídeo-de-demonstração)
- [Contexto](#-contexto)
- [Resultados](#-resultados)
- [Arquitetura](#-arquitetura)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Quick Start](#-quick-start)
- [Deploy (AWS + Docker)](#-deploy-aws--docker)
- [API Reference](#-api-reference)
- [Monitoramento](#-monitoramento)
- [Boas Práticas](#-boas-práticas)
- [Dependências](#-dependências)

---

## 🎬 Vídeo de Demonstração

Apresentação completa do projeto — arquitetura, pipeline de ML, API em produção na AWS
e monitoramento com Grafana/Prometheus:

▶️ **https://youtu.be/BVvGkcEbC0Q**

[![Vídeo de Demonstração](https://img.youtube.com/vi/BVvGkcEbC0Q/maxresdefault.jpg)](https://youtu.be/BVvGkcEbC0Q)

---

## 🎯 Contexto

Uma operadora de telecomunicações está perdendo clientes. Este projeto constrói um modelo preditivo baseado em **MLP (PyTorch)** para classificar clientes com risco de cancelamento, comparado a baselines clássicos (Scikit-Learn) e rastreado com MLflow.

| | |
|---|---|
| 📦 **Dataset** | Telco Customer Churn (IBM) — 7.043 registros, 19 features |
| 🎯 **Target** | `Churn` — `0` manteve · `1` cancelou |
| 📊 **Churn rate** | ~26% (classes desbalanceadas) |
| 💰 **Custo de negócio** | FP = R$ 10 · FN = R$ 100 |

---

## 🏆 Resultados

O modelo é selecionado pelo **menor custo de negócio**, não pela accuracy — porque um
Falso Negativo (cliente que cancela e passa despercebido) é **10× mais caro** que um
Falso Positivo.

| Modelo | AUC-ROC | PR-AUC | F1 | Accuracy | Custo (R$) |
|---|:---:|:---:|:---:|:---:|:---:|
| 🧠 **MLP (PyTorch)** — *servido* | 0.8398 | 0.6312 | 0.5969 | 0.7977 | **17.520** |
| 📈 Logistic Regression | 0.8424 | 0.6354 | 0.6009 | 0.8048 | 17.780 |
| 🌲 Gradient Boosting | 0.8418 | 0.6566 | 0.5783 | 0.8013 | 19.180 |
| 🌳 Random Forest | 0.8226 | 0.6133 | 0.5585 | 0.7857 | 19.490 |
| 🎲 DummyClassifier (baseline) | 0.5000 | 0.2654 | 0.0000 | 0.7346 | 37.400 |

> **Custo de negócio = `FP × R$10 + FN × R$100`** — penaliza cada tipo de erro pelo seu
> impacto real:
> - **FP (Falso Positivo)** — cliente que **não** ia cancelar é sinalizado como risco e recebe
>   uma ação de retenção desnecessária → custo de **R$ 10** (o gasto da oferta/contato).
> - **FN (Falso Negativo)** — cliente que **ia** cancelar passa despercebido e nenhuma ação é
>   tomada → custo de **R$ 100** (a perda do contrato).
>
> A **MLP tem o menor custo entre os modelos não-triviais** — o critério priorizado dado o
> impacto assimétrico do churn — uma redução de **53%** frente ao baseline. Métricas geradas
> por `make train` (`models/comparison_table.csv`).

---

## 🏗️ Arquitetura

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

### Estratégia de Deploy: **real-time (REST)**

A inferência foi implementada como **serviço REST em tempo real** (FastAPI `/predict`),
e não em modo batch. Justificativa:

| Critério | Por que real-time |
|---|---|
| **Decisão de negócio** | A ação de retenção é tomada *on-demand* (ex.: cliente liga no call center, abre o app) — exige resposta imediata, não um score do dia anterior |
| **Latência** | SLO alvo de **p99 < 500 ms** por requisição, viável com a MLP CPU-only |
| **Volume** | Score individual por cliente; payload pequeno e tráfego moderado cabem num único container |
| **Integração** | Endpoint HTTP é trivial de consumir pelo CRM / sistemas comerciais |
| **Observabilidade** | `/metrics` + Prometheus + Grafana monitoram latência e throughput em tempo real |

> **Alternativa batch** (não adotada): scoring periódico (ex.: semanal) de toda a base,
> gravando resultados numa tabela. Faria sentido para campanhas de marketing em massa,
> mas não atende o caso de uso prioritário de retenção on-demand. O pipeline atual pode
> ser reaproveitado em batch sem alteração — basta chamar `pipeline.transform` + modelo
> sobre um lote em vez de uma linha.

---

## 📁 Estrutura do Projeto

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
└── pyproject.toml
```

---

## ⚡ Quick Start

> Requer **Python 3.11+** e [uv](https://docs.astral.sh/uv/) instalado.

<details>
<summary><strong>🪟 Windows — instalar o <code>make</code> (pré-requisito)</strong></summary>

<br/>

O `make` **não vem instalado por padrão no Windows**. Antes de rodar qualquer comando `make`, instale o **GnuWin32**:

```powershell
# Opção 1 — via winget (recomendado)
winget install GnuWin32.Make

# Opção 2 — installer manual
# https://gnuwin32.sourceforge.net/packages/make.htm
```

Após a instalação, adicione o diretório ao `PATH` do sistema (caso não tenha sido feito automaticamente):

```
C:\Program Files (x86)\GnuWin32\bin
```

> **Como verificar:** abra um novo terminal e execute `make --version`. Se retornar a versão, está pronto.

</details>

Há duas formas de rodar o projeto localmente:

#### Opção A — Apenas a API (uv)

Mais leve, ideal para desenvolvimento. Sobe só a FastAPI, sem monitoramento.

```bash
# 1. Instalar dependências
uv sync --all-extras

# 2. Treinar o modelo (gera os artefatos em models/)
make train

# 3. Subir a API
make run
# → http://localhost:8000/docs
```

#### Opção B — Stack completa (Docker)

Sobe **API + NGINX + Prometheus + Grafana** — a mesma stack que roda em produção.
Requer **Docker Desktop** (veja a nota de Windows abaixo).

```bash
# 1. Instalar dependências e treinar (artefatos são copiados no build da imagem)
uv sync --all-extras
make train

# 2. Subir toda a stack em background
make docker-up

# 3. Verificar os containers e testar
make docker-ps
curl http://localhost/health
# → http://localhost/docs  (API)  ·  /grafana  ·  /prometheus
```

> 📖 Detalhes de cada serviço, URLs de produção e diagnóstico na seção
> [**Deploy (AWS + Docker)**](#-deploy-aws--docker).

### 🛠️ Comandos disponíveis

```bash
# ── Machine Learning ─────────────────────────────────────────────────────────
make train          # treina modelos e registra no MLflow → salva em models/
make run            # sobe a API em http://localhost:8000
make test           # roda todos os testes pytest
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make mlflow         # UI de experimentos em http://localhost:5000

# ── Docker ───────────────────────────────────────────────────────────────────
make docker-up      # sobe toda a stack (API + NGINX + Prometheus + Grafana)
make docker-down    # para e remove os containers (preserva volumes)
make docker-logs    # acompanha logs da API em tempo real
make docker-ps      # lista status de todos os containers
make docker-build   # build das imagens sem subir
make docker-clean   # para containers + remove volumes (reset completo)
```

> 🪟 **Windows:** os comandos `docker` exigem o
> [**Docker Desktop**](https://www.docker.com/products/docker-desktop/) instalado e em
> execução. Ele usa o **WSL2** como backend (kernel Linux necessário para rodar os
> containers) — habilite-o na instalação do Docker Desktop. Os comandos de ML/API
> (`make train`, `make run`, `make test`) rodam direto com o `uv`, sem necessidade de
> Docker/WSL2.

---

## 🚀 Deploy (AWS + Docker)

A mesma stack Docker Compose roda **localmente** e **em produção na AWS EC2**
(`us-east-2`) — toda roteada por NGINX na porta 80.

### 🌐 URLs — local vs. produção

| Serviço | Local | AWS (produção) |
|---|---|---|
| 📄 API + Swagger | http://localhost/docs | http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/docs |
| 💓 Health check | http://localhost/health | http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/health |
| 📊 Grafana | http://localhost/grafana | http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/grafana |
| 🔬 Prometheus | http://localhost/prometheus | http://ec2-3-21-102-246.us-east-2.compute.amazonaws.com/prometheus |

> 🔑 **Grafana** — usuário `admin` / senha `admin`. O dashboard **"Churn Prediction API"**
> já está provisionado: faça login e navegue em **Dashboards → Churn Prediction API**.

#### 🔑 Credenciais de acesso — Grafana

| Campo | Valor |
|---|---|
| Usuário | `users` |
| Senha | `users` |

> Usuário de acesso cadastrado para o professor avaliar o dashboard.

### 🐳 Subir a stack

> **Pré-requisito:** rodar `make train` pelo menos uma vez para gerar
> `models/churn_mlp.pt` e `models/churn_mlp_pipeline.joblib` antes do build.

```bash
# 1. Treinar o modelo (primeira vez ou ao atualizar)
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

### 📝 Notas técnicas

<details>
<summary><strong>Dockerfile</strong></summary>

- **Multi-stage build** — stage `builder` instala deps com `uv`; stage `runtime` copia apenas o `.venv`
- **torch CPU-only** — `pyproject.toml` fixa torch no índice `pytorch-cpu` (~2.5 GB de libs NVIDIA economizados); imagem final ~3.8 GB
- **Usuário não-root** — container roda como `appuser` (UID 1001)
- **Health check interno** — `/health` consultado a cada 30 s; containers `depends_on` aguardam status `healthy`

</details>

<details>
<summary><strong>docker-compose.yml</strong></summary>

- **Rede `internal`** — nenhum container (exceto NGINX) expõe porta para o host; todo tráfego passa pelo proxy
- **Prometheus sub-path** — flags `--web.external-url` e `--web.route-prefix=/` necessários para o UI funcionar via `/prometheus/`
- **Grafana sub-path** — variáveis `GF_SERVER_ROOT_URL` e `GF_SERVER_SERVE_FROM_SUB_PATH=true` habilitam o UI em `/grafana/`
- **Volumes persistentes** — `prometheus_data` e `grafana_data` preservam dados entre `docker compose down/up`

</details>

---

## 📡 API Reference

### `GET /health`

Verifica se a API e o modelo estão carregados.

```json
{ "status": "ok", "model_loaded": true }
```

---

### `POST /predict`

Classifica um cliente quanto ao risco de churn.

Os campos usam os aliases do CSV original (camelCase). `SeniorCitizen` é `int` (0 ou 1).

**Request body:**
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

**Response:**
```json
{ "churn_probability": 0.572, "churn_prediction": 1, "model_version": "1.0.0" }
```

---

### `GET /metrics`

Endpoint Prometheus — expõe throughput, latência e status HTTP de todas as rotas.

> 📖 Documentação interativa completa: **http://localhost/docs**

---

## 📊 Monitoramento

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

## ✅ Boas Práticas

| | Prática |
|:---:|---|
| 🔁 | Seed fixado em `42` para reprodutibilidade total |
| 📊 | Validação cruzada estratificada (5-fold) nos baselines |
| 🛑 | Early stopping no treinamento da MLP |
| 🪵 | Logging estruturado — zero `print()` |
| 🔍 | Linting com `ruff` sem erros |
| 🧪 | Testes automatizados: smoke, schema (pandera), API, métricas, preprocessing |
| 🐳 | Imagem Docker multi-stage, torch CPU-only, usuário não-root |
| 🔒 | `uv.lock` com torch CPU fixado — sem libs NVIDIA na imagem |
| 📡 | Monitoramento out-of-the-box: Prometheus + Grafana provisionados |

---

## 📦 Dependências Principais

| Pacote | Versão | Uso |
|---|:---:|---|
| `torch` (CPU-only) | ≥ 2.3 | MLP de churn — inferência sem GPU |
| `scikit-learn` | ≥ 1.4 | Pipelines de preprocessing e modelos baseline |
| `mlflow` | ≥ 2.13 | Tracking de experimentos e artefatos |
| `fastapi` + `uvicorn` | ≥ 0.111 | API de inferência assíncrona |
| `prometheus-fastapi-instrumentator` | ≥ 7.0 | Endpoint `/metrics` automático |
| `pandera` | ≥ 0.19 | Validação de schema do dataset |
| `uv` | — | Gerenciador de pacotes e lock file |

---

<div align="center">

Desenvolvido como parte do **FIAP Tech Challenge — Fase 01** · 2025

</div>
