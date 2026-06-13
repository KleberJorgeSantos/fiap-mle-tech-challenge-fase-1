# ML Canvas — Churn Prediction

> Baseado no framework ML Canvas (Louis Dorard). Última atualização: 2026-05-10.

---

## 1. Proposta de Valor

Identificar proativamente clientes com alto risco de cancelamento (churn) em uma operadora de telecomunicações, permitindo que a equipe comercial atue com ações de retenção direcionadas antes do cancelamento ocorrer.

**Impacto esperado:** redução do custo total de churn ao priorizar intervenções de retenção (R$10/cliente) sobre a perda do contrato (R$100/cliente).

---

## 2. Usuários e Stakeholders

| Papel | Necessidade |
|-------|-------------|
| Equipe Comercial / CRM | Lista priorizada de clientes em risco para contato proativo |
| Gestão de Produto | Visibilidade do churn por segmento de contrato e serviço |
| Time de Dados / MLOps | Monitoramento de drift e retreinamento periódico |

---

## 3. Definição do Problema

- **Tipo de tarefa:** Classificação binária supervisionada
- **Variável-alvo:** `Churn` — `1` (cliente cancelou) / `0` (cliente ativo)
- **Taxa de churn no dataset:** ~26% (desbalanceamento moderado)
- **Unidade de predição:** Cliente individual (uma linha por cliente)

**Pergunta de negócio:** *"Qual é a probabilidade de este cliente cancelar o serviço nos próximos meses?"*

---

## 4. Decisões que a Predição Apoia

| Decisão | Responsável | Frequência |
|---------|-------------|------------|
| Acionar oferta de retenção (desconto, upgrade) | CRM | Por lote semanal ou on-demand via API |
| Escalar cliente para gerente de conta | Comercial | Clientes com score > 0.7 |
| Segmentar campanha de fidelidade | Marketing | Mensal |

> **Regra de negócio:** threshold padrão = 0.5. Recomenda-se avaliar threshold 0.3–0.4 dado o custo assimétrico (FN 10× mais caro que FP).

---

## 5. Dados de Entrada

**Fonte:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` — IBM Watson Analytics  
**Volume:** 7.043 registros × 21 features originais

| Tipo | Features |
|------|----------|
| Numéricas (3) | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categóricas (16) | `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

**Features removidas:**
- `customerID` — identificador, sem poder preditivo

**Pré-processamento:**
- `TotalCharges`: conversão de string para float (valores em branco → `NaN`)
- `SeniorCitizen`: int (0/1) → string antes do pipeline sklearn
- Numéricas: `SimpleImputer(strategy="median")` → `StandardScaler`
- Categóricas: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder` → 46 features após encoding

---

## 6. Modelos

| Modelo | Tipo | Uso |
|--------|------|-----|
| DummyClassifier | Baseline trivial | Referência mínima |
| LogisticRegression | Linear | Baseline interpretável |
| RandomForest | Ensemble (árvores) | Baseline não-linear |
| GradientBoosting | Ensemble (boosting) | Baseline competitivo |
| **ChurnMLP** | Rede Neural (PyTorch) | **Modelo principal** |

**Arquitetura ChurnMLP:**
```
Input(46) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32)  → ReLU
          → Linear(1)   → Sigmoid
```

**Treinamento:** BCELoss, Adam (lr=1e-3), early stopping (patience=10, monitora val_loss), 5-fold StratifiedKFold nos baselines.

---

## 7. Métricas de Avaliação

### Métricas de ML

| Métrica | Justificativa |
|---------|---------------|
| AUC-ROC | Discriminação geral; robusto ao desbalanceamento |
| PR-AUC | Foco na classe positiva (churn); mais informativo que AUC com classes desbalanceadas |
| F1-score | Equilíbrio precisão/recall no threshold escolhido |
| Accuracy | Contexto geral (interpretar com cautela dado desbalanceamento) |

### Métrica de Negócio

```
Custo total = FP × R$10 + FN × R$100
```

- **FP** (falso alarme): cliente recebe oferta de retenção desnecessariamente → R$10
- **FN** (churn não detectado): contrato perdido sem intervenção → R$100

**Meta:** minimizar custo total, não apenas maximizar accuracy.

---

## 8. Avaliação Offline vs. Online

| Dimensão | Offline | Online |
|----------|---------|--------|
| Dados | Split estratificado 70/10/20 | Dados de produção com label adiado |
| Métricas | AUC-ROC, PR-AUC, F1, Custo | Churn real vs. previsto, ROI de campanhas |
| Frequência | A cada retreinamento | Semanal / Mensal |

---

## 9. Inferência

**Modo:** API REST (FastAPI) — endpoint `/predict`  
**Input:** JSON com os 19 campos do cliente (camelCase, aliases do CSV original)  
**Output:** `{ "churn_probability": 0.73, "churn_prediction": 1, "model_version": "1.0.0" }`  
**Latência alvo:** p99 < 500ms  
**Artefatos carregados no startup:**
- `models/churn_mlp.pt` — state_dict PyTorch
- `models/churn_mlp_pipeline.joblib` — ColumnTransformer sklearn

---

## 10. Feedback e Retreinamento

| Gatilho | Ação |
|---------|------|
| AUC-ROC em produção < 0.75 | Análise de drift → retreinamento |
| PSI de feature > 0.2 | Investigar mudança de distribuição |
| Churn real vs. previsto desvio > 5pp | Retreinamento com dados recentes |
| Novos produtos / mudança de preços | Retreinamento preventivo |

**Rastreamento de experimentos:** MLflow (`experiment = "churn-prediction"`, `mlflow.db` local)

---

## 11. Riscos e Limitações

| Risco | Mitigação |
|-------|-----------|
| Dataset estático — sem temporalidade | Monitorar drift; incorporar features de série temporal se disponível |
| Variáveis sensíveis (`gender`, `SeniorCitizen`) no modelo | Auditoria de fairness antes de produção |
| Desbalanceamento ~26% churn | Usar PR-AUC; ajustar threshold; avaliar class_weight |
| Generalização geográfica (dados dos EUA) | Validar com dados locais antes de deploy em outros mercados |
| Sem revisão humana no loop | Decisões de retenção devem ser revisadas pelo comercial |
