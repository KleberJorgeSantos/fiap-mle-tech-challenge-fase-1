# Model Card — Churn Prediction MLP

## Model Details

- **Tipo:** Multilayer Perceptron (MLP) para classificação binária
- **Framework:** PyTorch 2.x
- **Arquitetura:** Input → Linear(128) → BatchNorm → ReLU → Dropout(0.3) → Linear(64) → BatchNorm → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(1) → Sigmoid
- **Loss:** Binary Cross-Entropy (BCELoss)
- **Otimizador:** Adam (lr=1e-3)
- **Early stopping:** patience=10 epochs, monitorando val_loss
- **Versão:** 1.0.0
- **Desenvolvido por:** FIAP Tech Challenge Fase 01 — Grupo

## Uso Pretendido

**Casos de uso previstos:**
- Identificar clientes com alto risco de cancelamento do serviço
- Priorizar ações de retenção pela equipe comercial
- Calcular custo esperado de churn por coorte de clientes

**Casos de uso NÃO previstos:**
- Decisões automáticas que afetem clientes sem revisão humana
- Aplicação em domínios fora de telecomunicações

## Dados de Treinamento

- **Dataset:** Telco Customer Churn — IBM Watson Analytics
- **Volume:** 7.043 registros, 21 colunas
- **Período:** Dataset estático (snapshot histórico — sem data de corte conhecida)
- **Split:** 70% treino / 10% validação / 20% teste (estratificado)
- **Churn rate:** ~26% (desbalanceamento moderado)

**Features utilizadas:**
- Numéricas: `tenure`, `MonthlyCharges`, `TotalCharges`
- Categóricas: Gender, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing, Payment Method

**Features removidas (identificadores):**
- `customerID` — identificador sem poder preditivo

## Resultados de Avaliação

*Métricas do modelo servido (ChurnMLP) no conjunto de teste — geradas por `make train` (`models/comparison_table.csv`).*

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.8398 |
| PR-AUC | 0.6312 |
| F1-score | 0.5969 |
| Accuracy | 0.7977 |
| Business Cost (FP=10, FN=100) | R$ 17.520 |

> A MLP atinge o **menor custo de negócio entre os modelos não-triviais** (R$ 17.520),
> à frente de Logistic Regression (R$ 17.780) e Gradient Boosting (R$ 19.180). Tabela
> comparativa completa no README e em `models/comparison_table.csv`.

**Definição de custo de negócio:**
- Falso Negativo (FN): cliente churn não detectado → custo estimado R$100 (perda do contrato)
- Falso Positivo (FP): cliente retido desnecessariamente → custo estimado R$10 (ação de retenção)

## Limitações

- Dataset estático: não captura mudanças de comportamento ao longo do tempo
- Sem features temporais (séries temporais de uso, histórico de chamadas)
- Dados geograficamente concentrados nos EUA — pode não generalizar para outros mercados
- Desbalanceamento de classe (~26% churn) pode afetar o desempenho em populações com taxas muito diferentes

## Estratégia para Classes Desbalanceadas

Dado o desbalanceamento moderado (~26% churn), optou-se por **não** aplicar reamostragem (SMOTE/over/undersampling) nem ponderação de classe. A robustez é garantida por:

1. **Split e validação cruzada estratificados** — preservam a proporção das classes em treino/validação/teste e nos folds da CV.
2. **Métricas adequadas a desbalanceamento** — avaliação por **PR-AUC** e **F1** (não apenas accuracy), além de AUC-ROC.
3. **Análise de custo de negócio assimétrico** — FN custa 10× o FP (R$100 vs. R$10), o que orienta o ajuste de threshold para o ponto de menor custo esperado.

Reamostragem foi considerada desnecessária para esse nível de desbalanceamento; caso o modelo seja aplicado a populações com churn rate muito diferente, recomenda-se reavaliar essa decisão.

## Vieses Identificados

- **Viés de seleção:** dataset representa apenas clientes existentes — clientes que nunca contrataram não estão representados
- **Variáveis sensíveis:** `Gender` e `Senior Citizen` são features do modelo. Recomenda-se auditoria de equidade (fairness) antes de uso em produção
- **Distribuição temporal:** sem data de coleta conhecida; o modelo pode não capturar sazonalidade

## Cenários de Falha

- **Alta carga de FN:** threshold padrão de 0.5 pode ser inadequado — ajustar para 0.3-0.4 dependendo do custo de retenção da operadora
- **Data drift:** mudanças de produto (novos planos, preços) podem degradar rapidamente a performance
- **Features ausentes:** campos obrigatórios na API — requisições incompletas retornam erro 422

## Plano de Monitoramento

| Métrica | Alerta | Frequência |
|---------|--------|------------|
| AUC-ROC em produção | < 0.75 | Semanal |
| Churn rate real vs. previsto | Desvio > 5pp | Mensal |
| Distribuição de features (PSI) | PSI > 0.2 | Mensal |
| Taxa de erros da API | > 1% | Diário |
| Latência p99 | > 500ms | Diário |

**Playbook de resposta:**
1. Alerta de degradação → análise de data drift com PSI por feature
2. PSI alto → retreinar com dados recentes
3. AUC < 0.70 → escalar para revisão de arquitetura e features
