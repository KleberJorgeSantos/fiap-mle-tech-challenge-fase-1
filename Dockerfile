# ============================================================
# Dockerfile – Churn Prediction API
# ============================================================
# Pré-requisito: rodar `make train` para gerar os artefatos
#   em models/churn_mlp.pt e models/churn_mlp_pipeline.joblib
#
# Build:   docker build -t churn-api .
# Run:     docker run --rm -p 8000:8000 churn-api
# Teste:   curl http://localhost:8000/health
# ============================================================

# ── Stage 1: builder — instala dependências Python ──────────
FROM python:3.11-slim AS builder

# IMPORTANTE: WORKDIR deve ser /app (mesmo que o runtime) para que
# os shebangs do venv apontem para o path correto na imagem final.
WORKDIR /app

# Instala uv (gerenciador de pacotes rápido)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copia manifesto e lock file — o lock garante torch CPU-only
# (sem NVIDIA CUDA libs, economiza ~2.5 GB na imagem final)
COPY pyproject.toml uv.lock ./

# Instala todas as dependências de produção usando o lock file
# O uv.lock já resolve torch para pytorch-cpu index (CPU-only)
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime — imagem final enxuta ──────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copia o venv populado do stage de build
COPY --from=builder /app/.venv /app/.venv

# Ativa o venv e configura variáveis de ambiente
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copia código-fonte da aplicação
COPY src/ ./src/

# Copia artefatos de modelo pré-treinado
# (execute `make train` antes de `docker build`)
# Se models/ estiver vazio, a API sobe mas /predict retorna HTTP 503
COPY models/ ./models/

# Usuário não-root — boa prática de segurança em produção
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check interno usado por ECS/App Runner/ALB
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
