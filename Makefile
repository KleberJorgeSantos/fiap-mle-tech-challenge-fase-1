.PHONY: lint test train run install mlflow \
        docker-build docker-up docker-down docker-logs docker-ps docker-clean

# ── Desenvolvimento local ─────────────────────────────────────────────────────

install:
	uv sync --all-extras

lint:
	.venv/Scripts/ruff check src/ tests/

format:
	.venv/Scripts/ruff format src/ tests/

test:
	.venv/Scripts/pytest tests/ -v

train:
	.venv/Scripts/python -m src.train_pipeline

run:
	.venv/Scripts/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	.venv/Scripts/mlflow ui

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:   ## Constrói as imagens sem subir os containers
	docker compose build

docker-up:      ## Sobe toda a stack (API + NGINX + Prometheus + Grafana)
	docker compose up -d --build

docker-down:    ## Para e remove os containers (preserva volumes)
	docker compose down

docker-logs:    ## Acompanha logs da API em tempo real
	docker compose logs -f api

docker-ps:      ## Lista status de todos os containers da stack
	docker compose ps

docker-clean:   ## Para containers E remove volumes (reset completo)
	docker compose down -v
