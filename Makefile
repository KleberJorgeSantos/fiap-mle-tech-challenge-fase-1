.PHONY: lint test train run install mlflow

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v

train:
	uv run python -m src.train_pipeline

run:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	uv run mlflow ui
