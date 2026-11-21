# Convenience targets. Every target wraps a documented CLI command (see docs/DEPLOYMENT.md).
PY ?= python
.PHONY: install install-all demo pipeline backend frontend test test-fast lint format docker-up docker-pipeline mlflow-ui clean

install:            ## core + dev dependencies
	$(PY) -m pip install -e ".[audio,dev]"

install-all:        ## everything incl. torch/transformers/spaCy/DVC
	$(PY) -m pip install -e ".[all]"
	$(PY) -m spacy download en_core_web_sm

demo:               ## fast synthetic end-to-end run (CPU, a few minutes)
	cognitrack run-all --quick --workers 4

pipeline:           ## full synthetic run with default config
	cognitrack run-all --workers 4

backend:
	uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm install && npm run dev

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	ruff check .
	cd frontend && npx tsc --noEmit

format:
	ruff check --fix .
	ruff format .

docker-up:
	docker compose up --build

docker-pipeline:
	docker compose --profile pipeline run --rm pipeline

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

clean:
	rm -rf data/interim/* data/processed/* data/synthetic/* artifacts/models artifacts/reports mlruns data/cognitrack.db
