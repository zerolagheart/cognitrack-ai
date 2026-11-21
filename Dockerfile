# Backend + ML pipeline image (CPU). Pretrained checkpoints are NOT baked in; mount artifacts/checkpoints.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 COGNITRACK_ROOT=/app
# libsndfile for soundfile, ffmpeg for compressed audio decoding, libgomp for LightGBM/XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Dependency layer (cached unless pyproject changes). CPU-only torch keeps the image small.
COPY pyproject.toml README.md ./
RUN mkdir -p ml pipelines backend && touch ml/__init__.py pipelines/__init__.py backend/__init__.py \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[audio]" \
    && pip uninstall -y cognitrack-ai

COPY configs ./configs
COPY ml ./ml
COPY pipelines ./pipelines
COPY backend ./backend
RUN pip install --no-deps . && useradd --create-home --uid 10001 cognitrack \
    && mkdir -p data artifacts mlruns docs/reports && chown -R cognitrack:cognitrack /app
USER cognitrack

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
