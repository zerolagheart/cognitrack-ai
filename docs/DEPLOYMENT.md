# Deployment and operations

Everything runs locally and costs nothing.

* **Option A:** native Python + Node (SQLite by default).
* **Option B:** Docker Compose (PostgreSQL + backend + nginx frontend).

The ML pipeline can run separately from the web application in either mode.

## Prerequisites

| Option | Requirements |
|---|---|
| Native | Python 3.11+, Node 20+, `libsndfile` (bundled in Windows/macOS wheels; on Debian/Ubuntu `apt install libsndfile1 ffmpeg`) |
| Docker | Docker Desktop or Engine with Compose v2 |

## Option A — native

### 1. Environment and dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

Install the CPU build of PyTorch and the core extras:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[audio,nlp,dev]" torch
```

```bash
python -m spacy download en_core_web_sm
```

Optional extras:

* `pip install -e ".[deep]"` — Whisper / HuBERT / wav2vec 2.0 via Transformers.
* `pip install -e ".[mlops]"` — DVC.

Minimal install (no torch, no spaCy) also works: the MLP falls back to scikit-learn, NLP becomes rule-based, and attention fusion is skipped.

### 2. Configuration

```bash
cp .env.example .env
```

Leaving the API keys empty disables auth for local development.

### 3. Database initialisation

Tables are created automatically on backend start and on `cognitrack publish`. To use PostgreSQL instead of SQLite, set:

```
COGNITRACK_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/cognitrack
```

### 4. Pipeline

Run the whole pipeline in one command (CPU):

```bash
cognitrack run-all --quick --workers 4
```

Or run the stages one by one:

```bash
cognitrack generate-synthetic
```

```bash
cognitrack ingest
```

```bash
cognitrack extract-features --workers 4
```

```bash
cognitrack embed
```

```bash
cognitrack build-dataset
```

```bash
cognitrack train
```

```bash
cognitrack evaluate
```

```bash
cognitrack longitudinal
```

```bash
cognitrack monitor
```

```bash
cognitrack robustness
```

```bash
cognitrack publish
```

DVC equivalent (requires the `mlops` extra):

```bash
dvc repro
```

### 5. Backend

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

API docs are at http://127.0.0.1:8000/docs.

### 6. Frontend

```bash
cd frontend && npm install && npm run dev
```

The dashboard runs at http://localhost:5173. The Vite dev server proxies `/api` to port 8000; point it elsewhere with `VITE_PROXY_TARGET`.

## Option B — Docker Compose

Start the database, backend and frontend:

```bash
docker compose up --build -d
```

Run the synthetic pipeline and publish its results into PostgreSQL:

```bash
docker compose --profile pipeline run --rm pipeline
```

* Dashboard: http://localhost:8080 (nginx serves the build and proxies `/api`).
* API: http://localhost:8000.

`./data`, `./artifacts`, `./mlruns` and `./docs/reports` are bind-mounted, so pipeline outputs and the backend share state. PostgreSQL data lives in the `pgdata` volume. Change `POSTGRES_PASSWORD` in `.env`: the default is for local use only. The database port is not published to the host.

## Pretrained models (explicit, never automatic)

```bash
cognitrack download-models --embeddings --whisper --nlp --spacy
```

Checkpoints go to `artifacts/checkpoints/` (git-ignored). Then:

* set `embeddings.encoder: hubert` (or `wav2vec2`) in a config file and run `cognitrack embed`;
* set `transcription.enabled: true` to transcribe recordings that have no transcript.

If a checkpoint is missing, the stage explains which command to run, and the rest of the system keeps working.

## CPU and GPU execution

* **CPU (default):** everything works; HuBERT-base runs at about 1–2 s per 30 s clip.
* **GPU:** set `embeddings.device: cuda` and `transcription.device: cuda` (or leave them on `auto`). On Colab or Kaggle:
  1. Clone the repository and run `pip install -e ".[deep,audio,nlp]"`.
  2. Upload `data/interim/manifest.parquet` and the audio.
  3. Run `cognitrack embed --encoder hubert`.
  4. Download `data/processed/embedding_cache.parquet` back into your local checkout.

  The cache is keyed by audio hash and encoder version, so local runs reuse the vectors without a GPU. Do not upload restricted data to cloud notebooks unless the dataset agreement allows it.

## Operations

| Task | How |
|---|---|
| Health | `GET /api/v1/health` (Docker healthcheck uses it) |
| Logs | JSON lines on stderr (`docker compose logs backend`) |
| Experiment UI | `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |
| Drift | `cognitrack monitor`; dashboard → System Monitoring; Evidently HTML at `/api/v1/monitoring/evidently` |
| Reset demo | `make clean`, then `cognitrack run-all --quick` |
| Backups | `pg_dump` the `cognitrack` DB; `dvc push` data and artifacts to a self-hosted remote |
