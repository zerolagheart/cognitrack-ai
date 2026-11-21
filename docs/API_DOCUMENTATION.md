# API documentation

Base URL: `http://localhost:8000/api/v1`.

Interactive OpenAPI docs are at `/docs` (Swagger) and `/redoc`; the machine-readable spec is at `/openapi.json`.

## Authentication and roles

Send `X-API-Key: <key>`. Keys are configured per role through environment variables (`COGNITRACK_{VIEWER,RESEARCHER,ADMIN}_API_KEYS`, comma-separated).

| Role | Can |
|---|---|
| viewer | Aggregate endpoints: health, overview, datasets, reports, models, experiments, cohort longitudinal summary, monitoring |
| researcher | Everything a viewer can, plus participant-level data, recordings, uploads, processing, predictions and explanations |
| admin | Everything, plus `POST /admin/publish` |

If **no keys are configured in development**, authentication is disabled. Every request then acts as admin and a startup warning is logged. Production refuses to start without keys.

## Errors

Every error uses one envelope:

```json
{"error": {"code": "unavailable", "message": "No trained model is registered. Run: cognitrack run-all", "detail": {}, "request_id": "a1b2c3d4e5f6"}}
```

| Code | HTTP | Meaning |
|---|---|---|
| `request_validation` | 422 | Bad parameters or body |
| `validation_failed` | 422 | Domain validation failed (bad audio, missing consent, invalid pseudonym, …) |
| `not_found` | 404 | Unknown resource |
| `payload_too_large` | 413 | Upload is larger than `COGNITRACK_MAX_UPLOAD_MB` |
| `unavailable` | 503 | Dependency not ready (model, report, raw audio) |
| `http_error` | 401/403 | Authentication or authorisation failed |
| `internal_error` | 500 | Unexpected error; the response carries a reference ID |

## Endpoints

### System

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/health` | — | DB connectivity, model loaded, serving model, auth status, disclaimer |
| GET | `/overview` | viewer | Counts, data quality, serving model metrics, ablation, recent jobs, synthetic flag |
| GET | `/datasets` | viewer | Configured adapters, availability and licence notes; dataset profile and version |

### Participants

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/participants?search=&limit=&offset=` | researcher | Paginated pseudonymous participants |
| GET | `/participants/{pseudonym}` | researcher | Profile, sessions, recordings (quality, split, latest probability) |
| GET | `/participants/{pseudonym}/features?modality=acoustic\|linguistic\|conversational` | researcher | Session-level feature trends |
| GET | `/participants/{pseudonym}/longitudinal` | researcher | Live baseline-relative analysis (z-scores, EWMA, change points, session comparisons, missing-data indicators) |
| GET | `/participants/{pseudonym}/predictions` | researcher | Prediction history |

### Recordings and inference

| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/recordings/upload` | researcher | Multipart upload; returns `202` and a job (see below) |
| GET | `/recordings/{key}` | researcher | Metadata, quality, transcript, features with modalities, latest prediction |
| GET | `/recordings/{key}/waveform` | researcher | Min/max envelope, speech segments, pauses |
| POST | `/recordings/{key}/process` | researcher | Re-run feature extraction and prediction (needs raw audio) |
| POST | `/recordings/{key}/predict` | researcher | Score stored features with the serving model |
| GET | `/recordings/{key}/explanation` | researcher | Local explanation plus uncertainty assessment |
| POST | `/predict/features` | researcher | Score a supplied feature vector: `{"features": {...}, "quality_flags": []}` |
| GET | `/predictions?limit=&origin=batch\|api` | researcher | Recent predictions |

Upload form fields:

| Field | Required | Notes |
|---|---|---|
| `file` | yes | `.wav`, `.flac`, `.mp3` or `.ogg`; at most `COGNITRACK_MAX_UPLOAD_MB` |
| `participant` | yes | Pseudonym, `[A-Za-z0-9_-]{3,64}` |
| `consent_confirmed` | yes | Must be `true` |
| `session_date` | no | |
| `task` | no | |
| `transcript` | no | |

Uploads are written under a server-generated name, checked by decoding, and then processed in the background. The raw audio is deleted afterwards unless `COGNITRACK_RETAIN_RAW_AUDIO=true`. Poll `GET /monitoring/jobs/{id}` for progress.

A prediction response looks like this:

```json
{
  "probability": 0.71,
  "model_label": "elevated_signal",
  "uncertainty": {"abstain": true, "recommendation": "refer_for_clinician_review",
                  "reasons": ["ensemble disagreement (std 0.24)"], "ensemble_std": 0.24, "ood_percentile": 0.62,
                  "missing_feature_fraction": 0.0},
  "explanation": {"method": "shap", "base_value": -0.1, "output_space": "log-odds",
                  "contributions": [{"feature": "pause_rate_per_min", "shap": 0.42, "value": 18.3, "modality": "acoustic"}],
                  "modality_contributions": {"acoustic": 0.6, "linguistic": 0.2}},
  "disclaimer": "Research decision-support output. Not a diagnosis and not a medical device. ..."
}
```

### Models, reports and monitoring

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/models` | viewer | Registered model versions with summary metrics |
| GET | `/models/serving` | viewer | Serving model info |
| GET | `/experiments` | viewer | All experiments with CV/test metrics |
| GET | `/reports/{type}` | viewer | Latest published report |
| GET | `/longitudinal/cohort` | viewer | Cohort summary and per-participant status |
| GET | `/monitoring/jobs`, `/monitoring/jobs/{id}` | viewer | Processing and pipeline jobs |
| GET | `/monitoring/evidently` | viewer | Evidently drift HTML |
| POST | `/admin/publish` | admin | Load pipeline outputs into the DB and reload the predictor |

Valid `{type}` values for `/reports/{type}`: `data_quality`, `lineage`, `splits`, `feature_extraction`, `model_comparison`, `explainability`, `fairness`, `uncertainty`, `robustness`, `longitudinal`, `drift`, `model_card`.

## Logging

Logs are JSON lines. Each request logs its ID, method, path, status and latency. Query strings and bodies are never logged, and the `X-Request-ID` header is echoed back.
