# Contributing

## Ground rules

1. **No real patient data in the repository**, issues, PRs or test fixtures. Tests generate synthetic audio.
2. **No fabricated results.** Numbers in docs, the UI and PRs must come from a pipeline run, and synthetic results must be labelled as synthetic.
3. **No paid services** as required dependencies. Optional integrations must degrade gracefully.
4. **No clinical claims.** Use "statistical signal", "change relative to baseline" or "decision support", never "diagnosis" or "detects dementia".

## Development setup

```bash
pip install -e ".[audio,nlp,dev]" torch
```

```bash
python -m spacy download en_core_web_sm
```

```bash
cd frontend && npm install
```

## Checks (run before every PR — CI runs the same)

```bash
ruff check .
```

```bash
pytest -m "not slow"
```

```bash
pytest -m slow
```

```bash
cd frontend && npm run build
```

`pytest -m slow` runs a tiny synthetic pipeline and the API tests on SQLite (about 3–5 minutes on CPU).

## Code conventions

* Python 3.11, type hints and docstrings that explain *why*. Line length is 110 (ruff).
* Business logic lives in `ml/`. `pipelines/` orchestrates and `backend/app/services` adapts it for HTTP; do not reimplement feature or inference logic in the backend.
* Every parameter goes in `configs/default.yaml`, and paths resolve through `ml.config.resolve_path`; never hardcode local paths.
* Log through `ml.utils.get_logger` with `extra={"ctx": {...}}`, and never log transcripts, audio paths of uploads, or identifiers beyond pseudonyms.
* Missing values stay missing (`NaN` / `None`), never zero-filled outside model pipelines.

## Adding things

| Change | Steps |
|---|---|
| Dataset adapter | Subclass `DatasetAdapter` in `ml/data/adapters.py`, emit canonical columns, register it in `ADAPTERS`, add a config entry, add a test with a fabricated miniature layout |
| Feature | Implement it in the right `ml/` module, register it in `ml/features/registry.py`, document it in `DATA_DICTIONARY.md`, add a unit test with known input/output |
| Model | `ml/models/factory.py::build_estimator`, then `training.models` |
| Encoder | Implement `AudioEncoder`, register it in `ENCODERS`, give it a unique `version` string |
| API endpoint | Route in `backend/app/api/routes.py` (declare a role), logic in a service, schema in `schemas/api.py`, test in `backend/tests` |

## Architecture decisions

For significant changes, add an ADR to `docs/adr/` (copy `0000-template.md`).
