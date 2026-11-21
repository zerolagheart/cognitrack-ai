# Experiment tracking

## What is recorded

Every experiment (`ml/training/trainer.py::run_experiment`) records the following, in three places:

| Field | Where |
|---|---|
| Experiment ID (`<modality_set>__<strategy>__<model>`) | MLflow run name, registry, DB `experiments.experiment_key` |
| Model configuration, implementation (sklearn/xgboost/lightgbm/torch) | MLflow params, registry |
| Dataset version (content fingerprint of recording IDs × split × label) | MLflow param + tag, registry, DB |
| Feature configuration (modalities, exact feature columns, embedding encoder version) | registry `feature_columns`, `modality_columns`; index `embedding_encoder` |
| Training parameters, CV folds, calibration method | MLflow params |
| Random seed, config hash (audio, NLP, embeddings, split, training) | MLflow params, registry |
| CV, validation and test metrics; CIs; curves | MLflow metrics (`cv_*`, `test_*`, `test_participant_*`), registry, `model_comparison.json` |
| Model artifact (`model.joblib`: calibrated model, columns, OOD detector, SHAP background) | `artifacts/models/<id>/`, MLflow artifacts |

The three places are:

* **MLflow** — local SQLite backend `mlruns/mlflow.db`, artifacts in `mlruns/artifacts/`;
* **local registry** — `artifacts/models/registry.json` plus one bundle per experiment;
* **database** — `experiments` and `model_versions` tables, after `cognitrack publish`.

The serving model is also registered in the MLflow model registry as `cognitrack-serving` (best effort). The backend serves from the local registry, so no MLflow server is needed at inference time.

## Browsing

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Filter by tag: `tags.modality_set = "full"`, `tags.synthetic = "True"`. Compare `cv_auroc` across runs. Remember that `test_*` metrics are for reporting only, never for selection.

## Reproducing a run

1. Check out the commit and pull data with `dvc pull`, or regenerate synthetic data, which is deterministic given `project.seed`.
2. `dvc repro` (or `cognitrack run-all`). The DVC lock file pins input hashes.
3. Verify that `registry.json → dataset_version` and each experiment's `config_hash` match the original.

Bitwise reproducibility is expected for the scikit-learn, XGBoost and LightGBM models on the same platform. PyTorch models are seeded but may differ slightly across hardware or library versions.

## Adding an experiment

* **New model:** add a branch to `ml/models/factory.py::build_estimator` and its name to `training.models`.
* **New modality set:** add an entry under `training.modality_sets` in a config file, then run `cognitrack train --config my.yaml`.
* **New encoder:** implement `AudioEncoder`, register it in `ENCODERS`, then run `cognitrack embed --encoder <name>`.

Every new experiment automatically goes through the same CV → calibration → test protocol and is logged identically.
