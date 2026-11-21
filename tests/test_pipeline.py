"""End-to-end tests on a tiny synthetic cohort (marked slow: ~1-2 minutes on CPU)."""

import numpy as np
import pandas as pd
import pytest

from ml.data.splitting import assert_no_leakage
from ml.embeddings.cache import EmbeddingCache
from ml.embeddings.encoders import build_encoder
from ml.inference import Predictor
from ml.training.registry import ModelRegistry
from ml.utils import read_json

pytestmark = pytest.mark.slow


def test_ingestion_produces_valid_manifest(pipeline_outputs):
    m = pipeline_outputs["manifest"]
    assert pipeline_outputs["quality"]["passed"]
    assert m["is_synthetic"].all() and m["audio_sha256"].notna().all()
    assert (m["recording_duration"] > 0).all()


def test_features_cover_all_modalities(pipeline_outputs):
    f = pipeline_outputs["features"]
    assert {"pause_rate_per_min", "type_token_ratio", "mean_response_latency_s"} <= set(f.columns)
    usable = f[f["quality_usable"]]
    assert usable["speech_rate_wpm"].notna().mean() > 0.9


def test_dataset_split_is_participant_independent(pipeline_outputs):
    df = pipeline_outputs["dataset"]
    assert_no_leakage(df)
    assert any(c.startswith("emb_") for c in df.columns)


def test_embedding_cache_avoids_recompute(cfg, pipeline_outputs):
    cache = EmbeddingCache(cfg.path("processed") / "embedding_cache.parquet")
    enc = build_encoder("spectral_stats", cfg["embeddings"])
    manifest = pipeline_outputs["manifest"]
    assert all(cache.get(h, enc.version) is not None for h in manifest["audio_sha256"])
    n_before = len(cache)
    from pipelines.features import extract_embeddings

    extract_embeddings(cfg)
    assert len(EmbeddingCache(cfg.path("processed") / "embedding_cache.parquet")) == n_before


def test_experiments_record_required_metadata(pipeline_outputs):
    index = pipeline_outputs["index"]
    ok = [r for r in index["experiments"].values() if r["status"] == "ok"]
    assert ok
    for r in ok:
        for key in ("model", "dataset_version", "feature_columns", "seed", "cv", "test", "artifact_path", "experiment_id", "config_hash"):
            assert key in r
        assert 0 <= r["test"]["recording"]["auroc"] <= 1
    assert index["serving"]["experiment_id"] in index["experiments"]


def test_reports_written(cfg, pipeline_outputs):
    rd = cfg.path("report_data")
    for name in ("model_comparison", "explainability", "fairness", "uncertainty", "longitudinal", "drift", "model_card", "data_quality"):
        assert (rd / f"{name}.json").exists(), name
    comp = read_json(rd / "model_comparison.json")
    assert comp["synthetic_data"] and "SYNTHETIC" in comp["banner"]
    assert (cfg.path("reports") / "model_comparison.md").exists()


def test_predictor_inference_and_abstention(cfg, pipeline_outputs):
    predictor = Predictor(cfg)
    row = pipeline_outputs["dataset"].query("split == 'test' and quality_usable").iloc[0]
    pred = predictor.predict_features(row.to_dict(), [])
    assert 0 <= pred.probability <= 1
    assert pred.explanation["method"] in {"shap", "late_fusion_meta_coefficients", "attention_weights", "none"}
    assert "not a diagnosis" in pred.disclaimer.lower()
    bad = predictor.predict_features(row.to_dict(), ["clipping"])
    assert bad.uncertainty["abstain"] and any("audio quality" in r for r in bad.uncertainty["reasons"])
    empty = predictor.predict_features({}, [])
    assert empty.uncertainty["abstain"]  # all features missing


def test_model_bundle_roundtrip(cfg, pipeline_outputs):
    reg = ModelRegistry(cfg.path("models"))
    serving, members = reg.serving()
    df = pipeline_outputs["dataset"]
    p = serving.predict_proba(df[df["quality_usable"]].head(5))
    assert p.shape == (5,) and np.all((p >= 0) & (p <= 1))


def test_robustness_runs(cfg, pipeline_outputs):
    from pipelines.analytics import run_robustness_report

    report = run_robustness_report(cfg, pipeline_outputs["dataset"])
    names = {c["condition"] for c in report["conditions"]}
    assert {"clean", "noise_snr_10db", "missing_transcript", "truncated_50pct"} <= names
    clean = next(c for c in report["conditions"] if c["condition"] == "clean")
    assert clean.get("mean_abs_probability_shift", 0) == pytest.approx(0.0, abs=1e-9)


def test_longitudinal_synthetic_validation(cfg, pipeline_outputs):
    report = read_json(cfg.path("report_data") / "longitudinal.json")
    assert report["cohort_summary"]["n_participants"] == pipeline_outputs["manifest"]["participant_id"].nunique()
    assert "synthetic_engine_validation" in report


def test_predictions_cover_usable_recordings(cfg, pipeline_outputs):
    preds = pd.read_parquet(cfg.path("processed") / "predictions.parquet")
    df = pipeline_outputs["dataset"]
    assert set(preds["recording_id"]) == set(df.loc[df["quality_usable"].astype(bool), "recording_id"])
