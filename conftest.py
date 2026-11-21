"""Shared pytest fixtures.

A tiny synthetic cohort is generated once per test session in a temporary directory, run through the
real pipeline stages, and reused by ML, pipeline and API tests. Nothing touches the repository's data/.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml


def _tiny_overrides(root: Path) -> dict:
    return {
        "project": {"datasets": ["synthetic"], "seed": 7},
        "paths": {k: str(root / v) for k, v in {
            "raw": "raw", "interim": "interim", "processed": "processed", "synthetic": "synthetic", "artifacts": "artifacts",
            "models": "artifacts/models", "report_data": "artifacts/reports", "reports": "reports", "mlruns": "mlruns",
            "model_cache": "checkpoints"}.items()},
        "datasets": {"synthetic": {"adapter": "synthetic", "root": str(root / "synthetic")}},
        "synthetic": {"n_participants": 26, "sessions_min": 3, "sessions_max": 4, "session_duration_s": [8.0, 11.0],
                      "missing_session_prob": 0.0},
        "training": {"cv_folds": 3, "models": ["logreg", "lightgbm", "mlp"], "late_fusion_base_models": ["logreg"], "attention_fusion": True,
                     "modality_sets": {"acoustic": ["acoustic"], "linguistic": ["linguistic"], "embedding": ["embedding"],
                                       "full": ["acoustic", "linguistic", "conversational", "embedding"]}},
        "evaluation": {"n_bootstrap": 40, "robustness": {"n_recordings": 4, "snr_db": [10], "gain_db": [-12], "resample_hz": [8000]}},
    }


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cognitrack")


@pytest.fixture(scope="session")
def cfg(workspace: Path):
    from ml import config as config_module

    cfg_path = workspace / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_overrides(workspace)), encoding="utf-8")
    os.environ["COGNITRACK_CONFIG"] = str(cfg_path)
    config_module.get_config.cache_clear()
    return config_module.get_config()


@pytest.fixture(scope="session")
def pipeline_outputs(cfg):
    """Run synthetic generation → ingest → features → embeddings → split → train → evaluate."""
    from ml.training.trainer import train_all
    from pipelines.analytics import run_longitudinal, run_monitoring, write_model_card
    from pipelines.dataset import build_dataset
    from pipelines.evaluation import evaluate
    from pipelines.features import extract_embeddings, extract_features
    from pipelines.ingestion import generate_synthetic, ingest

    generate_synthetic(cfg)
    manifest, quality = ingest(cfg)
    features = extract_features(cfg)
    embeddings = extract_embeddings(cfg)
    df = build_dataset(cfg)
    index = train_all(df, cfg)
    evaluate(cfg, df)
    run_longitudinal(cfg, df)
    run_monitoring(cfg, df)
    write_model_card(cfg)
    return {"manifest": manifest, "quality": quality, "features": features, "embeddings": embeddings, "dataset": df, "index": index}


def tone_bursts(sr: int = 16000, f0: float = 150.0, bursts: int = 6, burst_s: float = 0.4, gap_s: float = 0.5, amp: float = 0.5) -> np.ndarray:
    """Harmonic voiced bursts separated by silence: a controllable speech-like test signal."""
    t = np.arange(int(burst_s * sr)) / sr
    burst = sum((1 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 8))
    burst = amp * burst / np.max(np.abs(burst)) * np.hanning(len(t)) ** 0.3
    gap = np.zeros(int(gap_s * sr))
    rng = np.random.default_rng(0)
    sig = np.concatenate([np.concatenate([burst, gap]) for _ in range(bursts)])
    return (sig + 0.0005 * rng.standard_normal(len(sig))).astype(np.float32)
