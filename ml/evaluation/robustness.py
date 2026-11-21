"""Recording robustness experiments.

Test-partition recordings are perturbed (additive noise at fixed SNRs, gain changes, band-limiting via
resampling, truncation as a corruption proxy, and dropped modalities), features are re-extracted with
the production code path, and the serving model is re-scored. Reported per condition: AUROC, mean
absolute probability shift vs. clean, decision flip rate and abstention rate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ml.config import Config, resolve_path
from ml.evaluation.metrics import binary_metrics
from ml.features.extraction import extract_from_signal
from ml.inference import Predictor
from ml.nlp.conversation import load_turns
from ml.preprocessing.audio import load_audio, resample
from ml.utils import get_logger

log = get_logger(__name__)

Perturbation = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def add_noise(snr_db: float) -> Perturbation:
    def f(sig: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        power = np.mean(sig**2) + 1e-12
        noise = rng.standard_normal(len(sig)) * np.sqrt(power / 10 ** (snr_db / 10))
        return np.clip(sig + noise, -1, 1).astype(np.float32)
    return f


def gain(db: float) -> Perturbation:
    return lambda sig, sr, rng: np.clip(sig * 10 ** (db / 20), -1, 1).astype(np.float32)


def band_limit(target_sr: int) -> Perturbation:
    return lambda sig, sr, rng: resample(resample(sig, sr, target_sr), target_sr, sr)[: len(sig)].astype(np.float32)


def truncate(fraction: float) -> Perturbation:
    return lambda sig, sr, rng: sig[: int(len(sig) * fraction)]


def conditions(cfg: Config) -> dict[str, dict[str, Any]]:
    rcfg = cfg["evaluation"]["robustness"]
    out: dict[str, dict[str, Any]] = {"clean": {}}
    out.update({f"noise_snr_{s}db": {"audio": add_noise(s)} for s in rcfg["snr_db"]})
    out.update({f"gain_{g:+d}db": {"audio": gain(g)} for g in rcfg["gain_db"]})
    out.update({f"bandlimit_{r}hz": {"audio": band_limit(r)} for r in rcfg["resample_hz"]})
    out["truncated_50pct"] = {"audio": truncate(0.5)}
    out["missing_transcript"] = {"drop_transcript": True}
    out["missing_speaker_turns"] = {"drop_turns": True}
    return out


def run_robustness(test_df: pd.DataFrame, cfg: Config, predictor: Predictor) -> dict[str, Any]:
    seed = cfg["project"]["seed"]
    n = min(cfg["evaluation"]["robustness"]["n_recordings"], len(test_df))
    sample = test_df.sample(n, random_state=seed) if n < len(test_df) else test_df
    y = (sample[cfg["split"]["label_column"]] == cfg["split"]["positive_label"]).astype(int).to_numpy()
    sr_target = cfg["audio"]["target_sample_rate"]
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in conditions(cfg)}
    for _, row in sample.iterrows():
        signal, sr, native_sr = load_audio(resolve_path(row["recording_path"]), sr_target)
        turns = load_turns(row["turns_path"])
        transcript = row["transcript"] if isinstance(row["transcript"], str) else None
        for name, cond in conditions(cfg).items():
            rng = np.random.default_rng(seed)
            sig = cond["audio"](signal, sr, rng) if "audio" in cond else signal
            rec = extract_from_signal(
                row["recording_id"], sig, sr, native_sr, cfg,
                transcript=None if cond.get("drop_transcript") else transcript,
                transcript_source="none" if cond.get("drop_transcript") else row["transcript_source"],
                turns=None if cond.get("drop_turns") else turns,
            )
            if rec.error:
                results[name].append({"probability": np.nan, "abstain": True, "error": rec.error})
                continue
            pred = predictor.predict_recording(rec, sig, sr)
            results[name].append({"probability": pred.probability, "abstain": pred.uncertainty["abstain"]})
        log.info("robustness recording done", extra={"ctx": {"recording_id": row["recording_id"]}})

    clean_p = np.array([r["probability"] for r in results["clean"]])
    summary = []
    for name, rows in results.items():
        p = np.array([r["probability"] for r in rows])
        valid = ~np.isnan(p)
        entry: dict[str, Any] = {
            "condition": name, "n": int(len(p)), "n_failed": int((~valid).sum()),
            "abstention_rate": float(np.mean([r["abstain"] for r in rows])),
        }
        if valid.sum() >= 5 and len(np.unique(y[valid])) == 2:
            m = binary_metrics(y[valid], p[valid])
            entry.update(auroc=m["auroc"], balanced_accuracy=m["balanced_accuracy"], brier=m["brier"])
        both = valid & ~np.isnan(clean_p)
        if both.any():
            entry["mean_abs_probability_shift"] = float(np.mean(np.abs(p[both] - clean_p[both])))
            entry["decision_flip_rate"] = float(np.mean((p[both] >= 0.5) != (clean_p[both] >= 0.5)))
        summary.append(entry)
    return {"model": predictor.model_info(), "n_recordings": int(n), "conditions": summary}
