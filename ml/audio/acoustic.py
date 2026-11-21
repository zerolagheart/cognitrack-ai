"""Acoustic feature extraction (librosa + optional openSMILE eGeMAPS).

Features describe *how* someone speaks: prosody (F0), voice quality (jitter, shimmer, HNR), energy,
spectral shape, MFCCs, speech/articulation rate and pause behaviour. They are measurement outputs, not
validated clinical indicators; several (jitter/shimmer) are sensitive to microphone and noise.

When speaker turns are available, prosodic, spectral and pause features are computed on the
participant's turns only, so interviewer speech does not contaminate them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks

from ml.audio.vad import detect_speech, pause_statistics
from ml.preprocessing.audio import extract_segments, frame_rms_db
from ml.utils import get_logger

log = get_logger(__name__)

N_MFCC_DEFAULT = 13
MIN_SPEECH_S = 0.5  # below this, prosodic/spectral statistics are too unstable to report


@lru_cache(maxsize=1)
def _opensmile_extractor() -> Any | None:
    try:
        import opensmile

        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals
        )
    except Exception as exc:  # ImportError or binary issues
        log.info("openSMILE unavailable; using librosa voice-quality proxies", extra={"ctx": {"reason": type(exc).__name__}})
        return None


def voice_quality_backend(setting: str) -> str:
    if setting == "off":
        return "librosa_proxy"
    available = _opensmile_extractor() is not None
    if setting == "on" and not available:
        raise RuntimeError("openSMILE requested (audio.opensmile=on) but not installed: pip install -e .[audio]")
    return "opensmile" if available else "librosa_proxy"


def _frames(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    return librosa.util.frame(signal, frame_length=frame, hop_length=hop).T


def pitch_track(signal: np.ndarray, sr: int, fmin: float, fmax: float, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """YIN F0 with an autocorrelation voicing decision.

    Returns ``(f0_hz, harmonicity)`` per frame; ``f0`` is NaN for unvoiced frames. Harmonicity is the
    normalised autocorrelation at the detected period (used for the HNR proxy).
    """
    frame = int(2 ** np.ceil(np.log2(sr / fmin * 2)))
    f0 = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop, center=True)
    frames = _frames(np.pad(signal, frame // 2, mode="reflect") if len(signal) > frame else signal, frame, hop)[: len(f0)]
    harm = np.zeros(len(f0))
    for i, (fr, f) in enumerate(zip(frames, f0, strict=False)):
        lag = int(round(sr / f))
        if lag <= 0 or lag >= len(fr) - 1:
            continue
        a, b = fr[:-lag], fr[lag:]
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12
        harm[i] = max(0.0, float(np.dot(a, b) / denom))
    energy_db = frame_rms_db(signal, sr, frame_ms=frame / sr * 1000, hop_ms=hop / sr * 1000)[: len(f0)]
    if len(energy_db) < len(f0):
        energy_db = np.pad(energy_db, (0, len(f0) - len(energy_db)), constant_values=-120)
    voiced = (harm > 0.6) & (energy_db > np.percentile(energy_db, 95) - 30)
    return np.where(voiced, f0, np.nan), harm


def _voice_quality_proxy(signal: np.ndarray, sr: int, f0: np.ndarray, harm: np.ndarray, hop: int) -> dict[str, float]:
    """Frame-level approximations of jitter / shimmer / HNR (documented as proxies, not Praat-equivalent)."""
    voiced = ~np.isnan(f0)
    out = {"jitter_local": np.nan, "shimmer_local_db": np.nan, "hnr_db": np.nan}
    if voiced.sum() < 5:
        return out
    periods = 1.0 / f0
    consecutive = voiced[1:] & voiced[:-1]
    if consecutive.any():
        dp = np.abs(np.diff(periods))[consecutive]
        out["jitter_local"] = float(np.nanmean(dp) / np.nanmean(periods[voiced]))
    frames = _frames(signal, int(0.03 * sr), hop)[: len(f0)]
    peak = np.max(np.abs(frames), axis=1)[: len(f0)] + 1e-9
    v = voiced[: len(peak)]
    cons = v[1:] & v[:-1]
    if cons.any():
        out["shimmer_local_db"] = float(np.mean(np.abs(20 * np.log10(peak[1:] / peak[:-1]))[cons]))
    r = np.clip(harm[voiced], 1e-4, 0.9999)
    out["hnr_db"] = float(np.mean(10 * np.log10(r / (1 - r))))
    return out


def _voice_quality_opensmile(signal: np.ndarray, sr: int) -> dict[str, float]:
    smile = _opensmile_extractor()
    feats = smile.process_signal(signal, sr).iloc[0]
    return {
        "jitter_local": float(feats.get("jitterLocal_sma3nz_amean", np.nan)),
        "shimmer_local_db": float(feats.get("shimmerLocaldB_sma3nz_amean", np.nan)),
        "hnr_db": float(feats.get("HNRdBACF_sma3nz_amean", np.nan)),
    }


def syllable_nuclei(signal: np.ndarray, sr: int, f0: np.ndarray, hop: int) -> int:
    """Count syllable nuclei as voiced intensity peaks (after de Jong & Wempe, 2009, simplified)."""
    db = frame_rms_db(signal, sr, frame_ms=hop / sr * 1000 * 2.5, hop_ms=hop / sr * 1000)
    n = min(len(db), len(f0))
    if n < 3:
        return 0
    db = np.convolve(db[:n], np.ones(3) / 3, mode="same")
    threshold = np.percentile(db, 95) - 25
    peaks, _ = find_peaks(db, prominence=2.0, distance=max(1, int(0.08 * sr / hop)))
    voiced = ~np.isnan(f0[:n])
    return int(sum(1 for p in peaks if db[p] > threshold and voiced[max(0, p - 2): p + 3].any()))


def extract_acoustic_features(
    signal: np.ndarray, sr: int, cfg: dict[str, Any], participant_segments: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Extract the acoustic feature vector for one recording.

    Args:
        signal: mono float32 signal at ``sr``.
        cfg: the ``audio`` config section.
        participant_segments: optional participant-turn (start, end) times; restricts analysis to them.
    """
    hop = int(sr * cfg["hop_ms"] / 1000)
    total_duration = len(signal) / sr
    if participant_segments:
        speech_signal = extract_segments(signal, sr, participant_segments)
        activity_parts = [
            detect_speech(signal[int(s * sr): int(e * sr)], sr, cfg["frame_ms"], cfg["hop_ms"],
                          cfg["vad_relative_threshold_db"], cfg["min_pause_s"])
            for s, e in participant_segments if e - s > 0.1
        ]
        from ml.audio.vad import SpeechActivity

        activity = SpeechActivity(
            segments=[seg for a in activity_parts for seg in a.segments],
            pauses=[p for a in activity_parts for p in a.pauses],
            frame_voiced=np.zeros(0, bool), hop_s=hop / sr,
        )
        analysis_duration = sum(e - s for s, e in participant_segments)
    else:
        speech_signal = signal
        activity = detect_speech(signal, sr, cfg["frame_ms"], cfg["hop_ms"], cfg["vad_relative_threshold_db"], cfg["min_pause_s"])
        analysis_duration = total_duration

    feats: dict[str, float] = {"recording_duration_s": total_duration, "analysed_duration_s": analysis_duration}
    feats.update(pause_statistics(activity, analysis_duration, cfg["long_pause_s"]))
    if speech_signal.size < sr * 0.5 or activity.speech_time < MIN_SPEECH_S:
        log.warning("too little speech for acoustic analysis", extra={"ctx": {"speech_seconds": round(activity.speech_time, 2)}})
        return feats  # remaining features stay missing rather than being fabricated

    f0, harm = pitch_track(speech_signal, sr, cfg["f0_min_hz"], cfg["f0_max_hz"], hop)
    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size >= 5:
        median = np.median(voiced_f0)
        semitones = 12 * np.log2(voiced_f0 / median)
        feats.update({
            "f0_mean_hz": float(voiced_f0.mean()),
            "f0_median_hz": float(median),
            "f0_std_hz": float(voiced_f0.std()),
            "f0_std_semitones": float(semitones.std()),
            "f0_range_semitones": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
            "voiced_frame_fraction": float(voiced_f0.size / len(f0)),
        })

    backend = voice_quality_backend(cfg.get("opensmile", "auto"))
    vq = _voice_quality_opensmile(speech_signal, sr) if backend == "opensmile" else _voice_quality_proxy(speech_signal, sr, f0, harm, hop)
    feats.update(vq)

    frame = int(sr * cfg["frame_ms"] / 1000)
    n_fft = int(2 ** np.ceil(np.log2(frame)))
    rms = librosa.feature.rms(y=speech_signal, frame_length=n_fft, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    active = rms_db > np.percentile(rms_db, 95) - cfg["vad_relative_threshold_db"]
    feats.update({
        "energy_mean_db": float(rms_db[active].mean()) if active.any() else np.nan,
        "energy_std_db": float(rms_db[active].std()) if active.any() else np.nan,
        "energy_dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)),
    })
    spec = np.abs(librosa.stft(speech_signal, n_fft=n_fft, hop_length=hop)) ** 2
    zcr = librosa.feature.zero_crossing_rate(speech_signal, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=spec)[0]
    m = active[: len(centroid)] if active[: len(centroid)].any() else slice(None)
    feats.update({
        "zcr_mean": float(zcr[m].mean()),
        "spectral_centroid_mean_hz": float(centroid[m].mean()),
        "spectral_centroid_std_hz": float(centroid[m].std()),
        "spectral_bandwidth_mean_hz": float(bandwidth[m].mean()),
        "spectral_rolloff_mean_hz": float(rolloff[m].mean()),
        "spectral_flatness_mean": float(flatness[m].mean()),
    })
    mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=int(cfg.get("n_mfcc", N_MFCC_DEFAULT)))
    mfcc_active = mfcc[:, m] if isinstance(m, np.ndarray) else mfcc
    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i + 1:02d}_mean"] = float(mfcc_active[i].mean())
        feats[f"mfcc_{i + 1:02d}_std"] = float(mfcc_active[i].std())

    n_syll = syllable_nuclei(speech_signal, sr, f0, hop)
    feats["syllable_count"] = float(n_syll)
    feats["speech_rate_syll_per_s"] = n_syll / analysis_duration if analysis_duration > 0 else np.nan
    feats["articulation_rate_syll_per_s"] = n_syll / activity.speech_time if activity.speech_time > 0 else np.nan
    return feats
