"""Energy-based voice activity detection and pause analysis.

A deliberately transparent VAD: frames whose RMS energy exceeds a noise-adaptive threshold (between the
recording's noise floor and its loud frames) are voiced. Short gaps are bridged and short bursts removed.
This is robust enough for clean interview audio; noisy field recordings would merit a learned VAD
(e.g. Silero), which can be swapped in behind :func:`detect_speech`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.preprocessing.audio import frame_rms_db

NOISE_FLOOR_FRACTION = 0.5
MIN_MARGIN_DB = 6.0


@dataclass
class SpeechActivity:
    segments: list[tuple[float, float]]  # voiced (start, end) in seconds
    pauses: list[float]  # pause durations >= min_pause between voiced segments
    frame_voiced: np.ndarray
    hop_s: float

    @property
    def speech_time(self) -> float:
        return float(sum(e - s for s, e in self.segments))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) indices of True runs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0], strict=True))


def detect_speech(
    signal: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10, relative_threshold_db: float = 28.0,
    min_pause_s: float = 0.25, min_speech_s: float = 0.05, absolute_floor_dbfs: float = -60.0,
) -> SpeechActivity:
    hop_s = hop_ms / 1000
    db = frame_rms_db(signal, sr, frame_ms, hop_ms)
    if db.size == 0:
        return SpeechActivity([], [], np.zeros(0, bool), hop_s)
    ref = np.percentile(db, 95)
    # Noise-adaptive threshold: halfway between the noise floor (10th percentile) and speech level, never
    # below ref - relative_threshold_db. A fixed offset from the speech level alone falls under the noise
    # floor at moderate SNR (~20 dB) and labels background noise as speech, erasing every pause.
    floor = np.percentile(db, 10)
    threshold = min(max(ref - relative_threshold_db, floor + NOISE_FLOOR_FRACTION * (ref - floor)), ref - MIN_MARGIN_DB)
    voiced = (db > threshold) & (db > absolute_floor_dbfs)

    # Bridge gaps shorter than min_pause (intra-word silences / plosive closures).
    bridge = int(round(min_pause_s / hop_s))
    for start, end in _runs(~voiced):
        if start > 0 and end < len(voiced) and end - start < bridge:
            voiced[start:end] = True
    # Remove isolated bursts shorter than min_speech (clicks).
    min_frames = max(1, int(round(min_speech_s / hop_s)))
    for start, end in _runs(voiced):
        if end - start < min_frames:
            voiced[start:end] = False

    segments = [(s * hop_s, e * hop_s + frame_ms / 1000 - hop_s) for s, e in _runs(voiced)]
    pauses = [segments[i + 1][0] - segments[i][1] for i in range(len(segments) - 1)]
    pauses = [p for p in pauses if p >= min_pause_s]
    return SpeechActivity(segments, pauses, voiced, hop_s)


def pause_statistics(activity: SpeechActivity, total_duration: float, long_pause_s: float) -> dict[str, float]:
    pauses = np.asarray(activity.pauses, dtype=float)
    minutes = max(total_duration / 60.0, 1e-6)
    speech = activity.speech_time
    return {
        "speech_time_s": speech,
        "voiced_fraction": speech / total_duration if total_duration > 0 else np.nan,
        "n_speech_segments": float(len(activity.segments)),
        "pause_count": float(len(pauses)),
        "pause_rate_per_min": len(pauses) / minutes,
        "mean_pause_s": float(pauses.mean()) if pauses.size else 0.0,
        "median_pause_s": float(np.median(pauses)) if pauses.size else 0.0,
        "pause_std_s": float(pauses.std()) if pauses.size > 1 else 0.0,
        "long_pause_count": float(np.sum(pauses >= long_pause_s)),
        "long_pause_rate_per_min": float(np.sum(pauses >= long_pause_s)) / minutes,
        "total_pause_time_s": float(pauses.sum()),
        "mean_speech_segment_s": speech / len(activity.segments) if activity.segments else 0.0,
    }
