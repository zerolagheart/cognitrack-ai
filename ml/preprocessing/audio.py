"""Audio loading, standardisation and quality assessment.

Every recording goes through :func:`load_audio` (mono, resampled, float32 in [-1, 1]) and
:func:`assess_quality`, which flags silence, clipping, low SNR, too-short and corrupted inputs.
Quality flags are carried with features so downstream models and the dashboard can warn or abstain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ml.utils import get_logger

log = get_logger(__name__)


class AudioLoadError(RuntimeError):
    """Raised for unreadable, empty or corrupted audio."""


@dataclass
class AudioQuality:
    duration_s: float
    native_sample_rate: int
    rms_dbfs: float
    peak: float
    clipping_fraction: float
    silence_fraction: float
    estimated_snr_db: float | None
    flags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not ({"corrupted", "too_short", "silent"} & set(self.flags))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usable"] = self.usable
        return d


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int, int]:
    """Load audio as mono float32 at ``target_sr``.

    Returns ``(signal, target_sr, native_sr)``. Falls back to librosa/audioread for formats libsndfile
    cannot decode. Raises :class:`AudioLoadError` with a descriptive message on failure.
    """
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path.name}")
    try:
        signal, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
        signal = signal.mean(axis=1)
    except Exception:
        try:
            import librosa

            signal, native_sr = librosa.load(str(path), sr=None, mono=True)
        except Exception as exc:
            raise AudioLoadError(f"Could not decode {path.name}: {type(exc).__name__}") from exc
    if signal.size == 0:
        raise AudioLoadError(f"{path.name} contains no samples")
    if not np.all(np.isfinite(signal)):
        raise AudioLoadError(f"{path.name} contains NaN/Inf samples (corrupted)")
    if native_sr != target_sr:
        signal = resample(signal, native_sr, target_sr)
    return signal.astype(np.float32), target_sr, int(native_sr)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    import librosa

    return librosa.resample(signal.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")


def frame_rms_db(signal: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10) -> np.ndarray:
    frame = max(int(sr * frame_ms / 1000), 1)
    hop = max(int(sr * hop_ms / 1000), 1)
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    n = 1 + (len(signal) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(signal[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def assess_quality(signal: np.ndarray, sr: int, native_sr: int, cfg: dict[str, Any]) -> AudioQuality:
    duration = len(signal) / sr
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    clipping = float(np.mean(np.abs(signal) >= cfg["clipping_level"])) if signal.size else 0.0
    rms_db = float(20 * np.log10(np.sqrt(np.mean(signal**2)) + 1e-12))
    frames_db = frame_rms_db(signal, sr, cfg["frame_ms"], cfg["hop_ms"])
    silence_fraction = float(np.mean(frames_db < cfg["silence_floor_dbfs"]))
    # SNR estimate: loud-frame energy vs quiet-frame energy (percentile heuristic, not a calibrated measure).
    snr = float(np.percentile(frames_db, 95) - np.percentile(frames_db, 10)) if frames_db.size > 10 else None

    flags: list[str] = []
    if duration < cfg["min_duration_s"]:
        flags.append("too_short")
    if duration > cfg["max_duration_s"]:
        flags.append("too_long")
    if peak < 1e-4 or rms_db < cfg["silence_floor_dbfs"] - 20:
        flags.append("silent")
    elif rms_db < cfg["silence_floor_dbfs"]:
        flags.append("low_level")
    if clipping > cfg["max_clipping_fraction"]:
        flags.append("clipping")
    if snr is not None and snr < cfg["min_snr_db"]:
        flags.append("low_snr")
    if native_sr < 8000:
        flags.append("low_sample_rate")
    return AudioQuality(duration, native_sr, rms_db, peak, clipping, silence_fraction, snr, flags)


def normalize_loudness(signal: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    """RMS normalisation (applied before embeddings; acoustic level features use the raw signal)."""
    rms = np.sqrt(np.mean(signal**2)) + 1e-12
    gain = 10 ** (target_dbfs / 20) / rms
    return np.clip(signal * gain, -1.0, 1.0).astype(np.float32)


def extract_segments(signal: np.ndarray, sr: int, segments: list[tuple[float, float]]) -> np.ndarray:
    """Concatenate the given (start, end) second ranges, e.g. participant-only speech."""
    parts = [signal[int(s * sr): int(e * sr)] for s, e in segments if e > s]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
