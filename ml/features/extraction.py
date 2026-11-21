"""Per-recording feature extraction: quality → (optional ASR) → acoustic → linguistic → conversational.

This module is shared by the batch pipeline (``pipelines/features.py``) and the API's on-demand
processing, so business logic lives in exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.audio.acoustic import MIN_SPEECH_S, extract_acoustic_features, voice_quality_backend
from ml.config import Config, resolve_path
from ml.nlp.conversation import extract_conversational_features, load_turns, participant_segments
from ml.nlp.linguistic import extract_linguistic_features
from ml.nlp.text import load_sentence_model
from ml.preprocessing.audio import AudioLoadError, assess_quality, load_audio
from ml.utils import get_logger

log = get_logger(__name__)


@dataclass
class RecordingFeatures:
    recording_id: str
    features: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)
    transcript: str | None = None
    transcript_source: str = "none"
    error: str | None = None

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"recording_id": self.recording_id, **self.features}
        row["quality_usable"] = bool(self.quality.get("usable", False))
        row["quality_flags"] = ",".join(self.quality.get("flags", []))
        row["quality_snr_db"] = self.quality.get("estimated_snr_db")
        row["quality_clipping_fraction"] = self.quality.get("clipping_fraction")
        row["quality_silence_fraction"] = self.quality.get("silence_fraction")
        row["backends"] = json.dumps(self.backends)
        row["extraction_error"] = self.error
        return row


def sentence_model_for(cfg: Config) -> Any | None:
    if not cfg["nlp"]["use_sentence_transformers"]:
        return None
    return load_sentence_model(cfg["nlp"]["sentence_model"], str(cfg.path("model_cache")))


def extract_from_signal(
    recording_id: str, signal: np.ndarray, sr: int, native_sr: int, cfg: Config,
    transcript: str | None = None, transcript_source: str = "none", turns: list[dict] | None = None,
    sentence_model: Any | None = None,
) -> RecordingFeatures:
    audio_cfg = cfg["audio"]
    out = RecordingFeatures(recording_id=recording_id, transcript=transcript, transcript_source=transcript_source)
    quality = assess_quality(signal, sr, native_sr, audio_cfg)
    out.quality = quality.to_dict()
    out.backends["voice_quality"] = voice_quality_backend(audio_cfg.get("opensmile", "auto"))
    if not quality.usable:
        out.error = f"unusable audio: {','.join(quality.flags)}"
        return out

    if not transcript and cfg["transcription"]["enabled"]:
        from ml.nlp.transcription import TranscriberUnavailable, transcribe

        try:
            asr = transcribe(signal, sr, cfg["transcription"], str(cfg.path("model_cache")))
            out.transcript, out.transcript_source = asr["text"] or None, "whisper"
            out.backends["asr"] = f"{asr['checkpoint']}@{asr['revision']}"
        except TranscriberUnavailable as exc:
            log.warning(str(exc))

    segments = participant_segments(turns)
    acoustic = extract_acoustic_features(signal, sr, audio_cfg, segments)
    if acoustic.get("speech_time_s", 0.0) < MIN_SPEECH_S:
        # Passing the hard quality gates is not enough: with no detectable speech, pause/voicing values would be
        # artefacts, so the recording is marked unusable instead of contributing fabricated zeros.
        out.quality["flags"] = [*out.quality.get("flags", []), "no_speech_detected"]
        out.quality["usable"] = False
        out.error = "unusable audio: no_speech_detected"
        return out
    linguistic, nlp_backends = extract_linguistic_features(
        out.transcript, cfg["nlp"], speech_duration_s=acoustic.get("analysed_duration_s"), sentence_model=sentence_model,
    )
    conversational = extract_conversational_features(turns, sentence_model)
    out.features = {**acoustic, **linguistic, **conversational}
    out.backends.update(nlp_backends)
    out.backends["speaker_turns"] = "dataset" if turns else "none"
    return out


def extract_recording(row: dict[str, Any], cfg: Config, sentence_model: Any | None = None) -> RecordingFeatures:
    """Extract features for one manifest row; failures are captured, not raised."""
    rid = row["recording_id"]
    try:
        signal, sr, native_sr = load_audio(resolve_path(row["recording_path"]), cfg["audio"]["target_sample_rate"])
    except AudioLoadError as exc:
        return RecordingFeatures(recording_id=rid, quality={"usable": False, "flags": ["corrupted"]}, error=str(exc))
    transcript = row.get("transcript") if isinstance(row.get("transcript"), str) else None
    try:
        return extract_from_signal(
            rid, signal, sr, native_sr, cfg, transcript=transcript,
            transcript_source=row.get("transcript_source") or "none", turns=load_turns(row.get("turns_path")),
            sentence_model=sentence_model,
        )
    except Exception as exc:  # keep batch extraction alive; error is recorded per recording
        log.exception("feature extraction failed", extra={"ctx": {"recording_id": rid}})
        return RecordingFeatures(recording_id=rid, error=f"{type(exc).__name__}: {exc}")
