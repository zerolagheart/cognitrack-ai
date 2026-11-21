"""Feature registry: which columns belong to which modality, and which are model inputs.

Raw counts that scale with recording length (word_count, pause_count, ...) are kept for description and
longitudinal display but excluded from model inputs, because recording length varies for reasons
unrelated to cognition (task, interviewer, protocol).
"""

from __future__ import annotations

import pandas as pd

ACOUSTIC_FEATURES: list[str] = [
    "voiced_fraction", "pause_rate_per_min", "mean_pause_s", "median_pause_s", "pause_std_s",
    "long_pause_rate_per_min", "mean_speech_segment_s",
    "f0_mean_hz", "f0_std_semitones", "f0_range_semitones", "voiced_frame_fraction",
    "jitter_local", "shimmer_local_db", "hnr_db",
    "energy_std_db", "energy_dynamic_range_db",
    "zcr_mean", "spectral_centroid_mean_hz", "spectral_centroid_std_hz", "spectral_bandwidth_mean_hz",
    "spectral_rolloff_mean_hz", "spectral_flatness_mean",
    "speech_rate_syll_per_s", "articulation_rate_syll_per_s",
] + [f"mfcc_{i:02d}_{s}" for i in range(1, 14) for s in ("mean", "std")]

LINGUISTIC_FEATURES: list[str] = [
    "type_token_ratio", "mattr", "brunet_w", "honore_r", "mean_sentence_length", "repetition_rate",
    "filler_rate", "pronoun_rate", "pronoun_noun_ratio", "indefinite_term_rate", "content_density",
    "verb_rate", "mean_parse_depth", "clauses_per_sentence", "incomplete_sentence_rate",
    "semantic_coherence", "speech_rate_wpm",
]

CONVERSATIONAL_FEATURES: list[str] = [
    "turns_per_min", "speaker_transitions_per_min", "mean_participant_turn_s", "mean_interlocutor_turn_s",
    "mean_response_latency_s", "median_response_latency_s", "response_latency_std_s",
    "mean_between_turn_silence_s", "interruptions_per_min", "participant_interrupted_rate",
    "conversational_balance", "topic_continuity",
]

DESCRIPTIVE_FEATURES: list[str] = [
    "recording_duration_s", "analysed_duration_s", "speech_time_s", "n_speech_segments", "pause_count",
    "long_pause_count", "total_pause_time_s", "syllable_count", "f0_median_hz", "f0_std_hz", "energy_mean_db",
    "word_count", "unique_word_count", "sentence_count", "turn_count", "participant_turn_count",
]

EMBEDDING_PREFIX = "emb_"

MODALITIES = ("acoustic", "linguistic", "conversational", "embedding")

# Human-readable labels used by reports and the dashboard.
FEATURE_LABELS: dict[str, str] = {
    "speech_rate_wpm": "Speech rate (words/min)",
    "pause_rate_per_min": "Pauses per minute",
    "mean_pause_s": "Mean pause duration (s)",
    "long_pause_rate_per_min": "Long pauses per minute",
    "f0_std_semitones": "Pitch variability (semitones)",
    "f0_range_semitones": "Pitch range (semitones)",
    "type_token_ratio": "Type-token ratio",
    "mattr": "Moving-average TTR",
    "filler_rate": "Filler-word rate",
    "indefinite_term_rate": "Indefinite-term rate (word-finding proxy)",
    "incomplete_sentence_rate": "Incomplete-sentence rate",
    "semantic_coherence": "Semantic coherence proxy",
    "mean_response_latency_s": "Mean response latency (s)",
    "articulation_rate_syll_per_s": "Articulation rate (syll/s)",
    "speech_rate_syll_per_s": "Speech rate (syll/s)",
    "conversational_balance": "Participant share of speech",
}


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(EMBEDDING_PREFIX)]


def modality_columns(modality: str, df: pd.DataFrame | None = None) -> list[str]:
    if modality == "acoustic":
        cols = ACOUSTIC_FEATURES
    elif modality == "linguistic":
        cols = LINGUISTIC_FEATURES
    elif modality == "conversational":
        cols = CONVERSATIONAL_FEATURES
    elif modality == "embedding":
        if df is None:
            raise ValueError("Embedding columns depend on the encoder; pass the feature DataFrame.")
        return embedding_columns(df)
    else:
        raise KeyError(f"Unknown modality '{modality}'. Known: {MODALITIES}")
    return [c for c in cols if df is None or c in df.columns]


def feature_modality(column: str) -> str:
    if column.startswith(EMBEDDING_PREFIX):
        return "embedding"
    for modality, cols in (("acoustic", ACOUSTIC_FEATURES), ("linguistic", LINGUISTIC_FEATURES), ("conversational", CONVERSATIONAL_FEATURES)):
        if column in cols:
            return modality
    return "descriptive"


def columns_for(modalities: list[str], df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each requested modality to its available, non-empty columns."""
    out = {}
    for m in modalities:
        cols = [c for c in modality_columns(m, df) if df[c].notna().any()]
        if cols:
            out[m] = cols
    return out


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))
