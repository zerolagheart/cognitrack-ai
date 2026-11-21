"""Conversational (turn-taking) features from speaker-attributed turns.

Only computed when the dataset supplies speaker attribution with timestamps (e.g. CHAT bullets, ADReSSo
segmentation files, synthetic turn files). Speaker turns are never inferred from unattributed audio:
if ``turns`` is missing the features are left missing.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ml.config import resolve_path
from ml.nlp.text import adjacent_similarity

PARTICIPANT_CODES = {"PAR", "PATIENT", "P"}
OVERLAP_TOLERANCE_S = 0.1


def load_turns(turns_path: str | None) -> list[dict[str, Any]] | None:
    if not turns_path:
        return None
    path = resolve_path(turns_path)
    if not path.exists():
        return None
    turns = json.loads(path.read_text(encoding="utf-8"))
    turns = [t for t in turns if t.get("start") is not None and t.get("end") is not None and t["end"] > t["start"]]
    return sorted(turns, key=lambda t: t["start"]) or None


def participant_segments(turns: list[dict[str, Any]] | None) -> list[tuple[float, float]] | None:
    if not turns:
        return None
    segs = [(float(t["start"]), float(t["end"])) for t in turns if str(t["speaker"]).upper() in PARTICIPANT_CODES]
    return segs or None


def extract_conversational_features(turns: list[dict[str, Any]] | None, sentence_model: Any | None = None) -> dict[str, float]:
    if not turns or len(turns) < 2:
        return {}
    is_par = [str(t["speaker"]).upper() in PARTICIPANT_CODES for t in turns]
    if not any(is_par) or all(is_par):
        return {}  # needs both participant and interlocutor turns
    span = turns[-1]["end"] - turns[0]["start"]
    minutes = max(span / 60.0, 1e-6)
    durations = np.array([t["end"] - t["start"] for t in turns])
    par_dur = durations[np.array(is_par)]
    other_dur = durations[~np.array(is_par)]

    latencies, gaps, transitions = [], [], 0
    interruptions_by_par = interruptions_of_par = 0
    for prev, cur, prev_par, cur_par in zip(turns, turns[1:], is_par, is_par[1:], strict=False):
        gap = cur["start"] - prev["end"]
        if prev_par != cur_par:
            transitions += 1
            if gap < -OVERLAP_TOLERANCE_S:
                if cur_par:
                    interruptions_by_par += 1
                else:
                    interruptions_of_par += 1
            if cur_par and not prev_par:
                latencies.append(max(gap, 0.0))
        gaps.append(max(gap, 0.0))
    latencies_arr = np.array(latencies) if latencies else np.array([np.nan])
    par_texts = [t.get("text") or "" for t, p in zip(turns, is_par, strict=True) if p]
    total_speech = durations.sum()
    return {
        "turn_count": float(len(turns)),
        "participant_turn_count": float(sum(is_par)),
        "turns_per_min": len(turns) / minutes,
        "speaker_transitions_per_min": transitions / minutes,
        "mean_participant_turn_s": float(par_dur.mean()),
        "mean_interlocutor_turn_s": float(other_dur.mean()),
        "mean_response_latency_s": float(np.nanmean(latencies_arr)) if latencies else float("nan"),
        "median_response_latency_s": float(np.nanmedian(latencies_arr)) if latencies else float("nan"),
        "response_latency_std_s": float(np.nanstd(latencies_arr)) if len(latencies) > 1 else float("nan"),
        "mean_between_turn_silence_s": float(np.mean(gaps)) if gaps else float("nan"),
        "interruptions_per_min": (interruptions_by_par + interruptions_of_par) / minutes,
        "participant_interrupted_rate": interruptions_of_par / max(sum(is_par), 1),
        "conversational_balance": float(par_dur.sum() / total_speech) if total_speech > 0 else float("nan"),
        "topic_continuity": adjacent_similarity(par_texts, sentence_model) if any(par_texts) else float("nan"),
    }
