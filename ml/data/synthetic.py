"""Synthetic demonstration dataset generator.

Generates *clearly labelled synthetic* longitudinal interview recordings so the full platform can be
exercised without access to restricted clinical corpora. Nothing produced here is clinical evidence:
group differences and trajectories are programmed in by construction (see ``TRAIT_SPEC``), so any model
that "detects" them is only demonstrating that the pipeline runs end to end.

Each participant has latent communication traits (articulation rate, pause behaviour, pitch variability,
lexical diversity, response latency, topic coherence). A subset follows a programmed "change"
trajectory in which traits drift across sessions. For each session the generator synthesises:

* a 16 kHz mono WAV of an interviewer/participant exchange built from harmonic voiced segments with
  formant-like spectral envelopes, F0 contours, jitter/shimmer perturbations, pauses and background noise;
* a participant transcript produced from a small template grammar (fillers, repetitions, indefinite
  "word-finding" terms, incomplete sentences and off-topic sentences are trait-controlled);
* a speaker-attributed turn file with start/end times (simulating a CHAT-style annotation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from ml.data.schema import LABEL_CONTROL, LABEL_IMPAIRED
from ml.utils import get_logger, relpath

log = get_logger(__name__)

SYNTHETIC_NOTICE = (
    "SYNTHETIC DEMONSTRATION DATA - programmatically generated, contains no real participants, "
    "and must never be interpreted as clinical evidence."
)

# trait: (control mean, impaired mean, between-person sd, within-person session sd)
TRAIT_SPEC: dict[str, tuple[float, float, float, float]] = {
    "articulation_wpm": (165.0, 140.0, 16.0, 6.0),
    "phrase_words": (6.5, 4.8, 1.0, 0.5),
    "pause_mean_s": (0.55, 0.85, 0.15, 0.06),
    "long_pause_prob": (0.05, 0.14, 0.04, 0.02),
    "f0_var_st": (2.6, 1.9, 0.45, 0.2),
    "zipf_a": (1.05, 1.3, 0.12, 0.05),
    "filler_rate": (0.03, 0.075, 0.02, 0.01),
    "indefinite_rate": (0.03, 0.08, 0.02, 0.01),
    "latency_s": (0.75, 1.3, 0.25, 0.12),
    "on_topic_prob": (0.93, 0.8, 0.05, 0.03),
    "incomplete_prob": (0.05, 0.16, 0.04, 0.02),
    "repetition_prob": (0.03, 0.08, 0.02, 0.01),
}
TRAIT_BOUNDS: dict[str, tuple[float, float]] = {
    "articulation_wpm": (80, 230),
    "phrase_words": (2.5, 10),
    "pause_mean_s": (0.2, 2.0),
    "long_pause_prob": (0.0, 0.5),
    "f0_var_st": (0.6, 5.0),
    "zipf_a": (0.8, 1.8),
    "filler_rate": (0.0, 0.25),
    "indefinite_rate": (0.0, 0.25),
    "latency_s": (0.25, 4.0),
    "on_topic_prob": (0.4, 1.0),
    "incomplete_prob": (0.0, 0.5),
    "repetition_prob": (0.0, 0.3),
}

TOPICS: dict[str, dict[str, list[str]]] = {
    "kitchen_scene": {
        "nouns": ["woman", "boy", "girl", "sink", "water", "plate", "window", "cupboard", "jar", "stool",
                  "floor", "curtain", "dish", "cup", "garden", "counter", "towel", "faucet", "kitchen", "cookie",
                  "shelf", "apron", "door", "tree", "path", "lid", "bowl", "spoon"],
        "verbs": ["is washing", "is reaching for", "is standing on", "is drying", "is falling off",
                  "is overflowing onto", "is holding", "is looking at", "is taking", "is spilling over",
                  "is grabbing", "is pointing at", "is climbing onto", "is cleaning"],
        "adjs": ["big", "small", "open", "wet", "tall", "full", "empty", "round", "dirty", "clean", "wooden"],
    },
    "daily_routine": {
        "nouns": ["breakfast", "coffee", "newspaper", "bus", "garden", "dog", "market", "friend", "radio",
                  "morning", "kettle", "walk", "neighbour", "park", "bread", "tea", "car", "shop", "letter",
                  "television", "book", "lunch", "phone", "grandson", "church", "library"],
        "verbs": ["usually make", "always read", "take", "walk to", "visit", "listen to", "buy", "drink",
                  "call", "water", "feed", "watch", "cook", "clean"],
        "adjs": ["early", "quiet", "long", "busy", "warm", "fresh", "local", "nice", "short", "old"],
    },
}
OFF_TOPIC = ["my sister lived near the sea", "we had a red car once", "the weather was cold last winter",
             "i used to work at the post office", "my father liked fishing", "there was a dance on saturday"]
INDEFINITES = ["thing", "stuff", "something", "that one", "whatsit", "the thing there"]
FILLERS = ["um", "uh", "er", "you know", "well"]
PROMPTS = {
    "kitchen_scene": ["tell me everything you see happening in this picture",
                      "what else is going on in the picture", "anything else you notice", "and over here"],
    "daily_routine": ["can you tell me about a typical morning", "what do you usually do after that",
                      "what about the afternoon", "tell me more about that"],
}
# (F1, F2, F3) formant centres in Hz for a few vowel-like spectral envelopes.
VOWELS = [(730, 1090, 2440), (530, 1840, 2480), (270, 2290, 3010), (570, 840, 2410), (300, 870, 2240)]


@dataclass
class SyntheticConfig:
    n_participants: int = 72
    sessions_min: int = 3
    sessions_max: int = 6
    change_fraction: float = 0.4
    session_duration_s: tuple[float, float] = (18.0, 30.0)
    sample_rate: int = 16000
    session_interval_days: tuple[int, int] = (60, 120)
    missing_session_prob: float = 0.08
    noise_snr_db: tuple[float, float] = (15.0, 35.0)
    seed: int = 42

    @classmethod
    def from_config(cls, cfg: dict[str, Any], seed: int) -> SyntheticConfig:
        return cls(
            n_participants=int(cfg["n_participants"]),
            sessions_min=int(cfg["sessions_min"]),
            sessions_max=int(cfg["sessions_max"]),
            change_fraction=float(cfg["change_fraction"]),
            session_duration_s=tuple(cfg["session_duration_s"]),
            sample_rate=int(cfg["sample_rate"]),
            session_interval_days=tuple(cfg["session_interval_days"]),
            missing_session_prob=float(cfg["missing_session_prob"]),
            noise_snr_db=tuple(cfg["noise_snr_db"]),
            seed=seed,
        )


# --------------------------------------------------------------------------------------------------
# Signal synthesis
# --------------------------------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, n: int, smooth: int) -> np.ndarray:
    noise = rng.standard_normal(n + smooth)
    kernel = np.hanning(max(smooth, 3))
    kernel /= kernel.sum()
    out = np.convolve(noise, kernel, mode="same")[:n]
    return out / (out.std() + 1e-9)


def _voiced(
    rng: np.random.Generator, dur: float, f0_start: float, f0_end: float, sr: int, amp: float,
    jitter: float = 0.01, shimmer: float = 0.06,
) -> np.ndarray:
    """Harmonic voiced segment with vowel-like formant envelope, jitter and shimmer."""
    n = max(int(dur * sr), 16)
    f0 = np.linspace(f0_start, f0_end, n) * (1.0 + jitter * _smooth_noise(rng, n, int(sr / max(f0_start, 60))))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    f1, f2, f3 = VOWELS[rng.integers(len(VOWELS))]
    mean_f0 = float(np.mean(f0))
    n_harm = int(min(30, (sr / 2 - 200) // mean_f0))
    sig = np.zeros(n)
    for k in range(1, n_harm + 1):
        fk = k * mean_f0
        env = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw in ((f1, 90), (f2, 120), (f3, 170)))
        gain = (0.15 + env) / k**0.8
        sig += gain * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    attack = min(int(0.02 * sr), n // 4)
    envelope = np.ones(n)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-attack:] = np.linspace(1, 0, attack)
    envelope *= 1.0 + shimmer * _smooth_noise(rng, n, int(0.01 * sr))
    sig *= envelope
    sig += 0.02 * rng.standard_normal(n) * envelope  # aspiration
    return amp * sig / (np.max(np.abs(sig)) + 1e-9)


def _word_audio(rng: np.random.Generator, n_syll: int, syll_dur: float, f0: float, f0_var_st: float,
                sr: int, amp: float) -> np.ndarray:
    parts = []
    for _ in range(n_syll):
        st = rng.normal(0, f0_var_st)
        start = f0 * 2 ** (st / 12)
        end = start * 2 ** (rng.normal(-0.4, 0.6) / 12)
        parts.append(_voiced(rng, syll_dur * rng.uniform(0.8, 1.2), start, end, sr, amp * rng.uniform(0.7, 1.0)))
        parts.append(np.zeros(int(rng.uniform(0.005, 0.02) * sr)))
    return np.concatenate(parts)


def _colored_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    spectrum /= np.sqrt(np.maximum(freqs, 1.0 / n))  # pink-ish
    noise = np.fft.irfft(spectrum, n)
    return noise / (noise.std() + 1e-12)


def _syllables(word: str) -> int:
    vowels = "aeiouy"
    count, prev = 0, False
    for ch in word.lower():
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    return max(1, count)


# --------------------------------------------------------------------------------------------------
# Language synthesis
# --------------------------------------------------------------------------------------------------


def _zipf_choice(rng: np.random.Generator, items: list[str], a: float) -> str:
    ranks = np.arange(1, len(items) + 1)
    p = 1.0 / ranks**a
    p /= p.sum()
    return items[rng.choice(len(items), p=p)]


def _sentence(rng: np.random.Generator, topic: str, traits: dict[str, float], vocab_order: list[str]) -> tuple[str, bool]:
    """Return (sentence, was_incomplete)."""
    if rng.random() > traits["on_topic_prob"]:
        words = OFF_TOPIC[rng.integers(len(OFF_TOPIC))].split()
    else:
        t = TOPICS[topic]
        noun = _zipf_choice(rng, vocab_order, traits["zipf_a"])
        noun2 = _zipf_choice(rng, vocab_order, traits["zipf_a"])
        if rng.random() < traits["indefinite_rate"] * 3:
            noun2 = INDEFINITES[rng.integers(len(INDEFINITES))]
        subj = "she" if rng.random() < 0.15 + traits["indefinite_rate"] * 2 else f"the {noun}"
        verb = t["verbs"][rng.integers(len(t["verbs"]))]
        adj = t["adjs"][rng.integers(len(t["adjs"]))]
        target = int(np.clip(round(rng.normal(traits["phrase_words"], 1.2)), 2, 12))
        words = f"{subj} {verb} the {adj} {noun2}".split()
        if target > len(words):
            extra = ["and", "the", _zipf_choice(rng, vocab_order, traits["zipf_a"])]
            words += extra[: target - len(words)]
        words = words[:target] if target < len(words) else words
    incomplete = rng.random() < traits["incomplete_prob"]
    if incomplete and len(words) > 2:
        words = words[: rng.integers(2, len(words))]
    out: list[str] = []
    for w in words:
        if rng.random() < traits["filler_rate"]:
            out.append(FILLERS[rng.integers(len(FILLERS))])
        out.append(w)
        if rng.random() < traits["repetition_prob"]:
            out.append(w)
    return " ".join(out), incomplete


# --------------------------------------------------------------------------------------------------
# Participants and sessions
# --------------------------------------------------------------------------------------------------


def _sample_traits(rng: np.random.Generator, impaired: bool) -> dict[str, float]:
    traits = {}
    for name, (ctrl, imp, between_sd, _) in TRAIT_SPEC.items():
        mean = imp if impaired else ctrl
        lo, hi = TRAIT_BOUNDS[name]
        traits[name] = float(np.clip(rng.normal(mean, between_sd), lo, hi))
    return traits


def _session_traits(rng: np.random.Generator, base: dict[str, float], drift: float) -> dict[str, float]:
    """Apply programmed drift (in units of the control→impaired gap) plus within-person noise."""
    out = {}
    for name, (ctrl, imp, _, within_sd) in TRAIT_SPEC.items():
        lo, hi = TRAIT_BOUNDS[name]
        out[name] = float(np.clip(base[name] + drift * (imp - ctrl) + rng.normal(0, within_sd), lo, hi))
    return out


def _render_session(
    rng: np.random.Generator, traits: dict[str, float], sex: str, cfg: SyntheticConfig, topic: str,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    sr = cfg.sample_rate
    target = rng.uniform(*cfg.session_duration_s)
    par_f0 = rng.normal(205, 18) if sex == "F" else rng.normal(118, 12)
    inv_f0 = rng.normal(165, 10)
    vocab = list(TOPICS[topic]["nouns"])
    rng.shuffle(vocab)
    segments: list[tuple[float, np.ndarray]] = []
    turns: list[dict[str, Any]] = []
    t = 0.3
    prompt_idx = 0
    syll_dur = 60.0 / (traits["articulation_wpm"] * 1.45)  # ~1.45 syllables per word

    while t < target - 2.0:
        prompt = PROMPTS[topic][prompt_idx % len(PROMPTS[topic])]
        prompt_idx += 1
        start = t
        for w in prompt.split():
            audio = _word_audio(rng, _syllables(w), 60.0 / (175 * 1.45), inv_f0, 2.0, sr, 0.45)
            segments.append((t, audio))
            t += len(audio) / sr + rng.uniform(0.04, 0.1)
        turns.append({"speaker": "INV", "start": round(start, 3), "end": round(t, 3), "text": prompt})

        t += float(np.clip(rng.lognormal(np.log(traits["latency_s"]), 0.35), 0.15, 6.0))
        start = t
        sentences = []
        n_sent = int(rng.integers(2, 5))
        for si in range(n_sent):
            sentence, _ = _sentence(rng, topic, traits, vocab)
            sentences.append(sentence)
            for w in sentence.split():
                filler = w in FILLERS
                sd = syll_dur * (2.2 if filler else 1.0)
                f0_word = par_f0 * (0.97 if filler else 1.0)
                audio = _word_audio(rng, 1 if filler else _syllables(w), sd, f0_word,
                                    0.4 if filler else traits["f0_var_st"], sr, 0.6)
                segments.append((t, audio))
                t += len(audio) / sr + rng.uniform(0.03, 0.12)
            if si < n_sent - 1:
                pause = rng.exponential(traits["pause_mean_s"]) + 0.2
                if rng.random() < traits["long_pause_prob"]:
                    pause += rng.uniform(2.0, 4.0)
                t += pause
            if t > target:
                break
        end = t
        turns.append({"speaker": "PAR", "start": round(start, 3), "end": round(end, 3), "text": ". ".join(sentences)})
        # Occasional interviewer overlap (interruption) or a normal gap before the next prompt.
        t = end - rng.uniform(0.2, 0.5) if rng.random() < 0.12 else end + rng.uniform(0.3, 0.9)

    total = int((max(t, target) + 0.5) * sr)
    signal = np.zeros(total)
    for start, audio in segments:
        i = int(start * sr)
        j = min(i + len(audio), total)
        signal[i:j] += audio[: j - i]
    snr_db = rng.uniform(*cfg.noise_snr_db)
    speech_power = np.mean(signal[np.abs(signal) > 0.01] ** 2) if np.any(np.abs(signal) > 0.01) else 1e-4
    noise = _colored_noise(rng, total) * np.sqrt(speech_power / 10 ** (snr_db / 10))
    signal = signal + noise
    signal = 0.7 * signal / (np.max(np.abs(signal)) + 1e-9)
    transcript = ". ".join(tr["text"] for tr in turns if tr["speaker"] == "PAR")
    return signal.astype(np.float32), turns, transcript


def generate_synthetic_dataset(out_dir: str | Path, cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate audio, transcripts, turn files and ``metadata.csv`` under ``out_dir``.

    Returns the metadata table (dataset-native format consumed by the synthetic adapter).
    """
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "turns").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    study_start = datetime(2024, 1, 8)

    for p in range(cfg.n_participants):
        pid = f"SYN-P{p + 1:03d}"
        impaired = bool(rng.random() < 0.45)
        has_change = bool(rng.random() < (cfg.change_fraction * (1.5 if impaired else 0.5)))
        sex = "F" if rng.random() < 0.55 else "M"
        age_band = str(rng.choice(["60-69", "70-79", "80-89"], p=[0.35, 0.45, 0.2]))
        education_band = str(rng.choice(["<=12y", "13-16y", ">16y"], p=[0.4, 0.4, 0.2]))
        base = _sample_traits(rng, impaired)
        n_sessions = int(rng.integers(cfg.sessions_min, cfg.sessions_max + 1))
        change_start = int(rng.integers(1, max(2, n_sessions - 1))) if has_change else None
        change_rate = float(rng.uniform(0.35, 0.7))
        enrol = study_start + timedelta(days=int(rng.integers(0, 120)))
        date = enrol
        for s in range(n_sessions):
            if s > 0:
                date += timedelta(days=int(rng.integers(*cfg.session_interval_days)))
            if s > 0 and rng.random() < cfg.missing_session_prob:
                continue  # skipped visit: leaves a gap in the timeline
            drift = change_rate * (s - change_start + 1) if has_change and s >= change_start else 0.0
            traits = _session_traits(rng, base, drift)
            topic = "kitchen_scene" if s % 2 == 0 else "daily_routine"
            signal, turns, transcript = _render_session(rng, traits, sex, cfg, topic)

            quality_issue = None
            roll = rng.random()
            if roll < 0.015:
                signal = np.clip(signal * 4.0, -1.0, 1.0)  # heavy clipping
                quality_issue = "clipping"
            elif roll < 0.025:
                signal = signal * 0.003  # near-silent recording
                quality_issue = "low_level"

            sid = f"{pid}-S{s + 1:02d}"
            rid = f"{sid}-R01"
            wav_path = out_dir / "audio" / f"{rid}.wav"
            turns_path = out_dir / "turns" / f"{rid}.json"
            sf.write(wav_path, signal, cfg.sample_rate, subtype="PCM_16")
            turns_path.write_text(json.dumps(turns, indent=1), encoding="utf-8")
            score = float(np.clip(rng.normal(27.5 - 6.0 * impaired - 2.5 * drift, 1.5), 0, 30))
            rows.append({
                "participant_id": pid,
                "session_id": sid,
                "recording_id": rid,
                "audio_file": relpath(wav_path),
                "turns_file": relpath(turns_path),
                "session_index": s,
                "session_date": date.date().isoformat(),
                "task": topic,
                "transcript": transcript,
                "group": LABEL_IMPAIRED if impaired else LABEL_CONTROL,
                "synthetic_score": round(score, 1),
                "age_band": age_band,
                "sex": sex,
                "education_band": education_band,
                "programmed_change": has_change,
                "programmed_change_start": change_start,
                "programmed_drift": round(drift, 3),
                "quality_issue": quality_issue,
            })
        log.info("synthetic participant generated", extra={"ctx": {"participant": pid, "sessions": n_sessions}})

    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "README.txt").write_text(
        SYNTHETIC_NOTICE + "\n\nGenerated by ml/data/synthetic.py with config:\n" + json.dumps(cfg.__dict__, indent=2),
        encoding="utf-8",
    )
    return meta
