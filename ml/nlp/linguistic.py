"""Linguistic feature extraction from participant transcripts.

Features quantify lexical diversity, fluency markers, syntactic complexity and coherence. Individually
none is diagnostic; several are confounded by education, language, task and ASR errors (see
docs/LIMITATIONS.md). Transcripts from ASR inherit Whisper's tendency to drop disfluencies, so filler and
repetition rates are not comparable between ``transcript_source = dataset`` and ``whisper``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ml.nlp.text import (
    FUNCTION_WORDS,
    INDEFINITE_TERMS,
    PRONOUNS,
    adjacent_similarity,
    parse,
    split_sentences,
    tokenize,
)

MATTR_WINDOW = 25


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio (Covington & McFall, 2010); length-robust unlike raw TTR."""
    if not tokens:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    ratios = [len(counts) / window]
    for i in range(window, len(tokens)):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def _count_fillers(text_tokens: list[str], fillers: list[str]) -> int:
    singles = {f for f in fillers if " " not in f}
    multi = [f.split() for f in fillers if " " in f]
    count = sum(t in singles and t != "like" and t != "well" for t in text_tokens)
    for phrase in multi:
        n = len(phrase)
        count += sum(text_tokens[i: i + n] == phrase for i in range(len(text_tokens) - n + 1))
    return count


def extract_linguistic_features(
    transcript: str | None, nlp_cfg: dict[str, Any], speech_duration_s: float | None = None, sentence_model: Any | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return ``(features, backend_info)``. Missing transcript → empty features (never fabricated)."""
    if not transcript or not transcript.strip():
        return {}, {"nlp_backend": "none"}
    tokens = tokenize(transcript)
    sentences, backend = parse(transcript, nlp_cfg["spacy_model"])
    n = len(tokens)
    if n == 0:
        return {"word_count": 0.0}, {"nlp_backend": backend}
    counts = Counter(tokens)
    types = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    sent_lengths = [len(s.tokens) for s in sentences] or [n]
    fillers = _count_fillers(tokens, nlp_cfg["filler_words"])
    immediate_reps = sum(tokens[i] == tokens[i - 1] for i in range(1, n))
    bigrams = list(zip(tokens, tokens[1:], strict=False))
    repeated_bigrams = sum(c - 1 for c in Counter(bigrams).values() if c > 1)
    pronouns = sum(t in PRONOUNS for t in tokens)
    indefinite = sum(t in INDEFINITE_TERMS for t in tokens)
    content = [t for t in tokens if t not in FUNCTION_WORDS]

    if sentences and sentences[0].pos:
        nouns = sum(p in {"NOUN", "PROPN"} for s in sentences for p in s.pos)
        verbs = sum(p == "VERB" for s in sentences for p in s.pos)
        content_pos = sum(p in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} for s in sentences for p in s.pos)
        depth = float(np.mean([s.depth for s in sentences if s.depth is not None]))
        content_density = content_pos / max(sum(len(s.pos) for s in sentences), 1)
        pronoun_noun_ratio = pronouns / nouns if nouns else float("nan")
        verb_rate = verbs / n
    else:
        depth = float("nan")
        content_density = len(content) / n
        pronoun_noun_ratio = float("nan")
        verb_rate = float("nan")

    feats: dict[str, float] = {
        "word_count": float(n),
        "unique_word_count": float(types),
        "type_token_ratio": types / n,
        "mattr": mattr(tokens),
        # Brunet's W (lower = richer) and Honoré's R (higher = richer); both less length-dependent than TTR.
        "brunet_w": float(n ** (types ** -0.165)),
        "honore_r": float(100 * math.log(n) / (1 - hapax / types)) if types and hapax < types else float("nan"),
        "sentence_count": float(len(sentences)),
        "mean_sentence_length": float(np.mean(sent_lengths)),
        "repetition_rate": (immediate_reps + repeated_bigrams) / n,
        "filler_rate": fillers / n,
        "pronoun_rate": pronouns / n,
        "pronoun_noun_ratio": pronoun_noun_ratio,
        "indefinite_term_rate": indefinite / n,  # word-finding proxy
        "content_density": content_density,
        "verb_rate": verb_rate,
        "mean_parse_depth": depth,
        "clauses_per_sentence": float(np.mean([s.n_clauses for s in sentences])) if sentences else float("nan"),
        "incomplete_sentence_rate": float(np.mean([(not s.has_verb) or len(s.tokens) < 3 for s in sentences])) if sentences else float("nan"),
        "semantic_coherence": adjacent_similarity(split_sentences(transcript), sentence_model),
    }
    if speech_duration_s and speech_duration_s > 0:
        feats["speech_rate_wpm"] = n / (speech_duration_s / 60.0)
    return feats, {"nlp_backend": backend, "similarity_backend": "sentence_transformers" if sentence_model else "tfidf"}
