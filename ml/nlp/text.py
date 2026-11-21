"""Text normalisation and NLP backends shared by linguistic and conversational features.

Backends degrade gracefully:

* spaCy (``en_core_web_sm``) for tokenisation, POS tags and dependency parses; otherwise a rule-based
  tokenizer with closed-class lexicons (documented as lower fidelity).
* sentence-transformers for semantic similarity when enabled and downloaded; otherwise TF-IDF cosine.

The active backend names are recorded with every feature row so results are never silently mixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ml.utils import get_logger

log = get_logger(__name__)

PRONOUNS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them",
    "their", "theirs", "themselves", "this", "that", "these", "those",
}
INDEFINITE_TERMS = ["thing", "things", "stuff", "something", "somebody", "someone", "whatsit", "thingy", "whatever"]
SUBORDINATORS = {"because", "although", "while", "when", "if", "since", "unless", "whereas", "which", "who",
                 "whom", "that", "where", "after", "before", "until"}
FUNCTION_WORDS = PRONOUNS | SUBORDINATORS | {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "is", "are", "was", "were",
    "be", "been", "am", "do", "does", "did", "have", "has", "had", "there", "here", "not", "so", "as", "by", "from",
    "up", "down", "over", "into", "onto", "off", "out", "about", "just", "very", "too", "also",
}
COMMON_VERBS = {"is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did", "go", "goes", "went",
                "make", "take", "see", "look", "get", "got", "say", "said", "think", "know", "come", "want",
                "use", "find", "give", "tell", "work", "call", "try", "ask", "feel", "leave", "put", "read", "buy",
                "walk", "drink", "cook", "clean", "watch", "visit", "feed", "listen", "like", "lived", "used", "liked"}

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*\.\s*")


@dataclass
class ParsedSentence:
    tokens: list[str]
    pos: list[str] = field(default_factory=list)  # Universal POS tags (empty for rule-based backend)
    depth: float | None = None  # max dependency depth
    has_verb: bool = False
    n_clauses: int = 1


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(normalize(text)) if s and s.strip()]


@lru_cache(maxsize=2)
def load_spacy(model: str) -> Any | None:
    try:
        import spacy

        return spacy.load(model, disable=["ner"])
    except Exception as exc:
        log.info("spaCy model unavailable; using rule-based NLP", extra={"ctx": {"model": model, "reason": type(exc).__name__}})
        return None


def _tree_depth(token: Any) -> int:
    # spaCy returns a new Token object on each `.head` access, so compare indices, not identity.
    depth = 0
    while token.head.i != token.i:
        token = token.head
        depth += 1
    return depth


def parse(text: str, spacy_model: str) -> tuple[list[ParsedSentence], str]:
    """Parse into sentences. Returns ``(sentences, backend_name)``."""
    nlp = load_spacy(spacy_model)
    sentences_text = split_sentences(text)
    if nlp is not None:
        parsed = []
        for doc in nlp.pipe(sentences_text):
            words = [t for t in doc if t.is_alpha]
            if not words:
                continue
            parsed.append(ParsedSentence(
                tokens=[t.lower_ for t in words],
                pos=[t.pos_ for t in words],
                depth=float(max(_tree_depth(t) for t in words)),
                has_verb=any(t.pos_ in {"VERB", "AUX"} for t in words),
                n_clauses=1 + sum(t.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj"} for t in words),
            ))
        return parsed, f"spacy:{spacy_model}"
    parsed = []
    for s in sentences_text:
        toks = tokenize(s)
        if not toks:
            continue
        has_verb = any(t in COMMON_VERBS or t.endswith("ing") or t.endswith("ed") for t in toks)
        parsed.append(ParsedSentence(tokens=toks, has_verb=has_verb, n_clauses=1 + sum(t in SUBORDINATORS for t in toks)))
    return parsed, "rule_based"


@lru_cache(maxsize=1)
def load_sentence_model(name: str, cache_dir: str | None = None) -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, cache_folder=cache_dir, local_files_only=True)
    except Exception as exc:
        log.info("sentence-transformer unavailable; using TF-IDF similarity", extra={"ctx": {"reason": type(exc).__name__}})
        return None


def adjacent_similarity(texts: list[str], sentence_model: Any | None = None) -> float:
    """Mean cosine similarity between consecutive texts (NaN if fewer than two)."""
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 2:
        return float("nan")
    if sentence_model is not None:
        emb = np.asarray(sentence_model.encode(texts, normalize_embeddings=True))
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        try:
            emb = TfidfVectorizer(token_pattern=r"[a-z]{2,}").fit_transform(texts).toarray()
        except ValueError:  # empty vocabulary
            return float("nan")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1, norms)
    sims = np.sum(emb[1:] * emb[:-1], axis=1)
    return float(np.mean(sims))
