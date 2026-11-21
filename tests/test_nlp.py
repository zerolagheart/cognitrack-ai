import math

import pytest

from ml.config import load_config
from ml.data.chat_parser import clean_chat_text, parse_chat
from ml.nlp.conversation import extract_conversational_features
from ml.nlp.linguistic import extract_linguistic_features, mattr

NLP_CFG = load_config()["nlp"]


def test_linguistic_features_basic_counts():
    text = "The boy is taking the cookie. Um the the girl is laughing. She is uh reaching for the thing."
    feats, backend = extract_linguistic_features(text, NLP_CFG, speech_duration_s=30.0)
    assert feats["word_count"] == 19
    assert feats["filler_rate"] == pytest.approx(2 / 19)
    assert feats["repetition_rate"] > 0  # "the the"
    assert feats["indefinite_term_rate"] == pytest.approx(1 / 19)
    assert feats["speech_rate_wpm"] == pytest.approx(19 / 0.5)
    assert 0 < feats["type_token_ratio"] <= 1
    assert backend["nlp_backend"] in {"rule_based", "spacy:en_core_web_sm"}


def test_missing_transcript_yields_no_features():
    feats, backend = extract_linguistic_features(None, NLP_CFG)
    assert feats == {} and backend["nlp_backend"] == "none"


def test_mattr_is_length_robust():
    words = ("a b c d e f g h i j " * 20).split()
    assert mattr(words, window=10) == pytest.approx(1.0)
    assert mattr(["x"] * 50, window=10) == pytest.approx(0.1)


def test_conversational_latency_and_interruptions():
    turns = [
        {"speaker": "INV", "start": 0.0, "end": 2.0, "text": "tell me about the picture"},
        {"speaker": "PAR", "start": 3.0, "end": 8.0, "text": "the boy is on the stool"},
        {"speaker": "INV", "start": 7.5, "end": 9.0, "text": "anything else"},  # overlaps → interruption of participant
        {"speaker": "PAR", "start": 10.0, "end": 14.0, "text": "the stool is falling over"},
    ]
    f = extract_conversational_features(turns)
    assert f["mean_response_latency_s"] == pytest.approx(1.0)
    assert f["participant_turn_count"] == 2
    assert f["participant_interrupted_rate"] == pytest.approx(0.5)
    assert 0 < f["conversational_balance"] < 1
    assert not math.isnan(f["topic_continuity"])


def test_conversational_requires_attributed_turns():
    assert extract_conversational_features(None) == {}
    assert extract_conversational_features([{"speaker": "PAR", "start": 0, "end": 1}]) == {}


def test_chat_parser(tmp_path):
    cha = tmp_path / "001-0.cha"
    cha.write_text(
        "@Begin\n@ID:\teng|Pitt|PAR|67;|female|ProbableAD||Participant|12||\n"
        "*INV:\ttell me what you see . \x150_1500\x15\n"
        "*PAR:\t&uh the boy [/] the boy is taking a cookie . \x152000_5200\x15\n%mor:\tdet|the\n"
        "*PAR:\tand xxx (.) the water's running over . \x155600_8000\x15\n@End\n", encoding="utf-8")
    t = parse_chat(cha)
    assert t.participants["PAR"]["sex"] == "female"
    assert t.speaker_text("PAR").startswith("uh the boy the boy is taking a cookie")
    assert "xxx" not in t.speaker_text("PAR")
    turns = t.turns()
    assert len(turns) == 3 and turns[1]["start"] == 2.0 and turns[1]["end"] == 5.2
    assert clean_chat_text("&=laughs it is (be)cause") == "it is because"


def test_spacy_parse_depth_terminates():
    """Regression: spaCy Token objects are not identity-stable, so depth must compare indices."""
    from ml.nlp.text import load_spacy, parse

    if load_spacy(NLP_CFG["spacy_model"]) is None:
        pytest.skip("spaCy model not installed")
    sentences, backend = parse("The boy who is standing on the stool is taking a cookie.", NLP_CFG["spacy_model"])
    assert backend.startswith("spacy")
    assert 1 <= sentences[0].depth < 10 and sentences[0].n_clauses >= 2
