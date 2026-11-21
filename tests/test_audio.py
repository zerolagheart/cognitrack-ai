import numpy as np
import pytest
import soundfile as sf

from conftest import tone_bursts
from ml.audio.acoustic import extract_acoustic_features
from ml.audio.vad import detect_speech, pause_statistics
from ml.config import load_config
from ml.preprocessing.audio import AudioLoadError, assess_quality, extract_segments, load_audio

AUDIO_CFG = load_config()["audio"]


def test_load_audio_resamples_and_downmixes(tmp_path):
    stereo = np.stack([tone_bursts(sr=22050), tone_bursts(sr=22050)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 22050)
    sig, sr, native = load_audio(path, 16000)
    assert sr == 16000 and native == 22050
    assert sig.ndim == 1 and sig.dtype == np.float32
    assert abs(len(sig) / sr - len(stereo) / 22050) < 0.01


def test_load_audio_rejects_missing_and_corrupt(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "missing.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF0000WAVEnot really audio")
    with pytest.raises(AudioLoadError):
        load_audio(bad)


@pytest.mark.parametrize("signal, flag", [
    (np.zeros(16000 * 3, dtype=np.float32), "silent"),
    (tone_bursts()[: 16000], "too_short"),
    (np.clip(tone_bursts() * 8, -1, 1), "clipping"),
])
def test_quality_flags(signal, flag):
    q = assess_quality(signal, 16000, 16000, AUDIO_CFG)
    assert flag in q.flags


def test_clean_signal_is_usable():
    q = assess_quality(tone_bursts(), 16000, 16000, AUDIO_CFG)
    assert q.usable and q.flags == []


def test_vad_finds_bursts_and_pauses():
    sig = tone_bursts(bursts=5, burst_s=0.4, gap_s=0.6)
    act = detect_speech(sig, 16000, min_pause_s=0.25)
    assert len(act.segments) == 5
    assert len(act.pauses) == 4
    assert all(0.45 < p < 0.75 for p in act.pauses)
    stats = pause_statistics(act, len(sig) / 16000, long_pause_s=2.0)
    assert stats["long_pause_count"] == 0
    assert 0.3 < stats["voiced_fraction"] < 0.55


def test_acoustic_pitch_estimate_matches_known_f0():
    feats = extract_acoustic_features(tone_bursts(f0=150.0), 16000, AUDIO_CFG)
    assert feats["f0_median_hz"] == pytest.approx(150.0, rel=0.05)
    assert feats["f0_std_semitones"] < 1.0  # steady tone
    for key in ("mfcc_01_mean", "spectral_centroid_mean_hz", "hnr_db", "speech_rate_syll_per_s", "pause_rate_per_min"):
        assert key in feats and np.isfinite(feats[key])


def test_acoustic_features_missing_not_fabricated_for_tiny_speech():
    sig = np.zeros(16000 * 3, dtype=np.float32)
    sig[:2000] = 0.3 * np.sin(np.arange(2000) * 0.06)
    feats = extract_acoustic_features(sig, 16000, AUDIO_CFG)
    assert "f0_mean_hz" not in feats  # left missing rather than invented


def test_participant_segments_restrict_analysis():
    sig = tone_bursts(bursts=6)
    full = extract_acoustic_features(sig, 16000, AUDIO_CFG)
    part = extract_acoustic_features(sig, 16000, AUDIO_CFG, participant_segments=[(0.0, 2.7)])
    assert part["analysed_duration_s"] == pytest.approx(2.7)
    assert part["analysed_duration_s"] < full["analysed_duration_s"]
    assert len(extract_segments(sig, 16000, [(0, 1), (2, 3)])) == 32000


def test_vad_finds_pauses_at_moderate_snr():
    """Regression: a fixed offset from the speech level fell below the noise floor at ~20 dB SNR."""
    rng = np.random.default_rng(1)
    sig = tone_bursts(bursts=5, burst_s=0.5, gap_s=0.6, amp=0.3)
    speech_rms = np.sqrt(np.mean(sig[np.abs(sig) > 0.01] ** 2))
    noisy = sig + rng.standard_normal(len(sig)).astype(np.float32) * speech_rms / 10  # 20 dB SNR
    act = detect_speech(noisy, 16000, min_pause_s=0.25)
    assert len(act.pauses) == 4


def test_recording_without_detectable_speech_is_unusable():
    from ml.features.extraction import extract_from_signal

    t = np.arange(16000 * 4) / 16000
    quiet = (0.0008 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)  # ~-65 dBFS: not "silent", but below the VAD floor
    rec = extract_from_signal("R1", quiet, 16000, 16000, load_config(), transcript="the boy is taking a cookie")
    assert not rec.quality["usable"] and "no_speech_detected" in rec.quality["flags"]
    assert rec.features == {}
