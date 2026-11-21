# Limitations

CogniTrack AI is a research and decision-support prototype. This page states what it cannot do, so its outputs are not over-interpreted.

## Scientific

* **No clinical validation.** No model in this repository has been validated against a clinical reference standard. With the default synthetic data, performance reflects effects programmed into the generator.
* **Small, narrow corpora.** Public datasets have roughly 10² participants, are mostly English, mostly picture description, and mostly cross-sectional. Estimates have wide intervals and may not transfer to other tasks, languages, devices or populations.
* **Confounding.** Speech rate, pausing, pitch and lexical diversity also vary with age, education, dialect, bilingualism, hearing, mood, fatigue, medication, motor speech disorders and recording conditions.
* **Label limitations.** Dataset labels such as "probable AD" are themselves clinical judgements of varying certainty. Cross-sectional labels do not describe change over time.
* **Longitudinal power.** With 3–6 sessions per person, change-point and trend tests have low power and high variance. Practice effects and task alternation can also look like change.

## Technical

* **Voice quality measures.** Without openSMILE, jitter, shimmer and HNR come from documented frame-level proxies that are not equivalent to Praat. They are microphone- and noise-sensitive, and the two backends are not comparable.
* **VAD.** The energy-based VAD can mistake background speech or music for speech. Strong noise inflates speech time and deflates pauses.
* **SNR estimate.** A percentile heuristic, not a calibrated measurement.
* **ASR.** Whisper may hallucinate on silence, drop disfluencies and err more on accented and older voices. Linguistic features from ASR and from manual transcripts must not be mixed.
* **Speaker attribution.** Conversational features need dataset-provided turns. No diarisation is performed, by design, so such features are missing for unattributed audio.
* **Synthetic audio.** The generator's harmonic "speech" is not natural speech. Pretrained encoders such as HuBERT behave differently on it, so the spectral-stats encoder is the default in demo mode.
* **Fairness analysis.** Only possible where demographics exist and subgroups reach the minimum size. Most public corpora lack the metadata for meaningful fairness conclusions.
* **Operational.** No schema migrations (`create_all`), API-key auth rather than SSO, single-node background processing.

## Known failure modes

| Situation | Effect | Mitigation in the system |
|---|---|---|
| Very noisy or clipped recording | Distorted acoustic features | Quality flags, exclusion, abstention |
| Very short recording (< 2 s speech) | Unstable statistics | Features left missing, abstention |
| Non-English speech | Rule-based or English NLP invalid | Language recorded; do not use the linguistic modality |
| New microphone or site | Feature drift | PSI/KS drift report, OOD abstention |
| Missing transcript or turns | Fewer modalities | Missing-feature fraction drives abstention; late fusion substitutes the base rate |
| Single noisy session | Spurious longitudinal flag | EWMA smoothing, explicit "statistical, not clinical" messaging |
