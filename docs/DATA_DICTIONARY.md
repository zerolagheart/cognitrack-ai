# Data dictionary

## 1. Canonical recording manifest

Schema: `ml/data/schema.py` (`MANIFEST_SCHEMA`, strict). Stored at `data/interim/manifest.parquet`.

| Column | Type | Nullable | Description |
|---|---|---|---|
| participant_id | str | no | Pseudonymous, dataset-prefixed ID (`SYN-P001`, `ADRSO-adrso024`, `DBPITT-001`). Pattern `[A-Za-z0-9_-]+` |
| session_id | str | no | One assessment visit |
| recording_id | str | no, unique | Recording identifier |
| recording_path | str | no | Path relative to repo root (absolute only for out-of-tree data) |
| dataset_source | str | no | Adapter name |
| dataset_split_hint | str | yes | Official dataset split (e.g. ADReSSo `train` / `test-dist`). Used for provenance only; CogniTrack builds its own participant-level splits |
| session_index | int ≥ 0 | no | Visit order within a participant |
| session_timestamp | datetime | yes | Visit date (synthetic dates for synthetic data) |
| recording_duration | float ≥ 0 | yes | Seconds, from the audio header |
| sample_rate | float > 0 | yes | Native sample rate (Hz) |
| task | str | yes | Elicitation task (`picture_description`, `kitchen_scene`, `daily_routine`) |
| language | str | yes | ISO 639-1 |
| transcript | str | yes | Participant speech only. Missing if the dataset has none and ASR is off |
| transcript_source | enum | no | `dataset`, `whisper` or `none` |
| turns_path | str | yes | JSON speaker-turn file: `[{speaker, start, end, text}]`. Only where the dataset supplies attribution |
| assessment_label | enum | yes | `cognitive_impairment` or `healthy_control`, exactly as the dataset provides. **Never inferred** |
| assessment_score | float | yes | Instrument score (e.g. MMSE) where legitimately provided |
| assessment_instrument | str | yes | Name of the instrument. Synthetic data uses `synthetic_score (not a clinical instrument)` |
| age_band / sex / education_band | str | yes | Coarse, non-identifying demographics |
| consent_research | bool | no | Research consent recorded |
| is_synthetic | bool | no | True for generated data |
| audio_sha256 | str | yes | Content hash: duplicate detection, embedding cache key, leakage checks |
| provenance | JSON str | no | Adapter, version, source file(s), licence note, quality notes |

## 2. Feature table

Stored at `data/processed/features.parquet`. The modality mapping lives in `ml/features/registry.py`. All features are measurements, **not validated clinical indicators**.

### Acoustic (computed on participant turns when attributed)

| Feature | Unit | Definition |
|---|---|---|
| voiced_fraction | 0–1 | VAD speech time ÷ analysed time |
| pause_rate_per_min | /min | Pauses ≥ `min_pause_s` (0.25 s) between speech segments |
| mean_pause_s / median_pause_s / pause_std_s | s | Pause duration statistics |
| long_pause_rate_per_min | /min | Pauses ≥ `long_pause_s` (2 s) |
| mean_speech_segment_s | s | Mean voiced run length |
| f0_mean_hz | Hz | Mean YIN F0 over voiced frames (harmonicity > 0.6) |
| f0_std_semitones / f0_range_semitones | st | Pitch variability, and 5th–95th percentile range relative to the median |
| voiced_frame_fraction | 0–1 | Frames with a periodic F0 |
| jitter_local | ratio | openSMILE `jitterLocal_sma3nz_amean` **or** frame-level period-perturbation proxy |
| shimmer_local_db | dB | openSMILE `shimmerLocaldB_sma3nz_amean` **or** frame-peak amplitude-perturbation proxy |
| hnr_db | dB | openSMILE `HNRdBACF_sma3nz_amean` **or** autocorrelation HNR proxy |
| energy_std_db / energy_dynamic_range_db | dB | Loudness variability |
| zcr_mean, spectral_centroid_mean_hz, spectral_centroid_std_hz, spectral_bandwidth_mean_hz, spectral_rolloff_mean_hz, spectral_flatness_mean | — | Spectral shape over active frames |
| mfcc_XX_mean / mfcc_XX_std | — | 13 MFCCs (40 mel bands) |
| speech_rate_syll_per_s | syll/s | Syllable nuclei ÷ analysed time |
| articulation_rate_syll_per_s | syll/s | Syllable nuclei ÷ speech time |

`backends.voice_quality` records which jitter/shimmer/HNR implementation ran. Values from different backends are **not comparable**.

### Linguistic

| Feature | Definition |
|---|---|
| type_token_ratio | unique ÷ total words (length-dependent) |
| mattr | moving-average TTR, window 25 |
| brunet_w | N^(V^−0.165); lower means richer |
| honore_r | 100·log N ÷ (1 − V₁/V); higher means richer |
| mean_sentence_length | words per sentence |
| repetition_rate | (immediate repeats + repeated bigrams) ÷ words |
| filler_rate | um/uh/er/… and multi-word fillers ÷ words ("like" and "well" excluded as single tokens) |
| pronoun_rate, pronoun_noun_ratio | pronouns ÷ words; pronouns ÷ nouns (spaCy only) |
| indefinite_term_rate | thing/stuff/something/… ÷ words (word-finding proxy) |
| content_density, verb_rate | content POS ÷ tokens; verbs ÷ words |
| mean_parse_depth, clauses_per_sentence | dependency-tree depth; 1 + subordinate clauses |
| incomplete_sentence_rate | sentences with no verb or fewer than 3 words |
| semantic_coherence | mean cosine similarity of adjacent sentences (TF-IDF or sentence-transformer) |
| speech_rate_wpm | transcript words ÷ analysed minutes |

### Conversational (dataset speaker turns only)

turns_per_min, speaker_transitions_per_min, mean_participant_turn_s, mean_interlocutor_turn_s, mean/median/std response latency (interlocutor end → participant start, clipped at 0), mean_between_turn_silence_s, interruptions_per_min (cross-speaker overlap > 0.1 s), participant_interrupted_rate, conversational_balance (participant share of speech time), topic_continuity (similarity of consecutive participant turns).

### Quality and metadata columns

`quality_usable`, `quality_flags` (`too_short`, `too_long`, `silent`, `low_level`, `clipping`, `low_snr`, `low_sample_rate`, `corrupted`), `quality_snr_db` (percentile heuristic, not calibrated), `quality_clipping_fraction`, `quality_silence_fraction`, `backends` (JSON), `extraction_error`.

### Embeddings

`emb_000…emb_NNN` and `embedding_encoder` (version string: checkpoint@revision-layer-pooling). The models reduce embeddings with PCA inside the pipeline (`emb_pc_XX` in explanations).

## 3. Synthetic metadata extras

`data/synthetic/metadata.csv` also contains `programmed_change`, `programmed_change_start` and `programmed_drift` (generator ground truth). These are used **only** to validate the longitudinal engine on synthetic data, and never as model features.

## 4. Report payloads

`artifacts/reports/*.json` hold data_quality, lineage, splits, feature_extraction, model_comparison, explainability, fairness, uncertainty, robustness, longitudinal, drift and model_card. They are published into `monitoring_reports.payload` and served at `GET /api/v1/reports/{type}`.
