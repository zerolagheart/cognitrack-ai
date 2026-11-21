# Model card — generated section

Generated 2026-11-12T13:12:45.444070+00:00 from the latest pipeline run.

> **SYNTHETIC DATA: these results come from programmatically generated data with group differences built in by construction. They demonstrate that the pipeline works and are NOT evidence of clinical validity.**

## Serving model

- Experiment: `full__early__random_forest`
- Model: random_forest (sklearn), strategy: early, modalities: acoustic, linguistic, conversational, embedding
- Features: 335; calibration: sigmoid
- Dataset version: `86f144e1e35d4c89`

## Test performance (held-out participants)

| Metric | Value | 95% CI |
|---|---|---|
| auroc | 0.966 | 0.886 – 1.000 |
| auprc | 0.921 | 0.698 – 1.000 |
| balanced_accuracy | 0.885 | 0.805 – 0.954 |
| sensitivity | 0.815 | 0.691 – 0.944 |
| specificity | 0.955 | 0.842 – 1.000 |
| f1 | 0.863 | 0.719 – 0.950 |
| brier | 0.076 | 0.028 – 0.137 |
| ece | 0.120 | 0.066 – 0.199 |

## Data partitions

| Split | Participants | Recordings | Labels (participants) |
|---|---|---|---|
| test | 15 | 73 | {'healthy_control': 9, 'cognitive_impairment': 6} |
| train | 45 | 191 | {'healthy_control': 25, 'cognitive_impairment': 20} |
| val | 12 | 51 | {'healthy_control': 7, 'cognitive_impairment': 5} |

## Uncertainty and abstention

- Abstention rate on test: 0.169
- Out-of-distribution rate on test: 0.028

## Subgroup performance (participant level)

Minimum subgroup size for reporting: 20 participants.

- sex = F (n=8): insufficient sample — not reported
- sex = M (n=7): insufficient sample — not reported
- age_band = 60-69 (n=7): insufficient sample — not reported
- age_band = 70-79 (n=6): insufficient sample — not reported
- age_band = 80-89 (n=2): insufficient sample — not reported
- education_band = 13-16y (n=8): insufficient sample — not reported
- education_band = <=12y (n=4): insufficient sample — not reported
- education_band = >16y (n=3): insufficient sample — not reported
- language = en (n=15): insufficient sample — not reported

## Robustness (re-extracted features under perturbation)

| Condition | AUROC | Mean |Δp| | Flip rate | Abstention |
|---|---|---|---|---|
| clean | 0.955 | 0.000 | 0.000 | 0.200 |
| noise_snr_20db | 0.938 | 0.045 | 0.025 | 1.000 |
| noise_snr_10db | 0.935 | 0.072 | 0.050 | 1.000 |
| noise_snr_5db | 0.932 | 0.104 | 0.075 | 1.000 |
| gain_-12db | 0.955 | 0.004 | 0.000 | 1.000 |
| gain_+6db | 0.949 | 0.034 | 0.025 | 0.400 |
| bandlimit_8000hz | 0.943 | 0.053 | 0.000 | 1.000 |
| truncated_50pct | 0.940 | 0.060 | 0.000 | 1.000 |
| missing_transcript | 0.908 | 0.109 | 0.100 | 0.150 |
| missing_speaker_turns | 0.923 | 0.095 | 0.075 | 0.250 |
