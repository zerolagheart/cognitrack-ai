# ADR-0001: Participant-independent splits and CV-based model selection

* Status: accepted
* Date: 2026-11-11

## Context

Speech from one person is highly self-similar (voice, vocabulary, recording set-up). If one person's recordings fall in both training and test data, the model can learn to recognise the speaker and appear to recognise the condition. Published speech-biomarker results are often inflated this way. The test set is also small, so choosing models by test score would overfit it.

## Decision

* All partitions and CV folds are grouped by participant (`split_participants`, `StratifiedGroupKFold`).
* `assert_no_leakage` runs on every modelling table and checks participant ID, recording ID and audio hash.
* The serving model is chosen by participant-grouped CV AUROC on the training partition. The simplest configuration within 0.01 of the best wins.
* Calibration uses the validation partition. Test metrics are computed once and carry cluster-bootstrap confidence intervals.

## Consequences

* Estimates are honest but noisy: intervals are wide on small corpora, and the reports show them.
* CV folds are capped by the participant count of the rarer class.

## Alternatives considered

Recording-level random splits (leaky); nested CV for everything (more compute, harder to explain; may be added for the final real-data analysis).
