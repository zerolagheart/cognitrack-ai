# ADR-0005: Missing data, labels and speaker turns are never fabricated

* Status: accepted
* Date: 2026-11-11

## Context

Clinical speech datasets are incomplete: some lack transcripts, speaker attribution, demographics or labels. Filling these gaps silently, for example by diarising to invent turns, pseudo-labelling, or zero-filling features, produces convincing but invalid results.

## Decision

* Missing clinical labels are excluded from supervised learning.
* Conversational features are computed only from dataset-provided turns.
* Acoustic statistics are left missing when there is too little speech.
* Unusable audio is excluded and counted.
* Imputation happens only inside model pipelines (fitted on training folds), and the missing-feature fraction is an abstention trigger at inference.
* Synthetic data is flagged at every layer.

## Consequences

Some modalities are unavailable for some datasets (e.g. no linguistic features for ADReSSo without ASR). The experiment grid records these as skipped experiments with a reason.
