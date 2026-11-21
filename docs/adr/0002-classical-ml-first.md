# ADR-0002: Classical models on interpretable features first; deep models must earn their place

* Status: accepted
* Date: 2026-11-11

## Context

Cognitive-speech corpora contain hundreds of participants at most. Deep models overfit at this scale, and clinicians need explanations.

## Decision

* Baselines are logistic regression, random forest, XGBoost and LightGBM on handcrafted acoustic, linguistic and conversational features.
* Deep components are a small PyTorch MLP, attention fusion, and *frozen* self-supervised embeddings (HuBERT, with wav2vec 2.0 as comparator). The embeddings are reduced by PCA inside the CV pipeline.
* Fine-tuning foundation models is out of scope for the CPU default.
* The selection rule prefers simpler models when performance is within the margin.

## Consequences

The defaults are fast, CPU-friendly and explainable with SHAP. Some gains from end-to-end learning may be missed until larger data is available.

## Alternatives considered

Fine-tuning HuBERT or WavLM end to end (GPU-bound, overfits small data); training from scratch (not feasible).
