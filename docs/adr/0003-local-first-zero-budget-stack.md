# ADR-0003: Local-first, zero-budget stack

* Status: accepted
* Date: 2026-11-11

## Context

The project must not need paid APIs, cloud services or GPUs.

## Decision

* MLflow with a SQLite backend; DVC with a local remote.
* PostgreSQL in Docker, with a SQLite fallback for Docker-free use.
* Local Hugging Face checkpoints, downloaded only by an explicit command.
* Evidently plus in-house PSI/KS for drift.
* FastAPI background tasks instead of a broker.

Optional components degrade with recorded backend names: openSMILE → librosa proxies, spaCy → rule-based NLP, torch → scikit-learn MLP, HuBERT → spectral-statistics encoder.

## Consequences

Anyone can run the full system on a laptop. Throughput is single-node, and the scale-out path (job queue, object storage) is documented in ARCHITECTURE.md.

## Alternatives considered

Hosted tracking (W&B), managed databases, commercial ASR (all rejected on cost and data-protection grounds).
