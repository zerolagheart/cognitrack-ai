"""CogniTrack command-line interface: every pipeline stage is a documented command.

Examples::

    cognitrack generate-synthetic
    cognitrack run-all --quick          # small synthetic end-to-end demo (CPU, a few minutes)
    cognitrack train --only full        # retrain one modality set
    cognitrack download-models --embeddings --whisper
"""

from __future__ import annotations

import json
import time

import typer

from ml.config import load_config
from ml.utils import configure_logging, get_logger, set_seed

app = typer.Typer(add_completion=False, help="CogniTrack AI pipeline commands (research prototype, not a medical device).")
log = get_logger("cognitrack.cli")
ConfigOpt = typer.Option(None, "--config", "-c", help="YAML file merged over configs/default.yaml")


def _cfg(config: str | None, overrides: dict | None = None):
    configure_logging()
    cfg = load_config(config, overrides)
    set_seed(cfg["project"]["seed"])
    return cfg


@app.command("generate-synthetic")
def generate_synthetic(config: str | None = ConfigOpt, participants: int | None = typer.Option(None, help="Override participant count")) -> None:
    """Generate the clearly-labelled synthetic demonstration dataset under data/synthetic."""
    from pipelines.ingestion import generate_synthetic as gen

    meta = gen(_cfg(config), participants)
    typer.echo(f"Synthetic dataset: {meta['participant_id'].nunique()} participants, {len(meta)} recordings (SYNTHETIC).")


@app.command()
def ingest(config: str | None = ConfigOpt, lenient: bool = typer.Option(False, help="Write manifest even if validation fails")) -> None:
    """Ingest configured datasets into the canonical manifest and validate it."""
    from pipelines.ingestion import ingest as run

    _, report = run(_cfg(config), strict=not lenient)
    typer.echo(json.dumps({"passed": report["passed"], "blocking_issues": report["blocking_issues"], **report["profile"]}, indent=2))


@app.command("extract-features")
def extract_features(config: str | None = ConfigOpt, workers: int = typer.Option(1, help="Parallel worker processes"),
                     limit: int | None = typer.Option(None, help="Process only the first N recordings")) -> None:
    """Audio quality checks + acoustic, linguistic and conversational features."""
    from pipelines.features import extract_features as run

    df = run(_cfg(config), config, workers, limit)
    typer.echo(f"Extracted features for {len(df)} recordings ({int(df['extraction_error'].notna().sum())} errors).")


@app.command()
def embed(config: str | None = ConfigOpt, encoder: str | None = typer.Option(None, help="spectral_stats | hubert | wav2vec2")) -> None:
    """Extract pooled audio embeddings (cached by audio hash + encoder version)."""
    from pipelines.features import extract_embeddings

    df = extract_embeddings(_cfg(config), encoder)
    typer.echo(f"Embeddings available for {len(df)} recordings.")


@app.command("build-dataset")
def build_dataset(config: str | None = ConfigOpt) -> None:
    """Merge manifest + features + embeddings and create participant-independent splits."""
    from pipelines.dataset import build_dataset as run

    run(_cfg(config))


@app.command()
def train(config: str | None = ConfigOpt, only: list[str] | None = typer.Option(None, help="Experiment ids or modality sets")) -> None:
    """Run the ablation × model × fusion experiment grid and register the serving model."""
    from ml.training.trainer import train_all
    from pipelines.dataset import load_dataset

    cfg = _cfg(config)
    index = train_all(load_dataset(cfg), cfg, only)
    typer.echo(f"Serving model: {index['serving']['experiment_id']} (CV AUROC {index['serving']['chosen_cv_score']:.3f})")


@app.command()
def evaluate(config: str | None = ConfigOpt) -> None:
    """Model comparison, ablation, explainability, fairness and uncertainty reports."""
    from pipelines.analytics import write_model_card
    from pipelines.dataset import load_dataset
    from pipelines.evaluation import evaluate as run

    cfg = _cfg(config)
    run(cfg, load_dataset(cfg))
    write_model_card(cfg)


@app.command()
def longitudinal(config: str | None = ConfigOpt) -> None:
    """Baseline-relative longitudinal change analysis for every participant."""
    from pipelines.analytics import run_longitudinal
    from pipelines.dataset import load_dataset

    cfg = _cfg(config)
    typer.echo(json.dumps(run_longitudinal(cfg, load_dataset(cfg))["cohort_summary"], indent=2))


@app.command()
def monitor(config: str | None = ConfigOpt) -> None:
    """Data / prediction drift report (PSI, KS, Evidently HTML)."""
    from pipelines.analytics import run_monitoring
    from pipelines.dataset import load_dataset

    cfg = _cfg(config)
    report = run_monitoring(cfg, load_dataset(cfg))
    typer.echo(f"{report['feature_drift']['n_drifted']} of {report['feature_drift']['n_features']} features flagged for drift.")


@app.command()
def robustness(config: str | None = ConfigOpt) -> None:
    """Perturbation robustness of the serving model (re-extracts features; slowest stage)."""
    from pipelines.analytics import run_robustness_report, write_model_card
    from pipelines.dataset import load_dataset

    cfg = _cfg(config)
    run_robustness_report(cfg, load_dataset(cfg))
    write_model_card(cfg)


@app.command()
def publish(config: str | None = ConfigOpt) -> None:
    """Load manifest, features, predictions, models and reports into the application database."""
    from backend.app.services.sync import publish_all

    _cfg(config)
    typer.echo(json.dumps(publish_all(), indent=2))


@app.command("run-all")
def run_all(config: str | None = ConfigOpt, quick: bool = typer.Option(False, help="Small synthetic cohort for a fast demo"),
            skip_robustness: bool = typer.Option(False), skip_publish: bool = typer.Option(False),
            workers: int = typer.Option(1)) -> None:
    """End-to-end: synthetic data → ingest → features → embeddings → split → train → evaluate → analytics → publish."""
    from ml.training.trainer import train_all
    from pipelines.analytics import run_longitudinal, run_monitoring, run_robustness_report, write_model_card
    from pipelines.dataset import build_dataset as build
    from pipelines.evaluation import evaluate as run_eval
    from pipelines.features import extract_embeddings, extract_features
    from pipelines.ingestion import generate_synthetic as gen
    from pipelines.ingestion import ingest as run_ingest

    overrides = None
    if quick:
        overrides = {"synthetic": {"n_participants": 40, "sessions_max": 5, "session_duration_s": [14.0, 20.0]},
                     "evaluation": {"n_bootstrap": 200, "robustness": {"n_recordings": 16}}, "training": {"cv_folds": 4}}
    cfg = _cfg(config, overrides)
    t0 = time.perf_counter()
    stages = []

    def stage(name: str, fn, *args):
        s = time.perf_counter()
        out = fn(*args)
        stages.append((name, round(time.perf_counter() - s, 1)))
        typer.echo(f"[{name}] done in {stages[-1][1]}s")
        return out

    if "synthetic" in cfg["project"]["datasets"]:
        stage("generate-synthetic", gen, cfg)
    stage("ingest", run_ingest, cfg)
    stage("extract-features", extract_features, cfg, config, workers)
    stage("embed", extract_embeddings, cfg)
    df = stage("build-dataset", build, cfg)
    stage("train", train_all, df, cfg)
    stage("evaluate", run_eval, cfg, df)
    stage("longitudinal", run_longitudinal, cfg, df)
    stage("monitor", run_monitoring, cfg, df)
    if not skip_robustness:
        stage("robustness", run_robustness_report, cfg, df)
    stage("model-card", write_model_card, cfg)
    if not skip_publish:
        from backend.app.services.sync import publish_all

        stage("publish", publish_all)
    typer.echo(f"Pipeline complete in {time.perf_counter() - t0:.0f}s: " + ", ".join(f"{n}={s}s" for n, s in stages))


@app.command("download-models")
def download_models(config: str | None = ConfigOpt, embeddings: bool = typer.Option(False, help="HuBERT + wav2vec 2.0"),
                    whisper: bool = typer.Option(False, help="Whisper ASR checkpoint"), nlp: bool = typer.Option(False, help="sentence-transformer"),
                    spacy_model: bool = typer.Option(False, "--spacy", help="spaCy English pipeline")) -> None:
    """Explicitly download pretrained checkpoints into artifacts/checkpoints (never done implicitly)."""
    cfg = _cfg(config)
    cache = str(cfg.path("model_cache"))
    targets = []
    if embeddings:
        targets += [(e["checkpoint"], e["revision"]) for k, e in cfg["embeddings"]["encoders"].items() if "checkpoint" in e]
    if whisper:
        targets.append((cfg["transcription"]["checkpoint"], cfg["transcription"]["revision"]))
    if nlp:
        targets.append((cfg["nlp"]["sentence_model"], "main"))
    if targets:
        from huggingface_hub import snapshot_download

        for repo, rev in targets:
            typer.echo(f"Downloading {repo}@{rev} → {cache}")
            snapshot_download(repo_id=repo, revision=rev, cache_dir=cache,
                              ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.onnx", "tf_model*", "flax_model*", "rust_model*"])
    if spacy_model:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "spacy", "download", cfg["nlp"]["spacy_model"]], check=True)
    if not targets and not spacy_model:
        typer.echo("Nothing selected. Use --embeddings, --whisper, --nlp and/or --spacy.")


if __name__ == "__main__":
    app()
