# Notebooks

Notebooks are for exploration only; production logic lives in `ml/` and `pipelines/`.

## Optional GPU embedding extraction (Colab / Kaggle, free tier)

```python
!git clone <your fork> cognitrack-ai && cd cognitrack-ai && pip install -e ".[deep,audio]"
# upload data/interim/manifest.parquet and the referenced audio (only if your data agreement permits cloud processing)
!cognitrack download-models --embeddings
!cognitrack embed --encoder hubert        # uses CUDA automatically (embeddings.device: auto)
# download data/processed/embedding_cache.parquet back to your local checkout
```

The cache is keyed by audio SHA-256 and encoder version, so the local pipeline reuses these vectors without a GPU.
Never assume GPU availability: every stage has a CPU path.
