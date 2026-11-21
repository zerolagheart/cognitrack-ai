"""Scikit-learn compatible PyTorch MLP classifier.

Used for the "MLP on handcrafted features" and "MLP on embeddings" experiments. If PyTorch is not
installed, :func:`make_mlp` returns scikit-learn's ``MLPClassifier`` (same role, CPU-only) and the
experiment metadata records which implementation ran.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary MLP with dropout, class-weighted BCE and early stopping on an internal validation split."""

    def __init__(self, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.2, epochs: int = 150,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 32, patience: int = 15,
                 random_state: int = 42) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _build(self, n_in: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = n_in
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> TorchMLPClassifier:
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        y_idx = y_idx.astype(np.float32)
        splitter = (GroupShuffleSplit(1, test_size=0.2, random_state=self.random_state) if groups is not None
                    else StratifiedShuffleSplit(1, test_size=0.2, random_state=self.random_state))
        tr, va = next(splitter.split(X, y_idx, groups))
        self.model_ = self._build(X.shape[1])
        pos = y_idx[tr].mean()
        pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y_idx[tr])
        Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y_idx[va])
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_state, wait = np.inf, None, 0
        self.history_: list[float] = []
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(Xv).squeeze(-1), yv))
            self.history_.append(val_loss)
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(-1).numpy()
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def make_mlp(params: dict[str, Any], seed: int) -> tuple[Any, str]:
    if TORCH_AVAILABLE:
        return TorchMLPClassifier(
            hidden_dims=tuple(params["hidden_dims"]), dropout=params["dropout"], epochs=params["epochs"], lr=params["lr"],
            weight_decay=params["weight_decay"], batch_size=params["batch_size"], patience=params["patience"], random_state=seed,
        ), "torch"
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=tuple(params["hidden_dims"]), alpha=params["weight_decay"], early_stopping=True,
                         max_iter=params["epochs"] * 5, random_state=seed), "sklearn"
