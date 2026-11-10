"""Logistic regression baseline (variant B). Reuses the same feature
windows as the GRU; takes only the last timestep (the signal-day state),
since plain logistic regression has no sequence mechanism."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from gru_classifier import CandidateWindowDataset


def fit_logreg_baseline(train_ds: CandidateWindowDataset) -> LogisticRegression:
    X = np.stack([window[-1] for window, _ in train_ds.samples])   # last timestep only
    y = np.array([label for _, label in train_ds.samples])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def logreg_take_probs(model: LogisticRegression, ds: CandidateWindowDataset) -> np.ndarray:
    X = np.stack([window[-1] for window, _ in ds.samples])
    return model.predict_proba(X)[:, 1]
