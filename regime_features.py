from __future__ import annotations

import numpy as np
import pandas as pd

from causal_hmm import CausalHMM

EPS = 1e-12


def fit_regime_model(close: pd.Series, train_end: str, n_states: int = 3) -> CausalHMM:
    """Fit CausalHMM on returns up to (and including) train_end ONLY.
    This is the one and only place fit() is called — everything after this
    function runs, parameters are frozen."""
    returns = close.pct_change().dropna()
    train_returns = returns.loc[:train_end]
    model = CausalHMM(n_states=n_states)
    model.fit(train_returns.to_numpy())
    return model


def build_regime_features(close: pd.Series, model: CausalHMM) -> pd.DataFrame:
    """Runs the FROZEN model's filter causally across the whole `close`
    series (train+val+test) and returns the 8-column feature frame, indexed
    to match close.pct_change().dropna()'s index (one row shorter than close
    itself, since the first close has no return)."""
    returns = close.pct_change().dropna()
    filtered = model.filter_sequence(returns.to_numpy())          # (T, K)
    order = model.order_by_mean_return()                           # cosmetic, stable column meaning
    A = model.transition_matrix()

    filtered_ordered = filtered[:, order]
    next_probs = filtered @ A                                      # causal: uses filtered_t and frozen A
    next_probs_ordered = next_probs[:, order]

    entropy = -(filtered * np.log(filtered + EPS)).sum(axis=1) / np.log(model.n_states)

    most_likely_state = filtered.argmax(axis=1)
    persistence = A[most_likely_state, most_likely_state]

    out = pd.DataFrame(index=returns.index)
    names = ["bear", "mid", "bull"] if model.n_states == 3 else [f"s{i}" for i in range(model.n_states)]
    for i, name in enumerate(names):
        out[f"regime_{name}"] = filtered_ordered[:, i]
        out[f"regime_next_{name}"] = next_probs_ordered[:, i]
    out["regime_entropy"] = entropy
    out["regime_persistence"] = persistence
    return out