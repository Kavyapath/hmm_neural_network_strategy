"""Shared fold-splitting machinery: the expanding-window Fold definition,
purging of trades that straddle a partition boundary, and dev-set threshold
selection. Used by both cli.py's single-fold train/evaluate pipeline and
ablation_runner.py's multi-variant comparison, so the splitting/purging/
threshold rule is identical everywhere it's applied."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train_start: str
    train_end: str
    dev_start: str
    dev_end: str
    test_start: str
    test_end: str


def purge_overlapping(labeled_trades: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Drops trades whose exit_date falls after `cutoff` — i.e. whose label
    used price action from beyond the partition boundary. Apply this to a
    fold's train-window trades against train_end, and to dev-window trades
    against dev_end, before those trades are used for anything."""
    cutoff_ts = pd.Timestamp(cutoff)
    return labeled_trades[labeled_trades["exit_date"] <= cutoff_ts].reset_index(drop=True)


def select_threshold(
    dev_take_probs: np.ndarray,
    dev_net_returns: np.ndarray,
    min_trades: int,
    grid: np.ndarray = np.linspace(0.3, 0.9, 25),
) -> tuple[float | None, float | None, str | None]:
    """Returns (best_threshold, best_expectancy, warning). warning is set
    (and threshold is None) if no candidate threshold meets min_trades."""
    best_threshold, best_expectancy = None, -np.inf
    for threshold in grid:
        accepted = dev_take_probs >= threshold
        if accepted.sum() < min_trades:
            continue
        expectancy = dev_net_returns[accepted].mean()
        if expectancy > best_expectancy:
            best_threshold, best_expectancy = float(threshold), float(expectancy)
    if best_threshold is None:
        return None, None, (
            f"no threshold in grid accepted >= {min_trades} trades on dev "
            f"(max accepted at any threshold: "
            f"{max((dev_take_probs >= t).sum() for t in grid)})"
        )
    return best_threshold, best_expectancy, None
