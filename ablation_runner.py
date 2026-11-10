"""Runs variants A-E through Step 7's fold machinery, logs every experiment.
See STEP_8_BASELINES_ABLATION.md. `evaluate_variant` mirrors
walk_forward.run_fold structurally (same purging, same splitting, same
threshold-selection rule) so A-E stay comparable; only the feature frame
and the fit/predict step change per variant."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feature_blocks import VARIANT_BLOCKS, build_feature_frame
from gru_classifier import CandidateWindowDataset, GRUConfig, train_take_skip_model
from logreg_baseline import fit_logreg_baseline, logreg_take_probs
from pullback_entry import PullbackConfig, candidate_signal
from regime_features import fit_regime_model
from trade_simulation import TradeSimConfig, build_labeled_dataset
from walk_forward import Fold, purge_overlapping, select_threshold
from volatility_features import fit_volatility_model


VARIANTS = ["A_rule_based", "B_logreg", "C_gru_no_regime", "D_gru_full", "E_gru_no_volatility"]

# One small, fixed search (not a sweep), per STEP_8_BASELINES_ABLATION.md's
# "A note on simple, small configuration search" — run on variant D only.
CONFIG_SEARCH_GRID = [
    {"hidden_size": hidden_size, "lr": lr}
    for hidden_size in (16, 32, 64)
    for lr in (1e-3, 3e-4)
]


@dataclass
class ExperimentLog:
    variant: str
    fold_name: str
    timestamp: str
    config: dict
    metrics: dict


def log_experiment(log_path: Path, entry: ExperimentLog) -> None:
    """Append one JSON line per experiment — never overwrite prior runs.
    'log every experiment' per the brief means this file should only grow."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def _insufficient(n_train: int, n_dev: int, n_test: int, warning: str) -> dict:
    return {
        "insufficient_data": True, "warning": warning,
        "n_train_trades": n_train, "n_dev_trades": n_dev, "n_test_trades": n_test,
        "selected_threshold": None, "dev_expectancy": None,
        "test_expectancy": None, "test_accuracy": None,
    }


def evaluate_variant(
    variant: str,
    fold: Fold,
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    sim_cfg: TradeSimConfig,
    pullback_cfg: PullbackConfig,
    gru_cfg: GRUConfig,
    min_trades_for_threshold: int,
    device: torch.device,
) -> dict:
    """Runs ONE variant on ONE fold. Reuses walk_forward's purge_overlapping
    and select_threshold unchanged — the walk-forward splits, purging, and
    threshold rule must be identical across A-E; only the candidate filter
    (feature frame + fit/predict step below) changes per variant."""
    blocks = VARIANT_BLOCKS[variant]
    train_trades_all, dev_trades_all, test_trades_all = [], [], []
    feature_frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ohlc = price_data[ticker]
        close = ohlc["close"]

        if blocks:
            regime_model = fit_regime_model(close, train_end=fold.train_end) if "regime" in blocks else None
            volatility_model = fit_volatility_model(close, fold.train_end) if "volatility" in blocks else None

            feature_frames[ticker] = build_feature_frame(
                close, blocks,
                regime_model=regime_model,
                volatility_model=volatility_model,
            )

        candidates = candidate_signal(close, benchmark_close, ohlc["high"], pullback_cfg)
        labeled = build_labeled_dataset(candidates, ohlc, benchmark_close, sim_cfg, ticker=ticker)

        train_slice = labeled[
            (labeled["signal_date"] >= fold.train_start) & (labeled["signal_date"] <= fold.train_end)
        ]
        dev_slice = labeled[
            (labeled["signal_date"] >= fold.dev_start) & (labeled["signal_date"] <= fold.dev_end)
        ]
        test_slice = labeled[
            (labeled["signal_date"] >= fold.test_start) & (labeled["signal_date"] <= fold.test_end)
        ]

        train_trades_all.append(purge_overlapping(train_slice, fold.train_end))
        dev_trades_all.append(purge_overlapping(dev_slice, fold.dev_end))
        test_trades_all.append(test_slice)

    train_trades = pd.concat(train_trades_all, ignore_index=True)
    dev_trades = pd.concat(dev_trades_all, ignore_index=True)
    test_trades = pd.concat(test_trades_all, ignore_index=True)

    if len(train_trades) < min_trades_for_threshold or len(dev_trades) < min_trades_for_threshold:
        return _insufficient(
            len(train_trades), len(dev_trades), len(test_trades),
            f"{variant}/{fold.name}: too few train/dev trades to proceed "
            f"(train={len(train_trades)}, dev={len(dev_trades)}, need >= {min_trades_for_threshold})",
        )

    if variant == "A_rule_based":
        # Every candidate is taken unconditionally — take_prob is implicitly
        # 1.0 for all of them — so there is no threshold to select and
        # nothing from torch/sklearn is ever touched.
        dev_expectancy = float(dev_trades["net_return"].mean())
        test_expectancy = float(test_trades["net_return"].mean()) if len(test_trades) else None
        test_accuracy = float(test_trades["label"].mean()) if len(test_trades) else None
        return {
            "insufficient_data": False, "warning": None,
            "n_train_trades": len(train_trades), "n_dev_trades": len(dev_trades), "n_test_trades": len(test_trades),
            "selected_threshold": None, "dev_expectancy": dev_expectancy,
            "test_expectancy": test_expectancy, "test_accuracy": test_accuracy,
        }

    train_ds = CandidateWindowDataset(train_trades, feature_frames, gru_cfg.lookback)
    dev_ds = CandidateWindowDataset(dev_trades, feature_frames, gru_cfg.lookback)

    if variant == "B_logreg":
        model = fit_logreg_baseline(train_ds)
        dev_probs = logreg_take_probs(model, dev_ds)
    else:  # C_gru_no_regime, D_gru_full, E_gru_no_volatility
        n_features = len(next(iter(feature_frames.values())).columns)
        local_gru_cfg = replace(gru_cfg, n_features=n_features)
        model, _history = train_take_skip_model(train_ds, dev_ds, local_gru_cfg, device)
        model.eval()
        with torch.no_grad():
            dev_logits = model(torch.stack([dev_ds[i][0] for i in range(len(dev_ds))]).to(device))
            dev_probs = model.take_prob(dev_logits).cpu().numpy()

    dev_net_returns = dev_trades["net_return"].to_numpy()[: len(dev_probs)]
    threshold, dev_expectancy, warning = select_threshold(dev_probs, dev_net_returns, min_trades_for_threshold)

    if threshold is None:
        return _insufficient(len(train_trades), len(dev_trades), len(test_trades), warning)

    test_ds = CandidateWindowDataset(test_trades, feature_frames, gru_cfg.lookback)
    if variant == "B_logreg":
        test_probs = logreg_take_probs(model, test_ds)
    else:
        with torch.no_grad():
            test_logits = model(torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device))
            test_probs = model.take_prob(test_logits).cpu().numpy()

    test_net_returns = test_trades["net_return"].to_numpy()[: len(test_probs)]
    test_labels = test_trades["label"].to_numpy()[: len(test_probs)]
    accepted_test = test_probs >= threshold
    test_expectancy = float(test_net_returns[accepted_test].mean()) if accepted_test.any() else None
    test_accuracy = (
        float(((test_probs >= threshold).astype(int) == test_labels).mean()) if len(test_labels) else None
    )

    return {
        "insufficient_data": False, "warning": None,
        "n_train_trades": len(train_trades), "n_dev_trades": len(dev_trades), "n_test_trades": len(test_trades),
        "selected_threshold": threshold, "dev_expectancy": dev_expectancy,
        "test_expectancy": test_expectancy, "test_accuracy": test_accuracy,
    }


def benchmark_buy_and_hold(benchmark_close: pd.Series, fold: Fold) -> dict:
    """Buy-and-hold on the benchmark over the fold's test window — not a
    candidate-based strategy at all, the thing every variant above has to
    beat to be worth anything."""
    start_price = benchmark_close.asof(pd.Timestamp(fold.test_start))
    end_price = benchmark_close.asof(pd.Timestamp(fold.test_end))
    return {"test_expectancy": float(end_price / start_price - 1)}


def run_all_variants(
    folds: list[Fold],
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    sim_cfg: TradeSimConfig,
    pullback_cfg: PullbackConfig,
    gru_cfg: GRUConfig,
    min_trades_for_threshold: int,
    device: torch.device,
    log_path: Path,
) -> list[dict]:
    """Loop variants x folds, dispatch to evaluate_variant, plus the
    buy-and-hold benchmark, logging + collecting every result."""
    results = []
    for variant in VARIANTS:
        for fold in folds:
            metrics = evaluate_variant(
                variant, fold, tickers, price_data, benchmark_close,
                sim_cfg, pullback_cfg, gru_cfg, min_trades_for_threshold, device,
            )
            log_experiment(log_path, ExperimentLog(
                variant=variant, fold_name=fold.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config=asdict(gru_cfg) if variant != "A_rule_based" else {},
                metrics=metrics,
            ))
            results.append({"variant": variant, "fold": fold.name, **metrics})

    for fold in folds:
        metrics = benchmark_buy_and_hold(benchmark_close, fold)
        log_experiment(log_path, ExperimentLog(
            variant="BENCHMARK_buy_and_hold", fold_name=fold.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={}, metrics=metrics,
        ))
        results.append({"variant": "BENCHMARK_buy_and_hold", "fold": fold.name, **metrics})
    return results


def run_config_search(
    folds: list[Fold],
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    sim_cfg: TradeSimConfig,
    pullback_cfg: PullbackConfig,
    base_gru_cfg: GRUConfig,
    min_trades_for_threshold: int,
    device: torch.device,
    log_path: Path,
) -> dict:
    """Small config search on variant D only (hidden_size x lr), picked by
    best mean DEV expectancy aggregated across folds. Never reads anything
    derived from the test window — see test_config_search_never_reads_test_metrics."""
    best_config, best_dev_expectancy = None, float("-inf")
    for config in CONFIG_SEARCH_GRID:
        trial_gru_cfg = replace(base_gru_cfg, **config)
        dev_expectancies = []
        for fold in folds:
            metrics = evaluate_variant(
                "D_gru_full", fold, tickers, price_data, benchmark_close,
                sim_cfg, pullback_cfg, trial_gru_cfg, min_trades_for_threshold, device,
            )
            log_experiment(log_path, ExperimentLog(
                variant="D_gru_full_search", fold_name=fold.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config=config, metrics=metrics,
            ))
            if not metrics["insufficient_data"] and metrics["dev_expectancy"] is not None:
                dev_expectancies.append(metrics["dev_expectancy"])

        mean_dev_expectancy = float(np.mean(dev_expectancies)) if dev_expectancies else float("-inf")
        if mean_dev_expectancy > best_dev_expectancy:
            best_config, best_dev_expectancy = config, mean_dev_expectancy

    return {"best_config": best_config, "best_dev_expectancy": best_dev_expectancy}
