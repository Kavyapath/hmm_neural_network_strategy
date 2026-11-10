"""CLI entry point. Usage:
    python cli.py preprocess --config config.yaml --out runs/<run_id>/
    python cli.py train       --config config.yaml --run-id <run_id>
    python cli.py evaluate    --run-id <run_id>
    python cli.py signals     --run-id <run_id> --as-of 2025-06-01

Wires Steps 1-9's already-tested modules together. See
STEP_10_PACKAGING.md and README.md's "Limitations" section for exactly
what this does and does not do — in particular, there is no real-ticker
downloader in Practice/ yet; every command runs on synthetic_data.py's
output (SYNTHETIC — NOT REAL MARKET DATA), selected deterministically by
config.yaml's data.seed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ablation_runner import ExperimentLog, log_experiment
from causal_hmm import CausalHMM
from feature_blocks import build_feature_frame
from gru_classifier import CandidateWindowDataset, GRUConfig, TakeSkipGRU, train_take_skip_model
from portfolio_backtest import PortfolioConfig, run_portfolio_backtest
from pullback_entry import PullbackConfig, RULE_DESCRIPTIONS, candidate_signal, candidate_signal_detail
from real_data import load_real_universe
from regime_features import fit_regime_model
from synthetic_data import generate_synthetic_universe
from trade_simulation import TradeSimConfig, average_true_range, build_labeled_dataset
from volatility_features import GarchParams, fit_volatility_model, plot_heteroscedasticity
from walk_forward import Fold, purge_overlapping, select_threshold


# ---------------------------------------------------------------------------
# Shared helpers (used by more than one subcommand)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("device: cuda requested but not available on this machine — falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _load_price_data(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Default source is synthetic_data.py (SYNTHETIC — NOT REAL MARKET DATA),
    deterministic by config.yaml's data.seed. Set `data.source: yfinance` in
    the config to pull real OHLC via real_data.py instead — off by default so
    every existing test and the smoke test stay network-independent."""
    d = cfg["data"]
    if d.get("source") in ("alpaca", "yfinance"):  # "yfinance" kept as an alias for old configs
        return load_real_universe(d["tickers"], d["benchmark"], d["start"], d["end"])
    return generate_synthetic_universe(
        d["tickers"], d["benchmark"], d["start"], d["end"], seed=d.get("seed", 0)
    )


def _single_fold(dates: pd.DatetimeIndex, name: str = "fold-0") -> Fold:
    """One expanding-window fold, split 60/20/20 by position along the
    available calendar. Step 7/8's multi-fold walk-forward loop is for
    ablation research; the CLI trains/evaluates one fold at a time."""
    n = len(dates)
    train_end_pos = int(n * 0.6)
    dev_end_pos = int(n * 0.8)
    return Fold(
        name=name,
        train_start=str(dates[0].date()), train_end=str(dates[train_end_pos].date()),
        dev_start=str(dates[train_end_pos + 1].date()), dev_end=str(dates[dev_end_pos].date()),
        test_start=str(dates[dev_end_pos + 1].date()), test_end=str(dates[-1].date()),
    )


def _build_feature_frames(
    cfg: dict, price_data: dict[str, pd.DataFrame], fold: Fold
) -> tuple[dict[str, pd.DataFrame], dict[str, CausalHMM], dict[str, GarchParams]]:
    blocks = frozenset(cfg["ablation"]["feature_blocks"])
    n_states = cfg["regime"]["n_states"]
    feature_frames: dict[str, pd.DataFrame] = {}
    regime_models: dict[str, CausalHMM] = {}
    volatility_models: dict[str, GarchParams] = {}
    for ticker, ohlc in price_data.items():
        close = ohlc["close"]
        regime_model = None
        if "regime" in blocks:
            regime_model = fit_regime_model(close, train_end=fold.train_end, n_states=n_states)
            regime_models[ticker] = regime_model
        volatility_model = None
        if "volatility" in blocks:
            volatility_model = fit_volatility_model(close, fold.train_end)
            volatility_models[ticker] = volatility_model
        feature_frames[ticker] = build_feature_frame(
            close, blocks,
            regime_model=regime_model,
            volatility_model=volatility_model,
        )
    return feature_frames, regime_models, volatility_models


def _labeled_trades_by_fold(
    cfg: dict, price_data: dict[str, pd.DataFrame], benchmark_close: pd.Series, fold: Fold
) -> dict[str, pd.DataFrame]:
    pullback_cfg = PullbackConfig(**cfg["pullback"])
    sim_cfg = TradeSimConfig(**cfg["trade_sim"])
    train_all, dev_all, test_all = [], [], []
    for ticker, ohlc in price_data.items():
        candidates = candidate_signal(ohlc["close"], benchmark_close, ohlc["high"], pullback_cfg)
        labeled = build_labeled_dataset(candidates, ohlc, benchmark_close, sim_cfg, ticker=ticker)
        train_slice = labeled[(labeled["signal_date"] >= fold.train_start) & (labeled["signal_date"] <= fold.train_end)]
        dev_slice = labeled[(labeled["signal_date"] >= fold.dev_start) & (labeled["signal_date"] <= fold.dev_end)]
        test_slice = labeled[(labeled["signal_date"] >= fold.test_start) & (labeled["signal_date"] <= fold.test_end)]
        train_all.append(purge_overlapping(train_slice, fold.train_end))
        dev_all.append(purge_overlapping(dev_slice, fold.dev_end))
        test_all.append(test_slice)
    return {
        "train": pd.concat(train_all, ignore_index=True),
        "dev": pd.concat(dev_all, ignore_index=True),
        "test": pd.concat(test_all, ignore_index=True),
    }


def _hmm_to_json(model: CausalHMM) -> dict:
    return {
        "n_states": model.n_states,
        "pi": model.pi_.tolist(),
        "A": model.A_.tolist(),
        "means": model.means_.tolist(),
        "vars": model.vars_.tolist(),
    }


def _hmm_from_json(d: dict) -> CausalHMM:
    model = CausalHMM(n_states=d["n_states"])
    model.pi_ = np.array(d["pi"])
    model.A_ = np.array(d["A"])
    model.means_ = np.array(d["means"])
    model.vars_ = np.array(d["vars"])
    return model


def _garch_to_json(model: GarchParams) -> dict:
    return asdict(model)


def _garch_from_json(d: dict) -> GarchParams:
    return GarchParams(**d)


def _plot_training_curves(history: dict, out_path: Path) -> None:
    """2-panel PNG: loss (top, lower-is-better) and accuracy (bottom,
    higher-is-better) vs. epoch, train and val overlaid on each with a
    legend, so a dropping loss line next to a rising accuracy line is
    visible as the same story told twice, not two unrelated numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    ax_loss.plot(epochs, history["train_loss"], marker="o", markersize=3, color="tab:blue", label="train_loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", markersize=3, color="tab:orange", label="val_loss")
    ax_loss.set_title("Training & validation loss (lower is better)")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right")
    ax_loss.annotate(
        f"{history['train_loss'][-1]:.4f}", (epochs[-1], history["train_loss"][-1]),
        textcoords="offset points", xytext=(6, 4), color="tab:blue", fontsize=9,
    )
    ax_loss.annotate(
        f"{history['val_loss'][-1]:.4f}", (epochs[-1], history["val_loss"][-1]),
        textcoords="offset points", xytext=(6, -10), color="tab:orange", fontsize=9,
    )

    ax_acc.plot(epochs, history["train_acc"], marker="o", markersize=3, color="tab:green", label="train_acc")
    ax_acc.plot(epochs, history["val_acc"], marker="o", markersize=3, color="tab:red", label="val_acc")
    ax_acc.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, label="chance (0.5)")
    ax_acc.set_title("Training & validation accuracy (higher is better)")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="lower right")
    ax_acc.annotate(
        f"{history['train_acc'][-1]:.3f}", (epochs[-1], history["train_acc"][-1]),
        textcoords="offset points", xytext=(6, 4), color="tab:green", fontsize=9,
    )
    ax_acc.annotate(
        f"{history['val_acc'][-1]:.3f}", (epochs[-1], history["val_acc"][-1]),
        textcoords="offset points", xytext=(6, -10), color="tab:red", fontsize=9,
    )

    fig.suptitle(f"Take/skip GRU training ({len(epochs)} epochs run)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_loss_variance(history: dict, out_path: Path) -> None:
    """1-panel PNG: std of the per-trade BCE loss, train vs. val, per epoch.
    _plot_training_curves's loss panel shows the MEAN error per epoch; two
    epochs with the same mean can still look very different underneath - one
    where every trade is off by about the same amount (homoscedastic), or
    one where most trades are nearly perfect and a few are wildly wrong
    (heteroscedastic). This chart is what tells those two cases apart. Only
    called when history has train_loss_std/val_loss_std (older runs trained
    before this was added won't)."""
    if "train_loss_std" not in history or "val_loss_std" not in history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss_std"]) + 1))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(epochs, history["train_loss_std"], marker="o", markersize=3, color="tab:blue",
            label="train loss std (spread across trades)")
    ax.plot(epochs, history["val_loss_std"], marker="o", markersize=3, color="tab:orange",
            label="val loss std (spread across trades)")
    ax.annotate(
        f"{history['train_loss_std'][-1]:.4f}", (epochs[-1], history["train_loss_std"][-1]),
        textcoords="offset points", xytext=(6, 4), color="tab:blue", fontsize=9,
    )
    ax.annotate(
        f"{history['val_loss_std'][-1]:.4f}", (epochs[-1], history["val_loss_std"][-1]),
        textcoords="offset points", xytext=(6, -10), color="tab:orange", fontsize=9,
    )
    ax.set_title("Per-trade loss variance per epoch - is the model's error size constant "
                 "across trades, or does it swing wildly for some? (training-phase "
                 "heteroscedasticity, distinct from the GARCH volatility-clustering check)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("std of per-trade BCE loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _score(model: TakeSkipGRU, ds: CandidateWindowDataset, device: torch.device) -> np.ndarray:
    if len(ds) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        windows = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        logits = model(windows)
        return model.take_prob(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    price_data, benchmark_close = _load_price_data(cfg)
    pullback_cfg = PullbackConfig(**cfg["pullback"])
    sim_cfg = TradeSimConfig(**cfg["trade_sim"])

    rows = []
    for ticker, ohlc in price_data.items():
        candidates = candidate_signal(ohlc["close"], benchmark_close, ohlc["high"], pullback_cfg)
        labeled = build_labeled_dataset(candidates, ohlc, benchmark_close, sim_cfg, ticker=ticker)
        rows.append(labeled)
    all_candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.parquet"
    all_candidates.to_parquet(out_path)
    print(f"preprocess: wrote {len(all_candidates)} labeled candidates for "
          f"{cfg['data']['tickers']} -> {out_path}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    device = _resolve_device(args.device)
    price_data, benchmark_close = _load_price_data(cfg)
    if args.train_start and args.train_end and args.dev_start and args.dev_end:
        # Explicit train/dev boundaries (e.g. from the Streamlit UI) instead of the
        # automatic 60/20/20 split. `test` is set equal to `dev` here - this run's
        # own "test" slice is just the in-sample dev reused for sanity, not a real
        # held-out period; use `evaluate-holdout` for a genuine future-period test.
        fold = Fold(
            name="custom", train_start=args.train_start, train_end=args.train_end,
            dev_start=args.dev_start, dev_end=args.dev_end,
            test_start=args.dev_start, test_end=args.dev_end,
        )
    else:
        fold = _single_fold(benchmark_close.index)
    feature_frames, regime_models, volatility_models = _build_feature_frames(cfg, price_data, fold)
    trades = _labeled_trades_by_fold(cfg, price_data, benchmark_close, fold)

    min_trades = cfg["walk_forward"]["min_trades_for_threshold"]
    log_path = run_dir / "experiment_log.jsonl"
    metadata = {
        "run_id": args.run_id,
        "device": str(device),
        "seed": cfg["model"]["seed"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "fold": asdict(fold),
    }

    if len(trades["train"]) < min_trades or len(trades["dev"]) < min_trades:
        metadata["insufficient_data"] = True
        metadata["warning"] = (
            f"too few train/dev trades to proceed "
            f"(train={len(trades['train'])}, dev={len(trades['dev'])}, need >= {min_trades})"
        )
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        log_experiment(log_path, ExperimentLog(
            variant="cli_train", fold_name=fold.name, timestamp=metadata["timestamp"],
            config=cfg["model"], metrics={"insufficient_data": True},
        ))
        print(f"train: insufficient data — {metadata['warning']}")
        return

    n_features = len(next(iter(feature_frames.values())).columns)
    gru_cfg = GRUConfig(n_features=n_features, **cfg["model"])

    train_ds = CandidateWindowDataset(trades["train"], feature_frames, gru_cfg.lookback)
    dev_ds = CandidateWindowDataset(trades["dev"], feature_frames, gru_cfg.lookback)
    model, history = train_take_skip_model(train_ds, dev_ds, gru_cfg, device)

    dev_probs = _score(model, dev_ds, device)
    dev_net_returns = trades["dev"]["net_return"].to_numpy()[: len(dev_probs)]
    lo, hi, n_grid = cfg["walk_forward"]["threshold_grid"]
    grid = np.linspace(lo, hi, int(n_grid))
    threshold, dev_expectancy, warning = select_threshold(dev_probs, dev_net_returns, min_trades, grid)

    if threshold is None:
        metadata["insufficient_data"] = True
        metadata["warning"] = warning
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        log_experiment(log_path, ExperimentLog(
            variant="cli_train", fold_name=fold.name, timestamp=metadata["timestamp"],
            config=cfg["model"], metrics={"insufficient_data": True},
        ))
        print(f"train: insufficient data — {warning}")
        return

    # model.pt — state_dict only, matching the main repo's checkpoint.py convention.
    torch.save(model.state_dict(), run_dir / "model.pt")

    # history.json — per-epoch train_loss/train_acc/val_loss/val_acc.
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # training_curves.png — loss (down) and accuracy (up) vs. epoch, side by side.
    _plot_training_curves(history, run_dir / "training_curves.png")

    # loss_variance.png — per-trade loss std per epoch (training-phase heteroscedasticity).
    _plot_loss_variance(history, run_dir / "loss_variance.png")

    # regime_model.json — one fitted CausalHMM per ticker (plain lists, JSON-stable).
    with open(run_dir / "regime_model.json", "w") as f:
        json.dump({t: _hmm_to_json(m) for t, m in regime_models.items()}, f, indent=2)

    # volatility_model.json — one fitted GARCH(1,1) per ticker, frozen at train_end.
    with open(run_dir / "volatility_model.json", "w") as f:
        json.dump({t: _garch_to_json(m) for t, m in volatility_models.items()}, f, indent=2)

    # heteroscedasticity_<ticker>.png — returns vs. the frozen vol forecast
    # over train+dev, the visual check that the GARCH fit is actually
    # tracking volatility clustering rather than just a flat average.
    for ticker, volatility_model in volatility_models.items():
        plot_heteroscedasticity(
            price_data[ticker]["close"], volatility_model,
            fold.train_end, fold.dev_end,
            run_dir / f"heteroscedasticity_{ticker}.png",
        )

    # feature_schema.json — exact ordered column names used to build tensors.
    feature_columns = list(next(iter(feature_frames.values())).columns)
    with open(run_dir / "feature_schema.json", "w") as f:
        json.dump({"columns": feature_columns, "lookback": gru_cfg.lookback}, f, indent=2)

    # scaler.json — NOT applied anywhere in Steps 3-8's actual training path
    # (no normalization layer was ever added); saved here for inspection only,
    # so this is explicitly a placeholder, not a claim that scaling happens.
    train_values = np.concatenate([
        feature_frames[t].to_numpy() for t in feature_frames
    ], axis=0)
    with open(run_dir / "scaler.json", "w") as f:
        json.dump({
            "applied_at_inference": False,
            "note": "Not applied by any Step 3-8 module — informational only.",
            "mean": train_values.mean(axis=0).tolist(),
            "std": train_values.std(axis=0).tolist(),
            "columns": feature_columns,
        }, f, indent=2)

    metadata["insufficient_data"] = False
    metadata["selected_threshold"] = threshold
    metadata["dev_expectancy"] = dev_expectancy
    metadata["warning"] = warning
    metadata["n_train_trades"] = len(trades["train"])
    metadata["n_dev_trades"] = len(trades["dev"])
    metadata["n_test_trades"] = len(trades["test"])
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log_experiment(log_path, ExperimentLog(
        variant="cli_train", fold_name=fold.name, timestamp=metadata["timestamp"],
        config=cfg["model"],
        metrics={"selected_threshold": threshold, "dev_expectancy": dev_expectancy},
    ))
    print(f"train: wrote artifacts to {run_dir} (threshold={threshold}, dev_expectancy={dev_expectancy})")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _write_trades_csv(path: Path, accepted: pd.DataFrame, take_probs: np.ndarray) -> None:
    """Writes the accepted (take_prob >= threshold) trades, sorted best-net_return
    first, so a UI can show a "best trades" table straight from the CSV."""
    out = accepted.copy()
    out["take_prob"] = take_probs[: len(out)]
    cols = [c for c in ["ticker", "signal_date", "entry_date", "exit_date",
                         "take_prob", "net_return", "label"] if c in out.columns]
    out = out[cols].sort_values("net_return", ascending=False)
    out.to_csv(path, index=False)


def _write_outcome_variance_report(
    reports_dir: Path, test_probs: np.ndarray, net_returns: np.ndarray, title_prefix: str = ""
) -> None:
    """Groups trades into take_prob buckets (same qcut binning as the calibration
    chart) and plots the STD of net_return within each bucket, not just its mean -
    i.e. whether outcome variance is roughly constant across confidence levels
    (homoscedastic) or grows/shrinks with confidence (heteroscedastic). Distinct
    from volatility_features.plot_heteroscedasticity (which checks the GARCH price-
    volatility fit): this checks whether take_prob says anything about the SIZE/
    consistency of a trade's outcome, not just its win-rate (see calibration.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if len(test_probs) >= 3:
        n_bins = min(5, max(1, len(test_probs) // 3))
        bins = pd.qcut(pd.Series(test_probs), n_bins, duplicates="drop")
        grouped = pd.Series(net_returns).groupby(bins)
        stats = grouped.agg(["mean", "std", "count"]).rename(
            columns={"mean": "mean_return", "std": "std_return", "count": "n_trades"}
        )
        stats.to_csv(reports_dir / "variance_by_bucket.csv")

        stats["std_return"].plot(
            ax=ax, kind="bar", color="#F58518",
            title=f"{title_prefix}Outcome variance: std of net_return by take_prob bucket",
        )
        ax.set_ylabel("std(net_return) within bucket")
        ax.set_xlabel("take_prob bucket (low -> high confidence)")
    fig.savefig(reports_dir / "outcome_variance.png", bbox_inches="tight")
    plt.close(fig)


def _attach_portfolio_columns(
    trades: pd.DataFrame, price_data: dict[str, pd.DataFrame], sim_cfg: TradeSimConfig, take_probs: np.ndarray
) -> pd.DataFrame:
    """portfolio_backtest.py needs `entry_price` (RAW, pre-cost) and `stop`,
    neither of which Step 5's TradeResult exposes. Both are deterministic
    functions of (ticker, signal_date, entry_date) that Step 5 already
    computed once internally; this re-derives just those two numbers from
    the same OHLC + ATR inputs, without touching trade_simulation.py."""
    out = trades.copy()
    out["take_prob"] = take_probs[: len(out)]
    raw_entry = np.empty(len(out))
    stop = np.empty(len(out))
    for ticker, ohlc in price_data.items():
        mask = (out["ticker"] == ticker).to_numpy()
        if not mask.any():
            continue
        atr_series = average_true_range(ohlc["high"], ohlc["low"], ohlc["close"], sim_cfg.atr_window)
        rows = out[mask]
        for idx in rows.index:
            sig_pos = ohlc.index.get_loc(out.at[idx, "signal_date"])
            atr_at_signal = atr_series.iloc[sig_pos]
            r = ohlc["open"].loc[out.at[idx, "entry_date"]]
            raw_entry[out.index.get_loc(idx)] = r
            stop[out.index.get_loc(idx)] = r - sim_cfg.stop_atr_mult * atr_at_signal
    out["entry_price"] = raw_entry
    out["stop"] = stop
    return out


def cmd_evaluate(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_id
    cfg = load_config(str(run_dir / "config_used.yaml"))
    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("insufficient_data"):
        print(f"evaluate: run {args.run_id} had insufficient data at train time — nothing to evaluate.")
        return

    device = _resolve_device(args.device)
    price_data, benchmark_close = _load_price_data(cfg)
    fold = Fold(**metadata["fold"])

    with open(run_dir / "regime_model.json") as f:
        regime_json = json.load(f)
    with open(run_dir / "volatility_model.json") as f:
        volatility_json = json.load(f)
    with open(run_dir / "feature_schema.json") as f:
        schema = json.load(f)

    blocks = frozenset(cfg["ablation"]["feature_blocks"])
    feature_frames = {}
    for ticker, ohlc in price_data.items():
        regime_model = _hmm_from_json(regime_json[ticker]) if ticker in regime_json else None
        volatility_model = _garch_from_json(volatility_json[ticker]) if ticker in volatility_json else None
        feature_frames[ticker] = build_feature_frame(
            ohlc["close"], blocks,
            regime_model=regime_model,
            volatility_model=volatility_model,
        )[schema["columns"]]

    trades = _labeled_trades_by_fold(cfg, price_data, benchmark_close, fold)
    test_ds = CandidateWindowDataset(trades["test"], feature_frames, schema["lookback"])

    n_features = len(schema["columns"])
    gru_cfg = GRUConfig(n_features=n_features, **cfg["model"])
    model = TakeSkipGRU(gru_cfg.n_features, gru_cfg.hidden_size, gru_cfg.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))

    test_probs = _score(model, test_ds, device)
    threshold = metadata["selected_threshold"]
    test_trades = trades["test"].iloc[: len(test_probs)].copy()
    accepted = test_trades[test_probs >= threshold].copy()
    accepted_probs = test_probs[test_probs >= threshold]

    sim_cfg = TradeSimConfig(**cfg["trade_sim"])
    portfolio_input = _attach_portfolio_columns(accepted, price_data, sim_cfg, accepted_probs)
    portfolio_cfg = PortfolioConfig(**cfg["portfolio"])
    result = run_portfolio_backtest(portfolio_input, portfolio_cfg) if len(portfolio_input) else None

    test_accuracy = float(((test_probs >= threshold).astype(int) == test_trades["label"].to_numpy()).mean()) if len(test_trades) else None
    test_expectancy = float(accepted["net_return"].mean()) if len(accepted) else None
    label_precision = float(accepted["label"].mean()) if len(accepted) else None
    profit_rate = float((accepted["net_return"] > 0).mean()) if len(accepted) else None

    bench_start = benchmark_close.asof(pd.Timestamp(fold.test_start))
    bench_end = benchmark_close.asof(pd.Timestamp(fold.test_end))
    benchmark_return = float(bench_end / bench_start - 1)

    equity_curve = result["equity_curve"] if result else pd.Series(dtype=float)
    final_equity = result["final_equity"] if result else portfolio_cfg.starting_cash
    max_drawdown = float((equity_curve / equity_curve.cummax() - 1).min()) if len(equity_curve) else 0.0

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.DataFrame([{
        "run_id": args.run_id, "fold_name": fold.name,
        "n_test_trades": len(test_trades), "n_admitted_portfolio": result["n_admitted"] if result else 0,
        "selected_threshold": threshold, "test_accuracy": test_accuracy, "test_expectancy": test_expectancy,
        "label_precision": label_precision, "profit_rate": profit_rate,
        "final_equity": final_equity, "max_drawdown": max_drawdown, "benchmark_return": benchmark_return,
    }])
    fold_metrics.to_csv(reports_dir / "fold_metrics.csv", index=False)
    _write_trades_csv(reports_dir / "accepted_trades.csv", accepted, accepted_probs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if len(equity_curve):
        equity_curve.plot(ax=ax, title="Equity curve (event-driven, entry-price valued)")
    ax.set_ylabel("equity ($)")
    fig.savefig(reports_dir / "equity_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    if len(equity_curve):
        drawdown = equity_curve / equity_curve.cummax() - 1
        drawdown.plot(ax=ax, title="Drawdown")
    ax.set_ylabel("drawdown")
    fig.savefig(reports_dir / "drawdown.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    if len(test_probs):
        n_bins = min(5, max(1, len(test_probs) // 3)) if len(test_probs) >= 3 else 1
        bins = pd.qcut(pd.Series(test_probs), n_bins, duplicates="drop")
        realized = pd.Series(test_trades["net_return"].to_numpy() > 0).groupby(bins).mean()
        realized.plot(ax=ax, kind="bar", title="Calibration: realized win-rate by take_prob bucket")
    ax.set_ylabel("fraction net_return > 0")
    fig.savefig(reports_dir / "calibration.png", bbox_inches="tight")
    plt.close(fig)

    _write_outcome_variance_report(reports_dir, test_probs, test_trades["net_return"].to_numpy())

    print(f"evaluate: wrote reports to {reports_dir}")


# ---------------------------------------------------------------------------
# evaluate-holdout: score a saved run's model on a later period it never
# trained or selected its threshold on (e.g. train/eval on 2015-2025, then
# hold out 2025-2026). Unlike `evaluate`, the price data and test-trade slice
# here are NOT the run's original fold - they're freshly pulled up to
# --test-end. The regime model stays the one fit at train time (fit_regime_model
# was already cut off at the original fold's train_end), so running it forward
# over new dates is plain causal inference, not a refit on future data.
# ---------------------------------------------------------------------------

def cmd_evaluate_holdout(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_id
    cfg = load_config(str(run_dir / "config_used.yaml"))
    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("insufficient_data"):
        print(f"evaluate-holdout: run {args.run_id} had insufficient data at train time — nothing to evaluate.")
        return

    device = _resolve_device(args.device)

    holdout_cfg = dict(cfg)
    holdout_cfg["data"] = dict(cfg["data"])
    holdout_cfg["data"]["start"] = args.buffer_start
    holdout_cfg["data"]["end"] = args.test_end
    price_data, benchmark_close = _load_price_data(holdout_cfg)

    with open(run_dir / "regime_model.json") as f:
        regime_json = json.load(f)
    with open(run_dir / "volatility_model.json") as f:
        volatility_json = json.load(f)
    with open(run_dir / "feature_schema.json") as f:
        schema = json.load(f)

    blocks = frozenset(cfg["ablation"]["feature_blocks"])
    feature_frames = {}
    for ticker, ohlc in price_data.items():
        regime_model = _hmm_from_json(regime_json[ticker]) if ticker in regime_json else None
        volatility_model = _garch_from_json(volatility_json[ticker]) if ticker in volatility_json else None
        feature_frames[ticker] = build_feature_frame(
            ohlc["close"], blocks,
            regime_model=regime_model,
            volatility_model=volatility_model,
        )[schema["columns"]]

    pullback_cfg = PullbackConfig(**cfg["pullback"])
    sim_cfg = TradeSimConfig(**cfg["trade_sim"])
    test_all = []
    for ticker, ohlc in price_data.items():
        candidates = candidate_signal(ohlc["close"], benchmark_close, ohlc["high"], pullback_cfg)
        labeled = build_labeled_dataset(candidates, ohlc, benchmark_close, sim_cfg, ticker=ticker)
        test_slice = labeled[
            (labeled["signal_date"] >= args.test_start) & (labeled["signal_date"] <= args.test_end)
        ]
        test_all.append(test_slice)
    test_trades = pd.concat(test_all, ignore_index=True) if test_all else pd.DataFrame()

    test_ds = CandidateWindowDataset(test_trades, feature_frames, schema["lookback"])

    n_features = len(schema["columns"])
    gru_cfg = GRUConfig(n_features=n_features, **cfg["model"])
    model = TakeSkipGRU(gru_cfg.n_features, gru_cfg.hidden_size, gru_cfg.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))

    test_probs = _score(model, test_ds, device)
    threshold = metadata["selected_threshold"]
    test_trades = test_trades.iloc[: len(test_probs)].copy()
    accepted = test_trades[test_probs >= threshold].copy()
    accepted_probs = test_probs[test_probs >= threshold]

    sim_cfg = TradeSimConfig(**cfg["trade_sim"])
    portfolio_input = _attach_portfolio_columns(accepted, price_data, sim_cfg, accepted_probs)
    portfolio_cfg = PortfolioConfig(**cfg["portfolio"])
    result = run_portfolio_backtest(portfolio_input, portfolio_cfg) if len(portfolio_input) else None

    test_accuracy = float(((test_probs >= threshold).astype(int) == test_trades["label"].to_numpy()).mean()) if len(test_trades) else None
    test_expectancy = float(accepted["net_return"].mean()) if len(accepted) else None
    label_precision = float(accepted["label"].mean()) if len(accepted) else None
    profit_rate = float((accepted["net_return"] > 0).mean()) if len(accepted) else None

    bench_start = benchmark_close.asof(pd.Timestamp(args.test_start))
    bench_end = benchmark_close.asof(pd.Timestamp(args.test_end))
    benchmark_return = float(bench_end / bench_start - 1)

    equity_curve = result["equity_curve"] if result else pd.Series(dtype=float)
    final_equity = result["final_equity"] if result else portfolio_cfg.starting_cash
    max_drawdown = float((equity_curve / equity_curve.cummax() - 1).min()) if len(equity_curve) else 0.0

    fold_name = f"holdout_{args.test_start}_{args.test_end}"
    reports_dir = run_dir / f"reports_{fold_name}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.DataFrame([{
        "run_id": args.run_id, "fold_name": fold_name,
        "n_test_trades": len(test_trades), "n_admitted_portfolio": result["n_admitted"] if result else 0,
        "selected_threshold": threshold, "test_accuracy": test_accuracy, "test_expectancy": test_expectancy,
        "label_precision": label_precision, "profit_rate": profit_rate,
        "final_equity": final_equity, "max_drawdown": max_drawdown, "benchmark_return": benchmark_return,
    }])
    fold_metrics.to_csv(reports_dir / "fold_metrics.csv", index=False)
    _write_trades_csv(reports_dir / "accepted_trades.csv", accepted, accepted_probs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if len(equity_curve):
        equity_curve.plot(ax=ax, title=f"Holdout equity curve ({args.test_start}..{args.test_end})")
    ax.set_ylabel("equity ($)")
    fig.savefig(reports_dir / "equity_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    if len(equity_curve):
        drawdown = equity_curve / equity_curve.cummax() - 1
        drawdown.plot(ax=ax, title="Holdout drawdown")
    ax.set_ylabel("drawdown")
    fig.savefig(reports_dir / "drawdown.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    if len(test_probs):
        n_bins = min(5, max(1, len(test_probs) // 3)) if len(test_probs) >= 3 else 1
        bins = pd.qcut(pd.Series(test_probs), n_bins, duplicates="drop")
        realized = pd.Series(test_trades["net_return"].to_numpy() > 0).groupby(bins).mean()
        realized.plot(ax=ax, kind="bar", title="Holdout calibration: realized win-rate by take_prob bucket")
    ax.set_ylabel("fraction net_return > 0")
    fig.savefig(reports_dir / "calibration.png", bbox_inches="tight")
    plt.close(fig)

    _write_outcome_variance_report(
        reports_dir, test_probs, test_trades["net_return"].to_numpy(), title_prefix="Holdout "
    )

    print(f"evaluate-holdout: {fold_name} | n_test_trades={len(test_trades)} "
          f"test_accuracy={test_accuracy} test_expectancy={test_expectancy} "
          f"final_equity={final_equity:.2f} max_drawdown={max_drawdown:.4f} "
          f"benchmark_return={benchmark_return:.4f}")
    print(f"evaluate-holdout: wrote reports to {reports_dir}")


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def cmd_signals(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_id
    cfg = load_config(str(run_dir / "config_used.yaml"))
    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("insufficient_data"):
        print(f"signals: run {args.run_id} had insufficient data at train time — nothing to score.")
        return

    device = _resolve_device(args.device)
    price_data, benchmark_close = _load_price_data(cfg)
    with open(run_dir / "regime_model.json") as f:
        regime_json = json.load(f)
    with open(run_dir / "volatility_model.json") as f:
        volatility_json = json.load(f)
    with open(run_dir / "feature_schema.json") as f:
        schema = json.load(f)

    blocks = frozenset(cfg["ablation"]["feature_blocks"])
    pullback_cfg = PullbackConfig(**cfg["pullback"])
    n_features = len(schema["columns"])
    gru_cfg = GRUConfig(n_features=n_features, **cfg["model"])
    model = TakeSkipGRU(gru_cfg.n_features, gru_cfg.hidden_size, gru_cfg.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    threshold = metadata["selected_threshold"]
    as_of = pd.Timestamp(args.as_of)

    found_any = False
    for ticker, ohlc in price_data.items():
        regime_model = _hmm_from_json(regime_json[ticker]) if ticker in regime_json else None
        volatility_model = _garch_from_json(volatility_json[ticker]) if ticker in volatility_json else None
        feature_frame = build_feature_frame(
            ohlc["close"], blocks,
            regime_model=regime_model,
            volatility_model=volatility_model,
        )[schema["columns"]]
        candidates = candidate_signal(ohlc["close"], benchmark_close, ohlc["high"], pullback_cfg)

        window_of_interest = candidates.index[(candidates.index <= as_of) & (candidates.index >= as_of - pd.Timedelta(days=10))]
        fired = candidates.loc[window_of_interest][candidates.loc[window_of_interest]]
        for sig_date in fired.index:
            if sig_date not in feature_frame.index:
                continue
            pos = feature_frame.index.get_loc(sig_date)
            if pos - schema["lookback"] + 1 < 0:
                continue
            window = feature_frame.iloc[pos - schema["lookback"] + 1: pos + 1].to_numpy(dtype=np.float32)
            with torch.no_grad():
                logit = model(torch.from_numpy(window).unsqueeze(0).to(device))
                take_prob = float(model.take_prob(logit).item())
            would_take = take_prob >= threshold
            found_any = True
            print(f"{ticker}\t{sig_date.date()}\ttake_prob={take_prob:.4f}\t"
                  f"threshold={threshold:.4f}\twould_take={would_take}")

    if not found_any:
        print(f"signals: no candidate signals for any ticker in the 10 sessions up to {args.as_of}")


# ---------------------------------------------------------------------------
# predict-next: given a saved run and an "as-of" date (normally the last day
# of a train/dev/holdout window, e.g. test_end), evaluates the pullback-entry
# rules and the GRU model AT that date and reports what it implies for the
# *next* trading session - because trade_simulation.py always enters on the
# session AFTER signal_date's open (see simulate_trade: entry_pos = sig_pos+1).
# So "signal fires on test_end" literally means "the strategy would buy at
# the open of the next session". No future price data is used or needed:
# every rule and every feature in the lookback window is computed using data
# up to and including as-of only.
# ---------------------------------------------------------------------------

def cmd_predict_next(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_id
    cfg = load_config(str(run_dir / "config_used.yaml"))
    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("insufficient_data"):
        print(f"predict-next: run {args.run_id} had insufficient data at train time - nothing to predict.")
        return

    device = _resolve_device(args.device)

    data_cfg = dict(cfg)
    data_cfg["data"] = dict(cfg["data"])
    data_cfg["data"]["start"] = args.buffer_start
    data_cfg["data"]["end"] = args.as_of
    price_data, benchmark_close = _load_price_data(data_cfg)

    with open(run_dir / "regime_model.json") as f:
        regime_json = json.load(f)
    with open(run_dir / "volatility_model.json") as f:
        volatility_json = json.load(f)
    with open(run_dir / "feature_schema.json") as f:
        schema = json.load(f)

    blocks = frozenset(cfg["ablation"]["feature_blocks"])
    pullback_cfg = PullbackConfig(**cfg["pullback"])
    n_features = len(schema["columns"])
    gru_cfg = GRUConfig(n_features=n_features, **cfg["model"])
    model = TakeSkipGRU(gru_cfg.n_features, gru_cfg.hidden_size, gru_cfg.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    threshold = metadata["selected_threshold"]

    as_of_ts = pd.Timestamp(args.as_of)
    next_session = (as_of_ts + pd.tseries.offsets.BDay(1)).date().isoformat()

    predictions = []
    for ticker, ohlc in price_data.items():
        if as_of_ts not in ohlc.index:
            predictions.append({
                "ticker": ticker, "as_of": args.as_of, "next_session": next_session,
                "status": "no_data",
                "note": f"No price bar for {ticker} on {args.as_of} (holiday/weekend/no data yet) - "
                        f"can't evaluate the entry rules for this date.",
            })
            continue

        regime_model = _hmm_from_json(regime_json[ticker]) if ticker in regime_json else None
        volatility_model = _garch_from_json(volatility_json[ticker]) if ticker in volatility_json else None
        feature_frame = build_feature_frame(
            ohlc["close"], blocks,
            regime_model=regime_model,
            volatility_model=volatility_model,
        )[schema["columns"]]
        detail = candidate_signal_detail(ohlc["close"], benchmark_close, ohlc["high"], pullback_cfg)
        row = detail.loc[as_of_ts]
        rule_cols = [c for c in detail.columns if c not in ("candidate", "rules_passed")]
        failed_rules = [c for c in rule_cols if not bool(row[c])]
        candidate_today = bool(row["candidate"])
        rules_passed = int(row["rules_passed"])

        take_prob = None
        if as_of_ts in feature_frame.index:
            pos = feature_frame.index.get_loc(as_of_ts)
            if pos - schema["lookback"] + 1 >= 0:
                window = feature_frame.iloc[pos - schema["lookback"] + 1: pos + 1].to_numpy(dtype=np.float32)
                with torch.no_grad():
                    logit = model(torch.from_numpy(window).unsqueeze(0).to(device))
                    take_prob = float(model.take_prob(logit).item())

        if not candidate_today:
            reasons = "; ".join(RULE_DESCRIPTIONS[r] + " - FAILED" for r in failed_rules)
            decision = "NO BUY"
            reason_text = (
                f"Only {rules_passed}/5 entry rules passed on {args.as_of} "
                f"(need >= {pullback_cfg.min_rules_required}), so nothing would be bought at "
                f"{next_session}'s open. Failed rule(s): {reasons}."
            )
            if take_prob is not None:
                reason_text += (
                    f" (For reference, the model's take-probability on this window is {take_prob:.1%}, "
                    f"but it's not meaningful here - the model was only trained on windows that cleared "
                    f"the min_rules_required bar.)"
                )
        elif take_prob is None:
            decision = "NO BUY"
            reason_text = (
                f"The pullback-entry pattern fired on {args.as_of} ({rules_passed}/5 rules passed, "
                f">= min_rules_required={pullback_cfg.min_rules_required}), but there isn't enough "
                f"price history yet to build the model's lookback window, so no probability can be "
                f"computed - treat as not confirmed."
            )
        elif take_prob >= threshold:
            decision = "BUY"
            reason_text = (
                f"Entry signal confirmed on {args.as_of} ({rules_passed}/5 pullback rules passed, "
                f"meeting min_rules_required={pullback_cfg.min_rules_required}). The model's "
                f"take-probability is {take_prob:.1%}, at or above the selected threshold of "
                f"{threshold:.1%}, so the strategy would buy at the open on {next_session}."
            )
        else:
            decision = "NO BUY"
            reason_text = (
                f"The pullback-entry pattern fired on {args.as_of} ({rules_passed}/5 rules passed), "
                f"but the model's take-probability is only {take_prob:.1%}, below the selected "
                f"threshold of {threshold:.1%}. The model isn't confident this particular setup has "
                f"positive edge, so the strategy would skip buying at {next_session}'s open."
            )

        predictions.append({
            "ticker": ticker, "as_of": args.as_of, "next_session": next_session,
            "status": "ok", "candidate_signal": candidate_today, "rules_passed": rules_passed,
            "min_rules_required": pullback_cfg.min_rules_required,
            "rules": {c: bool(row[c]) for c in rule_cols},
            "take_prob": take_prob, "threshold": threshold,
            "decision": decision, "reason": reason_text,
        })

    out_dir = run_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.as_of}.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    for p in predictions:
        if p["status"] == "ok":
            prob_str = f"{p['take_prob']:.4f}" if p["take_prob"] is not None else "n/a"
            print(f"predict-next\t{p['ticker']}\tas_of={p['as_of']}\tnext_session={p['next_session']}\t"
                  f"candidate={p['candidate_signal']}\trules_passed={p['rules_passed']}/5\t"
                  f"take_prob={prob_str}\tdecision={p['decision']}")
        else:
            print(f"predict-next\t{p['ticker']}\t{p['status']}: {p['note']}")
    print(f"predict-next: wrote {out_path}")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", default=str(uuid.uuid4())[:8])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-start", default=None, help="explicit fold override (all 4 required together)")
    p.add_argument("--train-end", default=None)
    p.add_argument("--dev-start", default=None)
    p.add_argument("--dev-end", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("evaluate-holdout")
    p.add_argument("--run-id", required=True)
    p.add_argument("--buffer-start", required=True,
                    help="start of the fresh data pull - earlier than --test-start to give "
                         "indicators/lookback a warm-up buffer")
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_evaluate_holdout)

    p = sub.add_parser("signals")
    p.add_argument("--run-id", required=True)
    p.add_argument("--as-of", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("predict-next")
    p.add_argument("--run-id", required=True)
    p.add_argument("--buffer-start", required=True,
                    help="start of the fresh data pull - earlier than --as-of to give "
                         "indicators/lookback a warm-up buffer")
    p.add_argument("--as-of", required=True,
                    help="the last known trading day (e.g. the test/holdout window's end date); "
                         "the prediction is for the NEXT trading session after this date")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_predict_next)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
