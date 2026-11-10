
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeSimConfig:
    atr_window: int = 14
    stop_atr_mult: float = 1.5
    target_r_mult: float = 2.0       # multiple of INITIAL STOP DISTANCE, not of ATR directly
    max_holding_sessions: int = 7     # entry session counts as session 1
    commission_bps: float = 1.0
    slippage_bps: float = 2.0


@dataclass
class TradeResult:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str          # "stop" | "target" | "timeout" | "unresolved"
    net_return: float | None
    r_multiple: float | None
    holding_sessions: int | None
    ambiguous_ordering: bool
    label: int | None         # 1 / 0, or None if unresolved (drop, don't label)
    benchmark_return: float | None
    ticker: str | None = None


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _apply_costs(price: float, side: str, cfg: TradeSimConfig) -> float:
    """side: 'buy' (entry) pays slippage upward, 'sell' (exit) pays it downward.
    Commission applied as a flat bps hit on the fill regardless of side."""
    slip = price * cfg.slippage_bps / 1e4
    comm = price * cfg.commission_bps / 1e4
    if side == "buy":
        return price + slip + comm
    return price - slip - comm


def simulate_trade(
    signal_date: pd.Timestamp,
    ohlc: pd.DataFrame,       # columns: open, high, low, close; index = trading calendar
    benchmark_close: pd.Series,
    cfg: TradeSimConfig = TradeSimConfig(),
    ticker: str | None = None,
) -> TradeResult:
    """Simulates ONE trade from a single candidate signal_date. ohlc must
    contain signal_date and at least max_holding_sessions+1 sessions after it
    for a fully-resolved result; otherwise returns an "unresolved" result."""
    dates = ohlc.index
    sig_pos = dates.get_loc(signal_date)

    atr_series = average_true_range(ohlc["high"], ohlc["low"], ohlc["close"], cfg.atr_window)
    atr_at_signal = atr_series.iloc[sig_pos]

    entry_pos = sig_pos + 1
    if entry_pos >= len(dates) or pd.isna(atr_at_signal):
        return TradeResult(signal_date, None, None, None, None, "unresolved",
                            None, None, None, False, None, None, ticker)

    entry_date = dates[entry_pos]
    raw_entry_price = ohlc["open"].iloc[entry_pos]
    entry_price = _apply_costs(raw_entry_price, "buy", cfg)

    stop = raw_entry_price - cfg.stop_atr_mult * atr_at_signal
    stop_distance = raw_entry_price - stop
    target = raw_entry_price + cfg.target_r_mult * stop_distance

    last_needed_pos = entry_pos + cfg.max_holding_sessions - 1
    if last_needed_pos >= len(dates):
        return TradeResult(signal_date, entry_date, entry_price, None, None, "unresolved",
                            None, None, None, False, None, None, ticker)

    for k in range(cfg.max_holding_sessions):
        pos = entry_pos + k
        holding_session = k + 1  # entry session counts as session 1
        bar = ohlc.iloc[pos]
        ambiguous = False

        if holding_session == 1:
            # same-session check uses the entry bar's own high/low AFTER entry at the open;
            # gap-through-open-on-entry-day is handled by raw_entry_price itself.
            pass

        gapped_through_stop = bar["open"] <= stop
        gapped_through_target = bar["open"] >= target

        if gapped_through_stop:
            exit_price_raw, reason = bar["open"], "stop"
        elif gapped_through_target:
            exit_price_raw, reason = bar["open"], "target"
        else:
            hit_stop = bar["low"] <= stop
            hit_target = bar["high"] >= target
            if hit_stop and hit_target:
                ambiguous = True
                exit_price_raw, reason = stop, "stop"       # conservative convention
            elif hit_stop:
                exit_price_raw, reason = stop, "stop"
            elif hit_target:
                exit_price_raw, reason = target, "target"
            else:
                if holding_session == cfg.max_holding_sessions:
                    exit_price_raw, reason = bar["close"], "timeout"
                else:
                    continue  # no exit condition yet, move to next session

        exit_date = dates[pos]
        exit_price = _apply_costs(exit_price_raw, "sell", cfg)
        net_return = exit_price / entry_price - 1
        r_multiple = (exit_price - raw_entry_price) / stop_distance

        bench_entry = benchmark_close.loc[entry_date]
        bench_exit = benchmark_close.loc[exit_date]
        bench_return = bench_exit / bench_entry - 1

        label = 1 if (reason == "target" and net_return > 0) else 0

        return TradeResult(
            signal_date, entry_date, entry_price, exit_date, exit_price, reason,
            net_return, r_multiple, holding_session, ambiguous, label, bench_return,
            ticker,
        )

    # Defensive: loop above always returns by the timeout branch on the last
    # iteration, so this line should be unreachable.
    raise RuntimeError("simulate_trade: holding loop exhausted without resolving — check max_holding_sessions logic.")


def build_labeled_dataset(
    candidates: pd.Series,     # boolean Series from pullback_entry.candidate_signal
    ohlc: pd.DataFrame,
    benchmark_close: pd.Series,
    cfg: TradeSimConfig = TradeSimConfig(),
    ticker: str | None = None,
) -> pd.DataFrame:
    """Runs simulate_trade() for every True day in `candidates`, drops
    unresolved trades, returns one row per resolved trade, each tagged with
    `ticker` so downstream code can disambiguate once multiple tickers'
    labeled frames are concatenated."""
    rows = []
    for signal_date in candidates.index[candidates]:
        result = simulate_trade(signal_date, ohlc, benchmark_close, cfg, ticker)
        if result.label is None:
            continue  # unresolved — dropped, not labeled as a loss
        rows.append(result.__dict__)
    return pd.DataFrame(rows)
