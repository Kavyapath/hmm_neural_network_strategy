from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PullbackConfig:
    ma_window: int = 50          # trend filter MA length
    rel_strength_window: int = 20   # trailing-return comparison window
    pullback_window: int = 3        # sessions t-3..t-1 checked for the dip
    prior_high_lookback: int = 10   # how far back close[t-4] must be a local high
    min_rules_required: int = 5     # how many of the 5 rules must be True; 5 = original strict AND


def _trailing_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window)


RULE_DESCRIPTIONS = {
    "rule1_benchmark_trend": "benchmark is above its own moving average (market-wide uptrend)",
    "rule2_stock_trend": "the stock itself is above its moving average (stock is in an uptrend)",
    "rule3_rel_strength": "the stock's trailing return beats the benchmark's (relative strength)",
    "rule4_pullback": "price made a recent high and has since pulled back below it",
    "rule5_trigger": "today's close broke back above yesterday's high (pullback resolved, trigger fired)",
}


def candidate_signal_detail(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    stock_high: pd.Series,
    cfg: PullbackConfig = PullbackConfig(),
) -> pd.DataFrame:
    """Same five entry rules as `candidate_signal`, but returns every rule's
    boolean column, the count of rules passed, and the combined `candidate`
    column (rules_passed >= cfg.min_rules_required) - so a caller can explain
    *why* a given day did or didn't fire. Used by cli.py's predict-next
    command to justify a buy / no-buy call in plain text."""
    stock_close, benchmark_close, stock_high = (
        s.reindex(stock_close.index) for s in (stock_close, benchmark_close, stock_high)
    )

    stock_ma = stock_close.rolling(cfg.ma_window).mean()
    bench_ma = benchmark_close.rolling(cfg.ma_window).mean()

    rule1_benchmark_trend = benchmark_close > bench_ma
    rule2_stock_trend = stock_close > stock_ma

    stock_ret20 = _trailing_return(stock_close, cfg.rel_strength_window)
    bench_ret20 = _trailing_return(benchmark_close, cfg.rel_strength_window)
    rule3_rel_strength = stock_ret20 > bench_ret20

    prior_high = stock_close.shift(4)
    local_max_before_prior_high = (
        stock_close.shift(5).rolling(cfg.prior_high_lookback).max()
    )
    was_prior_high = prior_high >= local_max_before_prior_high

    pullback_min = pd.concat(
        [stock_close.shift(k) for k in range(1, cfg.pullback_window + 1)], axis=1
    ).min(axis=1)
    rule4_pullback = was_prior_high & (pullback_min < prior_high)

    rule5_trigger = stock_close > stock_high.shift(1)

    detail = pd.DataFrame({
        "rule1_benchmark_trend": rule1_benchmark_trend,
        "rule2_stock_trend": rule2_stock_trend,
        "rule3_rel_strength": rule3_rel_strength,
        "rule4_pullback": rule4_pullback,
        "rule5_trigger": rule5_trigger,
    }).fillna(False)
    detail["rules_passed"] = detail.sum(axis=1)
    detail["candidate"] = detail["rules_passed"] >= cfg.min_rules_required
    return detail


def candidate_signal(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    stock_high: pd.Series,
    cfg: PullbackConfig = PullbackConfig(),
) -> pd.Series:
    """Returns a boolean Series aligned to stock_close.index: True on days
    where at least `cfg.min_rules_required` of the 5 entry rules fire
    (default 5 = original strict AND of all of them). stock_high must be the
    stock's own daily high (for rule 5, today's close vs yesterday's high)."""
    return candidate_signal_detail(stock_close, benchmark_close, stock_high, cfg)["candidate"]