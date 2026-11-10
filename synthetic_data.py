"""SYNTHETIC — NOT REAL MARKET DATA, NOT EVIDENCE OF PREDICTIVE PERFORMANCE.

Small, deterministic OHLC generator for smoke-testing the pipeline end to
end. Reuses the pattern from causal_hmm.py's own
`_demo_synthetic_causality_check`: a known Gaussian-emission regime process
with a fixed transition matrix and a fixed seed, so every run is exactly
reproducible. This is scaffolding to prove the pipeline runs correctly, not
a market model and not a source of any claim about real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRUE_A = np.array([
    [0.97, 0.02, 0.01],
    [0.02, 0.96, 0.02],
    [0.01, 0.02, 0.97],
])
_TRUE_MEANS = np.array([-0.010, 0.0004, 0.012])
_TRUE_STDS = np.array([0.012, 0.007, 0.011])


def _regime_switching_log_returns(n: int, seed: int) -> np.ndarray:
    """SYNTHETIC. A 3-state Gaussian-emission Markov chain with a fixed,
    known transition matrix — deterministic from `seed` alone."""
    rng = np.random.default_rng(seed)
    state = 1
    out = np.empty(n)
    for t in range(n):
        out[t] = rng.normal(_TRUE_MEANS[state], _TRUE_STDS[state])
        state = rng.choice(3, p=_TRUE_A[state])
    return out


def _ohlc_from_close(close: np.ndarray, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """SYNTHETIC. Derives plausible open/high/low bars around a given close
    series — small deterministic noise, not a real intraday model."""
    rng = np.random.default_rng(seed + 1)
    n = len(close)
    open_ = close * (1 + rng.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, size=n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def generate_synthetic_universe(
    tickers: list[str],
    benchmark: str,
    start: str,
    end: str,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """SYNTHETIC — NOT REAL MARKET DATA, NOT EVIDENCE OF PREDICTIVE PERFORMANCE.

    Returns (price_data, benchmark_close): price_data maps each ticker to an
    OHLC DataFrame, benchmark_close is a plain Series — both indexed on the
    same business-day calendar, both derived from independent synthetic
    regime-switching processes (fixed seed per symbol, so re-running this
    function with the same arguments is always bit-identical).
    """
    dates = pd.bdate_range(start, end)
    price_data: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        log_returns = _regime_switching_log_returns(len(dates), seed=seed + i * 1000)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        price_data[ticker] = _ohlc_from_close(close, dates, seed=seed + i * 1000)

    bench_log_returns = _regime_switching_log_returns(len(dates), seed=seed + 999_000)
    bench_close = 100.0 * np.exp(np.cumsum(bench_log_returns))
    benchmark_close = pd.Series(bench_close, index=dates, name=benchmark)

    return price_data, benchmark_close


if __name__ == "__main__":
    print("SYNTHETIC — NOT REAL MARKET DATA, NOT EVIDENCE OF PREDICTIVE PERFORMANCE")
    price_data, benchmark_close = generate_synthetic_universe(
        ["AAA", "BBB"], "SYN_BENCH", "2015-01-01", "2017-01-01", seed=0
    )
    for ticker, ohlc in price_data.items():
        print(ticker, ohlc.shape, ohlc.index[0].date(), "->", ohlc.index[-1].date())
    print("benchmark", benchmark_close.shape)
