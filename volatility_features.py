"""GARCH(1,1) volatility-clustering block. Forecasts MAGNITUDE of movement,
never direction. Fit walk-forward (expanding, train-only window, refit every
~21 sessions) exactly like causal_hmm.py's regime model — see
VOLATILITY_CLUSTERING.md for the full writeup."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

EPS = 1e-12
LJUNG_BOX_LAGS = 10
LJUNG_BOX_ALPHA = 0.05


@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float
    sigma0: float   # variance to seed the recursion with (sample var of the fit window)


def _negative_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10  # keep the optimizer inside the stationary region

    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    sigma2 = np.maximum(sigma2, EPS)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
    return -ll


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x ** 2))


def ljung_box_test(returns: np.ndarray, lags: int = LJUNG_BOX_LAGS) -> tuple[float, float]:
    """Ljung-Box Q-test for serial autocorrelation up to `lags`. Plain
    GARCH(1,1) only demeans the mean equation (fit_garch below) — that's only
    valid if returns have no significant autocorrelation left to model. This
    is the check for that assumption, not a substitute for an AR/ARX mean
    term if it fails. Returns (Q-stat, p-value); p-value < alpha rejects the
    null of no autocorrelation."""
    n = len(returns)
    acf = np.array([_autocorr(returns, k) for k in range(1, lags + 1)])
    q_stat = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    p_value = float(chi2.sf(q_stat, df=lags))
    return float(q_stat), p_value


def fit_garch(returns: np.ndarray) -> GarchParams:
    """Fits GARCH(1,1) by MLE on a 1-D array of (demeaned) daily returns.
    Caller is responsible for only passing TRAIN-WINDOW returns in."""
    q_stat, p_value = ljung_box_test(returns)
    if p_value < LJUNG_BOX_ALPHA:
        warnings.warn(
            f"Ljung-Box test rejects no-autocorrelation on this fit window "
            f"(Q={q_stat:.2f}, p={p_value:.4f} < {LJUNG_BOX_ALPHA}). Plain "
            f"GARCH only demeans the mean equation — consider an AR/ARX mean "
            f"term instead of fit_garch's unconditional demean.",
            stacklevel=2,
        )

    returns = returns - returns.mean()
    var0 = float(np.var(returns))
    x0 = np.array([0.1 * var0, 0.05, 0.90])
    bounds = [(1e-10, None), (0.0, 1.0), (0.0, 1.0)]
    result = minimize(_negative_log_likelihood, x0, args=(returns,),
                       bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta = result.x
    return GarchParams(omega=omega, alpha=alpha, beta=beta, sigma0=var0)


def fit_volatility_model(close: pd.Series, train_end: str) -> GarchParams:
    """Fit GARCH(1,1) on returns up to (and including) train_end ONLY.
    Mirrors regime_features.fit_regime_model's contract exactly: this is the
    one and only place fit() runs — everything downstream uses the frozen
    params."""
    returns = close.pct_change().dropna()
    train_returns = returns.loc[:train_end].to_numpy()
    return fit_garch(train_returns)


def _forecast_sigma2_series(returns: np.ndarray, params: GarchParams) -> np.ndarray:
    """Causal one-step-ahead variance recursion across the WHOLE series using
    FROZEN params. sigma2[t] is the forecast made using info up to t-1."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = params.sigma0
    for t in range(1, n):
        sigma2[t] = params.omega + params.alpha * returns[t - 1] ** 2 + params.beta * sigma2[t - 1]
    return sigma2


def build_volatility_features(
    close: pd.Series,
    model: GarchParams,
    periods_per_year: int = 252,
    percentile_window: int = 252,
    volatile_cutoff: float = 0.75,
) -> pd.DataFrame:
    """Runs the FROZEN model's variance recursion causally across the whole
    `close` series (train+dev+test) and returns a 3-column feature frame,
    indexed to match close.pct_change().dropna()'s index — same convention
    as regime_features.build_regime_features."""
    returns = close.pct_change().dropna()
    sigma2 = _forecast_sigma2_series(returns.to_numpy(), model)
    vol_fcst_annualized = np.sqrt(sigma2 * periods_per_year)

    vol_series = pd.Series(vol_fcst_annualized, index=returns.index)
    vol_percentile = vol_series.rolling(percentile_window, min_periods=percentile_window // 2).rank(pct=True)
    vol_regime_flag = (vol_percentile >= volatile_cutoff).astype(float)

    return pd.DataFrame({
        "garch_vol_fcst": vol_series,
        "vol_percentile_1y": vol_percentile,
        "vol_regime_flag": vol_regime_flag,
    }, index=returns.index)


def plot_heteroscedasticity(
    close: pd.Series,
    model: GarchParams,
    train_end: str,
    dev_end: str,
    out_path,
    periods_per_year: int = 252,
) -> None:
    """Saves a 2-panel PNG: daily returns on top, the frozen model's
    annualized vol forecast on the bottom, same time axis, with dashed
    vlines at train_end/dev_end. The visual point is the bottom panel
    tracking the bursty, clustered swings in the top panel — that
    clustering (volatility that is itself autocorrelated, unlike the
    returns themselves) is heteroscedasticity, and it's what GARCH is
    fit to capture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = close.pct_change().dropna()
    features = build_volatility_features(close, model, periods_per_year=periods_per_year)

    csv_path = Path(out_path).with_suffix(".csv")
    pd.DataFrame({
        "date": features.index,
        "daily_return": returns.reindex(features.index).to_numpy(),
        "garch_vol_fcst": features["garch_vol_fcst"].to_numpy(),
    }).to_csv(csv_path, index=False)

    fig, (ax_ret, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax_ret.plot(returns.index, returns.to_numpy(), linewidth=0.6, color="tab:blue")
    ax_ret.set_ylabel("daily return")
    ax_ret.set_title("Returns vs. GARCH volatility forecast (heteroscedasticity check)")
    ax_ret.axhline(0.0, linewidth=0.5, color="black")

    ax_vol.plot(
        features.index, features["garch_vol_fcst"].to_numpy(),
        linewidth=1.0, color="tab:red",
    )
    ax_vol.set_ylabel("annualized vol fcst")
    ax_vol.set_xlabel("date")

    for ax in (ax_ret, ax_vol):
        ax.axvline(pd.Timestamp(train_end), linestyle="--", color="grey", linewidth=0.8)
        ax.axvline(pd.Timestamp(dev_end), linestyle="--", color="grey", linewidth=0.8)
    ax_ret.text(pd.Timestamp(train_end), ax_ret.get_ylim()[1], "train|dev", fontsize=8, va="top")
    ax_ret.text(pd.Timestamp(dev_end), ax_ret.get_ylim()[1], "dev|test", fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)