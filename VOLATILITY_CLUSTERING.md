# Volatility Clustering instead of Seasonality — a walkthrough

This is the teaching doc. `MODULES.md` has the file-by-file wiring; this one
explains the *why* and the *math* so you can implement it yourself.

## 1. What "volatility clustering" actually means

Big moves tend to follow big moves, and calm tends to follow calm — not
because of any calendar effect, but because shocks to variance decay slowly.
Formally: returns `rₜ` are close to uncorrelated (hard to predict direction),
but squared/absolute returns `rₜ²` are strongly autocorrelated (very easy to
predict *magnitude*). This is one of the most robust stylized facts in all of
financial time series — far more robust than any day-of-week or
month-of-year effect, which is why it's a better block to replace
`seasonal_features.py` with than seasonality ever was.

Seasonal features had **zero fitting step** (pure calendar math) and
therefore zero causal mechanism — there's no economic reason Tuesday should
predict a pullback outcome. Volatility clustering **does** have a mechanism:
if yesterday was violent, today's stop/target distances (computed from ATR)
are more likely to get blown through in either direction, which directly
affects your `trade_simulation.py` outcomes.

## 2. The model: GARCH(1,1)

```
rₜ = μ + εₜ,          εₜ = σₜ zₜ,   zₜ ~ i.i.d.(0, 1)
σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²
```

- `ω` — long-run variance floor (how volatile the ticker is on average)
- `α` — how much yesterday's *shock* feeds into today's variance (the
  "reaction" term — big surprise yesterday → vol jumps today)
- `β` — how much yesterday's *variance itself* persists (the "memory" term —
  high β means storms take a long time to die down)

For a typical equity, `α + β` is close to 1 (0.95–0.99) — variance is highly
persistent, which is exactly the "clustering" you're trying to capture.
You fit `(ω, α, β)` by maximum likelihood on daily log returns.

**One-step-ahead forecast:** once fit, next-day variance is just
`σ̂ₜ₊₁² = ω + α·εₜ² + β·σₜ²` using today's realized shock and variance.
Take `√(σ̂ₜ₊₁²) × √252` to get an annualized vol forecast you can reason about
in percent terms.

## 3. Walk-forward fitting — non-negotiable

Exactly the same discipline your `causal_hmm.py` already follows: the GARCH
params must never see data from after the point you're forecasting.

- Refit `(ω, α, β)` every ~21 trading days (roughly monthly) on an
  **expanding window of past-only** returns.
- Between refits, just roll the variance recursion forward one step at a
  time using only `εₜ, σₜ` up to the current day — don't re-estimate daily,
  that's expensive and unnecessary.
- Fit **inside the fold**, on train-only data, mirroring
  `regime_features.fit_regime_model(close, train_end)` — never fit once on
  train+dev+test combined. If you fit on the whole series and then slice
  it, you've leaked every future storm into the model's "known" long-run
  variance `ω`, which quietly inflates backtest performance.
- Minimum history before the first fit is trustworthy: roughly 500 daily
  bars. For a ticker like MSTR with ~1580 total rows, that's fine for the
  full series but tight if you're also holding out 2 years for test — check
  your train window has enough length before relying on this.

## 4. From a vol forecast to a volatile/neutral label

Don't use an absolute vol threshold (e.g. "flag anything above 40%
annualized") — that doesn't transfer across tickers. MSTR's "calm" day is
still far more volatile than SPY's "stormy" day. Instead, rank each day's
forecast against *its own ticker's* trailing history:

```python
vol_percentile = forecast_vol.rolling(252).rank(pct=True)

regime = np.where(vol_percentile >= 0.75, "volatile", "neutral")
```

The 0.75 cutoff is your one real tuning knob. Start there; move it if the
regime almost never fires (raise cutoff's effective bite by lowering it) or
fires on more than ~30% of days (raise it).

## 5. Three ways to use the label — pick based on your data size

This is the part worth being deliberate about, given how few trades you
currently have per ticker (MSTR had 19 train / 11 dev candidates total).

| Approach | What it does | Data cost | Risk |
|---|---|---|---|
| **A. Feature column** | Feed `vol_percentile` / regime flag into the GRU as one more input column alongside technical + regime features | Low — same model, one more column | Low: model can ignore it if it's not useful |
| **B. Hard gate** | Reject/skip any candidate trade generated on a "volatile" day, before the model even scores it | None — no new fitting, just a filter | Low: simple, falsifiable, cheap to test first |
| **C. Regime-conditional policy** | Train a *separate* model (or separate thresholds) per regime — "how to trade in high vol vs low vol" | High — splits your already-thin trade count in two | **High**: with ~11-19 trades total, splitting into two buckets likely leaves each with too few trades to fit anything reliable. This is the version in your last question, and it's the one I'd hold off on for single-ticker runs. |

**Recommended order for your situation:** B first (cheapest, tells you
immediately whether high-vol days are where your losses cluster — directly
testable against the MSTR holdout trades you already have, all 5 of which
lost money), then A if B doesn't fully explain the losses, and only consider
C once you're training on enough tickers/history that each regime bucket has
enough resolved trades to support its own fit (rule of thumb: don't fit
anything with fewer than ~30-50 resolved trades going into it).

## 6. Concrete implementation checklist

1. New file `volatility_features.py`:
   - `fit_volatility_model(close, train_end)` → fits GARCH(1,1) on
     `close.pct_change()` up to `train_end` only, returns frozen params
     (mirrors `regime_features.fit_regime_model`).
   - `build_volatility_features(close, model)` → walks the frozen model
     forward across the full series, returns a DataFrame with
     `garch_vol_fcst`, `vol_percentile_1y`, `vol_regime_flag`.
2. `feature_blocks.py`: replace `"seasonal"` with `"volatility"` in
   `VARIANT_BLOCKS`, add a `"volatility"` branch in `build_feature_frame()`
   that takes a frozen `vol_model` the same way `"regime"` takes
   `regime_model`.
3. `walk_forward.py`: call `fit_volatility_model()` once per fold (train-only),
   right next to the existing `fit_regime_model()` call, hand both frozen
   models into `build_feature_frame`.
4. If testing approach B (gate) first: this doesn't even need step 2/3 — just
   compute the vol_regime_flag once per ticker and filter `candidate_signal`'s
   output in `pullback_entry.py` or right after it, before trade_simulation
   runs. Fastest way to get a real answer with the least new code.

## 7. Full code — what goes where

Dependency note: this uses only `numpy`, `pandas`, `scipy` (already in your
venv — checked, `scipy 1.17.1` is installed). No need to add the `arch`
package; GARCH(1,1) MLE is simple enough to hand-roll with
`scipy.optimize.minimize`, and hand-rolling keeps you in full control of the
walk-forward refit loop instead of fighting a library's own windowing.

### 7.1 New file: `volatility_features.py`

Place this next to `seasonal_features.py` / `regime_features.py`.

```python
"""GARCH(1,1) volatility-clustering block. Forecasts MAGNITUDE of movement,
never direction. Fit walk-forward (expanding, train-only window, refit every
~21 sessions) exactly like causal_hmm.py's regime model — see
VOLATILITY_CLUSTERING.md for the full writeup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1e-12


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


def fit_garch(returns: np.ndarray) -> GarchParams:
    """Fits GARCH(1,1) by MLE on a 1-D array of (demeaned) daily returns.
    Caller is responsible for only passing TRAIN-WINDOW returns in."""
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
```

### 7.2 Edit `feature_blocks.py`

Replace the seasonal import/usage with volatility, and add a `volatility`
branch mirroring the existing `regime` branch (it also needs a frozen,
fold-fit model passed in):

```python
# at the top — replace this line:
from seasonal_features import build_seasonal_features
# with:
from volatility_features import build_volatility_features, GarchParams

# VARIANT_BLOCKS — replace "seasonal" with "volatility" everywhere it appears:
VARIANT_BLOCKS: dict[str, frozenset[str]] = {
    "A_rule_based": frozenset(),
    "B_logreg": frozenset({"technical", "regime", "volatility"}),
    "C_gru_no_regime": frozenset({"technical", "volatility"}),
    "D_gru_full": frozenset({"technical", "regime", "volatility"}),
    "E_gru_no_seasonal": frozenset({"technical", "regime"}),
}

# build_feature_frame() — add a volatility_model parameter and branch,
# right alongside the existing regime_model parameter/branch:
def build_feature_frame(
    close: pd.Series,
    blocks: frozenset[str],
    regime_model: CausalHMM | None = None,
    volatility_model: GarchParams | None = None,
) -> pd.DataFrame:
    frames = []
    if "technical" in blocks:
        frames.append(build_technical_features(close))
    if "regime" in blocks:
        if regime_model is None:
            raise ValueError("'regime' block requested but no regime_model was given")
        frames.append(build_regime_features(close, regime_model))
    if "volatility" in blocks:
        if volatility_model is None:
            raise ValueError("'volatility' block requested but no volatility_model was given")
        frames.append(build_volatility_features(close, volatility_model))

    if not frames:
        return pd.DataFrame(index=close.index)

    combined = pd.concat(frames, axis=1, join="inner")
    return combined.dropna()
```

### 7.3 Edit `walk_forward.py`

Wherever the fold fits the regime model (search for `fit_regime_model(`),
add the matching volatility fit right next to it, and pass both frozen
models into `build_feature_frame`:

```python
from volatility_features import fit_volatility_model   # add to imports

# inside run_fold(), near the existing:
#   regime_model = fit_regime_model(close, fold.train_end, n_states=...)
# add:
    volatility_model = fit_volatility_model(close, fold.train_end)

# then wherever build_feature_frame(...) is called, add the kwarg:
    feature_frame = build_feature_frame(
        close, blocks,
        regime_model=regime_model,
        volatility_model=volatility_model,
    )
```

### 7.4 `cli.py` / `ablation_runner.py` / `app.py` / `config*.yaml`

No code changes needed — they treat block names as data. In any config you
write, swap `"seasonal"` → `"volatility"` under `ablation.feature_blocks`:

```yaml
ablation:
  feature_blocks: ["technical", "regime", "volatility"]
```

The Streamlit mode selector in `app.py`'s `MODE_TO_BLOCKS` dict should also
get `"seasonal"` renamed to `"volatility"` in whichever label maps to
`D_gru_full`'s block set.

### 7.5 If you want to try the gate first (Section 5, Approach B) — no model wiring needed

Cheapest path, skip 7.2/7.3 entirely for now. In `pullback_entry.py`, after
`candidate_signal()` returns its boolean Series, filter it using a
once-per-ticker volatility calc (reuse `fit_garch` + `build_volatility_features`
directly, no walk-forward fold plumbing required for a quick test):

```python
from volatility_features import fit_garch, build_volatility_features

returns = stock_close.pct_change().dropna()
params = fit_garch(returns.to_numpy())          # fine for a quick test; use
                                                  # fit_volatility_model(...) with
                                                  # a real train_end once you move
                                                  # this into the walk-forward loop
vol_features = build_volatility_features(stock_close, params)

candidates = candidate_signal(stock_close, benchmark_close, stock_high, cfg)
candidates = candidates & (vol_features["vol_regime_flag"] == 0)  # drop volatile days
```

This answers "does gating out high-vol days change the MSTR holdout result"
with the least code, before committing to the full feature-block wiring.

## 8. The one thing to never claim

GARCH forecasts *magnitude*, never *direction*. It will never tell you "MSTR
is about to go up" — only "MSTR is about to move a lot, so size down / sit
out / widen stops." Keep that framing in whatever report or UI label you
attach to this (same honesty rule the garch-method skill already enforces).
