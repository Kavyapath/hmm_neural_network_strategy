"""Technical indicators block — the "always on" feature group every
ablation variant (Step 8) uses underneath whatever regime/volatility columns
get toggled. Formulas match the main repo's data.py (`rsi14`, `mom20`,
`mom60`, `vol20`, `macd`, `dist_ma50`), rebuilt fresh here per
STEP_6_GRU_CLASSIFIER.md's note, so Practice/ doesn't pull in data.py's
unrelated yfinance/torch/Model imports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def _mom20(close: pd.Series) -> pd.Series:
    return close.pct_change(20)


def _mom60(close: pd.Series) -> pd.Series:
    return close.pct_change(60)


def _vol20(close: pd.Series) -> pd.Series:
    return np.log(close).diff().rolling(20).std()


def _macd(close: pd.Series) -> pd.Series:
    return (close.ewm(span=12).mean() - close.ewm(span=26).mean()) / close


def _dist_ma50(close: pd.Series) -> pd.Series:
    return close / close.rolling(50).mean() - 1


def build_technical_features(close: pd.Series) -> pd.DataFrame:
    """Causal (row t uses only close[:t]) technical-indicator frame, indexed
    like `close` itself. Early rows are NaN until each indicator's own
    rolling window has enough history — dropped downstream the same way
    every other feature block's warmup NaNs are dropped."""
    return pd.DataFrame({
        "rsi14": _rsi14(close),
        "mom20": _mom20(close),
        "mom60": _mom60(close),
        "vol20": _vol20(close),
        "macd": _macd(close),
        "dist_ma50": _dist_ma50(close),
    }, index=close.index)
