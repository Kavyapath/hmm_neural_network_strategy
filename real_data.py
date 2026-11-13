"""Real-ticker OHLC loader via Alpaca's market-data API (not yfinance - Yahoo's
endpoint was rate-limiting this environment). Same function names/signatures as
before, so cli.py's _load_price_data needs no changes beyond the `source` string.

Credentials: reads ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY from the environment
first; if unset, falls back to parsing C:\\Users\\User\\.secrets\\alpaca_paper.env
(same file AIStrategy/live_signal/alpaca_adapter.py uses) directly, so you don't
have to remember to `source` it before launching the Streamlit app. Paper-account
keys work fine here - this is a different host (data.alpaca.markets) from the
trading API, not a different credential.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ALPACA_DATA_BASE = "https://data.alpaca.markets"
SECRETS_FILE = Path(r"C:\Users\User\.secrets\alpaca_paper.env")


def _load_alpaca_env() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if key and secret:
        return key, secret

    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not available - set them in the "
            f"environment, or make sure they're in {SECRETS_FILE}."
        )
    return key, secret


def _alpaca_get(path: str, params: dict) -> dict:
    key, secret = _load_alpaca_env()
    url = f"{ALPACA_DATA_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited - back off and retry
                last_err = e
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            raise RuntimeError(f"Alpaca data API error {e.code} for {path}: "
                                f"{e.read().decode()[:300]}") from e
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError(f"Alpaca data API request to {path} failed after retries: {last_err}")


def load_real_ohlc(ticker: str, start: str, end: str, timeframe: str = "1Day",
                    feed: str = "iex") -> pd.DataFrame:
    """Returns a DataFrame with columns [open, high, low, close], indexed by
    date - the same shape synthetic_data.py's generate_synthetic_universe()
    produces per ticker, so this is a drop-in replacement for the synthetic
    source in cli.py's _load_price_data.

    feed="iex" is what free/paper Alpaca accounts are entitled to; pass
    feed="sip" if your account has that entitlement for full-exchange data.
    """
    rows = []
    page_token = None
    while True:
        params = {
            "timeframe": timeframe, "start": start, "end": end,
            "limit": 10000, "adjustment": "all", "feed": feed,
        }
        if page_token:
            params["page_token"] = page_token
        data = _alpaca_get(f"/v2/stocks/{ticker}/bars", params)
        rows.extend(data.get("bars") or [])
        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not rows:
        raise RuntimeError(
            f"load_real_ohlc({ticker!r}): Alpaca returned no bars for {start}..{end} "
            f"(feed={feed!r}) - check the symbol, date range, and feed entitlement."
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"]).dt.tz_convert(None).dt.normalize()
    df = (
        df.set_index("date")[["o", "h", "l", "c"]]
        .rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
        .sort_index()
    )
    return df.dropna()


def load_real_universe(
    tickers: list[str], benchmark: str, start: str, end: str
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Same return shape as synthetic_data.generate_synthetic_universe:
    (price_data: dict[ticker -> OHLC DataFrame], benchmark_close: Series).
    Aligns every ticker + the benchmark onto their shared trading calendar
    (inner join) so CandidateWindowDataset never sees a date one series has
    and another doesn't."""
    price_data = {t: load_real_ohlc(t, start, end) for t in tickers}
    benchmark_ohlc = load_real_ohlc(benchmark, start, end)

    shared_index = benchmark_ohlc.index
    for ohlc in price_data.values():
        shared_index = shared_index.intersection(ohlc.index)

    price_data = {t: ohlc.loc[shared_index] for t, ohlc in price_data.items()}
    benchmark_close = benchmark_ohlc.loc[shared_index, "close"].rename(benchmark)
    return price_data, benchmark_close


if __name__ == "__main__":
    ticker, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    df = load_real_ohlc(ticker, start, end)
    print(f"{ticker}: {df.shape[0]} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    print(df.head())
