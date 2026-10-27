"""Download and cache Binance.US candles."""
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.binance.us/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "6h": 21_600_000, "1d": 86_400_000}


def fetch(symbol="BTCUSDT", interval="1h", refresh=False, start=None):
    """Return OHLCV candles indexed by open time (UTC), cached as CSV.

    `start` (e.g. "2024-10-01") limits how far back short timeframes go.
    """
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{symbol}_{interval}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="time", parse_dates=True)

    rows = []
    start = int(pd.Timestamp(start).timestamp() * 1000) if start else 0
    while True:
        r = requests.get(API, params={"symbol": symbol, "interval": interval,
                                      "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + INTERVAL_MS[interval]
        if len(batch) < 1000:
            break
        time.sleep(0.1)

    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").astype(float)
    # Drop the still-forming candle so signals only use closed bars.
    df = df[df.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) <= pd.Timestamp.utcnow().tz_localize(None)]
    df.to_csv(path)
    return df


CB_API = "https://api.exchange.coinbase.com/products/{}/candles"


def fetch_coinbase(product="BTC-USD", interval="1d", start="2015-01-01", refresh=False):
    """Long-history candles from Coinbase (back to 2015) for higher-timeframe context."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"cb_{product}_{interval}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="time", parse_dates=True)

    step = INTERVAL_MS[interval] // 1000
    t = pd.Timestamp(start)
    end = pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    while t < end:
        t2 = min(t + pd.Timedelta(seconds=step * 300), end)
        r = requests.get(CB_API.format(product), timeout=30, params={
            "granularity": step, "start": t.isoformat(), "end": t2.isoformat()})
        r.raise_for_status()
        rows += r.json()
        t = t2
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index + pd.Timedelta(seconds=step) <= end]
    df.to_csv(path)
    return df


def resample(df, rule):
    """Combine candles into a longer timeframe, e.g. 3m -> 6m ("6min")."""
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def to_weekly(daily):
    """Monday-open weekly candles, matching TradingView's weekly chart."""
    return daily.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
