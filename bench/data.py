"""Download and cache Binance.US candles."""
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.binance.us/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def fetch(symbol="BTCUSDT", interval="1h", refresh=False):
    """Return OHLCV candles indexed by open time (UTC), cached as CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{symbol}_{interval}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="time", parse_dates=True)

    rows, start = [], 0
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
