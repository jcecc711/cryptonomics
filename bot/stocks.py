"""US stock candles for the watchlist and charts (analysis only; the bot trades BTC).

Data comes from Yahoo Finance's public chart endpoint: no key, regular market
hours only (9:30-16:00 New York). Intraday history is limited by Yahoo: 15m
candles go back about 60 days, 1h about 2 years; daily and weekly go back decades.

Timeframes Yahoo doesn't offer (3h, 4h, 6h, 8h, 12h) are built from 1h candles
within each trading day, like TradingView does for stocks: a 4h day is a
9:30-13:30 candle and a 13:30-16:00 one. 8h and 12h cover the whole session, so
they match the daily candle.
"""
import re
import time

import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
NY = "America/New_York"
NATIVE = {"15m": "15m", "1h": "60m", "1d": "1d", "1w": "1wk"}
BUILT = {"3h": 3, "4h": 4, "6h": 6, "8h": 8, "12h": 12}         # from 1h, per session
BAR_SEC = {"15m": 900, "60m": 3600, "1d": 6.5 * 3600, "1wk": 4 * 86400 + 17 * 3600}   # until the close
MAX_BACK = {"15m": 59 * 86400, "60m": 729 * 86400}               # Yahoo's intraday limits
ANALYSIS_TFS = ("1h", "4h", "1d", "1w")

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (compatible; cryptonomics-dashboard)"


class StockError(ValueError):
    pass


def is_ticker(s):
    return bool(TICKER.match(s or ""))


def _get(ticker, params):
    """One chart request; returns Yahoo's result dict. Kept separate so tests can replace it."""
    r = _session.get(URL + ticker, params=params, timeout=10)
    try:
        body = r.json()
    except ValueError:
        r.raise_for_status()
        raise StockError(f"bad response for {ticker}")
    err = (body.get("chart") or {}).get("error")
    if err or not body.get("chart", {}).get("result"):
        raise StockError(f"unknown stock {ticker!r}" if r.status_code == 404 or (err or {}).get("code") == "Not Found"
                         else f"{ticker}: {(err or {}).get('description', 'no data')}")
    return body["chart"]["result"][0]


def fetch(ticker, interval, start, end, closed_only=False, now=None):
    """OHLCV for [start, end] (unix seconds), UTC index, plus Yahoo's meta block."""
    res = _get(ticker, {"interval": interval, "period1": int(start), "period2": int(end),
                        "includePrePost": "false", "events": ""})
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    ts = res.get("timestamp") or []
    df = pd.DataFrame({k: q.get(k) or [None] * len(ts) for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(ts, unit="s", utc=True)).astype(float)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df.volume.fillna(0)
    df = _fold_live_point(df[~df.index.duplicated(keep="last")], interval)
    if closed_only and len(df):
        now = now or time.time()
        df = df[df.index.map(lambda t: t.timestamp() + BAR_SEC[interval] <= now)]
    if interval in ("1d", "1wk"):   # label daily / weekly candles by their date
        df.index = pd.to_datetime(df.index.tz_convert(NY).date).tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")]
    return df, res.get("meta", {})


def _on_grid(ts, interval):
    d = ts.tz_convert(NY)
    if interval == "1wk":
        return d.weekday() == 0 and d.hour == 0 and d.minute == 0
    if interval == "1d":
        return (d.hour, d.minute, d.second) in ((9, 30, 0), (0, 0, 0))
    step = 15 if interval == "15m" else 60
    return d.second == 0 and (d.minute - 30) % step == 0


def _fold_live_point(df, interval):
    """Yahoo appends the latest trade as an extra point off the candle grid (for weekly
    candles: today, not yet in the week's candle). Fold it into the candle it belongs to."""
    if len(df) < 2 or _on_grid(df.index[-1], interval):
        return df
    live, df = df.iloc[-1], df.iloc[:-1].copy()
    i = df.index[-1]
    df.loc[i, "high"] = max(df.at[i, "high"], live.high)
    df.loc[i, "low"] = min(df.at[i, "low"], live.low)
    df.loc[i, "close"] = live.close
    if interval == "1wk":
        df.loc[i, "volume"] += live.volume
    return df


def by_session(hourly, k):
    """Group 1h candles into k-hour candles that never cross a trading day."""
    if hourly.empty:
        return hourly
    day = hourly.index.tz_convert(NY).date
    pos = hourly.groupby(day).cumcount().to_numpy() // k
    keys = [f"{d}-{p}" for d, p in zip(day, pos)]
    g = hourly.groupby(keys, sort=False)
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out.index = g.apply(lambda x: x.index[0]).to_numpy()
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out.sort_index()


def _span_days(tf, limit):
    if tf in ("15m",):
        return limit / 26 * 7 / 5 * 1.15 + 4
    if tf == "1h" or tf in BUILT:
        return limit * BUILT.get(tf, 1) / 7 * 7 / 5 * 1.15 + 4
    if tf == "1d":
        return limit * 7 / 5 * 1.05 + 6
    return limit * 7 + 7


def candles(ticker, tf, end=None, limit=1000, closed_only=False, now=None):
    """(df, has_more) with at most `limit` candles ending at `end` (unix seconds, default now)."""
    now = now or time.time()
    interval = NATIVE.get(tf) or "60m"
    stop = min(end or now, now)
    floor = now - MAX_BACK[interval] if interval in MAX_BACK else 0
    start = max(stop - _span_days(tf, limit) * 86400, floor)
    if start >= stop:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), False
    df, _ = fetch(ticker, interval, start, stop, closed_only=closed_only, now=now)
    if tf in BUILT:
        df = by_session(df, BUILT[tf])
        if closed_only and len(df):   # the last group of a day that is still trading may be partial
            last = df.index[-1]
            if pd.Timestamp(now, unit="s", tz="UTC").tz_convert(NY).date() == last.tz_convert(NY).date() \
                    and now < _close_of(last):
                df = df.iloc[:-1]
    if end is not None:
        df = df[df.index.map(lambda t: t.timestamp() <= end)]
    more = len(df) > limit or start > floor + 86400
    return df.tail(limit), more and len(df) > 0


def _close_of(ts):
    d = ts.tz_convert(NY)
    return d.replace(hour=16, minute=0, second=0, microsecond=0).timestamp()


def quote(ticker, now=None):
    """Latest price, change since the previous close, day range and dollar volume."""
    now = now or time.time()
    df, meta = fetch(ticker, "1d", now - 12 * 86400, now, now=now)
    if df.empty:
        raise StockError(f"no prices for {ticker}")
    last = df.iloc[-1]
    px = float(meta.get("regularMarketPrice") or last.close)
    prev = float(df.close.iloc[-2]) if len(df) > 1 else None
    return {"symbol": ticker, "name": meta.get("longName") or meta.get("shortName") or ticker,
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
            "price": px, "change": 100 * (px / prev - 1) if prev else None,
            "high": float(last.high), "low": float(last.low), "volume": float(last.volume) * px}
