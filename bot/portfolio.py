"""Portfolio view: gains and losses by period, for BTC or any watchlist coin / stock.

For each period (1 week, 1 month, year to date, and each calendar year back to
2020) it reports:
  strategy  the bot's rules on that asset's own price history: DI crossover and
            ADX-14 trigger sleeves, equal shares, 0.02% fee + 0.01% slippage per
            side; coins on 4h / 8h / 12h / 1d (plus the halving bear filter for
            BTC), US stocks on 1d / 1w. Simulated, starting from $10,000.
  hold      buying at the start of the period and holding.
  account   the practice account's own numbers (BTC only, and only for periods
            the account has existed for the whole of).
"Stack" is the asset count: how much more (or less) of the asset the strategy's
dollars are worth at the end than the same dollars bought at the start.
"""
import threading
import time

import numpy as np
import pandas as pd

from bench.indicators import adx

from . import regime, stocks

START = pd.Timestamp("2019-06-01")          # history requested (Binance.US begins Sept 2019)
COST = 0.0003                               # fee + slippage per side
BASE_USD = 10_000.0
CRYPTO_TFS = ("4h", "8h", "12h", "1d")
STOCK_TFS = ("1d", "1w")
FIRST_YEAR = 2020

_lock = threading.Lock()
_cache = {}


def _cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _lock:
        _cache[key] = (time.time(), value)
    return value


def crypto_history(client, symbol, tf):
    """All closed candles since START, paging back 1000 at a time."""
    def load():
        frames, end = [], None
        while True:
            k = client.klines(symbol, tf, limit=1000, end_time=end)
            if k.empty:
                break
            if frames and k.index[0] >= frames[-1].index[0]:
                break                                       # no older candles
            frames.append(k)
            if k.index[0] <= START or len(k) < 900:     # (the first page can be one short: the forming candle)
                break
            end = int(k.index[0].timestamp() * 1000) - 1
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.concat(frames).sort_index()
        return df[~df.index.duplicated()]
    return _cached(("hist", symbol, tf), 3 * 3600, load)


def stock_history(ticker, tf):
    def load():
        df = stocks.candles(ticker, tf, limit=2600 if tf == "1d" else 600, closed_only=True)[0]
        df.index = df.index.tz_convert(None)              # naive UTC, like the crypto candles
        return df
    return _cached(("stock", ticker, tf), 3 * 3600, load)


def positions(df, kind, bear=None):
    """1 while a sleeve would hold the asset (decided on each candle's close), else 0."""
    d = adx(df, 38, 9)
    up = (d.di_plus > d.di_minus).to_numpy()
    a = d.adx.fillna(0).to_numpy()
    pos = np.zeros(len(df))
    on = False
    for i in range(1, len(df)):
        if on and not up[i]:
            on = False
        elif not on and up[i] and (not up[i - 1] if kind == "di" else a[i] > 14 >= a[i - 1]):
            on = True
        if bear is not None and bear[i]:
            on = False
        pos[i] = on
    return pd.Series(pos, df.index)


def sleeve_curve(df, pos):
    """Daily equity of one sleeve: held from the close a signal fires, costs on each switch."""
    r = df.close.pct_change().fillna(0)
    held = pos.shift(1).fillna(0)
    switches = pos.diff().abs().fillna(0)
    eq = (1 + held * r - switches * COST).cumprod()
    return eq.resample("1D").last().ffill()


def strategy_curve(frames, bear_filter):
    curves = []
    for tf, df in frames.items():
        if len(df) < 120:
            continue
        bear = np.array([regime.calendar_bear(t) for t in df.index.normalize()]) if bear_filter else None
        for kind in ("di", "adx"):
            curves.append(sleeve_curve(df, positions(df, kind, bear)))
    if not curves:
        return None
    rets = pd.concat([c.pct_change() for c in curves], axis=1, sort=True).dropna(how="all").fillna(0).mean(axis=1)
    return (1 + rets).cumprod(), len(curves)


def periods(today):
    today = pd.Timestamp(today).normalize()
    out = [("1W", "1 week", today - pd.Timedelta(days=7), None),
           ("1M", "1 month", today - pd.Timedelta(days=30), None),
           ("YTD", f"{today.year} to date", pd.Timestamp(f"{today.year}-01-01"), None)]
    for y in range(today.year - 1, FIRST_YEAR - 1, -1):
        out.append((str(y), str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31 23:59")))
    return out


def window(series, a, b):
    """Value just before the period starts, and the last value inside it."""
    before = series.loc[:a]
    inside = series.loc[a:b] if b is not None else series.loc[a:]
    if before.empty or inside.empty:
        return None
    return float(before.iloc[-1]), float(inside.iloc[-1]), before.index[-1], inside.index[-1]


def max_drawdown(series, a, b):
    w = series.loc[a:b] if b is not None else series.loc[a:]
    return float((w / w.cummax() - 1).min()) if len(w) else None


def build(symbol, frames, price, bear_filter, account=None, unit=None, name=None, sleeves_label="", holdings=None):
    """holdings: the account's (cash, btc) right now; without it the latest hourly snapshot is used."""
    sc = _cached(("sc", symbol, bear_filter), 600, lambda: strategy_curve(frames, bear_filter))   # the slow part
    daily = next(iter(frames.values())) if frames else None
    px = (daily.close.resample("1D").last().dropna() if daily is not None and len(daily) else pd.Series(dtype=float))
    if price and len(px):
        px.loc[pd.Timestamp.now("UTC").tz_localize(None).normalize()] = price   # today, at the live price
    out = []
    plist = periods(pd.Timestamp.now("UTC").tz_localize(None))
    if account is not None and len(account):           # the practice account from its first reading
        t0 = account.index[0]
        plist.insert(0, ("START", f"Since {t0:%b} {t0.day}", t0, None))
    for key, label, a, b in plist:
        row = {"key": key, "label": label, "start": str(a.date()), "end": str((b or pd.Timestamp.now()).date()),
               "strategy": None, "hold": None, "account": None}
        h = window(px, a, b)
        if h:
            p0, p1 = h[0], h[1]
            row["hold"] = {"ret": p1 / p0 - 1, "price_start": p0, "price_end": p1}
            if sc:
                s = window(sc[0], a, b)
                if s:
                    r = s[1] / s[0] - 1
                    units0, units1 = BASE_USD / p0, BASE_USD * (1 + r) / p1
                    row["strategy"] = {"ret": r, "usd": BASE_USD * r, "units_start": units0, "units_end": units1,
                                       "stack": units1 / units0 - 1, "dd": max_drawdown(sc[0], a, b)}
        if account is not None and len(account) and account.index[0] <= a + pd.Timedelta(hours=2):
            acc = account.loc[a - pd.Timedelta(hours=1):]   # the account existed for the whole period
            if len(acc):
                v0, q0 = float(acc.value.iloc[0]), float(acc.price.iloc[0])
                q1 = float(price or account.price.iloc[-1])
                cash, held = holdings if holdings else (float(account.cash.iloc[-1]), float(account.btc.iloc[-1]))
                v1 = cash + held * q1                          # what the account holds now, at the live price
                u0, u1 = v0 / q0, v1 / q1
                row["account"] = {"ret": v1 / v0 - 1, "usd": v1 - v0, "value": v1, "units_start": u0,
                                  "units_end": u1, "stack": u1 / u0 - 1, "held": held, "cash": cash}
        out.append(row)
    return {"symbol": symbol, "name": name or symbol, "unit": unit or symbol, "price": price,
            "sleeves": sc[1] if sc else 0, "sleeves_label": sleeves_label,
            "data_from": str(daily.index[0].date()) if daily is not None and len(daily) else None,
            "periods": out}


def portfolio(client, store, cfg, symbol, is_stock, price=None):
    if not is_stock and symbol == cfg.symbol:
        return _account_portfolio(client, store, cfg, symbol)   # never cached: it shows the live account

    def load():
        if is_stock:
            frames = {tf: stock_history(symbol, tf) for tf in STOCK_TFS}
            q = stocks.quote(symbol)
            return build(symbol, frames, q["price"], False, name=q["name"], unit=symbol,
                         sleeves_label="DI + ADX-14 on 1d / 1w")
        frames = {tf: crypto_history(client, symbol, tf) for tf in ("1d",) + tuple(t for t in CRYPTO_TFS if t != "1d")}
        bid, ask = client.book(symbol)
        px = float((bid + ask) / 2)
        account = None
        if symbol == cfg.symbol:
            rows = store.rows("SELECT ts, cash, btc, price, value FROM equity WHERE strategy='total' ORDER BY ts")
            if rows:
                account = pd.DataFrame(rows)
                account.index = pd.to_datetime(account.pop("ts"))
        base = symbol[:-4] if symbol.endswith(("USDT", "USDC")) else symbol[:-3]
        return build(symbol, frames, px, symbol == cfg.symbol and cfg.bear_filter == "halving", account,
                     unit=base, name=base, sleeves_label="DI + ADX-14 on 4h / 8h / 12h / 1d")
    return _cached(("portfolio", symbol), 600, load)


def _account_portfolio(client, store, cfg, symbol):
    frames = {tf: crypto_history(client, symbol, tf) for tf in ("1d",) + tuple(t for t in CRYPTO_TFS if t != "1d")}
    bid, ask = client.book(symbol)
    px = float((bid + ask) / 2)
    rows = store.rows("SELECT ts, cash, btc, price, value FROM equity WHERE strategy='total' ORDER BY ts")
    account, holdings = None, None
    if rows:
        account = pd.DataFrame(rows)
        account.index = pd.to_datetime(account.pop("ts"))
        names = [n for n, _, _ in cfg.sleeve_specs()] + ["manual"]
        holdings = (sum(float(store.get(f"{n}.cash") or 0) for n in names),
                    sum(float(store.get(f"{n}.btc") or 0) for n in names))
    base = symbol[:-4] if symbol.endswith(("USDT", "USDC")) else symbol[:-3]
    return build(symbol, frames, px, cfg.bear_filter == "halving", account, unit=base, name=base,
                 sleeves_label="DI + ADX-14 on 4h / 8h / 12h / 1d", holdings=holdings)
