"""Instead of sitting in cash after a sell, hold a small short (default 10%).

The live strategy: 8 sleeves (DI cross and ADX-14 trigger on 4h / 8h / 12h / 1d,
DI 38 / smoothing 9), halving-cycle bear filter (no longs on days 540-910 after
a halving). Each sleeve here is either:
  long   100% BTC, exactly as the bot trades
  short  -X% of the sleeve: opened when DI- crosses above DI+ (the sell), closed
         when DI+ crosses back above DI- (the DI sleeve buys at that moment; the
         ADX sleeve may stay in cash until ADX rises through 14)
  cash   otherwise
Shorts are allowed in the bear window too (longs are not).

The short's size is set at entry (X% of the sleeve's equity then) and not
rebalanced. Costs: longs 0.02% fee + 0.01% slippage per side; shorts would need
US futures: 0.05% + 0.01% per side on the short's size, plus a 10%/yr holding
cost on it (funding / basis).

    python bench/short_hedge.py
"""
import numpy as np
import pandas as pd
import requests

import data
from data import fetch_coinbase, resample
from indicators import adx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.regime import calendar_bear  # noqa: E402

TFS = {"4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
LONG_COST, SHORT_COST, CARRY = 0.0003, 0.0006, 0.10
SIZES = [0.0, 0.10, 0.25, 0.50]
PERIODS = [("2016-01-01", None, "2016 -> now"), ("2018-01-01", "2018-12-31", "2018 bear"),
           ("2019-01-01", "2021-12-31", "2019-21 bull"), ("2022-01-01", "2022-12-31", "2022 bear"),
           ("2023-01-01", "2025-10-05", "2023-Oct 25 bull"), ("2025-10-06", None, "Oct 2025 -> now (bear)")]


def hourly():
    """Cached Coinbase 1h candles, topped up to the latest closed hour."""
    df = fetch_coinbase(interval="1h")
    path = data.DATA_DIR / "cb_BTC-USD_1h.csv"
    t, end, rows = df.index[-1] + pd.Timedelta(hours=1), pd.Timestamp.now("UTC").tz_localize(None), []
    while t < end - pd.Timedelta(hours=1):
        t2 = min(t + pd.Timedelta(hours=300), end)
        r = requests.get(data.CB_API.format("BTC-USD"), timeout=30,
                         params={"granularity": 3600, "start": t.isoformat(), "end": t2.isoformat()})
        r.raise_for_status()
        rows += r.json()
        t = t2
    if rows:
        new = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        new["time"] = pd.to_datetime(new["time"], unit="s")
        new = new.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
        new = new[new.index + pd.Timedelta(hours=1) <= end]
        raw = pd.read_csv(path, index_col="time", parse_dates=True)
        raw = pd.concat([raw, new])
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        raw.to_csv(path)
        df = data.clean(raw)
    return df


def sleeve_positions(df, kind, bear):
    """+1 long / -1 short / 0 cash per candle, decided on the candle's close."""
    d = adx(df, 38, 9)
    up = (d.di_plus > d.di_minus).to_numpy()
    a = d.adx.to_numpy()
    n = len(df)
    pos = np.zeros(n, dtype=int)
    long_ = False
    for i in range(1, n):
        if kind == "di":
            enter = up[i] and not up[i - 1]
        else:
            enter = up[i] and a[i] > 14 >= a[i - 1]
        if long_ and not up[i]:
            long_ = False
        elif not long_ and enter:
            long_ = True
        if bear[i]:
            long_ = False
        pos[i] = 1 if long_ else (-1 if not up[i] else 0)
    return pos


def simulate(df, pos, size):
    """Daily equity and the list of short trades for one sleeve."""
    o, c, idx = df.open.to_numpy(), df.close.to_numpy(), df.index
    n = len(df)
    carry_bar = CARRY * pd.Timedelta(idx[1] - idx[0]).total_seconds() / (365 * 86400)
    eq = np.ones(n)
    cash, cur, entry_px, base_eq, entry_t = 1.0, 0, 0.0, 1.0, None
    shorts = []
    want = np.where(pos == -1, -size, pos.astype(float))
    want = np.where((pos == -1) & (size == 0), 0.0, want)
    for i in range(n - 1):
        tgt = want[i]
        if tgt != cur:
            px = o[i + 1]
            if cur > 0:
                cash = base_eq * px / entry_px * (1 - LONG_COST)
            elif cur < 0:
                pnl = base_eq * (-cur) * (1 - px / entry_px) - base_eq * (-cur) * SHORT_COST
                shorts.append({"opened": entry_t, "closed": idx[i + 1], "entry": entry_px, "exit": px,
                               "btc move %": 100 * (px / entry_px - 1),
                               "sleeve P/L %": 100 * ((cash + pnl) / start_cash - 1)})
                cash = cash + pnl
            cur = tgt
            if cur > 0:
                entry_px, base_eq = px, cash * (1 - LONG_COST)
            elif cur < 0:
                entry_px, entry_t, start_cash = px, idx[i + 1], cash
                cash -= cash * (-cur) * SHORT_COST
                base_eq = cash
        if cur > 0:
            eq[i] = base_eq * c[i] / entry_px
        elif cur < 0:
            cash -= base_eq * (-cur) * carry_bar
            eq[i] = cash + base_eq * (-cur) * (1 - c[i] / entry_px)
        else:
            eq[i] = cash
    eq[-1] = eq[-2]
    if cur < 0:
        shorts.append({"opened": entry_t, "closed": None, "entry": entry_px, "exit": c[-1],
                       "btc move %": 100 * (c[-1] / entry_px - 1),
                       "sleeve P/L %": 100 * ((cash + base_eq * (-cur) * (1 - c[-1] / entry_px)) / start_cash - 1)})
    return pd.Series(eq, idx).resample("1D").last().ffill(), pd.DataFrame(shorts)


def stats(eq, a, b):
    w = eq.loc[a:b]
    w = w / w.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    dd = (w / w.cummax() - 1).min()
    return 100 * (w.iloc[-1] - 1), 100 * (w.iloc[-1] ** (1 / yrs) - 1) if yrs >= 1 else np.nan, 100 * dd


def main():
    h1 = hourly()
    print(f"candles to {h1.index[-1]}, BTC ${h1.close.iloc[-1]:,.0f}")
    curves, trades = {}, {}
    for tf, rule in TFS.items():
        df = resample(h1, rule).loc["2015-06-01":]
        bear = np.array([calendar_bear(t) for t in df.index.normalize()])
        for kind in ("di", "adx"):
            pos = sleeve_positions(df, kind, bear)
            w = df.loc["2016-01-01":]
            p = pos[df.index.get_indexer(w.index)]
            for size in SIZES:
                eq, sh = simulate(w, p, size)
                curves[(size, kind, tf)] = eq
                trades[(size, kind, tf)] = sh
    bh = resample(h1, "1D").close.loc["2016-01-01":]

    ens = {}
    for size in SIZES:
        r = pd.concat([curves[(size, k, tf)].pct_change() for k in ("di", "adx") for tf in TFS], axis=1)
        ens[size] = (1 + r.fillna(0).mean(axis=1)).cumprod()

    pd.set_option("display.width", 250)
    rows = {}
    for size in SIZES:
        name = "cash (today)" if size == 0 else f"{size:.0%} short"
        for a, b, lab in PERIODS:
            t, c, d = stats(ens[size], a, b)
            rows.setdefault(name, {})[lab] = f"{c:.0f}%/yr, worst {d:.0f}%" if lab == "2016 -> now" else f"{t:+.0f}% / {d:.0f}%"
    for a, b, lab in PERIODS:
        t, c, d = stats(bh, a, b)
        rows.setdefault("BTC buy & hold", {})[lab] = f"{c:.0f}%/yr, worst {d:.0f}%" if lab == "2016 -> now" else f"{t:+.0f}% / {d:.0f}%"
    print("\n=== 8-sleeve strategy: cash vs a short when out of BTC.  Cells: return / worst drop ===")
    print(pd.DataFrame(rows).T.to_string())

    print("\n=== Year by year (return %) ===")
    yr = {("cash (today)" if s == 0 else f"{s:.0%} short"): ens[s].resample("YE").last().pct_change().mul(100)
          for s in SIZES}
    yr["BTC"] = bh.resample("YE").last().pct_change().mul(100)
    t = pd.DataFrame(yr)
    for k, s in zip(list(yr)[:-1], SIZES):     # first year from 1.0
        t.loc[t.index[0], k] = 100 * (ens[s].resample("YE").last().iloc[0] - 1)
    t.index = t.index.year
    print(t.T.round(0).to_string())

    print("\n=== 10% shorts since Oct 2025 (the current bear market), per sleeve ===")
    allsh = []
    for kind in ("di", "adx"):
        for tf in TFS:
            sh = trades[(0.10, kind, tf)]
            if len(sh):
                sh = sh[sh.opened >= "2025-10-01"].copy()
                sh.insert(0, "sleeve", f"{tf} {'DI' if kind == 'di' else 'ADX'}")
                allsh.append(sh)
    sh = pd.concat(allsh)
    sh["days"] = ((sh.closed.fillna(h1.index[-1]) - sh.opened).dt.total_seconds() / 86400).round(1)
    sh["closed"] = sh.closed.dt.strftime("%Y-%m-%d %H:%M").fillna("OPEN")
    sh["opened"] = sh.opened.dt.strftime("%Y-%m-%d %H:%M")
    print(sh.round(2).to_string(index=False))
    s = sh.groupby("sleeve").agg(shorts=("entry", "size"), wins=("btc move %", lambda x: int((x < 0).sum())),
                                 total_sleeve_pl=("sleeve P/L %", "sum"))
    print("\n", s.round(2).to_string())


if __name__ == "__main__":
    main()
