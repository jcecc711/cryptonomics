"""Short-timeframe in-and-out study (3m / 6m / 15m / 1h), long-only spot.

Costs decide everything at this scale, so each rule runs under three models:
  taker   market-style orders both ways: 0.02% fee + 0.01% slippage per side
  maker   0%-fee limit orders at the signal candle's close. A buy only fills if
          price trades *below* it within `wait` candles (a sell: above), else
          the entry is skipped / the exit crosses the spread as a taker.
  free    every order fills at the next open with no cost (upper bound only)

    python bench/short_term.py [since] [until]
"""
import sys

import numpy as np
import pandas as pd

import backtest
from data import fetch, resample
from indicators import adx, ema
from mtf import align

HOLDOUT_START = "2026-07-27"   # last 3 months held back
TAKER_COST = 0.0003


def run_limit(df, entry, exit_, wait=2, taker_fee=0.0002, slip=0.0001):
    """Maker-entry backtest: limit orders at the signal close, see module doc."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    ent, ext = entry.to_numpy(), exit_.to_numpy()
    n = len(df)
    eq = np.ones(n)
    cash, units, in_pos, entry_px, entry_t = 1.0, 0.0, False, 0.0, None
    trades, i = [], 0
    while i < n - 1:
        if not in_pos and ent[i]:
            lim, cash0 = c[i], cash
            for j in range(i + 1, min(i + 1 + wait, n)):
                if l[j] < lim:
                    units, cash, in_pos = cash / lim, 0.0, True
                    entry_px, entry_t = lim, df.index[j]
                    break
            if in_pos:
                eq[i:j] = cash0  # still in cash until the fill
                i = j
                eq[i] = units * c[i]
                i += 1
                continue
        elif in_pos and ext[i]:
            lim, px, j_end = c[i], None, min(i + 1 + wait, n)
            for j in range(i + 1, j_end):
                if h[j] > lim:
                    px, fee = lim, 0.0
                    break
            else:
                j = min(j_end, n - 1)
                px, fee = o[j] * (1 - slip), taker_fee
            cash = units * px * (1 - fee)
            trades.append({"entry_time": entry_t, "exit_time": df.index[j], "entry": entry_px,
                           "exit": px, "ret": px * (1 - fee) / entry_px - 1})
            for k in range(i, j):
                eq[k] = units * c[k]
            units, in_pos = 0.0, False
            i = j
            eq[i] = cash
            i += 1
            continue
        eq[i] = cash + units * c[i]
        i += 1
    eq[-1] = cash + units * c[-1]
    return pd.DataFrame(trades), pd.Series(eq, index=df.index)


def cross_up(a, b):
    return (a > b) & (a.shift() <= b.shift())


def cross_dn(a, b):
    return (a < b) & (a.shift() >= b.shift())


def rules(df, daily_up):
    c = df["close"]
    out = {}
    for L in (14, 38):
        d = adx(df, L)
        up, dn = cross_up(d.di_plus, d.di_minus), cross_dn(d.di_plus, d.di_minus)
        out[f"DI{L} cross"] = (up, dn)
        out[f"DI{L} cross, daily up only"] = (up & daily_up, dn | ~daily_up)
    e8, e21 = ema(c, 8), ema(c, 21)
    out["EMA 8/21 cross, daily up only"] = (cross_up(e8, e21) & daily_up, cross_dn(e8, e21) | ~daily_up)
    return out


def daily_trend(index, bar_len):
    d = fetch("BTCUSDT", "1d", refresh=True)
    di = adx(d, 38, 9)
    up = (di.di_plus > di.di_minus).astype(float).to_frame("up")
    return align(up, pd.Timedelta(days=1), index, bar_len)["up"].fillna(0).astype(bool)


def row(t, eq, w):
    s = backtest.stats(t, eq, w)
    s["avg trade%"] = 100 * t["ret"].mean() if len(t) else 0
    return {k: s[k] for k in ("trades/yr", "win%", "avg trade%", "profit factor", "CAGR%", "max DD%", "time in mkt%")}


def main():
    since = sys.argv[1] if len(sys.argv) > 1 else "2024-10-15"
    until = sys.argv[2] if len(sys.argv) > 2 else HOLDOUT_START
    base = fetch("BTCUSDT", "3m")
    pd.set_option("display.width", 250)
    for tf, rule in (("3m", None), ("6m", "6min"), ("15m", "15min"), ("1h", "1h")):
        df = base if rule is None else resample(base, rule)
        bar = pd.Timedelta(rule or "3min")
        daily_up = daily_trend(df.index, bar)
        w = df.loc[since:until]
        rows = {}
        for name, (e, x) in rules(df, daily_up).items():
            e, x = e.loc[w.index].fillna(False), x.loc[w.index].fillna(False)
            for model in ("taker", "maker", "free"):
                if model == "maker":
                    t, eq = run_limit(w, e, x)
                else:
                    cost = TAKER_COST if model == "taker" else 0.0
                    t, eq = backtest.run(w, e, x, fee=cost, slippage=0.0)
                rows[(name, model)] = row(t, eq, w)
        rows[("Buy & hold", "")] = {"CAGR%": backtest.buy_hold(w)["CAGR%"], "max DD%": backtest.buy_hold(w)["max DD%"]}
        print(f"\n=== {tf} candles  {w.index[0]:%Y-%m-%d} -> {w.index[-1]:%Y-%m-%d} ===")
        print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
