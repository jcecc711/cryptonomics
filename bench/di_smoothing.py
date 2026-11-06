"""Can extra smoothing or filtering make DI+ / DI- crossings more reliable?

Variants (DI length 38, ADX smoothing 9 throughout):
  base          the bot today: DI+ vs DI- as calculated
  ema3/5/8      DI+ and DI- each smoothed again with an N-candle EMA before comparing
  confirm2/3    a cross counts only after it has held N candles in a row
  gap1/2/3      buffer zone: long when DI+ leads by more than G points, out when DI- leads by more than G
  heikin-ashi   DI calculated from Heikin Ashi candles

Applied to both strategies (DI cross; ADX-14 trigger with a DI exit), on
4h / 8h / 12h / 1d, BTC 2016 -> Oct 2025 split into two halves, plus the 12
months after. A variant only matters if it helps on most timeframes in both halves.

    python bench/di_smoothing.py
"""
import numpy as np
import pandas as pd

import backtest
from adx14 import FEE, SLIP
from data import fetch_coinbase, resample
from indicators import adx, ema

TFS = {"4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
HALVES = [("2016-01-01", "2020-12-31", "2016-20"), ("2021-01-01", "2025-10-26", "2021-25")]
RECENT = "2025-10-27"


def heikin_ashi(df):
    c = (df.open + df.high + df.low + df.close) / 4
    o = np.empty(len(df))
    o[0] = (df.open.iloc[0] + df.close.iloc[0]) / 2
    cv = c.to_numpy()
    for i in range(1, len(df)):
        o[i] = (o[i - 1] + cv[i - 1]) / 2
    o = pd.Series(o, df.index)
    return pd.DataFrame({"open": o, "high": pd.concat([df.high, o, c], axis=1).max(axis=1),
                         "low": pd.concat([df.low, o, c], axis=1).min(axis=1), "close": c})


def state_from(enter_ok, exit_now):
    """Long from the first candle enter_ok is true until exit_now is true."""
    a, b = enter_ok.to_numpy(), exit_now.to_numpy()
    s = np.zeros(len(a), dtype=bool)
    on = False
    for i in range(len(a)):
        on = (not b[i]) if on else bool(a[i])
        s[i] = on
    return pd.Series(s, enter_ok.index)


def variants(df):
    base = adx(df, 38, 9)
    p, m, a = base.di_plus, base.di_minus, base.adx
    out = {"base": (p > m, p < m, a)}
    for n in (3, 5, 8):
        ps, ms = ema(p, n), ema(m, n)
        out[f"ema{n}"] = (ps > ms, ps < ms, a)
    for k in (2, 3):
        up, dn = p > m, p < m
        out[f"confirm{k}"] = (up.rolling(k).sum() == k, dn.rolling(k).sum() == k, a)
    for g in (1, 2, 3):
        out[f"gap{g}"] = (p - m > g, m - p > g, a)
    h = adx(heikin_ashi(df), 38, 9)
    out["heikin-ashi"] = (h.di_plus > h.di_minus, h.di_plus < h.di_minus, h.adx)
    return out


def signals(bull, bear, a, strat, level=14):
    if strat == "di":
        s = state_from(bull, bear)
    else:   # ADX trigger: enter when ADX rises through the level while bullish; exit when bearish
        trig = (a > level) & (a.shift() <= level) & bull
        s = state_from(trig, bear)
    return s & ~s.shift(fill_value=False), ~s & s.shift(fill_value=False)


def stats(eq, a=None, b=None):
    w = eq.loc[a:b] if a else eq
    w = w / w.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    dd = (w / w.cummax() - 1).min()
    tot = w.iloc[-1] - 1
    cagr = w.iloc[-1] ** (1 / yrs) - 1 if yrs > 1 else tot
    return cagr, dd, (cagr / -dd if dd < 0 else np.nan), tot


def main():
    h1 = fetch_coinbase(interval="1h")
    pd.set_option("display.width", 250)
    curves = {}   # (variant, strat, tf) -> daily equity over 2016 -> now
    trades = {}
    for tf, rule in TFS.items():
        df = resample(h1, rule).loc["2015-09-01":]
        for name, (bull, bear, a) in variants(df).items():
            for strat in ("di", "adx"):
                e, x = signals(bull, bear, a, strat)
                w = df.loc["2016-01-01":]
                t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index], fee=FEE, slippage=SLIP)
                curves[(name, strat, tf)] = eq.resample("1D").last().ffill()
                trades[(name, strat, tf)] = t

    names = list(variants(resample(h1, "1D").iloc[:400]).keys())
    rows, wins = {}, {}
    for name in names:
        for strat in ("di", "adx"):
            r = {}
            better = 0
            for a, b, lab in HALVES:
                cal = [stats(curves[(name, strat, tf)], a, b)[2] for tf in TFS]
                base = [stats(curves[("base", strat, tf)], a, b)[2] for tf in TFS]
                r[f"ret/DD {lab} (avg of 4 tf)"] = np.nanmean(cal)
                better += sum(c > bb + 1e-9 for c, bb in zip(cal, base))
            r["tf-halves beating base (of 8)"] = better if name != "base" else "-"
            tr = [trades[(name, strat, tf)] for tf in TFS]
            r["trades/yr (4 tf)"] = sum(len(t) for t in tr) / 9.8
            r["win %"] = 100 * np.mean([x for t in tr for x in (t.ret > 0)]) if sum(len(t) for t in tr) else np.nan
            rows[(strat, name)] = r

    # 8-sleeve 50/50 ensemble per variant
    for name in names:
        rets = pd.concat([curves[(name, s, tf)].pct_change() for s in ("di", "adx") for tf in TFS], axis=1).fillna(0).mean(axis=1)
        eq = (1 + rets).cumprod()
        c, d, cal, _ = stats(eq, "2016-01-01", "2025-10-26")
        rec = stats(eq, RECENT)[3]
        rows[("ENSEMBLE", name)] = {"CAGR % 2016-25": 100 * c, "max DD %": 100 * d, "ret/DD": cal, "last 12m %": 100 * rec}
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
