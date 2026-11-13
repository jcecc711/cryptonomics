"""Long and short on trend switches, mixing the 4h and 8h.

Each system holds 100% BTC when long and a 10% short when short (futures costs,
10%/yr holding cost), else cash. Signals use DI 38 / smoothing 9 on closed
candles; an 8h reading is used from the moment that 8h candle has closed.

  4h DI            long on a 4h DI+ cross up, short on a 4h DI- cross up; each
                   closes on the opposite 4h cross (the 4h DI sleeve, plus shorts)
  4h ADX           as 4h DI, but opens only when ADX rises through 14 with that
                   side's DI on top; closes on the opposite 4h DI cross
  8h DI            the 8h DI sleeve, plus shorts
  8h in, 4h out    opens on a fresh 8h DI cross, closes on the opposite 4h DI cross;
                   after a 4h exit it waits for the next 8h cross
  8h+4h agree      long while 8h and 4h both have DI+ on top, short while both have
                   DI- on top, else cash (re-enters when the 4h turns back)
  8h ADX in, 4h out   opens when the 8h ADX rises through 14 with that side on top,
                   closes on the opposite 4h DI cross

    python bench/mtf_switch.py
"""
import numpy as np
import pandas as pd

from data import resample
from indicators import adx
from short_hedge import hourly, simulate, stats

SHORT = 0.10


def readings(df):
    d = adx(df, 38, 9)
    return (d.di_plus > d.di_minus), d.adx


def on_4h(series, rule, idx4):
    """Higher-timeframe values on the 4h grid, from the 4h candle in which they become known."""
    s = series.copy()
    s.index = s.index + pd.Timedelta(rule) - pd.Timedelta("4h")
    return s.reindex(idx4, method="ffill")


def run_rule(name, up4, a4, up8, a8):
    n = len(up4)
    pos = np.zeros(n, int)
    cur = 0
    for i in range(1, n):
        x4u, x4d = up4[i] and not up4[i - 1], (not up4[i]) and up4[i - 1]
        x8u, x8d = up8[i] and not up8[i - 1], (not up8[i]) and up8[i - 1]
        t4u, t4d = a4[i] > 14 >= a4[i - 1] and up4[i], a4[i] > 14 >= a4[i - 1] and not up4[i]
        t8u, t8d = a8[i] > 14 >= a8[i - 1] and up8[i], a8[i] > 14 >= a8[i - 1] and not up8[i]
        # exits first
        if cur == 1 and not up4[i] and name != "8h DI":
            cur = 0
        if cur == -1 and up4[i] and name != "8h DI":
            cur = 0
        if name == "8h DI":
            if cur == 1 and not up8[i]:
                cur = 0
            if cur == -1 and up8[i]:
                cur = 0
        long_in = {"4h DI": x4u, "4h ADX": t4u, "8h DI": x8u, "8h in, 4h out": x8u and up4[i],
                   "8h+4h agree": up8[i] and up4[i], "8h ADX in, 4h out": t8u and up4[i]}[name]
        short_in = {"4h DI": x4d, "4h ADX": t4d, "8h DI": x8d, "8h in, 4h out": x8d and not up4[i],
                    "8h+4h agree": (not up8[i]) and not up4[i], "8h ADX in, 4h out": t8d and not up4[i]}[name]
        if long_in and cur != 1:
            cur = 1
        elif short_in and cur != -1:
            cur = -1
        pos[i] = cur
    return pos


def main():
    h1 = hourly()
    df4 = resample(h1, "4h").loc["2015-06-01":]
    df8 = resample(h1, "8h").loc["2015-06-01":]
    u4, a4 = readings(df4)
    u8, a8 = readings(df8)
    U8, A8 = on_4h(u8, "8h", df4.index).fillna(False).to_numpy(), on_4h(a8, "8h", df4.index).fillna(0).to_numpy()
    U4, A4 = u4.to_numpy(), a4.fillna(0).to_numpy()
    w = df4.loc["2016-01-01":]
    k = df4.index.get_indexer(w.index)
    rows = {}
    curves = {}
    for name in ["4h DI", "4h ADX", "8h DI", "8h in, 4h out", "8h+4h agree", "8h ADX in, 4h out"]:
        pos = run_rule(name, U4, A4, U8, A8)[k]
        for size, lab in ((0.0, "long / cash"), (SHORT, "long / 10% short")):
            eq, sh = simulate(w, pos, size)
            curves[(name, lab)] = eq
            t, c, d = stats(eq, "2016-01-01", None)
            r = {"%/yr 2016-now": c, "worst %": d}
            for a, b, l in (("2018-01-01", "2018-12-31", "2018"), ("2022-01-01", "2022-12-31", "2022"),
                            ("2021-01-01", "2021-12-31", "2021 bull"), ("2025-10-06", None, "Oct 25-now")):
                r[l] = stats(eq, a, b)[0]
            flips = np.sum(pos[1:] != pos[:-1])
            r["trades/yr"] = flips / ((w.index[-1] - w.index[0]).days / 365.25)
            if size:
                r["shorts won %"] = 100 * (sh["btc move %"] < 0).mean() if len(sh) else np.nan
                r["short P/L sum %"] = sh["sleeve P/L %"].sum() if len(sh) else 0
            rows[(name, lab)] = r
        cur = pos[-1]
        rows[(name, "long / 10% short")]["now"] = {1: "LONG", -1: "SHORT", 0: "cash"}[cur]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).T.round(1).to_string())


if __name__ == "__main__":
    main()
