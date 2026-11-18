"""Which rules come closest to hand-picked swing entries and exits on the 4h chart?

The target is the hindsight ideal: long from each swing low to the next swing high
(a zigzag of 8%+ moves, like entries at the bottoms and exits at the tops). No rule
can know a bottom in real time; this measures how close each one gets.

For every rule (long-only, 4h, fills at the next open, 0.03% per side), over
2016 -> Oct 2025 and over the chart window (June -> Nov 2026):
  agree %        share of time the rule's position matches the ideal (in or out)
  capture %      the rule's total return as a share of the ideal's (log terms)
  entry / exit   median distance from the swing low / high the rule bought / sold
  trades/yr, %/yr, worst drop, BTC stack (BTC at the end vs holding)

    python bench/swing_match.py
"""
import numpy as np
import pandas as pd

from adx_plus import build, rsi
from data import resample
from indicators import adx, ema, sma
from short_hedge import hourly

COST = 0.0003


def zigzag(df, th=0.08):
    """Swing pivots: [(index position, 'L' or 'H', price)] for moves of at least th."""
    hi, lo = df.high.to_numpy(), df.low.to_numpy()
    piv, trend = [], 0
    hmax, hi_i, lmin, lo_i = hi[0], 0, lo[0], 0
    for i in range(1, len(df)):
        if trend == 0:                                  # until the first th move decides the direction
            if hi[i] > hmax:
                hmax, hi_i = hi[i], i
            if lo[i] < lmin:
                lmin, lo_i = lo[i], i
            if lo_i < hi_i and hmax >= lmin * (1 + th):
                piv.append((lo_i, "L", lmin))
                trend = 1
            elif hi_i < lo_i and lmin <= hmax * (1 - th):
                piv.append((hi_i, "H", hmax))
                trend = -1
        elif trend == 1:                                # up leg: follow the high until a th drop
            if hi[i] > hmax:
                hmax, hi_i = hi[i], i
            elif lo[i] <= hmax * (1 - th):
                piv.append((hi_i, "H", hmax))
                trend, lmin, lo_i = -1, lo[i], i
        else:                                           # down leg: follow the low until a th rise
            if lo[i] < lmin:
                lmin, lo_i = lo[i], i
            elif hi[i] >= lmin * (1 + th):
                piv.append((lo_i, "L", lmin))
                trend, hmax, hi_i = 1, hi[i], i
    return piv


def ideal_position(df, th=0.08):
    pos = np.zeros(len(df))
    piv = zigzag(df, th)
    for (i, k, _), (j, k2, _) in zip(piv, piv[1:]):
        if k == "L" and k2 == "H":
            pos[i + 1:j + 1] = 1          # bought at the low candle's close, sold at the high
    return pd.Series(pos, df.index), piv


def state(enter, leave):
    a, b = enter.to_numpy(), leave.to_numpy()
    s, on = np.zeros(len(a)), False
    for i in range(len(a)):
        on = (not b[i]) if on else bool(a[i])
        s[i] = on
    return pd.Series(s, enter.index)


def rules(f, f12, f1d):
    """Long-only position rules for the 4h, each a 0/1 Series decided on the candle close."""
    c = f.close
    out = {}
    up = f.up
    out["DI 38 cross (live 4h sleeve)"] = up.astype(float)
    for n in (60, 100):
        d = adx(f, n, 9)
        out[f"DI {n} cross"] = (d.di_plus > d.di_minus).astype(float)
    out["DI 38 + 12h DI agrees"] = (up & f12).astype(float)
    out["DI 38 + daily DI agrees"] = (up & f1d).astype(float)
    for a, b in ((13, 55), (21, 55), (8, 21)):
        out[f"EMA {a} / EMA {b} cross"] = (ema(c, a) > ema(c, b)).astype(float)
    out["EMA 21 / MA 100 cross"] = (ema(c, 21) > sma(c, 100)).astype(float)
    for n, m in ((20, 10), (55, 20)):
        enter = c > c.shift(1).rolling(n).max()
        leave = c < c.shift(1).rolling(m).min()
        out[f"breakout {n}-high / {m}-low"] = state(enter, leave)
    r = f.rsi
    out["RSI swing 30 -> 70"] = state((r > 30) & (r.shift() <= 30), (r < 70) & (r.shift() >= 70))
    out["RSI swing 35 -> 65"] = state((r > 35) & (r.shift() <= 35), (r < 65) & (r.shift() >= 65))
    out["RSI 30 in, DI exit"] = state((r > 30) & (r.shift() <= 30), ~up & up.shift(fill_value=False))
    lo, hi = f.low.rolling(14).min(), f.high.rolling(14).max()
    k = 100 * (c - lo) / (hi - lo)
    k = k.rolling(3).mean()
    dd = k.rolling(3).mean()
    out["Stochastic 20 -> 80"] = state((k > dd) & (k.shift() <= dd.shift()) & (k < 20),
                                       (k < dd) & (k.shift() >= dd.shift()) & (k > 80))
    return out


def score(df, pos, ideal, piv, a, b):
    w = slice(a, b)
    p, q, d = pos.loc[w], ideal.loc[w], df.loc[w]
    held = p.shift(1).fillna(0)                          # fill at the next open ~ next candle
    r = np.log(d.close / d.close.shift()).fillna(0)
    switches = held.diff().abs().fillna(0)
    strat = (held * r).sum() + np.log(1 - COST) * switches.sum()
    ideal = (q.shift(1).fillna(0) * r).sum()
    eq = np.exp((held * r + np.log(1 - COST) * switches).cumsum())
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    # distances: each rule entry vs the nearest swing low before it, each exit vs the swing high before it
    lows = [(df.index[i], v) for i, k, v in piv if k == "L"]
    highs = [(df.index[i], v) for i, k, v in piv if k == "H"]
    ent = p.index[(p.diff() > 0).to_numpy()]
    ext = p.index[(p.diff() < 0).to_numpy()]

    def dist(times, refs, sign):
        out = []
        for t in times:
            prev = [v for tt, v in refs if tt <= t]
            if prev:
                out.append(sign * (d.close.get(t, np.nan) / prev[-1] - 1))
        return 100 * np.nanmedian(out) if out else np.nan
    return {"agree %": 100 * (p == q).mean(), "capture %": 100 * strat / ideal if ideal else np.nan,
            "entry vs low %": dist(ent, lows, 1), "exit vs high %": dist(ext, highs, -1),
            "trades/yr": (p.diff() > 0).sum() / max(yrs, 0.4),
            "%/yr": 100 * (eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 1 else 100 * (eq.iloc[-1] - 1),
            "worst %": 100 * (eq / eq.cummax() - 1).min(),
            "BTC stack": eq.iloc[-1] / (d.close.iloc[-1] / d.close.iloc[0])}


def main():
    h1 = hourly()
    f = build(resample(h1, "4h").loc["2015-06-01":])
    up_on = lambda rule: (lambda d: (d.di_plus > d.di_minus))(adx(resample(h1, rule).loc["2015-06-01":], 38, 9))
    def known(s, rule):            # a higher-timeframe reading, from the 4h candle in which it is known
        s = s.copy()
        s.index = s.index + pd.Timedelta(rule) - pd.Timedelta("4h")
        return s.reindex(f.index, method="ffill").fillna(False).astype(bool)
    f12, f1d = known(up_on("12h"), "12h"), known(up_on("1D"), "1D")
    ideal, piv = ideal_position(f, 0.08)
    R = rules(f, f12, f1d)
    pd.set_option("display.width", 250)
    for a, b, lab in (("2016-01-01", "2025-10-26", "2016 -> Oct 2025"), ("2026-06-04", None, "chart window: Jun -> Nov 2026")):
        rows = {name: score(f, pos, ideal, piv, a, b) for name, pos in R.items()}
        print(f"\n=== {lab}: closeness to the ideal 8% swings, and results ===")
        print(pd.DataFrame(rows).T.sort_values("capture %", ascending=False).round(1).to_string())
    print("\nIdeal 8% swings in the chart window (swing lows L, highs H):")
    print("  ".join(f"{k}{f.index[i]:%m-%d} {v:,.0f}" for i, k, v in piv if f.index[i] >= pd.Timestamp("2026-06-04")))


if __name__ == "__main__":
    main()
