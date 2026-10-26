"""Top-down (weekly -> daily -> 6H) trend scoring, per Joe's process.

Each timeframe gets a trend score from -3 to +3 built from its EMAs:
  angle   +1 / -1  21 EMA sloping up / down over the last `k` candles
  stack   +1 / -1  8 > 21 > 55 (or 8 < 21 < 55); 0 when mixed
  spread  +1 / -1  gap between 8 and 55 EMA widening in the trend's direction

Higher-timeframe scores are only used once their candle has closed, so the
6H bar never sees a weekly or daily value from the future.

    python bench/mtf.py [since] [until]
"""
import sys

import numpy as np
import pandas as pd

import backtest
from data import fetch_coinbase, to_weekly
from indicators import adx, ema, sma

DESIGN_END = "2025-10-26"  # last 12 months are held back for the final check


def trend_score(df, k=3):
    c = df["close"]
    e8, e21, e55 = ema(c, 8), ema(c, 21), ema(c, 55)
    angle = np.sign(e21 - e21.shift(k))
    stack = pd.Series(np.select([(e8 > e21) & (e21 > e55), (e8 < e21) & (e21 < e55)], [1, -1], 0), df.index)
    gap = (e8 - e55) / c
    spread = np.sign(gap) * (gap.abs() > gap.abs().shift(k))
    return pd.DataFrame({"angle": angle, "stack": stack, "spread": spread,
                         "score": angle + stack + spread})


def align(higher, bar_len, target_index, target_len):
    """Map higher-timeframe values onto target bars using only closed candles."""
    h = higher.copy()
    h.index = h.index + bar_len                     # value known at candle close
    t_close = target_index + target_len
    out = h.reindex(h.index.union(t_close)).ffill().reindex(t_close)
    out.index = target_index
    return out


def build():
    d = fetch_coinbase(interval="1d")
    h6 = fetch_coinbase(interval="6h")
    w = to_weekly(d)
    W, D, H = pd.Timedelta(days=7), pd.Timedelta(days=1), pd.Timedelta(hours=6)

    ws = trend_score(w)
    ws["above_200w"] = (w["close"] > sma(w["close"], 200)).astype(float)
    ws.loc[sma(w["close"], 200).isna(), "above_200w"] = np.nan
    # A "bounce": a weekly low within 10% of the 200W SMA that closes above it.
    m200 = sma(w["close"], 200)
    bounce = (w["low"] <= m200 * 1.10) & (w["close"] > m200)
    ws["bounce_2y"] = bounce.astype(float).rolling(104, min_periods=1).max()

    ds = trend_score(d)
    hs = trend_score(h6)
    di = adx(h6)

    f = pd.DataFrame(index=h6.index)
    f[["w_score", "above_200w", "bounce_2y"]] = align(ws[["score", "above_200w", "bounce_2y"]], W, h6.index, H)
    f["d_score"] = align(ds[["score"]], D, h6.index, H)["score"]
    f["h_score"] = hs["score"]
    c = h6["close"]
    f["h_cross_up"] = (ema(c, 8) > ema(c, 21)) & (ema(c, 8).shift() <= ema(c, 21).shift())
    f["h_cross_dn"] = (ema(c, 8) < ema(c, 21)) & (ema(c, 8).shift() >= ema(c, 21).shift())
    f["h_di_up"] = (di["di_plus"] > di["di_minus"]) & (di["di_plus"].shift() <= di["di_minus"].shift())
    return h6, f


def strategy(f, w_min=1, d_min=1, macro=None, pullback_bars=20, trigger="ema", exit_on="h_cross"):
    """Buy the 6H turn back up after a sell-off, when weekly and daily agree."""
    bias = (f["w_score"] >= w_min) & (f["d_score"] >= d_min)
    if macro:
        bias &= f[macro] == 1
    sold_off = (f["h_score"] <= -2).rolling(pullback_bars, min_periods=1).max().astype(bool)
    turn = f["h_cross_up"] if trigger == "ema" else f["h_di_up"]
    entry = bias & sold_off & turn
    if exit_on == "h_cross":
        exit_ = f["h_cross_dn"]
    elif exit_on == "d_score":
        exit_ = f["d_score"] <= 0
    else:  # whichever comes first: daily bias lost, or 6H trend fully down
        exit_ = (f["d_score"] <= 0) | (f["h_score"] <= -3)
    return entry, exit_


VARIANTS = {
    "1: W>=1 D>=1, 6H EMA turn, exit 6H cross":   dict(),
    "2: same, exit when daily bias lost":         dict(exit_on="d_score"),
    "3: same, exit daily lost or 6H fully down":  dict(exit_on="either"),
    "4: 3 + above 200W":                          dict(exit_on="either", macro="above_200w"),
    "5: 3 + within 2y of 200W bounce":            dict(exit_on="either", macro="bounce_2y"),
    "6: 3 with strong bias W>=2 D>=2":            dict(exit_on="either", w_min=2, d_min=2),
    "7: 3 with DI+ cross as the 6H trigger":      dict(exit_on="either", trigger="di"),
    "8: no top-down filter (W,D ignored)":        dict(exit_on="either", w_min=-9, d_min=-9),
}


def main():
    since = sys.argv[1] if len(sys.argv) > 1 else "2019-07-01"
    until = sys.argv[2] if len(sys.argv) > 2 else DESIGN_END
    h6, f = build()
    win = h6.loc[since:until]
    rows = {}
    for name, kw in VARIANTS.items():
        e, x = strategy(f, **kw)
        t, eq = backtest.run(win, e.loc[win.index].fillna(False), x.loc[win.index].fillna(False))
        rows[name] = backtest.stats(t, eq, win)
    rows["Buy & hold"] = backtest.buy_hold(win)
    pd.set_option("display.width", 220)
    print(f"6H execution, Coinbase BTC-USD  {win.index[0]:%Y-%m-%d} -> {win.index[-1]:%Y-%m-%d}")
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
