"""Candidate rule sets. Each returns (entry, exit) boolean Series."""
import pandas as pd

from indicators import adx, ema, sma


def cross_up(a, b):
    return (a > b) & (a.shift() <= b.shift())


def cross_down(a, b):
    return (a < b) & (a.shift() >= b.shift())


def ema_cross(df, fast=8, slow=21):
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    return cross_up(f, s), cross_down(f, s)


def ema_cross_adx(df, fast=8, slow=21, adx_min=20):
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    d = adx(df)
    trending_up = (d["adx"] > adx_min) & (d["di_plus"] > d["di_minus"])
    return (f > s) & cross_up(f, s).rolling(3).max().astype(bool) & trending_up, cross_down(f, s)


def ema_cross_adx_trend(df, fast=8, slow=21, adx_min=20, trend=200):
    entry, exit_ = ema_cross_adx(df, fast, slow, adx_min)
    above = df["close"] > sma(df["close"], trend)
    return entry & above, exit_


def ema_stack(df):
    """Long while 8 > 13 > 21 > 55 EMAs are stacked; exit when 8 drops below 21."""
    c = df["close"]
    e8, e13, e21, e55 = (ema(c, n) for n in (8, 13, 21, 55))
    stacked = (e8 > e13) & (e13 > e21) & (e21 > e55)
    return stacked & ~stacked.shift(fill_value=False), cross_down(e8, e21)


RULES = {
    "A: 8/21 EMA cross": ema_cross,
    "B: A + ADX>20 & DI+>DI-": ema_cross_adx,
    "C: B + above 200 MA": ema_cross_adx_trend,
    "D: 8>13>21>55 stack": ema_stack,
}


def di_signals(df, di_len=14, dashed=17, lower=14, tol=1.5, bounce_exit=True, lower_exit=False):
    """Joe's DI read.

    Buy when DI+ crosses above DI-. While long:
      - bounce exit: DI+ rises to within `tol` of the dashed line without
        breaking it, then turns down;
      - lower exit: after breaking the dashed line, DI+ falls below the lower line;
      - always exit if DI+ crosses back below DI-.
    """
    d = adx(df, di_len)
    p, m = d["di_plus"].to_numpy(), d["di_minus"].to_numpy()
    n = len(df)
    entry, exit_ = [False] * n, [False] * n
    long_, broke = False, False
    for i in range(1, n):
        if not long_:
            if p[i] > m[i] and p[i - 1] <= m[i - 1]:
                entry[i], long_, broke = True, True, p[i] > dashed
            continue
        broke = broke or p[i] > dashed
        out = p[i] < m[i]
        if bounce_exit and not broke and p[i - 1] >= dashed - tol and p[i] < p[i - 1]:
            out = True
        if lower_exit and broke and p[i] < lower:
            out = True
        if out:
            exit_[i], long_ = True, False
    return pd.Series(entry, df.index), pd.Series(exit_, df.index)


DI_RULES = {
    "E: DI+ crosses DI- (in & out)": lambda df, L: di_signals(df, L, bounce_exit=False),
    "F: E + bounce-off-dashed exit": lambda df, L: di_signals(df, L),
    "G: F + below-lower-line exit": lambda df, L: di_signals(df, L, lower_exit=True),
    "H: G + above 200 MA": None,
}
