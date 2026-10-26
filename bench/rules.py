"""Candidate rule sets. Each returns (entry, exit) boolean Series."""
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
