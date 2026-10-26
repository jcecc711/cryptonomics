"""Compare exits for the daily DI+ (length 38) crossover entry.

    python bench/exits.py
"""
import pandas as pd

import backtest
from data import fetch_coinbase, to_weekly
from indicators import adx, ema, sma
from mtf import align, trend_score

PERIODS = [("2019-07-01", "2025-10-26"), ("2019-07-01", "2022-06-30"), ("2022-07-01", "2025-10-26")]


def weekly_score(d):
    """Weekly EMA trend score (-3..+3) mapped onto daily bars, closed weeks only."""
    ws = trend_score(to_weekly(d))[["score"]]
    return align(ws, pd.Timedelta(days=7), d.index, pd.Timedelta(days=1))["score"]


def signals(d, mode, di_len=38, ma=50, trail=None, strong=None):
    """Stateful: buy on DI+ crossing above DI-, exit per `mode`.

    di          exit when DI+ crosses back below DI-
    ma          exit when the 21 EMA crosses below the `ma` SMA
    di_then_ma  DI exit until the 21 EMA gets above the SMA, then the 21/SMA cross
    switch      DI exit by default; while `strong` (weekly trend firmly up) and
                the 21 EMA is above the SMA, use the 21/SMA cross instead, with
                `trail` as its safety exit
    trail       also exit when the close falls this fraction below the highest
                close since entry (e.g. 0.15 = 15%)
    """
    di = adx(d, di_len)
    p, m = di["di_plus"].to_numpy(), di["di_minus"].to_numpy()
    c = d["close"].to_numpy()
    e, s = ema(d["close"], 21).to_numpy(), sma(d["close"], ma).to_numpy()
    n = len(d)
    st = strong.to_numpy() if strong is not None else [False] * n
    entry, exit_ = [False] * n, [False] * n
    long_ = armed = False
    peak = 0.0
    for i in range(1, n):
        if not long_:
            if p[i] > m[i] and p[i - 1] <= m[i - 1]:
                entry[i], long_, armed = True, True, e[i] > s[i]
                peak = c[i]
            continue
        peak = max(peak, c[i])
        armed = armed or e[i] > s[i]
        di_dn = p[i] < m[i] and p[i - 1] >= m[i - 1]
        ma_dn = e[i] < s[i] and e[i - 1] >= s[i - 1]
        stopped = bool(trail) and c[i] < peak * (1 - trail)
        if mode == "switch":
            out = (ma_dn or stopped) if (st[i] and e[i] > s[i]) else di_dn
        else:
            out = {"di": di_dn, "ma": ma_dn,
                   "di_then_ma": (di_dn and not armed) or (armed and ma_dn)}[mode] or stopped
        if out:
            exit_[i], long_ = True, False
    return pd.Series(entry, d.index), pd.Series(exit_, d.index)


VARIANTS = {
    "DI cross": ("di", None),
    "Switch: weekly>=2 -> 21/50 + 10% stop": ("switch2", 0.10),
    "Switch: weekly=3 -> 21/50 + 10% stop": ("switch3", 0.10),
    "21/50 cross": ("ma", None),
    "21/50 cross + 10% trailing stop": ("ma", 0.10),
    "21/50 cross + 15% trailing stop": ("ma", 0.15),
    "21/50 cross + 20% trailing stop": ("ma", 0.20),
    "21/50 cross + 25% trailing stop": ("ma", 0.25),
    "DI cross + 15% trailing stop": ("di", 0.15),
}

if __name__ == "__main__":
    d = fetch_coinbase(interval="1d")
    pd.set_option("display.width", 220)
    for since, until in PERIODS:
        w = d.loc[since:until]
        rows = {}
        ws = weekly_score(d)
        for name, (mode, trail) in VARIANTS.items():
            if mode.startswith("switch"):
                e, x = signals(d, "switch", trail=trail, strong=ws >= int(mode[-1]))
            else:
                e, x = signals(d, mode, trail=trail)
            t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index])
            rows[name] = backtest.stats(t, eq, w)
        rows["Buy & hold"] = backtest.buy_hold(w)
        print(f"{since} -> {until}")
        print(pd.DataFrame(rows).T.drop(columns=["trades/yr"]).round(1).to_string(), "\n")
