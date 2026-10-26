"""Compare exits for the daily DI+ (length 38) crossover entry.

    python bench/exits.py
"""
import pandas as pd

import backtest
from data import fetch_coinbase
from indicators import adx, ema, sma

PERIODS = [("2019-07-01", "2025-10-26"), ("2019-07-01", "2022-06-30"), ("2022-07-01", "2025-10-26")]


def signals(d, mode, di_len=38, ma=50):
    """Stateful: buy on DI+ crossing above DI-, exit per `mode`.

    di          exit when DI+ crosses back below DI-
    ma          exit when the 21 EMA crosses below the `ma` SMA
    di_then_ma  DI exit until the 21 EMA gets above the SMA, then the 21/SMA cross
    """
    di = adx(d, di_len)
    p, m = di["di_plus"].to_numpy(), di["di_minus"].to_numpy()
    e, s = ema(d["close"], 21).to_numpy(), sma(d["close"], ma).to_numpy()
    n = len(d)
    entry, exit_ = [False] * n, [False] * n
    long_ = armed = False
    for i in range(1, n):
        if not long_:
            if p[i] > m[i] and p[i - 1] <= m[i - 1]:
                entry[i], long_, armed = True, True, e[i] > s[i]
            continue
        armed = armed or e[i] > s[i]
        di_dn = p[i] < m[i] and p[i - 1] >= m[i - 1]
        ma_dn = e[i] < s[i] and e[i - 1] >= s[i - 1]
        out = {"di": di_dn, "ma": ma_dn,
               "di_then_ma": (di_dn and not armed) or (armed and ma_dn)}[mode]
        if out:
            exit_[i], long_ = True, False
    return pd.Series(entry, d.index), pd.Series(exit_, d.index)


VARIANTS = {
    "Exit: DI cross": "di",
    "Exit: 21 EMA crosses below 50 MA": "ma",
    "Exit: DI cross until 21>50, then 21/50 cross": "di_then_ma",
}

if __name__ == "__main__":
    d = fetch_coinbase(interval="1d")
    pd.set_option("display.width", 220)
    for since, until in PERIODS:
        w = d.loc[since:until]
        rows = {}
        for name, mode in VARIANTS.items():
            e, x = signals(d, mode)
            t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index])
            rows[name] = backtest.stats(t, eq, w)
        rows["Buy & hold"] = backtest.buy_hold(w)
        print(f"{since} -> {until}")
        print(pd.DataFrame(rows).T.drop(columns=["trades/yr"]).round(1).to_string(), "\n")
