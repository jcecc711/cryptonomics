"""Compare Joe's DI rules across timeframes and DI lengths."""
import sys

import pandas as pd

import backtest
from data import fetch
from indicators import sma
from rules import DI_RULES, di_signals

since = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
until = sys.argv[2] if len(sys.argv) > 2 else None
pd.set_option("display.width", 200)
for interval in ("6h", "1d"):
    df = fetch("BTCUSDT", interval)
    w = df.loc[since:until]
    for L in (14, 38):
        rows = {}
        for name, fn in DI_RULES.items():
            if fn is None:
                e, x = di_signals(df, L, lower_exit=True)
                e = e & (df["close"] > sma(df["close"], 200))
            else:
                e, x = fn(df, L)
            t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index])
            rows[name] = backtest.stats(t, eq, w)
        rows["Buy & hold"] = backtest.buy_hold(w)
        print(f"\n{interval}  DI length {L}  {w.index[0]:%Y-%m-%d} -> {w.index[-1]:%Y-%m-%d}")
        print(pd.DataFrame(rows).T.drop(columns=["avg win%", "avg loss%"]).round(2).to_string())
