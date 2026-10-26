"""DI length sensitivity and walk-forward test for the daily DI cross.

    python bench/di_sweep.py
"""
import pandas as pd

import backtest
from data import fetch_coinbase
from exits import signals
from mtf import DESIGN_END

LENGTHS = [7, 10, 14, 18, 21, 25, 30, 34, 38, 42, 46, 50, 60, 75]


def metrics(d, e, x, since, until):
    w = d.loc[since:until]
    t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index])
    s = backtest.stats(t, eq, w)
    s["ret/drop"] = s["CAGR%"] / -s["max DD%"] if s["max DD%"] < 0 else 0
    return s


if __name__ == "__main__":
    d = fetch_coinbase(interval="1d")
    sig = {L: signals(d, "di", di_len=L) for L in LENGTHS}
    pd.set_option("display.width", 220)

    rows = {}
    for L in LENGTHS:
        full = metrics(d, *sig[L], "2017-01-01", DESIGN_END)
        a = metrics(d, *sig[L], "2017-01-01", "2021-06-30")
        b = metrics(d, *sig[L], "2021-07-01", DESIGN_END)
        rows[L] = {"trades": full["trades"], "win%": full["win%"], "CAGR%": full["CAGR%"],
                   "max DD%": full["max DD%"], "ret/drop": full["ret/drop"],
                   "CAGR% 17-21": a["CAGR%"], "CAGR% 21-25": b["CAGR%"]}
    print(f"Daily DI cross by DI length, 2017 -> {DESIGN_END}")
    print(pd.DataFrame(rows).T.round(1).to_string(), "\n")

    # Walk-forward: each year, pick the length with the best return/drop over
    # the previous 3 years, then trade it for the next 12 months.
    wf, years = [], pd.date_range("2019-10-27", "2025-10-27", freq="12MS")
    eq_wf = []
    for start in pd.date_range("2019-10-27", periods=7, freq=pd.DateOffset(years=1)):
        end = start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if start > d.index[-1]:
            break
        train = {L: metrics(d, *sig[L], start - pd.DateOffset(years=3), start - pd.Timedelta(days=1))["ret/drop"]
                 for L in LENGTHS}
        pick = max(train, key=train.get)
        test = {L: metrics(d, *sig[L], start, end)["CAGR%"] for L in (pick, 14, 38)}
        wf.append({"year from": f"{start:%Y-%m}", "picked length": pick,
                   "picked CAGR%": test[pick], "fixed 38 CAGR%": test[38], "fixed 14 CAGR%": test[14],
                   "B&H %": backtest.buy_hold(d.loc[start:end])["total%"]})
    print("Walk-forward (pick the best length on the prior 3 years, trade it the next year)")
    print(pd.DataFrame(wf).round(1).to_string(index=False))
