"""Is the DI cross better on a lower timeframe than daily? Robustness checks.

1. Plateau: DI length 26-56 on 4h / 6h / 8h / 12h / 1d (is the edge smooth?)
2. Rolling 2-year windows 2016 -> Oct 2025: how often each beats the daily core
3. Recent 12 months (Oct 2025 -> now), not used by the scan
4. Ensemble: the same rule on 4h + 8h + 12h + 1d, money split equally

    python bench/di_timeframes.py
"""
import numpy as np
import pandas as pd

import backtest
from data import fetch_coinbase, resample
from indicators import adx
from strategy_scan import FEE, SLIP, dn_cross, up_cross

TFS = {"4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D"}


def equity(df, L, since, until=None):
    d = adx(df, L)
    e, x = up_cross(d.di_plus, d.di_minus), dn_cross(d.di_plus, d.di_minus)
    w = df.loc[since:until]
    t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index], fee=FEE, slippage=SLIP)
    return eq.resample("1D").last(), t


def m(eq):
    eq = eq / eq.iloc[0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return 100 * cagr, 100 * dd, cagr / -dd if dd < 0 else np.nan


def main():
    h1 = fetch_coinbase(interval="1h", refresh=True)
    frames = {tf: (h1 if tf == "1h" else resample(h1, r)) for tf, r in TFS.items()}
    pd.set_option("display.width", 250)

    print("1) Return / drawdown (calmar) by DI length, 2016 -> 2025-10-26")
    lengths = [26, 30, 34, 38, 42, 46, 50, 56]
    tab = {tf: {L: m(equity(df, L, "2016-01-01", "2025-10-26")[0])[2] for L in lengths} for tf, df in frames.items()}
    print(pd.DataFrame(tab).T.round(2).to_string())

    cands = {"1d DI38 (current core)": ("1d", 38), "12h DI38": ("12h", 38), "8h DI38": ("8h", 38),
             "6h DI38": ("6h", 38), "4h DI38": ("4h", 38)}
    eqs = {k: equity(frames[tf], L, "2016-01-01", "2025-10-26")[0] for k, (tf, L) in cands.items()}
    eqs = pd.DataFrame(eqs).ffill()
    rets = eqs.pct_change().fillna(0)
    eqs["Ensemble 4h+8h+12h+1d"] = (1 + rets[["4h DI38", "8h DI38", "12h DI38", "1d DI38 (current core)"]].mean(axis=1)).cumprod()

    print("\n2) Full period and rolling 2-year windows (every 2 weeks), 2016 -> 2025-10-26")
    n, rows = 730, {}
    starts = range(0, len(eqs) - n, 14)
    for k in eqs:
        full = m(eqs[k])
        win = [m(eqs[k].iloc[s:s + n]) for s in starts]
        core = [m(eqs["1d DI38 (current core)"].iloc[s:s + n]) for s in starts]
        rows[k] = {"CAGR%": full[0], "max DD%": full[1], "calmar": full[2],
                   "P(beats core return)": 100 * np.mean([a[0] > b[0] for a, b in zip(win, core)]),
                   "P(beats core ret/DD)": 100 * np.mean([a[2] > b[2] for a, b in zip(win, core)]),
                   "P(DD worse -35%)": 100 * np.mean([a[1] < -35 for a in win]),
                   "median 2y CAGR%": np.median([a[0] for a in win]),
                   "bad-case 2y CAGR%": np.percentile([a[0] for a in win], 10)}
    print(pd.DataFrame(rows).T.round(1).to_string())

    print("\n3) Recent 12 months, 2025-10-27 -> now (not used above)")
    rec = {}
    for k, (tf, L) in cands.items():
        eq, t = equity(frames[tf], L, "2025-10-27")
        c, dd, _ = m(eq)
        rec[k] = {"return%": 100 * (eq.iloc[-1] / eq.iloc[0] - 1), "max DD%": dd, "trades": len(t),
                  "win%": 100 * (t.ret > 0).mean() if len(t) else np.nan}
    bh = frames["1d"].close.loc["2025-10-27":]
    rec["Buy & hold"] = {"return%": 100 * (bh.iloc[-1] / bh.iloc[0] - 1), "max DD%": m(bh)[1]}
    print(pd.DataFrame(rec).T.round(1).to_string())


if __name__ == "__main__":
    main()
