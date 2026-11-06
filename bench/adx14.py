"""Does ADX reaching 14 mark the big moves? (Joe's observation.)

DI length 38, ADX smoothing 9, on 3h/4h/6h/8h/12h/1d, BTC 2016 -> Oct 2025
(Coinbase, cleaned), plus the 12 months after as a check.

1. How often ADX sits below 14 on each timeframe.
2. Event study: after ADX crosses 14 (up or down), how big is the next move,
   and does DI+ vs DI- call its direction?
3. The current DI-cross trades, split by ADX at entry.
4. ADX-14 as a trigger, alone and as a filter on the DI cross, at thresholds 10-18.

    python bench/adx14.py
"""
import numpy as np
import pandas as pd

import backtest
from data import fetch_coinbase, resample
from indicators import adx

TFS = {"3h": "3h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D"}
START, END, RECENT = "2016-01-01", "2025-10-26", "2025-10-27"
FEE, SLIP = 0.0002, 0.0001


def cross_up(a, level):
    return (a > level) & (a.shift() <= level)


def cross_dn(a, level):
    return (a < level) & (a.shift() >= level)


def frames():
    h1 = fetch_coinbase(interval="1h")
    out = {}
    for tf, rule in TFS.items():
        df = resample(h1, rule)
        d = adx(df, 38, 9)
        out[tf] = df.join(d)
    return out


def event_study(df, level=14, horizons=(6, 12, 24)):
    w = df.loc[START:END]
    up_ev, dn_ev = cross_up(w.adx, level), cross_dn(w.adx, level)
    bull = w.di_plus > w.di_minus
    rows = {}
    for h in horizons:
        fwd = w.close.shift(-h) / w.close - 1
        base = fwd.abs().median()
        for name, ev in (("ADX up through", up_ev), ("ADX down through", dn_ev)):
            f = fwd[ev].dropna()
            hit = (np.sign(f) == np.where(bull[ev].loc[f.index], 1, -1)).mean()
            rows[(name, h)] = {"events/yr": ev.sum() / 9.8, "median |move| %": 100 * f.abs().median(),
                               "vs any bar": f.abs().median() / base, "DI called direction %": 100 * hit}
    return rows


def di_cross_signals(df):
    up = df.di_plus > df.di_minus
    return up & ~up.shift(fill_value=False), ~up & up.shift(fill_value=False)


def trades_by_adx(df):
    w = df.loc[START:END]
    e, x = di_cross_signals(df)
    t, _ = backtest.run(w, e.loc[w.index], x.loc[w.index], fee=FEE, slippage=SLIP)
    if t.empty:
        return pd.DataFrame()
    # ADX on the signal candle, i.e. the candle before the fill
    sig_time = [w.index[w.index.get_loc(et) - 1] for et in t.entry_time]
    t["adx"] = w.adx.loc[sig_time].to_numpy()
    t["bucket"] = pd.cut(t.adx, [0, 12, 16, 20, 25, 100], labels=["<12", "12-16", "16-20", "20-25", "25+"])
    g = t.groupby("bucket", observed=False)
    return pd.DataFrame({"trades": g.size(), "win %": 100 * g.ret.apply(lambda r: (r > 0).mean()),
                         "avg trade %": 100 * g.ret.mean(), "share of profit %": 100 * g.ret.sum() / t.ret.sum()})


def strategies(df, level=14):
    """Long-only variants; each returns (entry, exit)."""
    up = df.di_plus > df.di_minus
    e0, x0 = di_cross_signals(df)
    a_up = cross_up(df.adx, level)
    a_dn = cross_dn(df.adx, level)
    recent_dip = (df.adx < level).rolling(6, min_periods=1).max().astype(bool)  # ADX was below in last 6 candles
    return {
        "DI cross (current)": (e0, x0),
        "ADX up thru L & DI+ > DI-, exit DI cross": (a_up & up, x0),
        "ADX up thru L & DI+ > DI-, exit ADX back below L or DI cross": (a_up & up, x0 | a_dn),
        "DI cross only if ADX dipped below L (last 6 candles)": (e0 & recent_dip, x0),
        "DI cross only if ADX above L": (e0 & (df.adx > level), x0),
        "DI cross OR (ADX up thru L & DI+ > DI-)": (e0 | (a_up & up), x0),
    }


def run(df, e, x, since, until=None):
    w = df.loc[since:until]
    t, eq = backtest.run(w, e.loc[w.index].fillna(False), x.loc[w.index].fillna(False), fee=FEE, slippage=SLIP)
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 1.5 else np.nan
    return {"CAGR %": 100 * cagr, "max DD %": 100 * dd, "ret/DD": cagr / -dd if dd < 0 else np.nan,
            "trades/yr": len(t) / yrs, "win %": 100 * (t.ret > 0).mean() if len(t) else np.nan,
            "total %": 100 * (eq.iloc[-1] - 1)}, eq


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    F = frames()

    print("=== 1. How often is ADX(38, smoothing 9) below 14? ===")
    print(pd.DataFrame({tf: {"median ADX": df.adx.loc[START:END].median(), "% of time below 14": 100 * (df.adx.loc[START:END] < 14).mean(),
                             "10th pct": df.adx.loc[START:END].quantile(.1), "90th pct": df.adx.loc[START:END].quantile(.9)}
                        for tf, df in F.items()}).T.round(1).to_string())

    print("\n=== 2. Event study: after ADX crosses 14 (move measured over the next N candles) ===")
    rows = {}
    for tf, df in F.items():
        for (name, h), v in event_study(df).items():
            rows[(tf, name, h)] = v
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\n=== 3. Current DI-cross trades by ADX at entry ===")
    for tf, df in F.items():
        print(f"-- {tf}")
        print(trades_by_adx(df).round(1).to_string())

    print("\n=== 4. As a trigger (threshold 14), 2016 -> Oct 2025 and the 12 months after ===")
    res, curves = {}, {}
    for tf, df in F.items():
        for name, (e, x) in strategies(df).items():
            full, eq = run(df, e, x, START, END)
            rec, _ = run(df, e, x, RECENT)
            res[(name, tf)] = {**{k: full[k] for k in ("CAGR %", "max DD %", "ret/DD", "trades/yr", "win %")},
                               "last 12m %": rec["total %"]}
            curves.setdefault(name, {})[tf] = eq.resample("1D").last()
    print(pd.DataFrame(res).T.round(2).to_string())

    print("\n=== Ensembles of 4h + 8h + 12h + 1d (equal split) ===")
    ens = {}
    for name, c in curves.items():
        e = pd.DataFrame({k: c[k] for k in ("4h", "8h", "12h", "1d")}).ffill()
        e = (1 + e.pct_change().fillna(0).mean(axis=1)).cumprod()
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        dd = (e / e.cummax() - 1).min()
        ens[name] = {"CAGR %": 100 * (e.iloc[-1] ** (1 / yrs) - 1), "max DD %": 100 * dd,
                     "ret/DD": (e.iloc[-1] ** (1 / yrs) - 1) / -dd}
    print(pd.DataFrame(ens).T.round(2).to_string())

    print("\n=== Threshold sensitivity: ret/DD by level, 2016 -> Oct 2025 ===")
    sens = {}
    for level in (10, 12, 14, 16, 18):
        for tf in ("4h", "8h", "12h", "1d"):
            for name, (e, x) in strategies(F[tf], level).items():
                if name == "DI cross (current)":
                    continue
                sens.setdefault(name.replace(" L", " L").strip(), {})[(tf, level)] = run(F[tf], e, x, START, END)[0]["ret/DD"]
    print(pd.DataFrame(sens).T.round(2).to_string())


if __name__ == "__main__":
    main()
