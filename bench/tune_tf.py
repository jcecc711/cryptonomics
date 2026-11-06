"""Tailor DI length / ADX smoothing / ADX level per timeframe, tested walk-forward.

For each timeframe (4h, 8h, 12h, 1d) and strategy:
  DI cross     DI length in LENGTHS (ADX settings don't affect it)
  ADX trigger  DI length x ADX smoothing x ADX level
Each year from 2020 (and the last 12 months), settings are chosen on the
previous 4 years only, scored by return / max drawdown averaged with their
neighbouring settings (so a lone lucky spike can't win), then traded on the
next, unseen year. Compared with the fixed 38 / 9 / 14.

    python bench/tune_tf.py
"""
import itertools

import numpy as np
import pandas as pd

import backtest
from adx14 import FEE, SLIP
from data import fetch_coinbase, resample
from indicators import adx

TFS = {"4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
LENGTHS = [14, 21, 28, 34, 38, 42, 50, 60]
SMOOTHING = [9, 14]
LEVELS = [10, 12, 14, 16, 18]
FIXED = (38, 9, 14)
TEST_YEARS = [(f"{y}-01-01", f"{y}-12-31", str(y)) for y in range(2020, 2025)] + \
             [("2025-01-01", "2025-10-26", "2025"), ("2025-10-27", None, "last 12m")]


def curves_for(df):
    """Daily equity curves for every setting, each run once over the full history."""
    out = {"di": {}, "adx": {}}
    for L, sm in itertools.product(LENGTHS, SMOOTHING):
        d = adx(df, L, sm)
        up = d.di_plus > d.di_minus
        cross_dn = ~up & up.shift(fill_value=False)
        if sm == SMOOTHING[0]:   # the DI cross ignores ADX smoothing
            e = up & ~up.shift(fill_value=False)
            out["di"][(L,)] = backtest.run(df, e, cross_dn, fee=FEE, slippage=SLIP)[1].resample("1D").last().ffill()
        for lv in LEVELS:
            e = (d.adx > lv) & (d.adx.shift() <= lv) & up
            out["adx"][(L, sm, lv)] = backtest.run(df, e, cross_dn, fee=FEE, slippage=SLIP)[1].resample("1D").last().ffill()
    return out


def score(eq, a, b):
    w = eq.loc[a:b]
    if len(w) < 30:
        return np.nan
    w = w / w.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    dd = (w / w.cummax() - 1).min()
    cagr = w.iloc[-1] ** (1 / yrs) - 1
    return cagr / -dd if dd < 0 else np.nan


def neighbours(key, keys):
    """Settings one step away in any dimension (and the setting itself)."""
    grids = [LENGTHS, SMOOTHING, LEVELS][:len(key)]
    out = []
    for k in keys:
        if all(abs(grids[i].index(k[i]) - grids[i].index(key[i])) <= 1 for i in range(len(key))):
            out.append(k)
    return out


def pick(curves, a, b):
    raw = {k: score(eq, a, b) for k, eq in curves.items()}
    smooth = {k: np.nanmean([raw[n] for n in neighbours(k, raw)]) for k in raw}
    return max(smooth, key=lambda k: -np.inf if np.isnan(smooth[k]) else smooth[k])


def seg_returns(eq, a, b):
    w = eq.loc[a:b]
    return w.pct_change().fillna(0)


def main():
    h1 = fetch_coinbase(interval="1h")
    pd.set_option("display.width", 250)
    picks, oos = {}, {"tuned": {}, "fixed": {}}
    full = {}
    for tf, rule in TFS.items():
        df = resample(h1, rule)
        C = curves_for(df)
        full[tf] = C
        for strat in ("di", "adx"):
            fixed_key = FIXED[:1] if strat == "di" else FIXED
            r_t, r_f = [], []
            for a, b, label in TEST_YEARS:
                train_end = pd.Timestamp(a) - pd.Timedelta(days=1)
                k = pick(C[strat], str((train_end - pd.DateOffset(years=4)).date()), str(train_end.date()))
                picks[(tf, strat, label)] = k
                r_t.append(seg_returns(C[strat][k], a, b))
                r_f.append(seg_returns(C[strat][fixed_key], a, b))
            oos["tuned"][(tf, strat)] = pd.concat(r_t)
            oos["fixed"][(tf, strat)] = pd.concat(r_f)

    print("=== Settings picked each year (DI length / ADX smoothing / ADX level), from the previous 4 years ===")
    tab = {}
    for (tf, strat, label), k in picks.items():
        tab.setdefault(f"{tf} {strat}", {})[label] = "/".join(map(str, k))
    print(pd.DataFrame(tab).T.to_string())

    def summarize(r):
        eq = (1 + r).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        dd = (eq / eq.cummax() - 1).min()
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        return {"CAGR %": 100 * cagr, "max DD %": 100 * dd, "ret/DD": cagr / -dd}

    print("\n=== Out-of-sample 2020 -> now: tuned each year vs fixed 38 / 9 / 14 ===")
    rows = {}
    for (tf, strat) in oos["tuned"]:
        t, f = summarize(oos["tuned"][(tf, strat)]), summarize(oos["fixed"][(tf, strat)])
        rows[f"{tf} {strat}"] = {"tuned CAGR %": t["CAGR %"], "tuned DD %": t["max DD %"], "tuned ret/DD": t["ret/DD"],
                                 "fixed CAGR %": f["CAGR %"], "fixed DD %": f["max DD %"], "fixed ret/DD": f["ret/DD"]}
    for which in ("tuned", "fixed"):
        allr = pd.concat(oos[which], axis=1).fillna(0).mean(axis=1)
        s = summarize(allr)
        rows.setdefault("ENSEMBLE (8 sleeves)", {}).update({f"{which} CAGR %": s["CAGR %"], f"{which} DD %": s["max DD %"],
                                                            f"{which} ret/DD": s["ret/DD"]})
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\n=== Year by year, ensemble of all 8 sleeves (return %) ===")
    yr = {}
    for which in ("tuned", "fixed"):
        allr = pd.concat(oos[which], axis=1).fillna(0).mean(axis=1)
        for a, b, label in TEST_YEARS:
            yr.setdefault(which, {})[label] = 100 * ((1 + allr.loc[a:b]).prod() - 1)
    print(pd.DataFrame(yr).T.round(1).to_string())

    print("\n=== Full 2016 -> Oct 2025 return/drawdown by DI length (DI cross) ===")
    print(pd.DataFrame({tf: {L: score(full[tf]["di"][(L,)], "2016-01-01", "2025-10-26") for L in LENGTHS} for tf in TFS}).T.round(2).to_string())
    print("\n=== Full-period return/drawdown, ADX trigger, smoothing 9: rows DI length, columns ADX level ===")
    for tf in TFS:
        print(f"-- {tf}")
        print(pd.DataFrame({lv: {L: score(full[tf]["adx"][(L, 9, lv)], "2016-01-01", "2025-10-26") for L in LENGTHS}
                            for lv in LEVELS}).round(2).to_string())


if __name__ == "__main__":
    main()
