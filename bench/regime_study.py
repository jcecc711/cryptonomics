"""Bull / bear regime detection, and what to do in a bear market.

Regime detectors (daily, using only the previous day's close):
  halving     the calendar: bear from 540 to 910 days after each halving
              (past tops came ~525-550 days after, bottoms ~890-925 days after;
              fitted on the same two cycles it's tested on, so optimistic)
  ma200       close below the 200-day average
  wk50        close below the 50-week (350-day) average
  band        close below both the 20-week SMA and 21-week EMA ("bull market support band")
  halving+ma200   both the calendar and the 200-day average say bear

What each sleeve does (DI(38) cross on 4h / 8h / 12h / 1d, equal split):
  long only (current)   long while DI+ > DI-, else cash
  bear: cash            as current, but no longs during a bear regime (spot-only)
  bear: flip            bull: long while DI+ > DI-; bear: short while DI- > DI+ (needs futures)
  bear: add shorts      long while DI+ > DI- always; plus short while DI- > DI+ in a bear regime

Shorts pay 0.05% fee + 0.01% slippage per side and a holding cost (run at 10%/yr
and 0%); longs 0.02% + 0.01%.

    python bench/regime_study.py
"""
import numpy as np
import pandas as pd

from data import fetch_coinbase, resample
from indicators import adx, ema, sma
from short_study import _run_positions

TFS = {"4h": ("4h", 2190), "8h": ("8h", 1095), "12h": ("12h", 730), "1d": ("1D", 365)}
HALVINGS = [pd.Timestamp(d) for d in ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")]
PERIODS = [("2016-01-01", "2025-10-26", "2016-25 (yr %)"), ("2018-01-01", "2018-12-31", "2018 bear"),
           ("2020-01-01", "2021-12-31", "2020-21 bull"), ("2022-01-01", "2022-12-31", "2022 bear"),
           ("2023-01-01", "2025-10-26", "2023-25 bull"), ("2025-10-27", None, "last 12m")]


def regimes(daily):
    c = daily.close
    days_since = pd.Series([min([(t - h).days for h in HALVINGS if h <= t], default=10_000) for t in c.index], c.index)
    cal = (days_since >= 540) & (days_since <= 910)
    band = (c < sma(c, 140)) & (c < ema(c, 147))
    out = {"halving": cal, "ma200": c < sma(c, 200), "wk50": c < sma(c, 350), "band": band,
           "halving+ma200": cal & (c < sma(c, 200))}
    return {k: v.shift(1).fillna(False).astype(bool) for k, v in out.items()}   # known at the next day


def positions(up, bear, mode):
    if mode == "long only (current)":
        return np.where(up, 1, 0)
    if mode == "bear: cash":
        return np.where(up & ~bear, 1, 0)
    if mode == "bear: flip":
        return np.where(bear, np.where(~up, -1, 0), np.where(up, 1, 0))
    if mode == "bear: add shorts":
        return np.where(up, 1, np.where(bear, -1, 0))
    raise ValueError(mode)


def stats(eq, label):
    eq = eq / eq.iloc[0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = 100 * (eq / eq.cummax() - 1).min()
    if label.startswith("2016-25"):
        cagr = 100 * (eq.iloc[-1] ** (1 / yrs) - 1)
        return f"{cagr:.0f}%/yr  DD {dd:.0f}%"
    return f"{100 * (eq.iloc[-1] - 1):+.0f}%  DD {dd:.0f}%"


def main(carry=0.10):
    h1 = fetch_coinbase(interval="1h")
    daily = resample(h1, "1D")
    R = regimes(daily)
    frames = {}
    for tf, (rule, bpy) in TFS.items():
        df = resample(h1, rule)
        d = adx(df, 38, 9)
        frames[tf] = (df, (d.di_plus > d.di_minus).to_numpy(), bpy)

    pd.set_option("display.width", 260)
    print("=== How often each detector called a bear market (share of days) ===")
    yrs = {y: f"{y}" for y in range(2016, 2027)}
    print(pd.DataFrame({k: {y: f"{100 * v.loc[str(y)].mean():.0f}%" for y in yrs} for k, v in R.items()}).T.to_string())

    modes = ["long only (current)", "bear: cash", "bear: flip", "bear: add shorts"]
    rows = {}
    for since, until, label in PERIODS:
        for rname, bear_d in R.items():
            for mode in modes:
                if mode == "long only (current)" and rname != "halving":
                    continue   # the baseline doesn't depend on the detector
                curves = {}
                for tf, (df, up, bpy) in frames.items():
                    w = df.loc[since:until]
                    idx = df.index.get_indexer(w.index)
                    bear = bear_d.reindex(w.index.floor("D")).fillna(False).to_numpy()
                    curves[tf] = _run_positions(w, positions(up[idx], bear, mode), carry, bpy)
                e = pd.DataFrame(curves).ffill()
                e = (1 + e.pct_change().fillna(0).mean(axis=1)).cumprod()
                key = "long only (current)" if mode.startswith("long only") else f"{rname} | {mode}"
                rows.setdefault(key, {})[label] = stats(e, label)
        bh = daily.close.loc[since:until]
        rows.setdefault("buy & hold", {})[label] = stats(bh, label)
    print(f"\n=== 4h+8h+12h+1d DI ensemble by regime rule (short holding cost {carry:.0%}/yr) ===")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(0.10)
    main(0.0)
