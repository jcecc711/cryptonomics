"""Would shorting downtrends help? The DI(38) cross, long-only vs short-only vs
long+short, on 4h / 8h / 12h / 1d and the four-timeframe ensemble.

Signals act on each closed candle and fill at the next open. A short opens
when DI- crosses above DI+ and closes when DI+ crosses back above.
Costs: long side as everywhere else (0.02% fee + 0.01% slippage per side).
Shorts would need US futures, so they pay 0.05% fee + 0.01% slippage per side
plus a holding cost (futures basis / funding) charged per day held, run at
10%/yr and at 0 for comparison. A short is 1x, unleveraged: its value is
entry_equity * (2 - price / entry_price).

    python bench/short_study.py
"""
import numpy as np
import pandas as pd

from data import fetch_coinbase, resample
from indicators import adx

TFS = {"4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
LONG_COST = 0.0003
SHORT_COST = 0.0006
PERIODS = [("2016-01-01", "2025-10-26", "2016-25 (all)"), ("2018-01-01", "2018-12-31", "2018 bear"),
           ("2020-01-01", "2021-12-31", "2020-21 bull"), ("2022-01-01", "2022-12-31", "2022 bear"),
           ("2023-01-01", "2025-10-26", "2023-25"), ("2025-10-27", None, "last 12 months")]


def simulate(df, up, mode, carry=0.10, bars_per_year=365):
    """mode: 'long', 'short' or 'both'. Returns a daily equity curve."""
    o, c = df.open.to_numpy(), df.close.to_numpy()
    want = np.where(up, 1, -1)
    if mode == "long":
        want = np.where(up, 1, 0)
    elif mode == "short":
        want = np.where(up, 0, -1)
    n = len(df)
    eq = np.ones(n)
    cash, pos, entry_px, entry_eq = 1.0, 0, 0.0, 1.0
    carry_bar = carry / bars_per_year
    for i in range(n - 1):
        target = want[i]
        if target != pos:
            px = o[i + 1]
            if pos == 1:
                cash = entry_eq * px / entry_px * (1 - LONG_COST)
            elif pos == -1:
                cash = entry_eq * (2 - px / entry_px) * (1 - SHORT_COST)
            pos = target
            if pos:
                entry_px = px
                entry_eq = cash * (1 - (LONG_COST if pos == 1 else SHORT_COST))
        if pos == -1:
            entry_eq *= 1 - carry_bar
        if pos == 1:
            eq[i] = entry_eq * c[i] / entry_px
        elif pos == -1:
            eq[i] = max(entry_eq * (2 - c[i] / entry_px), 0)
        else:
            eq[i] = cash
    eq[-1] = eq[-2]
    return pd.Series(eq, df.index).resample("1D").last().ffill()


def stats(eq):
    eq = eq / eq.iloc[0]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs >= 1 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return 100 * total, 100 * cagr, 100 * dd


def main():
    h1 = fetch_coinbase(interval="1h", refresh=True)
    frames = {tf: resample(h1, r) for tf, r in TFS.items()}
    ups = {}
    for tf, df in frames.items():
        d = adx(df, 38, 9)
        ups[tf] = (d.di_plus > d.di_minus).to_numpy()
    bpy = {"4h": 365 * 6, "8h": 365 * 3, "12h": 365 * 2, "1d": 365}
    daily = resample(h1, "1D").close
    pd.set_option("display.width", 250)

    for carry in (0.10, 0.0):
        rows = {}
        for since, until, label in PERIODS:
            curves = {}
            for mode in ("long", "short", "both"):
                per_tf = {}
                for tf, df in frames.items():
                    w = df.loc[since:until]
                    u = ups[tf][df.index.get_indexer(w.index)]
                    per_tf[tf] = simulate(w, u, mode, carry, bpy[tf])
                eqs = pd.DataFrame(per_tf).ffill()
                curves[mode] = (1 + eqs.pct_change().fillna(0).mean(axis=1)).cumprod()
                for tf in ("4h", "1d"):
                    rows.setdefault((f"{tf} {mode}",), {})[label] = stats(eqs[tf])
            for mode, eq in curves.items():
                rows.setdefault((f"ENSEMBLE {mode}",), {})[label] = stats(eq)
            bh = daily.loc[since:until]
            rows.setdefault(("Buy & hold",), {})[label] = stats(bh)
        print(f"\n=== Short holding cost {carry:.0%}/yr.  Cells: total return % / worst drop % "
              f"(full period also shows yearly %) ===")
        out = {}
        for k, v in rows.items():
            out[k[0]] = {lab: (f"{t:+.0f}% / {d:.0f}%" if lab != "2016-25 (all)" else f"{c:.0f}%/yr / {d:.0f}%")
                         for lab, (t, c, d) in v.items()}
        order = ["ENSEMBLE long", "ENSEMBLE short", "ENSEMBLE both", "4h long", "4h short", "4h both",
                 "1d long", "1d short", "1d both", "Buy & hold"]
        print(pd.DataFrame(out).T.loc[order].to_string())



def regime_test(carry=0.10):
    """Shorts allowed only while the daily close is below its 200-day average."""
    h1 = fetch_coinbase(interval="1h")
    daily = resample(h1, "1D")
    below = (daily.close < daily.close.rolling(200).mean()).astype(float).to_frame("b")
    rows = {}
    for since, until, label in PERIODS:
        curves = {}
        for mode in ("long", "both", "both, shorts only below 200-day avg"):
            per_tf = {}
            for tf, rule in TFS.items():
                df = resample(h1, rule)
                d = adx(df, 38, 9)
                up = (d.di_plus > d.di_minus).to_numpy()
                w = df.loc[since:until]
                idx = df.index.get_indexer(w.index)
                bpy = {"4h": 2190, "8h": 1095, "12h": 730, "1d": 365}[tf]
                if mode.startswith("both, shorts"):
                    # known at the close of the previous day only
                    b = below.reindex(w.index, method="ffill").shift(1).fillna(0)["b"].to_numpy().astype(bool)
                    # long when up; short when down AND bear regime; else cash
                    pos_up = up[idx]
                    want = np.where(pos_up, 1, np.where(b, -1, 0))
                    per_tf[tf] = _run_positions(w, want, carry, bpy)
                else:
                    per_tf[tf] = simulate(w, up[idx], mode, carry, bpy)
            eqs = pd.DataFrame(per_tf).ffill()
            curves[mode] = (1 + eqs.pct_change().fillna(0).mean(axis=1)).cumprod()
        for mode, eq in curves.items():
            t, c, d = stats(eq)
            rows.setdefault(f"ENSEMBLE {mode}", {})[label] = (f"{c:.0f}%/yr / {d:.0f}%" if label.startswith("2016-25")
                                                               else f"{t:+.0f}% / {d:.0f}%")
    pd.set_option("display.width", 250)
    print(f"\n=== Regime-filtered shorts, holding cost {carry:.0%}/yr ===")
    print(pd.DataFrame(rows).T.to_string())


def _run_positions(df, want, carry, bars_per_year):
    """Like simulate(), for an explicit target position array (+1 / 0 / -1)."""
    o, c = df.open.to_numpy(), df.close.to_numpy()
    n = len(df)
    eq = np.ones(n)
    cash, pos, entry_px, entry_eq = 1.0, 0, 0.0, 1.0
    carry_bar = carry / bars_per_year
    for i in range(n - 1):
        target = want[i]
        if target != pos:
            px = o[i + 1]
            if pos == 1:
                cash = entry_eq * px / entry_px * (1 - LONG_COST)
            elif pos == -1:
                cash = entry_eq * (2 - px / entry_px) * (1 - SHORT_COST)
            pos = target
            if pos:
                entry_px = px
                entry_eq = cash * (1 - (LONG_COST if pos == 1 else SHORT_COST))
        if pos == -1:
            entry_eq *= 1 - carry_bar
        eq[i] = (entry_eq * c[i] / entry_px if pos == 1 else
                 max(entry_eq * (2 - c[i] / entry_px), 0) if pos == -1 else cash)
    eq[-1] = eq[-2]
    return pd.Series(eq, df.index).resample("1D").last().ffill()


if __name__ == "__main__":
    main()
    regime_test()
