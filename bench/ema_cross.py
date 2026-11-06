"""EMA 8 / EMA 13 cross for long and short entries, from 15m to daily.

Rules (signal on each closed candle, fill at the next open):
  ema8/13        long while EMA 8 > EMA 13, short while below
  ema8/13+trend  the same, but only in the direction of the daily DI(38) trend
                 (as of the previous daily close); flat when they disagree
  DI cross       the bot's DI+ / DI- (38 / 9) cross, for comparison
Each runs long-only, short-only and long+short (always in the market).

Costs: longs 0.02% fee + 0.01% slippage per side (Binance.US spot taker);
shorts would need US futures: 0.05% + 0.01% per side plus 10%/yr holding cost.
A "no costs" run shows the raw edge before fees.

Data: Coinbase BTC-USD. 1h..1d back to 2016; 15m / 30m only from Nov 2024
(Coinbase serves 300 candles per request), so every timeframe is also compared
over that shorter common window.

    python bench/ema_cross.py
"""
import numpy as np
import pandas as pd

from data import fetch_coinbase, resample
from indicators import adx, ema

LONG_COST, SHORT_COST, CARRY = 0.0003, 0.0006, 0.10
LONG_TFS = {"1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
SHORT_TFS = {"15m": "15min", "30m": "30min"}
WINDOWS = [("2016-01-01", "2025-10-26", "2016 -> Oct 2025"), ("2025-10-27", None, "last 12 months"),
           ("2024-11-07", None, "Nov 2024 -> now")]


def simulate(df, want, costs=True, bars_per_year=365):
    """want: +1 / 0 / -1 per bar. Returns (daily equity, trades DataFrame)."""
    o, c = df.open.to_numpy(), df.close.to_numpy()
    lc, sc = (LONG_COST, SHORT_COST) if costs else (0.0, 0.0)
    carry = (CARRY if costs else 0.0) / bars_per_year
    n = len(df)
    eq = np.ones(n)
    cash, pos, entry_px, entry_eq, entry_i = 1.0, 0, 0.0, 1.0, 0
    trades = []
    for i in range(n - 1):
        target = want[i]
        if target != pos:
            px = o[i + 1]
            if pos:
                cash = entry_eq * (px / entry_px if pos == 1 else 2 - px / entry_px) * (1 - (lc if pos == 1 else sc))
                trades.append((df.index[entry_i], pos, cash / start_cash - 1))
            pos = target
            if pos:
                start_cash, entry_px, entry_i = cash, px, i + 1
                entry_eq = cash * (1 - (lc if pos == 1 else sc))
        if pos == -1:
            entry_eq *= 1 - carry
        eq[i] = (entry_eq * c[i] / entry_px if pos == 1 else
                 max(entry_eq * (2 - c[i] / entry_px), 0) if pos == -1 else cash)
    eq[-1] = eq[-2]
    t = pd.DataFrame(trades, columns=["time", "side", "ret"]).set_index("time")
    return pd.Series(eq, df.index).resample("1D").last().ffill(), t


def stats(eq, trades, a, b):
    w = eq.loc[a:b]
    w = w / w.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    total = w.iloc[-1] - 1
    cagr = w.iloc[-1] ** (1 / yrs) - 1 if yrs >= 1 else np.nan
    dd = (w / w.cummax() - 1).min()
    tr = trades.loc[a:b]
    return {"total": 100 * total, "cagr": 100 * cagr, "dd": 100 * dd,
            "trades/yr": len(tr) / max(yrs, 1e-9), "win %": 100 * (tr.ret > 0).mean() if len(tr) else np.nan,
            "avg trade %": 100 * tr.ret.mean() if len(tr) else np.nan}


def positions(up, mode, trend=None):
    if trend is not None:   # only trade with the daily trend
        want = np.where(up & trend, 1, np.where(~up & ~trend, -1, 0))
    else:
        want = np.where(up, 1, -1)
    if mode == "long":
        want = np.where(want == 1, 1, 0)
    elif mode == "short":
        want = np.where(want == -1, -1, 0)
    return want


def main():
    h1 = fetch_coinbase(interval="1h")
    m15 = fetch_coinbase(interval="15m", start="2024-11-01")
    daily = resample(h1, "1D")
    d = adx(daily, 38, 9)
    daily_up = (d.di_plus > d.di_minus).shift(1)       # known at the previous daily close
    frames = {tf: resample(m15, r) for tf, r in SHORT_TFS.items()}
    frames.update({tf: resample(h1, r).loc["2015-09-01":] for tf, r in LONG_TFS.items()})

    res = {}
    for tf, df in frames.items():
        bpy = 365 * 86400 / pd.Timedelta(df.index[1] - df.index[0]).total_seconds()
        up_ema = (ema(df.close, 8) > ema(df.close, 13)).to_numpy()
        di = adx(df, 38, 9)
        up_di = (di.di_plus > di.di_minus).to_numpy()
        trend = daily_up.reindex(df.index.floor("D")).fillna(False).astype(bool).to_numpy()
        rules = {"ema8/13": (up_ema, None), "ema8/13+trend": (up_ema, trend), "DI cross": (up_di, None)}
        start = "2016-01-01" if tf in LONG_TFS else "2024-11-07"
        w = df.loc[start:]
        k = df.index.get_indexer(w.index)
        for rule, (up, tr) in rules.items():
            for mode in ("long", "short", "both"):
                want = positions(up[k], mode, None if tr is None else tr[k])
                for costs in (True, False):
                    res[(tf, rule, mode, costs)] = simulate(w, want, costs, bpy)
        print("done", tf, flush=True)

    bh = daily.close
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    def table(tfs, a, b, label, costs=True, fields=("cagr", "dd", "trades/yr", "win %")):
        rows = {}
        for tf in tfs:
            for rule in ("ema8/13", "ema8/13+trend", "DI cross"):
                for mode in ("long", "short", "both"):
                    eq, tr = res[(tf, rule, mode, costs)]
                    s = stats(eq, tr, a, b)
                    rows[(tf, rule, mode)] = {f: s[f] for f in fields}
        b_eq = bh.loc[a:b]
        s = stats(b_eq, pd.DataFrame(columns=["ret"], index=pd.DatetimeIndex([])), a, b)
        rows[("BTC", "buy & hold", "")] = {f: s.get(f, np.nan) for f in fields}
        print(f"\n=== {label}{'' if costs else '  (NO COSTS: raw edge only)'} ===")
        print(pd.DataFrame(rows).T.round(1).to_string())

    table(LONG_TFS, "2016-01-01", "2025-10-26", "2016 -> Oct 2025, yearly % / worst drop %")
    table(LONG_TFS, "2016-01-01", "2025-10-26", "2016 -> Oct 2025", costs=False, fields=("cagr", "dd"))
    table(LONG_TFS, "2025-10-27", None, "Last 12 months, total %", fields=("total", "dd", "trades/yr", "win %"))
    table(list(SHORT_TFS) + ["1h", "4h", "1d"], "2024-11-07", None, "Nov 2024 -> now (2 years), yearly %",
          fields=("cagr", "dd", "trades/yr", "win %", "avg trade %"))
    table(list(SHORT_TFS) + ["1h", "4h", "1d"], "2024-11-07", None, "Nov 2024 -> now", costs=False, fields=("cagr", "dd"))

    print("\n=== Year by year, long+short, with costs (return %) ===")
    yr = {}
    for tf in LONG_TFS:
        for rule in ("ema8/13", "DI cross"):
            eq, _ = res[(tf, rule, "both", True)]
            yr[f"{tf} {rule}"] = (eq.resample("YE").last() / eq.resample("YE").last().shift(1).fillna(1) - 1).mul(100)
    yr["buy & hold"] = (bh.resample("YE").last() / bh.resample("YE").last().shift(1) - 1).mul(100)
    t = pd.DataFrame(yr).loc["2016":]
    t.index = t.index.year
    print(t.T.round(0).to_string())


if __name__ == "__main__":
    main()
