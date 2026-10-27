"""Broad scan: strategy families x timeframes, long-only spot, BTC 2016 -> Oct 2025.

Every family is run over a small grid of standard settings, and judged by the
*median* result across that grid (not its best setting), so a lucky parameter
can't win. Costs on every trade: 0.02% fee + 0.01% slippage per side.
Sub-daily timeframes are also run with the daily DI(38) trend filter (enter
only while the daily trend is up, exit when it turns down).

    python bench/strategy_scan.py            # writes data/scan_runs.csv
"""
import numpy as np
import pandas as pd

import backtest
from data import DATA_DIR, fetch_coinbase, resample
from indicators import adx, ema, rma, sma
from mtf import align

START, END = "2016-01-01", "2025-10-26"
SUBPERIODS = [("2016-01-01", "2017-12-31"), ("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"),
              ("2022-01-01", "2022-12-31"), ("2023-01-01", "2025-10-26")]
TIMEFRAMES = {"1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D"}
FEE, SLIP = 0.0002, 0.0001


def up_cross(a, b):
    return (a > b) & (a.shift() <= b.shift())


def dn_cross(a, b):
    return (a < b) & (a.shift() >= b.shift())


def atr(df, n):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def supertrend(df, n, mult):
    a = atr(df, n).to_numpy()
    hl2 = ((df.high + df.low) / 2).to_numpy()
    c = df.close.to_numpy()
    up, dn = hl2 - mult * a, hl2 + mult * a
    trend = np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        trend[i] = c[i] > dn[i - 1] if not trend[i - 1] else not c[i] < up[i - 1]
    return pd.Series(trend, df.index)


def rsi(c, n):
    d = c.diff()
    g, l = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + g / l)


def state_signals(state):
    state = state.astype(bool)
    return state & ~state.shift(fill_value=False), ~state & state.shift(fill_value=False)


def families(df):
    """{family: {setting: (entry, exit)}}"""
    c = df.close
    out = {}
    out["DI cross"] = {f"DI{L}": (up_cross(d.di_plus, d.di_minus), dn_cross(d.di_plus, d.di_minus))
                       for L in (14, 21, 28, 38, 50) for d in [adx(df, L)]}
    out["EMA cross"] = {f"{f}/{s}": (up_cross(ema(c, f), ema(c, s)), dn_cross(ema(c, f), ema(c, s)))
                        for f, s in ((8, 21), (13, 34), (21, 55), (34, 89), (50, 200))}
    out["Price vs EMA"] = {f"EMA{n}": (up_cross(c, ema(c, n)), dn_cross(c, ema(c, n))) for n in (20, 50, 100, 200)}
    out["Breakout"] = {f"{n}/{m}": (c > df.high.rolling(n).max().shift(), c < df.low.rolling(m).min().shift())
                       for n, m in ((20, 10), (40, 20), (55, 20), (100, 50), (200, 100))}
    out["Supertrend"] = {f"{n}x{k}": state_signals(supertrend(df, n, k))
                         for n, k in ((10, 2), (10, 3), (10, 4), (20, 3), (20, 5))}
    macd = {}
    for k in (1, 2, 3):
        m = ema(c, 12 * k) - ema(c, 26 * k)
        sig = ema(m, 9 * k)
        macd[f"{12 * k}/{26 * k}/{9 * k}"] = (up_cross(m, sig), dn_cross(m, sig))
    out["MACD"] = macd
    out["Momentum"] = {f"ROC{n}": state_signals(c > c.shift(n)) for n in (20, 50, 100, 200)}
    out["RSI dip-buy"] = {f"RSI{n}<{t}": ((rsi(c, n) < t) & (c > sma(c, 200)), c > sma(c, 5))
                          for n, t in ((2, 10), (2, 20), (3, 10), (3, 20))}
    return out


def metrics(eq, trades, years):
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    wins, losses = trades.ret[trades.ret > 0].sum(), -trades.ret[trades.ret <= 0].sum() if len(trades) else (0, 0)
    return {"CAGR%": 100 * cagr, "max DD%": 100 * dd, "calmar": cagr / -dd if dd < 0 else 0,
            "trades/yr": len(trades) / years, "PF": wins / losses if len(trades) and losses else np.nan}


def sub_cagr(eq, s0, s1):
    e = eq.loc[s0:s1]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return 100 * ((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1)


def main():
    h1 = fetch_coinbase(interval="1h")
    daily = resample(h1, "1D")
    di = adx(daily, 38, 9)
    daily_up_d = (di.di_plus > di.di_minus).astype(float).to_frame("up")
    rows = []
    for tf, rule in TIMEFRAMES.items():
        df = h1 if tf == "1h" else resample(h1, rule)
        bar = pd.Timedelta(rule)
        up = align(daily_up_d, pd.Timedelta(days=1), df.index, bar)["up"].fillna(0).astype(bool)
        w = df.loc[START:END]
        years = (w.index[-1] - w.index[0]).days / 365.25
        for fam, grid in families(df).items():
            for setting, (e, x) in grid.items():
                variants = [("none", e, x)]
                if tf != "1d":
                    variants.append(("daily filter", e & up, x | ~up))
                for filt, ee, xx in variants:
                    t, eq = backtest.run(w, ee.loc[w.index].fillna(False), xx.loc[w.index].fillna(False),
                                         fee=FEE, slippage=SLIP)
                    eqd = eq.resample("1D").last()
                    r = {"tf": tf, "family": fam, "setting": setting, "filter": filt, **metrics(eq, t, years)}
                    for s0, s1 in SUBPERIODS:
                        r[f"{s0[:4]}-{s1[2:4]}"] = sub_cagr(eqd, s0, s1)
                    rows.append(r)
        print("done", tf, flush=True)
    bh = daily.close.loc[START:END]
    print("Buy & hold:", metrics(bh / bh.iloc[0], pd.DataFrame({"ret": []}), (bh.index[-1] - bh.index[0]).days / 365.25))
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "scan_runs.csv", index=False)
    print(f"{len(out)} runs -> {DATA_DIR / 'scan_runs.csv'}")


if __name__ == "__main__":
    main()
