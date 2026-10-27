"""Short-timeframe dip buying inside the daily uptrend, long-only spot.

While the daily DI(38) trend is up, rest a buy limit `k` ATRs below the 21 EMA.
When a sell-off trades through it, buy (maker, 0% fee). Then rest a sell limit
at the 21 EMA (maker): the bounce back to the average is the profit.
Protection, as taker orders (0.02% fee + 0.01% slippage): a stop `stop` ATRs
under the entry, a time limit, and an exit if the daily trend turns down.

Fills are conservative: a limit only fills if price trades *through* it, and if
a candle reaches both the stop and the target, the stop is assumed first.

    python bench/dip_buy.py [since] [until]
"""
import sys

import numpy as np
import pandas as pd

import backtest
from data import fetch, resample
from indicators import ema, rma
from short_term import HOLDOUT_START, daily_trend

TAKER = 0.0002
SLIP = 0.0001


def atr(df, n=14):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def dip_buy(df, daily_up, k=2.0, stop=3.0, max_bars=60, maker_fee=0.0):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    mid, a = ema(df.close, 21).to_numpy(), atr(df).to_numpy()
    up = daily_up.to_numpy()
    n = len(df)
    eq = np.ones(n)
    cash, units, in_pos = 1.0, 0.0, False
    entry_px = stop_px = 0.0
    entry_t, held = None, 0
    trades = []

    def close_trade(j, px, fee):
        nonlocal cash, units, in_pos
        cash = units * px * (1 - fee)
        trades.append({"entry_time": entry_t, "exit_time": df.index[j], "entry": entry_px,
                       "exit": px, "ret": cash / (units * entry_px / (1 - maker_fee)) - 1})
        units, in_pos = 0.0, False

    for j in range(30, n):
        if not in_pos:
            buy = mid[j - 1] - k * a[j - 1]
            if up[j] and l[j] < buy:
                entry_px = min(o[j], buy)
                units, cash, in_pos = cash * (1 - maker_fee) / entry_px, 0.0, True
                stop_px, entry_t, held = entry_px - stop * a[j - 1], df.index[j], 0
        else:
            held += 1
            target = mid[j - 1]
            if l[j] <= stop_px:
                close_trade(j, min(o[j], stop_px) * (1 - SLIP), TAKER)
            elif h[j] > target:
                close_trade(j, max(o[j], target), maker_fee)
            elif held >= max_bars or not up[j]:
                close_trade(j, c[j] * (1 - SLIP), TAKER)
        eq[j] = cash + units * c[j]
    eq[:30] = 1.0
    return pd.DataFrame(trades), pd.Series(eq, index=df.index)


def row(t, eq, w):
    s = backtest.stats(t, eq, w)
    s["avg trade%"] = 100 * t["ret"].mean() if len(t) else 0
    return {k: s[k] for k in ("trades/yr", "win%", "avg trade%", "profit factor", "CAGR%", "max DD%", "time in mkt%")}


def main():
    since = sys.argv[1] if len(sys.argv) > 1 else "2024-10-15"
    until = sys.argv[2] if len(sys.argv) > 2 else HOLDOUT_START
    base = fetch("BTCUSDT", "3m")
    pd.set_option("display.width", 250)
    for tf, rule, max_bars in (("6m", "6min", 60), ("15m", "15min", 24), ("1h", "1h", 12)):
        df = resample(base, rule)
        up = daily_trend(df.index, pd.Timedelta(rule))
        always = pd.Series(True, index=df.index)
        w = df.loc[since:until]
        rows = {}
        for k in (2.0, 3.0, 4.0):
            for label, filt in (("daily up only", up), ("no filter", always)):
                t, eq = dip_buy(w, filt.loc[w.index], k=k, max_bars=max_bars)
                rows[f"buy {k:.0f} ATR below 21 EMA, {label}"] = row(t, eq, w)
        bh = backtest.buy_hold(w)
        rows["Buy & hold"] = {"CAGR%": bh["CAGR%"], "max DD%": bh["max DD%"]}
        print(f"\n=== {tf} candles (time limit {max_bars} candles)  {w.index[0]:%Y-%m-%d} -> {w.index[-1]:%Y-%m-%d} ===")
        print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
