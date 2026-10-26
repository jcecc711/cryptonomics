"""Long-only spot backtester.

Signals are evaluated on each closed candle; trades fill at the next candle's
open, which is what a live bot reacting to a candle close would get.
"""
import numpy as np
import pandas as pd


def run(df, entry, exit_, fee=0.0002, slippage=0.0002):
    """entry/exit_ are boolean Series. Cost per side = fee + slippage."""
    cost = fee + slippage
    in_pos = False
    trades, equity = [], np.ones(len(df))
    cash, units, entry_px, entry_t = 1.0, 0.0, 0.0, None
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    ent, ext = entry.to_numpy(), exit_.to_numpy()

    for i in range(len(df) - 1):
        if not in_pos and ent[i]:
            entry_px, entry_t = opens[i + 1], df.index[i + 1]
            units = cash * (1 - cost) / entry_px
            cash, in_pos = 0.0, True
        elif in_pos and ext[i]:
            px = opens[i + 1]
            cash = units * px * (1 - cost)
            trades.append({"entry_time": entry_t, "exit_time": df.index[i + 1],
                           "entry": entry_px, "exit": px,
                           "ret": px / entry_px * (1 - cost) ** 2 - 1})
            units, in_pos = 0.0, False
        equity[i] = cash + units * closes[i]
    equity[-1] = cash + units * closes[-1]
    if in_pos:  # still holding: mark to the last close so stats include it
        trades.append({"entry_time": entry_t, "exit_time": df.index[-1],
                       "entry": entry_px, "exit": closes[-1],
                       "ret": closes[-1] / entry_px * (1 - cost) ** 2 - 1, "open": True})
    return pd.DataFrame(trades), pd.Series(equity, index=df.index)


def stats(trades, equity, df):
    years = (df.index[-1] - df.index[0]).days / 365.25
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "trades": len(trades),
        "trades/yr": len(trades) / years,
        "win%": 100 * (trades["ret"] > 0).mean() if len(trades) else 0,
        "avg win%": 100 * trades.loc[trades["ret"] > 0, "ret"].mean() if len(trades) else 0,
        "avg loss%": 100 * trades.loc[trades["ret"] <= 0, "ret"].mean() if len(trades) else 0,
        "profit factor": (trades.loc[trades["ret"] > 0, "ret"].sum()
                          / -trades.loc[trades["ret"] <= 0, "ret"].sum()) if len(trades) else 0,
        "total%": 100 * (equity.iloc[-1] - 1),
        "CAGR%": 100 * (equity.iloc[-1] ** (1 / years) - 1),
        "max DD%": 100 * dd,
        "time in mkt%": None,
    }
    if len(trades):
        held = sum((t.exit_time - t.entry_time).total_seconds() for t in trades.itertuples())
        out["time in mkt%"] = 100 * held / (df.index[-1] - df.index[0]).total_seconds()
    return out


def buy_hold(df):
    eq = df["close"] / df["close"].iloc[0]
    years = (df.index[-1] - df.index[0]).days / 365.25
    return {"total%": 100 * (eq.iloc[-1] - 1),
            "CAGR%": 100 * (eq.iloc[-1] ** (1 / years) - 1),
            "max DD%": 100 * (eq / eq.cummax() - 1).min()}
