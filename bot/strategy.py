"""The rules, each run on several timeframes. The bot only acts on closed candles.

DI cross: hold BTC while DI+ is above DI-, cash while it is below. Tested in
bench/exits.py ("DI cross") and, across timeframes, in bench/di_timeframes.py.

ADX trigger: buy when ADX rises through 14 with DI+ on top; sell when DI+
crosses below DI-. Tested in bench/adx14.py and bench/adx_hybrid.py, where a
50/50 blend with the DI cross kept the same return for the risk with smaller
drawdowns.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.indicators import adx  # noqa: E402


def evaluate(candles, di_len=38, adx_smoothing=9):
    """Indicator readings on the last closed candle."""
    d = adx(candles, di_len, adx_smoothing)
    last, prev = d.iloc[-1], d.iloc[-2]
    return {
        "candle_time": candles.index[-1],
        "close": float(candles["close"].iloc[-1]),
        "readings": {"di_plus": round(float(last.di_plus), 2), "di_minus": round(float(last.di_minus), 2),
                     "adx": round(float(last.adx), 2)},
        "trend_up": bool(last.di_plus > last.di_minus),
        "crossed_up": bool(last.di_plus > last.di_minus and prev.di_plus <= prev.di_minus),
        "crossed_down": bool(last.di_plus < last.di_minus and prev.di_plus >= prev.di_minus),
    }


def decide(sig, holding, armed):
    """Return (action, armed).

    `armed` is False only when the bot started mid-uptrend and is waiting for a
    fresh cross; it re-arms as soon as DI+ is back below DI-.
    """
    if holding:
        return ("SELL" if not sig["trend_up"] else None), armed
    if not sig["trend_up"]:
        return None, True
    return ("BUY" if armed else None), armed


def summary(sig, trade, price):
    r = sig["readings"]
    di = f"DI+ {r['di_plus']:.1f} / DI- {r['di_minus']:.1f}"
    if trade:
        pnl = price / trade["entry_price"] - 1
        return f"{di}. Holding since {trade['entry_time'][:16]} at {trade['entry_price']:,.0f} ({pnl:+.1%})."
    if sig["trend_up"]:
        return f"{di}. In cash; DI+ is above DI- but waiting for a fresh cross up."
    return f"{di}. In cash, waiting for DI+ to cross above DI-."


# -- ADX trigger ---------------------------------------------------------------

def evaluate_adx(candles, di_len=38, adx_smoothing=9, level=14.0, lookback=60):
    """Readings on the last closed candle, plus recent trigger candles for catching up."""
    d = adx(candles, di_len, adx_smoothing)
    up = d.di_plus > d.di_minus
    trigger = (d.adx > level) & (d.adx.shift() <= level) & up
    last = d.iloc[-1]
    return {
        "candle_time": candles.index[-1],
        "close": float(candles["close"].iloc[-1]),
        "readings": {"di_plus": round(float(last.di_plus), 2), "di_minus": round(float(last.di_minus), 2),
                     "adx": round(float(last.adx), 2), "level": level},
        "trend_up": bool(up.iloc[-1]),
        "recent": trigger.tail(lookback),
    }


def decide_adx(sig, holding, state):
    """Return (action, state). `state` holds the last candle already processed,
    so a trigger on a candle the bot missed (e.g. while offline) still counts
    as long as DI+ is still on top; on the first run only the latest candle counts."""
    last = state.get("last")
    state = {"last": str(sig["candle_time"])}
    if holding:
        return ("SELL" if not sig["trend_up"] else None), state
    recent = sig["recent"]
    new = recent.iloc[-1:] if last is None else recent[recent.index > pd.Timestamp(last)]
    return ("BUY" if bool(new.any()) and sig["trend_up"] else None), state


def summary_adx(sig, trade, price):
    r = sig["readings"]
    head = f"ADX {r['adx']:.1f} · DI+ {r['di_plus']:.1f} / DI- {r['di_minus']:.1f}"
    if trade:
        pnl = price / trade["entry_price"] - 1
        return f"{head}. Holding since {trade['entry_time'][:16]} at {trade['entry_price']:,.0f} ({pnl:+.1%})."
    if sig["trend_up"]:
        return f"{head}. In cash; DI+ leads, waiting for ADX to rise through {r['level']:g}."
    return f"{head}. In cash; waiting for DI+ to lead and ADX to rise through {r['level']:g}."
