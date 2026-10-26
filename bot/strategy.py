"""The trading rule: daily DI+ / DI- crossover (DI length 38).

Hold BTC while DI+ is above DI-, hold cash while it is below. This is the rule
tested in bench/exits.py ("DI cross"); the bot only ever acts on closed candles.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.indicators import adx  # noqa: E402


def evaluate(candles, di_len=38, adx_smoothing=9):
    """Indicator readings on the last closed candle."""
    d = adx(candles, di_len, adx_smoothing)
    last, prev = d.iloc[-1], d.iloc[-2]
    return {
        "candle_time": candles.index[-1],
        "close": float(candles["close"].iloc[-1]),
        "di_plus": round(float(last.di_plus), 2),
        "di_minus": round(float(last.di_minus), 2),
        "adx": round(float(last.adx), 2),
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
