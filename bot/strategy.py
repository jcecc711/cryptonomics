"""The core rule: daily DI+ / DI- crossover (DI length 38).

Hold BTC while DI+ is above DI-, hold cash while it is below. This is the rule
tested in bench/exits.py ("DI cross"); the bot only ever acts on closed candles.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.indicators import adx, ema  # noqa: E402,F401  (ema is used by bot.swing)


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
    di = f"Daily DI+ {r['di_plus']:.1f} / DI- {r['di_minus']:.1f}"
    if trade:
        pnl = price / trade["entry_price"] - 1
        return f"{di}. Holding since {trade['entry_time'][:10]} at {trade['entry_price']:,.0f} ({pnl:+.1%})."
    if sig["trend_up"]:
        return f"{di}. In cash; DI+ is above DI- but waiting for a fresh cross up."
    return f"{di}. In cash, waiting for DI+ to cross above DI-."
