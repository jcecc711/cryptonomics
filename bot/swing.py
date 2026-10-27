"""The swing layer: 4-hour EMA 8/21 crossover, traded only while the daily DI
trend (the core's signal) is up.

Buy when the 8 EMA crosses above the 21 EMA on a closed 4h candle and the daily
DI+ is above DI-. Sell when the 8 EMA drops back below the 21, or the daily
trend turns down. This is the rule tested in bench/short_term.py.
"""
import pandas as pd

from .strategy import adx, ema


def daily_up_at(index, bar_len, daily, di_len=38, adx_smoothing=9):
    """Daily DI trend as known when each candle in `index` closed (closed days only)."""
    d = adx(daily, di_len, adx_smoothing)
    up = pd.DataFrame({"close_time": daily.index + pd.Timedelta(days=1),
                       "up": (d.di_plus > d.di_minus).to_numpy()})
    bars = pd.DataFrame({"close_time": index + bar_len})
    merged = pd.merge_asof(bars, up, on="close_time", direction="backward")
    return pd.Series(merged["up"].fillna(False).astype(bool).to_numpy(), index=index)


def evaluate(candles, daily, fast=8, slow=21, di_len=38, adx_smoothing=9, lookback=60):
    """Readings on the last closed 4h candle, plus recent crosses for catching up."""
    c = candles["close"]
    f, s = ema(c, fast), ema(c, slow)
    above = f > s
    cross_up = above & ~above.shift(fill_value=True)
    daily_up = daily_up_at(candles.index, pd.Timedelta(hours=4), daily, di_len, adx_smoothing)
    recent = pd.DataFrame({"cross_up": cross_up, "daily_up": daily_up}).tail(lookback)
    return {
        "candle_time": candles.index[-1],
        "close": float(c.iloc[-1]),
        "readings": {"ema_fast": round(float(f.iloc[-1]), 2), "ema_slow": round(float(s.iloc[-1]), 2),
                     "daily_up": bool(daily_up.iloc[-1])},
        "trend_up": bool(above.iloc[-1]),
        "daily_up": bool(daily_up.iloc[-1]),
        "recent": recent,
    }


def decide(sig, holding, state):
    """Return (action, state). `state` holds the last candle already processed,
    so crosses on candles the bot missed (e.g. while offline) still count."""
    last = state.get("last")
    state = {"last": str(sig["candle_time"])}
    if holding:
        return ("SELL" if not (sig["trend_up"] and sig["daily_up"]) else None), state
    recent = sig["recent"]
    new = recent.iloc[-1:] if last is None else recent[recent.index > pd.Timestamp(last)]
    crosses = new[new["cross_up"]]
    if len(crosses) and crosses["daily_up"].iloc[-1] and sig["trend_up"] and sig["daily_up"]:
        return "BUY", state
    return None, state


def summary(sig, trade, price):
    r = sig["readings"]
    ema_txt = f"4h EMA8 {r['ema_fast']:,.0f} {'>' if sig['trend_up'] else '<'} EMA21 {r['ema_slow']:,.0f}"
    if trade:
        pnl = price / trade["entry_price"] - 1
        return f"{ema_txt}. Holding since {trade['entry_time'][:16]} at {trade['entry_price']:,.0f} ({pnl:+.1%})."
    if not sig["daily_up"]:
        return f"{ema_txt}. In cash: daily trend is down, swing trades paused."
    return f"{ema_txt}. In cash, waiting for the 8 EMA to cross above the 21."
