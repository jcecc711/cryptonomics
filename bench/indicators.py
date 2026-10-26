"""Indicators matching TradingView's built-in definitions."""
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rma(s, n):
    """Wilder's moving average (TradingView ta.rma)."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, di_len=14, adx_len=14):
    """Return DataFrame with di_plus, di_minus, adx (TradingView ta.dmi)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, di_len)
    di_plus = 100 * rma(plus_dm, di_len) / atr
    di_minus = 100 * rma(minus_dm, di_len) / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    return pd.DataFrame({"di_plus": di_plus, "di_minus": di_minus,
                         "adx": rma(dx, adx_len)})
