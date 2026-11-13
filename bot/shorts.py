"""Tracked shorts: what a Coinbase BTC futures short would be doing. No orders.

Rule ("8h + 4h agree"): short while DI- is on top on both the 4h and the 8h
(closed candles, DI 38 / smoothing 9); close as soon as the 4h DI+ crosses back
above DI-. Size: whole Coinbase nano contracts (0.01 BTC each), BOT_SHORT_CONTRACTS.

Costs are estimates: fee + slippage per side on the short's value, and a funding
/ holding cost accrued per day held (BOT_SHORT_FUNDING_APR, default 10%/yr).
On its first run, a short the rule is already in is picked up from the candle
where the 4h and 8h first agreed (marked "picked up").
"""
import pandas as pd

from bench.indicators import adx

CONTRACT_BTC = 0.01
KEY = "short_tracker"


def down_now(c4, c8, di_len, smoothing):
    """(4h down, 8h down) on the latest closed candles, plus the 4h/8h series."""
    d4, d8 = adx(c4, di_len, smoothing), adx(c8, di_len, smoothing)
    dn4 = d4.di_minus > d4.di_plus
    dn8 = d8.di_minus > d8.di_plus
    return dn4, dn8, d4, d8


def streak_start(c4, dn4, dn8):
    """Open time and price of the 4h candle after the current both-down streak began."""
    dn8s = dn8.copy()
    dn8s.index = dn8s.index + pd.Timedelta("4h")        # an 8h reading is known once that candle closes
    both = dn4 & dn8s.reindex(c4.index, method="ffill").fillna(False).astype(bool)
    if not both.iloc[-1]:
        return None
    starts = both & ~both.shift(fill_value=False)
    t = starts[starts].index[-1]
    i = c4.index.get_loc(t)
    if i + 1 >= len(c4):
        return None                                      # signal on the latest candle: enter at the live price
    return c4.index[i + 1], float(c4.open.iloc[i + 1])


def mark(pos, price, now, cfg):
    """Profit / loss of an open or closed short at `price`, after estimated costs."""
    qty = pos["qty"]
    gross = qty * (pos["entry"] - price)
    days = max((pd.Timestamp(now) - pd.Timestamp(pos["opened"])).total_seconds() / 86400, 0)
    fees = qty * (pos["entry"] + price) * cfg.short_cost
    funding = qty * pos["entry"] * cfg.short_funding_apr * days / 365
    return {"gross": gross, "fees": fees, "funding": funding, "net": gross - fees - funding, "days": days,
            "pct": 100 * (pos["entry"] - price) / pos["entry"]}


def step(cfg, store, c4, c8, price, now):
    """Update the tracked short; returns (event, state). event: 'open', 'close' or None."""
    st = store.get(KEY) or {"open": None, "closed": []}
    dn4, dn8, d4, d8 = down_now(c4, c8, cfg.di_len, cfg.adx_smoothing)
    both = bool(dn4.iloc[-1] and dn8.iloc[-1])
    event = None
    pos = st["open"]
    if pos and not dn4.iloc[-1]:                         # 4h DI+ back on top: close
        m = mark(pos, price, now, cfg)
        st["closed"].insert(0, {**pos, "closed": str(now), "exit": price, **{k: round(v, 2) for k, v in m.items()}})
        st["open"], event = None, "close"
    elif not pos and both:
        first = not store.get(KEY)
        start = streak_start(c4, dn4, dn8) if first else None
        opened, entry = (str(start[0]), start[1]) if start else (str(now), price)
        st["open"] = {"opened": opened, "entry": entry, "contracts": cfg.short_contracts,
                      "qty": round(cfg.short_contracts * CONTRACT_BTC, 8), "picked_up": bool(start)}
        event = "open"
    st["signal"] = {"4h": "down" if dn4.iloc[-1] else "up", "8h": "down" if dn8.iloc[-1] else "up",
                    "di_plus_4h": round(float(d4.di_plus.iloc[-1]), 1), "di_minus_4h": round(float(d4.di_minus.iloc[-1]), 1),
                    "di_plus_8h": round(float(d8.di_plus.iloc[-1]), 1), "di_minus_8h": round(float(d8.di_minus.iloc[-1]), 1),
                    "adx_4h": round(float(d4.adx.iloc[-1]), 1), "as_of": str(c4.index[-1])}
    st["price"], st["updated"] = price, str(now)
    if st["open"]:
        st["open"]["mark"] = {k: round(v, 2) for k, v in mark(st["open"], price, now, cfg).items()}
    st["realized"] = round(sum(t["net"] for t in st["closed"]), 2)
    st["closed"] = st["closed"][:200]
    store.set(KEY, st)
    return event, st
