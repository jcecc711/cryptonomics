"""Bull / bear market regime: the halving cycle plus price trend checks.

The bear call comes from the halving calendar, the best regime signal in
bench/regime_study.py: days 540-910 after a halving have held the post-top
bear market in each past cycle (tops about 17-18 months after the halving,
bottoms about 30 months after). Price checks (200-day and 50-week averages, the
20-week/21-week "bull market support band") are shown as context; on their own
they fire during bull-market pullbacks too.
"""
import pandas as pd

from .strategy import adx  # noqa: F401  (keeps bench on the path)
from bench.indicators import ema, sma

HALVINGS = [pd.Timestamp(d) for d in ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")]
NEXT_HALVING_ESTIMATE = pd.Timestamp("2028-04-15")   # block 1,050,000; the exact date depends on block times
BEAR_START, BEAR_END = 540, 910                       # days after a halving


def phase(days):
    if days < 180:
        return "Post-halving build-up"
    if days < BEAR_START:
        return "Bull run"
    if days <= BEAR_END:
        return "Bear market"
    return "Recovery"


def calendar_bear(when):
    """True if `when` falls in a calendar bear window."""
    when = pd.Timestamp(when)
    past = [h for h in HALVINGS + [NEXT_HALVING_ESTIMATE] if h <= when]
    return bool(past) and BEAR_START <= (when - past[-1]).days <= BEAR_END


def assess(daily, when=None):
    """Regime readings from daily candles (closed), as of the last candle or `when`."""
    c = daily["close"]
    when = pd.Timestamp(when) if when is not None else c.index[-1] + pd.Timedelta(days=1)
    last_h = max(h for h in HALVINGS if h <= when)
    next_h = next((h for h in HALVINGS if h > when), NEXT_HALVING_ESTIMATE)
    days = (when - last_h).days
    cycle = (next_h - last_h).days
    bear = BEAR_START <= days <= BEAR_END

    price = float(c.iloc[-1])
    checks = []
    for name, value, n_needed in (("200-day average", sma(c, 200).iloc[-1], 200),
                                  ("50-week average", sma(c, 350).iloc[-1], 350),
                                  ("20-week average", sma(c, 140).iloc[-1], 140),
                                  ("21-week EMA", ema(c, 147).iloc[-1], 147)):
        if len(c) >= n_needed and value == value:
            checks.append({"name": name, "value": round(float(value), 2), "above": bool(price > value),
                           "gap": round(100 * (price / value - 1), 1)})
    above = sum(x["above"] for x in checks)

    if bear:
        call, bias = "Bear", "Halving calendar: post-top bear window. Bot takes no new longs and exits longs."
    elif not checks:
        call, bias = "Unknown", "Not enough price history for the trend checks."
    elif above == len(checks):
        call, bias = "Bull", "Price above all long-term averages, outside the bear window."
    elif above == 0:
        call, bias = "Caution", "Price below all long-term averages, though outside the bear window."
    else:
        call, bias = "Neutral", f"Price above {above} of {len(checks)} long-term averages."

    window_base = last_h if days <= BEAR_END else next_h
    return {
        "call": call, "bias": bias, "calendar_bear": bool(bear),
        "halving": str(last_h.date()), "days_since_halving": days, "cycle_days": cycle,
        "phase": phase(days),
        "bear_window": [str((window_base + pd.Timedelta(days=BEAR_START)).date()),
                        str((window_base + pd.Timedelta(days=BEAR_END)).date())],
        "bear_window_estimated": window_base == NEXT_HALVING_ESTIMATE,
        "next_halving": str(next_h.date()), "next_halving_estimated": next_h == NEXT_HALVING_ESTIMATE,
        "price": round(price, 2), "checks": checks,
        "as_of": str(c.index[-1].date()),
    }
