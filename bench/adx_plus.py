"""Can ADX be strengthened? Entry filters for "trend building steam" and exits for
"trend slowing / reversing", tested on the live 8-sleeve strategy.

How ADX is built (Wilder): +DM / -DM (the part of each candle's range beyond the
previous one), smoothed with an RMA of DI length (38) and divided by ATR -> DI+ / DI-.
DX = |DI+ - DI-| / (DI+ + DI-); ADX = RMA(DX, 9). So ADX measures how one-sided the
movement is (trend strength, not direction), and with two layers of smoothing it lags:
it peaks after the move and stays high while a trend is already ending.

Baseline (live): DI sleeve enters on DI+ crossing above DI-, ADX sleeve enters when ADX
rises through 14 with DI+ on top; both exit when DI+ crosses below DI-. 4h / 8h / 12h /
1d, equal shares, halving bear filter, 0.02% fee + 0.01% slippage per side.

Entry filters (applied to both sleeves' entries), standard settings, not tuned:
  RSI>50        RSI(14) above 50 at the entry (momentum agrees)
  ADX rising    ADX above its value 3 candles ago (trend gaining strength)
  CHOP<61.8     Choppiness Index(14) below 61.8 (market not ranging)
  ER>0.3        Kaufman Efficiency Ratio(20) above 0.3 (price moving efficiently)
  >EMA55        close above EMA 55
Extra exits (in addition to the DI cross down):
  ADX rollover  ADX was above 40 in the last 10 candles and is now 5+ points below that peak
  RSI<50        RSI(14) crosses below 50
  Chandelier    close below the 22-candle high minus 3 x ATR(22)
  DI14 cross    the faster DI(14) crosses down
A change counts only if it improves return / max drawdown in both halves
(2016-20, 2021-Oct 2025) on most timeframes. The last 12 months are reported apart.

Then long + short versions scored on bitcoin stacked (BTC at the end per BTC the starting
dollars bought): shorts always on, with ADX / RSI rules, or only in the halving bear window.

Also: what each ADX state has meant for the next 12 candles (event study).

    python bench/adx_plus.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
from data import resample
from indicators import adx, ema, rma
from short_hedge import hourly, simulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.regime import calendar_bear  # noqa: E402

TFS = {"4h": "4h", "8h": "8h", "12h": "12h", "1d": "1D"}
FEE, SLIP = 0.0002, 0.0001
HALVES = [("2016-01-01", "2020-12-31", "2016-20"), ("2021-01-01", "2025-10-26", "2021-25")]
RECENT = "2025-10-27"


# -- indicators -----------------------------------------------------------------

def rsi(c, n=14):
    d = c.diff()
    return 100 - 100 / (1 + rma(d.clip(lower=0), n) / rma(-d.clip(upper=0), n))


def true_range(df):
    pc = df.close.shift()
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def chop(df, n=14):
    tr = true_range(df).rolling(n).sum()
    rng = df.high.rolling(n).max() - df.low.rolling(n).min()
    return 100 * np.log10(tr / rng) / np.log10(n)


def efficiency(c, n=20):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def build(df):
    d = adx(df, 38, 9)
    f = df.join(d)
    f["up"] = f.di_plus > f.di_minus
    f["rsi"] = rsi(f.close)
    f["chop"] = chop(f)
    f["er"] = efficiency(f.close)
    f["ema55"] = ema(f.close, 55)
    f["atr22"] = rma(true_range(f), 22)
    fast = adx(df, 14, 9)
    f["up14"] = fast.di_plus > fast.di_minus
    days = f.index.normalize()
    bear = {dd: calendar_bear(dd) for dd in days.unique()}
    f["bear"] = [bear[dd] for dd in days]
    return f


# -- rules -------------------------------------------------------------------------

def edge(s):
    return s & ~s.shift(fill_value=False)


ENTRY_FILTERS = {
    "RSI>50": lambda f: f.rsi > 50,
    "ADX rising": lambda f: f.adx > f.adx.shift(3),
    "CHOP<61.8": lambda f: f.chop < 61.8,
    "ER>0.3": lambda f: f.er > 0.3,
    ">EMA55": lambda f: f.close > f.ema55,
}
EXITS = {
    "ADX rollover": lambda f: (f.adx.rolling(10).max() > 40) & (f.adx < f.adx.rolling(10).max() - 5),
    "RSI<50": lambda f: (f.rsi < 50) & (f.rsi.shift() >= 50),
    "Chandelier": lambda f: f.close < f.high.rolling(22).max() - 3 * f.atr22,
    "DI14 cross": lambda f: ~f.up14 & f.up14.shift(fill_value=False),
}


def signals(f, kind, filt=None, extra_exit=None):
    ok = filt(f) if filt else pd.Series(True, f.index)
    if kind == "di":
        enter = edge(f.up & ok)            # DI+ on top (and the filter agrees), freshly
    else:
        enter = (f.adx > 14) & (f.adx.shift() <= 14) & f.up & ok
    leave = ~f.up & f.up.shift(fill_value=False)
    if extra_exit:
        leave = leave | extra_exit(f)
    return enter & ~f.bear, leave | f.bear


def run_variant(frames, filt=None, extra_exit=None):
    curves = {}
    for tf, f in frames.items():
        w = f.loc["2016-01-01":]
        for kind in ("di", "adx"):
            e, x = signals(f, kind, filt, extra_exit)
            t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index], fee=FEE, slippage=SLIP)
            curves[(tf, kind)] = (eq.resample("1D").last().ffill(), len(t))
    return curves


def stats(eq, a=None, b=None):
    w = eq.loc[a:b] if a else eq
    w = w / w.iloc[0]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    dd = (w / w.cummax() - 1).min()
    total = w.iloc[-1] - 1
    cagr = w.iloc[-1] ** (1 / yrs) - 1 if yrs >= 1 else np.nan
    return cagr, dd, (cagr / -dd if dd < 0 and yrs >= 1 else np.nan), total


def ensemble(curves):
    r = pd.concat([c.pct_change() for c, _ in curves.values()], axis=1).fillna(0).mean(axis=1)
    return (1 + r).cumprod()


def pair(curves, tf):
    r = pd.concat([curves[(tf, k)][0].pct_change() for k in ("di", "adx")], axis=1).fillna(0).mean(axis=1)
    return (1 + r).cumprod()


# -- what ADX states have meant (event study) -------------------------------------------

def states(f, h=12):
    rising = f.adx > f.adx.shift(3)
    peak = f.adx.rolling(10).max()
    s = pd.Series("", f.index)
    s[f.up & rising & (f.adx < 25)] = "up trend building (ADX rising, <25)"
    s[f.up & rising & (f.adx >= 25)] = "up trend strong (ADX rising, 25+)"
    s[f.up & ~rising & (peak > 40)] = "up trend exhausting (ADX falling from 40+)"
    s[f.up & ~rising & (peak <= 40)] = "up trend fading (ADX falling, peak <40)"
    s[~f.up & rising] = "down trend building / strong"
    s[~f.up & ~rising] = "down trend fading"
    fwd = f.close.shift(-h) / f.close - 1
    change = f.up.ne(f.up.shift()).astype(float)
    flip = change[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1) > 0     # DI flips in the next h candles
    w = slice("2016-01-01", "2025-10-26")
    df = pd.DataFrame({"state": s, "fwd": fwd, "flip": flip}).loc[w]
    df = df[df.state != ""].dropna()
    g = df.groupby("state")
    df["flip"] = df.flip.astype(float)
    return pd.DataFrame({"share of time %": 100 * g.size() / len(df), f"median next {h} %": 100 * g.fwd.median(),
                         f"up after {h} %": 100 * g.fwd.apply(lambda x: (x > 0).mean()),
                         f"DI flips within {h} %": 100 * g.flip.mean()}).round(1)


# -- long + short: stack BTC in bull and bear markets ------------------------------------

def long_short_positions(f, kind, short_rule, long_filter=None, short_filter=None, cover_fast=False, bear_only=False):
    """+1 long / -1 short / 0 cash per candle. Longs as live (bear window: no longs).
    short_rule: None (no shorts), "mirror" (short on the sleeve's mirror entry: DI- crossing above DI+
    for the DI sleeve, ADX through 14 with DI- on top for the ADX sleeve) or "adx" (both sleeves short
    only on the ADX trigger). Shorts cover when DI+ crosses back above DI- (or DI14, if cover_fast)."""
    up, a = f.up.to_numpy(), f.adx.fillna(0).to_numpy()
    lf = long_filter(f).to_numpy() if long_filter else np.ones(len(f), bool)
    sf = short_filter(f).to_numpy() if short_filter else np.ones(len(f), bool)
    up14, bear = f.up14.to_numpy(), f.bear.to_numpy()
    pos = np.zeros(len(f), int)
    lg = sh = False
    for i in range(1, len(f)):
        trig = a[i] > 14 >= a[i - 1]
        if lg and not up[i]:
            lg = False
        elif not lg and lf[i] and up[i] and (trig if kind == "adx" else not up[i - 1]):
            lg = True
        if bear[i]:
            lg = False
        if short_rule:
            if sh and (up14[i] and not up14[i - 1] if cover_fast else up[i]):
                sh = False
            elif not sh and sf[i] and not up[i] and (bear[i] or not bear_only) \
                    and (trig if (kind == "adx" or short_rule == "adx") else up[i - 1]):
                sh = True
            if lg:
                sh = False
        pos[i] = 1 if lg else (-1 if sh else 0)
    return pos


def run_ls(frames, size, **kw):
    curves = {}
    for tf, f in frames.items():
        w = f.loc["2016-01-01":]
        k = f.index.get_indexer(w.index)
        bpy = 365 * 86400 / (w.index[1] - w.index[0]).total_seconds()
        for kind in ("di", "adx"):
            pos = long_short_positions(f, kind, **kw)[k]
            eq, sh = simulate(w, pos, size if kw.get("short_rule") else 0.0)
            curves[(tf, kind)] = (eq, int(np.sum(pos[1:] != pos[:-1])))
    return curves


def stack(eq, btc, a=None, b=None):
    """How many BTC the strategy ends with vs the BTC its starting dollars bought (1.0 = same)."""
    w, p = (eq.loc[a:b], btc.loc[a:b]) if a else (eq, btc)
    return (w.iloc[-1] / w.iloc[0]) / (p.iloc[-1] / p.iloc[0])


def main():
    h1 = hourly()
    frames = {tf: build(resample(h1, rule).loc["2015-06-01":]) for tf, rule in TFS.items()}
    pd.set_option("display.width", 250)

    variants = {"baseline (live)": (None, None)}
    variants.update({f"enter only if {k}": (fn, None) for k, fn in ENTRY_FILTERS.items()})
    variants.update({f"also exit on {k}": (None, fn) for k, fn in EXITS.items()})
    results = {name: run_variant(frames, *v) for name, v in variants.items()}

    def table(results):
        base = results["baseline (live)"]
        rows = {}
        for name, cur in results.items():
            eq = ensemble(cur)
            r = {}
            for a, b, lab in HALVES:
                c, d, cal, _ = stats(eq, a, b)
                r[f"{lab} %/yr"] = 100 * c
                r[f"{lab} worst %"] = 100 * d
                r[f"{lab} ret/DD"] = cal
            r["last 12m %"] = 100 * stats(eq, RECENT)[3]
            better = sum(stats(pair(cur, tf), a, b)[2] > stats(pair(base, tf), a, b)[2] + 1e-9
                         for tf in TFS for a, b, _ in HALVES)
            r["tf-halves better (of 8)"] = better if name != "baseline (live)" else "-"
            r["trades/yr"] = sum(n for _, n in cur.values()) / 9.8
            rows[name] = r
        return pd.DataFrame(rows).T

    print("=== 8-sleeve strategy: baseline vs one change at a time ===")
    t = table(results)
    print(t.round(2).to_string())

    # combine the changes that held up in both halves on most timeframes
    base_cal = [t.loc["baseline (live)", f"{lab} ret/DD"] for _, _, lab in HALVES]
    keep = [n for n in t.index if n != "baseline (live)" and t.loc[n, "tf-halves better (of 8)"] >= 6
            and all(t.loc[n, f"{lab} ret/DD"] > bc for (_, _, lab), bc in zip(HALVES, base_cal))]
    print("\nHeld up in both halves and on 6+ of 8 timeframe-halves:", keep or "none")
    ents = [n for n in keep if n.startswith("enter")]
    exs = [n for n in keep if n.startswith("also")]
    if ents and exs:
        combos = {}
        for en in ents:
            for ex in exs:
                combos[f"{en} + {ex.replace('also ', '')}"] = run_variant(frames, variants[en][0], variants[ex][1])
        combos["baseline (live)"] = results["baseline (live)"]
        print("\n=== Combinations ===")
        print(table(combos).round(2).to_string())

    # ---- long + short, scored on bitcoin stacked ----
    btc = frames["1d"].close.loc["2016-01-01":]
    PERIODS = [("2016-01-01", "2025-10-26", "2016-25"), ("2016-01-01", "2020-12-31", "2016-20"),
               ("2021-01-01", "2025-10-26", "2021-25"), ("2018-01-01", "2018-12-31", "2018 bear"),
               ("2022-01-01", "2022-12-31", "2022 bear"), ("2025-10-06", None, "Oct 25-now bear"),
               ("2020-01-01", "2021-11-09", "2020-21 bull"), ("2023-01-01", "2025-10-05", "2023-25 bull")]
    rsi_up, rsi_dn = (lambda f: f.rsi > 50), (lambda f: f.rsi < 50)
    ls = {"long only (live)": dict(size=0.0, short_rule=None),
          "long only, RSI>50 entries": dict(size=0.0, short_rule=None, long_filter=rsi_up),
          "long + 10% short (mirror)": dict(size=0.10, short_rule="mirror"),
          "long + 25% short (mirror)": dict(size=0.25, short_rule="mirror"),
          "long + 25% short, ADX-14 short trigger": dict(size=0.25, short_rule="adx"),
          "long + 25% short, ADX-14 trigger + RSI<50": dict(size=0.25, short_rule="adx", short_filter=rsi_dn),
          "long + 25% short, ADX-14 trigger, fast cover": dict(size=0.25, short_rule="adx", cover_fast=True),
          "long + 50% short, ADX-14 short trigger": dict(size=0.50, short_rule="adx"),
          "long + 25% short, bear window only": dict(size=0.25, short_rule="mirror", bear_only=True),
          "long + 50% short, bear window only": dict(size=0.50, short_rule="mirror", bear_only=True),
          "long + 100% short, bear window only": dict(size=1.00, short_rule="mirror", bear_only=True),
          "RSI>50 longs + 25% short, ADX-14 trigger": dict(size=0.25, short_rule="adx", long_filter=rsi_up)}
    rows, stacks = {}, {}
    for name, kw in ls.items():
        eq = ensemble(run_ls(frames, **kw))
        c, d, cal, _ = stats(eq, "2016-01-01", "2025-10-26")
        rows[name] = {"2016-25 %/yr": 100 * c, "worst %": 100 * d, "ret/DD": cal,
                      "2016-20 ret/DD": stats(eq, *HALVES[0][:2])[2], "2021-25 ret/DD": stats(eq, *HALVES[1][:2])[2],
                      "last 12m %": 100 * stats(eq, RECENT)[3]}
        stacks[name] = {lab: stack(eq, btc, a, b) for a, b, lab in PERIODS}
    print("\n=== Long + short: dollars ===")
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\n=== Long + short: BTC stack (BTC held at the end per BTC the starting dollars bought; 1.00 = same as holding) ===")
    print(pd.DataFrame(stacks).T.round(2).to_string())

    for tf in ("4h", "1d"):
        print(f"\n=== What ADX states have meant on the {tf} (2016 -> Oct 2025), next 12 candles ===")
        print(states(frames[tf]).to_string())


if __name__ == "__main__":
    main()
