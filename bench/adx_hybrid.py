"""Joe's ADX-14 + DI rules, and midpoints between the DI cross and the ADX trigger.

Rules simulated, long-only spot ("sell" = go to cash), DI 38 / ADX smoothing 9:

  DI cross (current)      buy DI+ cross up, sell DI+ cross down
  ADX trigger             buy when ADX rises through 14 with DI+ on top, sell on DI cross down
  Joe's ADX-14 rules      on any ADX cross of 14, take the side DI points to; a DI cross during
                          the trade means the trend broke: flip at the next ADX-14 cross; with no
                          DI cross, close when ADX crosses 14 the opposite way to the entry cross
  DI in, ADX out          buy DI cross up; sell on DI cross down, or once ADX has risen above 14,
                          when it falls back through 14
  ADX in, DI or ADX out   buy ADX up through 14 with DI+ on top (or DI cross up while ADX < 14);
                          sell on DI cross down or ADX falling back through 14
  50/50 blend             half the money in DI cross, half in ADX trigger

    python bench/adx_hybrid.py
"""
import numpy as np
import pandas as pd

import backtest
from adx14 import FEE, RECENT, SLIP, START, END, frames

LEVEL = 14
ENSEMBLE = ("4h", "8h", "12h", "1d")


def to_signals(df, target):
    t = pd.Series(target, df.index).astype(bool)
    return t & ~t.shift(fill_value=False), ~t & t.shift(fill_value=False)


def joe(df, level=LEVEL):
    p, m, a = df.di_plus.to_numpy(), df.di_minus.to_numpy(), df.adx.to_numpy()
    n, side, entry_dir, crossed = len(df), 0, 0, False
    target = np.zeros(n, dtype=bool)
    for i in range(1, n):
        up = p[i] > m[i]
        di_x = up != (p[i - 1] > m[i - 1])
        adx_dir = 1 if a[i] > level >= a[i - 1] else -1 if a[i] < level <= a[i - 1] else 0
        if side == 0:
            if adx_dir:
                side, entry_dir, crossed = (1 if up else -1), adx_dir, False
        else:
            if di_x:
                crossed = True
            if adx_dir:
                if crossed:                       # trend broke: flip to where DI points now
                    side, entry_dir, crossed = (1 if up else -1), adx_dir, False
                elif adx_dir != entry_dir:        # no DI cross: close on the opposite ADX cross
                    side = 0
        target[i] = side == 1
    return to_signals(df, target)


def di_in_adx_out(df, level=LEVEL):
    p, m, a = df.di_plus.to_numpy(), df.di_minus.to_numpy(), df.adx.to_numpy()
    n, long_, armed = len(df), False, False
    target = np.zeros(n, dtype=bool)
    for i in range(1, n):
        up, was_up = p[i] > m[i], p[i - 1] > m[i - 1]
        if not long_ and up and not was_up:
            long_, armed = True, a[i] > level
        elif long_:
            armed = armed or a[i] > level
            if not up or (armed and a[i] < level <= a[i - 1]):
                long_ = False
        target[i] = long_
    return to_signals(df, target)


def adx_in_either_out(df, level=LEVEL):
    p, m, a = df.di_plus.to_numpy(), df.di_minus.to_numpy(), df.adx.to_numpy()
    n, long_ = len(df), False
    target = np.zeros(n, dtype=bool)
    for i in range(1, n):
        up, was_up = p[i] > m[i], p[i - 1] > m[i - 1]
        adx_up = a[i] > level >= a[i - 1]
        adx_dn = a[i] < level <= a[i - 1]
        if not long_ and up and (adx_up or (not was_up and a[i] < level)):
            long_ = True
        elif long_ and (not up or adx_dn):
            long_ = False
        target[i] = long_
    return to_signals(df, target)


def di_cross(df):
    up = df.di_plus > df.di_minus
    return up & ~up.shift(fill_value=False), ~up & up.shift(fill_value=False)


def adx_trigger(df, level=LEVEL):
    up = df.di_plus > df.di_minus
    e = (df.adx > level) & (df.adx.shift() <= level) & up
    return e, ~up & up.shift(fill_value=False)


RULES = {"DI cross (current)": di_cross, "ADX trigger": adx_trigger, "Joe's ADX-14 rules": joe,
         "DI in, ADX out": di_in_adx_out, "ADX in, DI or ADX out": adx_in_either_out}


def curve(df, rule, since, until=None):
    e, x = rule(df)
    w = df.loc[since:until]
    t, eq = backtest.run(w, e.loc[w.index], x.loc[w.index], fee=FEE, slippage=SLIP)
    return eq.resample("1D").last().ffill(), t


def stats(eq):
    eq = eq / eq.iloc[0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return 100 * cagr, 100 * dd, cagr / -dd, 100 * (eq.iloc[-1] - 1)


def blend(curves):
    df = pd.DataFrame(curves).ffill()
    return (1 + df.pct_change().fillna(0).mean(axis=1)).cumprod()


def main():
    pd.set_option("display.width", 250)
    F = frames()
    per_tf, ens, ens_recent = {}, {}, {}
    for name, rule in RULES.items():
        cs, rs = {}, {}
        for tf in ("3h", "4h", "6h", "8h", "12h", "1d"):
            eq, t = curve(F[tf], rule, START, END)
            rec, _ = curve(F[tf], rule, RECENT)
            c, d, cal, _ = stats(eq)
            per_tf[(name, tf)] = {"CAGR %": c, "max DD %": d, "ret/DD": cal, "trades/yr": len(t) / 9.8,
                                  "win %": 100 * (t.ret > 0).mean() if len(t) else np.nan,
                                  "last 12m %": stats(rec)[3]}
            cs[tf], rs[tf] = eq, rec
        ens[name] = blend({k: cs[k] for k in ENSEMBLE})
        ens_recent[name] = blend({k: rs[k] for k in ENSEMBLE})
    ens["50/50 blend: DI cross + ADX trigger"] = blend({"a": ens["DI cross (current)"], "b": ens["ADX trigger"]})
    ens_recent["50/50 blend: DI cross + ADX trigger"] = blend({"a": ens_recent["DI cross (current)"], "b": ens_recent["ADX trigger"]})

    print("=== Each rule by timeframe, 2016 -> Oct 2025 (and the 12 months after) ===")
    print(pd.DataFrame(per_tf).T.round(2).to_string())

    print("\n=== Four-timeframe ensembles (4h + 8h + 12h + 1d) ===")
    base, n = ens["DI cross (current)"], 730
    starts = range(0, len(base) - n, 14)
    rows = {}
    for name, eq in ens.items():
        c, d, cal, _ = stats(eq)
        win = [stats(eq.iloc[s:s + n]) for s in starts]
        ref = [stats(base.iloc[s:s + n]) for s in starts]
        sub = {lab: stats(eq.loc[a:b])[3] for a, b, lab in [("2018-01-01", "2018-12-31", "2018 %"),
                                                            ("2022-01-01", "2022-12-31", "2022 %"),
                                                            ("2023-01-01", "2025-10-26", "2023-25 %")]}
        rows[name] = {"CAGR %": c, "max DD %": d, "ret/DD": cal, **sub, "last 12m %": stats(ens_recent[name])[3],
                      "beats current ret/DD (2y windows) %": 100 * np.mean([a[2] > b[2] for a, b in zip(win, ref)]),
                      "DD worse than -35% (2y windows) %": 100 * np.mean([a[1] < -35 for a in win])}
    print(pd.DataFrame(rows).T.round(1).to_string())


if __name__ == "__main__":
    main()
