"""Trailing stops on longs and shorts: cap each trade's loss near 3-4% and keep the trend.

Base strategy: the live 8 sleeves (DI cross and ADX-14 trigger on 4h / 8h / 12h / 1d,
equal shares), longs blocked in the halving bear window, optionally a 50% short
inside the bear window (opened on the sleeve's mirror signal, covered on DI+ crossing
back above DI-).

Fixed stop X%: the same, measured from the entry price only (the stop never moves).
Trailing stop X%: a long is sold if price trades X% below the highest price since
entry (a short is covered X% above the lowest). The stop fires inside the candle at
the stop price, like a resting stop order (or at the open if price gaps through it),
plus 0.05% slippage. The usual DI-cross exit still applies.

After a stop-out:
  wait     no new trade until the sleeve's next fresh entry signal
  resume   back in if price breaks past the extreme it made before the stop (a new high
           for a long, a new low for a short) while DI still points that way
Faster short cover: cover when the faster DI(14) turns up instead of DI(38).

Costs: longs 0.02% fee + 0.01% slippage per side; shorts 0.05% + 0.01% plus 10%/yr.

    python bench/trail_stops.py
"""
import numpy as np
import pandas as pd

from adx_plus import build, ensemble, stack, stats, HALVES, RECENT, TFS
from data import resample
from short_hedge import hourly

LONG_COST, SHORT_COST, STOP_SLIP, CARRY = 0.0003, 0.0006, 0.0005, 0.10


def simulate(f, kind, trail=None, reentry="wait", short_size=0.0, fast_cover=False, fixed=False):
    """Daily equity and the list of trade returns for one sleeve."""
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    up, up14, bear = f.up.to_numpy(), f.up14.to_numpy(), f.bear.to_numpy()
    a = f.adx.fillna(0).to_numpy()
    n = len(f)
    carry = CARRY * (f.index[1] - f.index[0]).total_seconds() / (365 * 86400)
    eq = np.ones(n)
    cash, side, entry, base, ext, size = 1.0, 0, 0.0, 1.0, 0.0, 0.0
    armed_long = armed_short = False      # "resume": waiting to re-enter past the old extreme
    resume_lvl = 0.0
    trades, stopped = [], 0
    pend = None                            # ("long"|"short"|"exit", price) decided at a close, filled next open

    def close_trade(px, cost):
        nonlocal cash, side
        if side == 1:
            new = base * px / entry * (1 - cost)
        else:
            new = cash + base * size * (1 - px / entry) - base * size * cost
        trades.append(new / cash_at_open[0] - 1)
        cash, side = new, 0

    cash_at_open = [1.0]
    for i in range(1, n):
        # 1) fills at this candle's open from the previous close's decision
        if pend:
            act = pend
            pend = None
            if act == "exit" and side:
                close_trade(o[i], LONG_COST if side == 1 else SHORT_COST)
            elif act in ("long", "short"):
                if side:
                    close_trade(o[i], LONG_COST if side == 1 else SHORT_COST)
                cash_at_open[0] = cash
                if act == "long":
                    side, entry, size = 1, o[i], 1.0
                    base = cash * (1 - LONG_COST)
                else:
                    side, entry, size = -1, o[i], short_size
                    cash -= cash * size * SHORT_COST
                    base = cash
                ext = o[i]
        # 2) the trailing stop, inside this candle
        if side and trail:
            if side == 1:
                stop = ext * (1 - trail)
                if l[i] <= stop:
                    px = min(o[i], stop) * (1 - STOP_SLIP)
                    resume_lvl, armed_long = ext, True
                    close_trade(px, LONG_COST)
                    stopped += 1
                elif not fixed:
                    ext = max(ext, h[i])
            else:
                stop = ext * (1 + trail)
                if h[i] >= stop:
                    px = max(o[i], stop) * (1 + STOP_SLIP)
                    resume_lvl, armed_short = ext, True
                    close_trade(px, SHORT_COST)
                    stopped += 1
                elif not fixed:
                    ext = min(ext, l[i])
        if side == -1:
            cash -= base * size * carry
        # 3) signals on this candle's close -> next open
        trig = a[i] > 14 >= a[i - 1]
        long_sig = up[i] and (trig if kind == "adx" else not up[i - 1]) and not bear[i]
        short_sig = short_size > 0 and bear[i] and not up[i] and (trig if kind == "adx" else up[i - 1])
        if reentry == "resume":
            if armed_long and up[i] and not bear[i] and c[i] > resume_lvl:
                long_sig = True
            if armed_short and not up[i] and bear[i] and short_size > 0 and c[i] < resume_lvl:
                short_sig = True
        if not up[i]:
            armed_long = False
        if up[i]:
            armed_short = False
        if side == 1 and (not up[i] or bear[i]):
            pend = "short" if short_sig else "exit"
        elif side == -1 and ((up14[i] and not up14[i - 1]) if fast_cover else up[i]):
            pend = "long" if long_sig else "exit"
        elif side == 0 and long_sig:
            pend, armed_long = "long", False
        elif side == 0 and short_sig:
            pend, armed_short = "short", False
        eq[i] = (base * c[i] / entry if side == 1 else
                 cash + base * size * (1 - c[i] / entry) if side == -1 else cash)
    eq[0] = 1.0
    return pd.Series(eq, f.index).resample("1D").last().ffill(), trades, stopped


def run(frames, **kw):
    curves, trades, stopped = {}, [], 0
    for tf, f in frames.items():
        w = f.loc["2016-01-01":]
        for kind in ("di", "adx"):
            eq, t, s = simulate(w, kind, **kw)
            curves[(tf, kind)] = (eq, len(t))
            trades += t
            stopped += s
    return curves, np.array(trades), stopped


def main():
    h1 = hourly()
    frames = {tf: build(resample(h1, rule).loc["2015-06-01":]) for tf, rule in TFS.items()}
    btc = frames["1d"].close.loc["2016-01-01":]
    V = {"long only (live)": {}}
    for x in (0.03, 0.04, 0.06, 0.10):
        V[f"long, {x:.0%} trail, wait"] = dict(trail=x)
    for x in (0.04, 0.06):
        V[f"long, {x:.0%} trail, resume"] = dict(trail=x, reentry="resume")
    for x in (0.04, 0.08):
        V[f"long, {x:.0%} fixed stop from entry"] = dict(trail=x, fixed=True)
    V["long + 50% bear short"] = dict(short_size=0.5)
    V["long + 50% bear short, 4% fixed stop"] = dict(short_size=0.5, trail=0.04, fixed=True)
    V["long + 50% bear short, fast cover"] = dict(short_size=0.5, fast_cover=True)
    for x in (0.04, 0.06, 0.10):
        V[f"long + 50% bear short, {x:.0%} trail, resume"] = dict(short_size=0.5, trail=x, reentry="resume")
    P = [("2016-01-01", "2025-10-26", "2016-25"), ("2018-01-01", "2018-12-31", "2018 bear"),
         ("2022-01-01", "2022-12-31", "2022 bear"), ("2020-01-01", "2021-11-09", "2020-21 bull"),
         ("2023-01-01", "2025-10-05", "2023-25 bull")]
    rows, stacks = {}, {}
    for name, kw in V.items():
        cur, tr, stopped = run(frames, **kw)
        eq = ensemble(cur)
        c, d, cal, _ = stats(eq, "2016-01-01", "2025-10-26")
        losses = tr[tr < 0]
        rows[name] = {"%/yr": 100 * c, "worst %": 100 * d, "ret/DD": cal,
                      "2016-20 ret/DD": stats(eq, *HALVES[0][:2])[2], "2021-25 ret/DD": stats(eq, *HALVES[1][:2])[2],
                      "last 12m %": 100 * stats(eq, RECENT)[3], "trades/yr": len(tr) / 10.9,
                      "stopped %": 100 * stopped / max(len(tr), 1), "win %": 100 * (tr > 0).mean(),
                      "avg loss %": 100 * losses.mean() if len(losses) else 0,
                      "worst trade %": 100 * tr.min()}
        stacks[name] = {lab: stack(eq, btc, a, b) for a, b, lab in P}
    pd.set_option("display.width", 260)
    print("=== Dollars and trades (sleeve-level trade returns) ===")
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\n=== BTC stack (BTC at the end per BTC the starting dollars bought; 1.00 = same as holding) ===")
    print(pd.DataFrame(stacks).T.round(2).to_string())


if __name__ == "__main__":
    main()
