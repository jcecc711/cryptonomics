"""Skim 10% of spot earnings into a leveraged-long "derivatives pot".

Spot: the live strategy (DI(38) cross, long-only, 4h + 8h + 12h + 1d sleeves,
equal weight). At each month end, if spot equity is above its high-water mark,
10% of the gain above the mark moves to the pot.

Pot: the same four signals, but each sleeve goes long with leverage L on
isolated margin (futures). Costs per sleeve:
  fee      0.05% of notional per side   (= 0.05% * L of margin)
  holding  10%/yr of notional while long (funding / futures basis)
  liquidation: if a candle's low reaches entry * (1 - 1/L + 0.5%), the sleeve's
           margin is lost; the sleeve restarts in cash and waits for the next
           cross up.
The pot keeps its own gains (no skimming back).

    python bench/leverage_skim.py
"""
import numpy as np
import pandas as pd

from data import fetch_coinbase, resample
from indicators import adx

TFS = {"4h": ("4h", 2190), "8h": ("8h", 1095), "12h": ("12h", 730), "1d": ("1D", 365)}
SPOT_COST = 0.0003          # 0.02% fee + 0.01% slippage per side
FUT_FEE = 0.0005 + 0.0001   # per side, on notional
CARRY = 0.10                # per year, on notional
MAINT = 0.005               # maintenance margin
SKIM = 0.10
PERIODS = [("2016-01-01", "2025-10-26", "2016-25 (all)"), ("2018-01-01", "2018-12-31", "2018 bear"),
           ("2020-01-01", "2021-12-31", "2020-21 bull"), ("2022-01-01", "2022-12-31", "2022 bear"),
           ("2023-01-01", "2025-10-26", "2023-25"), ("2025-10-27", None, "last 12 months")]


def sleeve_daily_returns(df, up, lev, bars_per_year, spot=False):
    """Daily returns of one sleeve and its number of liquidations.

    Signals at a candle's close fill at the next open. Like the live bot, a
    sleeve that starts mid-uptrend (or was just liquidated) waits for DI+ to
    drop below DI- and cross back up before buying.
    """
    o, l, c = df.open.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    n = len(df)
    r = np.zeros(n)
    cost = SPOT_COST if spot else FUT_FEE * lev
    carry_bar = 0.0 if spot else CARRY * lev / bars_per_year
    pos, wait = False, bool(up[0])
    cur = prev = 1.0
    entry_px = margin = liq_px = 0.0
    liqs = 0
    for i in range(n):
        if pos:
            if not spot and l[i] <= liq_px:
                r[i] = -1.0                       # margin lost
                liqs += 1
                pos, wait, cur, prev = False, True, 1.0, 1.0   # restart with a fresh unit
                continue
            margin *= 1 - carry_bar
            cur = margin * max(1 + lev * (c[i] / entry_px - 1), 0.0)
        r[i] = cur / prev - 1
        prev = cur
        if i == n - 1:
            break
        if wait and not up[i]:
            wait = False
        if up[i] and not pos and not wait:
            entry_px, margin, pos = o[i + 1], cur * (1 - cost), True
            liq_px = entry_px * (1 - 1 / lev + MAINT)
        elif not up[i] and pos:
            cur = margin * max(1 + lev * (o[i + 1] / entry_px - 1), 0.0) * (1 - cost)
            pos = False
    daily = (1 + pd.Series(r, df.index)).resample("1D").prod() - 1
    return daily, liqs


def run(h1, since, until, lev, skim=SKIM, cap=None):
    spot_r, pot_r, liqs = [], [], 0
    for tf, (rule, bpy) in TFS.items():
        df = resample(h1, rule)
        d = adx(df, 38, 9)
        up = (d.di_plus > d.di_minus).to_numpy()
        w = df.loc[since:until]
        u = up[df.index.get_indexer(w.index)]
        sr, _ = sleeve_daily_returns(w, u, 1, bpy, spot=True)
        pr, n = sleeve_daily_returns(w, u, lev, bpy)
        spot_r.append(sr)
        pot_r.append(pr)
        liqs += n
    spot_r = pd.concat(spot_r, axis=1).fillna(0).mean(axis=1)
    pot_r = pd.concat(pot_r, axis=1).fillna(0).mean(axis=1)

    S, P, hwm = 10_000.0, 0.0, 10_000.0
    rows = []
    for t in spot_r.index:
        S *= 1 + spot_r[t]
        P *= 1 + pot_r[t]
        if (t + pd.Timedelta(days=1)).month != t.month and S > hwm:
            move = skim * (S - hwm)
            S -= move
            P += move
            hwm = S
        if cap and (t + pd.Timedelta(days=1)).month != t.month and P > cap * (S + P):
            back = P - cap * (S + P)   # pot above its cap goes back to spot
            P -= back
            S += back
            hwm = max(hwm, S)
        rows.append((t, S, P))
    out = pd.DataFrame(rows, columns=["t", "spot", "pot"]).set_index("t")
    out["total"] = out.spot + out.pot
    return out, liqs


def dd(s):
    return 100 * (s / s.cummax() - 1).min()


def main():
    h1 = fetch_coinbase(interval="1h")
    pd.set_option("display.width", 250)
    table = {}
    for since, until, label in PERIODS:
        base, _ = run(h1, since, until, 1, skim=0)
        table.setdefault("No skim (current bot)", {})[label] = f"${base.total.iloc[-1]:,.0f} DD {dd(base.total):.0f}%"
        for lev in (1, 2, 3, 5, 10):
            for cap in (None, 0.2):
                out, liqs = run(h1, since, until, lev, cap=cap)
                end = out.iloc[-1]
                name = f"10% skim -> {lev}x pot" + (", pot capped at 20%" if cap else "")
                table.setdefault(name, {})[label] = (
                    f"${end.total:,.0f} (pot ${end.pot:,.0f}, {liqs} liq) DD {dd(out.total):.0f}%")
        print("done", label, flush=True)
    print("\nStart $10,000. Cells: end total (pot part, pot liquidations) and worst drop of the total")
    print(pd.DataFrame(table).T.to_string())


if __name__ == "__main__":
    main()
