"""Run every rule set and print a comparison table.

    python bench/run.py [--interval 1h] [--since 2021-01-01] [--refresh]
"""
import argparse

import pandas as pd

import backtest
from data import fetch
from rules import RULES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()

    df = fetch(a.symbol, a.interval, a.refresh)
    # Indicators are computed on full history, then the test window is sliced.
    rows = {}
    for name, rule in RULES.items():
        entry, exit_ = rule(df)
        window = df.loc[a.since:a.until]
        trades, eq = backtest.run(window, entry.loc[window.index], exit_.loc[window.index])
        rows[name] = backtest.stats(trades, eq, window)
    rows["Buy & hold"] = backtest.buy_hold(window)

    pd.set_option("display.width", 200)
    print(f"{a.symbol} {a.interval}  {window.index[0]:%Y-%m-%d} -> {window.index[-1]:%Y-%m-%d}")
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
