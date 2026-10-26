"""How likely is each exit to be the best one? Two tests:

1. Rolling windows: every 1-year and 2-year window of real history
   (2017 -> design end), started every two weeks.
2. Block bootstrap: synthetic price paths built from shuffled 60-day chunks
   of real daily candles, so trends survive but their order is random.

    python bench/exit_study.py [n_paths]
"""
import sys

import numpy as np
import pandas as pd

import backtest
from data import fetch_coinbase
from exits import signals, weekly_score
from mtf import DESIGN_END

CANDIDATES = {
    "DI cross": ("di", None, None),
    "21/50 + 10% stop": ("ma", 0.10, None),
    "Switch (weekly=3)": ("switch", 0.10, 3),
    "Switch (weekly>=2)": ("switch", 0.10, 2),
}


def equity_curves(d):
    ws = weekly_score(d)
    out = {}
    for name, (mode, trail, wmin) in CANDIDATES.items():
        strong = ws >= wmin if wmin else None
        e, x = signals(d, mode, trail=trail, strong=strong)
        out[name] = backtest.run(d, e, x)[1]
    out["Buy & hold"] = d["close"] / d["close"].iloc[0]
    return pd.DataFrame(out)


def window_metrics(eq):
    """eq: DataFrame of equity curves over one window."""
    years = len(eq) / 365.25
    ret = eq.iloc[-1] / eq.iloc[0]
    cagr = ret ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return pd.DataFrame({"cagr": cagr, "dd": dd, "calmar": cagr / -dd.where(dd < 0, -1e-9)})


def summarize(results):
    """results: list of window_metrics frames -> probability table."""
    names = results[0].index
    cagr = pd.DataFrame([r["cagr"] for r in results])
    dd = pd.DataFrame([r["dd"] for r in results])
    cal = pd.DataFrame([r["calmar"] for r in results])
    strat = [n for n in names if n != "Buy & hold"]
    best_ret = cagr[strat].idxmax(axis=1).value_counts(normalize=True)
    best_dd = dd[strat].idxmax(axis=1).value_counts(normalize=True)
    best_cal = cal[strat].idxmax(axis=1).value_counts(normalize=True)
    t = pd.DataFrame({
        "P(best return)": best_ret,
        "P(smallest drop)": best_dd,
        "P(best return/drop)": best_cal,
        "P(beats B&H return)": (cagr[strat].gt(cagr["Buy & hold"], axis=0)).mean(),
        "P(drop worse than -35%)": (dd < -0.35).mean(),
        "median yearly ret": cagr.median(),
        "bad case ret (10th pct)": cagr.quantile(0.10),
        "median worst drop": dd.median(),
    }).reindex(names).fillna(0)
    return (t * 100).round(0).astype(int)


def rolling(d, years):
    eq = equity_curves(d.loc["2016-06-01":DESIGN_END]).loc["2017-01-01":]
    n = int(years * 365)
    starts = range(0, len(eq) - n, 14)
    return summarize([window_metrics(eq.iloc[s:s + n]) for s in starts]), len(starts)


def bootstrap(d, n_paths=1000, block=60, warmup=420, years=2, seed=7):
    rng = np.random.default_rng(seed)
    prev = d["close"].shift()
    rel = pd.DataFrame({k: d[k] / prev for k in ("open", "high", "low", "close")}).dropna().to_numpy()
    length = warmup + int(years * 365)
    idx = pd.date_range("2001-01-01", periods=length, freq="D")
    results = []
    for _ in range(n_paths):
        starts = rng.integers(0, len(rel) - block, length // block + 1)
        r = np.concatenate([rel[s:s + block] for s in starts])[:length]
        close = 100 * np.cumprod(r[:, 3])
        prev_close = np.concatenate([[100], close[:-1]])
        path = pd.DataFrame({"open": r[:, 0] * prev_close, "high": r[:, 1] * prev_close,
                             "low": r[:, 2] * prev_close, "close": close, "volume": 0.0}, index=idx)
        eq = equity_curves(path).iloc[warmup:]
        results.append(window_metrics(eq))
    return summarize(results)


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    d = fetch_coinbase(interval="1d")
    pd.set_option("display.width", 250)
    for years in (1, 2):
        table, n = rolling(d, years)
        print(f"Real history, rolling {years}-year windows ({n} windows, 2017 -> {DESIGN_END}), all figures %")
        print(table.to_string(), "\n")
    print(f"Block bootstrap, {n_paths} synthetic 2-year paths (60-day chunks of 2015-2025 BTC), all figures %")
    print(bootstrap(d, n_paths).to_string())
