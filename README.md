# cryptonomics

Personal BTC spot trading system for Binance.US.

## Rule test bench

    python bench/run.py --since 2021-01-01            # 1-hour candles
    python bench/run.py --interval 1d --since 2021-01-01
    python bench/run.py --refresh                     # re-download candles

- `bench/data.py` downloads and caches Binance.US candles in `data/`.
- `bench/indicators.py` has EMA, SMA and ADX/DI matching TradingView.
- `bench/rules.py` holds the candidate buy/sell rule sets.
- `bench/backtest.py` acts on each closed candle and fills at the next open,
  charging 0.02% fee + 0.02% slippage per side.
