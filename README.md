# cryptonomics

Personal BTC spot trading system for Binance.US.

## The strategy

Daily DI+ / DI- crossover, DI length 38 (ADX smoothing 9):

- **Buy** when DI+ crosses above DI- on a closed daily candle.
- **Sell** when DI+ crosses back below DI-.
- Spot only: the bot is either in BTC or in cash.

Tested in `bench/` (2017 → Oct 2025, then a held-back final year): every DI
length from 7 to 75 was profitable, and 38 was among the best for return
versus drawdown. Expect drawdowns of roughly 35–40%.

## The bot

    pip install -r requirements.txt
    cp .env.example .env            # paper mode by default
    python -m bot.run once          # evaluate the last closed daily candle and act
    python -m bot.run loop          # keep running: checks hourly, signals change at 00:00 UTC
    python -m bot.run web           # dashboard at http://127.0.0.1:8080
    python -m bot.run status

- **Paper mode** (default) simulates orders against the live Binance.US order
  book, starting with `BOT_PAPER_START_USD`.
- **Orders** are maker-only limit orders at the best bid/ask (0% fee, no
  spread paid), re-posted when the market moves away. After
  `BOT_MAKER_TIMEOUT_MIN` the remainder crosses the spread (0.02% fee, capped
  at `BOT_MAX_SLIPPAGE`) unless `BOT_TAKER_FALLBACK=false`.
- **Starting mid-trend:** with `BOT_JOIN_OPEN_TREND=false` (default) the bot
  won't buy into a trend that is already running; it waits for the next fresh
  cross up.
- **Kill switch:** `touch data/STOP` blocks all new orders; delete it to resume.
- Everything is recorded in `data/bot_<mode>.sqlite` and shown on the dashboard.

### Going live

1. Create a Binance.US API key with **trading enabled, withdrawals disabled,
   and restricted to your server's IP address**.
2. Fund the account with the quote asset (USDT for `BTCUSDT`; or set
   `BOT_SYMBOL=BTCUSD` and `BOT_QUOTE=USD` to trade dollars directly).
3. In `.env`: `BOT_MODE=live`, the two keys, and a small `BOT_MAX_ORDER_USD`
   for the first trades.
4. Run on an always-on machine (a small VPS works). `deploy/` has systemd
   units for the loop and the dashboard. Set `DASHBOARD_PASSWORD` before
   exposing the dashboard beyond localhost, and put it behind HTTPS.

The bot only manages BTC it bought itself: a sell never exceeds the quantity
recorded for the open trade.

### Tests

    python -m unittest discover -s tests -t . -v

## Rule test bench

    python bench/run.py --since 2021-01-01            # 1-hour candles
    python bench/run.py --interval 1d --since 2021-01-01
    python bench/run.py --refresh                     # re-download candles
    python bench/exits.py                             # exit comparison
    python bench/exit_study.py                        # exit probability study
    python bench/di_sweep.py                          # DI length sweep + walk-forward

- `bench/data.py` downloads and caches Binance.US and Coinbase candles in `data/`.
- `bench/indicators.py` has EMA, SMA and ADX/DI matching TradingView.
- `bench/rules.py`, `bench/mtf.py` and `bench/exits.py` hold the rule sets tested.
- `bench/backtest.py` acts on each closed candle and fills at the next open,
  charging 0.02% fee + 0.02% slippage per side.
