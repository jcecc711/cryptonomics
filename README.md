# cryptonomics

Personal BTC spot trading system for Binance.US.

## The strategy

**DI+ / DI- crossover, DI length 38 (ADX smoothing 9), run on four
timeframes at once: 4h, 8h, 12h and daily.** Each timeframe is a "sleeve" with
25% of the money and its own cash and BTC. A sleeve buys when DI+ crosses above
DI- on its closed candle and sells when it crosses back below. Spot only: each
sleeve is either in BTC or in cash, so the account holds 0–100% BTC depending
on how many timeframes agree.

Tested in `bench/` on BTC 2016 → Oct 2025 (`strategy_scan.py`,
`di_timeframes.py`): out of 8 strategy families on 7 timeframes, the DI cross
on 4h–12h scored best. The four-timeframe ensemble returned about 85% a year
with a 39% worst drawdown, beat the daily-only version in about 80% of
two-year windows, and cut the chance of a drawdown worse than 35% from 42% to
13%. It lags the daily-only version in choppy, falling markets (+1% vs +7.5%
in the twelve months to Oct 2026).

## The bot

    pip install -r requirements.txt
    cp .env.example .env            # paper mode by default
    python -m bot.run once          # evaluate the latest closed candles and act
    python -m bot.run loop          # keep running: 5 minutes past every hour
    python -m bot.run web           # dashboard at http://127.0.0.1:8080
    python -m bot.run status

- **Paper mode** (default) simulates orders against the live Binance.US order
  book, starting with `BOT_PAPER_START_USD` split equally between the sleeves.
- **Timeframes:** `BOT_TIMEFRAMES` (default `4h,8h,12h,1d`).
- **Sleeve funding:** on its first live run each sleeve gets its equal share of
  the account's free quote balance; from then on it only spends its own cash
  and only sells BTC it bought itself.
- **Orders** are maker-only limit orders at the best bid/ask (0% fee, no
  spread paid), re-posted when the market moves away. After
  `BOT_MAKER_TIMEOUT_MIN` (default 10) the remainder crosses the spread (0.02%
  fee, capped at `BOT_MAX_SLIPPAGE`) unless `BOT_TAKER_FALLBACK=false`.
- **Starting mid-trend:** with `BOT_JOIN_OPEN_TREND=false` (default) the bot
  won't buy into a trend that is already running; it waits for the next fresh
  cross up.
- **Kill switch:** `touch data/STOP` blocks all new orders; delete it to resume.
- Everything is recorded in `data/bot_paper.sqlite` or `data/bot_live.sqlite` (tagged by sleeve)
  and shown on the dashboard. Paper and live never share balances or trades.

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

The bot only manages BTC it bought itself: a sell never exceeds what that
sleeve holds.

### Tests

    python -m unittest discover -s tests -t . -v

## Rule test bench

    python bench/run.py --since 2021-01-01            # 1-hour candles
    python bench/run.py --interval 1d --since 2021-01-01
    python bench/run.py --refresh                     # re-download candles
    python bench/exits.py                             # exit comparison
    python bench/exit_study.py                        # exit probability study
    python bench/di_sweep.py                          # DI length sweep + walk-forward
    python bench/short_term.py                        # 3m-1h crossovers under three cost models
    python bench/dip_buy.py                           # short-timeframe dip buying
    python bench/strategy_scan.py                     # 8 strategy families x 7 timeframes
    python bench/di_timeframes.py                     # DI cross across timeframes + ensemble

- `bench/data.py` downloads and caches Binance.US and Coinbase candles in `data/`.
- `bench/indicators.py` has EMA, SMA and ADX/DI matching TradingView.
- `bench/rules.py`, `bench/mtf.py` and `bench/exits.py` hold the rule sets tested.
- `bench/backtest.py` acts on each closed candle and fills at the next open,
  charging 0.02% fee + 0.02% slippage per side.
