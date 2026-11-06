# cryptonomics

Personal BTC spot trading system for Binance.US.

## The strategy

Two rules, each on four timeframes (4h, 8h, 12h, daily), DI length 38 and ADX
smoothing 9. Every rule-and-timeframe pair is a "sleeve" with an equal share of
the money (8 x 12.5%) and its own cash and BTC:

- **DI crossover:** buy when DI+ crosses above DI-, sell when it crosses back below.
- **ADX-14 trigger:** buy when ADX rises through 14 with DI+ on top, sell when
  DI+ crosses below DI-.

Spot only: each sleeve is either in BTC or in cash.

**Bear filter (halving cycle):** days 540-910 after each halving have held the
post-top bear market in every cycle so far (2018, 2022, 2025-26). In that window
the bot takes no new longs and exits open ones (`BOT_BEAR_FILTER=halving`; next
window about Oct 2029 - Oct 2030). Tested in `bench/regime_study.py`: about 99%
a year with a 30% worst drawdown versus 91% / 39% without it. The dashboard's
Market Regime panel shows the cycle position, the bull/bear call and price
against the 200-day, 50-week and 20/21-week averages.

Tested on BTC 2016 -> Oct 2025 (`bench/strategy_scan.py`, `di_timeframes.py`,
`adx14.py`, `adx_hybrid.py`). The DI crossover alone returned about 92% a year
with a 39% worst drawdown; the 50/50 blend with the ADX trigger about 77% a year
with a 32% worst drawdown: the same return for the risk, it never fell more than
35% within any two-year window, and it held up better in choppy and falling
markets. It gives up growth in strong bull runs.

## The bot

    pip install -r requirements.txt
    cp .env.example .env            # paper mode by default
    python -m bot.run once          # evaluate the latest closed candles and act
    python -m bot.run loop          # keep running: 5 minutes past every hour
    python -m bot.run web           # dashboard app at http://127.0.0.1:8080
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
- **Dashboard app:** overview (account vs. holding BTC, sleeves with DI gauges and
  candle-close countdowns, performance), charts (candles / Heikin Ashi, zoom and
  scroll back through history, resizable and full-screen panes, your MA/EMA set,
  volume, Bollinger, DI/ADX, RSI, MACD, Stochastic, ATR, horizontal / trend /
  Fibonacci / measure tools; click a drawing to move it, drag its end points,
  change color / thickness / style, extend it, lock it or delete it), a watchlist
  of Binance.US coins and US stocks scored with the bot's DI rule, trades with
  CSV export, activity, and trade alerts.
- **US stocks** (watchlist and charts only; the bot trades BTC): type a ticker
  such as `AAPL` in the watchlist (`$COIN` when a ticker is also a coin). Prices
  come from Yahoo Finance's public chart data, no key needed, regular hours only
  and a few minutes delayed. The DI read uses 1h / 4h / 1d / 1w; 3h-12h stock
  candles are built within each trading day (4h = 9:30-13:30 and 13:30-16:00 New
  York), and intraday history goes back about 60 days (15m) or 2 years (1h).
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
    python bench/ema_cross.py                         # EMA 8/13 cross, long / short, 15m-1d

- `bench/data.py` downloads and caches Binance.US and Coinbase candles in `data/`.
- `bench/indicators.py` has EMA, SMA and ADX/DI matching TradingView.
- `bench/rules.py`, `bench/mtf.py` and `bench/exits.py` hold the rule sets tested.
- `bench/backtest.py` acts on each closed candle and fills at the next open,
  charging 0.02% fee + 0.02% slippage per side.
