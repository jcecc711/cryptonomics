"""Run the bot.

    python -m bot.run once        evaluate the last closed daily candle and act on it
    python -m bot.run loop        run once now, then every hour (signals change daily at 00:00 UTC)
    python -m bot.run status      print the current state
    python -m bot.run web         start the dashboard

Paper trading is the default. Live trading needs BOT_MODE=live plus API keys.
Create data/STOP to stop all new orders.
"""
import fcntl
import logging
import sys
import time
from decimal import Decimal

from . import config, strategy
from .binanceus import BinanceUS
from .broker import LiveBroker, PaperBroker
from .execution import execute
from .store import Store, now

log = logging.getLogger("bot")


def make(cfg):
    client = BinanceUS(cfg.api_key, cfg.api_secret)
    store = Store(cfg.db_path)
    if cfg.mode == "live":
        broker = LiveBroker(client, cfg.symbol, cfg.base_asset, cfg.quote_asset)
    else:
        broker = PaperBroker(client, cfg.symbol, store, cfg.paper_maker_fee,
                             cfg.paper_taker_fee, cfg.paper_start_usd)
    return client, broker, store


def summary(sig, trade, price):
    di = f"DI+ {sig['di_plus']:.1f} / DI- {sig['di_minus']:.1f}"
    if trade:
        pnl = price / trade["entry_price"] - 1
        return f"{di}. Holding BTC since {trade['entry_time'][:10]} at {trade['entry_price']:,.0f} ({pnl:+.1%})."
    if sig["trend_up"]:
        return f"{di}. In cash; DI+ is above DI- but the bot is waiting for a fresh cross up."
    return f"{di}. In cash, waiting for DI+ to cross above DI-."


def run_once(cfg, client, broker, store):
    candles = client.klines(cfg.symbol, cfg.interval)
    sig = strategy.evaluate(candles, cfg.di_len, cfg.adx_smoothing)
    trade = store.current_trade()
    armed = store.get("armed")
    if armed is None:  # first run
        armed = cfg.join_open_trend or not sig["trend_up"]
    action, armed = strategy.decide(sig, trade is not None, armed)
    store.set("armed", armed)

    note = ""
    if action and cfg.stop_file.exists():
        note, action = f"kill switch ({cfg.stop_file.name}) is on, skipped {action}", None
    elif action:
        filters = client.filters(cfg.symbol)
        quote, base = broker.balances()
        if action == "BUY":
            budget = min(quote * Decimal(str(cfg.allocation)), Decimal(str(cfg.max_order_usd)))
            f = execute("BUY", budget, broker, filters, store, cfg)
            if f.base > 0:
                store.open_trade(f.avg_price, float(f.base - f.fee_base), float(f.quote) + f.fee_usd)
            note = f"bought {f.base} at {f.avg_price:,.2f}" if f.base else "buy did not fill"
        else:
            qty = min(base, Decimal(str(trade["qty"])))
            f = execute("SELL", qty, broker, filters, store, cfg)
            if f.base >= qty * Decimal("0.99"):
                store.close_trade(f.avg_price, float(f.quote) - f.fee_usd)
            elif f.base > 0:
                store.reduce_trade(float(f.base))
            note = f"sold {f.base} at {f.avg_price:,.2f}" if f.base else "sell did not fill"

    store.record_signal(sig, action or ("HOLD" if store.current_trade() else "WAIT"), note)
    bid, ask = broker.book()
    price = float((bid + ask) / 2)
    quote, base = broker.balances()
    store.snapshot(float(quote), float(base), price)
    text = summary(sig, store.current_trade(), price)
    store.set("last_run", {"time": now(), "summary": text, "note": note, "mode": cfg.mode})
    log.info("%s %s", text, note)
    return sig, action, note


def loop(cfg, client, broker, store, every_sec=3600):
    while True:
        try:
            run_once(cfg, client, broker, store)
        except Exception:
            log.exception("run failed; will retry next cycle")
        time.sleep(every_sec - time.time() % every_sec + 300)  # 5 minutes past each hour


def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "once"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load()

    if cmd == "web":
        from .web import serve
        serve(cfg)
        return
    client, broker, store = make(cfg)
    if cmd == "status":
        print(store.get("last_run", {}).get("summary", "No runs yet."))
        print("mode:", cfg.mode, "| balances (quote, base):", broker.balances())
        return

    lock = open(cfg.data_dir / f"bot_{cfg.mode}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another bot process is already running")
    if cmd == "once":
        run_once(cfg, client, broker, store)
    elif cmd == "loop":
        loop(cfg, client, broker, store)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
