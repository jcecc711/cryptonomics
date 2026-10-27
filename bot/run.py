"""Run the bot.

    python -m bot.run once        evaluate the latest closed candles and act on them
    python -m bot.run loop        run once now, then 5 minutes past every hour
    python -m bot.run status      print the current state
    python -m bot.run web         start the dashboard

The ensemble: the DI(38) cross runs on each timeframe in BOT_TIMEFRAMES
(default 4h, 8h, 12h, 1d), each as a sleeve with an equal share of the money
and its own cash and BTC. Paper is the default.
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


class Sleeve:
    """The DI cross on one timeframe, trading its own share of the money."""

    def __init__(self, name, share, mode, broker, store, cfg):
        self.name, self.share, self.mode = name, share, mode   # name is the timeframe, e.g. "4h"
        self.broker, self.store, self.cfg = broker, store, cfg

    # Sleeve balances: what this sleeve owns, separate from the account total.
    @property
    def cash(self):
        return Decimal(self.store.get(f"{self.name}.cash", "0"))

    @property
    def btc(self):
        return Decimal(self.store.get(f"{self.name}.btc", "0"))

    def set_balances(self, cash, btc):
        self.store.set(f"{self.name}.cash", str(cash))
        self.store.set(f"{self.name}.btc", str(btc))

    @property
    def funded(self):
        return self.store.get(f"{self.name}.cash") is not None

    def decide(self, sig, holding):
        key = f"{self.name}.state"
        armed = self.store.get(key)
        if armed is None:  # first run
            armed = self.cfg.join_open_trend or not sig["trend_up"]
        action, armed = strategy.decide(sig, holding, armed)
        self.store.set(key, armed)
        return action

    def step(self, client, filters_cache):
        candles = client.klines(self.cfg.symbol, self.name)
        sig = strategy.evaluate(candles, self.cfg.di_len, self.cfg.adx_smoothing)
        trade = self.store.current_trade(self.name)
        action = self.decide(sig, trade is not None)
        note = ""
        if action and self.cfg.stop_file.exists():
            note, action = f"kill switch ({self.cfg.stop_file.name}) is on, skipped {action}", None
        elif action:
            if "filters" not in filters_cache:
                filters_cache["filters"] = client.filters(self.cfg.symbol)
            note = self.trade(action, trade, filters_cache["filters"])
        self.store.record_signal(self.name, sig, action or ("HOLD" if self.store.current_trade(self.name) else "WAIT"),
                                 note)
        return sig, action, note

    def trade(self, action, trade, filters):
        quote, base = self.broker.balances()
        if action == "BUY":
            budget = min(self.cash, quote, Decimal(str(self.cfg.max_order_usd)))
            f = execute("BUY", budget, self.broker, filters, self.store, self.cfg, self.name)
            if f.base > 0:
                got = f.base - f.fee_base
                self.set_balances(self.cash - f.quote - f.fee_quote, self.btc + got)
                self.store.open_trade(self.name, f.avg_price, float(got), float(f.quote) + f.fee_usd)
            return f"bought {f.base} at {f.avg_price:,.2f}" if f.base else "buy did not fill"
        qty = min(self.btc, base)
        f = execute("SELL", qty, self.broker, filters, self.store, self.cfg, self.name)
        if f.base > 0:
            self.set_balances(self.cash + f.quote - f.fee_quote, self.btc - f.base)
            if f.base >= Decimal(str(trade["qty"])) * Decimal("0.99"):
                self.store.close_trade(self.name, f.avg_price, float(f.quote) - f.fee_usd)
            else:
                self.store.reduce_trade(self.name, float(f.base))
        return f"sold {f.base} at {f.avg_price:,.2f}" if f.base else "sell did not fill"


def make(cfg, client=None, store=None):
    client = client or BinanceUS(cfg.api_key, cfg.api_secret)
    store = store or Store(cfg.db_path)
    share = 1 / len(cfg.timeframes)
    live = LiveBroker(client, cfg.symbol, cfg.base_asset, cfg.quote_asset) if cfg.mode == "live" else None
    sleeves = []
    for tf in cfg.timeframes:
        broker = live or PaperBroker(client, cfg.symbol, store, cfg.paper_maker_fee, cfg.paper_taker_fee,
                                     cfg.paper_start_usd * share, key=f"paper.{tf}")
        sleeves.append(Sleeve(tf, share, cfg.mode, broker, store, cfg))
    return client, sleeves, store


def fund(sleeves):
    """Give new sleeves their starting cash. Live sleeves split one reading of
    the account's free quote balance, so funding order doesn't matter."""
    new_live = [s for s in sleeves if not s.funded and s.mode == "live"]
    if new_live:
        quote, _ = new_live[0].broker.balances()
        for s in new_live:
            s.set_balances(quote * Decimal(str(s.share)), Decimal(0))
            log.info("%s sleeve funded with %s (%.0f%% of free balance)", s.name, s.cash, 100 * s.share)
    for s in sleeves:
        if not s.funded:  # paper: its own simulated balance
            s.set_balances(s.broker.balances()[0], Decimal(0))


def run_once(cfg, client, sleeves, store):
    for broker in {id(s.broker): s.broker for s in sleeves}.values():
        for o in broker.cancel_leftovers():
            log.warning("cancelled leftover order %s (%s, filled %s)", o["orderId"], o["status"], o["executedQty"])
    fund(sleeves)
    cache, results, ts = {}, {}, now()
    for s in sleeves:
        try:
            results[s.name] = s.step(client, cache)
        except Exception as e:  # one sleeve failing shouldn't stop the others
            log.exception("%s sleeve failed", s.name)
            results[s.name] = (None, None, f"error: {e}")

    bid, ask = client.book(cfg.symbol)
    price = float((bid + ask) / 2)
    total_cash = total_btc = 0.0
    report = {}
    for s in sleeves:
        cash, btc = float(s.cash), float(s.btc)
        total_cash, total_btc = total_cash + cash, total_btc + btc
        store.snapshot(ts, s.name, cash, btc, price)
        sig, action, note = results[s.name]
        text = strategy.summary(sig, store.current_trade(s.name), price) if sig else "No reading this run."
        report[s.name] = {"summary": text, "note": note, "mode": s.mode, "share": s.share}
        log.info("[%s/%s] %s %s", s.name, s.mode, text, note)
    store.snapshot(ts, "total", total_cash, total_btc, price)
    store.set("last_run", {"time": ts, "sleeves": report})
    return results


def loop(cfg, client, sleeves, store, every_sec=3600):
    while True:
        try:
            run_once(cfg, client, sleeves, store)
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
    client, sleeves, store = make(cfg)
    if cmd == "status":
        last = store.get("last_run")
        if not last:
            print("No runs yet.")
        for name, r in (last or {}).get("sleeves", {}).items():
            print(f"[{name} {r['mode']} {r['share']:.0%}] {r['summary']}")
        for s in sleeves:
            print(f"{s.name}: cash {s.cash:.2f}, BTC {s.btc}")
        return

    lock = open(cfg.data_dir / "bot.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another bot process is already running")
    if cmd == "once":
        run_once(cfg, client, sleeves, store)
    elif cmd == "loop":
        loop(cfg, client, sleeves, store)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
