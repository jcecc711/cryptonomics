"""Run the bot.

    python -m bot.run once        evaluate the latest closed candles and act on them
    python -m bot.run loop        run once now, then 5 minutes past every hour
    python -m bot.run status      print the current state
    python -m bot.run web         start the dashboard

The ensemble: each strategy in BOT_STRATEGIES (DI cross and ADX-14 trigger)
runs on each timeframe in BOT_TIMEFRAMES (4h, 8h, 12h, 1d), each pair as a
sleeve with an equal share of the money and its own cash and BTC. Paper is the default.
Create data/STOP to stop all new orders.
"""
import fcntl
import logging
import sys
import time
from decimal import Decimal

from . import config, notify, regime, shorts, strategy
from .binanceus import BinanceUS
from .broker import LiveBroker, PaperBroker
from .execution import execute
from .store import Store, now

log = logging.getLogger("bot")


class Sleeve:
    """One strategy ("di" or "adx") on one timeframe, trading its own share of the money."""

    def __init__(self, name, share, mode, broker, store, cfg, kind="di", tf=None):
        self.name, self.share, self.mode = name, share, mode   # e.g. "4h" (DI) or "adx-4h"
        self.kind, self.tf = kind, tf or name
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

    def evaluate(self, candles):
        if self.kind == "adx":
            return strategy.evaluate_adx(candles, self.cfg.di_len, self.cfg.adx_smoothing, self.cfg.adx_level)
        return strategy.evaluate(candles, self.cfg.di_len, self.cfg.adx_smoothing)

    def summary(self, sig, trade, price):
        return (strategy.summary_adx if self.kind == "adx" else strategy.summary)(sig, trade, price)

    def decide(self, sig, holding):
        key = f"{self.name}.state"
        if self.kind == "adx":
            action, state = strategy.decide_adx(sig, holding, self.store.get(key, {}))
            self.store.set(key, state)
            return action
        armed = self.store.get(key)
        if armed is None:  # first run
            armed = self.cfg.join_open_trend or not sig["trend_up"]
        action, armed = strategy.decide(sig, holding, armed)
        self.store.set(key, armed)
        return action

    def step(self, client, filters_cache):
        key = ("candles", self.tf)        # sleeves on the same timeframe share one download
        if key not in filters_cache:
            filters_cache[key] = client.klines(self.cfg.symbol, self.tf)
        sig = self.evaluate(filters_cache[key])
        trade = self.store.current_trade(self.name)
        action = self.decide(sig, trade is not None)
        note = ""
        reg = filters_cache.get("regime")
        if reg and reg.get("calendar_bear") and self.cfg.bear_filter == "halving":
            # bear window: no new longs, and get out of any long still open
            if action == "BUY":
                action, note = None, "bear filter: skipped a buy signal (halving-cycle bear window)"
            elif trade is not None and action is None:
                action, note = "SELL", "bear filter: exiting (halving-cycle bear window)"
        if action and self.cfg.stop_file.exists():
            note, action = f"kill switch ({self.cfg.stop_file.name}) is on, skipped {action}", None
        elif action:
            if "filters" not in filters_cache:
                filters_cache["filters"] = client.filters(self.cfg.symbol)
            note = (note + "; " if note else "") + self.trade(action, trade, filters_cache["filters"])
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
    specs = cfg.sleeve_specs()
    share = 1 / len(specs)
    live = LiveBroker(client, cfg.symbol, cfg.base_asset, cfg.quote_asset) if cfg.mode == "live" else None
    sleeves = []
    for name, kind, tf in specs:
        broker = live or PaperBroker(client, cfg.symbol, store, cfg.paper_maker_fee, cfg.paper_taker_fee,
                                     cfg.paper_start_usd * share, key=f"paper.{name}")
        sleeves.append(Sleeve(name, share, cfg.mode, broker, store, cfg, kind, tf))
    if store.get(f"{MANUAL}.cash") is not None:          # a manual position is open
        sleeves.append(manual_sleeve(cfg, client, store, live))
    return client, sleeves, store


# -- manual position: bought and sold by hand, never by the rules --------------

MANUAL = "manual"


def manual_sleeve(cfg, client, store, live=None):
    broker = live or PaperBroker(client, cfg.symbol, store, cfg.paper_maker_fee, cfg.paper_taker_fee, 0,
                                 key=f"paper.{MANUAL}")
    return Sleeve(MANUAL, 0.0, cfg.mode, broker, store, cfg, kind=MANUAL, tf=MANUAL)


def _move_cash(sleeve, amount):
    """Add (or with a negative amount, take) cash to a paper sleeve and its simulated broker."""
    sleeve.set_balances(sleeve.cash + amount, sleeve.btc)
    q = f"paper.{sleeve.name}.quote"
    sleeve.store.set(q, str(Decimal(sleeve.store.get(q, "0")) + amount))


def manual_buy(cfg, client, sleeves, store, usd):
    """Lend `usd` from the rule sleeves sitting in cash (in proportion to their cash) and buy BTC with it."""
    if cfg.mode != "paper":
        raise SystemExit("Manual buys are for the practice account only (BOT_MODE=paper).")
    fund(sleeves)
    rule = [s for s in sleeves if s.kind != MANUAL]
    if str(usd).lower() == "all":   # every sleeve's cash
        usd = sum((s.cash for s in rule if not store.current_trade(s.name)), Decimal(0))
        usd = usd.quantize(Decimal("0.01"), rounding="ROUND_DOWN")
    usd = Decimal(str(usd)).quantize(Decimal("0.01"))
    if store.current_trade(MANUAL):
        raise SystemExit("There's already a manual position. Sell it first: python -m bot.run manual-sell")
    lenders = [s for s in rule if not store.current_trade(s.name) and s.cash > 0]
    avail = sum((s.cash for s in lenders), Decimal(0))
    if usd <= 0 or usd > avail:
        raise SystemExit(f"Choose an amount between 1 and {avail:,.2f} (the cash the sleeves have).")
    m = next((s for s in sleeves if s.kind == MANUAL), None) or manual_sleeve(cfg, client, store)
    if store.get(f"{MANUAL}.cash") is None:
        m.set_balances(Decimal(0), Decimal(0))
    lent = store.get(f"{MANUAL}.from") or {}
    left = usd
    for i, s in enumerate(lenders):
        part = left if i == len(lenders) - 1 else (usd * s.cash / avail).quantize(Decimal("0.01"))
        left -= part
        _move_cash(s, -part)
        lent[s.name] = float(Decimal(str(lent.get(s.name, 0))) + part)
    _move_cash(m, usd)
    store.set(f"{MANUAL}.from", lent)
    note = m.trade("BUY", None, client.filters(cfg.symbol))
    log.info("manual buy of %s: %s", usd, note)
    if not store.current_trade(MANUAL):                  # nothing filled: give the cash back
        _return_manual_cash(m, rule, store)
    return note


def manual_sell(cfg, client, sleeves, store):
    m = next((s for s in sleeves if s.kind == MANUAL), None)
    if m is None:
        raise SystemExit("There's no manual position.")
    trade = store.current_trade(MANUAL)
    note = m.trade("SELL", trade, client.filters(cfg.symbol)) if trade and m.btc > 0 else "nothing to sell"
    log.info("manual sell: %s", note)
    if not store.current_trade(MANUAL):
        _return_manual_cash(m, [s for s in sleeves if s.kind != MANUAL], store)
    return note


def _return_manual_cash(m, rule, store):
    """Hand the manual position's cash back to the sleeves it came from, in the same proportions."""
    lent = store.get(f"{MANUAL}.from") or {}
    total = sum(lent.values())
    cash, left = m.cash, m.cash
    named = [s for s in rule if s.name in lent]
    for i, s in enumerate(named):
        part = left if i == len(named) - 1 else (cash * Decimal(str(lent[s.name] / total))).quantize(Decimal("0.01"))
        left -= part
        _move_cash(s, part)
    _move_cash(m, -cash)
    for k in ("cash", "btc", "from"):
        store.set(f"{MANUAL}.{k}", None)


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
        if not s.funded and s.kind != MANUAL:  # paper: its own simulated balance
            s.set_balances(s.broker.balances()[0], Decimal(0))


def trade_lock(cfg):
    """Held while trading, so a manual command and the hourly run never trade at the same time."""
    f = open(cfg.data_dir / "trade.lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def run_once(cfg, client, sleeves, store):
    with trade_lock(cfg):
        return _run_once(cfg, client, sleeves, store)


def _run_once(cfg, client, sleeves, store):
    # a manual position opened or closed from the command line since the last run
    has = any(s.kind == MANUAL for s in sleeves)
    if store.get(f"{MANUAL}.cash") is not None and not has:
        live = next((s.broker for s in sleeves if s.mode == "live"), None)
        sleeves.append(manual_sleeve(cfg, client, store, live))
    elif store.get(f"{MANUAL}.cash") is None and has:
        sleeves[:] = [s for s in sleeves if s.kind != MANUAL]
    for broker in {id(s.broker): s.broker for s in sleeves}.values():
        for o in broker.cancel_leftovers():
            log.warning("cancelled leftover order %s (%s, filled %s)", o["orderId"], o["status"], o["executedQty"])
    fund(sleeves)
    cache, results, ts = {}, {}, now()
    try:
        cache[("candles", "1d")] = client.klines(cfg.symbol, "1d")
        cache["regime"] = regime.assess(cache[("candles", "1d")])
    except Exception:
        log.exception("regime check failed; trading without the bear filter this run")
    for s in sleeves:
        if s.kind == MANUAL:
            results[s.name] = (None, None, "manual position: sold only by hand")
            continue
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
        if s.kind == MANUAL:
            t = store.current_trade(s.name)
            text = (f"Manual position: {t['qty']:.6f} BTC since {t['entry_time'][:16]} at {t['entry_price']:,.0f} "
                    f"({price / t['entry_price'] - 1:+.1%})." if t else "Manual position: cash only.")
        else:
            text = s.summary(sig, store.current_trade(s.name), price) if sig else "No reading this run."
        report[s.name] = {"summary": text, "note": note, "mode": s.mode, "share": s.share}
        log.info("[%s/%s] %s %s", s.name, s.mode, text, note)
    store.snapshot(ts, "total", total_cash, total_btc, price)
    store.set("last_run", {"time": ts, "sleeves": report})
    notify.trade_alerts(cfg, results, price)
    if cfg.short_tracker:
        try:
            track_short(cfg, client, store, cache, price, ts)
        except Exception:
            log.exception("short tracker failed this run")
    for s in sleeves:   # errors and kill-switch skips: alert when they start, not every hour
        note = results[s.name][2] or ""
        problem = note if note.startswith(("error:", "kill switch")) else None
        notify.once_per_change(cfg, store, f"problem.{s.name}", problem, f"{s.name} sleeve: needs a look",
                               problem or "", priority="high", tags=("warning",))
    if cache.get("regime"):
        r = cache["regime"]
        notify.once_per_change(cfg, store, "regime", r["call"], f"Market regime: {r['call']}",
                               f"{r['phase']}, day {r['days_since_halving']} of the halving cycle. "
                               f"BTC ${price:,.0f}", tags=("compass",))
        store.set("regime", cache["regime"])
        log.info("regime: %s (%s, day %s of the halving cycle)", cache["regime"]["call"],
                 cache["regime"]["phase"], cache["regime"]["days_since_halving"])
    return results


def track_short(cfg, client, store, cache, price, ts):
    for tf in ("4h", "8h"):
        if ("candles", tf) not in cache:
            cache[("candles", tf)] = client.klines(cfg.symbol, tf)
    event, st = shorts.step(cfg, store, cache[("candles", "4h")], cache[("candles", "8h")], price, ts)
    pos = st["open"]
    if pos:
        m = pos["mark"]
        log.info("tracked short: %s contract(s) = %s BTC from %s at %.2f, now %.2f, P/L %+.2f (est.)",
                 pos["contracts"], pos["qty"], pos["opened"], pos["entry"], price, m["net"])
    else:
        log.info("tracked short: none (4h %s, 8h %s)", st["signal"]["4h"], st["signal"]["8h"])
    if event == "open":
        when = f"picked up from {pos['opened']} UTC" if pos["picked_up"] else "now"
        notify.send(cfg, f"SHORT {pos['qty']} BTC (tracked, not placed)",
                    f"8h and 4h both have DI- on top. A Coinbase short of {pos['contracts']} contract(s) "
                    f"= {pos['qty']} BTC at ${pos['entry']:,.0f} ({when}).\nCloses when 4h DI+ crosses back above DI-."
                    f"\nBTC ${price:,.0f}", priority="high", tags=("chart_with_downwards_trend",))
    elif event == "close":
        t = st["closed"][0]
        notify.send(cfg, f"COVER short: {'+' if t['net'] >= 0 else '-'}${abs(t['net']):,.2f} (tracked, not placed)",
                    f"4h DI+ crossed back above DI-. Short of {t['qty']} BTC from ${t['entry']:,.0f} closed at "
                    f"${t['exit']:,.0f} after {t['days']:.1f} days: {t['pct']:+.1f}% on the short, "
                    f"net ${t['net']:+,.2f} after estimated fees and funding.", priority="high",
                    tags=("chart_with_upwards_trend",))


def loop(cfg, client, sleeves, store, every_sec=3600):
    while True:
        try:
            run_once(cfg, client, sleeves, store)
            notify.once_per_change(cfg, store, "run", None, "", "")
        except Exception as e:
            log.exception("run failed; will retry next cycle")
            notify.once_per_change(cfg, store, "run", type(e).__name__, "Bot run failed",
                                   f"{e}\nIt will retry every hour; check the server log.", priority="high",
                                   tags=("warning",))
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
    if cmd == "alert-test":
        on = notify.channels(cfg)
        if not on:
            import secrets
            raise SystemExit("No alerts set up. In .env, set BOT_ALERT_EMAIL, BOT_SMTP_USER and "
                             "BOT_SMTP_PASSWORD for email, and/or BOT_NTFY_TOPIC for the ntfy app, e.g.\n"
                             f"  BOT_NTFY_TOPIC=cryptonomics-{secrets.token_hex(6)}")
        ok = notify.send(cfg, f"test ({cfg.mode})", "Alerts are working. You'll get one for every "
                         "trade, a regime change, or a problem.", tags=("white_check_mark",))
        print(f"sent via {', '.join(on)}" if ok else "failed: see the log line(s) above")
        return
    if cmd in ("manual-buy", "manual-sell"):
        with trade_lock(cfg):
            client, sleeves, store = make(cfg, client, store)   # fresh view inside the lock
            if cmd == "manual-buy":
                if len(argv) < 2:
                    raise SystemExit("Usage: python -m bot.run manual-buy <dollars | all>, e.g. manual-buy 2000")
                note = manual_buy(cfg, client, sleeves, store, argv[1].replace("$", "").replace(",", ""))
                t = store.current_trade(MANUAL)
                notify.send(cfg, f"Manual BUY ({cfg.mode.upper()})", note, priority="high", tags=("hand",))
            else:
                note = manual_sell(cfg, client, sleeves, store)
                notify.send(cfg, f"Manual SELL ({cfg.mode.upper()})", note, priority="high", tags=("hand",))
        print(note)
        return
    if cmd == "status":
        last = store.get("last_run")
        if not last:
            print("No runs yet.")
        for name, r in (last or {}).get("sleeves", {}).items():
            print(f"[{name} {r['mode']} {r['share']:.0%}] {r['summary']}")
        for s in sleeves:
            print(f"{s.name}: cash {s.cash:.2f}, BTC {s.btc}")
        st = store.get(shorts.KEY)
        if st:
            pos = st["open"]
            if pos:
                m = pos["mark"]
                print(f"tracked short (not placed): {pos['qty']} BTC from {pos['opened']} at {pos['entry']:,.2f}, "
                      f"BTC {st['price']:,.2f}, P/L {m['net']:+,.2f} est. ({m['pct']:+.1f}%)")
            else:
                print(f"tracked short: none (4h {st['signal']['4h']}, 8h {st['signal']['8h']})")
            print(f"tracked shorts closed: {len(st['closed'])}, realized {st['realized']:+,.2f} est.")
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
