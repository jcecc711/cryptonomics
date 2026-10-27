"""Order execution: work a maker-only order at the top of the book, then
optionally cross the spread if it hasn't filled in time.

Maker orders pay 0% on Binance.US and never pay the spread. The order is
re-posted whenever the market moves away from it.
"""
import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from .binanceus import BinanceError
from .broker import FINAL

log = logging.getLogger("bot.exec")


@dataclass
class Fill:
    base: Decimal = Decimal(0)        # gross base asset bought or sold
    quote: Decimal = Decimal(0)       # quote asset spent or received (before fees)
    fee_usd: float = 0.0
    fee_base: Decimal = Decimal(0)    # commission taken out of the base asset
    fee_quote: Decimal = Decimal(0)   # commission taken out of the quote asset

    @property
    def avg_price(self):
        return float(self.quote / self.base) if self.base else 0.0

    def add(self, order, broker):
        """Add one order's fills; returns (base filled, fee in USD) for that order."""
        b, q = Decimal(order["executedQty"]), Decimal(order["cummulativeQuoteQty"])
        usd = 0.0
        if b > 0:
            usd, in_base, in_quote = broker.fees(order["orderId"], q / b)
            self.base += b
            self.quote += q
            self.fee_usd += usd
            self.fee_base += in_base
            self.fee_quote += in_quote
        return b, usd


def _remaining_qty(side, budget, fill, price, filters):
    if side == "BUY":
        return filters.qty((budget - fill.quote) / price)
    return filters.qty(budget - fill.base)


def execute(side, budget, broker, filters, store, cfg, strategy="", clock=None, sleep=None):
    """Buy with `budget` quote (side BUY) or sell `budget` base (side SELL)."""
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    fill = Fill()
    deadline = clock() + cfg.maker_timeout_min * 60
    while clock() < deadline and not cfg.stop_file.exists():
        bid, ask = broker.book()
        price = filters.price(bid if side == "BUY" else ask)
        qty = _remaining_qty(side, budget, fill, price, filters)
        if not filters.ok(qty, price):
            break
        try:
            o = broker.place_maker(side, qty, price)
        except BinanceError as e:
            if e.would_take:  # the book moved between reading it and posting
                sleep(1)
                continue
            raise
        row = store.add_order(strategy, o["orderId"], side, "LIMIT_MAKER", price, qty, o["status"])
        log.info("maker %s %s @ %s", side, qty, price)

        while o["status"] not in FINAL and clock() < deadline and not cfg.stop_file.exists():
            sleep(cfg.poll_sec)
            o = broker.status(o["orderId"])
            if o["status"] in FINAL:
                break
            bid, ask = broker.book()
            if (side == "BUY" and bid > price) or (side == "SELL" and ask < price):
                break  # market moved away: re-post at the new best price
        if o["status"] not in FINAL:
            o = broker.cancel(o["orderId"])
        filled, fee = fill.add(o, broker)
        store.update_order(row, o["status"], o["executedQty"], o["cummulativeQuoteQty"], fee)
        log.info("maker %s %s, filled %s", side, o["status"], filled)

    if cfg.taker_fallback and not cfg.stop_file.exists():
        bid, ask = broker.book()
        limit = filters.price(ask * (1 + Decimal(str(cfg.max_slippage))) if side == "BUY"
                              else bid * (1 - Decimal(str(cfg.max_slippage))))
        qty = _remaining_qty(side, budget, fill, limit, filters)
        if filters.ok(qty, limit):
            o = broker.place_ioc(side, qty, limit)
            row = store.add_order(strategy, o["orderId"], side, "LIMIT_IOC", limit, qty, o["status"])
            filled, fee = fill.add(o, broker)
            store.update_order(row, o["status"], o["executedQty"], o["cummulativeQuoteQty"], fee)
            log.info("taker fallback %s %s, filled %s", side, o["status"], filled)
    return fill
