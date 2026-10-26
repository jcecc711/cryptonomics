"""Live and paper brokers with one interface.

Order dicts use Binance's shape: orderId, status, executedQty, cummulativeQuoteQty.
"""
import itertools
from decimal import Decimal

from .binanceus import BinanceError

FINAL = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}


class LiveBroker:
    def __init__(self, client, symbol, base, quote):
        self.c, self.symbol, self.base, self.quote = client, symbol, base, quote

    def book(self):
        return self.c.book(self.symbol)

    def balances(self):
        b = self.c.balances()
        return b.get(self.quote, Decimal(0)), b.get(self.base, Decimal(0))

    def place_maker(self, side, qty, price):
        return self.c.new_order(self.symbol, side, "LIMIT_MAKER", qty, price)

    def place_ioc(self, side, qty, price):
        return self.c.new_order(self.symbol, side, "LIMIT", qty, price, tif="IOC")

    def status(self, order_id):
        return self.c.get_order(self.symbol, order_id)

    def cancel(self, order_id):
        try:
            self.c.cancel(self.symbol, order_id)
        except BinanceError:
            pass  # most often: already filled
        return self.status(order_id)

    def fees(self, order_id, price):
        """(commission in USD, commission paid in the base asset) for an order's fills."""
        usd, in_base = 0.0, Decimal(0)
        for t in self.c.my_trades(self.symbol, order_id):
            amt = Decimal(t["commission"])
            if t["commissionAsset"] == self.base:
                in_base += amt
                usd += float(amt * Decimal(str(price)))
            elif t["commissionAsset"] == self.quote:
                usd += float(amt)
        return usd, in_base


class PaperBroker:
    """Simulates fills against the live Binance.US order book.

    A resting maker order fills once the book trades through its price (for a
    buy, the best bid drops below it or the best ask comes down to it). Balances
    live in the bot's store so they survive restarts.
    """

    def __init__(self, client, symbol, store, maker_fee=0.0, taker_fee=0.0002, start_usd=10_000.0):
        self.c, self.symbol, self.store = client, symbol, store
        self.maker_fee, self.taker_fee = Decimal(str(maker_fee)), Decimal(str(taker_fee))
        self.orders, self._ids = {}, itertools.count(1)
        if store.get("paper_quote") is None:
            store.set("paper_quote", str(start_usd))
            store.set("paper_base", "0")

    def book(self):
        return self.c.book(self.symbol)

    def balances(self):
        return Decimal(self.store.get("paper_quote")), Decimal(self.store.get("paper_base"))

    def _fill(self, o, price, fee):
        qty = Decimal(o["origQty"])
        quote, base = self.balances()
        if o["side"] == "BUY":
            quote -= qty * price
            base += qty * (1 - fee)
            o["fee_base"] = qty * fee
        else:
            base -= qty
            quote += qty * price * (1 - fee)
            o["fee_usd"] = float(qty * price * fee)
        self.store.set("paper_quote", str(quote))
        self.store.set("paper_base", str(base))
        o.update(status="FILLED", executedQty=str(qty), cummulativeQuoteQty=str(qty * price))

    def _new(self, side, qty, price, type_):
        quote, base = self.balances()
        if (side == "BUY" and qty * price > quote) or (side == "SELL" and qty > base):
            raise BinanceError(400, -2010, "Account has insufficient balance for requested action.")
        oid = next(self._ids)
        o = {"orderId": oid, "side": side, "type": type_, "price": str(price), "origQty": str(qty),
             "status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0"}
        self.orders[oid] = o
        return o

    def place_maker(self, side, qty, price):
        bid, ask = self.book()
        if (side == "BUY" and price >= ask) or (side == "SELL" and price <= bid):
            raise BinanceError(400, -2010, "Order would immediately match and take.")
        return dict(self._new(side, qty, price, "LIMIT_MAKER"))

    def place_ioc(self, side, qty, price):
        o = self._new(side, qty, price, "LIMIT")
        bid, ask = self.book()
        if side == "BUY" and ask <= price:
            self._fill(o, ask, self.taker_fee)
        elif side == "SELL" and bid >= price:
            self._fill(o, bid, self.taker_fee)
        else:
            o["status"] = "EXPIRED"
        return dict(o)

    def status(self, order_id):
        o = self.orders[order_id]
        if o["status"] == "NEW":
            bid, ask = self.book()
            p = Decimal(o["price"])
            if (o["side"] == "BUY" and (bid < p or ask <= p)) or (o["side"] == "SELL" and (ask > p or bid >= p)):
                self._fill(o, p, self.maker_fee)
        return dict(o)

    def cancel(self, order_id):
        o = self.orders[order_id]
        if o["status"] == "NEW":
            o["status"] = "CANCELED"
        return dict(o)

    def fees(self, order_id, price):
        o = self.orders[order_id]
        fee_base = o.get("fee_base", Decimal(0))
        return float(fee_base * Decimal(str(price))) + o.get("fee_usd", 0.0), fee_base
