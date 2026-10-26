"""Minimal Binance.US REST client (spot)."""
import hashlib
import hmac
import time
from decimal import ROUND_DOWN, Decimal
from urllib.parse import urlencode

import pandas as pd
import requests

BASE = "https://api.binance.us"
INTERVAL_MS = {"1h": 3_600_000, "6h": 21_600_000, "1d": 86_400_000}


class BinanceError(Exception):
    def __init__(self, status, code, msg):
        super().__init__(f"HTTP {status} code {code}: {msg}")
        self.status, self.code, self.msg = status, code, msg

    @property
    def would_take(self):
        """A LIMIT_MAKER order was rejected because it would cross the spread."""
        return self.code == -2010 and "immediately match" in self.msg.lower()


def sign(secret, query):
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class Filters:
    def __init__(self, tick, step, min_qty, min_notional):
        self.tick, self.step = Decimal(tick), Decimal(step)
        self.min_qty, self.min_notional = Decimal(min_qty), Decimal(min_notional)

    def price(self, p):
        return (Decimal(str(p)) / self.tick).to_integral_value(ROUND_DOWN) * self.tick

    def qty(self, q):
        return (Decimal(str(q)) / self.step).to_integral_value(ROUND_DOWN) * self.step

    def ok(self, qty, price):
        return qty >= self.min_qty and qty * price >= self.min_notional


class BinanceUS:
    def __init__(self, key="", secret="", session=None, base=BASE):
        self.key, self.secret, self.base = key, secret, base
        self.s = session or requests.Session()
        self._offset = None

    # -- transport ---------------------------------------------------------
    def _request(self, method, path, params=None, signed=False):
        params = dict(params or {})
        headers = {}
        if signed:
            if not (self.key and self.secret):
                raise RuntimeError("API key and secret are required for this call")
            params["timestamp"] = int(time.time() * 1000) + self.offset()
            params["recvWindow"] = 5000
            q = urlencode(params)
            params["signature"] = sign(self.secret, q)
            headers["X-MBX-APIKEY"] = self.key
        for attempt in range(3):
            try:
                r = self.s.request(method, self.base + path, params=params, headers=headers, timeout=15)
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            data = r.json() if r.content else {}
            if r.status_code >= 400:
                raise BinanceError(r.status_code, data.get("code"), data.get("msg", r.text))
            return data

    def offset(self):
        if self._offset is None:
            t0 = time.time() * 1000
            server = self._request("GET", "/api/v3/time")["serverTime"]
            self._offset = int(server - (t0 + time.time() * 1000) / 2)
        return self._offset

    # -- market data -------------------------------------------------------
    def klines(self, symbol, interval="1d", limit=1000):
        """Closed candles only, indexed by open time (UTC)."""
        rows = self._request("GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.set_index("time").astype(float)
        now = pd.Timestamp.now("UTC").tz_localize(None)
        return df[df.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) <= now]

    def book(self, symbol):
        b = self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
        return Decimal(b["bidPrice"]), Decimal(b["askPrice"])

    def filters(self, symbol):
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})["symbols"][0]
        f = {x["filterType"]: x for x in info["filters"]}
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {"minNotional": "0"}
        return Filters(f["PRICE_FILTER"]["tickSize"], f["LOT_SIZE"]["stepSize"],
                       f["LOT_SIZE"]["minQty"], notional["minNotional"])

    # -- account -----------------------------------------------------------
    def balances(self):
        acct = self._request("GET", "/api/v3/account", signed=True)
        return {b["asset"]: Decimal(b["free"]) for b in acct["balances"]}

    def new_order(self, symbol, side, type_, qty, price=None, tif=None, client_id=None):
        p = {"symbol": symbol, "side": side, "type": type_, "quantity": str(qty),
             "newOrderRespType": "FULL"}
        if price is not None:
            p["price"] = str(price)
        if tif:
            p["timeInForce"] = tif
        if client_id:
            p["newClientOrderId"] = client_id
        return self._request("POST", "/api/v3/order", p, signed=True)

    def get_order(self, symbol, order_id):
        return self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel(self, symbol, order_id):
        return self._request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def my_trades(self, symbol, order_id):
        return self._request("GET", "/api/v3/myTrades", {"symbol": symbol, "orderId": order_id}, signed=True)
