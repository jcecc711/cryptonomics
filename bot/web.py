"""Dashboard web app, served with the standard library.

    python -m bot.run web       -> http://127.0.0.1:8080

Endpoints (all behind DASHBOARD_PASSWORD when set, any username):
    GET  /                      the app (dashboard.html)
    GET  /api/status            bot state: sleeves, equity, trades, signals, orders, settings
    GET  /api/candles           ?symbol=BTCUSDT&tf=4h[&end=<ms>][&limit=1000] OHLCV, newest last;
                                `end` pages back through history; includes the forming candle
    GET  /api/price             ?symbol=BTCUSDT  live price and 24h change
    GET  /api/watchlist         watched coins and US stocks with price, change and DI trend
                                (coins: 4h/8h/12h/1d; stocks: 1h/4h/1d/1w)
    POST /api/watchlist         {"symbols": [...]} replace the list; a coin pair (SOLUSDT) or a
                                US stock ticker (AAPL), checked against Binance.US / Yahoo
    GET  /api/symbols           coins tradable on Binance.US (USDT / USD / USDC quotes)

Anything that isn't a Binance.US pair is treated as a US stock ticker (see stocks.py).

Keep the default 127.0.0.1 binding unless it sits behind a password and a
private network (Tailscale) or HTTPS.
"""
import base64
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bench.indicators import adx, ema, rma, sma

from . import regime as regime_mod
from . import stocks
from .binanceus import BinanceUS
from .store import Store

PAGE = Path(__file__).with_name("dashboard.html")
DEFAULT_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
                     "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]
ANALYSIS_TFS = ("4h", "8h", "12h", "1d")
CHART_TFS = ("15m", "1h", "3h", "4h", "6h", "8h", "12h", "1d", "1w")
BUILT = {"3h": ("1h", 3)}   # timeframes Binance.US doesn't offer: built from a smaller one
MAX_WATCHLIST = 30


class TTLCache:
    def __init__(self, maxsize=256):
        self.data, self.maxsize, self.lock = {}, maxsize, threading.Lock()

    def get(self, key, ttl, fn):
        now = time.time()
        with self.lock:
            hit = self.data.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        value = fn()
        with self.lock:
            if len(self.data) >= self.maxsize:
                self.data.pop(min(self.data, key=lambda k: self.data[k][0]))
            self.data[key] = (now, value)
        return value

    def clear(self):
        with self.lock:
            self.data.clear()


_cache = TTLCache()


def rsi(close, n=14):
    d = close.diff()
    gain, loss = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss)


def _num(x, nd=2):
    if x is None:
        return None
    x = float(x)
    return None if x != x else round(x, nd)


# -- data ------------------------------------------------------------------

def _built(client, symbol, tf, end, limit):
    """e.g. 3h candles from 1h ones, aligned to midnight UTC; a partial first group is dropped."""
    src, k = BUILT[tf]
    raw = client.klines(symbol, src, limit=min(1000, limit * k), end_time=end, closed_only=False)
    g = raw.resample(tf, label="left", closed="left")
    df = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    counts = g.size()
    if len(df) and counts.iloc[0] < k:
        df = df.iloc[1:]
    return df, len(raw) == min(1000, limit * k)


def _frame(symbol, tf, df, more):
    return {"symbol": symbol, "tf": tf, "has_more": more,
            "t": [int(t.timestamp()) for t in df.index],
            "o": df.open.round(8).tolist(), "h": df.high.round(8).tolist(),
            "l": df.low.round(8).tolist(), "c": df.close.round(8).tolist(),
            "v": df.volume.round(4).tolist()}


def is_stock(client, symbol):
    return symbol not in {x["symbol"] for x in tradable(client)}


def candles(client, symbol, tf, end=None, limit=1000, stock=False):
    def load():
        if stock:
            df, more = stocks.candles(symbol, tf, end=None if end is None else end / 1000, limit=limit)
            return {**_frame(symbol, tf, df, more), "kind": "stock"}
        if tf in BUILT:
            df, more = _built(client, symbol, tf, end, limit)
        else:
            df = client.klines(symbol, tf, limit=limit, end_time=end, closed_only=False)
            more = len(df) == limit
        return {**_frame(symbol, tf, df, more), "kind": "crypto"}
    # the latest page changes with every trade; older pages never change
    return _cache.get(("candles", symbol, tf, end, limit), (60 if stock else 15) if end is None else 3600, load)


def tradable(client):
    return _cache.get(("symbols",), 3600, client.tradable_symbols)


def _di_read(k, di_len, adx_smoothing):
    d = adx(k, di_len, adx_smoothing)
    up = d.di_plus > d.di_minus
    flips = up.ne(up.shift()).to_numpy().nonzero()[0]
    since = int(len(up) - 1 - flips[-1]) if len(flips) > 1 else None
    return {"up": bool(up.iloc[-1]), "di_plus": _num(d.di_plus.iloc[-1], 1),
            "di_minus": _num(d.di_minus.iloc[-1], 1), "adx": _num(d.adx.iloc[-1], 1), "bars_since_cross": since}


def _daily_context(out, c):
    out["rsi"] = _num(rsi(c).iloc[-1], 1)
    out["vs_ma200"] = _num(100 * (c.iloc[-1] / sma(c, 200).iloc[-1] - 1), 1)
    out["vs_ema21"] = _num(100 * (c.iloc[-1] / ema(c, 21).iloc[-1] - 1), 1)


def analyse_stock(symbol, di_len=38, adx_smoothing=9):
    """The same DI read for a US stock on 1h / 4h (per session) / 1d / 1w closed candles."""
    def load():
        out = {"tfs": {}}
        hourly, _ = stocks.candles(symbol, "1h", limit=2000, closed_only=True)
        frames = {"1h": hourly.tail(400), "4h": stocks.by_session(hourly, 4).tail(400),
                  "1d": stocks.candles(symbol, "1d", limit=400, closed_only=True)[0],
                  "1w": stocks.candles(symbol, "1w", limit=400, closed_only=True)[0]}
        for tf in stocks.ANALYSIS_TFS:
            if len(frames[tf]) > di_len:
                out["tfs"][tf] = _di_read(frames[tf], di_len, adx_smoothing)
        out["spark"] = [_num(x, 4) for x in hourly.close.tail(35)]     # last 5 sessions
        if len(frames["1d"]) > 30:
            _daily_context(out, frames["1d"].close)
        out["score"] = sum(v["up"] for v in out["tfs"].values())
        return out
    return _cache.get(("analysis", "stock", symbol), 600, load)


def analyse(client, symbol, di_len=38, adx_smoothing=9):
    """The bot's DI read on each timeframe, plus a few context numbers."""
    def load():
        out = {"tfs": {}}
        for tf in ANALYSIS_TFS:
            k = client.klines(symbol, tf, limit=400)
            out["tfs"][tf] = _di_read(k, di_len, adx_smoothing)
            if tf == "4h":
                out["spark"] = [_num(x, 8) for x in k.close.tail(42)]   # last 7 days
            if tf == "1d":
                _daily_context(out, k.close)
        out["score"] = sum(v["up"] for v in out["tfs"].values())
        return out
    return _cache.get(("analysis", symbol), 300, load)


def watchlist(cfg, store, client, lock):
    with lock:
        symbols = store.get("watchlist") or DEFAULT_WATCHLIST
    coins = [s for s in symbols if not is_stock(client, s)]
    tickers = {t["symbol"]: t for t in _cache.get(("tickers", tuple(coins)), 20,
                                                  lambda: client.tickers_24h(coins))} if coins else {}

    def stock(s):
        row = {"symbol": s, "kind": "stock", "bot": False}
        try:
            q = _cache.get(("quote", s), 60, lambda: stocks.quote(s))
            row.update({"name": q["name"], "price": _num(q["price"], 4), "change": _num(q["change"]),
                        "volume": _num(q["volume"], 0), "high": _num(q["high"], 4), "low": _num(q["low"], 4)})
            row.update(analyse_stock(s, cfg.di_len, cfg.adx_smoothing))
        except Exception as e:
            row["error"] = str(e)
        return row

    def one(s):
        if s not in coins:
            return stock(s)
        t = tickers.get(s, {})
        row = {"symbol": s, "price": _num(t.get("lastPrice"), 8), "change": _num(t.get("priceChangePercent")),
               "volume": _num(t.get("quoteVolume"), 0), "high": _num(t.get("highPrice"), 8),
               "low": _num(t.get("lowPrice"), 8), "bot": s == cfg.symbol, "kind": "crypto"}
        try:
            row.update(analyse(client, s, cfg.di_len, cfg.adx_smoothing))
        except Exception as e:  # one bad symbol shouldn't break the list
            row["error"] = str(e)
        return row

    with ThreadPoolExecutor(max_workers=6) as pool:  # gentle on Binance.US rate limits
        rows = list(pool.map(one, symbols))
    return {"symbols": rows, "timeframes": list(ANALYSIS_TFS), "stock_timeframes": list(stocks.ANALYSIS_TFS)}


def market_regime(cfg, client):
    """Fresh regime readings for the dashboard (the bot stores its own each run)."""
    return _cache.get(("regime", cfg.symbol), 300, lambda: regime_mod.assess(client.klines(cfg.symbol, "1d")))


def status(cfg, store, lock):
    with lock:
        sleeves, specs = [], cfg.sleeve_specs()
        for name, kind, tf in specs:
            eq = store.rows("SELECT ts, value, cash, btc, price FROM equity WHERE strategy=? ORDER BY ts", name)
            sleeves.append({
                "name": name, "kind": kind, "tf": tf, "share": 1 / len(specs),
                "trade": store.current_trade(name),
                "last": eq[-1] if eq else None, "start_value": eq[0]["value"] if eq else None,
                "signals": store.rows("SELECT * FROM signals WHERE strategy=? ORDER BY candle_time DESC LIMIT 20",
                                      name),
            })
        total = store.rows("SELECT ts, value, price FROM equity WHERE strategy='total' ORDER BY ts")
        return {
            "mode": cfg.mode, "symbol": cfg.symbol, "di_len": cfg.di_len,
            "kill_switch": cfg.stop_file.exists(),
            "last_run": store.get("last_run"),
            "sleeves": sleeves,
            "equity": total[-2000:],
            "trades": store.rows("SELECT * FROM trades ORDER BY id DESC"),
            "orders": store.rows("SELECT * FROM orders ORDER BY id DESC LIMIT 100"),
            "settings": {"mode": cfg.mode, "symbol": cfg.symbol, "timeframes": list(cfg.timeframes),
                         "strategies": list(cfg.strategies), "adx_level": cfg.adx_level,
                         "bear_filter": cfg.bear_filter,
                         "di_len": cfg.di_len, "adx_smoothing": cfg.adx_smoothing,
                         "join_open_trend": cfg.join_open_trend, "max_order_usd": cfg.max_order_usd,
                         "maker_timeout_min": cfg.maker_timeout_min, "taker_fallback": cfg.taker_fallback,
                         "max_slippage": cfg.max_slippage, "paper_start_usd": cfg.paper_start_usd,
                         "password_set": bool(cfg.dashboard_password)},
        }


# -- HTTP ------------------------------------------------------------------

def make_handler(cfg, store, client):
    lock = threading.Lock()  # one sqlite connection shared by the server's threads

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self):
            if not cfg.dashboard_password:
                return True
            h = self.headers.get("Authorization", "")
            if h.startswith("Basic "):
                try:
                    pw = base64.b64decode(h[6:]).decode().split(":", 1)[1]
                except Exception:
                    return False
                return hmac.compare_digest(pw, cfg.dashboard_password)
            return False

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _deny(self):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="cryptonomics"')
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _symbol(self, q):
            """(symbol, is_stock): a Binance.US pair, or something shaped like a stock ticker."""
            s = (q.get("symbol") or [cfg.symbol])[0].upper()
            if s == cfg.symbol or s in {x["symbol"] for x in tradable(client)}:
                return s, False
            if not stocks.is_ticker(s):
                raise ValueError(f"unknown symbol {s!r}")
            return s, True

        def do_GET(self):
            if not self._authorized():
                return self._deny()
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if url.path == "/api/status":
                    out = status(cfg, store, lock)
                    try:
                        out["regime"] = market_regime(cfg, client)
                    except Exception as e:  # Binance.US unreachable: fall back to the bot's last reading
                        with lock:
                            out["regime"] = store.get("regime")
                        out["regime_error"] = str(e)
                    return self._send(200, out)
                if url.path == "/api/candles":
                    tf = (q.get("tf") or ["4h"])[0]
                    if tf not in CHART_TFS:
                        raise ValueError(f"unknown timeframe {tf!r}")
                    end = int(q["end"][0]) if "end" in q else None
                    limit = max(10, min(1000, int((q.get("limit") or [1000])[0])))
                    s, stock = self._symbol(q)
                    return self._send(200, candles(client, s, tf, end, limit, stock=stock))
                if url.path == "/api/price":
                    s, stock = self._symbol(q)
                    if stock:
                        x = _cache.get(("quote", s), 15, lambda: stocks.quote(s))
                        return self._send(200, {"symbol": s, "price": x["price"], "change": x["change"], "time": time.time()})
                    t = _cache.get(("tickers", (s,)), 5, lambda: client.tickers_24h([s]))[0]
                    return self._send(200, {"symbol": s, "price": float(t["lastPrice"]),
                                            "change": float(t["priceChangePercent"]), "time": time.time()})
                if url.path == "/api/watchlist":
                    return self._send(200, watchlist(cfg, store, client, lock))
                if url.path == "/api/symbols":
                    return self._send(200, tradable(client))
                return self._send(404, {"error": "not found"})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # e.g. Binance.US unreachable: keep the page usable
                return self._send(502, {"error": str(e)})

        def do_POST(self):
            if not self._authorized():
                return self._deny()
            if urlparse(self.path).path != "/api/watchlist":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 10_000:
                    raise ValueError("request too large")
                symbols = json.loads(self.rfile.read(n) or b"{}").get("symbols")
                if not isinstance(symbols, list) or not 1 <= len(symbols) <= MAX_WATCHLIST:
                    raise ValueError(f"send 1-{MAX_WATCHLIST} symbols")
                known = {x["symbol"] for x in tradable(client)}
                clean = []
                for s in symbols:
                    s = str(s).upper().lstrip("$")
                    if s in clean:
                        continue
                    if s not in known:   # not a coin: it has to be a US stock Yahoo knows
                        if not stocks.is_ticker(s):
                            raise ValueError(f"unknown symbol {s!r}")
                        try:
                            _cache.get(("quote", s), 60, lambda: stocks.quote(s))
                        except stocks.StockError:
                            raise ValueError(f"{s!r} isn't a coin on Binance.US or a US stock")
                    clean.append(s)
                with lock:
                    store.set("watchlist", clean)
                return self._send(200, {"symbols": clean})
            except (ValueError, AttributeError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(502, {"error": str(e)})

        def log_message(self, *args):
            pass

    return Handler


def make_server(cfg, port=None, client=None):
    store = Store(cfg.db_path)
    client = client or BinanceUS()
    return ThreadingHTTPServer((cfg.dashboard_host, cfg.dashboard_port if port is None else port),
                               make_handler(cfg, store, client))


def serve(cfg):
    srv = make_server(cfg)
    print(f"Dashboard on http://{cfg.dashboard_host}:{srv.server_address[1]}  ({cfg.mode} mode)")
    srv.serve_forever()
