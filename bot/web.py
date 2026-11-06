"""Dashboard web app, served with the standard library.

    python -m bot.run web       -> http://127.0.0.1:8080

Endpoints (all behind DASHBOARD_PASSWORD when set, any username):
    GET  /                      the app (dashboard.html)
    GET  /api/status            bot state: sleeves, equity, trades, signals, orders, settings
    GET  /api/candles           ?symbol=BTCUSDT&tf=4h[&end=<ms>][&limit=1000] OHLCV, newest last;
                                `end` pages back through history; includes the forming candle
    GET  /api/price             ?symbol=BTCUSDT  live price and 24h change
    GET  /api/watchlist         watched coins with price, 24h change and DI trend on 4h/8h/12h/1d
    POST /api/watchlist         {"symbols": [...]} replace the watched coins
    GET  /api/symbols           coins tradable on Binance.US (USDT / USD / USDC quotes)

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


def candles(client, symbol, tf, end=None, limit=1000):
    def load():
        if tf in BUILT:
            df, more = _built(client, symbol, tf, end, limit)
        else:
            df = client.klines(symbol, tf, limit=limit, end_time=end, closed_only=False)
            more = len(df) == limit
        return {"symbol": symbol, "tf": tf, "has_more": more,
                "t": [int(t.timestamp()) for t in df.index],
                "o": df.open.round(8).tolist(), "h": df.high.round(8).tolist(),
                "l": df.low.round(8).tolist(), "c": df.close.round(8).tolist(),
                "v": df.volume.round(4).tolist()}
    # the latest page changes with every trade; older pages never change
    return _cache.get(("candles", symbol, tf, end, limit), 15 if end is None else 3600, load)


def tradable(client):
    return _cache.get(("symbols",), 3600, client.tradable_symbols)


def analyse(client, symbol, di_len=38, adx_smoothing=9):
    """The bot's DI read on each timeframe, plus a few context numbers."""
    def load():
        out = {"tfs": {}}
        for tf in ANALYSIS_TFS:
            k = client.klines(symbol, tf, limit=400)
            d = adx(k, di_len, adx_smoothing)
            up = d.di_plus > d.di_minus
            flips = up.ne(up.shift()).to_numpy().nonzero()[0]
            since = int(len(up) - 1 - flips[-1]) if len(flips) > 1 else None
            out["tfs"][tf] = {"up": bool(up.iloc[-1]), "di_plus": _num(d.di_plus.iloc[-1], 1),
                              "di_minus": _num(d.di_minus.iloc[-1], 1), "adx": _num(d.adx.iloc[-1], 1),
                              "bars_since_cross": since}
            if tf == "4h":
                out["spark"] = [_num(x, 8) for x in k.close.tail(42)]   # last 7 days
            if tf == "1d":
                c = k.close
                out["rsi"] = _num(rsi(c).iloc[-1], 1)
                out["vs_ma200"] = _num(100 * (c.iloc[-1] / sma(c, 200).iloc[-1] - 1), 1)
                out["vs_ema21"] = _num(100 * (c.iloc[-1] / ema(c, 21).iloc[-1] - 1), 1)
        out["score"] = sum(v["up"] for v in out["tfs"].values())
        return out
    return _cache.get(("analysis", symbol), 300, load)


def watchlist(cfg, store, client, lock):
    with lock:
        symbols = store.get("watchlist") or DEFAULT_WATCHLIST
    tickers = {t["symbol"]: t for t in _cache.get(("tickers", tuple(symbols)), 20,
                                                  lambda: client.tickers_24h(symbols))}
    def one(s):
        t = tickers.get(s, {})
        row = {"symbol": s, "price": _num(t.get("lastPrice"), 8), "change": _num(t.get("priceChangePercent")),
               "volume": _num(t.get("quoteVolume"), 0), "high": _num(t.get("highPrice"), 8),
               "low": _num(t.get("lowPrice"), 8), "bot": s == cfg.symbol}
        try:
            row.update(analyse(client, s, cfg.di_len, cfg.adx_smoothing))
        except Exception as e:  # one bad symbol shouldn't break the list
            row["error"] = str(e)
        return row

    with ThreadPoolExecutor(max_workers=6) as pool:  # gentle on Binance.US rate limits
        rows = list(pool.map(one, symbols))
    return {"symbols": rows, "timeframes": list(ANALYSIS_TFS)}


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
            s = (q.get("symbol") or [cfg.symbol])[0].upper()
            if s != cfg.symbol and s not in {x["symbol"] for x in tradable(client)}:
                raise ValueError(f"unknown symbol {s!r}")
            return s

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
                    return self._send(200, candles(client, self._symbol(q), tf, end, limit))
                if url.path == "/api/price":
                    s = self._symbol(q)
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
                    s = str(s).upper()
                    if s not in known:
                        raise ValueError(f"unknown symbol {s!r}")
                    if s not in clean:
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
