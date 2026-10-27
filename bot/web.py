"""Dashboard: one page plus a JSON endpoint, served with the standard library.

    python -m bot.run web       -> http://127.0.0.1:8080

Set DASHBOARD_PASSWORD to require a password (any username). Keep the default
127.0.0.1 binding unless it sits behind a password and HTTPS.
"""
import base64
import hmac
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import strategy
from .binanceus import BinanceUS
from .store import Store

PAGE = Path(__file__).with_name("dashboard.html")
_chart_cache = {"t": 0, "data": None}


def chart_data(cfg, client, bars=180):
    """Last `bars` candles with DI lines, for each timeframe in the ensemble."""
    if time.time() - _chart_cache["t"] < 300 and _chart_cache["data"]:
        return _chart_cache["data"]
    data = {}
    for tf in cfg.timeframes:
        candles = client.klines(cfg.symbol, tf)
        d = strategy.adx(candles, cfg.di_len, cfg.adx_smoothing).tail(bars)
        fmt = "%Y-%m-%d" if tf == "1d" else "%Y-%m-%d %H:%M"
        data[tf] = {"times": [t.strftime(fmt) for t in d.index],
                    "close": [round(x, 2) for x in candles["close"].tail(bars)],
                    "di_plus": [round(x, 2) for x in d["di_plus"]],
                    "di_minus": [round(x, 2) for x in d["di_minus"]],
                    "adx": [round(x, 2) for x in d["adx"]]}
    _chart_cache.update(t=time.time(), data=data)
    return data


def status(cfg, store, client):
    sleeves = []
    for name in cfg.timeframes:
        eq = store.rows("SELECT ts, value, cash, btc, price FROM equity WHERE strategy=? ORDER BY ts", name)
        sleeves.append({
            "name": name, "share": 1 / len(cfg.timeframes),
            "trade": store.current_trade(name),
            "last": eq[-1] if eq else None, "start_value": eq[0]["value"] if eq else None,
            "signals": store.rows("SELECT * FROM signals WHERE strategy=? ORDER BY candle_time DESC LIMIT 20", name),
        })
    total = store.rows("SELECT ts, value FROM equity WHERE strategy='total' ORDER BY ts")
    out = {
        "mode": cfg.mode, "symbol": cfg.symbol, "di_len": cfg.di_len,
        "kill_switch": cfg.stop_file.exists(),
        "last_run": store.get("last_run"),
        "sleeves": sleeves,
        "equity": total[-1000:],
        "trades": store.rows("SELECT * FROM trades ORDER BY id DESC"),
        "orders": store.rows("SELECT * FROM orders ORDER BY id DESC LIMIT 50"),
    }
    try:
        out["chart"] = chart_data(cfg, client)
    except Exception as e:  # dashboard still loads if Binance.US is unreachable
        out["chart"], out["chart_error"] = None, str(e)
    return out


def make_handler(cfg, store, client):
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

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorized():
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="cryptonomics"')
                self.end_headers()
                return
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                body = json.dumps(status(cfg, store, client), default=str).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

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
