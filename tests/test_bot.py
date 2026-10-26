"""Bot tests. No network needed.

    python -m unittest discover -s tests -v
"""
import base64
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))  # bench uses flat imports
from bench.exits import signals as bench_signals
from bench.indicators import adx
from bot import run, strategy, web
from bot.binanceus import BinanceError, Filters, sign
from bot.broker import PaperBroker
from bot.config import Config
from bot.execution import execute
from bot.store import Store


def synthetic_candles(n=900, seed=1):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


D = Decimal


class FakeClient:
    """Stands in for BinanceUS: an order book scripted over (fake) time, fixed candles.

    books: [(from_second, bid, ask), ...]; the book shown is the latest entry
    whose time has been reached on `clock`.
    """

    def __init__(self, books=((0, "100.00", "100.01"),), candles=None, clock=None):
        self.books = [(t, D(b), D(a)) for t, b, a in books]
        self.candles, self.clock = candles, clock or (lambda: 0)

    def book(self, symbol):
        now = self.clock()
        return [(b, a) for t, b, a in self.books if t <= now][-1]

    def filters(self, symbol):
        return Filters("0.01", "0.00001", "0.00001", "1")

    def klines(self, symbol, interval="1d", limit=1000):
        return self.candles


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_cfg(tmp, **kw):
    return Config(data_dir=Path(tmp), poll_sec=10, maker_timeout_min=1, **kw)


class TestSigning(unittest.TestCase):
    def test_matches_binance_docs_example(self):
        secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        q = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
             "&recvWindow=5000&timestamp=1499827319559")
        self.assertEqual(sign(secret, q), "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")

    def test_would_take_error(self):
        self.assertTrue(BinanceError(400, -2010, "Order would immediately match and take.").would_take)
        self.assertFalse(BinanceError(400, -2010, "Account has insufficient balance").would_take)


class TestFilters(unittest.TestCase):
    def test_rounds_down_to_exchange_steps(self):
        f = Filters("0.01", "0.00001", "0.00001", "1")
        self.assertEqual(f.price(80245.587), Decimal("80245.58"))
        self.assertEqual(f.qty(0.0123456), Decimal("0.01234"))
        self.assertFalse(f.ok(Decimal("0.00001"), Decimal("50000")))  # $0.50 < $1 minimum
        self.assertTrue(f.ok(Decimal("0.0001"), Decimal("50000")))


class TestStrategy(unittest.TestCase):
    def test_bot_trades_the_same_days_as_the_backtest(self):
        """Feeding decide() one candle at a time reproduces bench/exits.py exactly."""
        d = synthetic_candles()
        e, x = bench_signals(d, "di", di_len=38)
        di = adx(d, 38, 9)
        holding, bot_e, bot_x = False, [], []
        armed = not di.di_plus.iloc[0] > di.di_minus.iloc[0]  # same starting point as the backtest
        for i in range(1, len(d)):
            sig = {"trend_up": bool(di.di_plus.iloc[i] > di.di_minus.iloc[i])}
            action, armed = strategy.decide(sig, holding, armed)
            if action == "BUY":
                holding = True
                bot_e.append(d.index[i])
            elif action == "SELL":
                holding = False
                bot_x.append(d.index[i])
        self.assertGreater(len(bot_e), 5)
        self.assertEqual(bot_e, list(e[e].index))
        self.assertEqual(bot_x, list(x[x].index))

    def test_waits_for_fresh_cross_when_started_mid_trend(self):
        up, down = {"trend_up": True}, {"trend_up": False}
        self.assertEqual(strategy.decide(up, False, False), (None, False))
        self.assertEqual(strategy.decide(down, False, False), (None, True))
        self.assertEqual(strategy.decide(up, False, True), ("BUY", True))
        self.assertEqual(strategy.decide(down, True, True), ("SELL", True))


class TestPaperExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name)
        self.store = Store(self.cfg.db_path)
        self.filters = Filters("0.01", "0.00001", "0.00001", "1")

    def tearDown(self):
        self.tmp.cleanup()

    def broker(self, books):
        self.clock = Clock()
        return PaperBroker(FakeClient(books, clock=self.clock), "BTCUSDT", self.store, start_usd=1000)

    def run_exec(self, side, budget, b):
        return execute(side, Decimal(budget), b, self.filters, self.store, self.cfg, self.clock, self.clock.sleep)

    def test_maker_buy_fills_at_bid_with_no_fee(self):
        b = self.broker([(0, "100.00", "100.01"),   # post at 100.00
                         (5, "99.99", "100.00")])   # sellers trade through: filled
        f = self.run_exec("BUY", 1000, b)
        self.assertEqual(f.base, Decimal("10.00000"))
        self.assertEqual(f.avg_price, 100.0)
        self.assertEqual(f.fee_usd, 0.0)
        quote, base = b.balances()
        self.assertEqual((quote, base), (Decimal(0), Decimal(10)))

    def test_reposts_when_market_moves_away(self):
        b = self.broker([(0, "100.00", "100.01"),    # post at 100.00
                         (5, "100.50", "100.51"),    # bid moves up: cancel, re-post at 100.50
                         (25, "100.40", "100.50")])  # sellers trade through: filled at 100.50
        f = self.run_exec("BUY", 1000, b)
        orders = self.store.rows("SELECT price, status FROM orders ORDER BY id")
        self.assertEqual([o["status"] for o in orders], ["CANCELED", "FILLED"])
        self.assertEqual(orders[1]["price"], 100.5)
        self.assertEqual(f.avg_price, 100.5)

    def test_taker_fallback_after_timeout(self):
        b = self.broker([(0, "100.00", "100.01")])  # never trades through
        f = self.run_exec("BUY", 1000, b)
        types = [o["type"] for o in self.store.rows("SELECT type FROM orders ORDER BY id")]
        self.assertEqual(types[-1], "LIMIT_IOC")
        self.assertAlmostEqual(f.avg_price, 100.01)
        self.assertAlmostEqual(f.fee_usd, float(f.quote) * 0.0002, places=6)

    def test_no_taker_fallback_when_disabled(self):
        self.cfg.taker_fallback = False
        b = self.broker([(0, "100.00", "100.01")])
        f = self.run_exec("BUY", 1000, b)
        self.assertEqual(f.base, 0)
        self.assertEqual(b.balances()[0], Decimal(1000))

    def test_maker_sell(self):
        self.store.set("paper_quote", "0")
        self.store.set("paper_base", "2")
        b = self.broker([(0, "99.99", "100.00"),    # post sell at 100.00
                         (5, "100.00", "100.01")])  # buyers lift through: filled
        f = self.run_exec("SELL", 2, b)
        self.assertEqual(f.base, Decimal(2))
        self.assertEqual(b.balances(), (Decimal("200.00"), Decimal(0)))


class TestRunOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, join_open_trend=True)
        self.store = Store(self.cfg.db_path)
        d = synthetic_candles()
        di = adx(d, 38, 9)
        up = di.di_plus > di.di_minus
        # cut history at a day DI+ is above DI-, so the bot should buy
        self.candles = d.loc[: up[up].index[-1]]
        self.clock = Clock()
        self.client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], self.candles, self.clock)
        self.broker = PaperBroker(self.client, "BTCUSDT", self.store, start_usd=1000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_buys_then_holds_on_rerun(self):
        import bot.execution as ex
        orig = ex.time
        try:
            ex.time = type("T", (), {"monotonic": staticmethod(self.clock), "sleep": staticmethod(self.clock.sleep)})
            sig, action, _ = run.run_once(self.cfg, self.client, self.broker, self.store)
            self.assertEqual(action, "BUY")
            self.assertIsNotNone(self.store.current_trade())
            sig, action, _ = run.run_once(self.cfg, self.client, self.broker, self.store)
            self.assertIsNone(action)
        finally:
            ex.time = orig
        rows = self.store.rows("SELECT action FROM signals")
        self.assertEqual(rows, [{"action": "BUY"}])  # re-run did not overwrite the BUY
        self.assertIn("Holding BTC", self.store.get("last_run")["summary"])

    def test_kill_switch_blocks_orders(self):
        self.cfg.stop_file.touch()
        sig, action, note = run.run_once(self.cfg, self.client, self.broker, self.store)
        self.assertIsNone(action)
        self.assertIn("kill switch", note)
        self.assertEqual(self.store.rows("SELECT * FROM orders"), [])
        self.assertIsNone(self.store.current_trade())

    def test_first_run_mid_trend_waits_by_default(self):
        self.cfg.join_open_trend = False
        sig, action, _ = run.run_once(self.cfg, self.client, self.broker, self.store)
        self.assertIsNone(action)
        self.assertIn("waiting for a fresh cross", self.store.get("last_run")["summary"])


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        web._chart_cache.update(t=0, data=None)

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self, password=""):
        cfg = make_cfg(self.tmp.name, dashboard_password=password)
        srv = web.make_server(cfg, port=0, client=FakeClient(candles=synthetic_candles()))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def test_page_and_status(self):
        url = self.serve()
        self.assertIn(b"Cryptonomics Bot", urllib.request.urlopen(url + "/").read())
        s = json.loads(urllib.request.urlopen(url + "/api/status").read())
        self.assertEqual(s["mode"], "paper")
        self.assertEqual(len(s["chart"]["dates"]), 365)

    def test_password(self):
        url = self.serve(password="hunter2")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url + "/api/status")
        self.assertEqual(e.exception.code, 401)
        req = urllib.request.Request(url + "/api/status", headers={
            "Authorization": "Basic " + base64.b64encode(b"joe:hunter2").decode()})
        self.assertEqual(urllib.request.urlopen(req).status, 200)


if __name__ == "__main__":
    unittest.main()
