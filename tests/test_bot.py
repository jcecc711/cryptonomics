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
from bench.indicators import adx, ema
from bot import run, strategy, web
from bot.binanceus import BinanceError, Filters, sign
from bot.broker import PaperBroker
from bot.config import Config
from bot.execution import execute
from bot.store import Store


def synthetic_candles(n=900, seed=1, freq="D", vol=0.03):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(vol / 60, vol, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, vol * 0.7, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, vol * 0.7, n))
    idx = pd.date_range("2022-01-01", periods=n, freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def daily_from(h4):
    return h4.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})


D = Decimal


class FakeClient:
    """Stands in for BinanceUS: an order book scripted over (fake) time, fixed candles.

    books: [(from_second, bid, ask), ...]; the book shown is the latest entry
    whose time has been reached on `clock`.
    """

    def __init__(self, books=((0, "100.00", "100.01"),), candles=None, clock=None):
        self.books = [(t, D(b), D(a)) for t, b, a in books]
        # candles: one DataFrame (used for every interval) or {interval: DataFrame}
        self.candles, self.clock = candles, clock or (lambda: 0)

    def book(self, symbol):
        now = self.clock()
        return [(b, a) for t, b, a in self.books if t <= now][-1]

    def filters(self, symbol):
        return Filters("0.01", "0.00001", "0.00001", "1")

    def klines(self, symbol, interval="1d", limit=1000, end_time=None, closed_only=True):
        c = self.candles[interval] if isinstance(self.candles, dict) else self.candles
        if end_time is not None:
            c = c[c.index <= pd.Timestamp(end_time, unit="ms")]
        return c.tail(limit)

    def tickers_24h(self, symbols):
        return [{"symbol": s, "lastPrice": "100.0", "priceChangePercent": "1.5", "quoteVolume": "1000000",
                 "highPrice": "101", "lowPrice": "99"} for s in symbols]

    def tradable_symbols(self, quotes=("USDT", "USD", "USDC")):
        return [{"symbol": b + "USDT", "base": b, "quote": "USDT"} for b in ("BTC", "ETH", "SOL", "SUI")]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_cfg(tmp, **kw):
    kw.setdefault("timeframes", ("1d",))
    return Config(data_dir=Path(tmp), poll_sec=10, maker_timeout_min=1, **kw)


def ensemble_candles(n=3000):
    h4 = synthetic_candles(n=n, seed=3, freq="4h", vol=0.012)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return {"1h": synthetic_candles(n=1200, seed=5, freq="1h", vol=0.006),
            "4h": h4, "6h": h4.resample("6h").agg(agg).dropna(), "8h": h4.resample("8h").agg(agg),
            "12h": h4.resample("12h").agg(agg), "1d": daily_from(h4)}


class FakeTime:
    """Patch bot.execution's clock so order waits take no real time."""

    def __init__(self, clock):
        import bot.execution as ex
        self.ex, self.orig = ex, ex.time
        ex.time = type("T", (), {"monotonic": staticmethod(clock), "sleep": staticmethod(clock.sleep)})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.ex.time = self.orig


class TestSigning(unittest.TestCase):
    def test_matches_binance_docs_example(self):
        secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        q = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
             "&recvWindow=5000&timestamp=1499827319559")
        self.assertEqual(sign(secret, q), "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")

    def test_would_take_error(self):
        self.assertTrue(BinanceError(400, -2010, "Order would immediately match and take.").would_take)
        self.assertFalse(BinanceError(400, -2010, "Account has insufficient balance").would_take)


class TestSettingsFile(unittest.TestCase):
    def test_inline_comments_and_quotes(self):
        from bot.config import parse_dotenv
        env = parse_dotenv('A=1000000   # hard cap\n# full-line comment\nB="x # not a comment"\nC=\nD=a#b\n')
        self.assertEqual(env, {"A": "1000000", "B": "x # not a comment", "C": "", "D": "a#b"})

    def test_env_example_loads_cleanly(self):
        """The shipped .env.example, copied to .env as the setup guide says, must load."""
        import os
        from bot import config
        env = config.parse_dotenv((config.ROOT / ".env.example").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            env["BOT_DATA_DIR"] = tmp
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                cfg = config.load()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        self.assertEqual(cfg.mode, "paper")
        self.assertEqual(cfg.timeframes, ("4h", "8h", "12h", "1d"))
        self.assertEqual(cfg.max_order_usd, 1_000_000)
        self.assertEqual(cfg.maker_timeout_min, 10)


class TestKlines(unittest.TestCase):
    def test_drops_the_still_forming_candle_for_every_interval_the_bot_uses(self):
        from bot import binanceus

        for interval in ("1d", "4h"):
            ms = binanceus.INTERVAL_MS[interval]
            now = pd.Timestamp.now("UTC").tz_localize(None).floor(pd.Timedelta(milliseconds=ms))
            opens = [now - pd.Timedelta(milliseconds=ms * k) for k in (2, 1, 0)]  # last one still forming
            rows = [[int(t.timestamp() * 1000), "1", "2", "0.5", "1.5", "10"] for t in opens]
            client = binanceus.BinanceUS()
            client._request = lambda *a, **k: rows
            self.assertEqual(list(client.klines("BTCUSDT", interval).index), opens[:2])


class TestFilters(unittest.TestCase):
    def test_rounds_down_to_exchange_steps(self):
        f = Filters("0.01", "0.00001", "0.00001", "1")
        self.assertEqual(f.price(80245.587), Decimal("80245.58"))
        self.assertEqual(f.qty(0.0123456), Decimal("0.01234"))
        self.assertFalse(f.ok(Decimal("0.00001"), Decimal("50000")))  # $0.50 < $1 minimum
        self.assertTrue(f.ok(Decimal("0.0001"), Decimal("50000")))


class TestLiveBroker(unittest.TestCase):
    def test_cancels_only_its_own_leftover_orders(self):
        from bot.broker import LiveBroker

        class C:
            cancelled = []

            def open_orders(self, symbol):
                return [{"orderId": 1, "clientOrderId": "cnx123"}, {"orderId": 2, "clientOrderId": "web_manual"}]

            def cancel(self, symbol, oid):
                self.cancelled.append(oid)

            def get_order(self, symbol, oid):
                return {"orderId": oid, "status": "CANCELED", "executedQty": "0"}

        c = C()
        out = LiveBroker(c, "BTCUSDT", "BTC", "USDT").cancel_leftovers()
        self.assertEqual(c.cancelled, [1])
        self.assertEqual([o["orderId"] for o in out], [1])


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
        if hasattr(self, "store"):
            self.store.db.close()
        self.tmp.cleanup()

    def broker(self, books):
        self.clock = Clock()
        return PaperBroker(FakeClient(books, clock=self.clock), "BTCUSDT", self.store, start_usd=1000)

    def run_exec(self, side, budget, b):
        return execute(side, Decimal(budget), b, self.filters, self.store, self.cfg,
                       clock=self.clock, sleep=self.clock.sleep)

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
        self.store.set("paper.quote", "0")
        self.store.set("paper.base", "2")
        b = self.broker([(0, "99.99", "100.00"),    # post sell at 100.00
                         (5, "100.00", "100.01")])  # buyers lift through: filled
        f = self.run_exec("SELL", 2, b)
        self.assertEqual(f.base, Decimal(2))
        self.assertEqual(b.balances(), (Decimal("200.00"), Decimal(0)))


class TestRunOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, join_open_trend=True, paper_start_usd=1000)
        self.store = Store(self.cfg.db_path)
        d = synthetic_candles()
        di = adx(d, 38, 9)
        up = di.di_plus > di.di_minus
        # cut history at a day DI+ is above DI-, so the sleeve should buy
        self.candles = d.loc[: up[up].index[-1]]
        self.clock = Clock()
        self.client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], self.candles, self.clock)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def run_once(self):
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        with FakeTime(self.clock):
            return run.run_once(self.cfg, self.client, sleeves, self.store)

    def test_buys_then_holds_on_rerun(self):
        _, action, _ = self.run_once()["1d"]
        self.assertEqual(action, "BUY")
        self.assertIsNotNone(self.store.current_trade("1d"))
        _, action, _ = self.run_once()["1d"]
        self.assertIsNone(action)
        rows = self.store.rows("SELECT action FROM signals")
        self.assertEqual(rows, [{"action": "BUY"}])  # re-run did not overwrite the BUY
        self.assertIn("Holding", self.store.get("last_run")["sleeves"]["1d"]["summary"])

    def test_kill_switch_blocks_orders(self):
        self.cfg.stop_file.touch()
        _, action, note = self.run_once()["1d"]
        self.assertIsNone(action)
        self.assertIn("kill switch", note)
        self.assertEqual(self.store.rows("SELECT * FROM orders"), [])
        self.assertIsNone(self.store.current_trade("1d"))

    def test_first_run_mid_trend_waits_by_default(self):
        self.cfg.join_open_trend = False
        _, action, _ = self.run_once()["1d"]
        self.assertIsNone(action)
        self.assertIn("waiting for a fresh cross", self.store.get("last_run")["sleeves"]["1d"]["summary"])


class TestEnsemble(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, timeframes=("4h", "8h", "12h", "1d"), paper_start_usd=1000,
                            join_open_trend=True)
        self.store = Store(self.cfg.db_path)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_live_sleeves_split_one_balance_reading_equally(self):
        class Live:
            def balances(self):
                return Decimal("1000"), Decimal("0")

        live = Live()
        sleeves = [run.Sleeve(tf, 0.25, "live", live, self.store, self.cfg) for tf in self.cfg.timeframes]
        run.fund(sleeves)
        self.assertEqual([s.cash for s in sleeves], [Decimal("250.00")] * 4)
        run.fund(sleeves)  # funding happens once
        self.assertEqual(sleeves[0].cash, Decimal("250.00"))

    def test_paper_sleeves_get_equal_shares(self):
        _, sleeves, _ = run.make(self.cfg, FakeClient(), self.store)
        self.assertEqual([s.name for s in sleeves], ["4h", "8h", "12h", "1d"])
        self.assertEqual([s.broker.balances()[0] for s in sleeves], [Decimal("250.0")] * 4)

    def test_each_sleeve_reads_its_own_timeframe_and_stays_in_its_lane(self):
        candles = ensemble_candles()
        clock = Clock()
        client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], candles, clock)
        seen = []
        orig = client.klines
        client.klines = lambda symbol, interval="1d", limit=1000: seen.append(interval) or orig(symbol, interval, limit)
        _, sleeves, _ = run.make(self.cfg, client, self.store)
        with FakeTime(clock):
            results = run.run_once(self.cfg, client, sleeves, self.store)
        self.assertEqual(seen, ["4h", "8h", "12h", "1d"])
        for tf, (sig, action, _) in results.items():
            di = adx(candles[tf], 38, 9).iloc[-1]
            self.assertEqual(sig["readings"]["di_plus"], round(float(di.di_plus), 2))
            if action == "BUY":  # each sleeve spends only its own quarter (less at most a taker fee)
                row = self.store.rows("SELECT * FROM equity WHERE strategy=?", tf)[-1]
                self.assertAlmostEqual(row["cash"] + row["btc"] * 100.0, 250.0, delta=250 * 0.0003)
        self.assertTrue(any(a == "BUY" for _, a, _ in results.values()))
        rows = {r["strategy"]: r for r in self.store.rows("SELECT * FROM equity")}
        self.assertAlmostEqual(rows["total"]["value"], sum(rows[tf]["value"] for tf in self.cfg.timeframes))

    def test_paper_and_live_keep_separate_records(self):
        paper, live = make_cfg(self.tmp.name), make_cfg(self.tmp.name, mode="live")
        self.assertNotEqual(paper.db_path, live.db_path)

    def test_rejects_unknown_timeframes(self):
        import os
        from bot import config
        os.environ["BOT_TIMEFRAMES"] = "4h,5h"
        os.environ["BOT_DATA_DIR"] = self.tmp.name
        try:
            with self.assertRaises(SystemExit):
                config.load()
        finally:
            del os.environ["BOT_TIMEFRAMES"], os.environ["BOT_DATA_DIR"]


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        web._cache.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self, password=""):
        cfg = make_cfg(self.tmp.name, dashboard_password=password, timeframes=("4h", "8h", "12h", "1d"))
        self.candles = ensemble_candles()
        srv = web.make_server(cfg, port=0, client=FakeClient(candles=self.candles))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def get(self, url, expect=200):
        try:
            r = urllib.request.urlopen(url)
            body, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            body, code = e.read(), e.code
        self.assertEqual(code, expect, body[:200])
        return json.loads(body) if body[:1] in (b"{", b"[") else body

    def post(self, url, payload, expect=200):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req)
            body, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            body, code = e.read(), e.code
        self.assertEqual(code, expect, body[:200])
        return json.loads(body)

    def test_page_and_status(self):
        url = self.serve()
        self.assertIn(b"Cryptonomics", self.get(url + "/"))
        s = self.get(url + "/api/status")
        self.assertEqual([x["name"] for x in s["sleeves"]], ["4h", "8h", "12h", "1d"])
        self.assertEqual(s["mode"], "paper")
        self.assertEqual(s["settings"]["di_len"], 38)

    def test_candles_page_back_through_history(self):
        url = self.serve()
        latest = self.get(url + "/api/candles?symbol=BTCUSDT&tf=4h&limit=100")
        self.assertEqual(len(latest["t"]), 100)
        self.assertEqual(set(latest), {"symbol", "tf", "has_more", "t", "o", "h", "l", "c", "v"})
        end = latest["t"][0] * 1000 - 1
        older = self.get(url + f"/api/candles?symbol=BTCUSDT&tf=4h&limit=100&end={end}")
        self.assertLess(older["t"][-1], latest["t"][0])
        self.get(url + "/api/candles?symbol=NOPE&tf=4h", expect=400)
        self.get(url + "/api/candles?tf=5h", expect=400)

    def test_3h_candles_built_from_1h(self):
        url = self.serve()
        h1 = self.candles["1h"]
        r = self.get(url + "/api/candles?symbol=BTCUSDT&tf=3h&limit=50")
        self.assertTrue(all(t % 10800 == 0 for t in r["t"]))          # aligned to midnight UTC
        t0 = pd.Timestamp(r["t"][5], unit="s")
        grp = h1.loc[t0: t0 + pd.Timedelta(minutes=179)]
        self.assertEqual((r["o"][5], r["h"][5], r["l"][5], r["c"][5]),
                         (round(grp.open.iloc[0], 8), round(grp.high.max(), 8), round(grp.low.min(), 8), round(grp.close.iloc[-1], 8)))
        self.get(url + "/api/candles?symbol=BTCUSDT&tf=6h&limit=20")

    def test_watchlist_analysis_and_editing(self):
        url = self.serve()
        w = self.get(url + "/api/watchlist")
        self.assertEqual(w["timeframes"], ["4h", "8h", "12h", "1d"])
        btc = w["symbols"][0]
        self.assertTrue(btc["bot"])
        # the trend read matches the bot's own DI calculation
        di = adx(self.candles["4h"], 38, 9).iloc[-1]
        self.assertEqual(btc["tfs"]["4h"]["up"], bool(di.di_plus > di.di_minus))
        self.assertEqual(btc["score"], sum(v["up"] for v in btc["tfs"].values()))
        self.assertEqual(len(btc["spark"]), 42)
        self.assertEqual(self.post(url + "/api/watchlist", {"symbols": ["btcusdt", "SUIUSDT", "SUIUSDT"]}),
                         {"symbols": ["BTCUSDT", "SUIUSDT"]})
        self.assertEqual([r["symbol"] for r in self.get(url + "/api/watchlist")["symbols"]], ["BTCUSDT", "SUIUSDT"])
        self.post(url + "/api/watchlist", {"symbols": ["FAKEUSDT"]}, expect=400)
        self.post(url + "/api/watchlist", {"symbols": []}, expect=400)

    def test_price(self):
        url = self.serve()
        self.assertEqual(self.get(url + "/api/price?symbol=ETHUSDT")["price"], 100.0)

    def test_password_protects_everything(self):
        url = self.serve(password="hunter2")
        for path in ("/", "/api/status", "/api/candles?tf=4h", "/api/watchlist"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(url + path)
            self.assertEqual(e.exception.code, 401)
        req = urllib.request.Request(url + "/api/watchlist", data=b'{"symbols":["ETHUSDT"]}', method="POST")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        self.assertEqual(e.exception.code, 401)
        auth = {"Authorization": "Basic " + base64.b64encode(b"joe:hunter2").decode()}
        self.assertEqual(urllib.request.urlopen(urllib.request.Request(url + "/api/status", headers=auth)).status, 200)


if __name__ == "__main__":
    unittest.main()
