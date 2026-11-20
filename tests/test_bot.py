"""Bot tests. No network needed.

    python -m unittest discover -s tests -v
"""
import base64
import json
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))  # bench uses flat imports
from bench.exits import signals as bench_signals
from bench.indicators import adx, ema
from bot import notify, run, shorts, stocks, strategy, web
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

    def wallet(self):
        return {"USDT": (D("1500.5"), D("0")), "BTC": (D("0.01"), D("0.005")), "XYZ": (D("3"), D("0"))}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_cfg(tmp, **kw):
    kw.setdefault("timeframes", ("1d",))
    kw.setdefault("strategies", ("di",))
    kw.setdefault("price_guard", 0.0)   # the fake book (~100) and the synthetic candles live at different prices
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
        self.assertEqual(cfg.strategies, ("di", "adx"))
        self.assertEqual(cfg.bear_filter, "halving")
        self.assertEqual(cfg.adx_level, 14)


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


class TestAdxTrigger(unittest.TestCase):
    def test_bot_trades_the_same_candles_as_the_backtest(self):
        """Feeding one closed candle per run reproduces bench/adx_hybrid.py's ADX trigger:
        buy when ADX rises through 14 with DI+ on top, sell when DI+ crosses below DI-."""
        h4 = synthetic_candles(n=1500, seed=11, freq="4h", vol=0.012)
        d = adx(h4, 38, 9)
        up = d.di_plus > d.di_minus
        trig = (d.adx > 14) & (d.adx.shift() <= 14) & up
        ref_e, ref_x, holding = [], [], False
        for i in range(300, len(h4)):
            if not holding and trig.iloc[i]:
                holding = True
                ref_e.append(h4.index[i])
            elif holding and not up.iloc[i]:
                holding = False
                ref_x.append(h4.index[i])

        bot_e, bot_x, holding, state = [], [], False, {"last": str(h4.index[299])}
        for i in range(300, len(h4)):
            sig = strategy.evaluate_adx(h4.iloc[: i + 1], 38, 9, 14)
            action, state = strategy.decide_adx(sig, holding, state)
            if action == "BUY":
                holding = True
                bot_e.append(h4.index[i])
            elif action == "SELL":
                holding = False
                bot_x.append(h4.index[i])
        self.assertGreater(len(ref_e), 5)
        self.assertEqual(bot_e, ref_e)
        self.assertEqual(bot_x, ref_x)

    def sig(self, triggers, trend_up=True):
        t = pd.date_range("2026-01-01", periods=len(triggers), freq="4h")
        return {"candle_time": t[-1], "trend_up": trend_up, "recent": pd.Series(triggers, index=t)}, t

    def test_catches_a_trigger_it_missed_while_offline(self):
        sig, t = self.sig([False, True, False, False])
        self.assertEqual(strategy.decide_adx(sig, False, {"last": str(t[0])})[0], "BUY")
        self.assertIsNone(strategy.decide_adx(sig, False, {"last": str(t[1])})[0])      # already processed
        sig, t = self.sig([False, True, False, False], trend_up=False)
        self.assertIsNone(strategy.decide_adx(sig, False, {"last": str(t[0])})[0])     # DI- took over since

    def test_first_run_only_acts_on_the_latest_candle(self):
        sig, t = self.sig([True, False, False])
        self.assertIsNone(strategy.decide_adx(sig, False, {})[0])

    def test_sells_when_di_minus_takes_over(self):
        sig, _ = self.sig([False], trend_up=False)
        self.assertEqual(strategy.decide_adx(sig, True, {})[0], "SELL")


class TestRegime(unittest.TestCase):
    def test_calendar_matches_past_cycles(self):
        from bot import regime
        # tops just before the window, bear years inside it, bottoms at its end
        for day, bear in [("2017-12-17", False), ("2018-06-01", True), ("2018-12-15", True), ("2019-03-01", False),
                          ("2021-11-10", True), ("2022-06-18", True), ("2022-11-21", False),
                          ("2025-10-06", False), ("2026-05-01", True), ("2026-11-06", False), ("2029-12-01", True)]:
            self.assertEqual(regime.calendar_bear(day), bear, day)

    def test_assess_is_json_ready_and_calls_bear_in_the_window(self):
        from bot import regime
        d = synthetic_candles(n=900)                         # 2022-01-01 ...
        r = regime.assess(d.loc[:"2022-06-30"])
        json.dumps(r)
        self.assertEqual((r["call"], r["phase"], r["calendar_bear"]), ("Bear", "Bear market", True))
        self.assertEqual(r["bear_window"], ["2021-11-02", "2022-11-07"])
        r = regime.assess(d.loc[:"2023-06-30"])
        self.assertFalse(r["calendar_bear"])
        self.assertEqual(r["bear_window"][0], "2025-10-12")
        self.assertEqual(len(r["checks"]), 4)


class TestBearFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, join_open_trend=True, paper_start_usd=1000)
        self.store = Store(self.cfg.db_path)
        self.d = synthetic_candles(n=1600)                   # 2022-01-01 -> mid 2026
        di = adx(self.d, 38, 9)
        self.up = di.di_plus > di.di_minus

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def run_at(self, day):
        clock = Clock()
        client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], self.d.loc[:day], clock)
        _, sleeves, _ = run.make(self.cfg, client, self.store)
        with FakeTime(clock):
            return run.run_once(self.cfg, client, sleeves, self.store)["1d"]

    def up_day(self, start, end):
        days = self.up.loc[start:end]
        return days[days].index[-1]

    def test_skips_buys_in_a_bear_window(self):
        _, action, note = self.run_at(self.up_day("2022-03-01", "2022-10-31"))
        self.assertIsNone(action)
        self.assertIn("bear filter", note)
        self.assertEqual(self.store.rows("SELECT * FROM orders"), [])
        self.assertEqual(self.store.get("regime")["call"], "Bear")

    def test_buys_outside_and_exits_when_a_bear_window_starts(self):
        _, action, _ = self.run_at(self.up_day("2023-01-01", "2023-12-31"))
        self.assertEqual(action, "BUY")
        # a later day with DI+ still on top, but inside the 2025-26 bear window
        _, action, note = self.run_at(self.up_day("2025-10-13", "2026-05-01"))
        self.assertEqual(action, "SELL")
        self.assertIn("bear filter: exiting", note)
        self.assertIsNone(self.store.current_trade("1d"))

    def test_filter_off_ignores_the_calendar(self):
        self.cfg.bear_filter = "off"
        _, action, _ = self.run_at(self.up_day("2022-03-01", "2022-10-31"))
        self.assertEqual(action, "BUY")


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


class TestProtection(TestRunOnce):
    def scaled_client(self, last_close):
        """A fake book at the candles' own price level (so the price guard sees a normal market)."""
        px = f"{last_close:.2f}"
        return FakeClient([(0, px, f"{last_close + 0.01:.2f}")], self.candles, self.clock)

    def test_fixed_stop_sells_on_a_close_under_entry_and_waits_for_a_fresh_cross(self):
        self.assertEqual(self.run_once()["1d"][1], "BUY")
        t = self.store.current_trade("1d")
        last = float(self.candles.close.iloc[-1])
        with self.store.db:   # pretend the entry was 10% above where the market closed (DI+ is still on top)
            self.store.db.execute("UPDATE trades SET entry_price=? WHERE id=?", (last / 0.90, t["id"]))
        _, action, note = self.run_once()["1d"]
        self.assertEqual(action, "SELL")
        self.assertIn("stop: 1h close", note)
        self.assertIsNone(self.store.current_trade("1d"))
        _, action, _ = self.run_once()["1d"]
        self.assertIsNone(action)                       # DI+ still on top, but no buying straight back
        self.assertFalse(self.store.get("1d.state"))    # waits for DI+ to cross again

    def test_stop_ignores_a_close_above_the_level_and_can_be_switched_off(self):
        self.run_once()
        t = self.store.current_trade("1d")
        last = float(self.candles.close.iloc[-1])
        with self.store.db:
            self.store.db.execute("UPDATE trades SET entry_price=? WHERE id=?", (last / 0.95, t["id"]))   # down 5%
        self.assertIsNone(self.run_once()["1d"][1])
        with self.store.db:
            self.store.db.execute("UPDATE trades SET entry_price=? WHERE id=?", (last / 0.5, t["id"]))
        self.cfg.stop_pct = 0
        self.assertIsNone(self.run_once()["1d"][1])     # off: only the DI exit sells

    def test_price_guard_pauses_trading_when_the_book_is_far_from_the_last_close(self):
        last = float(self.candles.close.iloc[-1])
        self.cfg.price_guard = 0.10
        self.client = FakeClient([(0, f"{last * 0.2:.2f}", f"{last * 0.2 + 0.01:.2f}")], self.candles, self.clock)   # -80% flash
        _, action, note = self.run_once()["1d"]
        self.assertIsNone(action)
        self.assertIn("price guard", note)
        self.assertEqual(self.store.rows("SELECT * FROM orders"), [])
        self.client = self.scaled_client(last)          # the book is back to normal: the buy goes ahead
        self.assertEqual(self.run_once()["1d"][1], "BUY")


class TestAlerts(TestRunOnce):
    def run_once(self):
        self.sent = getattr(self, "sent", [])
        def post(url, data, timeout, headers):
            self.sent.append((url, headers["Title"], data.decode(), headers["Priority"]))
            return unittest.mock.Mock(raise_for_status=lambda: None)
        with unittest.mock.patch("bot.notify.requests.post", post):
            return super().run_once()

    def test_alerts_on_a_fill_once_per_regime_and_not_without_a_topic(self):
        super().run_once()                                   # no topic: nothing sent, nothing breaks
        self.assertFalse(hasattr(self, "sent"))
        self.store.db.execute("DELETE FROM state")
        self.store.db.execute("DELETE FROM trades")
        self.store.db.execute("DELETE FROM signals")
        self.store.db.commit()
        self.cfg.ntfy_topic = "cnx-test"
        self.run_once()
        titles = [s[1] for s in self.sent]
        self.assertTrue(all(s[0] == "https://ntfy.sh/cnx-test" for s in self.sent))
        buy = [s for s in self.sent if s[1].startswith("BUY BTC - 1d DI cross (PAPER)")]
        self.assertEqual(len(buy), 1, titles)
        self.assertIn("bought", buy[0][2])
        self.assertEqual(buy[0][3], "high")
        self.assertEqual(sum(t.startswith("Market regime") for t in titles), 1)
        n = len(self.sent)
        self.run_once()                                      # holding, same regime: quiet
        self.assertEqual(len(self.sent), n, [s[1] for s in self.sent[n:]])

    def test_kill_switch_alerts_once(self):
        self.cfg.ntfy_topic = "cnx-test"
        self.cfg.stop_file.touch()
        self.run_once()
        self.run_once()
        warn = [s for s in self.sent if "needs a look" in s[1]]
        self.assertEqual(len(warn), 1)
        self.assertIn("kill switch", warn[0][2])

    def test_email_alerts(self):
        mails = []
        class FakeSMTP:
            def __init__(self, host, port, timeout):
                self.calls = [("connect", host, port)]
            def starttls(self):
                self.calls.append("starttls")
            def login(self, user, pw):
                self.calls.append(("login", user, pw))
            def send_message(self, msg):
                mails.append((msg, self.calls))
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        self.cfg.alert_email, self.cfg.smtp_user = "me@example.com", "bot@example.com"
        self.assertEqual(notify.channels(self.cfg), [])            # no password yet: email is off
        self.cfg.smtp_password = "app-password"
        self.assertEqual(notify.channels(self.cfg), ["email"])
        with unittest.mock.patch("bot.notify.smtplib.SMTP", FakeSMTP):
            self.run_once()
        self.assertFalse(getattr(self, "sent", []))                 # ntfy stays off without a topic
        buy = [m for m, _ in mails if "BUY BTC - 1d DI cross (PAPER)" in m["Subject"]]
        self.assertEqual(len(buy), 1, [m["Subject"] for m, _ in mails])
        self.assertTrue(buy[0]["Subject"].startswith("[!] Cryptonomics:"))
        self.assertEqual((buy[0]["To"], buy[0]["From"]), ("me@example.com", "bot@example.com"))
        self.assertIn("bought", buy[0].get_content())
        self.assertEqual(mails[0][1], [("connect", "smtp.gmail.com", 587), "starttls",
                                       ("login", "bot@example.com", "app-password")])

    def test_failed_send_does_not_stop_the_bot(self):
        self.cfg.ntfy_topic = "cnx-test"
        def boom(*a, **k):
            raise OSError("network down")
        with unittest.mock.patch("bot.notify.requests.post", boom), FakeTime(self.clock):
            _, sleeves, _ = run.make(self.cfg, self.client, self.store)
            _, action, _ = run.run_once(self.cfg, self.client, sleeves, self.store)["1d"]
        self.assertEqual(action, "BUY")


class TestShortTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name)
        self.store = Store(self.cfg.db_path)
        self.h4 = synthetic_candles(n=3000, seed=5, freq="4h", vol=0.012)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        self.h8 = self.h4.resample("8h").agg(agg).dropna()
        d4, d8 = adx(self.h4, 38, 9), adx(self.h8, 38, 9)
        self.dn4 = d4.di_minus > d4.di_plus
        dn8 = d8.di_minus > d8.di_plus
        dn8.index = dn8.index + pd.Timedelta("4h")
        self.both = self.dn4 & dn8.reindex(self.h4.index, method="ffill").fillna(False).astype(bool)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def step(self, t):
        c4 = self.h4.loc[:t]
        c8 = self.h8[self.h8.index + pd.Timedelta("8h") <= t + pd.Timedelta("4h")]   # closed 8h candles only
        price = float(c4.close.iloc[-1])
        return shorts.step(self.cfg, self.store, c4, c8, price, str(t + pd.Timedelta("4h")))

    def test_picks_up_an_open_short_then_closes_on_the_4h_turn(self):
        b = self.both.iloc[200:]
        starts = b & ~b.shift(fill_value=False)
        start = starts[starts].index[3]
        later = b.loc[start:]
        t = later.index[min(5, (~later).argmax() - 1)]             # a few candles into the streak
        event, st = self.step(t)
        self.assertEqual(event, "open")
        pos = st["open"]
        self.assertTrue(pos["picked_up"])
        nxt = self.h4.index[self.h4.index.get_loc(start) + 1]
        self.assertEqual((pd.Timestamp(pos["opened"]), pos["entry"], pos["qty"]),
                         (nxt, float(self.h4.open.loc[nxt]), 0.01))
        # stays short until the 4h DI+ crosses back above DI-
        up_after = self.dn4.loc[t:]
        t_up = up_after[~up_after].index[0]
        for tt in self.h4.loc[t:t_up].index[1:-1]:
            self.assertIsNone(self.step(tt)[0])
        event, st = self.step(t_up)
        self.assertEqual(event, "close")
        c = st["closed"][0]
        exit_px = float(self.h4.close.loc[t_up])
        self.assertAlmostEqual(c["gross"], 0.01 * (c["entry"] - exit_px), places=2)
        self.assertLess(c["net"], c["gross"])                         # fees and funding come off
        self.assertIsNone(st["open"])
        self.assertEqual(st["realized"], c["net"])

    def test_new_short_after_the_first_run_opens_at_the_live_price(self):
        first_up = self.both[~self.both].index[300]
        self.assertIsNone(self.step(first_up)[0])
        b = self.both.loc[first_up:]
        t = b[b].index[0]
        event, st = self.step(t)
        self.assertEqual(event, "open")
        self.assertFalse(st["open"]["picked_up"])
        self.assertEqual(st["open"]["entry"], float(self.h4.close.loc[t]))

    def test_run_once_tracks_and_alerts(self):
        b = self.both.iloc[300:]
        t = b[b].index[2]
        cfg = make_cfg(self.tmp.name, timeframes=("1d",), alert_email="me@example.com",
                       smtp_user="bot@example.com", smtp_password="x")
        candles = {"4h": self.h4.loc[:t], "8h": self.h8[self.h8.index + pd.Timedelta("8h") <= t + pd.Timedelta("4h")],
                   "1d": self.h4.loc[:t].resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                                              "close": "last", "volume": "sum"}).dropna()}
        sent = []
        with unittest.mock.patch("bot.notify.send", lambda cfg, title, msg, *a, **k: sent.append(title)):
            _, sleeves, _ = run.make(cfg, FakeClient(candles=candles), self.store)
            run.run_once(cfg, FakeClient(candles=candles), sleeves, self.store)
        st = self.store.get("short_tracker")
        self.assertIsNotNone(st["open"])
        self.assertTrue(any(s.startswith("SHORT 0.01 BTC (tracked, not placed)") for s in sent), sent)
        self.assertEqual(self.store.rows("SELECT * FROM orders"), [])     # nothing is ever placed


class TestManualPosition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, strategies=("di", "adx"), paper_start_usd=1000)
        self.store = Store(self.cfg.db_path)
        d = synthetic_candles()
        di = adx(d, 38, 9)
        down = di.di_plus < di.di_minus
        self.candles = d.loc[: down[down].index[-1]]          # DI- on top: the rules want no BTC
        self.clock = Clock()
        self.client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], self.candles, self.clock)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def run_once(self, sleeves):
        with FakeTime(self.clock):
            return run.run_once(self.cfg, self.client, sleeves, self.store)

    def total(self, price=100.0):
        snap = self.store.rows("SELECT value FROM equity WHERE strategy='total' ORDER BY ts DESC LIMIT 1")
        return snap[0]["value"]

    def test_buy_is_kept_by_the_rules_and_sell_returns_the_cash(self):
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        self.run_once(sleeves)                                # funds the two sleeves: 500 + 500
        with self.store.db:                                   # later runs share this second: keep the first apart
            self.store.db.execute("UPDATE equity SET ts='2026-01-01 00:00:00'")
        with FakeTime(self.clock):
            note = run.manual_buy(self.cfg, self.client, sleeves, self.store, "300")
        self.assertIn("bought", note)
        t = self.store.current_trade("manual")
        self.assertIsNotNone(t)
        cash = {s.name: float(s.cash) for s in sleeves}
        self.assertEqual((cash["1d"], cash["adx-1d"]), (350.0, 350.0))   # lent 150 each
        self.assertEqual(self.store.get("manual.from"), {"1d": 150.0, "adx-1d": 150.0})
        # the hourly run picks up the manual position and never sells it
        self.run_once(sleeves)
        self.run_once(sleeves)
        self.assertIsNotNone(self.store.current_trade("manual"))
        self.assertTrue(any(s.kind == "manual" for s in sleeves))
        self.assertAlmostEqual(self.total(), 1000, delta=1)   # account total includes the manual BTC
        self.assertIn("Manual position", self.store.get("last_run")["sleeves"]["manual"]["summary"])
        s = {x["name"]: x for x in web.status(self.cfg, self.store, threading.Lock())["sleeves"]}
        self.assertEqual((s["manual"]["share"], s["manual"]["start_value"]), (0.0, 300.0))
        self.assertEqual(s["1d"]["balance"]["cash"], 350.0)                 # live balances, not the hourly snapshot
        self.assertGreater(s["manual"]["balance"]["btc"], 2.9)
        self.assertAlmostEqual(s["1d"]["start_value"], 350.0)
        with FakeTime(self.clock):
            note = run.manual_sell(self.cfg, self.client, sleeves, self.store)
        self.assertIn("sold", note)
        self.assertIsNone(self.store.current_trade("manual"))
        self.assertIsNone(self.store.get("manual.cash"))
        back = {s.name: float(s.cash) for s in sleeves if s.kind != "manual"}
        self.assertAlmostEqual(back["1d"] + back["adx-1d"], 1000, delta=1)
        self.assertAlmostEqual(back["1d"], back["adx-1d"], delta=0.02)
        self.run_once(sleeves)
        self.assertFalse(any(s.kind == "manual" for s in sleeves))
        self.assertEqual([r["strategy"] for r in self.store.rows("SELECT strategy FROM trades")], ["manual"])

    def test_buy_all(self):
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        self.run_once(sleeves)
        with FakeTime(self.clock):
            run.manual_buy(self.cfg, self.client, sleeves, self.store, "all")
        self.assertEqual(self.store.get("manual.from"), {"1d": 500.0, "adx-1d": 500.0})
        self.assertTrue(all(float(s.cash) == 0 for s in sleeves if s.kind != "manual"))
        self.assertGreater(self.store.current_trade("manual")["qty"], 9.9)    # ~1000 / 100

    def test_stop_sells_on_a_close_below_and_not_above(self):
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        self.run_once(sleeves)
        with FakeTime(self.clock):
            run.manual_buy(self.cfg, self.client, sleeves, self.store, "all")
        last = float(self.candles.close.iloc[-1])            # the fake client serves these for 1h too
        self.store.set("manual.stop", last * 0.9)            # close is above the stop: keep
        self.run_once(sleeves)
        self.assertIsNotNone(self.store.current_trade("manual"))
        self.assertIn("stop at", self.store.get("last_run")["sleeves"]["manual"]["note"])
        s = {x["name"]: x for x in web.status(self.cfg, self.store, threading.Lock())["sleeves"]}
        self.assertAlmostEqual(s["manual"]["stop"], last * 0.9)
        sent = []
        self.store.set("manual.stop", last * 1.1)            # close is below the stop: sell
        with unittest.mock.patch("bot.notify.send", lambda cfg, title, msg, *a, **k: sent.append(title)):
            self.run_once(sleeves)
        self.assertIsNone(self.store.current_trade("manual"))
        self.assertIsNone(self.store.get("manual.stop"))
        self.assertFalse(any(x.kind == "manual" for x in sleeves))
        self.assertTrue(any("STOP" in t for t in sent), sent)
        self.assertAlmostEqual(sum(float(x.cash) for x in sleeves), 1000, delta=2)   # cash back with the sleeves
        self.run_once(sleeves)                                # and the next run is normal

    def test_portfolio_shows_the_buy_before_the_next_hourly_run(self):
        from bot import portfolio
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        self.run_once(sleeves)
        with FakeTime(self.clock):
            run.manual_buy(self.cfg, self.client, sleeves, self.store, "all")
        portfolio._cache.clear()
        r = portfolio.portfolio(self.client, self.store, self.cfg, self.cfg.symbol, False)
        acct = next(p for p in r["periods"] if p["key"] == "START")["account"]
        self.assertAlmostEqual(acct["held"], self.store.current_trade("manual")["qty"], places=6)
        self.assertLess(acct["cash"], 1)                    # all of it went into BTC

    def test_limits(self):
        _, sleeves, _ = run.make(self.cfg, self.client, self.store)
        self.run_once(sleeves)
        with self.assertRaises(SystemExit):
            run.manual_buy(self.cfg, self.client, sleeves, self.store, "5000")       # more than the cash


class FakeLiveExchange(FakeClient):
    """A signed-in exchange: real-account balances, LIMIT_MAKER / IOC orders, and a fixed amount of
    BTC for sale at the bid (so a maker-only order can fill partly)."""

    def __init__(self, candles, clock, usdt="2500", liquidity="0.02"):
        super().__init__([(0, "100.00", "100.01")], candles, clock)
        self.bal = {"USDT": D(usdt), "BTC": D(0)}
        self.liquidity, self.orders, self.ids = D(liquidity), {}, 0

    def balances(self):
        return dict(self.bal)

    def new_order(self, symbol, side, type_, qty, price=None, tif=None, client_id=None):
        self.ids += 1
        o = {"orderId": self.ids, "clientOrderId": client_id, "side": side, "type": type_, "tif": tif,
             "origQty": str(qty), "price": str(price), "status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0"}
        self.orders[self.ids] = o
        return dict(o)

    def get_order(self, symbol, oid):
        o = self.orders[oid]
        if o["status"] in ("NEW", "PARTIALLY_FILLED") and o["side"] == "BUY":
            want = D(o["origQty"]) - D(o["executedQty"])
            got = min(want, self.liquidity)
            if got > 0:
                self.liquidity -= got
                px = D(o["price"])
                o["executedQty"] = str(D(o["executedQty"]) + got)
                o["cummulativeQuoteQty"] = str(D(o["cummulativeQuoteQty"]) + got * px)
                self.bal["USDT"] -= got * px
                self.bal["BTC"] += got
                o["status"] = "FILLED" if D(o["executedQty"]) >= D(o["origQty"]) else "PARTIALLY_FILLED"
        return dict(o)

    def cancel(self, symbol, oid):
        o = self.orders[oid]
        if o["status"] not in ("FILLED",):
            o["status"] = "CANCELED"

    def open_orders(self, symbol):
        return []

    def my_trades(self, symbol, oid):
        return []


class TestLiveManualBuy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, mode="live", api_key="k", api_secret="s", strategies=("di", "adx"),
                            max_order_usd=320)
        self.store = Store(self.cfg.db_path)
        d = synthetic_candles()
        di = adx(d, 38, 9)
        down = di.di_plus < di.di_minus
        self.candles = d.loc[: down[down].index[-1]]           # the rules want no BTC
        self.clock = Clock()

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def buy(self, ex, **kw):
        with FakeTime(self.clock):
            _, sleeves, _ = run.make(self.cfg, ex, self.store)
            run.run_once(self.cfg, ex, sleeves, self.store)       # funds the sleeves from the real balance
            note = run.manual_buy(self.cfg, ex, sleeves, self.store, "all", **kw)
        return note, sleeves

    def test_maker_only_buy_with_all_the_cash_never_crosses_the_spread(self):
        ex = FakeLiveExchange(self.candles, self.clock, liquidity="100")
        note, sleeves = self.buy(ex)
        types = {o["type"] for o in ex.orders.values()}
        self.assertEqual(types, {"LIMIT_MAKER"})                 # no IOC / taker order
        self.assertTrue(all(D(o["price"]) == D("100.00") for o in ex.orders.values()))   # at the bid, not the ask
        t = self.store.current_trade("manual")
        self.assertAlmostEqual(t["qty"], 25.0, places=1)         # $2,500 at $100: the $320 sleeve cap didn't apply
        self.assertIn("bought", note)
        self.assertEqual(sum(float(s.cash) for s in sleeves if s.kind != "manual"), 0.0)

    def test_part_fill_returns_the_unspent_cash_to_the_sleeves(self):
        ex = FakeLiveExchange(self.candles, self.clock, liquidity="10")     # only $1,000 for sale at the bid
        note, sleeves = self.buy(ex, wait_min=5)
        self.assertAlmostEqual(self.store.current_trade("manual")["qty"], 10.0)
        self.assertIn("not spent went back", note)
        rule = [s for s in sleeves if s.kind != "manual"]
        self.assertAlmostEqual(sum(float(s.cash) for s in rule), 1500, delta=0.05)
        self.assertEqual([float(s.cash) for s in rule], [750.0, 750.0])     # two sleeves of 1,250, each lent 500
        self.assertEqual({o["type"] for o in ex.orders.values()}, {"LIMIT_MAKER"})
        lent = self.store.get("manual.from")
        self.assertAlmostEqual(sum(lent.values()), 1000, delta=0.05)


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
        # daily candles come first (for the regime check) and the 1d sleeve reuses them
        self.assertEqual(seen, ["1d", "4h", "8h", "12h"])
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

    def test_blend_makes_eight_equal_sleeves(self):
        self.cfg.strategies = ("di", "adx")
        _, sleeves, _ = run.make(self.cfg, FakeClient(), self.store)
        self.assertEqual([s.name for s in sleeves], ["4h", "8h", "12h", "1d", "adx-4h", "adx-8h", "adx-12h", "adx-1d"])
        self.assertEqual([s.kind for s in sleeves], ["di"] * 4 + ["adx"] * 4)
        self.assertEqual({s.tf for s in sleeves}, {"4h", "8h", "12h", "1d"})
        self.assertEqual([s.broker.balances()[0] for s in sleeves], [Decimal("125.0")] * 8)

    def test_blend_runs_and_total_adds_up(self):
        self.cfg.strategies = ("di", "adx")
        candles = ensemble_candles()
        clock = Clock()
        client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], candles, clock)
        _, sleeves, _ = run.make(self.cfg, client, self.store)
        with FakeTime(clock):
            results = run.run_once(self.cfg, client, sleeves, self.store)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(sig is not None for sig, _, _ in results.values()))
        self.assertIn("adx", results["adx-4h"][0]["readings"])
        rows = {r["strategy"]: r for r in self.store.rows("SELECT * FROM equity")}
        names = [s.name for s in sleeves]
        self.assertAlmostEqual(rows["total"]["value"], sum(rows[n]["value"] for n in names))
        self.assertAlmostEqual(rows["total"]["value"], 1000.0, delta=1000 * 0.0003)

    def test_rejects_unknown_strategies(self):
        import os
        from bot import config
        os.environ["BOT_STRATEGIES"] = "di,rsi"
        os.environ["BOT_DATA_DIR"] = self.tmp.name
        try:
            with self.assertRaises(SystemExit):
                config.load()
        finally:
            del os.environ["BOT_STRATEGIES"], os.environ["BOT_DATA_DIR"]

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


class FakeYahoo:
    """Answers like Yahoo's chart endpoint for one ticker: regular-session 1h, 15m, 1d, 1wk
    candles on weekdays, with the extra off-grid "latest trade" point Yahoo appends."""
    def __init__(self, ticker="AAPL", days=900, now=pd.Timestamp("2026-11-06 19:00", tz="UTC")):
        self.ticker, self.now = ticker, now
        days_ny = pd.bdate_range(end=now.tz_convert("America/New_York").date(), periods=days)
        rng = np.random.default_rng(3)
        rows = []
        for d in days_ny:
            for h in range(7):
                ts = pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30 + 60 * h)
                if ts <= now:
                    rows.append(ts.tz_convert("UTC"))
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(rows))))
        self.h1 = pd.DataFrame({"open": px * (1 + rng.normal(0, 0.001, len(px))), "close": px,
                                "volume": rng.integers(1000, 5000, len(px)).astype(float)}, index=pd.DatetimeIndex(rows))
        self.h1["high"] = self.h1[["open", "close"]].max(axis=1) * 1.002
        self.h1["low"] = self.h1[["open", "close"]].min(axis=1) * 0.998

    def frame(self, interval):
        if interval in ("60m", "15m"):
            return self.h1
        ny = self.h1.index.tz_convert("America/New_York")
        key = ny.normalize() if interval == "1d" else (ny.normalize() - pd.to_timedelta(ny.weekday, unit="D"))
        g = self.h1.groupby(key)
        df = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        if interval == "1d":
            df.index = df.index + pd.Timedelta(hours=9, minutes=30)
        return df.tz_convert("UTC") if df.index.tz else df

    def __call__(self, ticker, params):
        if ticker != self.ticker:
            raise stocks.StockError(f"unknown stock {ticker!r}")
        df = self.frame(params["interval"])
        df = df[(df.index >= pd.Timestamp(params["period1"], unit="s", tz="UTC")) &
                (df.index <= pd.Timestamp(params["period2"], unit="s", tz="UTC"))]
        ts = [int(x.timestamp()) for x in df.index]
        q = {k: df[k].tolist() for k in ("open", "high", "low", "close", "volume")}
        if ts:   # Yahoo's trailing point: the latest trade, off the candle grid
            ts.append(int(self.now.timestamp()) + 7)
            for k in ("open", "high", "low", "close"):
                q[k].append(q["close"][-1])
            q["volume"].append(0)
        return {"meta": {"longName": "Apple Inc.", "regularMarketPrice": q["close"][-1] if ts else None},
                "timestamp": ts, "indicators": {"quote": [q]}}


class TestStocks(unittest.TestCase):
    def setUp(self):
        self.y = FakeYahoo()
        patcher = unittest.mock.patch.object(stocks, "_get", self.y)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = self.y.now.timestamp()

    def test_4h_candles_stay_inside_each_session(self):
        df, more = stocks.candles("AAPL", "4h", limit=40, now=self.now)
        ny = df.index.tz_convert("America/New_York")
        self.assertEqual(sorted(set(ny.strftime("%H:%M"))), ["09:30", "13:30"])
        day = self.y.h1[self.y.h1.index.tz_convert("America/New_York").date == ny[-2].date()]
        first = day.iloc[:4]
        self.assertAlmostEqual(df.loc[df.index[-2], "high"], first.high.max())
        self.assertAlmostEqual(df.loc[df.index[-2], "close"], first.close.iloc[-1])
        self.assertTrue(more)

    def test_live_point_is_folded_and_closed_only_drops_the_forming_candle(self):
        h, _ = stocks.candles("AAPL", "1h", limit=50, now=self.now)
        self.assertEqual(h.index[-1], self.y.h1.index[-1])                 # no extra off-grid candle
        w, _ = stocks.candles("AAPL", "1w", limit=10, now=self.now)
        self.assertEqual(w.index[-1].weekday(), 0)                          # labelled by Monday's date
        wc, _ = stocks.candles("AAPL", "1w", limit=10, now=self.now, closed_only=True)
        self.assertEqual(wc.index[-1], w.index[-2])                        # this week is still trading
        d, _ = stocks.candles("AAPL", "1d", limit=10, now=self.now, closed_only=True)
        self.assertLess(d.index[-1].date(), pd.Timestamp(self.now, unit="s").date())

    def test_paging_back(self):
        latest, _ = stocks.candles("AAPL", "1d", limit=100, now=self.now)
        older, more = stocks.candles("AAPL", "1d", end=latest.index[0].timestamp() - 1, limit=100, now=self.now)
        self.assertEqual(len(older), 100)
        self.assertLess(older.index[-1], latest.index[0])

    def test_unknown_ticker(self):
        with self.assertRaises(stocks.StockError):
            stocks.quote("NOPE")
        self.assertFalse(stocks.is_ticker("BAD TICKER"))


class TestClients(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_cfg(self.tmp.name, join_open_trend=True, dashboard_password="admin-pw")
        d = synthetic_candles()
        di = adx(d, 38, 9)
        up = di.di_plus > di.di_minus
        self.candles = d.loc[: up[up].index[-1]]           # DI+ on top: the sleeves buy
        self.clock = Clock()
        self.client = FakeClient([(0, "100.00", "100.01"), (5, "99.99", "100.00")], self.candles, self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accounts_are_separate_and_report(self):
        from bot import clients
        a, pw_a = clients.add(self.cfg, "Ann Able", 250000)
        b, _ = clients.add(self.cfg, "Ann Able", 1000000)      # same name: its own id
        self.assertEqual((a["id"], b["id"]), ("ann-able", "ann-able-2"))
        self.assertTrue(clients.check_password(clients.get(self.cfg, "ann-able"), pw_a))
        self.assertFalse(clients.check_password(clients.get(self.cfg, "ann-able"), "wrong"))
        self.assertNotIn(pw_a, clients.registry_path(self.cfg).read_text())   # only a hash is stored
        with FakeTime(self.clock):
            clients.run_all(self.cfg, self.client, run.make, run.run_once)
        for c, start in ((a, 250000), (b, 1000000)):
            ccfg = clients.client_cfg(self.cfg, c)
            self.assertTrue(ccfg.db_path.exists())
            st = Store(ccfg.db_path)
            self.assertIsNotNone(st.current_trade("1d"))      # its own sleeve bought, with its own money
            self.assertAlmostEqual(st.rows("SELECT value FROM equity WHERE strategy='total'")[0]["value"], start, delta=start * 0.01)
            st.db.close()
            r = clients.report(self.cfg, c, 100.0)
            self.assertAlmostEqual(r["start_usd"], start)
            self.assertAlmostEqual(r["value"], start, delta=start * 0.01)
            self.assertGreater(r["in_btc"], 0.9)
            self.assertEqual([p["label"] for p in r["periods"]][-1], "Since inception")
            html = clients.statement(r, r["months"][0]["month"])
            self.assertIn("SIMULATED (PAPER) ACCOUNT", html)
            self.assertIn(c["name"], html)
        self.assertFalse(Path(self.cfg.db_path).exists())       # the main account was not touched
        self.cfg.mode = "live"
        with self.assertRaises(SystemExit):
            clients.add(self.cfg, "Live Person", 5000)          # paper only

    def test_client_login_sees_only_the_portal(self):
        from bot import clients
        c, pw = clients.add(self.cfg, "Bea Brown", 500000)
        clients.add(self.cfg, "Cal Cole", 250000)
        srv = web.make_server(self.cfg, port=0, client=self.client)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}"

        def get(path, user, password):
            auth = {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}
            import http.client
            con = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
            con.request("GET", path, headers=auth)
            r = con.getresponse()
            body = r.read()
            extra = con.sock.recv(1) if con.sock else b""      # a second response on the same connection
            con.close()
            self.assertEqual(extra, b"", f"more than one response for {path}")
            return r.status, body
        code, body = get("/", c["id"], pw)
        self.assertEqual(code, 200)
        self.assertIn(b"Investor Portal", body)                 # a client's home page is the portal
        code, body = get("/api/me", c["id"], pw)
        me = json.loads(body)
        self.assertEqual((code, me["name"], me["start_usd"]), (200, "Bea Brown", 500000))
        code, _ = get("/api/me?client=cal-cole", c["id"], pw)
        self.assertEqual(json.loads(_)["name"], "Bea Brown")     # ?client= is ignored for clients
        for path in ("/api/status", "/api/clients", "/api/candles?tf=4h", "/api/watchlist"):
            self.assertEqual(get(path, c["id"], pw)[0], 403, path)
        self.assertEqual(get("/api/me", c["id"], "wrong")[0], 401)
        self.assertEqual(get("/statement", c["id"], pw)[0], 200)
        # the admin sees everyone, and any client's portal
        code, body = get("/api/clients", "joe", "admin-pw")
        self.assertEqual([x["name"] for x in json.loads(body)["clients"]], ["Bea Brown", "Cal Cole"])
        self.assertEqual(json.loads(get("/api/me?client=cal-cole", "joe", "admin-pw")[1])["name"], "Cal Cole")
        self.assertIn(b"Cryptonomics", get("/", "joe", "admin-pw")[1])
        self.assertEqual(get("/api/me?client=nobody", "joe", "admin-pw")[0], 400)


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        web._cache.clear()
        patcher = unittest.mock.patch.object(stocks, "_get", FakeYahoo(now=pd.Timestamp.now(tz="UTC").floor("h")))
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_status_lists_both_strategies(self):
        cfg = make_cfg(self.tmp.name, timeframes=("4h", "1d"), strategies=("di", "adx"))
        srv = web.make_server(cfg, port=0, client=FakeClient(candles=ensemble_candles()))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        s = self.get(f"http://127.0.0.1:{srv.server_address[1]}/api/status")
        self.assertEqual([(x["name"], x["kind"], x["tf"], x["share"]) for x in s["sleeves"]],
                         [("4h", "di", "4h", 0.25), ("1d", "di", "1d", 0.25),
                          ("adx-4h", "adx", "4h", 0.25), ("adx-1d", "adx", "1d", 0.25)])
        self.assertEqual(s["settings"]["strategies"], ["di", "adx"])

    def test_page_and_status(self):
        url = self.serve()
        self.assertIn(b"Cryptonomics", self.get(url + "/"))
        s = self.get(url + "/api/status")
        self.assertEqual([x["name"] for x in s["sleeves"]], ["4h", "8h", "12h", "1d"])
        self.assertEqual(s["mode"], "paper")
        self.assertIn("start", s)                                  # first total reading (None before any run)
        self.assertEqual(s["settings"]["di_len"], 38)
        self.assertIn(s["regime"]["call"], ("Bull", "Neutral", "Caution", "Bear"))
        self.assertEqual(s["settings"]["bear_filter"], "halving")

    def test_candles_page_back_through_history(self):
        url = self.serve()
        latest = self.get(url + "/api/candles?symbol=BTCUSDT&tf=4h&limit=100")
        self.assertEqual(len(latest["t"]), 100)
        self.assertEqual(set(latest), {"symbol", "tf", "has_more", "t", "o", "h", "l", "c", "v", "kind"})
        end = latest["t"][0] * 1000 - 1
        older = self.get(url + f"/api/candles?symbol=BTCUSDT&tf=4h&limit=100&end={end}")
        self.assertLess(older["t"][-1], latest["t"][0])
        self.get(url + "/api/candles?symbol=NOPE&tf=4h", expect=400)            # neither a coin nor a stock
        self.get(url + "/api/candles?symbol=BAD!&tf=4h", expect=400)
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
        # US stocks: anything that isn't a Binance.US pair is checked as a stock ticker
        self.assertEqual(self.post(url + "/api/watchlist", {"symbols": ["BTCUSDT", "aapl", "$AAPL"]}),
                         {"symbols": ["BTCUSDT", "AAPL"]})
        aapl = self.get(url + "/api/watchlist")["symbols"][1]
        self.assertEqual((aapl["kind"], aapl["name"]), ("stock", "Apple Inc."))
        self.assertEqual(list(aapl["tfs"]), ["1h", "4h", "1d", "1w"])
        self.assertIsNotNone(aapl["price"])
        self.post(url + "/api/watchlist", {"symbols": ["MSFT"]}, expect=400)   # Yahoo doesn't know it here
        c = self.get(url + "/api/candles?symbol=AAPL&tf=4h&limit=30")
        self.assertEqual((c["kind"], len(c["t"])), ("stock", 30))
        self.assertEqual(self.get(url + "/api/price?symbol=AAPL")["symbol"], "AAPL")
        self.post(url + "/api/watchlist", {"symbols": []}, expect=400)

    def test_cash_since_for_buying_power(self):
        cfg = make_cfg(self.tmp.name, timeframes=("4h", "1d"))
        store = Store(cfg.db_path)
        self.addCleanup(store.db.close)
        store.snapshot("2026-11-05 16:05:00", "4h", 1250, 0, 76000)
        store.snapshot("2026-11-05 16:05:00", "1d", 1250, 0, 76000)
        store.open_trade("4h", 80000, 0.015, 1200)
        store.close_trade("4h", 70000, 1050)
        store.snapshot("2026-11-16 20:05:00", "4h", 1100, 0, 65000)
        s = {x["name"]: x for x in web.status(cfg, store, threading.Lock())["sleeves"]}
        self.assertEqual((s["4h"]["cash_since"]["price"], s["4h"]["cash_since"]["how"]), (70000, "sold"))
        self.assertEqual(s["1d"]["cash_since"], {"time": "2026-11-05 16:05:00", "price": 76000, "how": "started"})
        store.open_trade("1d", 65000, 0.019, 1250)
        s = {x["name"]: x for x in web.status(cfg, store, threading.Lock())["sleeves"]}
        self.assertIsNone(s["1d"]["cash_since"])                  # holding BTC: no buying-power reference

    def test_portfolio_periods(self):
        url = self.serve()
        r = self.get(url + "/api/portfolio?symbol=BTCUSDT")
        keys = [p["key"] for p in r["periods"]]
        self.assertEqual(keys[:3], ["1W", "1M", "YTD"])
        self.assertEqual(keys[-1], "2020")
        self.assertEqual(r["unit"], "BTC")
        for p in r["periods"]:
            s, h = p["strategy"], p["hold"]
            if s:   # stack = what the strategy's dollars buy at the end vs the same dollars at the start
                self.assertAlmostEqual(1 + s["stack"], (1 + s["ret"]) / (1 + h["ret"]), places=9)
                self.assertAlmostEqual(s["usd"], 10000 * s["ret"], places=6)
        self.assertTrue(any(p["strategy"] for p in r["periods"]))
        st = self.get(url + "/api/portfolio?symbol=AAPL")
        self.assertEqual((st["unit"], st["name"]), ("AAPL", "Apple Inc."))
        self.assertTrue(any(p["strategy"] for p in st["periods"]))
        self.get(url + "/api/portfolio?symbol=BAD!", expect=400)

    def test_wallet_is_read_and_valued_for_the_admin_only(self):
        url = self.serve()
        self.assertEqual(self.get(url + "/api/wallet"), {"configured": False})     # no API key: nothing to read
        web._cache.clear()
        cfg = make_cfg(self.tmp.name, api_key="k", api_secret="s", dashboard_password="pw")
        from bot import clients
        c, cpw = clients.add(cfg, "Dee Dunn", 100000)
        srv = web.make_server(cfg, port=0, client=FakeClient(candles=ensemble_candles()))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/wallet"
        auth = lambda u, p: {"Authorization": "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()}
        w = json.loads(urllib.request.urlopen(urllib.request.Request(base, headers=auth("joe", "pw"))).read())
        got = {a["asset"]: a for a in w["assets"]}
        self.assertEqual(got["USDT"]["value"], 1500.5)
        self.assertAlmostEqual(got["BTC"]["total"], 0.015)                 # free + in open orders
        self.assertAlmostEqual(got["BTC"]["value"], 1.5)                   # at the fake 100.0 price
        self.assertIsNone(got["XYZ"]["value"])                             # no market to price it
        self.assertAlmostEqual(w["total"], 1502.0)
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base, headers=auth(c["id"], cpw)))
        self.assertEqual(e.exception.code, 403)                            # clients never see it

    def test_lookalike_key_gets_a_clear_error(self):
        from bot.binanceus import BinanceUS
        with self.assertRaisesRegex(RuntimeError, "look-alike"):
            BinanceUS("k\u0435y", "secret").balances()                    # Cyrillic "e"

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
