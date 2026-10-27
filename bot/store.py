"""SQLite record of the bot's state, signals, orders, trades and equity.

Rows are tagged with the strategy (sleeve) they belong to: "core" or "swing".
"""
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals (
    strategy TEXT, candle_time TEXT, close REAL, readings TEXT, action TEXT, note TEXT, created TEXT,
    PRIMARY KEY (strategy, candle_time));
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT, exchange_id TEXT, side TEXT, type TEXT,
    price REAL, qty REAL, status TEXT, filled_qty REAL DEFAULT 0, filled_quote REAL DEFAULT 0,
    commission_usd REAL DEFAULT 0, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT, entry_time TEXT, entry_price REAL, qty REAL,
    cost_usd REAL, exit_time TEXT, exit_price REAL, proceeds_usd REAL, ret REAL);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT, strategy TEXT, cash REAL, btc REAL, price REAL, value REAL,
    PRIMARY KEY (ts, strategy));
"""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # -- key/value state ---------------------------------------------------
    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, json.dumps(value)))

    # -- records -----------------------------------------------------------
    def record_signal(self, strategy, sig, action, note=""):
        with self.db:
            # Re-runs on the same candle refresh the row, but never overwrite the BUY/SELL it acted on.
            self.db.execute(
                "INSERT INTO signals VALUES (?,?,?,?,?,?,?) ON CONFLICT(strategy, candle_time) DO UPDATE SET"
                " action=excluded.action, note=excluded.note, created=excluded.created"
                " WHERE signals.action NOT IN ('BUY', 'SELL')",
                (strategy, str(sig["candle_time"]), sig["close"], json.dumps(sig["readings"]),
                 action, note, now()))

    def add_order(self, strategy, exchange_id, side, type_, price, qty, status):
        with self.db:
            cur = self.db.execute(
                "INSERT INTO orders (strategy, exchange_id, side, type, price, qty, status, created, updated)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (strategy, str(exchange_id), side, type_, float(price or 0), float(qty), status, now(), now()))
            return cur.lastrowid

    def update_order(self, row_id, status, filled_qty, filled_quote, commission_usd=0.0):
        with self.db:
            self.db.execute("UPDATE orders SET status=?, filled_qty=?, filled_quote=?, commission_usd=?,"
                            " updated=? WHERE id=?",
                            (status, float(filled_qty), float(filled_quote), float(commission_usd), now(), row_id))

    def open_trade(self, strategy, entry_price, qty, cost_usd):
        with self.db:
            self.db.execute("INSERT INTO trades (strategy, entry_time, entry_price, qty, cost_usd)"
                            " VALUES (?,?,?,?,?)", (strategy, now(), entry_price, qty, cost_usd))

    def current_trade(self, strategy):
        row = self.db.execute("SELECT * FROM trades WHERE strategy=? AND exit_time IS NULL"
                              " ORDER BY id DESC LIMIT 1", (strategy,)).fetchone()
        return dict(row) if row else None

    def close_trade(self, strategy, exit_price, proceeds_usd):
        t = self.current_trade(strategy)
        if not t:
            return
        with self.db:
            self.db.execute("UPDATE trades SET exit_time=?, exit_price=?, proceeds_usd=?, ret=? WHERE id=?",
                            (now(), exit_price, proceeds_usd, proceeds_usd / t["cost_usd"] - 1, t["id"]))

    def reduce_trade(self, strategy, sold_qty):
        """A sell only partly filled: keep the trade open with what's left."""
        t = self.current_trade(strategy)
        if t:
            left = t["qty"] - sold_qty
            with self.db:
                self.db.execute("UPDATE trades SET qty=?, cost_usd=? WHERE id=?",
                                (left, t["cost_usd"] * left / t["qty"], t["id"]))

    def snapshot(self, ts, strategy, cash, btc, price):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)",
                            (ts, strategy, float(cash), float(btc), float(price), float(cash + btc * price)))

    def rows(self, sql, *args):
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]
