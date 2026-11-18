"""Client accounts: separately managed paper accounts with their own investor logins.

Each client is a complete, separate account: its own folder and database under
data/clients/<id>/, its own sleeves and balances, trading the same strategy as
the main account. Nothing is pooled. Clients are paper-only for now.

    python -m bot.run client-add "Jane Smith" 250000     # prints her login once
    python -m bot.run client-list
    python -m bot.run client-password jane-smith          # new password

The registry (names, starting capital, password hashes) is data/clients.json.
Passwords are stored only as salted PBKDF2 hashes.
"""
import dataclasses
import hashlib
import hmac
import json
import logging
import re
import secrets
from datetime import datetime, timezone

import pandas as pd

from .store import Store

log = logging.getLogger("bot")
ITERATIONS = 200_000


# -- registry -------------------------------------------------------------------

def registry_path(cfg):
    return cfg.data_dir / "clients.json"


def load(cfg):
    p = registry_path(cfg)
    return json.loads(p.read_text()) if p.exists() else []


def save(cfg, clients):
    p = registry_path(cfg)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(clients, indent=2))
    tmp.chmod(0o600)
    tmp.replace(p)


def get(cfg, cid):
    return next((c for c in load(cfg) if c["id"] == cid), None)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def set_password(c, password=None):
    password = password or secrets.token_urlsafe(12)
    c["salt"] = secrets.token_hex(16)
    c["pw_hash"] = _hash(password, c["salt"])
    return password


def check_password(c, password):
    return bool(c and c.get("pw_hash")) and hmac.compare_digest(_hash(password, c["salt"]), c["pw_hash"])


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "client"


def add(cfg, name, start_usd, password=None):
    if cfg.mode != "paper":
        raise SystemExit("Client accounts are paper-only for now (BOT_MODE=paper).")
    start_usd = float(start_usd)
    if start_usd < 100:
        raise SystemExit("Starting capital must be at least $100.")
    clients = load(cfg)
    cid, n = slug(name), 2
    base = cid
    while any(c["id"] == cid for c in clients):
        cid, n = f"{base}-{n}", n + 1
    c = {"id": cid, "name": name, "start_usd": start_usd, "mode": "paper",
         "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    password = set_password(c, password)
    clients.append(c)
    save(cfg, clients)
    client_cfg(cfg, c).data_dir.mkdir(parents=True, exist_ok=True)
    return c, password


# -- each client's own account ------------------------------------------------------

def client_cfg(cfg, c):
    """The main settings, pointed at the client's own folder and capital. Alerts and the
    short tracker stay with the main account; the main kill switch stops clients too."""
    return dataclasses.replace(cfg, data_dir=cfg.data_dir / "clients" / c["id"], paper_start_usd=c["start_usd"],
                               mode="paper", ntfy_topic="", alert_email="", short_tracker=False,
                               stop_path=cfg.stop_file)


def run_all(cfg, client, make, run_once):
    """One run for every client account, each in its own database."""
    for c in load(cfg):
        ccfg = client_cfg(cfg, c)
        ccfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = Store(ccfg.db_path)
        try:
            _, sleeves, _ = make(ccfg, client, store)
            run_once(ccfg, client, sleeves, store)
        except Exception:
            log.exception("client %s: run failed", c["id"])
        finally:
            store.db.close()


# -- reporting -------------------------------------------------------------------

def balances(ccfg, store):
    names = [n for n, _, _ in ccfg.sleeve_specs()] + ["manual"]
    cash = sum(float(store.get(f"{n}.cash") or 0) for n in names)
    btc = sum(float(store.get(f"{n}.btc") or 0) for n in names)
    return cash, btc


def _value_at(eq, t, default):
    before = eq.loc[:t]
    return float(before.value.iloc[-1]) if len(before) else default


def report(cfg, c, price):
    """Everything the investor portal shows for one client, at the live price."""
    ccfg = client_cfg(cfg, c)
    store = Store(ccfg.db_path)
    try:
        rows = store.rows("SELECT ts, value, price, cash, btc FROM equity WHERE strategy='total' ORDER BY ts")
        trades = store.rows("SELECT strategy, entry_time, entry_price, qty, cost_usd, exit_time, exit_price, "
                            "proceeds_usd, ret FROM trades ORDER BY id DESC")
        cash, btc = balances(ccfg, store)
    finally:
        store.db.close()
    start = c["start_usd"]
    funded = cash + btc > 0
    value = cash + btc * price if funded else start
    now = pd.Timestamp.now("UTC").tz_localize(None)
    eq = pd.DataFrame(rows)
    if len(eq):
        eq.index = pd.to_datetime(eq.pop("ts"))
    inception = eq.index[0] if len(eq) else pd.Timestamp(c["created"])
    price0 = float(eq.price.iloc[0]) if len(eq) else price

    def period(label, t0):
        if t0 <= inception:
            v0, since = start, True
        else:
            v0, since = _value_at(eq, t0, start), False
        return {"label": label, "start": str(max(t0, inception).date()), "since_inception": since,
                "start_value": v0, "gain": value - v0, "ret": value / v0 - 1 if v0 else 0}

    q0 = pd.Timestamp(year=now.year, month=3 * ((now.month - 1) // 3) + 1, day=1)
    periods = [period("1 week", now - pd.Timedelta(days=7)), period("1 month", now - pd.Timedelta(days=30)),
               period("Month to date", pd.Timestamp(year=now.year, month=now.month, day=1)),
               period("Quarter to date", q0), period("Year to date", pd.Timestamp(year=now.year, month=1, day=1)),
               period("Since inception", inception)]
    months = []
    m = pd.Timestamp(year=inception.year, month=inception.month, day=1)
    while m <= now:
        nxt = m + pd.offsets.MonthBegin(1)
        v0 = start if m <= inception else _value_at(eq, m, start)
        v1 = value if nxt > now else _value_at(eq, nxt - pd.Timedelta(seconds=1), v0)
        months.append({"month": m.strftime("%Y-%m"), "start_value": v0, "end_value": v1, "gain": v1 - v0,
                       "ret": v1 / v0 - 1 if v0 else 0,
                       "trades": sum(1 for t in trades if (t["exit_time"] or t["entry_time"])[:7] == m.strftime("%Y-%m"))})
        m = nxt
    daily = eq[["value", "price"]].resample("1D").last().dropna() if len(eq) else pd.DataFrame(columns=["value", "price"])
    return {
        "id": c["id"], "name": c["name"], "mode": c.get("mode", "paper"), "inception": str(inception.date()),
        "start_usd": start, "value": value, "gain": value - start, "ret": value / start - 1,
        "cash": cash if funded else start, "btc": btc, "price": price, "in_btc": btc * price / value if value else 0,
        "btc_equiv": value / price, "btc_equiv_start": start / price0,
        "hold_value": start * price / price0, "hold_ret": price / price0 - 1,
        "periods": periods, "months": months[::-1], "trades": trades[:200],
        "open": [t for t in trades if not t["exit_time"]],
        # daily account value, and what the starting capital would be worth just holding BTC
        "equity": [{"t": str(t.date()), "v": round(float(r.value), 2), "hold": round(start * float(r.price) / price0, 2)}
                   for t, r in daily.iterrows()],
        "price0": price0,
    }


# -- monthly statement (printable HTML; "Save as PDF" from the browser) -------------

def statement(rep, month):
    from html import escape as h
    m = next((x for x in rep["months"] if x["month"] == month), None)
    if not m:
        raise ValueError(f"no statement for {month!r}")
    start, end = pd.Timestamp(month + "-01"), pd.Timestamp(month + "-01") + pd.offsets.MonthBegin(1)
    in_month = [t for t in rep["trades"] if start <= pd.Timestamp(t["exit_time"] or t["entry_time"]) < end]
    money = lambda x: ("-" if x < 0 else "") + f"${abs(x):,.2f}"
    pct = lambda x: f"{100 * x:+.2f}%"
    rows = "".join(
        f"<tr><td>{h(t['strategy'])}</td><td>{h(t['entry_time'][:16])}</td><td class=n>{money(t['entry_price'])}</td>"
        f"<td class=n>{t['qty']:.6f}</td><td>{h((t['exit_time'] or 'open')[:16])}</td>"
        f"<td class=n>{money(t['exit_price']) if t['exit_price'] else '–'}</td>"
        f"<td class=n>{pct(t['ret']) if t['ret'] is not None else '–'}</td></tr>" for t in in_month)
    label = start.strftime("%B %Y")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Statement {h(label)} - {h(rep['name'])}</title>
<style>
body {{ font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; color: #141824; background: #fff; max-width: 820px; margin: 24px auto; padding: 0 16px; }}
h1 {{ font-size: 22px; margin: 0; }} .muted {{ color: #6b7385; }} .paper {{ display: inline-block; background: #fff4e5; color: #b25e00; border: 1px solid #f5c27a;
  border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0 22px; }} td, th {{ padding: 7px 8px; border-bottom: 1px solid #e6e8ef; text-align: left; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #6b7385; }} .n {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sum td:first-child {{ color: #6b7385; }} .sum td:last-child {{ text-align: right; font-weight: 700; }}
.top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; border-bottom: 3px solid #f7931a; padding-bottom: 12px; margin-bottom: 14px; }}
.print {{ margin: 8px 0 18px; }} @media print {{ .print {{ display: none; }} body {{ margin: 0; }} }}
</style></head><body>
<div class="top"><div><h1>Monthly statement</h1><div class="muted">{h(label)} · {h(rep['name'])} · account {h(rep['id'])}</div></div>
<div style="text-align:right"><b>Cryptonomics</b><br><span class="paper">SIMULATED (PAPER) ACCOUNT</span></div></div>
<button class="print" onclick="print()">Print / Save as PDF</button>
<table class="sum">
<tr><td>Value at start of month</td><td>{money(m['start_value'])}</td></tr>
<tr><td>Value at end of month{' (to date)' if m is rep['months'][0] else ''}</td><td>{money(m['end_value'])}</td></tr>
<tr><td>Gain / loss this month</td><td>{money(m['gain'])} ({pct(m['ret'])})</td></tr>
<tr><td>Deposits / withdrawals</td><td>$0.00</td></tr>
<tr><td>Fees</td><td>$0.00</td></tr>
<tr><td>Starting capital ({h(rep['inception'])})</td><td>{money(rep['start_usd'])}</td></tr>
<tr><td>Gain / loss since inception (to date)</td><td>{money(rep['gain'])} ({pct(rep['ret'])})</td></tr>
<tr><td>Holding BTC over the same time (for comparison)</td><td>{pct(rep['hold_ret'])}</td></tr>
</table>
<h3>Trades this month</h3>
<table><thead><tr><th>Sleeve</th><th>Bought (UTC)</th><th class=n>Price</th><th class=n>BTC</th><th>Sold (UTC)</th><th class=n>Price</th><th class=n>Return</th></tr></thead>
<tbody>{rows or '<tr><td colspan=7 class=muted>No trades this month.</td></tr>'}</tbody></table>
<p class="muted" style="font-size:12px">This is a simulated (paper) account: no real money is invested and the results are hypothetical.
Trades are simulated against the live Binance.US order book. Past or simulated performance does not guarantee future results.
Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC.</p>
</body></html>"""
