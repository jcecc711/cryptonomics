"""Phone alerts through ntfy (https://ntfy.sh): free, no account.

Install the ntfy app on the phone, subscribe to a topic, and put the same topic
in .env as BOT_NTFY_TOPIC. Anyone who knows the topic name can read the alerts,
so use a long random one (python -m bot.run alert-test prints a suggestion).
An alert that can't be sent is logged and never stops the bot.
"""
import logging

import requests

log = logging.getLogger("bot")


def send(cfg, title, message, priority="default", tags=()):
    """Send one alert; returns True if ntfy accepted it. No topic set: does nothing."""
    if not cfg.ntfy_topic:
        return False
    try:
        r = requests.post(f"{cfg.ntfy_server.rstrip('/')}/{cfg.ntfy_topic}", data=message.encode(), timeout=10,
                          headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority,
                                   "Tags": ",".join(tags)})
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("alert not sent: %s", e)
        return False


def trade_alerts(cfg, results, price):
    """One alert per sleeve that acted (or wanted to) this run."""
    mode = "PAPER" if cfg.mode == "paper" else "LIVE"
    for name, (sig, action, note) in results.items():
        if not action:
            continue
        filled = bool(note) and ("bought" in note or "sold" in note)
        side = action
        strat = "ADX-14 trigger" if name.startswith("adx-") else "DI cross"
        tf = name.split("-")[-1]
        title = f"{side} BTC - {tf} {strat} ({mode})"
        body = note if note else side
        if not filled:
            title = f"{side} signal, no fill - {tf} {strat} ({mode})"
        send(cfg, title, f"{body}\nBTC ${price:,.0f}",
             priority="high" if filled else "default",
             tags=("chart_with_upwards_trend",) if side == "BUY" else ("chart_with_downwards_trend",))


def once_per_change(cfg, store, key, value, title, message, priority="default", tags=()):
    """Alert only when `value` differs from what was alerted last time (errors, regime)."""
    last = store.get(f"alerted.{key}")
    if value == last:
        return False
    store.set(f"alerted.{key}", value)
    if value is None:
        return False
    return send(cfg, title, message, priority, tags)
