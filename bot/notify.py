"""Alerts by email and/or ntfy push notifications.

Email: BOT_ALERT_EMAIL plus an SMTP login (BOT_SMTP_USER / BOT_SMTP_PASSWORD;
Gmail and Google Workspace by default, which need an app password).
ntfy (https://ntfy.sh, free app, no account): BOT_NTFY_TOPIC. Anyone who knows
the topic name can read those alerts, so use a long random one.
Either channel can be on, or both. An alert that can't be sent is logged and
never stops the bot.
"""
import logging
import smtplib
from email.message import EmailMessage

import requests

log = logging.getLogger("bot")


def channels(cfg):
    out = []
    if cfg.alert_email and cfg.smtp_user and cfg.smtp_password:
        out.append("email")
    if cfg.ntfy_topic:
        out.append("ntfy")
    return out


def send(cfg, title, message, priority="default", tags=()):
    """Send one alert on every configured channel; True if any of them delivered it."""
    ok = False
    if "email" in channels(cfg):
        ok |= _email(cfg, title, message, priority)
    if "ntfy" in channels(cfg):
        ok |= _ntfy(cfg, title, message, priority, tags)
    return ok


def _email(cfg, title, message, priority):
    msg = EmailMessage()
    msg["Subject"] = ("[!] " if priority == "high" else "") + "Cryptonomics: " + title
    msg["From"] = cfg.smtp_user
    msg["To"] = cfg.alert_email
    msg.set_content(message + "\n\n-- your Cryptonomics bot")
    try:
        if cfg.smtp_port == 465:
            server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20)
            server.starttls()
        with server:
            server.login(cfg.smtp_user, cfg.smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        log.warning("email alert not sent: %s", e)
        return False


def _ntfy(cfg, title, message, priority, tags):
    try:
        r = requests.post(f"{cfg.ntfy_server.rstrip('/')}/{cfg.ntfy_topic}", data=message.encode(), timeout=10,
                          headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority,
                                   "Tags": ",".join(tags)})
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("ntfy alert not sent: %s", e)
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
