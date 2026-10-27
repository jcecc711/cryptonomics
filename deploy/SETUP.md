# Server setup

About 20 minutes. Any small Linux cloud server works (DigitalOcean, AWS
Lightsail, Vultr, Linode, Hetzner US): pick **Ubuntu 24.04**, **1 GB RAM**,
a **US region**, and make sure it has a fixed public IPv4 address (most do by
default). That address is what you'll lock your Binance.US API key to.

## 1. Create the server and log in

Create it in the provider's dashboard with your SSH key, then from your computer:

    ssh root@YOUR_SERVER_IP

## 2. Install and create a user for the bot

    apt update && apt install -y python3-pandas python3-numpy python3-requests git
    adduser --disabled-password --gecos "" crypto
    mkdir -p /opt/cryptonomics && chown crypto:crypto /opt/cryptonomics

## 3. Get the code

The repository is private, so the server needs read access. Easiest: on GitHub
open the repo → Settings → Deploy keys → Add deploy key (read-only), using the
public key printed here:

    sudo -u crypto ssh-keygen -t ed25519 -N "" -f /home/crypto/.ssh/id_ed25519
    cat /home/crypto/.ssh/id_ed25519.pub

Then:

    sudo -u crypto git clone -b claude/relaxed-curie-7wi1ki git@github.com:jcecc711/cryptonomics.git /opt/cryptonomics

## 4. Configure

    cd /opt/cryptonomics
    sudo -u crypto cp .env.example .env
    sudo -u crypto nano .env

For the practice run, leave `BOT_MODE=paper`. Your API keys can go in now or
later. Lock the file down:

    chmod 600 .env

Check it works (this places practice orders only):

    sudo -u crypto python3 -m unittest discover -s tests -t .
    sudo -u crypto python3 -m bot.run once

## 5. Run it all the time

    cp deploy/cryptonomics-bot.service deploy/cryptonomics-web.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now cryptonomics-bot cryptonomics-web
    systemctl status cryptonomics-bot          # should say "active (running)"
    journalctl -u cryptonomics-bot -f          # live log; Ctrl+C to stop watching

Both restart automatically after a crash or a server reboot.

## 6. See the dashboard

The dashboard only listens on the server itself, so nobody else can reach it.
To view it, open a tunnel from your computer:

    ssh -L 8080:127.0.0.1:8080 root@YOUR_SERVER_IP

and browse to http://localhost:8080 while that window is open.

## 7. Binance.US API key

On Binance.US: API Management → create a key with **trading enabled,
withdrawals disabled**, and **restrict access to YOUR_SERVER_IP**. Put the key
and secret in `.env`, then `systemctl restart cryptonomics-bot`.

## 8. Going live (after the practice period)

In `.env` set `BOT_MODE=live` and a small `BOT_MAX_ORDER_USD` (e.g. 100) for
the first trades, fund the account with USDT, then:

    systemctl restart cryptonomics-bot cryptonomics-web

Live and practice use separate balances: on its first live run each sleeve
takes 25% of the account's free USDT.

## Everyday commands

    touch /opt/cryptonomics/data/STOP          # emergency stop: no new orders
    rm /opt/cryptonomics/data/STOP             # resume
    sudo -u crypto python3 -m bot.run status   # one-line summary per sleeve
    cd /opt/cryptonomics && sudo -u crypto git pull && systemctl restart cryptonomics-bot cryptonomics-web   # update
