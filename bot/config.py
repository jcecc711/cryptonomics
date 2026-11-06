"""Settings, read from environment variables (or a .env file in the repo root)."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_dotenv(text):
    """KEY=value lines; '#' starts a comment (whole line, or after a space in an unquoted value)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'") and v[0] in v[1:]:
            v = v[1:v.index(v[0], 1)]
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        out[k.strip()] = v
    return out


def load_dotenv(path=ROOT / ".env"):
    if path.exists():
        for k, v in parse_dotenv(path.read_text()).items():
            os.environ.setdefault(k, v)


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    mode: str = "paper"                 # "paper" or "live"
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    di_len: int = 38
    adx_smoothing: int = 9

    # The ensemble: each strategy runs on each of these timeframes, every
    # (strategy, timeframe) pair trading an equal share of the money ("sleeve").
    #   di   buy when DI+ crosses above DI-, sell when it crosses back below
    #   adx  buy when ADX rises through `adx_level` with DI+ on top, sell when DI+ crosses below DI-
    timeframes: tuple = ("4h", "8h", "12h", "1d")
    strategies: tuple = ("di", "adx")
    adx_level: float = 14.0
    # "halving": in the halving-cycle bear window (days 540-910 after a halving)
    # take no new longs and exit open ones. "off": ignore the regime.
    bear_filter: str = "halving"

    # Buy only on a fresh DI+ cross. If the bot starts while DI+ is already
    # above DI-, it waits for the next cross unless this is on.
    join_open_trend: bool = False
    max_order_usd: float = 1_000_000.0  # hard cap per buy, per sleeve

    # Execution
    maker_timeout_min: float = 10.0     # how long to work a maker-only order
    poll_sec: float = 10.0
    taker_fallback: bool = True         # cross the spread if the maker order hasn't filled
    max_slippage: float = 0.002         # taker fallback limit, 0.2% through the book

    # Paper trading
    paper_start_usd: float = 10_000.0
    paper_taker_fee: float = 0.0002
    paper_maker_fee: float = 0.0

    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    data_dir: Path = ROOT / "data"

    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    dashboard_password: str = field(default="", repr=False)

    def sleeve_specs(self):
        """[(name, kind, timeframe)]; DI sleeves are named by timeframe, ADX ones 'adx-<tf>'."""
        return [(tf if kind == "di" else f"{kind}-{tf}", kind, tf) for kind in self.strategies for tf in self.timeframes]

    @property
    def db_path(self):
        """Paper and live keep separate records, so switching to live starts clean."""
        return self.data_dir / f"bot_{self.mode}.sqlite"

    @property
    def stop_file(self):
        """Create this file to stop all new orders (kill switch)."""
        return self.data_dir / "STOP"


def load():
    load_dotenv()
    e = os.environ.get
    cfg = Config(
        mode=e("BOT_MODE", "paper").lower(),
        symbol=e("BOT_SYMBOL", "BTCUSDT"),
        base_asset=e("BOT_BASE", "BTC"),
        quote_asset=e("BOT_QUOTE", "USDT"),
        di_len=int(e("BOT_DI_LEN", 38)),
        adx_smoothing=int(e("BOT_ADX_SMOOTHING", 9)),
        timeframes=tuple(t.strip() for t in e("BOT_TIMEFRAMES", "4h,8h,12h,1d").split(",") if t.strip()),
        strategies=tuple(t.strip().lower() for t in e("BOT_STRATEGIES", "di,adx").split(",") if t.strip()),
        adx_level=float(e("BOT_ADX_LEVEL", 14)),
        bear_filter=e("BOT_BEAR_FILTER", "halving").lower(),
        join_open_trend=_bool("BOT_JOIN_OPEN_TREND", False),
        max_order_usd=float(e("BOT_MAX_ORDER_USD", 1_000_000)),
        maker_timeout_min=float(e("BOT_MAKER_TIMEOUT_MIN", 10)),
        poll_sec=float(e("BOT_POLL_SEC", 10)),
        taker_fallback=_bool("BOT_TAKER_FALLBACK", True),
        max_slippage=float(e("BOT_MAX_SLIPPAGE", 0.002)),
        paper_start_usd=float(e("BOT_PAPER_START_USD", 10_000)),
        api_key=e("BINANCEUS_API_KEY", ""),
        api_secret=e("BINANCEUS_API_SECRET", ""),
        data_dir=Path(e("BOT_DATA_DIR", ROOT / "data")),
        dashboard_host=e("DASHBOARD_HOST", "127.0.0.1"),
        dashboard_port=int(e("DASHBOARD_PORT", 8080)),
        dashboard_password=e("DASHBOARD_PASSWORD", ""),
    )
    if cfg.mode not in ("paper", "live"):
        raise SystemExit(f"BOT_MODE must be 'paper' or 'live', got {cfg.mode!r}")
    if cfg.mode == "live" and not (cfg.api_key and cfg.api_secret):
        raise SystemExit("Live mode needs BINANCEUS_API_KEY and BINANCEUS_API_SECRET")
    from .binanceus import INTERVAL_MS
    bad = [t for t in cfg.timeframes if t not in INTERVAL_MS]
    if bad or not cfg.timeframes or len(set(cfg.timeframes)) != len(cfg.timeframes):
        raise SystemExit(f"BOT_TIMEFRAMES must be distinct values from {sorted(INTERVAL_MS)}, got {cfg.timeframes}")
    if cfg.bear_filter not in ("halving", "off"):
        raise SystemExit(f"BOT_BEAR_FILTER must be 'halving' or 'off', got {cfg.bear_filter!r}")
    if not cfg.strategies or set(cfg.strategies) - {"di", "adx"} or len(set(cfg.strategies)) != len(cfg.strategies):
        raise SystemExit(f"BOT_STRATEGIES must be one or both of di, adx; got {cfg.strategies}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
