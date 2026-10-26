"""Settings, read from environment variables (or a .env file in the repo root)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    mode: str = "paper"                 # "paper" or "live"
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    interval: str = "1d"
    di_len: int = 38
    adx_smoothing: int = 9

    # Buy only on a fresh DI+ cross. If the bot starts while DI+ is already
    # above DI-, it waits for the next cross unless this is on.
    join_open_trend: bool = False
    allocation: float = 1.0             # share of free quote balance to spend on a buy
    max_order_usd: float = 1_000_000.0  # hard cap per buy

    # Execution
    maker_timeout_min: float = 60.0     # how long to work a maker-only order
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

    @property
    def db_path(self):
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
        join_open_trend=_bool("BOT_JOIN_OPEN_TREND", False),
        allocation=float(e("BOT_ALLOCATION", 1.0)),
        max_order_usd=float(e("BOT_MAX_ORDER_USD", 1_000_000)),
        maker_timeout_min=float(e("BOT_MAKER_TIMEOUT_MIN", 60)),
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
    if not 0 < cfg.allocation <= 1:
        raise SystemExit("BOT_ALLOCATION must be between 0 and 1")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
