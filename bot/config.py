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
    mode: str = "paper"                 # default for both sleeves: "paper" or "live"
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    interval: str = "1d"
    di_len: int = 38
    adx_smoothing: int = 9

    # Buy only on a fresh DI+ cross. If the bot starts while DI+ is already
    # above DI-, it waits for the next cross unless this is on.
    join_open_trend: bool = False
    max_order_usd: float = 1_000_000.0  # hard cap per buy, per sleeve

    # Sleeves: each strategy trades its own share of the money, and can be
    # paper or live on its own (e.g. core live while swing proves itself).
    core_share: float = 0.7
    core_mode: str = ""                 # "" = use `mode`
    swing_enabled: bool = True
    swing_share: float = 0.3
    swing_mode: str = ""
    swing_fast: int = 8
    swing_slow: int = 21
    swing_maker_timeout_min: float = 15.0

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
        return self.data_dir / "bot.sqlite"

    def sleeve_mode(self, name):
        return getattr(self, f"{name}_mode") or self.mode

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
        max_order_usd=float(e("BOT_MAX_ORDER_USD", 1_000_000)),
        core_share=float(e("BOT_CORE_SHARE", 0.7)),
        core_mode=e("BOT_CORE_MODE", "").lower(),
        swing_enabled=_bool("BOT_SWING_ENABLED", True),
        swing_share=float(e("BOT_SWING_SHARE", 0.3)),
        swing_mode=e("BOT_SWING_MODE", "").lower(),
        swing_fast=int(e("BOT_SWING_FAST", 8)),
        swing_slow=int(e("BOT_SWING_SLOW", 21)),
        swing_maker_timeout_min=float(e("BOT_SWING_MAKER_TIMEOUT_MIN", 15)),
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
    modes = [cfg.mode, cfg.sleeve_mode("core"), cfg.sleeve_mode("swing")]
    if any(m not in ("paper", "live") for m in modes):
        raise SystemExit(f"BOT_MODE / BOT_CORE_MODE / BOT_SWING_MODE must be 'paper' or 'live', got {modes}")
    if "live" in modes[1:] and not (cfg.api_key and cfg.api_secret):
        raise SystemExit("Live mode needs BINANCEUS_API_KEY and BINANCEUS_API_SECRET")
    shares = cfg.core_share + (cfg.swing_share if cfg.swing_enabled else 0)
    if not (0 < cfg.core_share <= 1 and 0 <= cfg.swing_share <= 1 and shares <= 1.0001):
        raise SystemExit("BOT_CORE_SHARE and BOT_SWING_SHARE must be between 0 and 1 and add up to at most 1")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
