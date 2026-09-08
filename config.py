import os
from dataclasses import dataclass


def _f(k, d):
    return float(os.environ.get(k, d))


def _i(k, d):
    return int(os.environ.get(k, d))


@dataclass
class Config:
    mode: str = os.environ.get("MODE", "paper")              # paper | live
    exchange_id: str = os.environ.get("EXCHANGE_ID", "binance")  # paper uses public prices; ccxt dropped coindcx, live will need a REST shim
    symbols: tuple = ("BTC/USDT", "ETH/USDT")
    timeframe: str = os.environ.get("TIMEFRAME", "4h")       # swing, not scalp
    candle_limit: int = 200

    starting_capital: float = _f("STARTING_CAPITAL", 95.0)   # USDT, ~ Rs 8000
    position_frac: float = _f("POSITION_FRAC", 0.45)         # of equity per entry
    max_positions: int = _i("MAX_POSITIONS", 2)
    max_trades_per_week: int = _i("MAX_TRADES_PER_WEEK", 4)
    min_order: float = _f("MIN_ORDER", 5.0)                  # USDT

    rsi_period: int = 14
    rsi_buy: float = _f("RSI_BUY", 30)
    rsi_sell: float = _f("RSI_SELL", 70)

    taker_fee: float = _f("TAKER_FEE", 0.005)                # CoinDCX ~0.5%
    tds_rate: float = _f("TDS_RATE", 0.01)                   # India, per sell
    min_net_profit: float = _f("MIN_NET_PROFIT", 0.015)      # skip sells that don't clear costs

    pause_life: float = _f("PAUSE_LIFE", 0.75)               # equity/start below -> pause
    resume_life: float = _f("RESUME_LIFE", 0.85)             # ... above -> resume (hysteresis)

    poll_seconds: int = _i("POLL_SECONDS", 3600)
    db_path: str = os.environ.get("DB_PATH", "agent.db")

    @classmethod
    def load(cls):
        return cls()
