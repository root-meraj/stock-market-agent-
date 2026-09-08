"""Run: python test_agent.py   (plain asserts, no framework, no network)."""
from config import Config
from db import DB
from broker import Broker
from risk import RiskManager
from strategy import rsi


class FakeEx:
    def __init__(self, prices):
        self.prices = prices

    def fetch_ticker(self, s):
        return {"last": self.prices[s]}

    def fetch_ohlcv(self, s, tf, limit=None):
        return [[0, 0, 0, 0, c, 0] for c in range(1, 80)]


def test_rsi():
    assert rsi([1, 2], 14) is None
    assert rsi(list(range(1, 40)), 14) > 99            # only gains
    assert rsi(list(range(40, 1, -1)), 14) < 1         # only losses
    assert rsi([5] * 40, 14) == 50.0                   # flat


def test_life_meter_hysteresis():
    cfg, db = Config(), DB(":memory:")
    r = RiskManager(cfg, db)
    s = cfg.starting_capital
    assert r.update_pause(s)[1] is False
    assert r.update_pause(s * 0.70)[1] is True         # below pause_life
    assert r.update_pause(s * 0.80)[1] is True         # dead band -> stays paused
    assert r.update_pause(s * 0.90)[1] is False        # above resume_life


def test_risk_approve():
    cfg, db = Config(), DB(":memory:")
    r = RiskManager(cfg, db)
    assert r.approve("buy", "BTC/USDT", 100.0, 100.0, {}, 50000) == ("buy", 45.0)
    assert r.approve("buy", "BTC/USDT", 100.0, 1.0, {}, 50000) is None          # cash < min_order
    pos = {"BTC/USDT": {"qty": 0.001, "avg_price": 50000}}
    assert r.approve("buy", "BTC/USDT", 100.0, 100.0, pos, 50000) is None       # already holding
    assert r.approve("sell", "BTC/USDT", 100.0, 100.0, pos, 50500) is None      # +1% < costs
    assert r.approve("sell", "BTC/USDT", 100.0, 100.0, pos, 52000)[0] == "sell"  # +4% ok


def test_paper_round_trip():
    cfg, db = Config(), DB(":memory:")
    ex = FakeEx({"BTC/USDT": 50000.0})
    b = Broker(cfg, db, exchange=ex)
    start = b.cash()
    b.execute("BTC/USDT", "buy", 45.0, reason="test")
    assert abs(b.cash() - (start - 45.0 - 45.0 * cfg.taker_fee)) < 1e-6
    qty = b.positions()["BTC/USDT"]["qty"]
    ex.prices["BTC/USDT"] = 55000.0
    b.execute("BTC/USDT", "sell", qty, reason="test")
    assert "BTC/USDT" not in b.positions()
    proceeds = qty * 55000.0
    expected = (start - 45.0 - 45.0 * cfg.taker_fee) + proceeds \
        - proceeds * cfg.taker_fee - proceeds * cfg.tds_rate
    assert abs(b.cash() - expected) < 1e-6
    assert db.trades_since(3600) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
