import argparse
import time
import traceback

from config import Config
from db import DB
from broker import Broker
from strategy import Strategy
from risk import RiskManager


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def run_once(cfg, db, broker, strat, risk):
    positions = broker.positions()
    prices = {s: broker.price(s) for s in cfg.symbols}
    equity = broker.cash() + sum(positions[s]["qty"] * prices[s] for s in positions)
    life, paused = risk.update_pause(equity)
    db.snapshot(equity, broker.cash(), life, paused)
    log(f"equity={equity:.2f} life={life:.0%} paused={paused} holding={list(positions)}")

    if paused:
        log("life below threshold - trading paused until equity recovers")
        return

    for sym in cfg.symbols:
        signal, r = strat.signal(broker.closes(sym))
        if not signal:
            continue
        decision = risk.approve(signal, sym, equity, broker.cash(), positions, prices[sym])
        if not decision:
            log(f"{sym} rsi={r:.1f} {signal} -> skipped by risk manager")
            continue
        side, amount = decision
        fill = broker.execute(sym, side, amount, rsi=r, reason=f"rsi={r:.1f}")
        log(f"{sym} {side} qty={fill['qty']:.6f} @ {fill['price']:.2f} "
            f"fee={fill['fee']:.3f} tds={fill['tds']:.3f}")
        positions = broker.positions()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one cycle and exit (cron mode)")
    args = ap.parse_args()

    cfg = Config.load()
    db = DB(cfg.db_path)
    broker = Broker(cfg, db)
    strat = Strategy(cfg)
    risk = RiskManager(cfg, db)
    log(f"started mode={cfg.mode} exchange={cfg.exchange_id} symbols={list(cfg.symbols)} "
        f"tf={cfg.timeframe}")

    while True:
        try:
            run_once(cfg, db, broker, strat, risk)
        except Exception:
            log("cycle error:\n" + traceback.format_exc())
        if args.once:
            break
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
