"""Event-driven backtester.
Reuses Strategy + RiskManager as-is; only the cost numbers change (via dataclass
replace so RiskManager's sell gate sees the India-flavored fee/tds automatically).

  python backtest.py                    # single run w/ current config
  python backtest.py --sweep            # grid over rsi_buy/rsi_sell/min_net_profit
  python backtest.py --years 3 --tf 1h  # more history
"""
import argparse
import csv
import itertools
import os
import time
from dataclasses import replace
from datetime import datetime

import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from risk import RiskManager
from strategy import Strategy, rsi_series


# India cost model (non-negotiable, overrides config.py defaults)
FEE_RATE = 0.002                          # 0.2% per side
GST = 0.18                                # 18% on fees
EFF_FEE = FEE_RATE * (1 + GST)            # ~0.236% effective
TDS_RATE = 0.01                           # 1% on sell notional

DATA_DIR = "data"
RESULTS_DIR = "results"


# --- OHLCV cache ---------------------------------------------------------

def cache_path(symbol, tf):
    return os.path.join(DATA_DIR, f"{symbol.replace('/', '')}_{tf}.csv")


def fetch_ohlcv(symbol, tf="1h", years=2.0, exchange_id="binance"):
    """Return list of [ts_ms, o, h, l, c, v]. Caches to CSV, top-up on re-run."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = cache_path(symbol, tf)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [[int(r[0])] + [float(x) for x in r[1:]] for r in csv.reader(f)]

    ex = getattr(ccxt, exchange_id)()
    tf_ms = ex.parse_timeframe(tf) * 1000
    now_ms = int(time.time() * 1000)
    want_start = now_ms - int(years * 365.25 * 86400 * 1000)
    since_ms = rows[-1][0] + tf_ms if rows else want_start

    while since_ms < now_ms:
        batch = ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + tf_ms
        time.sleep(ex.rateLimit / 1000)

    # dedupe (top-up may overlap by one) and trim to the requested window
    seen = set()
    uniq = []
    for r in rows:
        if r[0] not in seen and r[0] >= want_start:
            seen.add(r[0])
            uniq.append(r)
    uniq.sort(key=lambda r: r[0])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(uniq)
    return uniq


# --- In-memory broker matching the surface risk+run use -------------------

class BacktestBroker:
    def __init__(self, starting_cash):
        self._cash = starting_cash
        self._pos = {}
        self._prices = {}
        self.trades = []

    def set_prices(self, prices):
        self._prices = prices

    def cash(self):
        return self._cash

    def positions(self):
        return {s: dict(p) for s, p in self._pos.items() if p["qty"] > 1e-12}

    def price(self, sym):
        return self._prices[sym]

    def execute(self, symbol, side, amount, ts=0, rsi=None):
        px = self._prices[symbol]
        qty = amount / px if side == "buy" else amount
        notional = qty * px
        fee = notional * EFF_FEE
        tds = notional * TDS_RATE if side == "sell" else 0.0
        realized = None
        if side == "buy":
            old = self._pos.get(symbol, {"qty": 0.0, "avg_price": 0.0})
            new_qty = old["qty"] + qty
            new_avg = (old["qty"] * old["avg_price"] + qty * px) / new_qty
            self._pos[symbol] = {"qty": new_qty, "avg_price": new_avg}
            self._cash -= notional + fee
        else:
            old = self._pos[symbol]
            realized = (px - old["avg_price"]) * qty - fee - tds
            new_qty = old["qty"] - qty
            if new_qty < 1e-12:
                del self._pos[symbol]
            else:
                self._pos[symbol]["qty"] = new_qty
            self._cash += notional - fee - tds
        self.trades.append({
            "ts": ts, "symbol": symbol, "side": side, "qty": qty, "price": px,
            "fee": fee, "tds": tds, "rsi": rsi, "realized": realized,
        })


class _FakeDB:
    """Just enough of db.DB to keep RiskManager happy in-memory."""

    def __init__(self, trades):
        self._trades = trades
        self._flags = {}

    def trades_since(self, seconds):
        if not self._trades:
            return 0
        cutoff = self._trades[-1]["ts"] - seconds * 1000
        return sum(1 for t in self._trades if t["ts"] > cutoff)

    def get_flag(self, k, d=None):
        return self._flags.get(k, d)

    def set_flag(self, k, v):
        self._flags[k] = str(v)


# --- The replay -----------------------------------------------------------

def run(cfg, ohlcv_map, rsi_by_symbol=None):
    """rsi_by_symbol: optional {symbol: full RSI series} so a sweep computes RSI once
    per (symbol, period) instead of every combo. Falls back to computing it here.
    ponytail: full-history RSI, not a rolling candle_limit window like live. Wilder RSI
    converges within ~100 candles so the seed difference is <0.5pt in the interior;
    fine for tuning. Match live exactly only if a threshold sits right on that edge."""
    symbols = list(ohlcv_map.keys())
    all_ts = sorted({r[0] for series in ohlcv_map.values() for r in series})
    idx = {s: 0 for s in symbols}
    if rsi_by_symbol is None:
        rsi_by_symbol = {s: rsi_series([c[4] for c in ser], cfg.rsi_period)
                         for s, ser in ohlcv_map.items()}

    broker = BacktestBroker(cfg.starting_capital)
    strat = Strategy(cfg)
    risk = RiskManager(cfg, _FakeDB(broker.trades))

    equity_series = []
    for ts in all_ts:
        current = {}
        for s in symbols:
            series = ohlcv_map[s]
            while idx[s] < len(series) and series[idx[s]][0] <= ts:
                idx[s] += 1
            i = idx[s] - 1
            if i >= 0:
                current[s] = series[i][4]
        if not current:
            continue
        broker.set_prices(current)

        positions = broker.positions()
        equity = broker.cash() + sum(p["qty"] * current[s] for s, p in positions.items())
        equity_series.append((ts, equity))

        _, paused = risk.update_pause(equity)
        if paused:
            continue

        for sym in symbols:
            if sym not in current:
                continue
            i = idx[sym] - 1
            signal, r = strat.signal_from_rsi(rsi_by_symbol[sym][i])
            if not signal:
                continue
            decision = risk.approve(signal, sym, equity, broker.cash(), positions, current[sym])
            if not decision:
                continue
            side, amount = decision
            broker.execute(sym, side, amount, ts=ts, rsi=r)
            positions = broker.positions()

    return broker, equity_series


# --- Stats and outputs ----------------------------------------------------

def stats(equity_series, trades, starting_cap, tf_hours=1):
    if len(equity_series) < 2:
        return {"n_trades": 0, "final_equity": starting_cap, "total_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade_pnl": 0.0}
    eq = np.array([e for _, e in equity_series])
    ret = np.diff(eq) / eq[:-1]
    periods_per_year = (365.25 * 24) / tf_hours
    sharpe = float(ret.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((peak - eq) / peak).max())
    sells = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sells if (t["realized"] or 0) > 0]
    return {
        "n_trades": len(trades),
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / starting_cap - 1),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": (len(wins) / len(sells)) if sells else 0.0,
        "avg_trade_pnl": float(np.mean([t["realized"] for t in sells])) if sells else 0.0,
    }


def save_outputs(broker, equity_series, s, prefix=""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = f"{prefix}_" if prefix else ""
    with open(f"{RESULTS_DIR}/{p}trades.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["ts", "symbol", "side", "qty", "price", "fee", "tds", "rsi", "realized"])
        w.writeheader()
        w.writerows(broker.trades)
    ts = [datetime.utcfromtimestamp(t / 1000) for t, _ in equity_series]
    eq = [e for _, e in equity_series]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, eq, linewidth=1)
    ax.axhline(equity_series[0][1], color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(f"equity  ret={s['total_return']:.1%}  sharpe={s['sharpe']:.2f}  "
                 f"mdd={s['max_drawdown']:.1%}  n={s['n_trades']}")
    ax.set_ylabel("USDT")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/{p}equity.png", dpi=100)
    plt.close(fig)


def buy_and_hold(ohlcv_map):
    """Equal-weighted B&H across the symbols; net of one buy fee, one sell fee, one TDS."""
    rets = []
    for series in ohlcv_map.values():
        if len(series) < 2:
            continue
        gross = series[-1][4] / series[0][4] - 1
        rets.append(gross - 2 * EFF_FEE - TDS_RATE)
    return sum(rets) / len(rets) if rets else 0.0


# --- Grid sweep -----------------------------------------------------------

GRID_BUY = [20, 25, 30, 35, 40, 45]
GRID_SELL = [65, 70, 75, 80, 85]
GRID_MNP = [0.01, 0.02, 0.03, 0.04, 0.05]
GRID_PERIOD = [9, 14, 21]


def _run_grid(cfg, ohlcv_map, tf_hours, verbose=False):
    """450-combo grid on the given OHLCV. Returns (results sorted by alpha desc, bh_return)."""
    bh = buy_and_hold(ohlcv_map)
    # RSI depends only on period, not the buy/sell/mnp thresholds -> compute once per period.
    rsi_cache = {(s, p): rsi_series([c[4] for c in ser], p)
                 for s, ser in ohlcv_map.items() for p in set(GRID_PERIOD)}
    combos = list(itertools.product(GRID_BUY, GRID_SELL, GRID_MNP, GRID_PERIOD))
    results = []
    for i, (b, s_, mnp, p) in enumerate(combos):
        c = replace(cfg, rsi_buy=b, rsi_sell=s_, min_net_profit=mnp, rsi_period=p)
        rsi_by_symbol = {s: rsi_cache[(s, p)] for s in ohlcv_map}
        broker, eq = run(c, ohlcv_map, rsi_by_symbol)
        st = stats(eq, broker.trades, c.starting_capital, tf_hours=tf_hours)
        results.append({
            "rsi_buy": b, "rsi_sell": s_, "min_net_profit": mnp, "rsi_period": p,
            "bh_return": bh, "alpha": st["total_return"] - bh, **st,
        })
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(combos)}")
    results.sort(key=lambda r: r["alpha"], reverse=True)
    return results, bh


def sweep(cfg, ohlcv_map, tf_hours):
    n = len(GRID_BUY) * len(GRID_SELL) * len(GRID_MNP) * len(GRID_PERIOD)
    print(f"Grid: {len(GRID_BUY)}x{len(GRID_SELL)}x{len(GRID_MNP)}x{len(GRID_PERIOD)} = {n} combos")
    results, bh = _run_grid(cfg, ohlcv_map, tf_hours, verbose=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\nBuy-and-hold benchmark (equal-weighted, post-fee): {bh:+.1%}")
    print(f"\nTop 5 by alpha (strategy return - B&H):")
    for r in results[:5]:
        print(f"  buy={r['rsi_buy']} sell={r['rsi_sell']} mnp={r['min_net_profit']:.0%} "
              f"p={r['rsi_period']:2d}  ret={r['total_return']:+.1%} bh={r['bh_return']:+.1%} "
              f"alpha={r['alpha']:+.1%} sharpe={r['sharpe']:+.2f} n={r['n_trades']}")

    best = max(results, key=lambda r: r["sharpe"])
    c = replace(cfg, rsi_buy=best["rsi_buy"], rsi_sell=best["rsi_sell"],
                min_net_profit=best["min_net_profit"], rsi_period=best["rsi_period"])
    broker, eq = run(c, ohlcv_map)
    s_ = stats(eq, broker.trades, c.starting_capital, tf_hours=tf_hours)
    save_outputs(broker, eq, s_, prefix="best_sharpe")
    print(f"\nBest-Sharpe: buy={c.rsi_buy} sell={c.rsi_sell} mnp={c.min_net_profit:.0%} "
          f"p={c.rsi_period}  sharpe={s_['sharpe']:+.2f} ret={s_['total_return']:+.1%}")
    print(f"-> {RESULTS_DIR}/best_sharpe_equity.png (eyeball for smooth vs one-lucky-trade)")
    print(f"-> {RESULTS_DIR}/sweep.csv")
    return results


def walk_forward(cfg, ohlcv_map, tf_hours):
    """4 non-overlapping windows of ~6 months; train on first 5/6, test on last 1/6."""
    all_ts = sorted({r[0] for series in ohlcv_map.values() for r in series})
    span = all_ts[-1] - all_ts[0]
    win_ms = span // 4
    train_frac = 5 / 6

    print(f"\nWalk-forward: 4 windows of ~6mo (5mo train / 1mo test)")
    rows = []
    for w in range(4):
        w_start = all_ts[0] + w * win_ms
        w_end = w_start + win_ms
        split = w_start + int(win_ms * train_frac)
        train_map = {s: [r for r in series if w_start <= r[0] < split]
                     for s, series in ohlcv_map.items()}
        test_map = {s: [r for r in series if split <= r[0] < w_end]
                    for s, series in ohlcv_map.items()}
        if any(len(v) < 100 for v in train_map.values()):
            print(f"  window {w+1}: too little data, skipping")
            continue

        print(f"  window {w+1}: training on "
              f"{datetime.utcfromtimestamp(w_start/1000):%Y-%m-%d} -> "
              f"{datetime.utcfromtimestamp(split/1000):%Y-%m-%d} ...")
        train_res, _ = _run_grid(cfg, train_map, tf_hours)
        best = train_res[0]
        params = {k: best[k] for k in ("rsi_buy", "rsi_sell", "min_net_profit", "rsi_period")}

        c = replace(cfg, **params)
        broker, eq = run(c, test_map)
        test_st = stats(eq, broker.trades, c.starting_capital, tf_hours=tf_hours)
        test_bh = buy_and_hold(test_map)
        rows.append({
            "window": w + 1,
            "train_start": datetime.utcfromtimestamp(w_start/1000).strftime("%Y-%m-%d"),
            "test_start": datetime.utcfromtimestamp(split/1000).strftime("%Y-%m-%d"),
            **params,
            "train_return": best["total_return"], "train_alpha": best["alpha"],
            "test_return": test_st["total_return"], "test_bh": test_bh,
            "test_alpha": test_st["total_return"] - test_bh,
            "test_sharpe": test_st["sharpe"], "test_n_trades": test_st["n_trades"],
        })

    if not rows:
        print("no windows produced results")
        return []

    print(f"\n  {'win':3s}  {'test_period':22s}  {'best_params':30s}  "
          f"{'train_ret':>9s} {'train_alpha':>11s} {'test_ret':>9s} {'test_alpha':>10s}")
    for r in rows:
        pstr = f"b={r['rsi_buy']} s={r['rsi_sell']} mnp={r['min_net_profit']:.0%} p={r['rsi_period']}"
        period = f"{r['test_start']} + 1mo"
        print(f"  {r['window']:<3d}  {period:22s}  {pstr:30s}  "
              f"{r['train_return']:+9.1%} {r['train_alpha']:+11.1%} "
              f"{r['test_return']:+9.1%} {r['test_alpha']:+10.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/walk_forward.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    train_avg = float(np.mean([r["train_return"] for r in rows]))
    test_avg = float(np.mean([r["test_return"] for r in rows]))
    distinct = len({(r["rsi_buy"], r["rsi_sell"], r["min_net_profit"], r["rsi_period"]) for r in rows})
    print(f"\n  train avg ret {train_avg:+.1%}  |  test avg ret {test_avg:+.1%}")
    print(f"  distinct best-param sets across {len(rows)} windows: {distinct}")
    if test_avg < train_avg - 0.02 or distinct >= max(3, len(rows)):
        print("  -> overfitting signal: params drift or test underperforms train")
    else:
        print("  -> holds up: params stable and test tracks train")
    print(f"-> {RESULTS_DIR}/walk_forward.csv")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--exchange", default="binance")
    args = ap.parse_args()

    cfg = replace(Config.load(), taker_fee=EFF_FEE, tds_rate=TDS_RATE, timeframe=args.tf)
    tf_hours = ccxt.Exchange.parse_timeframe(args.tf) / 3600

    ohlcv_map = {}
    for sym in cfg.symbols:
        print(f"loading {sym} {args.tf} ...")
        ohlcv_map[sym] = fetch_ohlcv(sym, tf=args.tf, years=args.years, exchange_id=args.exchange)
        print(f"  {len(ohlcv_map[sym])} candles "
              f"({datetime.utcfromtimestamp(ohlcv_map[sym][0][0] / 1000):%Y-%m-%d} -> "
              f"{datetime.utcfromtimestamp(ohlcv_map[sym][-1][0] / 1000):%Y-%m-%d})")

    if args.sweep:
        sweep(cfg, ohlcv_map, tf_hours)
    if args.walk_forward:
        walk_forward(cfg, ohlcv_map, tf_hours)
    if not (args.sweep or args.walk_forward):
        broker, eq = run(cfg, ohlcv_map)
        s = stats(eq, broker.trades, cfg.starting_capital, tf_hours=tf_hours)
        save_outputs(broker, eq, s)
        print()
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:16s} {v:+.4f}")
            else:
                print(f"  {k:16s} {v}")
        print(f"\n-> {RESULTS_DIR}/equity.png, {RESULTS_DIR}/trades.csv")


if __name__ == "__main__":
    main()
