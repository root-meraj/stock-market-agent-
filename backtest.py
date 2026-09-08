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
from dataclasses import fields, replace
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

def run(cfg, ohlcv_map, signals_by_symbol=None):
    """signals_by_symbol: {symbol: ["buy"/"sell"/None per candle index]}, precomputed by
    a strategy generator. If omitted, defaults to the RSI strategy over its own cfg params.
    Either way RiskManager gates every proposed order (sizing, TDS profit gate, life meter,
    trade budget) — the strategy only proposes, risk disposes.
    ponytail: full-history indicators, not a rolling candle_limit window like live. Wilder RSI
    (and EMA) converge fast so the seed difference is negligible in the interior; fine for
    tuning. Match live exactly only if a threshold sits right on that edge."""
    symbols = list(ohlcv_map.keys())
    all_ts = sorted({r[0] for series in ohlcv_map.values() for r in series})
    idx = {s: 0 for s in symbols}
    if signals_by_symbol is None:
        strat = Strategy(cfg)
        signals_by_symbol = {}
        for s, ser in ohlcv_map.items():
            rser = rsi_series([c[4] for c in ser], cfg.rsi_period)
            signals_by_symbol[s] = [strat.signal_from_rsi(r)[0] for r in rser]

    broker = BacktestBroker(cfg.starting_capital)
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
            signal = signals_by_symbol[sym][idx[sym] - 1]
            if not signal:
                continue
            decision = risk.approve(signal, sym, equity, broker.cash(), positions, current[sym])
            if not decision:
                continue
            side, amount = decision
            broker.execute(sym, side, amount, ts=ts)
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


# --- Strategy family ------------------------------------------------------
# Each generator returns ["buy"/"sell"/None per candle index]; RiskManager gates them all.
# Mean-reversion (RSI) keeps the TDS profit gate; trend/breakout disable it via
# min_net_profit=-1.0 (config-only, RiskManager untouched) so they can stop out of losers.

CFG_FIELDS = {f.name for f in fields(Config)}


def _apply(cfg, params):
    return replace(cfg, **{k: v for k, v in params.items() if k in CFG_FIELDS})


def _ema(values, period):
    k = 2.0 / (period + 1)
    out, e = [], None
    for v in values:
        e = v if e is None else v * k + e * (1.0 - k)
        out.append(e)
    return out


def rsi_signals(series, params, cfg):
    strat = Strategy(cfg)                         # reuse live threshold logic verbatim
    rser = rsi_series([c[4] for c in series], cfg.rsi_period)
    return [strat.signal_from_rsi(r)[0] for r in rser]


def ema_signals(series, params, cfg):
    closes = [c[4] for c in series]
    fast, slow = _ema(closes, params["fast"]), _ema(closes, params["slow"])
    out = [None] * len(closes)
    for i in range(params["slow"], len(closes)):
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            out[i] = "buy"
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            out[i] = "sell"
    return out


def donchian_signals(series, params, cfg):
    n = params["period"]
    highs = [c[2] for c in series]
    lows = [c[3] for c in series]
    closes = [c[4] for c in series]
    out = [None] * len(series)
    # ponytail: O(n) window max/min per bar. Fine at these sizes; use a monotonic deque if it bites.
    for i in range(n, len(series)):
        if closes[i] > max(highs[i - n:i]):
            out[i] = "buy"
        elif closes[i] < min(lows[i - n:i]):
            out[i] = "sell"
    return out


def _grid(pairs):
    """Cartesian product of {key: [values]} -> list of param dicts."""
    keys = list(pairs)
    return [dict(zip(keys, combo)) for combo in itertools.product(*pairs.values())]


# Full RSI sweep grid (the --sweep command); survey grids are smaller (pulse check, not tuning).
GRID_BUY = [20, 25, 30, 35, 40, 45]
GRID_SELL = [65, 70, 75, 80, 85]
GRID_MNP = [0.01, 0.02, 0.03, 0.04, 0.05]
GRID_PERIOD = [9, 14, 21]

STRATEGIES = {
    "rsi": (rsi_signals, _grid({"rsi_buy": [25, 35], "rsi_sell": [70, 80],
                                "rsi_period": [14, 21], "min_net_profit": [0.01]})),
    "ema": (ema_signals, _grid({"fast": [10, 20], "slow": [50, 100, 200],
                                "min_net_profit": [-1.0]})),
    "donchian": (donchian_signals, _grid({"period": [20, 40, 55], "min_net_profit": [-1.0]})),
}


def _run_grid(cfg, ohlcv_map, tf_hours, gen, grid, verbose=False):
    """Run a strategy's grid on the OHLCV. Returns (results sorted by alpha desc, bh_return)."""
    bh = buy_and_hold(ohlcv_map)
    results = []
    for i, params in enumerate(grid):
        c = _apply(cfg, params)
        sigs = {s: gen(ser, params, c) for s, ser in ohlcv_map.items()}
        broker, eq = run(c, ohlcv_map, sigs)
        st = stats(eq, broker.trades, c.starting_capital, tf_hours=tf_hours)
        results.append({**params, "bh_return": bh, "alpha": st["total_return"] - bh, **st})
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(grid)}")
    results.sort(key=lambda r: r["alpha"], reverse=True)
    return results, bh


def _walk_forward(cfg, ohlcv_map, tf_hours, gen, grid, n_windows=4, train_frac=5 / 6):
    """Generic walk-forward: n windows, train grid on first train_frac, test on the rest.
    Returns per-window rows (params dict + train/test metrics)."""
    all_ts = sorted({r[0] for series in ohlcv_map.values() for r in series})
    span = all_ts[-1] - all_ts[0]
    win_ms = span // n_windows
    param_keys = list(grid[0].keys())
    rows = []
    for w in range(n_windows):
        w_start = all_ts[0] + w * win_ms
        w_end = w_start + win_ms
        split = w_start + int(win_ms * train_frac)
        train_map = {s: [r for r in series if w_start <= r[0] < split]
                     for s, series in ohlcv_map.items()}
        test_map = {s: [r for r in series if split <= r[0] < w_end]
                    for s, series in ohlcv_map.items()}
        if any(len(v) < 100 for v in train_map.values()):
            continue
        train_res, _ = _run_grid(cfg, train_map, tf_hours, gen, grid)
        best = train_res[0]
        params = {k: best[k] for k in param_keys}
        c = _apply(cfg, params)
        test_sigs = {s: gen(ser, params, c) for s, ser in test_map.items()}
        broker, eq = run(c, test_map, test_sigs)
        test_st = stats(eq, broker.trades, c.starting_capital, tf_hours=tf_hours)
        test_bh = buy_and_hold(test_map)
        rows.append({
            "window": w + 1,
            "train_start": datetime.utcfromtimestamp(w_start / 1000).strftime("%Y-%m-%d"),
            "test_start": datetime.utcfromtimestamp(split / 1000).strftime("%Y-%m-%d"),
            "params": params,
            "train_return": best["total_return"], "train_alpha": best["alpha"],
            "test_return": test_st["total_return"], "test_bh": test_bh,
            "test_alpha": test_st["total_return"] - test_bh,
            "test_sharpe": test_st["sharpe"], "test_n_trades": test_st["n_trades"],
        })
    return rows


def sweep(cfg, ohlcv_map, tf_hours):
    gen, _ = STRATEGIES["rsi"]
    grid = _grid({"rsi_buy": GRID_BUY, "rsi_sell": GRID_SELL,
                  "min_net_profit": GRID_MNP, "rsi_period": GRID_PERIOD})
    print(f"Grid: {len(grid)} combos")
    results, bh = _run_grid(cfg, ohlcv_map, tf_hours, gen, grid, verbose=True)

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


def _pstr(params):
    parts = []
    for k, v in params.items():
        if k == "min_net_profit":
            if v < 0:
                continue                                  # sentinel: profit gate disabled
            parts.append(f"mnp={v:.0%}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def walk_forward(cfg, ohlcv_map, tf_hours, strategy="rsi"):
    """4 non-overlapping windows of ~6 months; train grid on first 5/6, test on last 1/6."""
    gen, grid = STRATEGIES[strategy]
    print(f"\nWalk-forward ({strategy}): 4 windows of ~6mo (5mo train / 1mo test)")
    rows = _walk_forward(cfg, ohlcv_map, tf_hours, gen, grid)
    if not rows:
        print("no windows produced results")
        return []

    print(f"\n  {'win':3s}  {'test_period':22s}  {'best_params':34s}  "
          f"{'train_ret':>9s} {'train_alpha':>11s} {'test_ret':>9s} {'test_alpha':>10s}")
    for r in rows:
        print(f"  {r['window']:<3d}  {r['test_start'] + ' + 1mo':22s}  {_pstr(r['params']):34s}  "
              f"{r['train_return']:+9.1%} {r['train_alpha']:+11.1%} "
              f"{r['test_return']:+9.1%} {r['test_alpha']:+10.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    flat = [{**{k: v for k, v in r.items() if k != "params"}, **r["params"]} for r in rows]
    with open(f"{RESULTS_DIR}/walk_forward.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    train_avg = float(np.mean([r["train_return"] for r in rows]))
    test_avg = float(np.mean([r["test_return"] for r in rows]))
    distinct = len({tuple(sorted(r["params"].items())) for r in rows})
    print(f"\n  train avg ret {train_avg:+.1%}  |  test avg ret {test_avg:+.1%}")
    print(f"  distinct best-param sets across {len(rows)} windows: {distinct}")
    if test_avg < train_avg - 0.02 or distinct >= max(3, len(rows)):
        print("  -> overfitting signal: params drift or test underperforms train")
    else:
        print("  -> holds up: params stable and test tracks train")
    print(f"-> {RESULTS_DIR}/walk_forward.csv")
    return rows


# --- Market survey --------------------------------------------------------

SURVEY_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]  # LINK: mid-cap (MATIC->POL renamed)
SURVEY_TFS = ["15m", "1h", "4h", "1d"]


def _cell(cfg, ohlcv_map, tf_hours, strategy):
    """One (tf x asset x strategy) cell: out-of-sample walk-forward summary."""
    gen, grid = STRATEGIES[strategy]
    rows = _walk_forward(cfg, ohlcv_map, tf_hours, gen, grid)
    if not rows:
        return {"test_alpha": 0.0, "wf_gap": 0.0, "test_return": 0.0, "bh_return": 0.0,
                "sharpe": 0.0, "n_trades": 0, "n_windows": 0, "param_churn": 0}
    train_avg = float(np.mean([r["train_return"] for r in rows]))
    test_avg = float(np.mean([r["test_return"] for r in rows]))
    return {
        "test_alpha": float(np.mean([r["test_alpha"] for r in rows])),   # out-of-sample vs B&H
        "wf_gap": train_avg - test_avg,                                  # overfit gap
        "test_return": test_avg,
        "bh_return": float(np.mean([r["test_bh"] for r in rows])),
        "sharpe": float(np.mean([r["test_sharpe"] for r in rows])),
        "n_trades": int(sum(r["test_n_trades"] for r in rows)),
        "n_windows": len(rows),
        "param_churn": len({tuple(sorted(r["params"].items())) for r in rows}),
    }


def _survey_heatmap(cells):
    """One asset x timeframe heatmap per strategy, colored by out-of-sample alpha."""
    strategies = list(STRATEGIES)
    lut = {(c["strategy"], c["timeframe"], c["asset"]): c for c in cells}
    fig, axes = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 4), squeeze=False)
    vmax = max((abs(c["test_alpha"]) for c in cells), default=0.05) or 0.05
    for ax, sname in zip(axes[0], strategies):
        grid = np.array([[lut.get((sname, tf, a), {}).get("test_alpha", np.nan)
                          for tf in SURVEY_TFS] for a in SURVEY_ASSETS])
        ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SURVEY_TFS)), SURVEY_TFS)
        ax.set_yticks(range(len(SURVEY_ASSETS)), [a.split("/")[0] for a in SURVEY_ASSETS])
        ax.set_title(sname)
        for yi, a in enumerate(SURVEY_ASSETS):
            for xi, tf in enumerate(SURVEY_TFS):
                c = lut.get((sname, tf, a))
                if c:
                    ax.text(xi, yi, f"{c['test_alpha']:+.0%}\nn{c['n_trades']}",
                            ha="center", va="center", fontsize=7)
    fig.suptitle("Out-of-sample alpha vs buy-and-hold (green=beats B&H after fees)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/survey_heatmap.png", dpi=100)
    plt.close(fig)


def survey(years, exchange):
    """Sweep timeframe x asset x strategy; dump one CSV. Goal: which shape has a pulse?"""
    cells = []
    for tf in SURVEY_TFS:
        tf_hours = ccxt.Exchange.parse_timeframe(tf) / 3600
        for asset in SURVEY_ASSETS:
            print(f"loading {asset} {tf} ...")
            try:
                series = fetch_ohlcv(asset, tf=tf, years=years, exchange_id=exchange)
            except Exception as e:
                print(f"  skip {asset} {tf}: {e}")
                continue
            if len(series) < 400:
                print(f"  skip {asset} {tf}: only {len(series)} candles")
                continue
            ohlcv_map = {asset: series}
            cfg = replace(Config.load(), taker_fee=EFF_FEE, tds_rate=TDS_RATE, timeframe=tf)
            for strat in STRATEGIES:
                m = _cell(cfg, ohlcv_map, tf_hours, strat)
                row = {"timeframe": tf, "asset": asset, "strategy": strat, **m}
                cells.append(row)
                pulse = "PULSE" if m["test_alpha"] > 0 and m["n_trades"] >= 4 else "."
                print(f"  {strat:9s} test_alpha={m['test_alpha']:+.1%} "
                      f"wf_gap={m['wf_gap']:+.1%} sharpe={m['sharpe']:+.2f} "
                      f"n={m['n_trades']:<4d} {pulse}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cells.sort(key=lambda r: r["test_alpha"], reverse=True)
    with open(f"{RESULTS_DIR}/survey.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    _survey_heatmap(cells)

    winners = [c for c in cells if c["test_alpha"] > 0 and c["n_trades"] >= 4]
    print(f"\n{'='*60}")
    print(f"Cells with a pulse (out-of-sample alpha > 0, >=4 trades): {len(winners)}/{len(cells)}")
    for c in winners[:10]:
        print(f"  {c['timeframe']:4s} {c['asset']:9s} {c['strategy']:9s}  "
              f"alpha={c['test_alpha']:+.1%} gap={c['wf_gap']:+.1%} "
              f"sharpe={c['sharpe']:+.2f} n={c['n_trades']}")
    if not winners:
        print("  NONE. No (tf x asset x strategy) beats buy-and-hold out-of-sample after fees.")
        print("  -> The honest answer: this problem shape has no pulse. Pivot before ML.")
    print(f"-> {RESULTS_DIR}/survey.csv")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--survey", action="store_true", help="tf x asset x strategy market survey")
    ap.add_argument("--strategy", default="rsi", choices=list(STRATEGIES),
                    help="strategy for --walk-forward / single run")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--exchange", default="binance")
    args = ap.parse_args()

    if args.survey:
        survey(args.years, args.exchange)
        return

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
        walk_forward(cfg, ohlcv_map, tf_hours, strategy=args.strategy)
    if not (args.sweep or args.walk_forward):
        if args.strategy == "rsi":
            broker, eq = run(cfg, ohlcv_map)                  # cfg's own RSI params (config.py)
        else:
            gen, grid = STRATEGIES[args.strategy]
            c = _apply(cfg, grid[0])
            sigs = {s: gen(ser, grid[0], c) for s, ser in ohlcv_map.items()}
            broker, eq = run(c, ohlcv_map, sigs)
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
