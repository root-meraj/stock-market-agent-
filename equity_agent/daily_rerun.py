"""ONE_DAY equity survey, properly sized. Closes the gap where daily had n=0 (test windows
were ~1 month, too short for swing signals to fire).

  python equity_agent/daily_rerun.py [--zero-brokerage] [--years 8]

- 8y cached daily history, rolling folds: train 24mo / test 12mo, stepped 12mo, up to 4 folds
- Per-stock strategies (rsi/ema/donchian): reuse the crypto grid + walk-forward primitives
- Cross-sectional momentum: rank 10 stocks by trailing 12m return monthly, hold top 3 EW,
  scored vs equal-weight B&H of all 10 (its own portfolio simulator)
- Pulse gate: n>=8 trades across the whole out-of-sample test (summed over folds)
- Output: results/equity_daily_rerun.csv + equity_daily_rerun_heatmap.png
"""
import argparse
import csv
import os
import sys
from bisect import bisect_right
from dataclasses import replace
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from angel_client import AngelClient
from config import Config

ASSETS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
          "ITC", "LT", "SBIN", "BHARTIARTL", "MARUTI"]
TF_HOURS = 24
DAY = 86_400_000
RESULTS = os.path.join(ROOT, "results")
PER_STOCK = ["rsi", "ema", "donchian"]


def folds(all_ts, train_days=730, test_days=365, step_days=365, max_folds=4):
    """Rolling (train, test) windows anchored to the data end, most-recent first, then reversed."""
    start, end = all_ts[0], all_ts[-1]
    out = []
    k = 0
    while len(out) < max_folds:
        test_end = end - k * step_days * DAY
        test_start = test_end - test_days * DAY
        train_start = test_start - train_days * DAY
        if train_start < start:
            break
        out.append((train_start, test_start, test_end))
        k += 1
    return list(reversed(out))


def _slice(series, lo, hi):
    return [r for r in series if lo <= r[0] < hi]


# --- cross-sectional momentum (portfolio strategy) ------------------------

def _asof(series, ts):
    """Close as of the last bar at or before ts (series sorted by ts)."""
    i = bisect_right([r[0] for r in series], ts) - 1
    return series[i][4] if i >= 0 else None


def momentum_curve(daily, test_start, test_end, cost, top=3, lookback_days=365, cap=95.0):
    """Monthly rebalance into the top-`top` stocks by trailing return. Returns
    (equity_series, n_legs). Fully invested; fees paid out of portfolio value."""
    syms = list(daily)
    days = sorted({r[0] for s in syms for r in daily[s] if test_start <= r[0] < test_end})
    if len(days) < 40:
        return [], 0
    holdings = {}            # sym -> shares
    equity_series = []
    n_legs = 0
    last_month = None
    for ts in days:
        prices = {s: _asof(daily[s], ts) for s in syms}
        prices = {s: p for s, p in prices.items() if p}
        month = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m")
        if month != last_month:                                  # rebalance at each new month
            last_month = month
            ranked = sorted(
                (s for s in prices
                 if _asof(daily[s], ts - lookback_days * DAY)),
                key=lambda s: prices[s] / _asof(daily[s], ts - lookback_days * DAY),
                reverse=True)
            chosen = ranked[:top]
            if chosen:
                v = cap if not holdings else sum(holdings.get(s, 0) * prices[s] for s in holdings)
                buy_n = sell_n = 0.0
                for s in set(holdings) | set(chosen):
                    cur = holdings.get(s, 0) * prices.get(s, 0)
                    tgt = (v / len(chosen)) if s in chosen else 0.0
                    if tgt > cur:
                        buy_n += tgt - cur
                    else:
                        sell_n += cur - tgt
                    if abs(tgt - cur) > 1e-9:
                        n_legs += 1
                v_after = v - cost.buy_cost(buy_n) - cost.sell_cost(sell_n)
                holdings = {s: (v_after / len(chosen)) / prices[s] for s in chosen}
        equity = sum(holdings.get(s, 0) * prices[s] for s in holdings if s in prices)
        equity_series.append((ts, equity or cap))
    return equity_series, n_legs


# --- per-stock walk-forward (reuses bt primitives) ------------------------

def per_stock_cell(cfg, series, strategy, cost, fold_windows):
    gen, grid = bt.STRATEGIES[strategy]
    train_rets, test_rets, test_alphas, sharpes, n = [], [], [], [], 0
    for train_start, test_start, test_end in fold_windows:
        train = _slice(series, train_start, test_start)
        test = _slice(series, test_start, test_end)
        if len(train) < 200 or len(test) < 60:
            continue
        res, _ = bt._run_grid(cfg, {"S": train}, TF_HOURS, gen, grid, cost=cost)
        best = res[0]
        params = {k: best[k] for k in grid[0]}
        c = bt._apply(cfg, params)
        sigs = {"S": gen(test, params, c)}
        broker, eq = bt.run(c, {"S": test}, sigs, cost=cost)
        st = bt.stats(eq, broker.trades, c.starting_capital, tf_hours=TF_HOURS, cost=cost)
        bh = bt.buy_and_hold({"S": test}, cost)
        train_rets.append(best["total_return"])
        test_rets.append(st["total_return"])
        test_alphas.append(st["total_return"] - bh)
        sharpes.append(st["sharpe"])
        n += st["n_trades"]
    if not test_alphas:
        return None
    return {"test_alpha": float(np.mean(test_alphas)),
            "wf_gap": float(np.mean(train_rets) - np.mean(test_rets)),
            "sharpe": float(np.mean(sharpes)), "n_trades": n, "n_folds": len(test_alphas)}


def momentum_cell(daily, cost, fold_windows):
    test_alphas, sharpes, n = [], [], 0
    for _, test_start, test_end in fold_windows:
        eq, legs = momentum_curve(daily, test_start, test_end, cost,
                                  cap=Config.load().starting_capital)
        if len(eq) < 40:
            continue
        st = bt.stats(eq, [], Config.load().starting_capital, tf_hours=TF_HOURS, cost=cost)
        bh = bt.buy_and_hold({s: _slice(daily[s], test_start, test_end) for s in daily}, cost)
        test_alphas.append(st["total_return"] - bh)
        sharpes.append(st["sharpe"])
        n += legs
    if not test_alphas:
        return None
    return {"test_alpha": float(np.mean(test_alphas)), "wf_gap": 0.0,
            "sharpe": float(np.mean(sharpes)), "n_trades": n, "n_folds": len(test_alphas)}


# --- heatmap --------------------------------------------------------------

def heatmap(cells, mom, path):
    lut = {(c["strategy"], c["asset"]): c for c in cells}
    fig, ax = plt.subplots(figsize=(6, 6))
    cols = PER_STOCK + ["momentum"]
    vmax = max([abs(c["test_alpha"]) for c in cells] + [abs(mom["test_alpha"])] if mom else
               [abs(c["test_alpha"]) for c in cells], default=0.05) or 0.05
    grid = np.full((len(ASSETS), len(cols)), np.nan)
    for yi, a in enumerate(ASSETS):
        for xi, s in enumerate(PER_STOCK):
            c = lut.get((s, a))
            if c:
                grid[yi, xi] = c["test_alpha"]
    if mom:
        grid[:, 3] = mom["test_alpha"]                      # portfolio: one value down the column
    ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(range(len(ASSETS)), ASSETS)
    for yi, a in enumerate(ASSETS):
        for xi, s in enumerate(PER_STOCK):
            c = lut.get((s, a))
            if c:
                ax.text(xi, yi, f"{c['test_alpha']:+.0%}\nn{c['n_trades']}",
                        ha="center", va="center", fontsize=7)
    if mom:
        ax.text(3, len(ASSETS) // 2, f"PORTFOLIO\n{mom['test_alpha']:+.1%}\nn{mom['n_trades']}",
                ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title("ONE_DAY out-of-sample alpha vs B&H\n(train 24mo / test 12mo, rolling folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _selftest():
    """No network. One stock rips, three are flat -> momentum must hold the winner and beat EW."""
    start = 1_600_000_000_000
    daily = {}
    for i, sym in enumerate(["WIN", "F1", "F2", "F3"]):
        rows = []
        for d in range(500):
            ts = start + d * DAY
            price = 100.0 * (1.02 ** (d / 21)) if sym == "WIN" else 100.0
            rows.append([ts, price, price, price, price, 0])
        daily[sym] = rows
    cost = bt.CostModel(kind="equity", stcg=0.0)
    test_start = start + 300 * DAY
    test_end = start + 500 * DAY
    eq, legs = momentum_curve(daily, test_start, test_end, cost, top=1, cap=100.0)
    assert eq and eq[-1][1] > eq[0][1], "momentum should compound the winner"
    bh = bt.buy_and_hold({s: _slice(daily[s], test_start, test_end) for s in daily}, cost)
    mom_ret = eq[-1][1] / eq[0][1] - 1
    assert mom_ret > bh, (mom_ret, bh)                # holding the winner beats equal-weight
    assert legs > 0
    print("selftest ok:", f"momentum={mom_ret:+.1%} vs EW B&H={bh:+.1%}, legs={legs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--zero-brokerage", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run network-free momentum check")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    cost = bt.CostModel(kind="equity", zero_brokerage=args.zero_brokerage, stcg=0.20)
    gate_fee = cost.sell_cost(1.0)
    client = AngelClient().login()
    print(f"logged in as {client.profile_name()}\n")

    to = datetime.today()
    frm = to - timedelta(days=int(args.years * 365.25))
    daily = {}
    for sym in ASSETS:
        print(f"loading {sym} ONE_DAY ...")
        daily[sym] = client.fetch_ohlcv(sym, "ONE_DAY", frm.strftime("%Y-%m-%d"),
                                        to.strftime("%Y-%m-%d"))
    all_ts = sorted({r[0] for s in daily.values() for r in s})
    fw = folds(all_ts)
    print(f"\n{len(fw)} rolling folds (train 24mo / test 12mo):")
    for ts0, ts1, ts2 in fw:
        print(f"  train {datetime.fromtimestamp(ts0/1000):%Y-%m} -> "
              f"{datetime.fromtimestamp(ts1/1000):%Y-%m}  |  test -> "
              f"{datetime.fromtimestamp(ts2/1000):%Y-%m}")

    cfg = replace(Config.load(), taker_fee=gate_fee, tds_rate=0.0, timeframe="ONE_DAY")
    cells = []
    print()
    for sym in ASSETS:
        for strat in PER_STOCK:
            m = per_stock_cell(cfg, daily[sym], strat, cost, fw)
            if not m:
                continue
            cells.append({"asset": sym, "strategy": strat, **m})
            pulse = "PULSE" if m["test_alpha"] > 0 and m["n_trades"] >= 8 else "."
            print(f"  {sym:10s} {strat:9s} alpha={m['test_alpha']:+.1%} "
                  f"gap={m['wf_gap']:+.1%} sharpe={m['sharpe']:+.2f} n={m['n_trades']:<3d} {pulse}")

    mom = momentum_cell(daily, cost, fw)
    if mom:
        pulse = "PULSE" if mom["test_alpha"] > 0 and mom["n_trades"] >= 8 else "."
        print(f"\n  {'TOP3':10s} {'momentum':9s} alpha={mom['test_alpha']:+.1%} "
              f"sharpe={mom['sharpe']:+.2f} n={mom['n_trades']:<3d} {pulse}  (vs EW B&H of all 10)")

    os.makedirs(RESULTS, exist_ok=True)
    all_rows = cells + ([{"asset": "TOP3_PORTFOLIO", "strategy": "momentum", **mom}] if mom else [])
    all_rows.sort(key=lambda r: r["test_alpha"], reverse=True)
    with open(os.path.join(RESULTS, "equity_daily_rerun.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    heatmap(cells, mom, os.path.join(RESULTS, "equity_daily_rerun_heatmap.png"))

    winners = [r for r in all_rows if r["test_alpha"] > 0 and r["n_trades"] >= 8]
    print(f"\n{'='*60}\nCells with a pulse (alpha>0, n>=8 out-of-sample): {len(winners)}/{len(all_rows)}")
    for r in winners[:12]:
        print(f"  {r['asset']:14s} {r['strategy']:9s} alpha={r['test_alpha']:+.1%} "
              f"gap={r['wf_gap']:+.1%} sharpe={r['sharpe']:+.2f} n={r['n_trades']}")
    if not winners:
        print("  NONE beats B&H out-of-sample with enough trades to trust.")
    print(f"-> {os.path.join(RESULTS, 'equity_daily_rerun.csv')}")
    print(f"-> {os.path.join(RESULTS, 'equity_daily_rerun_heatmap.png')}")


if __name__ == "__main__":
    main()
