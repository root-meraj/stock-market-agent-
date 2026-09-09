"""Post-Earnings-Announcement-Drift test. The first non-price hypothesis.

Buy stocks that beat consensus EPS (surprise > threshold), enter the session AFTER the
announcement (point-in-time), hold H days; measure abnormal return vs the equal-weight
universe over the same window, net of trading cost. Sweep (surprise threshold x holding
period) and walk-forward over the event timeline to check out-of-sample stability.

  python brain_agent/pead.py            # sweep + walk-forward
  python brain_agent/pead.py --selftest # no network
"""
import argparse
import bisect
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from fundamentals import NIFTY, load_earnings, load_prices

COST = bt.CostModel(kind="equity", stcg=0.0)
ROUNDTRIP = COST.buy_cost(1.0) + COST.sell_cost(1.0)      # ~0.3% charged to the event leg
THRESHOLDS = [0.0, 2.0, 5.0, 10.0]                        # EPS surprise %
HOLDS = [20, 30, 40, 60]                                  # trading days


def _asof(series, ts):
    i = bisect.bisect_right([r[0] for r in series], ts) - 1
    return series[i][4] if i >= 0 else None


def event_alpha(prices, sym, announce_ms, H):
    """Abnormal return of holding `sym` for H days starting the session AFTER the
    announcement, minus the equal-weight universe over the same calendar window."""
    series = prices[sym]
    i = bisect.bisect_right([r[0] for r in series], announce_ms)   # first bar strictly after
    if i <= 0 or i + H >= len(series):
        return None
    entry_ts, entry_px = series[i][0], series[i][4]
    exit_ts, exit_px = series[i + H][0], series[i + H][4]
    stock = exit_px / entry_px - 1 - ROUNDTRIP
    uni = [b / a - 1 for a, b in
           ((_asof(s, entry_ts), _asof(s, exit_ts)) for s in prices.values())
           if a and b]
    return stock - (sum(uni) / len(uni)) if uni else None


def build_events(universe=NIFTY):
    prices = {s: load_prices(s) for s in universe if load_prices(s)}
    events = []
    for s in prices:
        for ms, est, rep, sur in load_earnings(s):
            surprise = sur if sur is not None else (
                (rep - est) / abs(est) * 100 if est else None)
            if surprise is not None:
                events.append((ms, s, surprise))
    events.sort()
    return events, prices


def evaluate(events, prices, thr, H):
    return [a for a in
            (event_alpha(prices, s, ms, H) for ms, s, sur in events if sur >= thr)
            if a is not None]


def summarize(alphas):
    if len(alphas) < 5:
        return None
    a = np.array(alphas)
    se = a.std() / np.sqrt(len(a))
    return {"n": len(a), "mean_alpha": float(a.mean()), "hit": float((a > 0).mean()),
            "t": float(a.mean() / se) if se > 0 else 0.0}


def sweep(events, prices):
    print(f"{len(events)} earnings events across {len(prices)} stocks\n")
    print(f"  {'thr%':>5} {'hold':>5} {'n':>5} {'mean_alpha':>11} {'hit%':>6} {'t-stat':>7}")
    best = None
    for thr in THRESHOLDS:
        for H in HOLDS:
            s = summarize(evaluate(events, prices, thr, H))
            if not s:
                continue
            print(f"  {thr:>5.0f} {H:>5} {s['n']:>5} {s['mean_alpha']:>+10.2%} "
                  f"{s['hit']:>5.0%} {s['t']:>+7.2f}")
            if best is None or s["t"] > best[0]:
                best = (s["t"], thr, H, s)
    if best:
        _, thr, H, s = best
        print(f"\n  strongest full-sample cell: thr={thr:.0f}% hold={H}d  "
              f"mean_alpha={s['mean_alpha']:+.2%} t={s['t']:+.2f} n={s['n']}")
    return best


def walk_forward(events, prices, n_folds=4):
    """Expanding-window: tune (thr,H) on earlier events, test on the next block."""
    print(f"\nWalk-forward ({n_folds} folds, expanding train / out-of-sample test):")
    n = len(events)
    bounds = [int(n * k / (n_folds + 1)) for k in range(n_folds + 2)]
    combos = [(thr, H) for thr in THRESHOLDS for H in HOLDS]
    test_means = []
    print(f"  {'fold':>4} {'train_ev':>8} {'best (thr,H)':>14} {'train_a':>8} "
          f"{'test_ev':>7} {'test_a':>8} {'test_t':>7}")
    for k in range(n_folds):
        train = events[:bounds[k + 1]]
        test = events[bounds[k + 1]:bounds[k + 2]]
        scored = [(summarize(evaluate(train, prices, thr, H)), thr, H) for thr, H in combos]
        scored = [(s, thr, H) for s, thr, H in scored if s and s["n"] >= 15]
        if not scored or len(test) < 10:
            continue
        s_tr, thr, H = max(scored, key=lambda x: x[0]["mean_alpha"])
        s_te = summarize(evaluate(test, prices, thr, H))
        if not s_te:
            continue
        test_means.append(s_te["mean_alpha"])
        print(f"  {k+1:>4} {len(train):>8} {f'({thr:.0f},{H})':>14} {s_tr['mean_alpha']:>+7.2%} "
              f"{s_te['n']:>7} {s_te['mean_alpha']:>+7.2%} {s_te['t']:>+7.2f}")
    if test_means:
        m = float(np.mean(test_means))
        print(f"\n  mean out-of-sample alpha across folds: {m:+.2%}")
        print("  -> PEAD has a pulse" if m > 0.005 else
              "  -> no reliable out-of-sample drift (alpha ~0 after costs)")
    return test_means


def _selftest():
    base, day = 1_600_000_000_000, 86_400_000
    up = [[base + i * day, 0, 0, 0, 100 * (1.10 ** (i / 20)), 0] for i in range(80)]
    flat = [[base + i * day, 0, 0, 0, 100.0, 0] for i in range(80)]
    prices = {"UP": up, "F1": flat, "F2": flat}
    a = event_alpha(prices, "UP", base + 5 * day, 20)          # beat-stock drifts, universe flat
    assert a and a > 0.05, a
    assert event_alpha(prices, "UP", base + 79 * day, 20) is None   # not enough forward data
    print("pead selftest ok: event alpha =", round(a, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    events, prices = build_events()
    sweep(events, prices)
    walk_forward(events, prices)


if __name__ == "__main__":
    main()
