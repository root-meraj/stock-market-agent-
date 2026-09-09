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
from fundamentals import NIFTY, load_earnings_reported, load_prices, smallmid_universe

COST = bt.CostModel(kind="equity", stcg=0.0)
ROUNDTRIP = COST.buy_cost(1.0) + COST.sell_cost(1.0)      # ~0.3% charged to the event leg
THRESHOLDS = [0.0, 2.0, 5.0, 10.0]                        # EPS surprise %
HOLDS = [20, 30, 40, 60]                                  # trading days
DAY = 86_400_000


def _asof(series, ts):
    i = bisect.bisect_right([r[0] for r in series], ts) - 1
    return series[i][4] if i >= 0 else None


def event_alpha(prices, sym, announce_ms, H, slippage=0.0):
    """Abnormal return of holding `sym` for H days starting the session AFTER the
    announcement, minus the equal-weight universe over the same window. `slippage` is the
    adverse fill fraction on EACH of entry and exit (on top of the fee/STT cost model)."""
    series = prices[sym]
    i = bisect.bisect_right([r[0] for r in series], announce_ms)   # first bar strictly after
    if i <= 0 or i + H >= len(series):
        return None
    entry_ts, entry_px = series[i][0], series[i][4]
    exit_ts, exit_px = series[i + H][0], series[i + H][4]
    stock = exit_px / entry_px - 1 - ROUNDTRIP - 2 * slippage
    uni = [b / a - 1 for a, b in
           ((_asof(s, entry_ts), _asof(s, exit_ts)) for s in prices.values())
           if a and b]
    return stock - (sum(uni) / len(uni)) if uni else None


def _liquid_enough(series, min_years=3, min_adv=2e7):
    """>=3y of history and median daily traded value (close*volume) above Rs2cr."""
    if len(series) < int(250 * min_years * 0.9):
        return False
    if series[-1][0] - series[0][0] < int(min_years * 365 * 0.9) * DAY:
        return False
    tv = sorted(r[4] * r[5] for r in series if r[5] > 0)
    return bool(tv) and tv[len(tv) // 2] >= min_adv


def _yoy_ref(rows, i):
    """Reported EPS ~1 year before rows[i] (300-430 days prior), closest to 365d."""
    ms = rows[i][0]
    best = None
    for j in range(i - 1, -1, -1):
        gap = ms - rows[j][0]
        if gap > 430 * DAY:
            break
        if 300 * DAY <= gap <= 430 * DAY:
            if best is None or abs(gap - 365 * DAY) < abs(best[1]):
                best = (rows[j][2], gap - 365 * DAY)
    return best[0] if best else None


def build_events(universe):
    """Returns (events, prices, coverage). events: (ms, sym, surprise%, method).
    method is 'consensus' (Yahoo estimate) or 'yoy' (actual vs same quarter last year)."""
    prices, dropped = {}, {"no_price": 0, "history/liquidity": 0}
    for s in universe:
        ser = load_prices(s)
        if not ser:
            dropped["no_price"] += 1
        elif not _liquid_enough(ser):
            dropped["history/liquidity"] += 1
        else:
            prices[s] = ser
    events, cov = [], {"consensus": 0, "yoy": 0, "unusable": 0}
    for s in prices:
        rows = load_earnings_reported(s)
        for i, (ms, est, rep, sur) in enumerate(rows):
            if est is not None:
                surprise = sur if sur is not None else (rep - est) / abs(est) * 100 if est else None
                method = "consensus"
            else:
                ref = _yoy_ref(rows, i)
                surprise = (rep - ref) / abs(ref) * 100 if ref else None
                method = "yoy"
            if surprise is None:
                cov["unusable"] += 1
            else:
                events.append((ms, s, surprise, method))
                cov[method] += 1
    events.sort()
    return events, prices, {"stocks": len(prices), "dropped": dropped, **cov}


def precompute(events, prices):
    """Per (event, H) abnormal return BEFORE cost/slippage. The universe benchmark depends
    only on the dates, so computing it once here turns each sweep cell into an O(n) filter.
    Uses prebuilt ts/close arrays so each as-of lookup is one bisect (no list rebuild)."""
    tsL = {s: [r[0] for r in prices[s]] for s in prices}
    clL = {s: [r[4] for r in prices[s]] for s in prices}
    others = list(prices)
    per_H = {H: [] for H in HOLDS}                        # H -> [(ms, surprise, alpha_gross)]
    for ms, s, sur, _ in events:
        tsl, cl = tsL[s], clL[s]
        i = bisect.bisect_right(tsl, ms)
        if i <= 0 or i >= len(cl):
            continue
        entry_ts, entry_px = tsl[i], cl[i]
        for H in HOLDS:
            if i + H >= len(cl):
                continue
            exit_ts = tsl[i + H]
            stock = cl[i + H] / entry_px - 1
            tot = cnt = 0
            for o in others:
                otl, ocl = tsL[o], clL[o]
                a = bisect.bisect_right(otl, entry_ts) - 1
                b = bisect.bisect_right(otl, exit_ts) - 1
                if a >= 0 and b >= 0 and ocl[a] > 0:
                    tot += ocl[b] / ocl[a] - 1
                    cnt += 1
            if cnt:
                per_H[H].append((ms, sur, stock - tot / cnt))
    return per_H


def evaluate(per_H, thr, H, slippage=0.0, lo=None, hi=None):
    drag = ROUNDTRIP + 2 * slippage
    return [ag - drag for ms, sur, ag in per_H[H]
            if sur >= thr and (lo is None or lo <= ms < hi)]


def summarize(alphas):
    if len(alphas) < 5:
        return None
    a = np.array(alphas)
    se = a.std() / np.sqrt(len(a))
    return {"n": len(a), "mean_alpha": float(a.mean()), "hit": float((a > 0).mean()),
            "t": float(a.mean() / se) if se > 0 else 0.0}


def sweep(per_H, slippage):
    print(f"\n=== slippage {slippage:.1%} per side (on top of the ~0.3% cost model) ===")
    print(f"  {'thr%':>5} {'hold':>5} {'n':>5} {'mean_alpha':>11} {'hit%':>6} {'t-stat':>7}")
    best = None
    for thr in THRESHOLDS:
        for H in HOLDS:
            s = summarize(evaluate(per_H, thr, H, slippage))
            if not s:
                continue
            print(f"  {thr:>5.0f} {H:>5} {s['n']:>5} {s['mean_alpha']:>+10.2%} "
                  f"{s['hit']:>5.0%} {s['t']:>+7.2f}")
            if best is None or s["t"] > best[0]:
                best = (s["t"], thr, H, s)
    if best:
        _, thr, H, s = best
        print(f"  strongest: thr={thr:.0f}% hold={H}d  mean_alpha={s['mean_alpha']:+.2%} "
              f"t={s['t']:+.2f} n={s['n']}")
    return best


def walk_forward(per_H, slippage, n_folds=4):
    all_ms = sorted(ms for rows in per_H.values() for ms, _, _ in rows)
    if len(all_ms) < 50:
        return None
    q = [all_ms[min(len(all_ms) - 1, int(len(all_ms) * k / (n_folds + 1)))]
         for k in range(n_folds + 2)]
    combos = [(thr, H) for thr in THRESHOLDS for H in HOLDS]
    test_means = []
    for k in range(n_folds):
        scored = [(summarize(evaluate(per_H, thr, H, slippage, q[0], q[k + 1])), thr, H)
                  for thr, H in combos]
        scored = [(s, thr, H) for s, thr, H in scored if s and s["n"] >= 15]
        if not scored:
            continue
        s_tr, thr, H = max(scored, key=lambda x: x[0]["mean_alpha"])
        s_te = summarize(evaluate(per_H, thr, H, slippage, q[k + 1], q[k + 2]))
        if s_te:
            test_means.append(s_te["mean_alpha"])
    return float(np.mean(test_means)) if test_means else None


def _selftest():
    base, day = 1_600_000_000_000, 86_400_000
    up = [[base + i * day, 0, 0, 0, 100 * (1.10 ** (i / 20)), 0] for i in range(80)]
    flat = [[base + i * day, 0, 0, 0, 100.0, 0] for i in range(80)]
    prices = {"UP": up, "F1": flat, "F2": flat}
    a = event_alpha(prices, "UP", base + 5 * day, 20)          # beat-stock drifts, universe flat
    assert a and a > 0.05, a
    assert event_alpha(prices, "UP", base + 5 * day, 20, slippage=0.05) < a  # slippage eats alpha
    assert event_alpha(prices, "UP", base + 79 * day, 20) is None   # not enough forward data
    rows = [(base, None, 10.0, None), (base + 365 * day, None, 15.0, None)]
    assert _yoy_ref(rows, 1) == 10.0                            # YoY picks last year's EPS
    print("pead selftest ok: event alpha =", round(a, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="smallmid", choices=["smallmid", "nifty"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    universe = smallmid_universe() if args.universe == "smallmid" else NIFTY
    events, prices, cov = build_events(universe)
    total = cov["consensus"] + cov["yoy"]
    print("*** SURVIVORSHIP BIAS: current index constituents only. Names delisted/demoted/"
          "crashed out are absent, which inflates alpha. Free point-in-time membership is\n"
          "    unavailable (paid NSE product). Treat any positive result as an UPPER BOUND. ***\n")
    print(f"universe {args.universe}: {len(universe)} constituents -> {cov['stocks']} pass "
          f"3y-history + Rs2cr-median-turnover filter")
    print(f"  dropped: {cov['dropped']}")
    print(f"events: {total}  |  consensus {cov['consensus']} ({cov['consensus']/total:.0%})  "
          f"YoY-proxy {cov['yoy']} ({cov['yoy']/total:.0%})  |  unusable {cov['unusable']}")
    print("precomputing forward returns ...")
    per_H = precompute(events, prices)

    for slip in (0.0, 0.003, 0.005):
        sweep(per_H, slip)
        wf = walk_forward(per_H, slip)
        if wf is not None:
            print(f"  walk-forward out-of-sample mean alpha: {wf:+.2%}")


if __name__ == "__main__":
    main()
